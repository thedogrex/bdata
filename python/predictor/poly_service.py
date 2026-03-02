import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from db import DbProvider
from app.poly_client import PolymarketClient, MarketData
import app.config as config


db = DbProvider()


def _now_ts() -> int:
    return int(time.time())


def _compute_timestamps(now: Optional[int] = None, count: int = 5) -> List[int]:
    now = _now_ts() if now is None else int(now)
    interval = int(getattr(config, "POLY_INTERVAL_SECONDS", 300))
    if interval <= 0:
        interval = 300
    base = (now // interval) * interval
    return [base + i * interval for i in range(count)]


def _slug_for_ts(ts: int) -> str:
    template = getattr(config, "POLY_SLUG_TEMPLATE", "btc-updown-5m-{ts}")
    try:
        return template.format(ts=ts)
    except Exception:
        return f"btc-updown-5m-{ts}"


def current_active_ts(now: Optional[int] = None) -> int:
    now = _now_ts() if now is None else int(now)
    interval = int(getattr(config, "POLY_INTERVAL_SECONDS", 300))
    if interval <= 0:
        interval = 300
    return (now // interval) * interval


def _get_current_ts() -> int:
    return current_active_ts()


def _infer_resolved_outcome(m: MarketData) -> Optional[str]:
    """Infer resolved outcome from Gamma outcomePrices.

    Returns 'UP' or 'DOWN' when there is a clear winner, otherwise None.
    """
    if not m or not getattr(m, "outcomes", None):
        return None

    # Prefer explicit market resolution inputs when available.
    try:
        final_price = getattr(m, "final_price", None)
        target_price = getattr(m, "target_price", None)
        if final_price is not None and target_price is not None:
            return "DOWN" if float(final_price) < float(target_price) else "UP"
    except Exception:
        pass

    try:
        best = max(m.outcomes, key=lambda o: float(getattr(o, "price", 0.0)))
        best_price = float(getattr(best, "price", 0.0))
        name = (getattr(best, "name", "") or "").upper()
        # Conservative threshold to avoid marking unresolved markets.
        if best_price < 0.95:
            return None
        if "UP" in name:
            return "UP"
        if "DOWN" in name:
            return "DOWN"
        return None
    except Exception:
        return None


async def _check_and_store_market_resolution(slug: str, now_ts: int) -> Optional[str]:
    """Check market resolution via Gamma and store it in poly_markets.

    Always updates last_resolution_check_ts. Sets resolved_outcome only if inferred.
    """
    client = PolymarketClient()
    resolved: Optional[str] = None
    try:
        m = await asyncio.to_thread(client.fetch_market, slug)
        resolved = _infer_resolved_outcome(m)
    except Exception:
        resolved = None

    if resolved:
        await db.execute(
            """
            UPDATE poly_markets
            SET resolved_outcome=%s,
                last_resolution_check_ts=%s
            WHERE slug=%s
            """,
            (resolved, int(now_ts), slug),
        )
    else:
        await db.execute(
            """
            UPDATE poly_markets
            SET last_resolution_check_ts=%s
            WHERE slug=%s
            """,
            (int(now_ts), slug),
        )

    return resolved


async def _take_orderbook_snapshot_for_slug(slug: str) -> None:
    out_rows = await db.fetchall(
        "SELECT asset_id FROM poly_outcomes WHERE slug=%s",
        (slug,),
    )
    asset_ids = [r[0] for r in out_rows]
    if not asset_ids:
        return
    obs = await asyncio.to_thread(fetch_orderbooks, asset_ids)
    for ob in obs:
        asset_id = str(ob.get("asset_id") or "")
        if asset_id:
            await save_orderbook_snapshot(slug, asset_id, ob)


def _price_to_cents(price: Any) -> Optional[float]:
    try:
        p = float(price)
    except Exception:
        return None

    if p <= 1.0:
        return p * 100.0
    return p


async def upsert_market(m: MarketData) -> None:
    await db.execute(
        """
        INSERT INTO poly_markets (slug, ts, end_date, question, description, closed)
        VALUES (%s,%s,%s,%s,%s,%s) AS new
        ON DUPLICATE KEY UPDATE
            end_date=new.end_date,
            question=new.question,
            description=new.description,
            closed=new.closed
        """,
        (m.slug, int(m.timestamp), m.end_date, m.question, m.description, int(m.closed)),
    )

    for o in m.outcomes:
        await db.execute(
            """
            INSERT INTO poly_outcomes (slug, asset_id, name)
            VALUES (%s,%s,%s) AS new
            ON DUPLICATE KEY UPDATE
                name=new.name
            """,
            (m.slug, o.asset_id, o.name),
        )


async def refresh_tracked_markets() -> List[Dict[str, Any]]:
    client = PolymarketClient()
    rows: List[Dict[str, Any]] = []
    for ts in _compute_timestamps(count=3):
        slug = _slug_for_ts(ts)
        try:
            m = await asyncio.to_thread(client.fetch_market, slug)
            await upsert_market(m)
            rows.append({"slug": m.slug, "ts": m.timestamp, "closed": int(m.closed)})
        except Exception:
            rows.append({"slug": slug, "ts": ts, "closed": None})
    return rows


def fetch_orderbooks(asset_ids: List[str]) -> List[Dict[str, Any]]:
    if not asset_ids:
        return []
    url = "https://clob.polymarket.com/books"
    body = [{"token_id": str(a)} for a in asset_ids]
    resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(body), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []


async def save_orderbook_snapshot(slug: str, asset_id: str, data: Dict[str, Any]) -> None:
    asks = data.get("asks", [])

    # Convert prices to cents
    asks_cents = [{"price": _price_to_cents(a["price"]), "size": a["size"]} for a in asks]

    # Get best ask (lowest price)
    best_ask = min((a["price"] for a in asks_cents), default=None) if asks_cents else None

    ts = int(time.time())

    await db.execute(
        """
        INSERT INTO poly_orderbook_snapshots
            (slug, asset_id, ts, best_bid_cents, best_ask_cents, mid_cents, bids_json, asks_json)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            slug,
            str(asset_id),
            ts,
            None,  # best_bid_cents = NULL
            best_ask,
            None,  # mid_cents = NULL (no bids to calculate mid)
            None,  # bids_json = NULL
            json.dumps(asks_cents, ensure_ascii=False),
        ),
    )


async def _try_autopredict(slug: str) -> None:
    """Run autopredict for a market if enabled in settings."""
    try:
        settings = await get_settings()
        if not settings.get("autopredict"):
            return
        # Skip if already predicted (any stored runs)
        row = await db.fetchone(
            "SELECT 1 FROM poly_pred_runs WHERE slug=%s LIMIT 1", (slug,)
        )
        if row:
            return
        print(f"[autopredict] Running batch prediction for {slug}")
        await batch_predict_for_market(slug=slug, quantum=False, table="c_5m")
    except Exception as e:
        print(f"[autopredict] Error for {slug}: {e}")


async def _try_autopredict_after_end(ended_market_ts: int) -> None:
    """After a market ends (ts), predict for the market starting +5 minutes (H2)."""
    try:
        settings = await get_settings()
        if not settings.get("autopredict"):
            return

        min_target_ts = int(ended_market_ts) + 300
        targets = await db.fetchall(
            """
            SELECT slug, ts
            FROM poly_markets
            WHERE closed=0 AND ts >= %s
            ORDER BY ts ASC
            LIMIT 1
            """,
            (int(min_target_ts),),
        )

        # If we don't yet have the upcoming market, refresh and retry once.
        if not targets or len(targets) < 1:
            try:
                await refresh_tracked_markets()
            except Exception:
                pass
            targets = await db.fetchall(
                """
                SELECT slug, ts
                FROM poly_markets
                WHERE closed=0 AND ts >= %s
                ORDER BY ts ASC
                LIMIT 1
                """,
                (int(min_target_ts),),
            )

        for slug, _ts in targets:
            await _try_autopredict(str(slug))
    except Exception as e:
        print(f"[autopredict] after_end error (ended_ts={ended_market_ts}): {e}")


async def poll_loop(stop_event: asyncio.Event, orderbook_interval_sec: int = 3) -> None:
    orderbook_interval_sec = int(orderbook_interval_sec)
    if orderbook_interval_sec <= 0:
        orderbook_interval_sec = 3

    # Track last update times for different market types
    last_active_update = 0
    last_future_update = 0
    last_market_refresh = 0
    last_resolution_scan = 0
    last_seen_active_ts: Optional[int] = None
    autopredicted_slugs: set = set()  # slugs we already auto-predicted
    autopredicted_ended_ts: set[int] = set()  # ended market ts we already processed
    
    while not stop_event.is_set():
        try:
            current_time = int(time.time())

            # Refresh tracked markets less frequently to avoid blocking snapshot polling.
            # (Gamma API calls can be slow and were causing ~30s gaps between snapshots.)
            if current_time - last_market_refresh >= 60:
                await refresh_tracked_markets()
                last_market_refresh = current_time
            
            # Get current active timestamp
            current_ts = _get_current_ts()
            current_time = int(time.time())

            # If the active market advanced, take one final snapshot for the market that just ended.
            if last_seen_active_ts is None:
                last_seen_active_ts = current_ts
            elif current_ts != last_seen_active_ts:
                ended_ts = int(last_seen_active_ts)
                ended_slug = _slug_for_ts(ended_ts)
                try:
                    await _take_orderbook_snapshot_for_slug(ended_slug)
                except Exception:
                    pass
                # Backend autopredict: 1 second after end, run for markets at +10 minutes (H2) and next.
                if ended_ts not in autopredicted_ended_ts:
                    autopredicted_ended_ts.add(ended_ts)
                    try:
                        await _try_autopredict_after_end(ended_ts)
                    except Exception:
                        pass
                last_seen_active_ts = current_ts
            
            # Get all markets (open only, skip closed/past)
            m_rows = await db.fetchall(
                "SELECT slug, ts, closed FROM poly_markets ORDER BY ts DESC"
            )
            
            # Separate markets by type
            active_markets = []
            future_markets = []
            
            for slug, ts, closed in m_rows:
                # Skip closed/past markets
                if closed or ts < current_ts:
                    continue
                    
                if ts == current_ts:
                    active_markets.append((slug, ts))
                else:
                    # Only download snapshots shortly before market start
                    # (skip markets starting in 11+ minutes)
                    seconds_to_start = int(ts) - int(current_time)
                    if 0 < seconds_to_start <= 10 * 60:
                        future_markets.append((slug, ts))
            
            # Update active market every 3 seconds
            if current_time - last_active_update >= orderbook_interval_sec and active_markets:
                for slug, ts in active_markets[:1]:  # Only current active market
                    await _take_orderbook_snapshot_for_slug(slug)
                last_active_update = current_time
            
            # Update future markets every 10 seconds (next 2 markets)
            if current_time - last_future_update >= 10 and future_markets:
                for slug, ts in future_markets[:2]:  # Next 2 future markets
                    await _take_orderbook_snapshot_for_slug(slug)
                last_future_update = current_time

            # Keep sets bounded to avoid memory leak
            if len(autopredicted_slugs) > 50:
                autopredicted_slugs = set(list(autopredicted_slugs)[-30:])
            if len(autopredicted_ended_ts) > 100:
                autopredicted_ended_ts = set(list(autopredicted_ended_ts)[-60:])

            # Resolution polling: once per minute scan DONE markets without resolution.
            if current_time - last_resolution_scan >= 60:
                last_resolution_scan = current_time
                done_rows = await db.fetchall(
                    """
                    SELECT slug
                    FROM poly_markets
                    WHERE (closed=1 OR ts < %s)
                      AND (resolved_outcome IS NULL OR resolved_outcome='')
                      AND (last_resolution_check_ts IS NULL OR last_resolution_check_ts < %s)
                    ORDER BY ts DESC
                    LIMIT 20
                    """,
                    (int(current_ts), int(current_time) - 60),
                )
                for (slug,) in done_rows:
                    try:
                        await _check_and_store_market_resolution(str(slug), current_time)
                    except Exception:
                        pass

        except Exception as e:
            print(f"Error in poll loop: {e}")
            pass

        try:
            # Tick every 1s so backend countdown/autopredict is responsive.
            await asyncio.wait_for(stop_event.wait(), timeout=1)
        except asyncio.TimeoutError:
            continue


async def get_settings() -> Dict[str, Any]:
    """Get autopredict settings from DB."""
    row = await db.fetchone(
        "SELECT autopredict, strategy, params_json, window_size FROM poly_settings WHERE id='default'"
    )
    if not row:
        return {"autopredict": False, "strategy": "rsi_mean_reversion", "params": None, "window_size": 1000}
    autopredict, strategy, params_json, window_size = row
    params = None
    if params_json:
        try:
            params = json.loads(params_json)
        except Exception:
            pass
    return {
        "autopredict": bool(autopredict),
        "strategy": strategy or "rsi_mean_reversion",
        "params": params,
        "window_size": int(window_size) if window_size else 1000,
    }


async def save_settings(autopredict: bool, strategy: str, params: Optional[dict], window_size: int) -> Dict[str, Any]:
    """Save autopredict settings to DB."""
    params_json = json.dumps(params, ensure_ascii=False) if params else None
    await db.execute(
        """
        INSERT INTO poly_settings (id, autopredict, strategy, params_json, window_size)
        VALUES ('default', %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            autopredict=VALUES(autopredict),
            strategy=VALUES(strategy),
            params_json=VALUES(params_json),
            window_size=VALUES(window_size)
        """,
        (int(autopredict), strategy, params_json, int(window_size)),
    )
    return await get_settings()


async def list_markets(limit: int = 50) -> List[Dict[str, Any]]:
    current_ts = _get_current_ts()
    rows = await db.fetchall(
        """
        SELECT
            pm.slug,
            pm.ts,
            pm.end_date,
            pm.question,
            pm.closed,
            pm.resolved_outcome,
            pm.prediction_outcome,
            pm.prediction_ts,
            MAX(CASE WHEN pr.id IS NULL THEN 0 ELSE 1 END) AS has_pred_any,
            MAX(CASE WHEN pr.id IS NOT NULL AND pr.prediction IN ('UP','DOWN') THEN 1 ELSE 0 END) AS has_pred_defined,
            MAX(
                CASE
                    WHEN pr.id IS NOT NULL
                     AND pr.prediction IN ('UP','DOWN')
                     AND pm.resolved_outcome IN ('UP','DOWN')
                     AND pr.prediction = pm.resolved_outcome
                    THEN 1 ELSE 0
                END
            ) AS has_pred_correct
        FROM poly_markets pm
        LEFT JOIN poly_pred_runs pr ON pr.slug = pm.slug
        GROUP BY
            pm.slug, pm.ts, pm.end_date, pm.question, pm.closed,
            pm.resolved_outcome, pm.prediction_outcome, pm.prediction_ts
        ORDER BY pm.ts DESC
        LIMIT %s
        """,
        (int(limit),),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        (
            slug,
            ts,
            end_date,
            question,
            closed,
            resolved_outcome,
            prediction_outcome,
            prediction_ts,
            has_pred_any,
            has_pred_defined,
            has_pred_correct,
        ) = r
        status = "open"
        
        # Mark past markets as ended
        if closed:
            status = "ended"
        elif ts < current_ts:
            # Past but not marked as closed yet
            status = "ended"
        
        out.append({
            "slug": slug, 
            "ts": int(ts), 
            "end_date": end_date, 
            "question": question, 
            "closed": int(closed),
            "resolved_outcome": resolved_outcome,
            "prediction_outcome": prediction_outcome,
            "prediction_ts": int(prediction_ts) if prediction_ts is not None else None,
            "has_pred": bool(has_pred_any),
            "has_pred_defined": bool(has_pred_defined),
            "has_pred_correct": bool(has_pred_correct),
            "pred_badge": (
                None
                if not has_pred_any
                else (
                    "green"
                    if (resolved_outcome in ("UP", "DOWN") and has_pred_defined and has_pred_correct)
                    else (
                        "red"
                        if (resolved_outcome in ("UP", "DOWN") and has_pred_defined and not has_pred_correct)
                        else "neutral"
                    )
                )
            ),
            "status": status
        })
    return out


async def get_market(slug: str) -> Optional[Dict[str, Any]]:
    m = await db.fetchone(
        "SELECT slug, ts, end_date, question, description, closed, resolved_outcome, prediction_outcome, prediction_ts FROM poly_markets WHERE slug=%s",
        (slug,),
    )
    if not m:
        return None

    o_rows = await db.fetchall(
        "SELECT asset_id, name FROM poly_outcomes WHERE slug=%s ORDER BY asset_id",
        (slug,),
    )

    return {
        "slug": m[0],
        "ts": int(m[1]),
        "end_date": m[2],
        "question": m[3],
        "description": m[4],
        "closed": int(m[5]),
        "resolved_outcome": m[6],
        "prediction_outcome": m[7],
        "prediction_ts": int(m[8]) if m[8] is not None else None,
        "outcomes": [{"asset_id": r[0], "name": r[1]} for r in o_rows],
    }


async def get_market_live(slug: str) -> Optional[Dict[str, Any]]:
    """Fetch live market data from Polymarket (includes outcome prices)."""
    client = PolymarketClient()
    try:
        m = await asyncio.to_thread(client.fetch_market, slug)
    except Exception:
        return None

    resolved = _infer_resolved_outcome(m)

    return {
        "slug": m.slug,
        "ts": int(m.timestamp),
        "end_date": m.end_date,
        "question": m.question,
        "description": m.description,
        "closed": int(m.closed),
        "resolved_outcome": resolved,
        "outcomes": [{"asset_id": o.asset_id, "name": o.name, "price": float(o.price)} for o in m.outcomes],
    }


async def get_price_series(asset_id: str, minutes: int = 60, limit: int = 2000) -> List[Dict[str, Any]]:
    minutes = int(minutes)
    if minutes <= 0:
        minutes = 60

    ts_from = int(time.time()) - minutes * 60

    rows = await db.fetchall(
        """
        SELECT ts, best_ask_cents
        FROM poly_orderbook_snapshots
        WHERE asset_id=%s AND ts >= %s
        ORDER BY ts ASC
        LIMIT %s
        """,
        (str(asset_id), int(ts_from), int(limit)),
    )

    return [
        {
            "ts": int(r[0]),
            "best_bid_cents": None,  # Always NULL
            "best_ask_cents": float(r[1]) if r[1] is not None else None,
            "mid_cents": None,      # Always NULL
        }
        for r in rows
    ]


async def create_sim_trade(
    slug: str, asset_id: str, side: str, qty: float,
    outcome_side: str | None = None, requested_price: float | None = None,
) -> Dict[str, Any]:
    slug = str(slug or "").strip()
    if not slug:
        raise ValueError("slug is required")

    side = (side or "").upper().strip()
    if side != "BUY":
        raise ValueError("Only BUY orders are supported (asks-only)")

    qty = float(qty)
    if qty <= 0:
        raise ValueError("qty must be > 0")

    outcome_side = (outcome_side or "").upper().strip() or None

    # Validate asset belongs to this market
    market = await get_market(slug)
    if not market:
        raise ValueError(f"Unknown market slug: {slug}")
    outcomes = market.get("outcomes") or []
    if str(asset_id) not in {str(o.get("asset_id")) for o in outcomes}:
        raise ValueError("asset_id does not belong to this market")

    row = await db.fetchone(
        """
        SELECT slug, ts, best_ask_cents, asks_json
        FROM poly_orderbook_snapshots
        WHERE asset_id=%s
        ORDER BY ts DESC
        LIMIT 1
        """,
        (str(asset_id),),
    )
    if not row:
        raise RuntimeError("No orderbook data for this asset_id yet")

    slug = row[0]
    snap_ts = int(row[1])
    best_ask = row[2]
    asks_raw = row[3]

    # Parse asks from snapshot
    import json as _json
    asks = []
    if asks_raw:
        try:
            asks = _json.loads(asks_raw) if isinstance(asks_raw, str) else asks_raw
        except Exception:
            asks = []

    if not asks:
        raise RuntimeError("Order book has 0 records — buy impossible")

    # Sort asks ascending by price
    asks_sorted = sorted(asks, key=lambda a: float(a.get("price", 9999)))

    if requested_price is not None:
        req_p = float(requested_price)
        # Check if the requested price still exists in the current snapshot
        available = [a for a in asks_sorted if float(a.get("price", 9999)) <= req_p]
        if not available:
            raise RuntimeError("Price has been changed")

        # Fill at the best available price <= requested price, check size
        remaining_qty = qty
        fill_price = None
        for ask in available:
            ask_size = float(ask.get("size", 0))
            ask_price = float(ask.get("price", 0))
            if ask_size >= remaining_qty:
                fill_price = ask_price
                break
            remaining_qty -= ask_size
            fill_price = ask_price

        if remaining_qty > 0 and fill_price is not None:
            # Check total available size across all qualifying asks
            total_size = sum(float(a.get("size", 0)) for a in available)
            if total_size < qty:
                raise RuntimeError("Not enough in order book")

        if fill_price is None:
            raise RuntimeError("No ask price available")
    else:
        fill_price = best_ask
        if fill_price is None:
            raise RuntimeError("No ask price available for this asset_id yet")

    fill = float(fill_price)
    trade_ts = int(time.time())

    trade_id = await db.execute(
        """
        INSERT INTO poly_sim_trades
            (ts, slug, asset_id, side, outcome_side, qty, requested_price_cents, fill_price_cents, snapshot_ts)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (trade_ts, slug, str(asset_id), side, outcome_side, qty,
         float(requested_price) if requested_price is not None else None,
         fill, snap_ts),
    )

    return {
        "id": int(trade_id),
        "ts": trade_ts,
        "slug": slug,
        "asset_id": str(asset_id),
        "side": side,
        "outcome_side": outcome_side,
        "qty": qty,
        "requested_price_cents": float(requested_price) if requested_price is not None else None,
        "fill_price_cents": fill,
        "snapshot_ts": snap_ts,
    }


async def list_sim_trades(limit: int = 200) -> List[Dict[str, Any]]:
    rows = await db.fetchall(
        """
        SELECT id, ts, slug, asset_id, side, outcome_side, qty, fill_price_cents
        FROM poly_sim_trades
        ORDER BY ts DESC
        LIMIT %s
        """,
        (int(limit),),
    )
    return [
        {
            "id": int(r[0]),
            "ts": int(r[1]),
            "slug": r[2],
            "asset_id": r[3],
            "side": r[4],
            "outcome_side": r[5],
            "qty": float(r[6]),
            "fill_price_cents": float(r[7]),
        }
        for r in rows
    ]


async def get_sim_positions(slug: str | None = None) -> List[Dict[str, Any]]:
    slug = str(slug).strip() if slug is not None else None
    if slug:
        rows = await db.fetchall(
            """
            SELECT slug,
                   asset_id,
                   SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END) as pos_qty,
                   SUM(CASE WHEN side='BUY' THEN qty*fill_price_cents ELSE -qty*fill_price_cents END) as cashflow
            FROM poly_sim_trades
            WHERE slug=%s
            GROUP BY slug, asset_id
            """,
            (slug,),
        )
    else:
        rows = await db.fetchall(
            """
            SELECT NULL as slug,
                   asset_id,
                   SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END) as pos_qty,
                   SUM(CASE WHEN side='BUY' THEN qty*fill_price_cents ELSE -qty*fill_price_cents END) as cashflow
            FROM poly_sim_trades
            GROUP BY asset_id
            """
        )

    out: List[Dict[str, Any]] = []
    for r in rows:
        row_slug = str(r[0]) if r[0] is not None else None
        asset_id = str(r[1])
        pos_qty = float(r[2]) if r[2] is not None else 0.0
        cashflow = float(r[3]) if r[3] is not None else 0.0

        last = await db.fetchone(
            """
            SELECT best_ask_cents
            FROM poly_orderbook_snapshots
            WHERE asset_id=%s
            ORDER BY ts DESC
            LIMIT 1
            """,
            (asset_id,),
        )

        mark = None
        if last:
            mark = last[0]

        mark = float(mark) if mark is not None else None

        m2m = None
        if mark is not None:
            m2m = pos_qty * mark

        pnl = None
        if m2m is not None:
            pnl = m2m - cashflow

        out.append(
            {
                "slug": row_slug,
                "asset_id": asset_id,
                "pos_qty": pos_qty,
                "cashflow_cents": cashflow,
                "mark_cents": mark,
                "m2m_cents": m2m,
                "pnl_cents": pnl,
            }
        )

    out.sort(key=lambda x: abs(x.get("pnl_cents") or 0), reverse=True)
    return out


async def get_sim_markets_with_positions() -> List[str]:
    """List market slugs where there is at least one non-zero position."""
    rows = await db.fetchall(
        """
        SELECT slug
        FROM (
            SELECT slug,
                   asset_id,
                   SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END) as pos_qty
            FROM poly_sim_trades
            GROUP BY slug, asset_id
        ) t
        WHERE ABS(pos_qty) > 0
        GROUP BY slug
        ORDER BY slug ASC
        """
    )
    return [str(r[0]) for r in rows if r and r[0]]


async def predict_for_market(
    slug: str,
    strategy_name: str = "rsi_mean_reversion",
    strategy_params: Optional[Dict[str, Any]] = None,
    window_size: int = 1000,
    horizon: int = 1,
    table: str = "c_5m",
) -> Dict[str, Any]:
    """Run a single-candle prediction for the candle at the market's timestamp.

    The market slug encodes the timestamp (e.g. btc-updown-5m-1739919600).
    We need exactly *window_size* consecutive 5-min candles ending with the
    candle whose open_time == market_ts (in microseconds).

    Steps:
      1. Auto-sync missing candles from Binance REST API.
      2. Load *window_size* candles ending at market_ts (inclusive).
      3. Verify continuity (no gaps).
      4. Fit strategy on first N-1 candles, predict on the last one.
    """
    import pandas as pd
    import numpy as np
    from predictor.features import add_technical_features
    from predictor.strategies import get_strategy, STRATEGY_REGISTRY
    from predictor.data_loader import add_direction
    from predictor.candle_sync import sync_candles_up_to

    horizon = max(1, min(3, int(horizon)))

    if strategy_name not in STRATEGY_REGISTRY:
        return {"error": f"Unknown strategy: {strategy_name}"}

    # Resolve market timestamp (epoch seconds from slug)
    m_row = await db.fetchone(
        "SELECT ts FROM poly_markets WHERE slug=%s", (slug,)
    )
    if not m_row:
        return {"error": f"Market not found: {slug}"}
    market_ts = int(m_row[0])

    # 5-minute candle interval in microseconds
    interval_us = 5 * 60 * 1_000_000  # 300 000 000
    market_ts_us = market_ts * 1_000_000

    # --- Step 1: Auto-sync candles from Binance API ---
    sync_info = {}
    try:
        sync_result = await sync_candles_up_to(market_ts, window_candles=window_size + 50, table=table)
        sync_info["sync"] = sync_result
        if sync_result.get("downloaded", 0) > 0:
            # Also fill any internal gaps
            fill_result = await check_and_fill_gaps(market_ts, window_candles=window_size, table=table)
            sync_info["gap_fill"] = fill_result
    except Exception as e:
        sync_info["sync_error"] = str(e)

    # --- Step 2: Load candles ending at market_ts ---
    need = window_size

    rows = await db.fetchall(
        f"""
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM {table}
        WHERE open_time <= %s
        ORDER BY open_time DESC
        LIMIT %s
        """,
        (market_ts_us, need),
    )

    if not rows or len(rows) < need:
        have = len(rows) if rows else 0
        # Find latest candle to help diagnose
        latest_row = await db.fetchone(f"SELECT MAX(open_time) FROM {table}")
        latest_us = int(latest_row[0]) if latest_row and latest_row[0] else 0
        latest_dt = pd.Timestamp(latest_us, unit="us").strftime("%Y-%m-%d %H:%M:%S") if latest_us else "none"
        market_dt = pd.Timestamp(market_ts_us, unit="us").strftime("%Y-%m-%d %H:%M:%S")
        return {
            "error": f"Not enough candles before market timestamp. Need {need}, have {have}. "
                     f"Market ts: {market_dt} (epoch {market_ts}). "
                     f"Latest candle in DB: {latest_dt}.",
            "need": need,
            "have": have,
            "market_ts": market_ts,
            "latest_candle_us": latest_us,
            **sync_info,
        }

    # Reverse to ascending order
    rows = list(reversed(rows))

    # --- Step 2b: Verify last candle aligns with market_ts ---
    # For horizon H, we can tolerate up to (H-1) missing candles at the end,
    # because the model predicts H candles ahead of the signal candle.
    #   H1 → needs exact candle at market_ts  (0 missing allowed)
    #   H2 → can work with 1 missing candle   (signal shifted back by 1)
    #   H3 → can work with 1-2 missing candles (signal shifted back by 1-2)
    last_candle_open_us = int(rows[-1][0])
    missing_candles = max(0, int((market_ts_us - last_candle_open_us) / interval_us))
    shifted = missing_candles > 0

    if missing_candles > 0 and missing_candles >= horizon:
        last_dt = pd.Timestamp(last_candle_open_us, unit="us").strftime("%Y-%m-%d %H:%M:%S")
        market_dt = pd.Timestamp(market_ts_us, unit="us").strftime("%Y-%m-%d %H:%M:%S")
        diff_min = round((market_ts_us - last_candle_open_us) / 1_000_000 / 60, 1)
        return {
            "error": f"Candle at market timestamp not found. Market ts: {market_dt}, "
                     f"latest candle: {last_dt} ({diff_min} min before). "
                     f"Missing {missing_candles} candle(s) — need at most {horizon - 1} "
                     f"missing for H{horizon}.",
            "market_ts_us": market_ts_us,
            "last_candle_us": last_candle_open_us,
            "missing_candles": missing_candles,
            "horizon": horizon,
            **sync_info,
        }

    # Build DataFrame
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quota_volume", "trades", "taker_base_volume", "taker_quota_volume"
    ])
    for c in ["open", "high", "low", "close", "volume", "quota_volume",
              "taker_base_volume", "taker_quota_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)

    # --- Step 3: Continuity check ---
    open_times = df["open_time"].values.astype(np.int64)
    diffs = np.diff(open_times)
    bad_idx = np.where(diffs != interval_us)[0]
    if len(bad_idx) > 0:
        first_bad = int(bad_idx[0])
        gap_from_us = int(open_times[first_bad])
        gap_to_us = int(open_times[first_bad + 1])
        gap_from_dt = pd.Timestamp(gap_from_us, unit="us").strftime("%Y-%m-%d %H:%M:%S")
        gap_to_dt = pd.Timestamp(gap_to_us, unit="us").strftime("%Y-%m-%d %H:%M:%S")
        actual_gap_min = round((gap_to_us - gap_from_us) / 1_000_000 / 60, 1)
        return {
            "error": f"Candle gap detected at position {first_bad}: {gap_from_dt} -> {gap_to_dt} "
                     f"({actual_gap_min} min instead of 5 min). {len(bad_idx)} gap(s) total in {need} candles.",
            "gap_from_ts": gap_from_us,
            "gap_to_ts": gap_to_us,
            "gaps_total": int(len(bad_idx)),
            **sync_info,
        }

    # --- Step 4: Run strategy ---
    df = add_direction(df)
    df = df.reset_index(drop=True)
    df_feat = add_technical_features(df)

    # --- Exact backtester mechanism (run_backtest lines 128-162) ---
    # Train on all candles BEFORE the last one, predict on the LAST candle only.
    # This is identical to how the backtester evaluates each candle:
    #   train = df_feat[i-window : i]   (exclusive of candle i)
    #   pred  = strategy.predict(df_feat[[i]])
    df_train = df_feat.iloc[:-1].reset_index(drop=True)
    df_predict = df_feat.iloc[[-1]].reset_index(drop=True)

    strategy = get_strategy(strategy_name, strategy_params)
    strategy.fit(df_train, horizon=horizon)
    pred_arr = strategy.predict(df_predict, horizon=horizon)
    prob_arr = strategy.predict_proba(df_predict, horizon=horizon)

    pred = int(pred_arr[0])
    prob = float(prob_arr[0])
    label = "UP" if pred == 1 else ("DOWN" if pred == 0 else "UNDEFINED")

    # --- Diagnostics: show WHY this candle got this signal ---
    period = strategy.params.get("rsi_period", 14)
    rsi_col = f"rsi_{period}" if f"rsi_{period}" in df_predict.columns else "rsi_14"
    pred_rsi = float(np.nan_to_num(df_predict[rsi_col].values[0], nan=50.0))
    pred_bb = float(np.nan_to_num(df_predict["bb_pos"].values[0], nan=0.5)) if "bb_pos" in df_predict.columns else 0.5

    base_oversold = strategy.params.get("rsi_oversold", 30)
    base_overbought = strategy.params.get("rsi_overbought", 70)
    rsi_p10 = getattr(strategy, '_rsi_p10', None)
    rsi_p90 = getattr(strategy, '_rsi_p90', None)
    if rsi_p10 is not None:
        effective_oversold = (base_oversold + rsi_p10) / 2
        effective_overbought = (base_overbought + rsi_p90) / 2
    else:
        effective_oversold = base_oversold
        effective_overbought = base_overbought

    # Also show last 10 candles context (what happened just before)
    context_start = max(0, len(df_feat) - 10)
    tail_detail = []
    for k in range(context_start, len(df_feat)):
        row = df_feat.iloc[k]
        ot = int(row["open_time"])
        dt_str = pd.Timestamp(ot, unit="us").strftime("%m-%d %H:%M")
        r_rsi = float(np.nan_to_num(row[rsi_col], nan=50.0))
        r_bb = float(np.nan_to_num(row.get("bb_pos", 0.5), nan=0.5))
        # Re-predict each context candle to show what backtest would have said
        df_k = df_feat.iloc[[k]].reset_index(drop=True)
        k_pred = int(strategy.predict(df_k, horizon=horizon)[0])
        k_prob = float(strategy.predict_proba(df_k, horizon=horizon)[0])
        tail_detail.append({
            "dt": dt_str,
            "rsi": round(r_rsi, 1),
            "bb": round(r_bb, 3),
            "prob": round(k_prob, 4),
            "pred": k_pred,
        })

    # Collect RSI stats from the last 10 candles for diagnostics
    tail_rsi_vals = [d["rsi"] for d in tail_detail if d["rsi"] is not None]
    diag = {
        "base_oversold": round(float(base_oversold), 1),
        "base_overbought": round(float(base_overbought), 1),
        "effective_oversold": round(effective_oversold, 1),
        "effective_overbought": round(effective_overbought, 1),
        "rsi_p10": round(rsi_p10, 1) if rsi_p10 is not None else None,
        "rsi_p90": round(rsi_p90, 1) if rsi_p90 is not None else None,
        "train_size": len(df_train),
        "tail_size": len(tail_detail),
        "pred_rsi": round(pred_rsi, 1),
        "pred_bb": round(pred_bb, 3),
        "tail_rsi_min": round(min(tail_rsi_vals), 1) if tail_rsi_vals else None,
        "tail_rsi_max": round(max(tail_rsi_vals), 1) if tail_rsi_vals else None,
        "tail_rsi_last": round(tail_rsi_vals[-1], 1) if tail_rsi_vals else None,
        "tail_detail": tail_detail,
    }

    signals_in_tail = sum(1 for d in tail_detail if d["pred"] != -1)
    total_in_tail = len(tail_detail)
    candles_ago = 0 if pred != -1 else -1

    last_candle_ts_us = int(open_times[-1])
    last_candle_dt = pd.Timestamp(last_candle_ts_us, unit="us").strftime("%Y-%m-%d %H:%M:%S")

    # Persist prediction result to market row so UI can show it later.
    prediction_ts = int(time.time())
    try:
        await db.execute(
            """
            UPDATE poly_markets
            SET prediction_outcome=%s,
                prediction_ts=%s
            WHERE slug=%s
            """,
            (label, int(prediction_ts), slug),
        )
    except Exception:
        pass

    # Signal candle = the last candle (we predict on exactly one candle)
    signal_candle_dt = last_candle_dt if pred != -1 else None

    ret = {
        "prediction": label,
        "probability": round(prob, 4),
        "strategy": strategy_name,
        "params": strategy.params,
        "window_size": window_size,
        "horizon": horizon,
        "candles_used": need,
        "last_candle_ts": last_candle_ts_us,
        "last_candle_dt": last_candle_dt,
        "signal_candle_dt": signal_candle_dt,
        "candles_ago": candles_ago,
        "signals_in_tail": signals_in_tail,
        "tail_size": total_in_tail,
        "market_slug": slug,
        "market_ts": market_ts,
        "prediction_ts": int(prediction_ts),
        "shifted": shifted,
        "missing_candles": missing_candles,
        "diag": diag,
        **sync_info,
    }
    if shifted:
        ret["shift_note"] = (
            f"Signal candle shifted back by {missing_candles} "
            f"({missing_candles * 5} min). Using H{horizon} to compensate."
        )

    # Persist full payload for instant UI analysis later
    try:
        payload_json = json.dumps(ret, ensure_ascii=False)
        await db.execute(
            """
            INSERT INTO poly_predictions (slug, prediction_ts, payload_json)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                prediction_ts=%s,
                payload_json=%s
            """,
            (slug, int(prediction_ts), payload_json, int(prediction_ts), payload_json),
        )
    except Exception:
        pass

    return ret


async def get_saved_prediction(slug: str) -> Dict[str, Any]:
    row = await db.fetchone(
        "SELECT payload_json FROM poly_predictions WHERE slug=%s LIMIT 1",
        (slug,),
    )
    if not row or not row[0]:
        return {"error": "Prediction not found"}
    try:
        return json.loads(row[0])
    except Exception:
        return {"error": "Prediction payload corrupted"}


# ==================== PREDICTION TEMPLATES ====================

async def list_pred_templates() -> List[Dict[str, Any]]:
    rows = await db.fetchall(
        "SELECT id, name, strategy, params_json, window_size, horizon, active, sort_order "
        "FROM poly_pred_templates ORDER BY sort_order ASC, id ASC"
    )
    out = []
    for r in rows:
        tid, name, strategy, params_json, window_size, horizon, active, sort_order = r
        params = None
        if params_json:
            try:
                params = json.loads(params_json)
            except Exception:
                pass
        out.append({
            "id": int(tid),
            "name": name,
            "strategy": strategy,
            "params": params,
            "window_size": int(window_size) if window_size else 1000,
            "horizon": int(horizon) if horizon else 1,
            "active": bool(active),
            "sort_order": int(sort_order) if sort_order else 0,
        })
    return out


async def create_pred_template(
    name: str,
    strategy: str = "rsi_mean_reversion",
    params: Optional[Dict] = None,
    window_size: int = 1000,
    horizon: int = 1,
) -> Dict[str, Any]:
    params_json = json.dumps(params, ensure_ascii=False) if params else None
    horizon = max(1, min(3, int(horizon)))
    await db.execute(
        "INSERT INTO poly_pred_templates (name, strategy, params_json, window_size, horizon, active, sort_order) "
        "VALUES (%s, %s, %s, %s, %s, 1, 0)",
        (name, strategy, params_json, int(window_size), horizon),
    )
    return {"status": "ok"}


async def update_pred_template(
    template_id: int,
    name: Optional[str] = None,
    strategy: Optional[str] = None,
    params: Optional[Dict] = None,
    window_size: Optional[int] = None,
    horizon: Optional[int] = None,
    active: Optional[bool] = None,
    sort_order: Optional[int] = None,
) -> Dict[str, Any]:
    sets = []
    vals = []
    if name is not None:
        sets.append("name=%s"); vals.append(name)
    if strategy is not None:
        sets.append("strategy=%s"); vals.append(strategy)
    if params is not None:
        sets.append("params_json=%s"); vals.append(json.dumps(params, ensure_ascii=False))
    if window_size is not None:
        sets.append("window_size=%s"); vals.append(int(window_size))
    if horizon is not None:
        sets.append("horizon=%s"); vals.append(max(1, min(3, int(horizon))))
    if active is not None:
        sets.append("active=%s"); vals.append(int(active))
    if sort_order is not None:
        sets.append("sort_order=%s"); vals.append(int(sort_order))
    if not sets:
        return {"status": "nothing to update"}
    vals.append(int(template_id))
    await db.execute(
        f"UPDATE poly_pred_templates SET {', '.join(sets)} WHERE id=%s",
        tuple(vals),
    )
    return {"status": "ok"}


async def delete_pred_template(template_id: int) -> Dict[str, Any]:
    await db.execute("DELETE FROM poly_pred_templates WHERE id=%s", (int(template_id),))
    return {"status": "ok"}


async def toggle_pred_template(template_id: int) -> Dict[str, Any]:
    await db.execute(
        "UPDATE poly_pred_templates SET active = NOT active WHERE id=%s",
        (int(template_id),),
    )
    return {"status": "ok"}


# ==================== BATCH / QUANTUM PREDICT ====================

async def batch_predict_for_market(
    slug: str,
    quantum: bool = False,
    table: str = "c_5m",
) -> Dict[str, Any]:
    """Run predictions for all active templates on a market.
    If quantum=True, simulate missing candle as green/red scenarios.
    Records each run (with timing) to poly_pred_runs.
    """
    templates = await list_pred_templates()
    active = [t for t in templates if t["active"]]
    if not active:
        return {"error": "No active prediction templates. Create and enable at least one."}

    batch_id = str(uuid.uuid4())
    results = []

    _RUN_SQL = """
        INSERT INTO poly_pred_runs
          (slug, batch_id, template_id, template_name, strategy, params_json,
           window_size, horizon, quantum, quantum_scenario,
           prediction, probability, started_at, finished_at, duration_ms, error, result_json)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    for tpl in active:
        params_json_str = json.dumps(tpl["params"], ensure_ascii=False) if tpl["params"] else None
        started_at = datetime.utcnow()

        if quantum:
            qr = await quantum_predict_for_market(
                slug=slug,
                strategy_name=tpl["strategy"],
                strategy_params=tpl["params"],
                window_size=tpl["window_size"],
                horizon=tpl["horizon"],
                table=table,
            )
            finished_at = datetime.utcnow()
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)
            results.append({
                "template_id": tpl["id"],
                "template_name": tpl["name"],
                "horizon": tpl["horizon"],
                "quantum": True,
                "result": qr,
            })
            if qr.get("error"):
                await db.execute(_RUN_SQL, (
                    slug, batch_id, tpl["id"], tpl["name"], tpl["strategy"], params_json_str,
                    tpl["window_size"], tpl["horizon"], 1, None,
                    None, None, started_at, finished_at, duration_ms, qr["error"], None,
                ))
            else:
                for sc_name, sc in (qr.get("scenarios") or {}).items():
                    await db.execute(_RUN_SQL, (
                        slug, batch_id, tpl["id"], tpl["name"], tpl["strategy"], params_json_str,
                        tpl["window_size"], tpl["horizon"], 1, sc_name,
                        sc.get("prediction"), sc.get("probability"),
                        started_at, finished_at, duration_ms, None,
                        json.dumps(sc, ensure_ascii=False),
                    ))
        else:
            r = await predict_for_market(
                slug=slug,
                strategy_name=tpl["strategy"],
                strategy_params=tpl["params"],
                window_size=tpl["window_size"],
                horizon=tpl["horizon"],
                table=table,
            )
            finished_at = datetime.utcnow()
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)
            results.append({
                "template_id": tpl["id"],
                "template_name": tpl["name"],
                "horizon": tpl["horizon"],
                "quantum": False,
                "result": r,
            })
            await db.execute(_RUN_SQL, (
                slug, batch_id, tpl["id"], tpl["name"], tpl["strategy"], params_json_str,
                tpl["window_size"], tpl["horizon"], 0, None,
                r.get("prediction") if not r.get("error") else None,
                r.get("probability") if not r.get("error") else None,
                started_at, finished_at, duration_ms,
                r.get("error") or None,
                json.dumps(r, ensure_ascii=False) if not r.get("error") else None,
            ))

    # For non-quantum: cache vote summary on poly_markets row
    if not quantum:
        up = sum(1 for e in results if e["result"].get("prediction") == "UP")
        dn = sum(1 for e in results if e["result"].get("prediction") == "DOWN")
        unk = len(results) - up - dn
        votes_json = json.dumps({"up": up, "down": dn, "unk": unk,
                                  "batch_id": batch_id,
                                  "ts": int(datetime.utcnow().timestamp())},
                                 ensure_ascii=False)
        try:
            await db.execute(
                "UPDATE poly_markets SET pred_votes=%s WHERE slug=%s",
                (votes_json, slug),
            )
        except Exception:
            pass  # column may not exist yet — apply SQL migration first

    return {"slug": slug, "quantum": quantum, "batch_id": batch_id, "results": results}


async def get_pred_runs_for_market(slug: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Return prediction run history for a market, newest first."""
    rows = await db.fetchall(
        """
        SELECT id, batch_id, template_id, template_name, strategy, params_json,
               window_size, horizon, quantum, quantum_scenario,
               prediction, probability, started_at, finished_at, duration_ms,
               error, result_json
        FROM poly_pred_runs
        WHERE slug=%s
        ORDER BY started_at DESC
        LIMIT %s
        """,
        (slug, int(limit)),
    )
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "batch_id": r[1],
            "template_id": r[2],
            "template_name": r[3],
            "strategy": r[4],
            "params": json.loads(r[5]) if r[5] else None,
            "window_size": r[6],
            "horizon": r[7],
            "quantum": bool(r[8]),
            "quantum_scenario": r[9],
            "prediction": r[10],
            "probability": float(r[11]) if r[11] is not None else None,
            "started_at": r[12].isoformat() if r[12] else None,
            "finished_at": r[13].isoformat() if r[13] else None,
            "duration_ms": r[14],
            "error": r[15],
            "result": json.loads(r[16]) if r[16] else None,
        })
    return out


async def quantum_predict_for_market(
    slug: str,
    strategy_name: str = "rsi_mean_reversion",
    strategy_params: Optional[Dict[str, Any]] = None,
    window_size: int = 1000,
    horizon: int = 1,
    table: str = "c_5m",
) -> Dict[str, Any]:
    """Quantum predict: simulate two scenarios for the missing candle.
    Scenario A: next candle is GREEN (close > open, medium size).
    Scenario B: next candle is RED   (close < open, medium size).
    Run prediction for each and return both results.
    """
    import pandas as pd
    import numpy as np
    from predictor.features import add_technical_features
    from predictor.strategies import get_strategy, STRATEGY_REGISTRY
    from predictor.data_loader import add_direction
    from predictor.candle_sync import sync_candles_up_to, check_and_fill_gaps

    if strategy_name not in STRATEGY_REGISTRY:
        return {"error": f"Unknown strategy: {strategy_name}"}

    horizon = max(1, min(3, int(horizon)))

    m_row = await db.fetchone("SELECT ts FROM poly_markets WHERE slug=%s", (slug,))
    if not m_row:
        return {"error": f"Market not found: {slug}"}
    market_ts = int(m_row[0])

    interval_us = 5 * 60 * 1_000_000
    interval_s = 300
    market_ts_us = market_ts * 1_000_000

    # Sync candles up to market_ts (the candle at market_ts might not exist yet)
    sync_info = {}
    try:
        sync_result = await sync_candles_up_to(market_ts, window_candles=window_size + 50, table=table)
        sync_info["sync"] = sync_result
    except Exception as e:
        sync_info["sync_error"] = str(e)

    # Load candles up to market_ts (inclusive, but may not have it)
    need = window_size
    rows = await db.fetchall(
        f"""
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM {table}
        WHERE open_time <= %s
        ORDER BY open_time DESC
        LIMIT %s
        """,
        (market_ts_us, need),
    )
    if not rows or len(rows) < 2:
        return {"error": "Not enough candles for quantum predict", **sync_info}

    rows = list(reversed(rows))

    # Quantum predict is ONLY available for markets that come AFTER the current
    # active (live) market.  For the active market and all past markets every
    # candle is already present, so regular predict should be used instead.
    active_ts = current_active_ts()
    if market_ts <= active_ts:
        active_dt = pd.Timestamp(active_ts * 1_000_000, unit="us").strftime("%Y-%m-%d %H:%M:%S")
        market_dt = pd.Timestamp(market_ts_us, unit="us").strftime("%Y-%m-%d %H:%M:%S")
        return {
            "error": (
                "Quantum predict is not available for the current or past markets. "
                "Use it only for future markets (after the active market). "
                f"Active market: {active_dt}, this market: {market_dt}."
            ),
            "market_ts": market_ts,
            "market_dt": market_dt,
            "active_ts": active_ts,
            "active_dt": active_dt,
        }

    # Synthesize the missing candle at market_ts (it hasn't formed yet)
    synth_open_us = market_ts_us

    last_close = float(rows[-1][4])
    # Compute average candle body size from last 20 candles
    recent = rows[-20:]
    avg_body = np.mean([abs(float(r[4]) - float(r[1])) for r in recent])
    if avg_body < 1:
        avg_body = last_close * 0.001
    avg_vol = np.mean([float(r[5]) for r in recent])

    scenarios = {}
    for scenario_name, direction_up in [("green", True), ("red", False)]:
        synth_open = last_close
        if direction_up:
            synth_close = last_close + avg_body
            synth_high = synth_close + avg_body * 0.3
            synth_low = synth_open - avg_body * 0.2
        else:
            synth_close = last_close - avg_body
            synth_high = synth_open + avg_body * 0.2
            synth_low = synth_close - avg_body * 0.3

        synth_row = (
            synth_open_us,
            synth_open,
            synth_high,
            synth_low,
            synth_close,
            avg_vol,
            (synth_open_us // 1000) + interval_s * 1000 - 1,
            avg_vol * synth_close,
            100,
            avg_vol * 0.5,
            avg_vol * synth_close * 0.5,
        )

        scenario_rows = list(rows) + [synth_row]
        # Keep only the last `need` rows
        scenario_rows = scenario_rows[-need:]

        df = pd.DataFrame(scenario_rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quota_volume", "trades", "taker_base_volume", "taker_quota_volume"
        ])
        for c in ["open", "high", "low", "close", "volume", "quota_volume",
                   "taker_base_volume", "taker_quota_volume"]:
            df[c] = df[c].astype(float)
        df["trades"] = df["trades"].astype(int)

        # Continuity check (skip for quantum — synthetic candle may cause gap)
        df = add_direction(df)
        df = df.reset_index(drop=True)
        df_feat = add_technical_features(df)

        df_train = df_feat.iloc[:-1].reset_index(drop=True)
        df_predict = df_feat.iloc[[-1]].reset_index(drop=True)

        try:
            strategy = get_strategy(strategy_name, strategy_params)
            strategy.fit(df_train, horizon=horizon)
            pred_arr = strategy.predict(df_predict, horizon=horizon)
            prob_arr = strategy.predict_proba(df_predict, horizon=horizon)

            pred = int(pred_arr[0])
            prob = float(prob_arr[0])
            label = "UP" if pred == 1 else ("DOWN" if pred == 0 else "UNDEFINED")
        except Exception as e:
            label = "ERROR"
            prob = 0.0
            pred = -1

        scenarios[scenario_name] = {
            "prediction": label,
            "probability": round(prob, 4),
            "strategy": strategy_name,
            "horizon": horizon,
            "synth_candle": {
                "open": round(synth_open, 2),
                "close": round(synth_close, 2),
                "high": round(synth_high, 2),
                "low": round(synth_low, 2),
                "direction": "UP" if direction_up else "DOWN",
            },
        }

    return {
        "market_slug": slug,
        "market_ts": market_ts,
        "scenarios": scenarios,
        **sync_info,
    }


async def get_prediction_candles(
    slug: str,
    window_size: int = 1000,
    tail: int = 200,
    table: str = "c_5m",
) -> Dict[str, Any]:
    """Return the last *tail* candles (from the *window_size* window) used for
    prediction on *slug*.  The frontend draws a chart from these."""
    import pandas as pd

    m_row = await db.fetchone("SELECT ts FROM poly_markets WHERE slug=%s", (slug,))
    if not m_row:
        return {"error": f"Market not found: {slug}"}
    market_ts = int(m_row[0])
    market_ts_us = market_ts * 1_000_000

    rows = await db.fetchall(
        f"""
        SELECT open_time, open, high, low, close, volume
        FROM {table}
        WHERE open_time <= %s
        ORDER BY open_time DESC
        LIMIT %s
        """,
        (market_ts_us, window_size),
    )

    if not rows:
        return {"error": "No candles found", "market_ts": market_ts}

    rows = list(reversed(rows))
    # Take only the last `tail` candles for the chart
    rows = rows[-tail:]

    candles = []
    for r in rows:
        ot = int(r[0])
        candles.append({
            "t": ot // 1_000_000,  # epoch seconds for JS
            "o": float(r[1]),
            "h": float(r[2]),
            "l": float(r[3]),
            "c": float(r[4]),
            "v": float(r[5]),
        })

    return {
        "slug": slug,
        "market_ts": market_ts,
        "window_size": window_size,
        "total_in_window": len(list(reversed(rows))),
        "candles": candles,
    }


async def get_orderbook_analysis(slug: str, asset_id: str, minutes: int = 60) -> List[Dict[str, Any]]:
    """Get order book data aggregated by minute for analysis"""
    rows = await db.fetchall(
        """
        SELECT 
            ts,
            best_bid_cents,
            best_ask_cents,
            mid_cents,
            bids_json,
            asks_json
        FROM poly_orderbook_snapshots
        WHERE slug=%s AND asset_id=%s
        ORDER BY ts ASC
        """,
        (slug, asset_id),
    )
    
    out = []
    for r in rows:
        ts, best_bid, best_ask, mid, bids_json, asks_json = r
        
        # Parse order book data
        try:
            asks = json.loads(asks_json) if asks_json else []
        except:
            asks = []
        
        # Calculate best ask as min price from asks list
        computed_best_ask = None
        try:
            ask_prices = [float(a["price"]) for a in asks if a.get("price") is not None]
            if ask_prices:
                computed_best_ask = min(ask_prices)
        except (ValueError, TypeError):
            computed_best_ask = best_ask  # fallback to stored value

        # Calculate ask depth
        ask_depth = 0
        try:
            ask_depth = sum(float(a["size"]) for a in asks if a.get("price") and a.get("size"))
        except (ValueError, TypeError):
            ask_depth = 0
        
        out.append({
            "ts": int(ts),
            "best_bid_cents": None,  # Always NULL
            "best_ask_cents": computed_best_ask,
            "mid_cents": None,      # Always NULL
            "spread_cents": None,   # Cannot calculate without bids
            "bid_depth": 0,         # Always 0
            "ask_depth": ask_depth,
            "bids": [],             # Always empty
            "asks": asks,
        })
    
    return out


async def get_latest_orderbook(slug: str, asset_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest order book snapshot for an outcome"""
    row = await db.fetchone(
        """
        SELECT 
            ts,
            best_bid_cents,
            best_ask_cents,
            mid_cents,
            bids_json,
            asks_json
        FROM poly_orderbook_snapshots
        WHERE slug=%s AND asset_id=%s
        ORDER BY ts DESC
        LIMIT 1
        """,
        (slug, asset_id),
    )
    
    if not row:
        return None
    
    ts, best_bid, best_ask, mid, bids_json, asks_json = row
    
    # Parse order book data
    try:
        asks = json.loads(asks_json) if asks_json else []
    except:
        asks = []
    
    # Recalculate best ask as min price
    computed_best_ask = None
    try:
        ask_prices = [float(a["price"]) for a in asks if a.get("price") is not None]
        if ask_prices:
            computed_best_ask = min(ask_prices)
    except (ValueError, TypeError):
        computed_best_ask = best_ask

    return {
        "ts": int(ts),
        "best_bid_cents": None,  # Always NULL
        "best_ask_cents": computed_best_ask,
        "mid_cents": None,      # Always NULL
        "bids": [],             # Always empty
        "asks": asks,
    }
