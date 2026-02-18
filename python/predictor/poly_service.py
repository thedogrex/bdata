import asyncio
import json
import time
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
    for ts in _compute_timestamps(count=5):
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

    # Get best ask
    best_ask = asks_cents[0]["price"] if asks_cents else None

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


async def poll_loop(stop_event: asyncio.Event, orderbook_interval_sec: int = 3) -> None:
    orderbook_interval_sec = int(orderbook_interval_sec)
    if orderbook_interval_sec <= 0:
        orderbook_interval_sec = 3

    # Track last update times for different market types
    last_active_update = 0
    last_future_update = 0
    
    while not stop_event.is_set():
        try:
            await refresh_tracked_markets()
            
            # Get current active timestamp
            current_ts = _get_current_ts()
            current_time = int(time.time())
            
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
                    future_markets.append((slug, ts))
            
            # Update active market every 3 seconds
            if current_time - last_active_update >= orderbook_interval_sec and active_markets:
                for slug, ts in active_markets[:1]:  # Only current active market
                    out_rows = await db.fetchall(
                        "SELECT asset_id FROM poly_outcomes WHERE slug=%s",
                        (slug,),
                    )
                    asset_ids = [r[0] for r in out_rows]
                    if asset_ids:
                        obs = await asyncio.to_thread(fetch_orderbooks, asset_ids)
                        for ob in obs:
                            asset_id = str(ob.get("asset_id") or "")
                            if asset_id:
                                await save_orderbook_snapshot(slug, asset_id, ob)
                last_active_update = current_time
            
            # Update future markets every 30 seconds (next 2 markets)
            if current_time - last_future_update >= 30 and future_markets:
                for slug, ts in future_markets[:2]:  # Next 2 future markets
                    out_rows = await db.fetchall(
                        "SELECT asset_id FROM poly_outcomes WHERE slug=%s",
                        (slug,),
                    )
                    asset_ids = [r[0] for r in out_rows]
                    if asset_ids:
                        obs = await asyncio.to_thread(fetch_orderbooks, asset_ids)
                        for ob in obs:
                            asset_id = str(ob.get("asset_id") or "")
                            if asset_id:
                                await save_orderbook_snapshot(slug, asset_id, ob)
                last_future_update = current_time

        except Exception as e:
            print(f"Error in poll loop: {e}")
            pass

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=orderbook_interval_sec)
        except asyncio.TimeoutError:
            continue


async def list_markets(limit: int = 50) -> List[Dict[str, Any]]:
    current_ts = _get_current_ts()
    rows = await db.fetchall(
        "SELECT slug, ts, end_date, question, closed FROM poly_markets ORDER BY ts DESC LIMIT %s",
        (int(limit),),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        slug, ts, end_date, question, closed = r
        status = "open"
        
        # Mark past markets as [DONE]
        if closed:
            status = "[DONE]"
        elif ts < current_ts:
            # Past but not marked as closed yet
            status = "[DONE]"
        
        out.append({
            "slug": slug, 
            "ts": int(ts), 
            "end_date": end_date, 
            "question": question, 
            "closed": int(closed),
            "status": status
        })
    return out


async def get_market(slug: str) -> Optional[Dict[str, Any]]:
    m = await db.fetchone(
        "SELECT slug, ts, end_date, question, description, closed FROM poly_markets WHERE slug=%s",
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
        "outcomes": [{"asset_id": r[0], "name": r[1]} for r in o_rows],
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


async def create_sim_trade(slug: str, asset_id: str, side: str, qty: float) -> Dict[str, Any]:
    slug = str(slug or "").strip()
    if not slug:
        raise ValueError("slug is required")

    side = (side or "").upper().strip()
    if side != "BUY":
        raise ValueError("Only BUY orders are supported (asks-only)")

    qty = float(qty)
    if qty <= 0:
        raise ValueError("qty must be > 0")

    # Validate asset belongs to this market
    market = await get_market(slug)
    if not market:
        raise ValueError(f"Unknown market slug: {slug}")
    outcomes = market.get("outcomes") or []
    if str(asset_id) not in {str(o.get("asset_id")) for o in outcomes}:
        raise ValueError("asset_id does not belong to this market")

    row = await db.fetchone(
        """
        SELECT slug, ts, best_ask_cents
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

    fill = best_ask
    if fill is None:
        raise RuntimeError("No ask price available for this asset_id yet")

    trade_ts = int(time.time())

    trade_id = await db.execute(
        """
        INSERT INTO poly_sim_trades
            (ts, slug, asset_id, side, qty, fill_price_cents, snapshot_ts)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (trade_ts, slug, str(asset_id), side, qty, float(fill), snap_ts),
    )

    return {
        "id": int(trade_id),
        "ts": trade_ts,
        "slug": slug,
        "asset_id": str(asset_id),
        "side": side,
        "qty": qty,
        "fill_price_cents": float(fill),
        "snapshot_ts": snap_ts,
    }


async def list_sim_trades(limit: int = 200) -> List[Dict[str, Any]]:
    rows = await db.fetchall(
        """
        SELECT id, ts, slug, asset_id, side, qty, fill_price_cents
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
            "qty": float(r[5]),
            "fill_price_cents": float(r[6]),
        }
        for r in rows
    ]


async def get_sim_positions() -> List[Dict[str, Any]]:
    rows = await db.fetchall(
        """
        SELECT asset_id,
               SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END) as pos_qty,
               SUM(CASE WHEN side='BUY' THEN qty*fill_price_cents ELSE -qty*fill_price_cents END) as cashflow
        FROM poly_sim_trades
        GROUP BY asset_id
        """
    )

    out: List[Dict[str, Any]] = []
    for r in rows:
        asset_id = str(r[0])
        pos_qty = float(r[1]) if r[1] is not None else 0.0
        cashflow = float(r[2]) if r[2] is not None else 0.0

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


async def get_orderbook_analysis(slug: str, asset_id: str, minutes: int = 60) -> List[Dict[str, Any]]:
    """Get order book data aggregated by minute for analysis"""
    since_ts = int(time.time()) - (minutes * 60)
    
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
        WHERE slug=%s AND asset_id=%s AND ts>=%s
        ORDER BY ts ASC
        """,
        (slug, asset_id, since_ts),
    )
    
    out = []
    for r in rows:
        ts, best_bid, best_ask, mid, bids_json, asks_json = r
        
        # Parse order book data
        try:
            asks = json.loads(asks_json) if asks_json else []
        except:
            asks = []
        
        # Calculate ask depth
        ask_depth = 0
        try:
            ask_depth = sum(float(a["size"]) for a in asks if a.get("price") and a.get("size"))
        except (ValueError, TypeError):
            ask_depth = 0
        
        out.append({
            "ts": int(ts),
            "best_bid_cents": None,  # Always NULL
            "best_ask_cents": best_ask,
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
    
    return {
        "ts": int(ts),
        "best_bid_cents": None,  # Always NULL
        "best_ask_cents": best_ask,
        "mid_cents": None,      # Always NULL
        "bids": [],             # Always empty
        "asks": asks,
    }
