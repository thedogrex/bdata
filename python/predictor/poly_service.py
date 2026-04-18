import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from db import DbProvider

import app.config as config
from predictor import telegram_bot
from predictor.poly_client import PolymarketClient, MarketData
from predictor.utils.async_utils import resolve_awaitable

db = DbProvider()
logger = logging.getLogger("poly_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

PRED_DATA_LOG_DIR = Path(__file__).resolve().parent / "pred_data_logs"

MAX_PRICE_CAP_CENTS = 53
BET_SIZE_CONFIRM_TTL_SEC = 60
MSK_UTC_OFFSET_HOURS = 3
MSK_TZ_NAME = "+03:00"
DAILY_START_HOUR_MSK = 8
DAILY_REPORT_MINUTE_MSK = 5

_current_bet_size_request: Optional[Dict[str, Any]] = None
_request_cleanup_deadline: Optional[datetime] = None
_daily_balance_date: Optional[date] = None
_daily_balance_start_usd: Optional[float] = None
_last_balance_report_day: Optional[date] = None

DEFAULT_LIVE_TRADE_SETTINGS = {
    "auto_place": False,
    "bet_size_usd": 5.0,
    "price_cap_cents": 52,
}


def _now_ts() -> int:
    return int(time.time())


def _utcnow() -> datetime:
    return datetime.utcnow()


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _maybe_log_prediction_window(
    slug: str,
    market_ts: int,
    rows: List[Tuple[Any, ...]],
    window_size: int,
    table: str,
    prediction_label: str,
) -> None:
    if not getattr(config, "LOG_PRED_DATA_FILES", False):
        return

    try:
        PRED_DATA_LOG_DIR.mkdir(parents=True, exist_ok=True)

        label_safe = (prediction_label or "undefined").strip().lower()
        if label_safe not in {"up", "down", "undefined"}:
            label_safe = "undefined"
        prefix = f"{market_ts}_{label_safe}_"
        prediction_index = 0
        while (PRED_DATA_LOG_DIR / f"{prefix}{prediction_index}.json").exists():
            prediction_index += 1

        sorted_rows = sorted(rows, key=lambda r: int(r[0]), reverse=True)

        candles_payload: List[Dict[str, Any]] = []
        for idx, row in enumerate(sorted_rows):
            open_time_us = int(row[0])
            close_val = row[4]
            try:
                close_float = float(close_val) if close_val is not None else None
            except (TypeError, ValueError):
                close_float = None
            candles_payload.append(
                {
                    "idx": idx,
                    "ts": open_time_us // 1_000_000,
                    "ts_us": open_time_us,
                    "close": close_float,
                }
            )

        payload = {
            "slug": slug,
            "market_ts": market_ts,
            "prediction_index": prediction_index,
            "window_size": window_size,
            "table": table,
            "prediction_label": prediction_label,
            "candles_total": len(candles_payload),
            "logged_at_ts": int(time.time()),
            "candles": candles_payload,
        }

        log_path = PRED_DATA_LOG_DIR / f"{prefix}{prediction_index}.json"
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("[pred_log] failed to persist prediction window", exc_info=True)


def _current_request_status() -> Optional[str]:
    global _current_bet_size_request
    if not _current_bet_size_request:
        return None
    status = _current_bet_size_request.get("status")
    if status in ("approved", "rejected", "expired"):
        return status
    expires_at: Optional[datetime] = _current_bet_size_request.get("expires_at")
    if expires_at and expires_at < _utcnow():
        _current_bet_size_request["status"] = "expired"
        _current_bet_size_request.setdefault("resolved_at", _utcnow())
        return "expired"
    return status or "pending"


def _schedule_cleanup(seconds: int = 30) -> None:
    global _request_cleanup_deadline
    _request_cleanup_deadline = _utcnow() + timedelta(seconds=max(1, int(seconds)))


def _cleanup_request_if_needed(force: bool = False) -> None:
    global _current_bet_size_request, _request_cleanup_deadline
    if not _current_bet_size_request:
        _request_cleanup_deadline = None
        return
    status = _current_request_status()
    if status == "pending" and not force:
        return
    deadline = _request_cleanup_deadline
    if not deadline:
        if status in ("approved", "rejected", "expired"):
            _schedule_cleanup(5)
            deadline = _request_cleanup_deadline
        else:
            return
    if force or (deadline and deadline <= _utcnow()):
        _current_bet_size_request = None
        _request_cleanup_deadline = None


def _reset_request_if_done() -> None:
    _cleanup_request_if_needed(force=True)


def _get_cached_balance_display() -> str:
    from predictor import live_trading

    bal = live_trading.get_cached_collateral_balance_usd()
    if bal is None:
        return "н/д"
    return f"{bal:.2f}"


def _msk_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=MSK_UTC_OFFSET_HOURS)


def _balance_day_for_time(now_msk: Optional[datetime] = None) -> date:
    now = now_msk or _msk_now()
    day = now.date()
    if now.hour < DAILY_START_HOUR_MSK:
        day = day - timedelta(days=1)
    return day


async def _ensure_daily_balance_state(current_balance: Optional[float]) -> None:
    global _daily_balance_date, _daily_balance_start_usd

    day = _balance_day_for_time()
    if _daily_balance_date == day and _daily_balance_start_usd is not None:
        return

    row = await db.fetchone(
        "SELECT start_balance_usd FROM poly_daily_balance_digest WHERE digest_date=%s",
        (day,),
    )

    if row and row[0] is not None:
        _daily_balance_date = day
        _daily_balance_start_usd = float(row[0])
        return

    if current_balance is None:
        return

    start_value = float(current_balance)
    if row:
        await db.execute(
            "UPDATE poly_daily_balance_digest SET start_balance_usd=%s WHERE digest_date=%s",
            (start_value, day),
        )
    else:
        await db.execute(
            "INSERT INTO poly_daily_balance_digest (digest_date, start_balance_usd) VALUES (%s, %s)",
            (day, start_value),
        )

    _daily_balance_date = day
    _daily_balance_start_usd = start_value


async def _get_start_balance_for_day(day: date) -> Optional[float]:
    row = await db.fetchone(
        "SELECT start_balance_usd FROM poly_daily_balance_digest WHERE digest_date=%s",
        (day,),
    )
    if not row or row[0] is None:
        return None
    return float(row[0])


async def _was_report_sent(day: date) -> bool:
    row = await db.fetchone(
        "SELECT report_sent_at FROM poly_daily_balance_digest WHERE digest_date=%s",
        (day,),
    )
    return bool(row and row[0])


async def _mark_report_sent(day: date) -> None:
    await db.execute(
        """
        INSERT INTO poly_daily_balance_digest (digest_date, start_balance_usd, report_sent_at)
        VALUES (%s, %s, UTC_TIMESTAMP())
        ON DUPLICATE KEY UPDATE report_sent_at=UTC_TIMESTAMP()
        """,
        (day, None),
    )


async def _maybe_send_daily_balance_report() -> None:
    if not config.TELEGRAM_DAILY_REPORTS_ENABLED:
        return
    global _last_balance_report_day

    now_msk = _msk_now()
    if now_msk.hour < DAILY_START_HOUR_MSK or (
        now_msk.hour == DAILY_START_HOUR_MSK and now_msk.minute < DAILY_REPORT_MINUTE_MSK
    ):
        return

    current_day = _balance_day_for_time(now_msk)
    report_day = current_day - timedelta(days=1)
    if _last_balance_report_day == report_day:
        return

    if await _was_report_sent(report_day):
        _last_balance_report_day = report_day
        return

    prev_start = await _get_start_balance_for_day(report_day)
    current_start = await _get_start_balance_for_day(current_day)
    if prev_start is None or current_start is None:
        return

    delta = current_start - prev_start
    date_label = report_day.strftime("%d.%m.%Y")
    text = (
        f"Дневной отчёт ({date_label} МСК)\n"
        f"Старт предыдущего дня: {prev_start:.2f}$\n"
        f"Старт сегодняшнего дня: {current_start:.2f}$\n"
        f"Δ за день: {delta:+.2f}$"
    )
    try:
        await telegram_bot.notify_info_chats(text)
        await _mark_report_sent(report_day)
        _last_balance_report_day = report_day
    except Exception as exc:
        logger.warning("Failed to send daily balance report: %s", exc)


def get_bet_size_request_state() -> Optional[Dict[str, Any]]:
    _cleanup_request_if_needed()
    status = _current_request_status()
    if not status:
        return None
    req = dict(_current_bet_size_request or {})
    if not req:
        return None
    expires_at = req.get("expires_at")
    ttl_sec = None
    if isinstance(expires_at, datetime):
        ttl_sec = max(0, int((expires_at - _utcnow()).total_seconds()))
    return {
        "id": req.get("id"),
        "status": status,
        "requested_bet_size": req.get("requested_bet_size"),
        "previous_bet_size": req.get("previous_bet_size"),
        "expires_in_sec": ttl_sec,
        "requested_at": req.get("requested_at").isoformat() if isinstance(req.get("requested_at"), datetime) else None,
        "actor": req.get("actor"),
        "resolved_at": req.get("resolved_at").isoformat() if isinstance(req.get("resolved_at"), datetime) else None,
        "reason": req.get("reason"),
    }


async def _notify_admin_request(req: Dict[str, Any]) -> None:
    balance_display = req.get("balance_display", "н/д")
    text = (
        f"Текущий баланс: {balance_display}$\n"
        f"Вы хотите поменять размер ставки на {req.get('requested_bet_size'):,.2f}$?"
    )
    keyboard = telegram_bot.build_bet_size_keyboard(req["id"])
    await telegram_bot.notify_admin(text, reply_markup=keyboard)


async def _notify_info_change(bet_size: float, balance_display: str) -> None:
    text = (
        f"Текущий баланс: {balance_display}$\n"
        f"Размер ставки был изменён на {bet_size:.2f}$"
    )
    await telegram_bot.notify_info_chats(text)


def _has_pending_request() -> bool:
    status = _current_request_status()
    return status == "pending"


async def request_bet_size_change(auto_place: bool, bet_size_usd: float, price_cap_cents: int) -> Dict[str, Any]:
    global _current_bet_size_request

    current_settings = await get_live_trade_settings()
    prev_bet = float(current_settings.get("bet_size_usd", DEFAULT_LIVE_TRADE_SETTINGS["bet_size_usd"]))
    bet_changed = float(bet_size_usd) != float(prev_bet)

    if not bet_changed:
        saved = await save_live_trade_settings(
            auto_place=auto_place,
            bet_size_usd=bet_size_usd,
            price_cap_cents=price_cap_cents,
        )
        _reset_request_if_done()
        return {"status": "saved", "settings": saved}

    if _has_pending_request():
        return {"status": "pending", "request": get_bet_size_request_state()}
    _cleanup_request_if_needed(force=True)

    request_id = _new_request_id()
    expires_at = _utcnow() + timedelta(seconds=BET_SIZE_CONFIRM_TTL_SEC)
    req = {
        "id": request_id,
        "requested_bet_size": float(bet_size_usd),
        "previous_bet_size": prev_bet,
        "status": "pending",
        "requested_at": _utcnow(),
        "expires_at": expires_at,
        "actor": None,
        "auto_place": bool(auto_place),
        "price_cap_cents": int(price_cap_cents),
        "balance_display": _get_cached_balance_display(),
    }
    _current_bet_size_request = req
    try:
        await _notify_admin_request(req)
    except Exception as exc:
        logger.error("Failed to notify admin: %s", exc)
    return {
        "status": "pending",
        "request": get_bet_size_request_state(),
    }


async def approve_bet_size_request(request_id: str, actor: Optional[str] = None) -> Dict[str, Any]:
    global _current_bet_size_request
    if not _current_bet_size_request or _current_bet_size_request.get("id") != request_id:
        return {"status": "missing"}
    if _current_request_status() != "pending":
        status = _current_request_status()
        return {"status": status or "unknown"}

    _current_bet_size_request["status"] = "approved"
    _current_bet_size_request["actor"] = actor
    _current_bet_size_request["resolved_at"] = _utcnow()
    await save_live_trade_settings(
        auto_place=_current_bet_size_request.get("auto_place", False),
        bet_size_usd=float(_current_bet_size_request["requested_bet_size"]),
        price_cap_cents=int(_current_bet_size_request.get("price_cap_cents", DEFAULT_LIVE_TRADE_SETTINGS["price_cap_cents"])),
    )
    try:
        await _notify_info_change(
            float(_current_bet_size_request["requested_bet_size"]),
            _current_bet_size_request.get("balance_display", "н/д"),
        )
    except Exception as exc:
        logger.error("Failed to send info notification: %s", exc)

    _schedule_cleanup()
    resp = get_bet_size_request_state() or {"status": "approved"}
    return resp


async def reject_bet_size_request(request_id: str, actor: Optional[str] = None, reason: str | None = None) -> Dict[str, Any]:
    global _current_bet_size_request
    if not _current_bet_size_request or _current_bet_size_request.get("id") != request_id:
        return {"status": "missing"}
    status = _current_request_status()
    if status != "pending":
        return {"status": status or "unknown"}

    _current_bet_size_request["status"] = "rejected"
    _current_bet_size_request["actor"] = actor
    _current_bet_size_request["reason"] = reason
    _current_bet_size_request["resolved_at"] = _utcnow()
    _schedule_cleanup()
    resp = get_bet_size_request_state() or {"status": "rejected"}
    return resp


async def cancel_bet_size_request() -> Dict[str, Any]:
    global _current_bet_size_request
    if not _current_bet_size_request:
        return {"status": "missing"}
    _current_bet_size_request["status"] = "rejected"
    _current_bet_size_request["actor"] = "frontend"
    _current_bet_size_request["resolved_at"] = _utcnow()
    _current_bet_size_request.setdefault("reason", "cancelled_by_ui")
    _schedule_cleanup()
    resp = get_bet_size_request_state() or {"status": "rejected"}
    return resp


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
    """Infer resolved outcome from explicit Polymarket market resolution fields.

    Uses final_price vs target_price from Gamma market payload.
    Returns 'UP' or 'DOWN' when both values are present, otherwise None.
    """
    if not m:
        return None

    # Prefer explicit market resolution inputs when available.
    try:
        final_price = getattr(m, "final_price", None)
        target_price = getattr(m, "target_price", None)
        if final_price is not None and target_price is not None:
            return "DOWN" if float(final_price) < float(target_price) else "UP"
    except Exception:
        pass

    # Fallback: check if market is closed and try to infer from outcome prices
    try:
        if m.closed and m.outcomes and len(m.outcomes) >= 2:
            # For binary markets, the outcome with higher price is the winner
            prices = [(o.name, float(o.price)) for o in m.outcomes]
            prices.sort(key=lambda x: x[1], reverse=True)
            if prices[0][1] > prices[1][1]:
                winner = prices[0][0].upper()
                if winner in ("YES", "UP"):
                    return "UP"
                elif winner in ("NO", "DOWN"):
                    return "DOWN"
    except Exception:
        pass

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
        logger.debug("Resolution check for %s: closed=%s, final_price=%s, target_price=%s, inferred=%s",
                     slug, getattr(m, 'closed', None), getattr(m, 'final_price', None),
                     getattr(m, 'target_price', None), resolved)
    except Exception as e:
        logger.warning("Resolution fetch failed for %s: %s", slug, e)
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


async def refresh_tracked_markets(now: Optional[int] = None, count: int = 3) -> List[Dict[str, Any]]:
    client = PolymarketClient()
    rows: List[Dict[str, Any]] = []
    for ts in _compute_timestamps(now=now, count=int(count)):
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

        min_target_ts = int(ended_market_ts) + 600
        max_target_ts = min_target_ts + 100

        targets = await db.fetchall(
            """
            SELECT slug, ts
            FROM poly_markets
            WHERE closed=0 AND ts >= %s AND ts <= %s
            ORDER BY ts ASC
            LIMIT 1
            """,
            (int(min_target_ts), int(max_target_ts)),
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
                WHERE closed=0 AND ts >= %s AND ts <= %s
                ORDER BY ts ASC
                LIMIT 1
                """,
                (int(min_target_ts), int(max_target_ts)),
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
    last_active_missing_refresh = 0
    last_balance_refresh = 0
    last_daily_report_check = 0
    last_order_flow_report = 0
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

            # Daily balance report check (every 5 minutes 30 seconds).
            if config.TELEGRAM_DAILY_REPORTS_ENABLED and current_time - last_daily_report_check >= 330:
                try:
                    await _maybe_send_daily_balance_report()
                except Exception as exc:
                    logger.warning("[poll_loop] daily balance report check failed: %s", exc)
                finally:
                    last_daily_report_check = current_time

            if getattr(config, "TELEGRAM_ORDER_FLOW_INFO", None) and current_time - last_order_flow_report >= 3600:
                last_order_flow_report = current_time
                try:
                    from predictor import live_trading

                    text = await live_trading.get_today_order_flow_report_text()
                    for chat_id in getattr(config, "TELEGRAM_ORDER_FLOW_INFO", []) or []:
                        try:
                            await telegram_bot.send_message(chat_id, text)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("[poll_loop] order flow telegram report failed: %s", exc)

            # Refresh cached collateral balance every 15 minutes to reuse in Telegram confirmations.
            if current_time - last_balance_refresh >= 900:
                try:
                    from predictor import live_trading  # local import to avoid circular dep

                    refreshed = await live_trading.refresh_collateral_balance()
                    cached_balance = live_trading.get_cached_collateral_balance_usd()
                    await _ensure_daily_balance_state(cached_balance)
                    if refreshed is not None:
                        logger.debug("[poll_loop] collateral balance refreshed: %.2f$", float(refreshed))
                except Exception as exc:
                    logger.warning("[poll_loop] failed to refresh collateral balance: %s", exc)
                finally:
                    last_balance_refresh = current_time
            
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

                # After end: ensure upcoming markets exist in DB and prefetch ask history (snapshots)
                try:
                    await refresh_tracked_markets(now=int(current_ts), count=3)
                except Exception:
                    pass
                try:
                    upcoming_ts = _compute_timestamps(now=int(current_ts), count=3)
                    for ts in upcoming_ts:
                        asyncio.create_task(_take_orderbook_snapshot_for_slug(_slug_for_ts(int(ts))))
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
                    # (skip markets starting in 21+ minutes)
                    seconds_to_start = int(ts) - int(current_time)
                    if 0 < seconds_to_start <= 20 * 60:
                        future_markets.append((slug, ts))

            # If active market isn't present in DB (or missing outcomes), refresh it periodically.
            # Without outcomes in poly_outcomes, snapshots can't be saved.
            if not active_markets and current_time - last_active_missing_refresh >= 10:
                last_active_missing_refresh = current_time
                try:
                    await refresh_tracked_markets(now=int(current_ts), count=3)
                except Exception as e:
                    print(f"[poly] refresh_tracked_markets for active failed: {e}")
            
            # Update active market every 3 seconds
            if current_time - last_active_update >= orderbook_interval_sec:
                if active_markets:
                    for slug, ts in active_markets[:1]:  # Only current active market
                        try:
                            await _take_orderbook_snapshot_for_slug(slug)
                        except Exception as e:
                            print(f"[poly] active orderbook snapshot failed (slug={slug}): {e}")
                    last_active_update = current_time
                else:
                    # Log when active market is missing to help diagnose
                    if current_time % 10 == 0:  # Log every 10 seconds to avoid spam
                        print(f"[poly] No active market found (current_ts={current_ts})")
            
            # Update future markets every 10 seconds (next 4 markets)
            if current_time - last_future_update >= 10 and future_markets:
                for slug, ts in future_markets[:4]:  # Next 4 future markets
                    try:
                        await _take_orderbook_snapshot_for_slug(slug)
                    except Exception as e:
                        print(f"[poly] future orderbook snapshot failed (slug={slug}): {e}")
                last_future_update = current_time

            # Keep sets bounded to avoid memory leak
            if len(autopredicted_slugs) > 50:
                autopredicted_slugs = set(list(autopredicted_slugs)[-30:])
            if len(autopredicted_ended_ts) > 100:
                autopredicted_ended_ts = set(list(autopredicted_ended_ts)[-60:])

            # Resolution polling: once per minute scan DONE markets without resolution.
            if current_time - last_resolution_scan >= 60:
                last_resolution_scan = current_time
                interval = int(getattr(config, "POLY_INTERVAL_SECONDS", 300))
                if interval <= 0:
                    interval = 300
                min_auto_resolve_ts = int(current_ts) - interval * 100  # Extended to catch older markets
                done_rows = await db.fetchall(
                    """
                    SELECT slug
                    FROM poly_markets
                    WHERE (closed=1 OR ts < %s)
                      AND ts >= %s
                      AND (resolved_outcome IS NULL OR resolved_outcome='')
                      AND (last_resolution_check_ts IS NULL OR last_resolution_check_ts < %s)
                    ORDER BY ts DESC
                    LIMIT 20
                    """,
                    (int(current_ts), int(min_auto_resolve_ts), int(current_time) - 60),
                )
                logger.debug("Resolution scan: found %d markets to check (current_ts=%s, min_ts=%s)",
                             len(done_rows), current_ts, min_auto_resolve_ts)
                resolved_count = 0
                for (slug,) in done_rows:
                    try:
                        result = await _check_and_store_market_resolution(str(slug), current_time)
                        if result:
                            resolved_count += 1
                    except Exception as e:
                        logger.warning("Resolution check error for %s: %s", slug, e)
                if done_rows:
                    logger.info("Resolution scan complete: checked %d markets, resolved %d",
                                len(done_rows), resolved_count)

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
        VALUES ('default', %s, %s, %s, %s) AS new
        ON DUPLICATE KEY UPDATE
            autopredict=new.autopredict,
            strategy=new.strategy,
            params_json=new.params_json,
            window_size=new.window_size
        """,
        (int(autopredict), strategy, params_json, int(window_size)),
    )
    return await get_settings()


def _normalize_live_trade_settings(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_LIVE_TRADE_SETTINGS)
    if data:
        merged.update({k: data.get(k) for k in merged.keys() if data.get(k) is not None})

    bet_size = float(merged.get("bet_size_usd", DEFAULT_LIVE_TRADE_SETTINGS["bet_size_usd"]) or DEFAULT_LIVE_TRADE_SETTINGS["bet_size_usd"])
    bet_size = max(0.0, bet_size)
    price_cap = int(merged.get("price_cap_cents", DEFAULT_LIVE_TRADE_SETTINGS["price_cap_cents"]) or DEFAULT_LIVE_TRADE_SETTINGS["price_cap_cents"])
    price_cap = max(1, min(MAX_PRICE_CAP_CENTS, price_cap))

    return {
        "auto_place": bool(merged.get("auto_place", False)),
        "bet_size_usd": bet_size,
        "price_cap_cents": price_cap,
    }


async def get_live_trade_settings() -> Dict[str, Any]:
    row = await db.fetchone(
        "SELECT auto_place, bet_size_usd, price_cap_cents FROM poly_live_trade_settings WHERE id='default'"
    )
    if not row:
        return dict(DEFAULT_LIVE_TRADE_SETTINGS)
    auto_place, bet_size, price_cap = row
    return _normalize_live_trade_settings(
        {
            "auto_place": bool(auto_place),
            "bet_size_usd": bet_size,
            "price_cap_cents": price_cap,
        }
    )


async def save_live_trade_settings(
    auto_place: bool,
    bet_size_usd: float,
    price_cap_cents: int,
) -> Dict[str, Any]:
    normalized = _normalize_live_trade_settings(
        {
            "auto_place": auto_place,
            "bet_size_usd": bet_size_usd,
            "price_cap_cents": price_cap_cents,
        }
    )

    await db.execute(
        """
        INSERT INTO poly_live_trade_settings
            (id, auto_place, bet_size_usd, price_cap_cents)
        VALUES ('default', %s, %s, %s) AS new
        ON DUPLICATE KEY UPDATE
            auto_place=new.auto_place,
            bet_size_usd=new.bet_size_usd,
            price_cap_cents=new.price_cap_cents
        """,
        (
            int(bool(normalized["auto_place"])),
            float(normalized["bet_size_usd"]),
            int(normalized["price_cap_cents"]),
        ),
    )

    return await get_live_trade_settings()


async def list_markets(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
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
        OFFSET %s
        """,
        (int(limit), int(max(0, offset))),
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


async def list_prediction_updates(since_ts: int = 0, limit: int = 20) -> Dict[str, Any]:
    """Return newly created predictions for future markets since `since_ts`.

    This is used by the frontend to reliably detect backend-generated predictions.
    Only defined predictions (UP/DOWN) are returned.
    """
    now_ts = int(time.time())
    since_ts = int(since_ts or 0)
    limit = max(1, min(100, int(limit or 20)))

    emulate_down = bool(getattr(config, "EMULATE_DOWN", False))
    outcomes_sql = "('UP','DOWN','UNDEFINED')" if emulate_down else "('UP','DOWN')"

    rows = await db.fetchall(
        """
        SELECT slug, ts, prediction_outcome, prediction_ts
        FROM poly_markets
        WHERE prediction_ts IS NOT NULL
          AND prediction_ts > %s
          AND prediction_outcome IN """ + outcomes_sql + """
          AND ts > %s
          AND closed = 0
        ORDER BY prediction_ts ASC
        LIMIT %s
        """,
        (since_ts, now_ts, limit),
    )

    updates: List[Dict[str, Any]] = []
    for slug, ts, outcome, pts in rows:
        updates.append({
            "slug": str(slug),
            "ts": int(ts),
            "prediction_outcome": str(outcome),
            "prediction_ts": int(pts) if pts is not None else None,
        })

    max_ts = since_ts
    for u in updates:
        if u.get("prediction_ts") and int(u["prediction_ts"]) > max_ts:
            max_ts = int(u["prediction_ts"])

    return {
        "now": now_ts,
        "since": since_ts,
        "cursor": max_ts,
        "updates": updates,
        "emulate_down": emulate_down,
    }


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


async def set_market_resolution_manual(slug: str, outcome: str) -> Dict[str, Any]:
    side = str(outcome or "").strip().upper()
    if side not in ("UP", "DOWN"):
        return {"error": "outcome must be UP or DOWN", "slug": slug}

    row = await db.fetchone(
        "SELECT slug, ts, closed, resolved_outcome FROM poly_markets WHERE slug=%s",
        (slug,),
    )
    if not row:
        return {"error": "Market not found", "slug": slug}

    now_ts = int(time.time())
    await db.execute(
        """
        UPDATE poly_markets
        SET resolved_outcome=%s,
            last_resolution_check_ts=%s
        WHERE slug=%s
        """,
        (side, now_ts, slug),
    )

    updated = await db.fetchone(
        "SELECT slug, ts, closed, resolved_outcome, last_resolution_check_ts FROM poly_markets WHERE slug=%s",
        (slug,),
    )
    if not updated:
        return {"error": "Market not found after update", "slug": slug}

    return {
        "slug": str(updated[0]),
        "ts": int(updated[1]) if updated[1] is not None else None,
        "closed": int(updated[2]) if updated[2] is not None else 0,
        "resolved_outcome": updated[3],
        "resolved_now": side,
        "last_resolution_check_ts": int(updated[4]) if updated[4] is not None else now_ts,
        "manual": True,
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
    from predictor.candle_sync import sync_candles_up_to, check_and_fill_gaps

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

    async def _load_rows():
        return await db.fetchall(
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

    rows = await _load_rows()

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
    def _compute_missing(_rows):
        _last_open = int(_rows[-1][0])
        _missing = max(0, int((market_ts_us - _last_open) / interval_us))
        return _last_open, _missing

    last_candle_open_us, missing_candles = _compute_missing(rows)
    shifted = missing_candles > 0

    if missing_candles > 0 and missing_candles > horizon:
        # Candle might appear a moment later. Retry sync once after 1 second.
        try:
            await asyncio.sleep(1)
            retry_sync = await sync_candles_up_to(market_ts, window_candles=window_size + 50, table=table)
            sync_info["sync_retry"] = retry_sync
            if retry_sync.get("downloaded", 0) > 0:
                try:
                    sync_info["gap_fill_retry"] = await check_and_fill_gaps(
                        market_ts, window_candles=window_size, table=table
                    )
                except Exception:
                    pass
            rows2 = await _load_rows()
            if rows2 and len(rows2) >= need:
                rows2 = list(reversed(rows2))
                last_candle_open_us, missing_candles = _compute_missing(rows2)
                shifted = missing_candles > 0
                rows = rows2
        except Exception as e:
            sync_info["sync_retry_error"] = str(e)

    if missing_candles > 0 and missing_candles > horizon:
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

    bb_series_arr = None
    if hasattr(strategy, "_compute_bb_pos"):
        try:
            bb_calc = strategy._compute_bb_pos(df_feat)
            bb_series_arr = np.array(bb_calc, dtype=float, copy=False)
            if len(bb_series_arr) != len(df_feat):
                bb_series_arr = None
        except Exception:
            bb_series_arr = None
    if bb_series_arr is None and "bb_pos" in df_feat.columns:
        bb_series_arr = df_feat["bb_pos"].values.astype(float, copy=True)

    vol_fast_arr = vol_slow_arr = vol_ratio_arr = None
    if hasattr(strategy, "_get_vol_metrics"):
        try:
            vol_metrics = strategy._get_vol_metrics(df_feat)
            if vol_metrics:
                vol_fast_arr = np.array(vol_metrics.get("fast"), dtype=float, copy=False)
                vol_slow_arr = np.array(vol_metrics.get("slow"), dtype=float, copy=False)
                vol_ratio_arr = np.array(vol_metrics.get("ratio"), dtype=float, copy=False)
                if len(vol_fast_arr) != len(df_feat):
                    vol_fast_arr = None
                if len(vol_slow_arr) != len(df_feat):
                    vol_slow_arr = None
                if len(vol_ratio_arr) != len(df_feat):
                    vol_ratio_arr = None
        except Exception:
            vol_fast_arr = vol_slow_arr = vol_ratio_arr = None

    pred_arr = await resolve_awaitable(strategy.predict(df_predict, horizon=horizon))
    prob_arr = await resolve_awaitable(strategy.predict_proba(df_predict, horizon=horizon))

    pred = int(pred_arr[0])
    prob = float(prob_arr[0])
    label = "UP" if pred == 1 else ("DOWN" if pred == 0 else "UNDEFINED")

    _maybe_log_prediction_window(
        slug=slug,
        market_ts=market_ts,
        rows=rows,
        window_size=window_size,
        table=table,
        prediction_label=label,
    )

    # --- Diagnostics: show WHY this candle got this signal ---
    period = strategy.params.get("rsi_period", 14)
    rsi_col = f"rsi_{period}" if f"rsi_{period}" in df_predict.columns else "rsi_14"
    pred_rsi = float(np.nan_to_num(df_predict[rsi_col].values[0], nan=50.0))
    if bb_series_arr is not None and len(bb_series_arr) == len(df_feat):
        pred_bb = float(np.nan_to_num(bb_series_arr[-1], nan=0.5))
    else:
        pred_bb = float(np.nan_to_num(df_predict.get("bb_pos", pd.Series([0.5])).values[0], nan=0.5)) if "bb_pos" in df_predict.columns else 0.5

    if vol_fast_arr is not None and len(vol_fast_arr) == len(df_feat):
        pred_vol = float(np.nan_to_num(vol_fast_arr[-1], nan=0.0))
    else:
        pred_vol = 0.0

    if vol_ratio_arr is not None and len(vol_ratio_arr) == len(df_feat):
        pred_vol_ratio = float(np.nan_to_num(vol_ratio_arr[-1], nan=0.0))
    else:
        pred_vol_ratio = 0.0

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
        if bb_series_arr is not None and len(bb_series_arr) > k:
            r_bb = float(np.nan_to_num(bb_series_arr[k], nan=0.5))
        else:
            r_bb = float(np.nan_to_num(row.get("bb_pos", 0.5), nan=0.5))
        if vol_fast_arr is not None and len(vol_fast_arr) > k:
            r_vol = float(np.nan_to_num(vol_fast_arr[k], nan=0.0))
        else:
            r_vol = float(np.nan_to_num(row.get("volatility_20", 0.0), nan=0.0))
        if vol_ratio_arr is not None and len(vol_ratio_arr) > k:
            r_ratio = float(np.nan_to_num(vol_ratio_arr[k], nan=0.0))
        else:
            r_ratio = None
        # Re-predict each context candle to show what backtest would have said
        df_k = df_feat.iloc[[k]].reset_index(drop=True)
        k_pred = int((await resolve_awaitable(strategy.predict(df_k, horizon=horizon)))[0])
        k_prob = float((await resolve_awaitable(strategy.predict_proba(df_k, horizon=horizon)))[0])
        tail_detail.append({
            "dt": dt_str,
            "rsi": round(r_rsi, 1),
            "bb": round(r_bb, 3),
            "vol": round(r_vol, 4),
            "vol_ratio": round(r_ratio, 4) if r_ratio is not None else None,
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
        "pred_volatility": round(pred_vol, 4),
        "pred_volatility_ratio": round(pred_vol_ratio, 4),
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

    # Backend auto-trade (optional): if confirmation is disabled, place order immediately.
    try:
        if not bool(getattr(config, "NEED_CONFIRMATION", True)):
            asyncio.create_task(_auto_trade_after_prediction(slug=slug, prediction=label))
    except Exception:
        pass

    return ret


async def _auto_trade_after_prediction(slug: str, prediction: str) -> None:
    try:
        pred = str(prediction or "").upper()
        emulate_down = bool(getattr(config, "EMULATE_DOWN", False))
        if pred == "UNDEFINED" and emulate_down:
            logger.info("[auto_trade] emulate DOWN for undefined prediction", {"slug": slug})
            pred = "DOWN"
        if pred not in ("UP", "DOWN"):
            logger.info("[auto_trade] skip: invalid prediction", {"slug": slug, "prediction": pred})
            return

        live_settings = await get_live_trade_settings()
        if not live_settings.get("auto_place"):
            logger.info("[auto_trade] skip: auto_place disabled", {"slug": slug})
            return

        # Trade only future markets
        m_row = await db.fetchone("SELECT ts, closed FROM poly_markets WHERE slug=%s", (slug,))
        if not m_row:
            logger.info("[auto_trade] skip: market metadata missing", {"slug": slug})
            return
        market_ts = int(m_row[0]) if m_row[0] is not None else 0
        closed = int(m_row[1]) if len(m_row) > 1 and m_row[1] is not None else 0
        if closed:
            logger.info("[auto_trade] skip: market already closed", {"slug": slug})
            return
        now_utc = int(time.time())
        if not (market_ts and now_utc < market_ts):
            logger.info(
                "[auto_trade] skip: market not in future",
                {"slug": slug, "market_ts": market_ts, "now": now_utc},
            )
            return

        # Resolve outcome asset_id for side
        o_rows = await db.fetchall("SELECT asset_id, name FROM poly_outcomes WHERE slug=%s", (slug,))
        if not o_rows:
            logger.info("[auto_trade] skip: no outcomes found", {"slug": slug})
            return
        up_id = None
        down_id = None
        for asset_id, name in o_rows:
            n = str(name or "").upper()
            if "UP" in n and not up_id:
                up_id = str(asset_id)
            if "DOWN" in n and not down_id:
                down_id = str(asset_id)
        # Fallback to first two
        if (not up_id or not down_id) and len(o_rows) >= 2:
            up_id = up_id or str(o_rows[0][0])
            down_id = down_id or str(o_rows[1][0])

        asset_id = down_id if pred == "DOWN" else up_id
        if not asset_id:
            logger.info("[auto_trade] skip: missing asset_id for outcome", {"slug": slug, "prediction": pred})
            return

        # Import here to avoid circular imports
        from predictor import live_trading

        if pred not in ("UP", "DOWN"):
            logger.info("[auto_trade] skip: non-tradable prediction", {"slug": slug, "prediction": pred})
            return

        bet_size_usd = float(live_settings.get("bet_size_usd", DEFAULT_LIVE_TRADE_SETTINGS["bet_size_usd"]) or DEFAULT_LIVE_TRADE_SETTINGS["bet_size_usd"])
        bet_size_usd = max(0.0, bet_size_usd)
        price_cap_cents = int(live_settings.get("price_cap_cents", 52) or 52)
        price_cap_cents = max(1, min(MAX_PRICE_CAP_CENTS, price_cap_cents))
        price_threshold = price_cap_cents / 100.0

        logger.info(
            "[auto_trade] placing order",
            {
                "slug": slug,
                "prediction": pred,
                "bet_size_usd": bet_size_usd,
                "price_cap_cents": price_cap_cents,
                "asset_id": asset_id,
            },
        )

        result = await live_trading.buy_after_prediction(
            slug=slug,
            asset_id=str(asset_id),
            outcome_side=pred,
            prediction_direction=pred,
            snapshot_price=0.0,
            price_threshold=price_threshold,
            batch_id=None,
            template_id=None,
            enable_price_wait=True,
        )
        logger.info("[auto_trade] order result", {"slug": slug, "result": result})
    except Exception as e:
        logger.exception("[auto_trade] exception", exc_info=e)
        return


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
            pred_arr = await resolve_awaitable(strategy.predict(df_predict, horizon=horizon))
            prob_arr = await resolve_awaitable(strategy.predict_proba(df_predict, horizon=horizon))

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


async def get_predictions_analytics(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    hour_from: Optional[int] = None,
    hour_to: Optional[int] = None,
) -> Dict[str, Any]:
    """Return prediction analytics with optional date and hour-of-day filters.

    Filters apply to pr.started_at (when the prediction was made).
    Only counts non-quantum, non-error, UP/DOWN predictions made before the market start.
    """
    where = [
        "pr.prediction IN ('UP','DOWN')",
        "pr.quantum = 0",
        "pr.error IS NULL",
        "pr.started_at < FROM_UNIXTIME(pm.ts)",
    ]
    params: list = []

    msk_started = f"CONVERT_TZ(pr.started_at, '+00:00', '{MSK_TZ_NAME}')"

    if date_from:
        where.append(f"DATE({msk_started}) >= %s")
        params.append(date_from)
    if date_to:
        where.append(f"DATE({msk_started}) <= %s")
        params.append(date_to)
    if hour_from is not None:
        where.append(f"HOUR({msk_started}) >= %s")
        params.append(int(hour_from))
    if hour_to is not None:
        where.append(f"HOUR({msk_started}) <= %s")
        params.append(int(hour_to))

    where_sql = " AND ".join(where)

    # ── Overall summary ──────────────────────────────────────────────────────
    summary_q = f"""
        SELECT
            COUNT(*)                                                      AS total_predictions,
            COUNT(DISTINCT pm.slug)                                       AS total_markets,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL THEN 1 ELSE 0 END), 0) AS resolved_count,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL
                      AND pr.prediction = pm.resolved_outcome THEN 1 ELSE 0 END), 0) AS correct_count,
            COUNT(DISTINCT DATE({msk_started}))                          AS active_days,
            COUNT(DISTINCT CONCAT(DATE({msk_started}), '-', LPAD(HOUR({msk_started}), 2, '0'))) AS active_day_hours
        FROM poly_pred_runs pr
        JOIN poly_markets pm ON pm.slug = pr.slug
        WHERE {where_sql}
    """
    row = await db.fetchone(summary_q, params or None)
    total_predictions = int(row[0]) if row else 0
    total_markets     = int(row[1]) if row else 0
    resolved_count    = int(row[2] or 0) if row else 0
    correct_count     = int(row[3] or 0) if row else 0
    active_days       = int(row[4]) if row else 0
    active_hour_slots = int(row[5]) if row else 0

    correct_pct = round(correct_count / resolved_count * 100, 1) if resolved_count > 0 else None

    # ── Per-day breakdown ────────────────────────────────────────────────────
    day_q = f"""
        SELECT
            DATE({msk_started})                                           AS day,
            COUNT(*)                                                      AS predictions,
            COUNT(DISTINCT pm.slug)                                       AS markets,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL THEN 1 ELSE 0 END), 0) AS resolved,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL
                      AND pr.prediction = pm.resolved_outcome THEN 1 ELSE 0 END), 0) AS correct
        FROM poly_pred_runs pr
        JOIN poly_markets pm ON pm.slug = pr.slug
        WHERE {where_sql}
        GROUP BY DATE({msk_started})
        ORDER BY day ASC
    """
    day_rows = await db.fetchall(day_q, params or None)
    per_day = []
    for r in day_rows:
        res = int(r[3] or 0)
        cor = int(r[4] or 0)
        per_day.append({
            "day":         str(r[0]),
            "predictions": int(r[1]),
            "markets":     int(r[2]),
            "resolved":    res,
            "correct":     cor,
            "correct_pct": round(cor / res * 100, 1) if res > 0 else None,
        })

    # ── Per-hour breakdown ───────────────────────────────────────────────────
    hour_q = f"""
        SELECT
            HOUR({msk_started})                                           AS hr,
            COUNT(*)                                                      AS predictions,
            COUNT(DISTINCT pm.slug)                                       AS markets,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL THEN 1 ELSE 0 END), 0) AS resolved,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL
                      AND pr.prediction = pm.resolved_outcome THEN 1 ELSE 0 END), 0) AS correct
        FROM poly_pred_runs pr
        JOIN poly_markets pm ON pm.slug = pr.slug
        WHERE {where_sql}
        GROUP BY HOUR({msk_started})
        ORDER BY hr ASC
    """
    hour_rows = await db.fetchall(hour_q, params or None)
    per_hour = []
    for r in hour_rows:
        res = int(r[3] or 0)
        cor = int(r[4] or 0)
        per_hour.append({
            "hour":        int(r[0]),
            "predictions": int(r[1]),
            "markets":     int(r[2]),
            "resolved":    res,
            "correct":     cor,
            "correct_pct": round(cor / res * 100, 1) if res > 0 else None,
        })

    avg_per_day  = round(sum(d.get("markets", 0) for d in per_day) / len(per_day), 1) if per_day else 0
    avg_per_hour = round(total_markets / active_hour_slots, 1) if active_hour_slots > 0 else 0

    # ── Per-template breakdown ────────────────────────────────────────────────
    tpl_q = f"""
        SELECT
            pr.template_name,
            COUNT(*)                                                      AS predictions,
            COUNT(DISTINCT pm.slug)                                       AS markets,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL THEN 1 ELSE 0 END), 0) AS resolved,
            COALESCE(SUM(CASE WHEN pm.resolved_outcome IS NOT NULL
                      AND pr.prediction = pm.resolved_outcome THEN 1 ELSE 0 END), 0) AS correct
        FROM poly_pred_runs pr
        JOIN poly_markets pm ON pm.slug = pr.slug
        WHERE {where_sql}
        GROUP BY pr.template_name
        ORDER BY predictions DESC
    """
    tpl_rows = await db.fetchall(tpl_q, params or None)
    per_template = []
    for r in tpl_rows:
        res = int(r[3] or 0)
        cor = int(r[4] or 0)
        per_template.append({
            "template":    r[0] or "—",
            "predictions": int(r[1]),
            "markets":     int(r[2]),
            "resolved":    res,
            "correct":     cor,
            "correct_pct": round(cor / res * 100, 1) if res > 0 else None,
        })

    return {
        "summary": {
            "total_predictions": total_predictions,
            "total_markets":     total_markets,
            "resolved_count":    resolved_count,
            "correct_count":     correct_count,
            "correct_pct":       correct_pct,
            "avg_per_day":       avg_per_day,
            "avg_per_hour":      avg_per_hour,
        },
        "per_day":      per_day,
        "per_hour":     per_hour,
        "per_template": per_template,
    }


async def get_ask_price_analysis(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    hour_from: Optional[int] = None,
    hour_to: Optional[int] = None,
    window_sec: int = 10,
) -> Dict[str, Any]:
    """For each prediction, wait window_sec seconds after finished_at and then find the
    FIRST orderbook snapshot at/after that time. Parse asks_json to compute available amounts per price
    level. Returns summary stats, cumulative amount buckets, a depth table, and per-day.
    """
    pred_where = [
        "pr.prediction IN ('UP','DOWN')",
        "pr.quantum = 0",
        "pr.error IS NULL",
        "pr.finished_at IS NOT NULL",
        "pr.started_at < FROM_UNIXTIME(pm.ts)",
    ]
    pred_params: list = []

    msk_finished = f"CONVERT_TZ(pr.finished_at, '+00:00', '{MSK_TZ_NAME}')"

    if date_from:
        pred_where.append(f"DATE({msk_finished}) >= %s")
        pred_params.append(date_from)
    if date_to:
        pred_where.append(f"DATE({msk_finished}) <= %s")
        pred_params.append(date_to)
    if hour_from is not None:
        pred_where.append(f"HOUR({msk_finished}) >= %s")
        pred_params.append(int(hour_from))
    if hour_to is not None:
        pred_where.append(f"HOUR({msk_finished}) <= %s")
        pred_params.append(int(hour_to))

    where_sql = " AND ".join(pred_where)
    ws = int(window_sec)

    def _f(v, digits=2):
        return round(float(v), digits) if v is not None else None

    # ── Step 1: count total predictions matching filters ─────────────────────
    total_q = f"""
        SELECT COUNT(DISTINCT pr.id)
        FROM poly_pred_runs pr
        JOIN poly_markets pm ON pm.slug = pr.slug
        WHERE {where_sql}
    """
    total_row = await db.fetchone(total_q, pred_params or None)
    total_preds = int(total_row[0] or 0) if total_row else 0

    # ── Step 2: fetch first snapshot (asks_json) per prediction ──────────────
    # Join to the earliest snapshot within the window so we get price+size detail.
    snap_q = f"""
        SELECT pr.id, obs.asks_json, obs.best_ask_cents, DATE({msk_finished}) AS day
        FROM poly_pred_runs pr
        JOIN poly_markets pm ON pm.slug = pr.slug
        JOIN poly_outcomes po
            ON  po.slug = pr.slug
            AND (
                (pr.prediction = 'UP'   AND (UPPER(po.name) LIKE '%%UP%%'   OR UPPER(po.name) LIKE '%%YES%%'))
             OR (pr.prediction = 'DOWN' AND (UPPER(po.name) LIKE '%%DOWN%%' OR UPPER(po.name) LIKE '%%NO%%'))
            )
        JOIN (
            SELECT pr2.id AS pred_id, pr2.slug AS slug, po2.asset_id AS asset_id, MIN(obs2.ts) AS first_ts
            FROM poly_pred_runs pr2
            JOIN poly_outcomes po2
                ON  po2.slug = pr2.slug
                AND (
                    (pr2.prediction = 'UP'   AND (UPPER(po2.name) LIKE '%%UP%%'   OR UPPER(po2.name) LIKE '%%YES%%'))
                 OR (pr2.prediction = 'DOWN' AND (UPPER(po2.name) LIKE '%%DOWN%%' OR UPPER(po2.name) LIKE '%%NO%%'))
                )
            JOIN poly_orderbook_snapshots obs2
                ON  obs2.slug = pr2.slug
                AND obs2.asset_id = po2.asset_id
                AND obs2.ts   >= UNIX_TIMESTAMP(pr2.finished_at) + {ws}
                AND obs2.asks_json IS NOT NULL
            GROUP BY pr2.id, pr2.slug, po2.asset_id
        ) fs ON fs.pred_id = pr.id
        JOIN poly_orderbook_snapshots obs
            ON  obs.slug = fs.slug
            AND obs.asset_id = fs.asset_id
            AND obs.ts   = fs.first_ts
        WHERE {where_sql}
        LIMIT 20000
    """
    snap_rows = await db.fetchall(snap_q, pred_params or None)

    # ── Step 3: process asks_json in Python ───────────────────────────────────
    THRESHOLDS = [51.0, 52.0, 53.0]
    # depth_buckets: price_bucket -> list of sizes at exactly that level
    from collections import defaultdict
    depth_totals: dict = defaultdict(list)   # bucket -> [cumulative_size_per_pred]

    preds_with_snap = 0
    min_asks: list = []
    cumul: dict = {t: [] for t in THRESHOLDS}   # threshold -> [cumul_size]

    per_day_raw: dict = defaultdict(lambda: {
        "preds": 0,
        "asks": [],
        **{f"cumul_{int(t)}": [] for t in THRESHOLDS},
    })

    for row in snap_rows:
        pred_id, asks_json_str, best_ask_cents, day = row
        day_str = str(day)

        try:
            asks = json.loads(asks_json_str) if asks_json_str else []
        except Exception:
            asks = []

        if not asks:
            continue

        # Normalise: ensure price is float cents
        levels: list[tuple[float, float]] = []
        for a in asks:
            try:
                p = float(a.get("price") or 0)
                s = float(a.get("size") or 0)
                if p > 0 and s > 0:
                    levels.append((p, s))
            except (TypeError, ValueError):
                continue

        if not levels:
            continue

        preds_with_snap += 1
        best_ask = min(p for p, _ in levels)
        min_asks.append(best_ask)

        # Cumulative sizes at each threshold
        for t in THRESHOLDS:
            total_size = sum(s for p, s in levels if p <= t)
            cumul[t].append(total_size)

        # Depth table: cumulative size at each 0.5c bucket (44–62c)
        bucket_range = [round(44.0 + i * 0.5, 1) for i in range(37)]  # 44.0 .. 62.0
        for b in bucket_range:
            c_size = sum(s for p, s in levels if p <= b)
            if c_size > 0:
                depth_totals[b].append(c_size)

        # Per-day
        d = per_day_raw[day_str]
        d["preds"] += 1
        d["asks"].append(best_ask)
        for t in THRESHOLDS:
            total_size = sum(s for p, s in levels if p <= t)
            d[f"cumul_{int(t)}"].append(total_size)

    # ── Step 4: aggregate ─────────────────────────────────────────────────────
    coverage_pct = round(preds_with_snap / total_preds * 100, 1) if total_preds > 0 else 0
    avg_min_ask  = _f(sum(min_asks) / len(min_asks)) if min_asks else None
    overall_min  = _f(min(min_asks)) if min_asks else None

    def _agg_cumul(threshold):
        vals = [v for v in cumul[threshold] if v > 0]
        cnt  = len(vals)
        avg  = _f(sum(vals) / cnt, 1) if cnt > 0 else None
        pct  = round(cnt / preds_with_snap * 100, 1) if preds_with_snap > 0 else 0
        avg_ask_vals = [a for a, c in zip(min_asks, cumul[threshold]) if a <= threshold]
        avg_ask = _f(sum(avg_ask_vals) / len(avg_ask_vals)) if avg_ask_vals else None
        return {"cnt": cnt, "pct": pct, "avg_amount": avg, "avg_ask": avg_ask}

    buckets = {str(int(t)): _agg_cumul(t) for t in THRESHOLDS}

    # Depth table: for each 0.5c bucket, how many predictions had liquidity + avg cumulative amount
    depth = []
    for b in sorted(depth_totals.keys()):
        vals = depth_totals[b]
        depth.append({
            "price":      b,
            "count":      len(vals),
            "pct":        round(len(vals) / preds_with_snap * 100, 1) if preds_with_snap > 0 else 0,
            "avg_cumul":  _f(sum(vals) / len(vals), 1),
        })

    # Per-day summary
    per_day = []
    for day_str in sorted(per_day_raw.keys()):
        d = per_day_raw[day_str]
        n = d["preds"]
        asks_list = d["asks"]
        entry = {
            "day":     day_str,
            "preds":   n,
            "avg_ask": _f(sum(asks_list) / len(asks_list)) if asks_list else None,
        }
        for t in THRESHOLDS:
            vals = [v for v in d[f"cumul_{int(t)}"] if v > 0]
            entry[f"cnt_le{int(t)}"]    = len(vals)
            entry[f"avg_amt_le{int(t)}"] = _f(sum(vals) / len(vals), 1) if vals else None
        per_day.append(entry)

    return {
        "window_sec":      ws,
        "summary": {
            "total_preds":     total_preds,
            "preds_with_snap": preds_with_snap,
            "coverage_pct":    coverage_pct,
            "avg_min_ask":     avg_min_ask,
            "overall_min_ask": overall_min,
        },
        "buckets":  buckets,   # {"51": {cnt, pct, avg_amount, avg_ask}, "52": ..., "53": ...}
        "depth":    depth,     # [{price, count, pct, avg_cumul}, ...]
        "per_day":  per_day,
    }


async def get_kelly_simulation(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    hour_from: Optional[int] = None,
    hour_to: Optional[int] = None,
    start_bank: float = 100.0,
    max_bet: Optional[float] = None,
    fee_rate: float = 0.0156,
    max_price_cents: float = 51.0,
    hk_pct: float = 0.017,
    fk_pct: float = 0.0334,
) -> Dict[str, Any]:
    """Simulate Half-Kelly and Full-Kelly strategies using real orderbook depth.

    For each resolved prediction the first snapshot at/after finished_at is used.
    The order book is walked in price order; only levels ≤ max_price_cents are used.
    If no such level exists the trade is skipped.  On win, shares pay $1 each; on
    loss the cost is lost.  Returns per-trade detail rows for a full audit table.
    """
    pred_where = [
        "pr.prediction IN ('UP','DOWN')",
        "pr.quantum = 0",
        "pr.error IS NULL",
        "pr.finished_at IS NOT NULL",
        "pr.probability IS NOT NULL",
        "pm.resolved_outcome IS NOT NULL",
        "pm.resolved_outcome IN ('UP','DOWN')",
        "pr.started_at < FROM_UNIXTIME(pm.ts)",
    ]
    pred_params: list = []

    msk_finished = f"CONVERT_TZ(pr.finished_at, '+00:00', '{MSK_TZ_NAME}')"

    if date_from:
        pred_where.append(f"DATE({msk_finished}) >= %s")
        pred_params.append(date_from)
    if date_to:
        pred_where.append(f"DATE({msk_finished}) <= %s")
        pred_params.append(date_to)
    if hour_from is not None:
        pred_where.append(f"HOUR({msk_finished}) >= %s")
        pred_params.append(int(hour_from))
    if hour_to is not None:
        pred_where.append(f"HOUR({msk_finished}) <= %s")
        pred_params.append(int(hour_to))

    where_sql = " AND ".join(pred_where)
    mpc = float(max_price_cents)
    fr  = float(fee_rate)
    hkf = float(hk_pct)
    fkf = float(fk_pct)

    sim_q = f"""
        SELECT
            pr.id,
            pr.slug,
            pr.prediction,
            pr.probability,
            pm.resolved_outcome,
            obs.asks_json,
            pr.finished_at,
            obs.ts AS applied_ts
        FROM poly_pred_runs pr
        JOIN poly_markets pm ON pm.slug = pr.slug
        JOIN (
            SELECT pr2.id AS pred_id, po2.asset_id AS asset_id, MIN(obs2.ts) AS first_ts
            FROM poly_pred_runs pr2
            JOIN poly_outcomes po2
                ON  po2.slug = pr2.slug
                AND (
                    (pr2.prediction = 'UP'   AND (UPPER(po2.name) LIKE '%%UP%%'   OR UPPER(po2.name) LIKE '%%YES%%'))
                 OR (pr2.prediction = 'DOWN' AND (UPPER(po2.name) LIKE '%%DOWN%%' OR UPPER(po2.name) LIKE '%%NO%%'))
                )
            JOIN poly_orderbook_snapshots obs2
                ON  obs2.slug     = pr2.slug
                AND obs2.asset_id = po2.asset_id
                AND obs2.ts       >= UNIX_TIMESTAMP(pr2.finished_at)
                AND obs2.asks_json IS NOT NULL
            GROUP BY pr2.id, po2.asset_id
        ) fs ON fs.pred_id = pr.id
        JOIN poly_orderbook_snapshots obs
            ON  obs.slug     = pr.slug
            AND obs.asset_id = fs.asset_id
            AND obs.ts       = fs.first_ts
            AND obs.asks_json IS NOT NULL
        WHERE {where_sql}
        ORDER BY pr.finished_at ASC
        LIMIT 20000
    """
    rows = await db.fetchall(sim_q, pred_params or None)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fill_book(levels: list, budget: float):
        """Walk sorted ask levels (already filtered by max_price) and fill.
        Returns (shares, dollars_spent, weighted_avg_fill_cents)."""
        remaining = budget
        total_shares = 0.0
        total_spent  = 0.0
        for price_c, size in levels:
            if remaining < 1e-6:
                break
            cost_per_share = price_c / 100.0
            if cost_per_share <= 0:
                continue
            shares_here = min(float(size), remaining / cost_per_share)
            cost_here   = shares_here * cost_per_share
            total_shares += shares_here
            total_spent  += cost_here
            remaining    -= cost_here
        avg_fill = (total_spent / total_shares * 100.0) if total_shares > 0 else 0.0
        return total_shares, total_spent, avg_fill

    def _kelly_frac(prob: float, ask_c: float) -> float:
        if ask_c <= 0 or ask_c >= 100:
            return 0.0
        f = (100.0 * prob - ask_c) / (100.0 - ask_c)
        return max(f, 0.0)

    # ── simulation state ──────────────────────────────────────────────────────
    bank_hk = float(start_bank)
    bank_fk = float(start_bank)
    trades_detail: list = []
    total_resolved = len(rows)
    skipped_price = 0

    for row in rows:
        pred_id, slug, prediction, probability, resolved_outcome, asks_json_str, finished_at, applied_ts = row
        day_str = str(finished_at)[:10] if finished_at else ""
        time_str = str(finished_at)[11:19] if finished_at and len(str(finished_at)) >= 19 else ""
        applied_time_utc = datetime.utcfromtimestamp(int(applied_ts)).strftime("%Y-%m-%d %H:%M:%S") if applied_ts is not None else ""

        try:
            prob = float(probability)
        except (TypeError, ValueError):
            prob = 0.5  # fallback – should not happen given SQL filter

        try:
            asks_raw = json.loads(asks_json_str) if asks_json_str else []
        except Exception:
            asks_raw = []

        # Parse and filter levels to max_price_cents
        levels = []
        for a in asks_raw:
            try:
                p = float(a.get("price") or 0)
                s = float(a.get("size")  or 0)
                if p > 0 and s > 0 and p <= mpc:
                    levels.append((p, s))
            except (TypeError, ValueError):
                continue
        levels.sort(key=lambda x: x[0])

        if not levels:
            skipped_price += 1
            trade_row: Dict[str, Any] = {
                "date":        day_str,
                "time":        time_str,
                "applied_time": applied_time_utc,
                "slug":        slug,
                "pred":        prediction,
                "outcome":     resolved_outcome,
                "correct":     (prediction == resolved_outcome),
                "best_ask":    None,
                "skipped":     True,
                "skip_reason": f"ask > {mpc}c",
                "hk_bet":      0.0,
                "hk_shares":   0.0,
                "hk_fill":     0.0,
                "hk_fee":      0.0,
                "hk_profit":   0.0,
                "hk_bank":     round(bank_hk, 2),
                "fk_bet":      0.0,
                "fk_shares":   0.0,
                "fk_fill":     0.0,
                "fk_fee":      0.0,
                "fk_profit":   0.0,
                "fk_bank":     round(bank_fk, 2),
            }
            trades_detail.append(trade_row)
            continue

        best_ask_c = levels[0][0]

        # Fixed bet sizing (percent of current bank)
        fk = fkf
        hk = hkf

        is_correct = (prediction == resolved_outcome)

        trade_row: Dict[str, Any] = {
            "date":       day_str,
            "time":       time_str,
            "applied_time": applied_time_utc,
            "slug":       slug,
            "pred":       prediction,
            "outcome":    resolved_outcome,
            "correct":    is_correct,
            "best_ask":   round(best_ask_c, 2),
            "skipped":    False,
            "skip_reason": None,
        }

        for label, frac, bank in [("hk", hk, bank_hk), ("fk", fk, bank_fk)]:
            raw_bet = frac * bank
            bet = min(raw_bet, float(max_bet)) if max_bet is not None else raw_bet
            bet = max(bet, 0.0)

            if bet < 1e-6:
                trade_row[f"{label}_bet"]    = 0.0
                trade_row[f"{label}_shares"] = 0.0
                trade_row[f"{label}_fill"]   = 0.0
                trade_row[f"{label}_fee"]    = 0.0
                trade_row[f"{label}_profit"] = 0.0
                trade_row[f"{label}_bank"]   = round(bank, 2)
                continue

            shares, spent, avg_fill_c = _fill_book(levels, bet)

            if is_correct:
                fee = spent * fr
                profit = shares - spent - fee
            else:
                fee = 0.0
                profit = -spent

            trade_row[f"{label}_bet"]    = round(spent, 2)
            trade_row[f"{label}_shares"] = round(shares, 2)
            trade_row[f"{label}_fill"]   = round(avg_fill_c, 2)
            trade_row[f"{label}_fee"]    = round(fee, 2)
            trade_row[f"{label}_profit"] = round(profit, 2)

            if label == "hk":
                bank_hk = max(bank_hk + profit, 0.0)
                trade_row["hk_bank"] = round(bank_hk, 2)
            else:
                bank_fk = max(bank_fk + profit, 0.0)
                trade_row["fk_bank"] = round(bank_fk, 2)

        trades_detail.append(trade_row)

    total_trades = len(trades_detail)
    total_wins   = sum(1 for t in trades_detail if t["correct"])

    def _pct(v, total):
        return round(v / total * 100, 1) if total > 0 else 0

    return {
        "start_bank":      start_bank,
        "max_bet":         max_bet,
        "fee_rate":        fr,
        "max_price_cents": mpc,
        "hk_pct":          hkf,
        "fk_pct":          fkf,
        "total_resolved":  total_resolved,
        "total_trades":    total_trades,
        "total_wins":      total_wins,
        "win_pct":         _pct(total_wins, total_trades),
        "skipped_price":   skipped_price,
        "half_kelly": {
            "end_bank": round(bank_hk, 2),
            "roi_pct":  round((bank_hk - start_bank) / start_bank * 100, 2) if start_bank > 0 else 0,
        },
        "full_kelly": {
            "end_bank": round(bank_fk, 2),
            "roi_pct":  round((bank_fk - start_bank) / start_bank * 100, 2) if start_bank > 0 else 0,
        },
        "trades": trades_detail,
    }


async def get_order_market_pricing(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    price_threshold_cents: float = 52.0,
) -> Dict[str, Any]:
    """Analyse correctly predicted markets: what % had the ask price drop ≤ threshold
    during the market's active window.

    For each market with a correct prediction (prediction == resolved_outcome):
      1. Find the matching outcome asset_id (UP→Up/Yes, DOWN→Down/No).
      2. Look at all orderbook snapshots during the market lifetime (pm.ts → pm.ts + interval).
      3. Find the minimum best_ask_cents recorded.
      4. If min_ask ≤ price_threshold_cents, the market counts as "fillable" (limit order would fill).

    Returns summary stats + per-market detail rows.
    """
    interval = int(getattr(config, "POLY_INTERVAL_SECONDS", 300))
    if interval <= 0:
        interval = 300

    msk_started = f"CONVERT_TZ(pr.started_at, '+00:00', '{MSK_TZ_NAME}')"

    # Base filters: correct predictions only, resolved markets
    where = [
        "pr.prediction IN ('UP','DOWN')",
        "pr.quantum = 0",
        "pr.error IS NULL",
        "pm.resolved_outcome IS NOT NULL",
        "pm.resolved_outcome IN ('UP','DOWN')",
        "pr.started_at < FROM_UNIXTIME(pm.ts)",
    ]
    params: list = []

    if date_from:
        where.append(f"DATE({msk_started}) >= %s")
        params.append(date_from)
    if date_to:
        where.append(f"DATE({msk_started}) <= %s")
        params.append(date_to)

    where_sql = " AND ".join(where)

    # Deduplicate per market (slug): one row per correctly-predicted market
    # Use the first prediction (earliest started_at) per slug
    q = f"""
        SELECT
            sub.slug,
            sub.prediction,
            sub.resolved_outcome,
            sub.market_ts,
            sub.day,
            sub.template_name,
            sub.asset_id,
            (
                SELECT MIN(obs.best_ask_cents)
                FROM poly_orderbook_snapshots obs
                WHERE obs.slug = sub.slug
                  AND obs.asset_id = sub.asset_id
                  AND obs.ts >= sub.market_ts
                  AND obs.ts <= sub.market_ts + {interval}
                  AND obs.best_ask_cents IS NOT NULL
                  AND obs.best_ask_cents > 0
            ) AS min_ask_cents
        FROM (
            SELECT
                pr.slug,
                pr.prediction,
                pm.resolved_outcome,
                pm.ts AS market_ts,
                DATE({msk_started}) AS day,
                pr.template_name,
                po.asset_id,
                ROW_NUMBER() OVER (PARTITION BY pr.slug ORDER BY pr.started_at ASC) AS rn
            FROM poly_pred_runs pr
            JOIN poly_markets pm ON pm.slug = pr.slug
            JOIN poly_outcomes po
                ON  po.slug = pr.slug
                AND (
                    (pr.prediction = 'UP'   AND (UPPER(po.name) LIKE '%%UP%%'   OR UPPER(po.name) LIKE '%%YES%%'))
                 OR (pr.prediction = 'DOWN' AND (UPPER(po.name) LIKE '%%DOWN%%' OR UPPER(po.name) LIKE '%%NO%%'))
                )
            WHERE {where_sql}
        ) sub
        WHERE sub.rn = 1
        ORDER BY sub.market_ts ASC
        LIMIT 20000
    """
    rows = await db.fetchall(q, params or None)

    threshold = float(price_threshold_cents)

    total_predictions = len(rows)
    total_correct = 0
    total_incorrect = 0
    correct_no_data = 0
    incorrect_no_data = 0
    fillable_correct = 0
    fillable_incorrect = 0
    profit_total = 0.0
    profit_trades = 0
    markets: List[Dict[str, Any]] = []

    per_day_raw: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "correct": 0,
            "incorrect": 0,
            "correct_no_data": 0,
            "incorrect_no_data": 0,
            "fillable_correct": 0,
            "fillable_incorrect": 0,
            "min_asks": [],
            "profit": 0.0,
        }
    )

    for row in rows:
        slug, prediction, resolved_outcome, market_ts, day, template_name, asset_id, min_ask = row
        day_str = str(day) if day else ""

        is_correct = bool(prediction == resolved_outcome)
        if is_correct:
            total_correct += 1
        else:
            total_incorrect += 1

        min_ask_f = float(min_ask) if min_ask is not None else None
        is_fillable = min_ask_f is not None and min_ask_f <= threshold
        has_data = min_ask_f is not None

        trade_profit: Optional[float] = None
        if not has_data:
            if is_correct:
                correct_no_data += 1
            else:
                incorrect_no_data += 1
        if is_fillable:
            price_dollars = threshold / 100.0
            if price_dollars > 0:
                profit_trades += 1
                if is_correct:
                    fillable_correct += 1
                    trade_profit = (1.0 / price_dollars) - 1.0
                else:
                    fillable_incorrect += 1
                    trade_profit = -1.0
                profit_total += trade_profit

        profit_value = round(trade_profit, 4) if trade_profit is not None else None

        markets.append({
            "slug": slug,
            "prediction": prediction,
            "resolved_outcome": resolved_outcome,
            "market_ts": int(market_ts),
            "day": day_str,
            "template": template_name or "—",
            "asset_id": asset_id,
            "min_ask_cents": round(min_ask_f, 2) if min_ask_f is not None else None,
            "fillable": is_fillable,
            "correct": is_correct,
            "profit_usd": profit_value,
        })

        d = per_day_raw[day_str]
        if is_correct:
            d["correct"] += 1
            if not has_data:
                d["correct_no_data"] += 1
        else:
            d["incorrect"] += 1
            if not has_data:
                d["incorrect_no_data"] += 1
        if is_fillable:
            if is_correct:
                d["fillable_correct"] += 1
            else:
                d["fillable_incorrect"] += 1
        if not has_data:
            pass
        if min_ask_f is not None:
            d["min_asks"].append(min_ask_f)
        if trade_profit is not None:
            d["profit"] += trade_profit

    with_data_correct = total_correct - correct_no_data
    with_data_incorrect = total_incorrect - incorrect_no_data
    with_data = with_data_correct + with_data_incorrect

    fillable_pct_correct = round(fillable_correct / with_data_correct * 100, 1) if with_data_correct > 0 else None
    fillable_pct_incorrect = (
        round(fillable_incorrect / with_data_incorrect * 100, 1) if with_data_incorrect > 0 else None
    )

    per_day = []
    for day_str in sorted(per_day_raw.keys()):
        d = per_day_raw[day_str]
        wd_correct = d["correct"] - d["correct_no_data"]
        wd_incorrect = d["incorrect"] - d["incorrect_no_data"]
        pct_correct = (
            round(d["fillable_correct"] / wd_correct * 100, 1) if wd_correct > 0 else None
        )
        pct_incorrect = (
            round(d["fillable_incorrect"] / wd_incorrect * 100, 1) if wd_incorrect > 0 else None
        )
        profit_trades_day = d["fillable_correct"] + d["fillable_incorrect"]
        avg_profit_day = (
            round(d["profit"] / profit_trades_day, 3) if profit_trades_day > 0 else None
        )
        avg_min = round(sum(d["min_asks"]) / len(d["min_asks"]), 2) if d["min_asks"] else None
        per_day.append({
            "day": day_str,
            "correct": d["correct"],
            "incorrect": d["incorrect"],
            "with_data_correct": wd_correct,
            "with_data_incorrect": wd_incorrect,
            "fillable_correct": d["fillable_correct"],
            "fillable_incorrect": d["fillable_incorrect"],
            "fillable_pct_correct": pct_correct,
            "fillable_pct_incorrect": pct_incorrect,
            "profit_usd": round(d["profit"], 2) if d["profit"] else 0.0,
            "avg_profit_usd": avg_profit_day,
            "avg_min_ask": avg_min,
        })

    # Distribution histogram: count markets per 0.5c bucket
    histogram: List[Dict[str, Any]] = []
    if markets:
        asks_with_data = [m["min_ask_cents"] for m in markets if m["min_ask_cents"] is not None and m["correct"]]
        if asks_with_data:
            bucket_range = [round(44.0 + i * 0.5, 1) for i in range(37)]  # 44.0 .. 62.0
            for b in bucket_range:
                cnt = sum(1 for a in asks_with_data if a <= b)
                histogram.append({
                    "price": b,
                    "count": cnt,
                    "pct": round(cnt / len(asks_with_data) * 100, 1),
                })

    profit_avg = round(profit_total / profit_trades, 3) if profit_trades > 0 else None
    profit_win_rate = fillable_correct / profit_trades if profit_trades > 0 else None

    summary = {
        "total_predictions": total_predictions,
        "total_correct_predictions": total_correct,
        "total_incorrect_predictions": total_incorrect,
        "with_orderbook_data": with_data,
        "with_data_correct": with_data_correct,
        "with_data_incorrect": with_data_incorrect,
        "no_data": total_predictions - with_data,
        "fillable_count": fillable_correct,
        "fillable_pct": fillable_pct_correct,
        "fillable_incorrect_count": fillable_incorrect,
        "fillable_incorrect_pct": fillable_pct_incorrect,
        "fillable_total_count": fillable_correct + fillable_incorrect,
        "profit_total_usd": round(profit_total, 2),
        "profit_final_usd": round(profit_total, 2),
        "profit_avg_per_trade_usd": profit_avg,
        "profit_trades": profit_trades,
        "profit_win_rate": profit_win_rate,
        "profit_win_trades": fillable_correct,
        "profit_loss_trades": fillable_incorrect,
    }

    return {
        "price_threshold_cents": threshold,
        "interval_sec": interval,
        "summary": summary,
        "per_day": per_day,
        "histogram": histogram,
        "markets": markets,
    }
