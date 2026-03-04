"""
Polymarket Trading Client for predictor module.

Wraps py_clob_client SDK to:
 - Fetch market data from Gamma API
 - Create & sign orders via CLOB API (limit / market / FOK)
 - Track positions in DB
 - Verbose console logging for debugging

Env vars required in .env:
  POLY_PRIVATE_KEY     - Wallet private key for signing
  POLY_FUNDER          - Address that holds funds (proxy wallet address)
  POLY_SIGNATURE_TYPE  - 0=EOA, 1=email/Magic, 2=browser proxy (default 0)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("poly_client")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Data classes (kept from app/poly_client.py)
# ---------------------------------------------------------------------------

@dataclass
class MarketOutcome:
    name: str
    price: float
    asset_id: str


@dataclass
class MarketData:
    slug: str
    timestamp: int
    end_date: str
    question: str
    description: str
    outcomes: List[MarketOutcome]
    closed: int
    final_price: Optional[float] = None
    target_price: Optional[float] = None


# ---------------------------------------------------------------------------
# CLOB client singleton (lazy init)
# ---------------------------------------------------------------------------

_clob_client = None


def _get_clob_client():
    """Lazy-init ClobClient with credentials from .env."""
    global _clob_client
    if _clob_client is not None:
        return _clob_client

    from py_clob_client.client import ClobClient

    host = os.getenv("POLY_CLOB_HOST", "https://clob.polymarket.com")
    chain_id = int(os.getenv("POLY_CHAIN_ID", "137"))
    private_key = os.getenv("POLY_PRIVATE_KEY", "")
    funder = os.getenv("POLY_FUNDER", "")
    sig_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))

    if not private_key:
        logger.error("POLY_PRIVATE_KEY is not set in .env — trading disabled")
        return None

    logger.info("Initialising ClobClient  host=%s  chain=%d  sig_type=%d  funder=%s",
                host, chain_id, sig_type, funder[:10] + "..." if funder else "(none)")

    client = ClobClient(
        host,
        key=private_key,
        chain_id=chain_id,
        signature_type=sig_type,
        funder=funder or None,
    )

    # Derive / load API credentials (HMAC key+secret+passphrase)
    try:
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        logger.info("API creds derived OK  api_key=%s...", creds.api_key[:12] if creds.api_key else "?")
    except Exception as e:
        logger.error("Failed to derive API creds: %s", e, exc_info=True)
        return None

    _clob_client = client
    return _clob_client


# ---------------------------------------------------------------------------
# Market data (Gamma API — read-only, no auth needed)
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com/markets?slug="


class PolymarketClient:
    """Combined market-data + trading client."""

    # ---- config helpers ----
    @staticmethod
    def _interval() -> int:
        return int(os.getenv("POLY_INTERVAL_SECONDS", "300")) or 300

    @staticmethod
    def _slug_template() -> str:
        return os.getenv("POLY_SLUG_TEMPLATE", "btc-updown-5m-{ts}")

    # ---- timestamp / slug ----
    def get_current_market_timestamp(self) -> int:
        now = int(time.time())
        interval = self._interval()
        return (now // interval) * interval

    def get_slug_for_timestamp(self, ts: int) -> str:
        try:
            return self._slug_template().format(ts=ts)
        except Exception:
            return f"btc-updown-5m-{ts}"

    # ---- fetch market from Gamma ----
    def fetch_market(self, slug: str) -> MarketData:
        url = GAMMA_BASE + slug
        logger.debug("GET %s", url)
        resp = requests.get(url, timeout=15)
        logger.debug("Gamma response status=%d  len=%d", resp.status_code, len(resp.content))

        if resp.status_code != 200:
            raise RuntimeError(f"Gamma API error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        if not data:
            raise RuntimeError(f"Market not found: {slug}")

        raw = data[0]

        def _pf(v):
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        final_price = _pf(raw.get("finalPrice")) or _pf(raw.get("final_price"))
        target_price = _pf(raw.get("targetPrice")) or _pf(raw.get("target_price"))

        outcome_names = json.loads(raw["outcomes"])
        outcome_prices = json.loads(raw["outcomePrices"])
        asset_ids = json.loads(raw["clobTokenIds"])

        outcomes = [
            MarketOutcome(name=outcome_names[i], price=float(outcome_prices[i]), asset_id=asset_ids[i])
            for i in range(len(outcome_names))
        ]

        try:
            timestamp = int(raw["slug"].split("-")[-1])
        except Exception:
            timestamp = 0

        logger.info("Market loaded  slug=%s  closed=%s  outcomes=%d  prices=%s",
                     slug, raw.get("closed"), len(outcomes),
                     [f"{o.name}={o.price}" for o in outcomes])

        return MarketData(
            slug=raw["slug"],
            timestamp=timestamp,
            end_date=raw["endDate"],
            question=raw["question"],
            description=raw["description"],
            outcomes=outcomes,
            closed=1 if raw.get("closed") is True else 0,
            final_price=final_price,
            target_price=target_price,
        )

    def fetch_current_active_market(self) -> MarketData:
        ts = self.get_current_market_timestamp()
        slug = self.get_slug_for_timestamp(ts)
        return self.fetch_market(slug)

    # ------------------------------------------------------------------
    # CLOB Trading
    # ------------------------------------------------------------------

    def _clob(self):
        c = _get_clob_client()
        if c is None:
            raise RuntimeError("CLOB client not initialised (check POLY_PRIVATE_KEY in .env)")
        return c

    # ---- helpers ----
    def get_tick_size(self, token_id: str) -> str:
        """Get tick size for a token (e.g. '0.01')."""
        try:
            ts = self._clob().get_tick_size(token_id)
            logger.debug("tick_size(%s) = %s", token_id[:16], ts)
            return ts
        except Exception as e:
            logger.warning("get_tick_size failed, defaulting to 0.01: %s", e)
            return "0.01"

    def get_neg_risk(self, token_id: str) -> bool:
        """Check if token uses negative risk."""
        try:
            nr = self._clob().get_neg_risk(token_id)
            logger.debug("neg_risk(%s) = %s", token_id[:16], nr)
            return nr
        except Exception as e:
            logger.warning("get_neg_risk failed, defaulting to False: %s", e)
            return False

    # ---- market buy (FOK) ----
    def buy_market(self, token_id: str, amount: float, worst_price: float = 0.99) -> Dict[str, Any]:
        """
        Place a FOK market BUY order.

        Args:
            token_id:    CLOB token ID (asset_id from MarketOutcome)
            amount:      Dollar amount to spend (e.g. 5.0 = $5)
            worst_price: Max price per share (slippage protection, 0..1)

        Returns:
            CLOB API response dict  {success, orderID, status, errorMsg, ...}
        """
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        tick_size = self.get_tick_size(token_id)
        neg_risk = self.get_neg_risk(token_id)

        logger.info(">>> BUY MARKET  token=%s  amount=$%.2f  worst_price=%.4f  tick=%s  neg_risk=%s",
                     token_id[:16], amount, worst_price, tick_size, neg_risk)

        try:
            signed = self._clob().create_market_order(
                MarketOrderArgs(
                    token_id=token_id,
                    amount=amount,
                    price=worst_price,
                    side=BUY,
                    fee_rate_bps=0,
                    nonce=0,
                )
            )
            logger.debug("Signed order created, posting to CLOB...")
            resp = self._clob().post_order(signed, OrderType.FOK)
            logger.info("<<< BUY MARKET response: %s", json.dumps(resp, default=str))
            return resp
        except Exception as e:
            logger.error("BUY MARKET failed: %s", e, exc_info=True)
            return {"success": False, "errorMsg": str(e)}

    # ---- limit buy (GTC) ----
    def buy_limit(self, token_id: str, price: float, size: float) -> Dict[str, Any]:
        """
        Place a GTC limit BUY order.

        Args:
            token_id: CLOB token ID
            price:    Price per share (e.g. 0.52)
            size:     Number of shares

        Returns:
            CLOB API response dict
        """
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        tick_size = self.get_tick_size(token_id)
        neg_risk = self.get_neg_risk(token_id)

        logger.info(">>> BUY LIMIT  token=%s  price=%.4f  size=%.2f  tick=%s  neg_risk=%s",
                     token_id[:16], price, size, tick_size, neg_risk)

        try:
            signed = self._clob().create_order(
                OrderArgs(token_id=token_id, price=price, size=size, side=BUY)
            )
            resp = self._clob().post_order(signed, OrderType.GTC)
            logger.info("<<< BUY LIMIT response: %s", json.dumps(resp, default=str))
            return resp
        except Exception as e:
            logger.error("BUY LIMIT failed: %s", e, exc_info=True)
            return {"success": False, "errorMsg": str(e)}

    # ---- sell (FOK) ----
    def sell_market(self, token_id: str, shares: float, worst_price: float = 0.01) -> Dict[str, Any]:
        """
        Place a FOK market SELL order.

        Args:
            token_id:    CLOB token ID
            shares:      Number of shares to sell
            worst_price: Min price per share (slippage protection)
        """
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL

        tick_size = self.get_tick_size(token_id)
        neg_risk = self.get_neg_risk(token_id)

        logger.info(">>> SELL MARKET  token=%s  shares=%.2f  worst_price=%.4f  tick=%s",
                     token_id[:16], shares, worst_price, tick_size)

        try:
            signed = self._clob().create_market_order(
                MarketOrderArgs(
                    token_id=token_id,
                    amount=shares,
                    price=worst_price,
                    side=SELL,
                    fee_rate_bps=0,
                    nonce=0,
                )
            )
            resp = self._clob().post_order(signed, OrderType.FOK)
            logger.info("<<< SELL MARKET response: %s", json.dumps(resp, default=str))
            return resp
        except Exception as e:
            logger.error("SELL MARKET failed: %s", e, exc_info=True)
            return {"success": False, "errorMsg": str(e)}

    # ---- open orders ----
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders for the authenticated user."""
        try:
            from py_clob_client.clob_types import OpenOrderParams
            orders = self._clob().get_orders(OpenOrderParams())
            logger.info("Open orders: %d", len(orders) if orders else 0)
            return orders or []
        except Exception as e:
            logger.error("get_open_orders failed: %s", e)
            return []

    # ---- cancel ----
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        try:
            resp = self._clob().cancel(order_id)
            logger.info("Cancel order %s: %s", order_id, resp)
            return resp
        except Exception as e:
            logger.error("cancel_order failed: %s", e)
            return {"success": False, "errorMsg": str(e)}

    def cancel_all_orders(self) -> Dict[str, Any]:
        try:
            resp = self._clob().cancel_all()
            logger.info("Cancel all orders: %s", resp)
            return resp
        except Exception as e:
            logger.error("cancel_all failed: %s", e)
            return {"success": False, "errorMsg": str(e)}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Fetch a single order by id (L2)."""
        try:
            client = self._clob()
            if hasattr(client, "get_order"):
                resp = client.get_order(order_id)
            elif hasattr(client, "getOrder"):
                resp = client.getOrder(order_id)
            else:
                raise RuntimeError("CLOB client missing get_order")
            logger.debug("get_order(%s): %s", order_id, resp)
            return resp if isinstance(resp, dict) else {"data": resp}
        except Exception as e:
            logger.error("get_order failed: %s", e)
            return {"success": False, "errorMsg": str(e)}

    # ---- trades ----
    def get_trades(self) -> List[Dict[str, Any]]:
        try:
            trades = self._clob().get_trades()
            logger.info("Trades fetched: %d", len(trades) if trades else 0)
            return trades or []
        except Exception as e:
            logger.error("get_trades failed: %s", e)
            return []

    # ---- orderbook ----
    def get_orderbook(self, token_id: str) -> Dict[str, Any]:
        try:
            book = self._clob().get_order_book(token_id)
            logger.debug("Orderbook for %s: bids=%d asks=%d",
                         token_id[:16],
                         len(book.bids) if book and book.bids else 0,
                         len(book.asks) if book and book.asks else 0)
            return book
        except Exception as e:
            logger.error("get_orderbook failed: %s", e)
            return {}

    def get_midpoint(self, token_id: str) -> Optional[float]:
        try:
            mid = self._clob().get_midpoint(token_id)
            logger.debug("Midpoint %s = %s", token_id[:16], mid)
            return float(mid) if mid else None
        except Exception as e:
            logger.error("get_midpoint failed: %s", e)
            return None

    def get_best_ask(self, token_id: str) -> Optional[float]:
        """Get best ask price for a token."""
        try:
            price = self._clob().get_price(token_id, side="BUY")
            logger.debug("Best ask (BUY price) %s = %s", token_id[:16], price)
            return float(price) if price else None
        except Exception as e:
            logger.error("get_best_ask failed: %s", e)
            return None

    # ---- balances / allowances ----
    def get_balance_allowance(self) -> Dict[str, Any]:
        """Return balance & allowance info for collateral + conditional assets."""
        client = self._clob()
        out: Dict[str, Any] = {"collateral": None, "conditional": None}
        for asset_type in ("COLLATERAL", "CONDITIONAL"):
            try:
                # NOTE: Conditional token balances/allowances are ERC1155 operations and require
                # a concrete tokenId/assetId. A generic "CONDITIONAL" request can fail with:
                #   "assetId invalid value -1"
                # For the wallet summary page we primarily need USDC collateral, so we skip
                # conditional balances here and return a helpful note instead of erroring.
                if asset_type == "CONDITIONAL":
                    out["conditional"] = {
                        "note": "Conditional token balance/allowance requires a specific tokenId (asset_id). Skipped in generic wallet summary."
                    }
                    continue

                # Prefer typed params object. Passing a dict can break on some SDK versions
                # with errors like: "'dict' object has no attribute 'signature_type'".
                from py_clob_client.clob_types import BalanceAllowanceParams

                if hasattr(client, "get_balance_allowance"):
                    resp = client.get_balance_allowance(BalanceAllowanceParams(asset_type=asset_type))
                elif hasattr(client, "getBalanceAllowance"):
                    resp = client.getBalanceAllowance(BalanceAllowanceParams(asset_type=asset_type))
                else:
                    raise RuntimeError("CLOB client missing get_balance_allowance")

                key = "collateral" if asset_type == "COLLATERAL" else "conditional"
                out[key] = resp
                logger.debug("balance_allowance(%s)=%s", asset_type, resp)
            except Exception as e:
                logger.error("get_balance_allowance(%s) failed: %s", asset_type, e)
                key = "collateral" if asset_type == "COLLATERAL" else "conditional"
                out[key] = {"error": str(e)}

        return out
