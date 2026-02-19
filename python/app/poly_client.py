import time
import requests
import json
from dataclasses import dataclass
from typing import List, Optional
import app.config as config


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


class PolymarketClient:
    BASE_URL = "https://gamma-api.polymarket.com/markets?slug="

    # ---------------------------------------------
    # 1. Вычисляем timestamp текущего рынка
    # ---------------------------------------------
    def get_current_market_timestamp(self) -> int:
        now = int(time.time())  # UTC timestamp
        interval = int(getattr(config, "POLY_INTERVAL_SECONDS", 300))
        if interval <= 0:
            interval = 300
        return (now // interval) * interval

    def get_slug_for_timestamp(self, ts: int) -> str:
        template = getattr(config, "POLY_SLUG_TEMPLATE", "btc-updown-5m-{ts}")
        try:
            return template.format(ts=ts)
        except Exception:
            return f"btc-updown-5m-{ts}"

    # ---------------------------------------------
    # 2. Загрузка одного рынка по slug
    # ---------------------------------------------
    def fetch_market(self, slug: str) -> MarketData:
        url = self.BASE_URL + slug
        resp = requests.get(url)

        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}")

        data = resp.json()

        if not data:
            raise RuntimeError(f"Market not found: {slug}")

        raw = data[0]

        def _parse_float(v):
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        # Try to read market resolution inputs (if provided by Gamma)
        final_price = _parse_float(raw.get("finalPrice") if isinstance(raw, dict) else None)
        if final_price is None:
            final_price = _parse_float(raw.get("final_price") if isinstance(raw, dict) else None)

        target_price = _parse_float(raw.get("targetPrice") if isinstance(raw, dict) else None)
        if target_price is None:
            target_price = _parse_float(raw.get("target_price") if isinstance(raw, dict) else None)

        # parse outcomes
        outcome_names = json.loads(raw["outcomes"])
        outcome_prices = json.loads(raw["outcomePrices"])
        asset_ids = json.loads(raw["clobTokenIds"])

        outcomes = [
            MarketOutcome(
                name=outcome_names[i],
                price=float(outcome_prices[i]),
                asset_id=asset_ids[i]
            )
            for i in range(len(outcome_names))
        ]

        timestamp = int(raw["slug"].split("-")[-1])

        return MarketData(
            slug=raw["slug"],
            timestamp=timestamp,
            end_date=raw["endDate"],
            question=raw["question"],
            description=raw["description"],
            outcomes=outcomes,
            closed= 1 if raw["closed"] == True else 0,
            final_price=final_price,
            target_price=target_price,
        )

    # ---------------------------------------------
    # 3. Получить текущий активный рынок
    # ---------------------------------------------
    def fetch_current_active_market(self) -> MarketData:
        ts = self.get_current_market_timestamp()
        slug = self.get_slug_for_timestamp(ts)
        return self.fetch_market(slug)
