import time
import requests
import json
from dataclasses import dataclass
from typing import List
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
            closed= 1 if raw["closed"] == True else 0
        )

    # ---------------------------------------------
    # 3. Получить текущий активный рынок
    # ---------------------------------------------
    def fetch_current_active_market(self) -> MarketData:
        ts = self.get_current_market_timestamp()
        slug = self.get_slug_for_timestamp(ts)
        return self.fetch_market(slug)
