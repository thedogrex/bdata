# Polymarket CLOB V2 Migration Guide

This project now targets Polymarket's CLOB V2 API/contract stack. Follow the checklist below whenever you deploy, run smoke tests, or onboard a new machine.

## 1. Dependencies

1. `pip install -r requirements.txt`
   - Ensures `py_clob_client_v2==1.0.0`, `py_builder_relayer_client`, `eth-account>=0.13`, and the upgraded signing/fee helpers are available.
2. Remove any old `py_clob_client` wheels from your virtualenv to avoid namespace collisions.

## 2. Environment variables (`python/.env`)

Required:
- `POLY_PRIVATE_KEY` – signer private key (0x-prefixed hex).
- `POLY_FUNDER` – wallet that actually holds pUSD (proxy wallet / smart account). Keep it empty if signer and funder are the same.
- `POLY_SIGNATURE_TYPE=1` – Magic/email custodial wallets must sign with `SignatureType=1`.
- `POLY_CLOB_HOST=https://clob.polymarket.com`
- `POLY_CHAIN_ID=137` (Polygon mainnet).

Optional (builder attribution):
- `POLY_BUILDER_ADDRESS` – builder wallet (used for revenue share payouts).
- `POLY_BUILDER_CODE` – 32-byte hex builder code (exactly 66 chars including `0x`).

> Invalid builder codes are ignored and logged. Keep the code private—it maps fills back to your builder account.

Redeem-all (builder relayer) credentials:
- `POLY_BUILDER_API_KEY`, `POLY_BUILDER_API_SECRET`, `POLY_BUILDER_API_PASSPHRASE` – issued via the Builder portal.
- `POLY_RELAYER_HOST` – override only if directed by Polymarket (defaults to `https://relayer-v2.polymarket.com`).
- `POLY_REDEEM_COOLDOWN_SEC` – optional throttle between manual redemptions (defaults to 420s = 7 minutes).

Auto-redeem (background task):
- `POLY_AUTO_REDEEM_ENABLED=true` – starts a background loop on startup that automatically redeems positions.
- `POLY_AUTO_REDEEM_INTERVAL_SEC` – how often to run redeem-all (defaults to 420s = 7 minutes).

## 3. Wallet & collateral prep

1. Fund the signer/funder with **USDC.e** on Polygon.
2. Wrap USDC.e → **pUSD** via the [Collateral Onramp `wrap()`](https://docs.polymarket.com/concepts/pusd) contract (only needed once per wallet).
3. Approve the exchange contract if prompted by the SDK.

## 4. Smoke test

1. Export `.env` (or run via `python-dotenv`).
2. `python main_poly.py`
   - Confirms we can fetch Gamma markets and construct a `ClobClient` with derived API creds.
3. Optional dry run: call `predictor.poly_client.PolymarketClient().buy_limit(...)` against a test market (builder program provides sandbox markets at `https://clob-v2.polymarket.com`).

## 5. Operational checklist

- ✅ `pip list | grep py-clob` shows only `py-clob-client-v2`.
- ✅ `POLY_SIGNATURE_TYPE=1` in production `.env` (per user request).
- ✅ Builder fields omitted unless you have an issued code.
- ✅ All orders include millisecond timestamps (handled automatically by `py_clob_client_v2`).
- ✅ Fees are auto-calculated server side; no hard-coded `feeRateBps` in strategy configs.

## 6. Redeem-all helper

- API: `POST /api/poly/live/redeem_all?force=false`
  - Triggers the builder relayer to redeem every resolved position that still holds shares.
  - Enforces a 7-minute cooldown (set `force=true` if you need to override).
  - Response includes per-market status and total redeemed count.
- Make sure the Builder relayer credentials are in `.env` before wiring the "Redeem All" button to this endpoint.

- Auto-redeem: Set `POLY_AUTO_REDEEM_ENABLED=true` to start a background task that runs redeem-all automatically every `POLY_AUTO_REDEEM_INTERVAL_SEC` seconds (default 7 minutes). The task starts on FastAPI startup and logs its activity to the console.

## 7. References

- Official migration notes: <https://docs.polymarket.com/v2-migration>
- Contracts list: <https://docs.polymarket.com/resources/contracts>
- pUSD details: <https://docs.polymarket.com/concepts/pusd>

Keep this guide up-to-date whenever Polymarket releases breaking changes (new host, collateral token, or SDK revision).
