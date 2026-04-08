import os
import json
from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------------------------------------------------------  Setup

IS_LOCAL: bool = os.getenv("IS_LOCAL", False)
DB_USER: str = os.getenv("DB_USER", "")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_NAME: str = os.getenv("DB_NAME", "")

FASTAPI_ROOT: str = os.getenv("FASTAPI_ROOT", "")

API_KEY : str = os.getenv("API_KEY", "no-api-key")
WALLET_ADDRESS : str = os.getenv("WALLET_ADDRESS", "no-wallet-address")

WSS_API_KEY : str = os.getenv("WSS_API_KEY", "no-WSS_API_KEY")
WSS_API_SECRET : str = os.getenv("WSS_API_SECRET", "no-WSS_API_SECRET")
WSS_API_PASSPHRASE : str = os.getenv("WSS_API_PASSPHRASE", "no-WSS_API_PASSPHRASE")

TARGET_PRICES = [52, 51, 50, 49, 48]  # in cents

POLY_INTERVAL_SECONDS: int = int(os.getenv("POLY_INTERVAL_SECONDS", "300"))
POLY_SLUG_TEMPLATE: str = os.getenv("POLY_SLUG_TEMPLATE", "btc-updown-5m-{ts}")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


EMULATE_DOWN: bool = _env_bool("EMULATE_DOWN", False)

NEED_CONFIRMATION: bool = _env_bool("NEED_CONFIRMATION", True)

BUY_MARKET: bool = _env_bool("BUY_MARKET", True)

LOG_PRED_DATA_FILES: bool = _env_bool("LOG_PRED_DATA_FILES", False)

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_ADMIN_CHAT_ID: str = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
TELEGRAM_PRINT_CHAT_ID: bool = _env_bool("TELEGRAM_PRINT_CHAT_ID", False)
TELEGRAM_DAILY_REPORTS_ENABLED: bool = _env_bool("TELEGRAM_DAILY_REPORTS_ENABLED", False)

DEBUG_BINANCE_PRICE: bool = _env_bool("DEBUG_BINANCE_PRICE", False)

def _env_json_list(name: str) -> list[str]:
    raw = os.getenv(name, "[]")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    # Support comma-separated fallback
    return [s.strip() for s in raw.split(',') if s.strip()]

TELEGRAM_INFO_CHAT_IDS: list[str] = _env_json_list("TELEGRAM_INFO_CHAT_IDS")

TELEGRAM_ORDER_FLOW_INFO: list[str] = _env_json_list("TELEGRAM_ORDER_FLOW_INFO")
