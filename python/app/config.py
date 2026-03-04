import os
from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------------------------------------------------------  Setup

IS_LOCAL: bool = os.getenv("IS_LOCAL", False)
DB_USER: str = os.getenv("DB_USER", "")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_NAME: str = os.getenv("DB_NAME", "")


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
