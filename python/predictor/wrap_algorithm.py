#!/usr/bin/env python3
"""
USDC.e → pUSD Wrapping Algorithm using Polymarket Relayer.

This script handles the complete wrapping flow:
1. Detects wallet type (EOA vs Proxy)
2. Checks balances in funder and proxy wallets
3. If using proxy wallet and USDC.e is in funder, guides user on funding
4. If USDC.e is in correct wallet, executes approve + deposit via relayer
5. All operations are gas-less (relayer pays MATIC)

Usage:
    python wrap_algorithm.py --info           # Check balances
    python wrap_algorithm.py --amount 23.5    # Wrap specific amount
    python wrap_algorithm.py --all            # Wrap all available
"""

import os
import sys
import argparse
import time
import requests
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from eth_abi import encode as eth_encode
from eth_utils import keccak

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, RelayerTxType, SafeTransaction
from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig as BuilderSigningConfig

load_dotenv()

# Contract addresses
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
PUSD = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
ONRAMP = "0x93070a847efEf7F70739046A929D47a521F5B8ee"  # Correct contract from docs

# Function selectors
SEL_APPROVE = keccak(text="approve(address,uint256)")[:4]
SEL_BALANCE = keccak(text="balanceOf(address)")[:4]
SEL_WRAP = keccak(text="wrap(address,address,uint256)")[:4]

POLYGON_RPC_ENDPOINTS = [
    "https://polygon.llamarpc.com",
    "https://polygon.drpc.org",
    "https://polygon.meowrpc.com",
    "https://rpc.ankr.com/polygon",
]
DEFAULT_RELAYER_URL = "https://relayer-v2.polymarket.com"


@dataclass
class WrapConfig:
    private_key: str
    funder_address: str
    signature_type: int
    builder_api_key: str
    builder_api_secret: str
    builder_api_passphrase: str
    relayer_url: str
    chain_id: int


def get_env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def load_config() -> WrapConfig:
    private_key = get_env_value("POLY_PRIVATE_KEY", "POLYMARKET_PRIVATE_KEY")
    funder_address = get_env_value("POLY_FUNDER", "POLYMARKET_FUNDER_ADDRESS")
    sig_raw = get_env_value("POLY_SIGNATURE_TYPE") or "0"
    builder_key = get_env_value("POLY_BUILDER_API_KEY", "POLYMARKET_BUILDER_API_KEY")
    builder_secret = get_env_value("POLY_BUILDER_API_SECRET", "POLYMARKET_BUILDER_SECRET")
    builder_passphrase = get_env_value("POLY_BUILDER_API_PASSPHRASE", "POLYMARKET_BUILDER_PASSPHRASE")
    relayer_url = get_env_value("POLY_RELAYER_HOST") or DEFAULT_RELAYER_URL
    chain_raw = get_env_value("POLY_CHAIN_ID") or "137"

    missing = []
    if not private_key:
        missing.append("POLY_PRIVATE_KEY")
    if not funder_address:
        missing.append("POLY_FUNDER")
    if not builder_key:
        missing.append("POLY_BUILDER_API_KEY")
    if not builder_secret:
        missing.append("POLY_BUILDER_API_SECRET")
    if not builder_passphrase:
        missing.append("POLY_BUILDER_API_PASSPHRASE")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    return WrapConfig(
        private_key=private_key,
        funder_address=funder_address,
        signature_type=int(sig_raw),
        builder_api_key=builder_key,
        builder_api_secret=builder_secret,
        builder_api_passphrase=builder_passphrase,
        relayer_url=relayer_url,
        chain_id=int(chain_raw),
    )


def call_rpc(payload: dict) -> Optional[dict]:
    for rpc_url in POLYGON_RPC_ENDPOINTS:
        try:
            resp = requests.post(rpc_url, json=payload, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and "result" not in data:
                continue
            return data
        except Exception:
            continue
    return None


def get_balance(token: str, wallet: str) -> int:
    data = "0x" + SEL_BALANCE.hex() + wallet[2:].lower().zfill(64)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": token, "data": data}, "latest"]
    }
    resp = call_rpc(payload)
    if not resp or "result" not in resp:
        return 0
    return int(resp["result"], 16)


def build_relayer_client(config: WrapConfig) -> tuple[RelayClient, str]:
    wallet_type = RelayerTxType.PROXY if config.signature_type == 1 else RelayerTxType.SAFE
    builder_config = BuilderSigningConfig(
        local_builder_creds=BuilderApiKeyCreds(
            key=config.builder_api_key,
            secret=config.builder_api_secret,
            passphrase=config.builder_api_passphrase,
        )
    )
    client = RelayClient(
        config.relayer_url,
        chain_id=config.chain_id,
        private_key=config.private_key,
        builder_config=builder_config,
        relay_tx_type=wallet_type,
    )
    
    # Get proxy address if using proxy wallet
    proxy = config.funder_address
    if config.signature_type == 1:
        for attr in ["address", "proxy_address", "wallet_address"]:
            if hasattr(client, attr):
                val = getattr(client, attr)
                if val:
                    proxy = val
                    break
    
    return client, proxy


def execute_relayer_transaction(client: RelayClient, txn: SafeTransaction, label: str) -> dict:
    for attempt in range(2):
        try:
            print(f"  → Executing: {label}")
            resp = client.execute([txn], label)
            resp.wait()
            
            # Check for failure - the relayer response may indicate failure
            failed = False
            error_msg = None
            tx_hash = None
            
            # Extract hash first
            for attr in ["transaction_hash", "hash", "tx_hash"]:
                if hasattr(resp, attr):
                    val = getattr(resp, attr)
                    if val:
                        tx_hash = val
                        break
            
            # Check response status attributes
            for attr in ["_status", "status", "state"]:
                if hasattr(resp, attr):
                    status_val = getattr(resp, attr)
                    if status_val:
                        status_str = str(status_val).upper()
                        if any(x in status_str for x in ["FAILED", "REVERT", "ERROR"]):
                            failed = True
                            error_msg = f"Status: {status_str}"
                            print(f"  ❌ {error_msg}")
            
            # Check receipt status
            if hasattr(resp, "receipt") and resp.receipt:
                rcpt = resp.receipt
                if isinstance(rcpt, dict):
                    status = rcpt.get("status")
                    if status in [0, "0x0", "0x00", False]:
                        failed = True
                        error_msg = "Transaction reverted on-chain"
                        print(f"  ❌ {error_msg}")
            
            # Check for explicit error in response
            if hasattr(resp, "error"):
                err = getattr(resp, "error")
                if err:
                    failed = True
                    error_msg = str(err)
                    print(f"  ❌ Error: {error_msg}")
            
            # Check response string for "failed onchain" pattern
            resp_str = str(resp).lower()
            if "failed onchain" in resp_str:
                failed = True
                error_msg = "Transaction failed on-chain"
                print(f"  ❌ {error_msg}")
            
            if failed:
                return {"ok": False, "error": error_msg or "Transaction failed", "hash": tx_hash}
            
            print(f"  ✅ Success: {tx_hash[:20]}..." if tx_hash else "  ✅ Success")
            return {"ok": True, "hash": tx_hash}
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            if attempt == 0:
                time.sleep(60)
                continue
            return {"ok": False, "error": str(e)}
    
    return {"ok": False, "error": "Rate limit exceeded"}


def build_approve_tx(token: str, spender: str, amount: int) -> SafeTransaction:
    args = eth_encode(["address", "uint256"], [spender, amount])
    data = "0x" + (SEL_APPROVE + args).hex()
    return SafeTransaction(to=token, operation=OperationType.Call, data=data, value="0")


def build_deposit_tx(amount: int, wallet: str) -> SafeTransaction:
    args = eth_encode(["address", "address", "uint256"], [USDC_E, wallet, amount])
    data = "0x" + (SEL_WRAP + args).hex()
    return SafeTransaction(to=ONRAMP, operation=OperationType.Call, data=data, value="0")


def main():
    parser = argparse.ArgumentParser(description="Wrap USDC.e to pUSD via relayer")
    parser.add_argument("--info", action="store_true", help="Show balances")
    parser.add_argument("--amount", type=float, help="Amount in USDC.e")
    parser.add_argument("--all", action="store_true", help="Wrap all available")
    args = parser.parse_args()
    
    print("=" * 70)
    print("WRAPPING ALGORITHM: USDC.e → pUSD")
    print("=" * 70)
    
    config = load_config()
    client, proxy = build_relayer_client(config)
    
    wallet_type = "Proxy" if config.signature_type == 1 else "EOA"
    print(f"\nConfiguration:")
    print(f"  Funder:    {config.funder_address}")
    print(f"  Proxy:     {proxy if proxy != config.funder_address else 'Same as funder'}")
    print(f"  Wallet:    {wallet_type} (sig_type={config.signature_type})")
    print(f"  Relayer:   {config.relayer_url}")
    print("-" * 70)
    
    # Check balances
    print("\n[1] Checking balances...")
    funder_usdc = get_balance(USDC_E, config.funder_address)
    proxy_usdc = get_balance(USDC_E, proxy)
    proxy_pusd = get_balance(PUSD, proxy)
    
    print(f"  Funder USDC.e: {funder_usdc / 1e6:.6f}")
    print(f"  Proxy  USDC.e: {proxy_usdc / 1e6:.6f}")
    print(f"  Proxy  pUSD:   {proxy_pusd / 1e6:.6f}")
    
    if args.info:
        print("\n" + "-" * 70)
        if config.signature_type == 1:
            print("PROXY WALLET MODE:")
            print("  - Relayer executes from PROXY wallet")
            print("  - USDC.e must be in PROXY to wrap gas-less")
            print(f"  - Your proxy: {proxy}")
            if proxy_usdc > 0:
                print(f"  ✅ Ready to wrap {proxy_usdc/1e6:.6f} USDC.e")
            elif funder_usdc > 0:
                print(f"  ⚠️  USDC.e in funder ({funder_usdc/1e6:.6f})")
                print(f"  → Send USDC.e to proxy: {proxy}")
                print("  → Or use direct Web3 (requires MATIC)")
        else:
            print("EOA WALLET MODE:")
            print("  - Relayer executes from FUNDER wallet")
            print("  - USDC.e must be in FUNDER to wrap")
            if funder_usdc > 0:
                print(f"  ✅ Ready to wrap {funder_usdc/1e6:.6f} USDC.e")
        return
    
    # Determine target wallet and balance
    target_wallet = proxy if config.signature_type == 1 else config.funder_address
    target_balance = proxy_usdc if config.signature_type == 1 else funder_usdc
    
    # Determine amount
    if args.all:
        amount_raw = max(0, target_balance - 10000)
    elif args.amount:
        amount_raw = int(args.amount * 1_000_000)
    else:
        print("❌ Specify --amount or --all")
        sys.exit(1)
    
    amount_display = amount_raw / 1e6
    
    if amount_raw <= 0:
        print("❌ Invalid amount")
        sys.exit(1)
    
    if amount_raw > target_balance:
        print(f"❌ Insufficient USDC.e")
        print(f"   Have: {target_balance/1e6:.6f}")
        print(f"   Need: {amount_display:.6f}")
        print(f"\nTarget wallet: {target_wallet}")
        if config.signature_type == 1:
            print("Send USDC.e to proxy address above")
        sys.exit(1)
    
    print(f"\n[2] Preparing to wrap {amount_display:.6f} USDC.e → pUSD")
    print(f"   From: {target_wallet}")
    print("-" * 70)
    
    confirm = input("Confirm? [y/N]: ")
    if confirm.lower() != "y":
        print("Aborted")
        sys.exit(0)
    
    # Step 1: Approve
    print("\n[3] Approving ONRAMP to spend USDC.e...")
    approve_tx = build_approve_tx(USDC_E, ONRAMP, amount_raw)
    result1 = execute_relayer_transaction(client, approve_tx, "approve_onramp")
    
    if not result1["ok"]:
        print(f"\n❌ Approve failed: {result1.get('error')}")
        sys.exit(1)
    
    time.sleep(2)
    
    # Step 2: Wrap
    print("\n[4] Wrapping USDC.e → pUSD...")
    deposit_tx = build_deposit_tx(amount_raw, target_wallet)
    result2 = execute_relayer_transaction(client, deposit_tx, "wrap")
    
    if not result2["ok"]:
        print(f"\n❌ Wrap failed: {result2.get('error')}")
        print(f"\nNote: Approve succeeded. Retry wrap if needed.")
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 70)
    print("✅ WRAP COMPLETE")
    print("=" * 70)
    print(f"Wrapped: {amount_display:.6f} USDC.e → pUSD")
    print(f"Wallet:  {target_wallet}")
    if result1.get("hash"):
        print(f"Approve: https://polygonscan.com/tx/{result1['hash']}")
    if result2.get("hash"):
        print(f"Deposit: https://polygonscan.com/tx/{result2['hash']}")
    print("-" * 70)
    
    final_pusd = get_balance(PUSD, proxy if config.signature_type == 1 else config.funder_address)
    print(f"pUSD balance: {final_pusd / 1e6:.6f}")


if __name__ == "__main__":
    main()
