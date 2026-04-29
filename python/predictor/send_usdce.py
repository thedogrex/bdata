#!/usr/bin/env python3
"""
Send USDC.e to another address using Polymarket Relayer.

This script sends USDC.e from your wallet to another address gas-less via the relayer.

Usage:
    python send_usdce.py <recipient_address> <amount>
    
Example:
    python send_usdce.py 0x1234...abcd 10.5
"""

import os
import sys
import argparse
import time
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from eth_abi import encode as eth_encode
from eth_utils import keccak
import requests

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, RelayerTxType, SafeTransaction
from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig as BuilderSigningConfig

load_dotenv()

# Contract addresses
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Function selectors
SEL_TRANSFER = keccak(text="transfer(address,uint256)")[:4]
SEL_BALANCE = keccak(text="balanceOf(address)")[:4]

POLYGON_RPC_ENDPOINTS = [
    "https://polygon.llamarpc.com",
    "https://polygon.drpc.org",
    "https://polygon.meowrpc.com",
    "https://rpc.ankr.com/polygon",
]
DEFAULT_RELAYER_URL = "https://relayer-v2.polymarket.com"


@dataclass
class SendSettings:
    private_key: str
    funder_address: str
    signature_type: int
    builder_api_key: str
    builder_api_secret: str
    builder_api_passphrase: str
    relayer_url: str
    chain_id: int


def _env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def load_config() -> SendSettings:
    private_key = _env_value("POLY_PRIVATE_KEY", "POLYMARKET_PRIVATE_KEY")
    funder_address = _env_value("POLY_FUNDER", "POLYMARKET_FUNDER_ADDRESS")
    sig_raw = _env_value("POLY_SIGNATURE_TYPE") or "0"
    builder_key = _env_value("POLY_BUILDER_API_KEY", "POLYMARKET_BUILDER_API_KEY")
    builder_secret = _env_value("POLY_BUILDER_API_SECRET", "POLYMARKET_SECRET")
    builder_passphrase = _env_value("POLY_BUILDER_API_PASSPHRASE", "POLYMARKET_PASSPHRASE")
    relayer_url = _env_value("POLY_RELAYER_HOST") or DEFAULT_RELAYER_URL
    chain_raw = _env_value("POLY_CHAIN_ID") or "137"

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
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    try:
        signature_type = int(sig_raw)
    except (TypeError, ValueError):
        raise RuntimeError("POLY_SIGNATURE_TYPE must be an integer")

    try:
        chain_id = int(chain_raw)
    except (TypeError, ValueError):
        raise RuntimeError("POLY_CHAIN_ID must be an integer")

    return SendSettings(
        private_key=private_key,
        funder_address=funder_address,
        signature_type=signature_type,
        builder_api_key=builder_key,
        builder_api_secret=builder_secret,
        builder_api_passphrase=builder_passphrase,
        relayer_url=relayer_url,
        chain_id=chain_id,
    )


def build_relayer_client(config: SendSettings):
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
        if hasattr(client, "address") and client.address:
            proxy = client.address
    
    return client, proxy


def get_balance(token: str, wallet: str) -> int:
    for rpc_url in POLYGON_RPC_ENDPOINTS:
        try:
            selector = SEL_BALANCE.hex()
            addr = wallet[2:].lower().zfill(64)
            data = "0x" + selector + addr
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_call",
                "params": [{"to": token, "data": data}, "latest"],
            }
            resp = requests.post(rpc_url, json=payload, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            if "result" in data:
                return int(data["result"], 16)
        except Exception as e:
            print(f"RPC {rpc_url} failed: {e}")
            continue
    return 0


def execute_relayer_transaction(client: RelayClient, txn: SafeTransaction, label: str) -> dict:
    for attempt in range(2):
        try:
            print(f"  → Executing: {label}")
            resp = client.execute([txn], label)
            resp.wait()
            
            # Check for failure
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


def build_transfer_tx(to_address: str, amount: int) -> SafeTransaction:
    args = eth_encode(["address", "uint256"], [to_address, amount])
    data = "0x" + (SEL_TRANSFER + args).hex()
    return SafeTransaction(to=USDC_E, operation=OperationType.Call, data=data, value="0")


def main():
    parser = argparse.ArgumentParser(description="Send USDC.e to another address via relayer")
    parser.add_argument("recipient", help="Recipient address")
    parser.add_argument("amount", type=float, help="Amount in USDC.e")
    args = parser.parse_args()
    
    recipient = args.recipient.strip()
    amount = args.amount
    
    # Validate recipient address
    if not recipient.startswith("0x") or len(recipient) != 42:
        print("❌ Invalid recipient address. Must be 42 characters starting with 0x")
        sys.exit(1)
    
    print("=" * 70)
    print("SEND USDC.e TO ANOTHER ADDRESS")
    print("=" * 70)
    
    config = load_config()
    client, sender = build_relayer_client(config)
    
    wallet_type = "Proxy" if config.signature_type == 1 else "EOA"
    print(f"\nConfiguration:")
    print(f"  Sender:    {sender}")
    print(f"  Recipient: {recipient}")
    print(f"  Wallet:    {wallet_type} (sig_type={config.signature_type})")
    print(f"  Relayer:   {config.relayer_url}")
    print("-" * 70)
    
    # Check balance
    print("\n[1] Checking USDC.e balance...")
    balance = get_balance(USDC_E, sender)
    print(f"  Sender USDC.e: {balance / 1e6:.6f}")
    
    amount_raw = int(amount * 1e6)
    
    if balance < amount_raw:
        print(f"\n❌ Insufficient balance. Need {amount}, have {balance / 1e6:.6f}")
        sys.exit(1)
    
    print(f"\n[2] Preparing to send {amount:.6f} USDC.e")
    print(f"   From: {sender}")
    print(f"   To:   {recipient}")
    print("-" * 70)
    
    confirm = input("Confirm? [y/N]: ")
    if confirm.lower() != "y":
        print("Aborted")
        sys.exit(0)
    
    # Send
    print("\n[3] Sending USDC.e...")
    transfer_tx = build_transfer_tx(recipient, amount_raw)
    result = execute_relayer_transaction(client, transfer_tx, "transfer_usdce")
    
    if not result["ok"]:
        print(f"\n❌ Transfer failed: {result.get('error')}")
        sys.exit(1)
    
    # Check final balance
    time.sleep(2)
    final_balance = get_balance(USDC_E, sender)
    recipient_balance = get_balance(USDC_E, recipient)
    
    print("\n" + "=" * 70)
    print("✅ TRANSFER COMPLETE")
    print("=" * 70)
    print(f"Sent: {amount:.6f} USDC.e")
    print(f"From: {sender}")
    print(f"To:   {recipient}")
    if result.get("hash"):
        print(f"Tx:   https://polygonscan.com/tx/{result['hash']}")
    print("-" * 70)
    print(f"Sender balance:   {final_balance / 1e6:.6f} USDC.e")
    print(f"Recipient balance: {recipient_balance / 1e6:.6f} USDC.e")
    print("=" * 70)


if __name__ == "__main__":
    main()
