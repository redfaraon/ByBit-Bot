#!/usr/bin/env python3
"""
Per-user runner: applies user credentials, reads engine trade_plan and executes
decisions against the user's Bybit account. Creates/uses `runtime/<user>/bybit.log`.

Usage (Linux):
  python3 bybit_userbot.py --user <USER_ID>

This script is intentionally small: it relies on helpers from `bybitbot_impl.py`.
"""
import os
import json
import time
from pathlib import Path
import argparse

try:
    import ccxt
    import bybitbot_impl as impl
except Exception as exc:
    print(f"Import error: {exc}")
    raise

RUNTIME = Path("runtime")
ENGINE_DIR = RUNTIME / "engine"
TRADE_PLAN = ENGINE_DIR / "trade_plan.json"


def load_user_env_from_users_config(user_id: str) -> None:
    cfg = impl._load_users_config()
    for entry in cfg.get("users") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id")) == str(user_id):
            # apply secrets.env (users/<id>/secrets.env)
            secret_path = Path("users") / str(user_id) / impl.USERS_DEFAULT_SECRET
            if secret_path.exists():
                try:
                    for line in secret_path.read_text(encoding="utf-8").splitlines():
                        if not line or line.strip().startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        os.environ[k.strip()] = v.strip()
                except Exception:
                    pass
            # apply public env file if present
            public_path = Path("users") / str(user_id) / impl.USERS_PUBLIC_ENV_FILE
            if public_path.exists():
                try:
                    for line in public_path.read_text(encoding="utf-8").splitlines():
                        if not line or line.strip().startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
                except Exception:
                    pass
            # set BYBITBOT_USER_ID so bybitbot_impl helpers can use it
            os.environ["BYBITBOT_USER_ID"] = str(user_id)
            os.environ["BYBITBOT_MULTIUSER"] = "1"
            return
    # If not found, still try implicit secrets location
    secret_path = Path("users") / str(user_id) / impl.USERS_DEFAULT_SECRET
    if secret_path.exists():
        try:
            for line in secret_path.read_text(encoding="utf-8").splitlines():
                if not line or line.strip().startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip()
        except Exception:
            pass
    os.environ["BYBITBOT_USER_ID"] = str(user_id)
    os.environ["BYBITBOT_MULTIUSER"] = "1"


def create_user_exchange():
    # bybitbot_impl.init_exchange reads BYBIT_API_KEY and BYBIT_API_SECRET from env
    return impl.init_exchange()


def round_qty_for_symbol(symbol: str, qty: float) -> float:
    rules = impl._get_symbol_trade_rules(None, symbol) if False else impl._get_symbol_trade_rules  # placeholder
    # Use helper from module if present
    try:
        rules = impl._get_symbol_trade_rules(create_user_exchange(), symbol)
    except Exception:
        rules = {}
    min_qty = rules.get("min_qty") or 0.0
    step = rules.get("qty_step") or 0.0
    if step and step > 0:
        try:
            steps = int(round(qty / step))
            qty = max(step, steps * step)
        except Exception:
            pass
    if min_qty and qty < min_qty:
        qty = min_qty
    return float(qty)


def execute_trade_plan_for_user(user_id: str, dry_run: bool = False):
    if not TRADE_PLAN.exists():
        raise FileNotFoundError(f"Trade plan not found: {TRADE_PLAN}. Run engine first.")
    payload = json.loads(TRADE_PLAN.read_text(encoding="utf-8"))
    trade_plan = payload.get("trade_plan") or payload.get("selection") or payload
    if not trade_plan:
        raise RuntimeError("No trade_plan in snapshot")
    decisions = (trade_plan or {}).get("trade_plan") or trade_plan.get("decisions") or trade_plan.get("decisions") or trade_plan.get("selection", {}).get("decisions") or trade_plan.get("decisions")
    # alternative fallback: top-level 'trade_plan' key in snapshot
    if not decisions and isinstance(payload.get("trade_plan"), dict):
        decisions = payload["trade_plan"].get("trade_plan") or payload["trade_plan"].get("decisions")
    if not decisions:
        decisions = []
    # Prepare user exchange
    ex = create_user_exchange()
    # Get user equity to size positions
    equity, available_margin, _ = impl.fetch_usdt_equity(ex)
    impl._append_user_bybit_log(user_id, f"USERBOT starting run: equity={equity} available={available_margin}")
    # For each decision, try to execute. Decisions format varies; be defensive.
    for dec in decisions:
        try:
            symbol = dec.get("symbol") or dec.get("pair") or dec.get("ticker")
            if not symbol:
                continue
            symbol = impl.normalize_symbol(symbol, record_missing=False) or symbol
            action = (dec.get("action") or dec.get("side") or dec.get("direction") or "buy").lower()
            order_type = (dec.get("type") or dec.get("order_type") or "market").lower()
            # Determine notional in USDT
            notional = None
            for key in ("notional_usdt", "notional", "notionalUsd", "quote", "size_usdt"):
                if key in dec and dec.get(key) is not None:
                    try:
                        notional = float(dec.get(key))
                        break
                    except Exception:
                        notional = None
            if notional is None:
                pct = dec.get("notional_pct") or dec.get("allocation_pct") or dec.get("pct")
                try:
                    pct_v = float(pct)
                except Exception:
                    pct_v = None
                if pct_v is not None and equity and equity > 0:
                    # pct may be e.g. 0.05 or 5
                    if pct_v > 1:
                        pct_v = pct_v / 100.0
                    notional = max(0.0, float(equity) * float(pct_v))
            if notional is None:
                # fallback to a small allocation
                notional = min(50.0, float(equity) * 0.01 if equity and equity > 0 else 50.0)
            # Fetch price
            try:
                ticker = ex.fetch_ticker(symbol)
                price = ticker.get("last") or ticker.get("close") or ticker.get("info", {}).get("lastPrice")
            except Exception:
                price = None
            if not price:
                # try market info
                mkt = None
                try:
                    mkt = ex.market(symbol)
                except Exception:
                    mkt = None
                price = (mkt or {}).get("info", {}).get("lastPrice") if mkt else None
            if not price:
                impl._append_user_bybit_log(user_id, f"SKIP {symbol}: cannot determine price for decision {dec}")
                continue
            # Compute qty: for perpetual linear contracts, qty = notional / price
            qty = float(notional) / float(price) if price and float(price) > 0 else 0.0
            # Round/adjust qty according to market rules
            try:
                # attempt to use real rules
                rules = impl._get_symbol_trade_rules(ex, symbol)
                step = rules.get("qty_step") or rules.get("min_qty") or None
                if step and step > 0:
                    qty = max(step, (int(qty / step) * step))
            except Exception:
                pass
            qty = max(0.0, float(qty))
            if qty <= 0:
                impl._append_user_bybit_log(user_id, f"SKIP {symbol}: computed qty 0 for notional {notional}")
                continue
            if dry_run:
                impl._append_user_bybit_log(user_id, f"DRY RUN {symbol} {action} {order_type} qty={qty:.6f} price={price} notional={notional}")
                continue
            # Build params (support reduceOnly/trailing if present)
            params = dict(dec.get("params") or {})
            # place order
            try:
                if order_type == "market":
                    res = ex.create_order(symbol, "market", action, qty, None, params)
                else:
                    # limit or others
                    price_for_order = dec.get("price") or dec.get("limit") or price
                    res = ex.create_order(symbol, "limit", action, qty, float(price_for_order), params)
                impl._append_user_bybit_log(user_id, f"ORDER {symbol} {action} {order_type} qty={qty:.6f} price={price_for_order if order_type!="market" else 'market'} -> {res}")
            except Exception as exc:
                impl._append_user_bybit_log(user_id, f"ORDER FAIL {symbol}: {exc}")
        except Exception as exc_outer:
            impl._append_user_bybit_log(user_id, f"Decision processing error: {exc_outer}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a per-user bot that consumes engine trade plan")
    parser.add_argument("--user", required=True, help="User id to run for (must match users/users.json or users/<id>/secrets.env)")
    parser.add_argument("--dry-run", action="store_true", help="Do not send orders; just log intended actions")
    args = parser.parse_args()
    load_user_env_from_users_config(args.user)
    try:
        execute_trade_plan_for_user(args.user, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Userbot run failed: {exc}")
        raise
