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
from dotenv import dotenv_values

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


def execute_trade_plan_for_user(user_id: str, dry_run: bool = False):
    if not TRADE_PLAN.exists():
        raise FileNotFoundError(f"Trade plan not found: {TRADE_PLAN}. Run engine first.")
    payload = json.loads(TRADE_PLAN.read_text(encoding="utf-8"))
    trade_plan = payload.get("trade_plan") or payload.get("selection") or payload
    if not trade_plan:
        raise RuntimeError("No trade_plan in snapshot")
    snapshot = payload
    ex = create_user_exchange()
    impl.apply_trade_plan_snapshot(ex, snapshot, user_id=user_id, dry_run=dry_run)


def execute_trade_plan(user_id: str, trade_plan_file: Path):
    """Execute a trade plan for a specific user."""
    if not trade_plan_file.exists():
        print(f"[USERBOT] Trade plan file not found: {trade_plan_file}")
        return

    try:
        with trade_plan_file.open("r", encoding="utf-8") as f:
            trade_plan = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[USERBOT] Failed to parse trade plan for user {user_id}: {exc}")
        return

    print(f"[USERBOT] Executing trade plan for user {user_id}: {trade_plan}")
    # Here you would add logic to execute the trades in the plan.


# Example usage in the userbot
if __name__ == "__main__":
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
    USER_ID = "example_user"
    TRADE_PLAN_FILE = Path(f"runtime/{USER_ID}/trade_plan.json")
    execute_trade_plan(USER_ID, TRADE_PLAN_FILE)
