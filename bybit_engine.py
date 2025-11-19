#!/usr/bin/env python3
"""
Engine-only runner: builds universe via OpenAI and persists selection + trade plan
into `runtime/engine/selection.json` and `runtime/engine/trade_plan.json` for userbots to consume.

Run on Linux as a scheduled service or cron.
"""
import json
import datetime
from pathlib import Path
import sys
from dotenv import dotenv_values

try:
    import ccxt
    import bybitbot_impl as impl
except Exception as exc:
    print(f"Engine import failed: {exc}", file=sys.stderr)
    raise

RUNTIME = Path("runtime")
ENGINE_DIR = RUNTIME / "engine"
ENGINE_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_SELECTION = ENGINE_DIR / "selection.json"
SNAPSHOT_TRADE_PLAN = ENGINE_DIR / "trade_plan.json"


def public_exchange():
    # Public-bybit client (no API keys) — sufficient for market data
    ex = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    try:
        impl._enable_exchange_logging(ex)
    except Exception:
        pass
    return ex


def run_once(pairs=None, quiet=False):
    ex = public_exchange()
    # positions_map empty: engine doesn't know per-user exposure
    positions_map = {}
    equity = 0.0
    available_margin = 0.0
    universe_cache = impl.load_universe_cache()
    symbol_candidates = list(set((impl.PAIR_LIST or []) + (impl.BASE_PAIR_CANDIDATES or [])))
    if pairs:
        symbol_candidates = [p for p in pairs if p]
    symbol_candidates = sorted(symbol_candidates)
    news_digest = impl._build_news_digest(symbol_candidates)
    res = impl.ai_update_universe(
        exchange=ex,
        symbols=symbol_candidates,
        positions_map=positions_map,
        equity=equity,
        available_margin=available_margin,
        universe_cache=universe_cache,
        news_digest=news_digest,
    )
    if not res:
        raise RuntimeError("Engine: ai_update_universe returned no result")
    selection_result, universe_state, news_requests = res
    # persist universe cache (used by main bot as well)
    universe_state = universe_state or {}
    universe_state["engine_generated_at"] = datetime.datetime.datetime.now(datetime.timezone.utc).isoformat()
    impl.save_universe_cache(universe_state)
    # Build portfolio bundle and trade plan (model-driven allocations)
    bundle, open_orders_cache = impl.build_portfolio_bundle(ex, selection_result or {}, {}, news_cache=None)
    trade_plan = impl.ai_plan_trades(
        ex,
        bundle,
        equity,
        available_margin,
        positions_snapshot={},
        pending_orders=open_orders_cache,
        stage="initial",
    )
    # write snapshots
    snapshot = {
        "selection": selection_result,
        "universe": universe_state,
        "news_requests": news_requests,
        "bundle": bundle,
        "trade_plan": trade_plan,
    }
    try:
        SNAPSHOT_SELECTION.write_text(json.dumps({"selection": selection_result, "universe": universe_state}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        SNAPSHOT_TRADE_PLAN.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    if not quiet:
        print("Engine: wrote selection ->", SNAPSHOT_SELECTION)
        print("Engine: wrote trade_plan ->", SNAPSHOT_TRADE_PLAN)
    return snapshot


def _load_user_registry(users_config_file: Path, users_dir: Path) -> dict[str, dict]:
    """Load user profiles from a JSON configuration file."""
    if not users_config_file.exists():
        return {}
    try:
        payload = json.loads(users_config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[USERS] Failed to parse {users_config_file}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(payload, dict):
        print(f"[USERS] Invalid registry format in {users_config_file}", file=sys.stderr)
        return {}
    registry = {}
    for entry in payload.get("users", []):
        if not isinstance(entry, dict):
            continue
        user_id = str(entry.get("id") or "").strip()
        if not user_id:
            continue
        registry[user_id] = {
            "enabled": entry.get("enabled", True),
            "label": entry.get("label", user_id),
            "env_overrides": entry.get("env", {}),
            "state_dir": users_dir / user_id
        }
    return registry


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Run engine-only universe selection + trade planning")
    p.add_argument("--pairs", help="Comma-separated list of symbols to consider", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    pairs = None
    if args.pairs:
        pairs = [s.strip() for s in args.pairs.split(",") if s.strip()]
    run_once(pairs=pairs, quiet=args.quiet)
