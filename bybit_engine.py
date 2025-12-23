#!/usr/bin/env python3

"""
Standalone engine runner that reuses EngineCore without importing the main bot.
"""
from __future__ import annotations

MODULE_VERSION = "1.3.10"

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values

try:
    import ccxt
except Exception as exc:
    print(f"[ENGINE] ccxt import failed: {exc}", file=sys.stderr)
    raise

from engine_core import EngineCore, EngineSettings


REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.getenv("BYBITBOT_STATE_DIR", REPO_ROOT / "runtime"))
RUNTIME_DIR = Path(os.getenv("BYBITBOT_STATE_DIR", REPO_ROOT / "runtime"))
MAIN_LOG_MIRROR = Path(os.getenv("BYBIT_MAIN_LOG", REPO_ROOT / "bybit.log"))
MAIN_LOG_MAX_BYTES = int(float(os.getenv("BYBIT_MAIN_LOG_MAX_MB", "8")) * 1024 * 1024)
MAIN_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_MAIN_LOG_BACKUPS", "5")))


def _maybe_rotate_file(path: Path, max_bytes: int, backups: int) -> None:
    if max_bytes <= 0 or backups <= 0:
        return
    try:
        if not path.exists():
            return
        size = path.stat().st_size
    except OSError:
        return
    if size < max_bytes:
        return
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    rotated = path.with_name(f"{path.name}.{timestamp}")
    try:
        path.rename(rotated)
    except OSError:
        return
    try:
        rotated_files = sorted(
            [candidate for candidate in path.parent.glob(f"{path.name}.*") if candidate.is_file()],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for extra in rotated_files[backups:]:
        try:
            extra.unlink(missing_ok=True)
        except OSError:
            continue

class EngineWithMirror(EngineCore):
    """EngineCore that mirrors its log output into the main bybit.log as well."""

    def __init__(self, *args, mirror_path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mirror_path = mirror_path

    def _log(self, message: str) -> None:  # override
        try:
            print(message)
        except Exception:
            pass
        if self._mirror_path:
            try:
                self._mirror_path.parent.mkdir(parents=True, exist_ok=True)
                _maybe_rotate_file(self._mirror_path, MAIN_LOG_MAX_BYTES, MAIN_LOG_BACKUPS)
                with self._mirror_path.open("a", encoding="utf-8") as fp:
                    fp.write(message.rstrip() + "`n")
            except Exception:
                pass

def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        values = dotenv_values(env_path)
    except Exception:
        return
    for key, value in values.items():
        if value is None or key in os.environ:
            continue
        os.environ[key] = value


def create_public_exchange():
    params = {
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    ex = ccxt.bybit(params)
    ex.aiohttp_proxy = os.getenv("BYBITBOT_PROXY")
    verbose = os.getenv("BYBITBOT_CCXT_VERBOSE")
    if verbose and verbose.lower() in {"1", "true", "yes"}:
        ex.verbose = True
    return ex


def parse_symbol_override(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    result = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            result.append(token)
    return result or None


def run_cycle(engine: EngineCore, pairs: list[str] | None, quiet: bool) -> dict:
    snapshot = engine.build_trade_plan(symbol_candidates=pairs)
    if not quiet:
        selection = snapshot.get("selection") or {}
        print(
            "[ENGINE] snapshot generated:",
            f"{len(selection.get('pairs') or [])} pairs; volatility={selection.get('volatility')};",
            f"next_run≈{selection.get('next_run_minutes')} min",
        )
        news_items = snapshot.get("news") or {}
        trade_plan_payload = snapshot.get("trade_plan") or {}
        decisions = (trade_plan_payload.get("trade_plan") or trade_plan_payload.get("decisions")) or []
        print(f"[ENGINE] news items: {len(news_items)}; trade plan decisions: {len(decisions)}")
    return snapshot


def run_loop(
    engine: EngineCore,
    pairs: list[str] | None,
    quiet: bool,
    *,
    default_interval: float,
    min_interval: float,
    max_interval: float,
) -> None:
    delay_minutes = default_interval
    consecutive_errors = 0
    while True:
        cycle_start = time.perf_counter()
        try:
            snapshot = run_cycle(engine, pairs, quiet)
            consecutive_errors = 0
            delay_minutes = _resolve_next_delay(snapshot, default_interval, min_interval, max_interval)
        except Exception as exc:
            consecutive_errors += 1
            wait = min(max_interval, min_interval * (2 ** min(consecutive_errors, 5)))
            delay_minutes = wait
            print(f"[ENGINE] Cycle failed: {exc}")
        elapsed = time.perf_counter() - cycle_start
        sleep_minutes = max(min_interval, min(delay_minutes, max_interval))
        sleep_seconds = max(5.0, sleep_minutes * 60 - elapsed)
        if not quiet:
            print(f"[ENGINE] Sleeping for {sleep_seconds/60:.2f} minutes")
        time.sleep(sleep_seconds)


def _resolve_next_delay(snapshot: dict, default_interval: float, min_interval: float, max_interval: float) -> float:
    selection = snapshot.get("selection") or {}
    raw_next = selection.get("next_run_minutes") or (snapshot.get("universe") or {}).get("next_run_minutes")
    try:
        next_minutes = float(raw_next)
    except Exception:
        next_minutes = default_interval
    next_minutes = max(min_interval, min(max_interval, next_minutes or default_interval))
    return next_minutes


def main(argv: Iterable[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Autonomous Bybit engine runner")
    parser.add_argument("--pairs", help="Comma-separated list of symbols to consider", default=None)
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--loop", action="store_true", help="Run continuously instead of a single pass")
    parser.add_argument("--interval", type=float, default=float(os.getenv("ENGINE_DEFAULT_INTERVAL", "15")), help="Default interval between cycles (minutes)")
    parser.add_argument("--min-interval", type=float, default=float(os.getenv("ENGINE_MIN_INTERVAL", "5")), help="Minimum interval between cycles (minutes)")
    parser.add_argument("--max-interval", type=float, default=float(os.getenv("ENGINE_MAX_INTERVAL", "30")), help="Maximum interval between cycles (minutes)")
    args = parser.parse_args(argv)
    pairs = parse_symbol_override(args.pairs)
    exchange = create_public_exchange()
    settings = EngineSettings()
    engine = EngineWithMirror(exchange, settings=settings, runtime_dir=RUNTIME_DIR)
    if args.loop:
        run_loop(
            engine,
            pairs,
            args.quiet,
            default_interval=args.interval,
            min_interval=args.min_interval,
            max_interval=args.max_interval,
        )
    else:
        run_cycle(engine, pairs, args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
