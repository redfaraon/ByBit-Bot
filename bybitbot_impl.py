# -*- coding: utf-8 -*-
# Version: 2025.10.22.7
"""
Bybit Intraday AI Trading Bot — 30m, 5 пар USDT Perpetual
Сбалансированный интрадей-бот с поддержкой OpenAI GPT, Telegram и расширенным контекстом.
"""

import os
import shutil
import subprocess
import sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# --- Импорты ---
import math, time, json, traceback, datetime, random, warnings, re, numbers, hashlib
from pathlib import Path
from typing import Optional, Tuple, Any, Sequence
import pandas as pd
import ccxt
import requests
from colorama import Fore, Style, init
from openai import OpenAI
from dotenv import load_dotenv
try:
    from zoneinfo import ZoneInfo  # type: ignore
except ImportError:
    ZoneInfo = None  # type: ignore

try:
    import tiktoken  # type: ignore
except ImportError:
    tiktoken = None
try:
    import feedparser  # type: ignore
except ImportError:
    feedparser = None

# Версия бота: обновляйте при каждом релизе/значимых изменениях
BOT_VERSION = "2025.10.26.1"
BOT_CHANGELOG = (
    "Changelog is now sourced from the latest git commits."
)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
CHANGELOG_FILE = SCRIPT_DIR / "CHANGELOG.txt"
_LAST_COMMIT_HASH: Optional[str] = None

BASE_PAIR_CANDIDATES = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "BNB/USDT:USDT",
    "LTC/USDT:USDT",
    # "ADA/USDT:USDT",
    # "TRX/USDT:USDT",
    # "POL/USDT:USDT",
    # "LINK/USDT:USDT",
    # "AVAX/USDT:USDT",
    # "APT/USDT:USDT",
    # "ARB/USDT:USDT",
    # "OP/USDT:USDT",
    # "SUI/USDT:USDT",
    # "DOT/USDT:USDT",
    # "ATOM/USDT:USDT",
    # "UNI/USDT:USDT",
    # "FIL/USDT:USDT",
    # "NEAR/USDT:USDT",
    # "ETC/USDT:USDT",
    # "AAVE/USDT:USDT",
    # "XLM/USDT:USDT",
    # "HBAR/USDT:USDT",
]

BASE_INDICATOR_CANDIDATES = [
    "ema20",
    "ema50",
    "ema200",
    "sma50",
    "rsi14",
    "stoch14",
    "macd",
    "atr14",
    "bbands",
    "vwma20",
    "supertrend",
]

BASE_TIMEFRAME_CANDIDATES = ["5m", "15m", "30m", "1h", "2h", "4h", "1d"]
PAIR_TICKER_MAP = {pair: pair.split("/")[0].split(":")[0].upper() for pair in BASE_PAIR_CANDIDATES}
TICKER_TO_SYMBOL: dict[str, str] = {}
for pair, ticker in PAIR_TICKER_MAP.items():
    if not pair:
        continue
    if ticker:
        upper_ticker = ticker.upper()
        TICKER_TO_SYMBOL.setdefault(upper_ticker, pair)
        TICKER_TO_SYMBOL.setdefault(f"{upper_ticker}USDT", pair)
    sanitized = re.sub(r"[^A-Z0-9]", "", pair.upper())
    if sanitized:
        TICKER_TO_SYMBOL.setdefault(sanitized, pair)

PAIR_CANDIDATE_LIMIT = int(os.getenv("PAIR_CANDIDATE_LIMIT", "25"))
PAIR_PREFETCH_LIMIT = int(os.getenv("PAIR_PREFETCH_LIMIT", "30"))

SYMBOL_ALIASES = {
    "MATIC/USDT:USDT": "POL/USDT:USDT",
}

for alias, target in SYMBOL_ALIASES.items():
    PAIR_TICKER_MAP.setdefault(alias, alias.split("/")[0].split(":")[0].upper())
    PAIR_TICKER_MAP.setdefault(target, target.split("/")[0].split(":")[0].upper())

for pair, ticker in PAIR_TICKER_MAP.items():
    if not pair:
        continue
    if ticker:
        upper_ticker = ticker.upper()
        TICKER_TO_SYMBOL.setdefault(upper_ticker, pair)
        TICKER_TO_SYMBOL.setdefault(f"{upper_ticker}USDT", pair)
    sanitized = re.sub(r"[^A-Z0-9]", "", pair.upper())
    if sanitized:
        TICKER_TO_SYMBOL.setdefault(sanitized, pair)

for pair, ticker in PAIR_TICKER_MAP.items():
    if not pair:
        continue
    if ticker:
        upper_ticker = ticker.upper()
        TICKER_TO_SYMBOL.setdefault(upper_ticker, pair)
        TICKER_TO_SYMBOL.setdefault(f"{upper_ticker}USDT", pair)
    sanitized = re.sub(r"[^A-Z0-9]", "", pair.upper())
    if sanitized:
        TICKER_TO_SYMBOL.setdefault(sanitized, pair)


TIMEFRAME_NORMALIZATION_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "45m": "45m",
    "60m": "1h",
    "60": "1h",
    "1h": "1h",
    "1hr": "1h",
    "1hour": "1h",
    "1hours": "1h",
    "2h": "2h",
    "120m": "2h",
    "4h": "4h",
    "240m": "4h",
    "240": "4h",
    "4hour": "4h",
    "4hours": "4h",
    "6h": "6h",
    "360m": "6h",
    "12h": "12h",
    "720m": "12h",
    "1d": "1d",
    "24h": "1d",
    "1440m": "1d",
    "1440": "1d",
    "1day": "1d",
    "30": "30m",
}


def normalize_requested_timeframe(tf_value: Any, default: str = "4h") -> str:
    """Convert various timeframe spellings into ccxt-friendly identifiers."""
    if tf_value is None:
        return default
    text = str(tf_value).strip()
    if not text:
        return default
    lowered = text.lower().strip()
    if lowered.startswith("higher_tf"):
        parts = lowered.split(":", 1)
        lowered = parts[1] if len(parts) == 2 and parts[1] else lowered.replace("higher_tf", "", 1)
        lowered = lowered.strip()
    lowered = lowered.replace(" ", "").replace("-", "")
    replacements = {
        "hours": "h",
        "hour": "h",
        "hrs": "h",
        "hr": "h",
        "minutes": "m",
        "minute": "m",
        "mins": "m",
        "min": "m",
    }
    for source, target in replacements.items():
        if source in lowered:
            lowered = lowered.replace(source, target)
    if lowered in TIMEFRAME_NORMALIZATION_MAP:
        return TIMEFRAME_NORMALIZATION_MAP[lowered]
    match = re.fullmatch(r'(\d+)([mh])?', lowered)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "h" or (unit is None and value >= 60 and value % 60 == 0):
            hours = value if unit == "h" else value // 60
            return f"{hours}h"
        if unit == "m" or unit is None:
            return f"{value}m"
    return lowered or default

# Подавляем FutureWarning от pandas
warnings.filterwarnings("ignore", category=FutureWarning)
init(autoreset=True)

LOG_TZINFO = None
LOG_TIMEZONE = ""
_LOG_TZ_WARNING_EMITTED = False

DEFAULT_NEXT_RUN_MINUTES = 28.0
RUNTIME_STATUS_FILE = Path(__file__).with_name("runtime_status.json")
CHANGELOG_STATE_FILE = Path(__file__).with_name("changelog_state.json")
UNIVERSE_CACHE_FILE = Path(__file__).with_name("universe_cache.json")


class ProtectionMissingError(RuntimeError):
    """Raised when open positions remain without mandatory protective orders."""


def _is_reduce_only(order) -> bool:
    try:
        value = order.get("reduceOnly")
    except AttributeError:
        return False
    return value in (True, "true", "1", 1)


def _has_stop_flag(order) -> bool:
    order_type = (order.get("type") or "").lower()
    stop_price = safe_float(order.get("stopPrice") or order.get("triggerPrice") or order.get("stopLoss"))
    if order_type in ("stop", "stoploss", "stop_limit", "stoplimit"):
        return True
    return stop_price is not None


def _has_trailing_flag(order) -> bool:
    order_type = (order.get("type") or "").lower()
    trailing_val = order.get("trailingStop")
    try:
        if trailing_val not in (None, ""):
            float(trailing_val)
            return True
    except (TypeError, ValueError):
        pass
    return order_type == "trailingstop"


def ensure_version_backup() -> None:
    """Create a versioned backup of this script if it does not already exist."""
    try:
        script_path = Path(__file__).resolve()
    except (NameError, OSError):
        return

    backup_dir = script_path.parent / "backups"
    try:
        backup_dir.mkdir(exist_ok=True)
    except Exception:
        return

    backup_name = f"{script_path.stem}_v{BOT_VERSION}{script_path.suffix}"
    backup_path = backup_dir / backup_name
    if backup_path.exists():
        return

    try:
        shutil.copy2(script_path, backup_path)
        print(f"[INFO] Created backup {backup_path.name}")
    except Exception as exc:
        print(f"[WARN] Failed to create backup '{backup_path.name}': {exc}")


def safe_float(val):
    try:
        if val is None or val == "":
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            cleaned = val.replace(',', '').strip()
            return float(cleaned)
    except (ValueError, TypeError):
        return None
    return None


def safe_int(val):
    try:
        if val is None or val == "":
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            if math.isnan(val):
                return None
            return int(round(val))
        if isinstance(val, str):
            cleaned = val.replace(',', '').strip()
            if not cleaned:
                return None
            return int(float(cleaned))
    except (ValueError, TypeError):
        return None
    return None


def _resolve_symbol_leverage(decision: dict, symbol_meta: dict, current_position: dict | None, default: int | None = None) -> int:
    fallback = default if default is not None else LEVERAGE
    candidates = [
        (decision or {}).get("leverage"),
        ((decision or {}).get("config") or {}).get("leverage") if isinstance(decision, dict) else None,
        ((decision or {}).get("risk") or {}).get("leverage") if isinstance(decision, dict) else None,
        (symbol_meta or {}).get("leverage"),
        ((symbol_meta or {}).get("config") or {}).get("leverage") if isinstance(symbol_meta, dict) else None,
        (symbol_meta or {}).get("risk", {}).get("leverage") if isinstance(symbol_meta, dict) else None,
        (current_position or {}).get("leverage") if isinstance(current_position, dict) else None,
        fallback,
    ]
    for value in candidates:
        leverage_val = safe_int(value)
        if leverage_val and leverage_val > 0:
            return leverage_val
    return max(1, safe_int(fallback) or 1)


def _set_symbol_leverage(exchange, symbol: str, leverage: int, current_position: dict | None = None):
    leverage_val = safe_int(leverage)
    if leverage_val is None or leverage_val <= 0:
        return
    current_lev = safe_float((current_position or {}).get("leverage") if isinstance(current_position, dict) else None)
    if current_lev is not None and math.isfinite(current_lev) and abs(current_lev - leverage_val) < 1e-6:
        return
    try:
        exchange.set_leverage(leverage_val, symbol)
    except Exception as e:
        code = get_bybit_retcode(e)
        if code == 110043:
            log(f"ℹ️ Плечо {leverage_val}x уже установлено для {symbol} (код {code})", Fore.LIGHTBLACK_EX)
        else:
            log(f"⚠️ Не удалось установить плечо {leverage_val}x для {symbol}: {e}", Fore.YELLOW)



def _extract_decision_position_size(decision, symbol_meta=None):
    """Extract an absolute size or notional for the primary action from AI payload."""
    sources: list[dict] = []
    if isinstance(decision, dict):
        sources.append(decision)
        for key in ("config", "sizing", "risk"):
            value = decision.get(key)
            if isinstance(value, dict):
                sources.append(value)
    if isinstance(symbol_meta, dict):
        sources.append(symbol_meta)
        for key in ("config", "sizing", "risk"):
            value = symbol_meta.get(key)
            if isinstance(value, dict):
                sources.append(value)
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("amount", "qty", "quantity", "contracts", "size", "units"):
            val = source.get(key)
            amount = safe_float(val)
            if amount and amount > 0:
                return amount, None
        for key in ("notional", "notional_usdt", "quote", "notionalQuote"):
            val = source.get(key)
            notional = safe_float(val)
            if notional and notional > 0:
                return None, notional
    return None, None

def detect_unprotected_positions(exchange, positions_map) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for symbol, position in (positions_map or {}).items():
        amount = safe_float((position or {}).get("amount") or (position or {}).get("contracts"))
        if amount is None or not math.isfinite(amount) or abs(amount) == 0:
            continue
        try:
            orders = fetch_open_orders_for_symbol(exchange, symbol)
        except Exception as exc:
            missing.append((symbol, f"orders unavailable: {exc}"))
            continue
        has_stop = False
        has_trailing = False
        for order in orders or []:
            if not isinstance(order, dict):
                continue
            reduce_only = _is_reduce_only(order)
            if reduce_only and _has_stop_flag(order):
                has_stop = True
            if _has_trailing_flag(order) and (reduce_only or order.get("reduceOnly") is None):
                has_trailing = True
        if not (has_stop or has_trailing):
            missing.append((symbol, "no stop-loss or trailing-stop"))
    return missing


def _read_change_log_entry() -> Tuple[str, str]:
    version = BOT_VERSION
    changelog_text = BOT_CHANGELOG
    try:
        raw = CHANGELOG_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return version, changelog_text
    if not raw:
        return version, changelog_text
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    if not blocks:
        return version, changelog_text
    latest_block = blocks[-1]
    lines = [line.strip() for line in latest_block.splitlines() if line.strip()]
    if not lines:
        return version, changelog_text
    new_version = lines[0]
    body = "\n".join(lines[1:]).strip()
    if not body:
        body = _build_commit_changelog()
    return new_version, body


def _collect_news_pairs(limit: int = 40) -> set[str]:
    dynamic_pairs: set[str] = set()
    try:
        rss_payload = get_news_from_rss("", limit)
    except Exception as exc:
        log(f"⚠️ Не удалось собрать RSS-новости для расширения универсума: {exc}", Fore.YELLOW)
        rss_payload = {}
    items = (rss_payload or {}).get("items") or []
    for item in items:
        title = ((item or {}).get("title") or "").upper()
        if not title:
            continue
        for pair, ticker in PAIR_TICKER_MAP.items():
            if ticker and ticker in title:
                dynamic_pairs.add(pair)
    return dynamic_pairs


def _build_commit_changelog(limit: int = 8) -> str:
    limit = max(1, limit)
    try:
        result = subprocess.run(
            ["git", "log", "-n", str(limit), "--pretty=format:%cs %h %s"],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
            check=True,
        )
    except Exception as exc:
        return f"git log unavailable: {exc}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return "No recent commits available."
    return "Recent commits:\n" + "\n".join(f"- {line}" for line in lines)


def _current_git_head() -> Optional[str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
            check=True,
        )
        return head.stdout.strip()
    except Exception:
        return None


def maybe_refresh_metadata() -> dict[str, Any]:
    global BOT_VERSION, BOT_CHANGELOG, _LAST_COMMIT_HASH

    previous_hash = _LAST_COMMIT_HASH
    head = _current_git_head()
    commit_changed = previous_hash is not None and head is not None and head != previous_hash
    if head is not None:
        _LAST_COMMIT_HASH = head

    new_version, new_changelog = _read_change_log_entry()
    metadata_changed = (new_version != BOT_VERSION) or (new_changelog != BOT_CHANGELOG)

    if not commit_changed and not metadata_changed:
        return {
            "commit_changed": False,
            "metadata_changed": False,
            "version_changed": False,
            "reload_required": False,
            "current_hash": head,
            "previous_hash": previous_hash,
        }

    previous_version = BOT_VERSION
    BOT_VERSION = new_version or BOT_VERSION
    BOT_CHANGELOG = new_changelog or BOT_CHANGELOG
    os.environ["BYBITBOT_CHANGELOG_VERSION"] = BOT_VERSION
    os.environ["BYBITBOT_CHANGELOG_TEXT"] = BOT_CHANGELOG

    version_changed = BOT_VERSION != previous_version
    if version_changed:
        ensure_version_backup()
        log(f"🆕 Обнаружена новая версия: {previous_version} → {BOT_VERSION}", Fore.LIGHTBLUE_EX)
        send_tg(f"🆕 Обновлена версия до {BOT_VERSION}")
    elif metadata_changed:
        log("ℹ️ Обновлён changelog без изменения версии.", Fore.LIGHTBLACK_EX)
        send_tg("ℹ️ Обновлён changelog без изменения версии.")
    elif commit_changed and previous_hash is not None:
        log("ℹ️ Обновлена HEAD коммита без изменения changelog.", Fore.LIGHTBLACK_EX)

    return {
        "commit_changed": bool(commit_changed),
        "metadata_changed": bool(metadata_changed),
        "version_changed": bool(version_changed),
        "reload_required": bool(commit_changed),
        "current_hash": head,
        "previous_hash": previous_hash,
    }


def _parse_version_tuple(version: str) -> Tuple[int, ...]:
    parts = []
    for chunk in version.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            digits = "".join(ch for ch in chunk if ch.isdigit())
            parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _find_previous_version_script(current_version: str, script_path: Path) -> Optional[Tuple[str, Path]]:
    backup_dir = script_path.parent / "backups"
    if not backup_dir.exists():
        return None
    current_tuple = _parse_version_tuple(current_version)
    prefix = f"{script_path.stem}_v"
    suffix = script_path.suffix
    candidates = []
    for backup_file in backup_dir.glob(f"{prefix}*{suffix}"):
        name = backup_file.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        version_str = name[len(prefix):-len(suffix)]
        if not version_str:
            continue
        version_tuple = _parse_version_tuple(version_str)
        if version_tuple < current_tuple:
            candidates.append((version_tuple, version_str, backup_file))
    if not candidates:
        return None
    _, version_str, backup_path = max(candidates, key=lambda item: item[0])
    return version_str, backup_path


def _run_previous_version_script(script_path: Path) -> int:
    python_executable = sys.executable or "python"
    result = subprocess.run([python_executable, str(script_path)], check=False)
    return result.returncode


def _build_news_digest(symbols):
    digest = {}
    for sym in symbols:
        try:
            news_payload = get_news(sym)
            if not isinstance(news_payload, dict):
                continue
            summary = news_payload.get("summary") or ""
            items = news_payload.get("items") or []
            trimmed_items = []
            for item in items[: max(1, min(len(items), 3))]:
                trimmed_items.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "published_at": item.get("published_at"),
                })
            digest[sym] = {
                "summary": summary,
                "items": trimmed_items
            }
        except Exception as exc:
            log(f"[WARN] Failed to fetch news headlines for {sym}: {exc}", Fore.YELLOW)
    return digest


def _parse_indicator_name(name: str) -> Tuple[str, Optional[int]]:
    base = "".join(ch for ch in name if ch.isalpha()).lower()
    digits = "".join(ch for ch in name if ch.isdigit())
    length = int(digits) if digits else None
    return base, length


def _expand_indicator_entries(entry) -> list:
    result = []
    if entry is None:
        return result
    if isinstance(entry, str):
        val = entry.strip()
        if val:
            result.append(val)
        return result
    if isinstance(entry, (list, tuple, set)):
        for item in entry:
            result.extend(_expand_indicator_entries(item))
        return result
    if isinstance(entry, dict):
        name = entry.get("indicator") or entry.get("name")
        if name:
            lengths = (
                entry.get("length")
                or entry.get("period")
                or entry.get("window")
                or entry.get("n")
                or entry.get("size")
            )
            if isinstance(lengths, (list, tuple, set)):
                for length in lengths:
                    try:
                        val = int(length)
                        result.append(f"{name}{val}")
                    except (TypeError, ValueError):
                        continue
            elif lengths is not None:
                try:
                    val = int(lengths)
                    result.append(f"{name}{val}")
                except (TypeError, ValueError):
                    result.append(str(name))
            else:
                result.append(str(name))
        nested = entry.get("indicators") or entry.get("names")
        if nested:
            result.extend(_expand_indicator_entries(nested))
        for key, value in entry.items():
            if key in ("indicator", "name", "indicators", "names", "length", "period", "window", "n", "size"):
                continue
            if isinstance(value, (list, tuple, set)):
                for val in value:
                    result.extend(_expand_indicator_entries({"indicator": key, "length": val}))
            elif isinstance(value, (int, float)):
                result.extend(_expand_indicator_entries({"indicator": key, "length": value}))
        return result
    return result


def _apply_indicator_to_df(df: pd.DataFrame, indicator_name: str) -> Optional[str]:
    if df.empty:
        return None
    base, length = _parse_indicator_name(indicator_name)
    try:
        if base == "ema" and length:
            col = f"ema{length}"
            df[col] = ema(df["close"], length)
            return col
        if base == "sma" and length:
            col = f"sma{length}"
            df[col] = df["close"].rolling(length).mean()
            return col
        if base == "rsi":
            period = length or 14
            col = f"rsi{period}"
            df[col] = rsi(df["close"], period)
            return col
        if base == "atr":
            period = length or 14
            col = f"atr{period}"
            df[col] = atr(df, period)
            return col
        if base == "stoch":
            period = length or 14
            col = f"stoch{period}"
            low_min = df["low"].rolling(period).min()
            high_max = df["high"].rolling(period).max()
            df[col] = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, pd.NA)
            return col
    except Exception as exc:
        log(f"[WARN] Failed to apply indicator {indicator_name}: {exc}", Fore.YELLOW)
    return None


def _serialize_df(df: pd.DataFrame, limit: int = 80):
    if df.empty:
        return []
    slice_df = df.tail(limit)
    result = []
    for row in slice_df.to_dict("records"):
        clean_row = {}
        for key, value in row.items():
            if isinstance(value, (pd.Timestamp, datetime.datetime)):
                clean_row[key] = value.isoformat()
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                continue
            else:
                if pd.isna(value):
                    continue
                clean_row[key] = float(value) if isinstance(value, (int, float)) else value
        result.append(clean_row)
    return result


def prepare_symbol_dataset(exchange, symbol: str, timeframes: list, indicators: list, news_cache=None):
    dataset = {"symbol": symbol, "timeframes": {}, "indicators": [], "errors": []}
    indicators = indicators or []
    timeframes = timeframes or [TIMEFRAME, "4h"]
    seen_cols = set()
    for tf in timeframes:
        try:
            df_tf = fetch_df(exchange, symbol, tf)
        except Exception as exc:
            err = f"fetch_df({tf}) failed: {exc}"
            dataset["errors"].append(err)
            log(f"[WARN] {symbol}: {err}", Fore.YELLOW)
            continue
        applied = []
        for ind in indicators:
            col = _apply_indicator_to_df(df_tf, ind)
            if col:
                applied.append(col)
                seen_cols.add(col)
        dataset["timeframes"][tf] = {
            "bars": _serialize_df(df_tf)
        }
        if applied:
            dataset["timeframes"][tf]["indicators"] = applied
    dataset["indicators"] = sorted(seen_cols)
    dataset["position"] = None
    dataset["open_orders"] = []
    if news_cache and symbol in news_cache:
        dataset["news"] = news_cache[symbol]
    return dataset


def ai_update_universe(exchange, symbols, positions_map, equity, available_margin, universe_cache, news_digest=None):
    if not AI_KEY:
        log("[AI] OPENAI_API_KEY missing for universe update", Fore.RED)
        return None
    client = OpenAI(api_key=AI_KEY, timeout=30)
    if not news_digest:
        news_digest = _build_news_digest(symbols)
    positions_compact: list[dict[str, Any]] = []
    symbol_set = set(symbols) | set(positions_map.keys())
    for sym in sorted(symbol_set):
        pos = positions_map.get(sym)
        if not pos:
            continue
        positions_compact.append(
            {
                "symbol": sym,
                "side": pos.get("side"),
                "amount": pos.get("amount"),
                "entryPrice": pos.get("entryPrice"),
                "leverage": pos.get("leverage"),
                "unrealizedPnl": pos.get("unrealizedPnl"),
            }
        )
    timestamp_now = datetime.datetime.now(datetime.timezone.utc)
    request_payload = {
        "candidate_pairs": sorted(symbol_set),
        "indicator_candidates": [
            {"indicator": "ema", "length": 20},
            {"indicator": "ema", "length": 50},
            "volume",
            "rsi14",
            "macd",
            "atr14",
            "stoch14",
        ],
        "timeframe_candidates": ["15m", "30m", "1h", "2h", "4h"],
        "equity_usdt": equity,
        "available_margin_usdt": available_margin,
        "positions": positions_compact,
        "news_headlines": news_digest,
        "session_timestamp": timestamp_now.isoformat(),
        "session_weekday": timestamp_now.strftime("%A"),
        "session_hour": timestamp_now.strftime("%H:%M"),
        "universe_cache": None,
    }
    system_msg = (
        "You are an AI portfolio strategist for Bybit. Update the trading universe based on the provided "
        "candidate pairs, the existing universe cache, and the supplied news headlines. "
        "Respond strictly in JSON with keys: universe (pairs, global_timeframes, global_indicators, notes), "
        "targets (per-symbol recommendations), and news_requests (symbols that require full news text)."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
    ]
    hard_limit = TOKEN_LIMIT if TOKEN_LIMIT > 0 else None
    per_cap = _current_request_token_cap()
    if per_cap:
        hard_limit = per_cap if hard_limit is None else min(hard_limit, per_cap)
    token_estimate = estimate_tokens(messages, AI_MODEL)
    if hard_limit:
        shrink_attempts = 0
        while token_estimate > hard_limit and shrink_attempts < 3:
            for sym, payload in list(news_digest.items()):
                summary = payload.get("summary") or ""
                payload["items"] = (payload.get("items") or [])[:1]
                if len(summary) > 160:
                    news_digest[sym]["summary"] = summary[:157] + "..."
            messages[1]["content"] = json.dumps(request_payload, ensure_ascii=False)
            token_estimate = estimate_tokens(messages, AI_MODEL)
            shrink_attempts += 1
    if per_cap and token_estimate > per_cap:
        log(f"[AI] universe update: unable to compress request below {per_cap} tokens", Fore.YELLOW)
        return None
    if not _ensure_token_budget(token_estimate, AI_MODEL, "universe update"):
        return None
    _log_ai_request(AI_MODEL, token_estimate, "universe update")
    try:
        res = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as exc:
        log(f"[ERROR] OpenAI universe update: {exc}", Fore.RED)
        return None
    _register_ai_usage(AI_MODEL, getattr(res, "usage", None), "universe update")
    try:
        payload = res.choices[0].message.content
    except Exception:
        return None
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        log(f"[WARN] JSON decode (universe update): {exc}", Fore.YELLOW)
        return None
    selection_payload = result.get("selection") or result
    universe_payload = result.get("universe") or {}
    news_requests = result.get("news_requests") or []
    if isinstance(selection_payload, dict):
        selection_payload["_news_digest"] = news_digest
    if not universe_payload.get("pairs"):
        fallback_pairs = selection_payload.get("pairs") or [
            item.get("symbol")
            for item in (selection_payload.get("targets") or [])
            if isinstance(item, dict) and item.get("symbol")
        ]
        universe_payload["pairs"] = fallback_pairs
    universe_payload.setdefault("global_timeframes", selection_payload.get("global_timeframes") or [])
    universe_payload.setdefault("global_indicators", selection_payload.get("global_indicators") or [])
    return selection_payload, universe_payload, news_requests


def build_portfolio_bundle(exchange, selection_result, positions_map, news_cache=None, extra_symbols=None):
    targets = (selection_result or {}).get("targets") or []
    global_timeframes = set((selection_result or {}).get("global_timeframes") or [])
    global_indicators = set((selection_result or {}).get("global_indicators") or [])
    bundle = {"symbols": [], "meta": {}}
    open_orders_cache = {}
    processed_symbols: set[str] = set()
    for target in targets:
        symbol = target.get("symbol")
        if not symbol:
            continue
        base_timeframes = list(global_timeframes)
        timeframes = list(dict.fromkeys(base_timeframes + (target.get("timeframes") or [])))
        baseline_indicators = (list(global_indicators) or list(BASE_INDICATOR_CANDIDATES))[:3]
        indicator_candidates = (target.get("indicators") or []) + list(global_indicators)
        indicators = list(dict.fromkeys(baseline_indicators + indicator_candidates))
        dataset = prepare_symbol_dataset(exchange, symbol, timeframes, indicators, news_cache=news_cache)
        dataset["target"] = {
            "notional_pct": target.get("notional_pct"),
            "timeframes": timeframes,
            "indicators": indicators,
            "notes": target.get("notes"),
        }
        position_payload = positions_map.get(symbol)
        if position_payload:
            dataset["position"] = position_payload
        try:
            open_orders = fetch_open_orders_for_symbol(exchange, symbol)
        except Exception as exc:
            log(f"[WARN] fetch_open_orders {symbol}: {exc}", Fore.YELLOW)
            open_orders = []
        dataset["open_orders"] = open_orders
        open_orders_cache[symbol] = open_orders
        bundle["symbols"].append(dataset)
        processed_symbols.add(symbol)
    for extra_symbol in extra_symbols or []:
        if not extra_symbol or extra_symbol in processed_symbols:
            continue
        timeframes = [TIMEFRAME, "4h"]
        indicators = list(global_indicators)
        dataset = prepare_symbol_dataset(exchange, extra_symbol, timeframes, indicators, news_cache=news_cache)
        dataset.setdefault("meta", {})["source"] = "non_reduce_orders"
        position_payload = positions_map.get(extra_symbol)
        if position_payload:
            dataset["position"] = position_payload
        try:
            open_orders = fetch_open_orders_for_symbol(exchange, extra_symbol)
        except Exception as exc:
            log(f"[WARN] fetch_open_orders {extra_symbol}: {exc}", Fore.YELLOW)
            open_orders = []
        dataset["open_orders"] = open_orders
        open_orders_cache[extra_symbol] = open_orders
        bundle["symbols"].append(dataset)
        processed_symbols.add(extra_symbol)
    bundle["meta"] = {
        "global_timeframes": list(global_timeframes),
        "global_indicators": list(global_indicators),
        "confidence": (selection_result or {}).get("confidence"),
        "reason": (selection_result or {}).get("reason"),
    }
    return bundle, open_orders_cache


def augment_bundle_with_needs(exchange, bundle, needs, news_cache=None, news_full=None):
    if not needs:
        return bundle
    symbol_map = {entry['symbol']: entry for entry in bundle.get('symbols', []) if entry.get('symbol')}
    for need in needs:
        if isinstance(need, dict):
            symbol = need.get('symbol')
            if symbol not in symbol_map:
                continue
            dataset = symbol_map[symbol]
            additional_tf_raw = need.get('higher_tf')
            additional_tf_list: list[str] = []
            if additional_tf_raw:
                if isinstance(additional_tf_raw, (list, tuple, set)):
                    additional_tf_list = [str(tf) for tf in additional_tf_raw if tf]
                else:
                    additional_tf_list = [str(additional_tf_raw)]
            timeframe_requests: list[str] = []
            for tf in additional_tf_list:
                mapped_tf = normalize_requested_timeframe(tf, default="4h")
                if mapped_tf and mapped_tf not in timeframe_requests:
                    timeframe_requests.append(mapped_tf)
            raw_timeframes = need.get('timeframes') or []
            if isinstance(raw_timeframes, (str, bytes)):
                raw_timeframes = [raw_timeframes]
            for tf in raw_timeframes:
                mapped_tf = normalize_requested_timeframe(tf, default=TIMEFRAME)
                if mapped_tf and mapped_tf not in timeframe_requests:
                    timeframe_requests.append(mapped_tf)
            requested_timeframes = timeframe_requests or [TIMEFRAME]
            requested_indicators = need.get('indicators') or []
            if isinstance(requested_indicators, (str, bytes)):
                requested_indicators = [requested_indicators]
            for tf in requested_timeframes:
                tf_label = str(tf)
                try:
                    df_tf = fetch_df(exchange, symbol, tf_label)
                except Exception as exc:
                    dataset.setdefault('errors', []).append(f"needs fetch_df({tf_label}): {exc}")
                    continue
                applied = []
                for ind in requested_indicators:
                    col = _apply_indicator_to_df(df_tf, ind)
                    if col:
                        applied.append(col)
                dataset.setdefault('timeframes', {})[tf_label] = {'bars': _serialize_df(df_tf)}
                if applied:
                    dataset['timeframes'][tf_label]['indicators'] = applied
            if need.get('funding'):
                try:
                    dataset['funding'] = get_funding_rate(exchange, symbol)
                except Exception as exc:
                    dataset.setdefault('errors', []).append(f"needs funding: {exc}")
            if need.get('open_interest'):
                try:
                    dataset['open_interest'] = get_open_interest(exchange, symbol)
                except Exception as exc:
                    dataset.setdefault('errors', []).append(f"needs open_interest: {exc}")
            if need.get('news'):
                if news_full and symbol in news_full:
                    dataset['news'] = news_full[symbol]
                elif news_cache:
                    dataset['news'] = news_cache.get(symbol) or dataset.get('news')
            continue
        if isinstance(need, str):
            bundle.setdefault('meta', {}).setdefault('extra_requests', []).append(need)
    return bundle



def _shrink_bundle_for_tokens(bundle, max_bars=60):
    for entry in bundle.get("symbols", []):
        for tf_data in entry.get("timeframes", {}).values():
            bars = tf_data.get("bars") or []
            if len(bars) > max_bars:
                tf_data["bars"] = bars[-max_bars:]


def ai_plan_trades(
    exchange,
    bundle,
    equity,
    available_margin,
    positions_snapshot=None,
    pending_orders=None,
    stage="initial",
):
    if not AI_KEY:
        log("?? �� 㪠��� OPENAI_API_KEY (stage plan)", Fore.RED)
        return None
    client = OpenAI(api_key=AI_KEY, timeout=40)
    positions_payload = _compact_positions_snapshot(positions_snapshot)
    pending_orders_payload = _compact_orders_snapshot(pending_orders)
    payload = {
        "stage": stage,
        "equity_usdt": equity,
        "available_margin_usdt": available_margin,
        "data": bundle,
    }
    if positions_payload:
        payload["positions_snapshot"] = positions_payload
    if pending_orders_payload:
        payload["pending_orders"] = pending_orders_payload
    system_msg = (
        "You are a trade analyst for Bybit. For each symbol evaluate positions, orders, candles, and indicators. "
        "Return strict JSON in the form:\n"
        '  "decisions": [\n'
        '    {\n'
        '      "symbol": "PAIR",\n'
        '      "action": "open|close|manage|reduce|skip",\n'
        '      "side": "buy|sell",\n'
        '      "notional_pct": float,\n'
        '      "reason": "short explanation",\n'
        '      "tp_atr": float,\n'
        '      "sl_atr": float,\n'
        '      "orders": [ {...} ],\n'
        '      "cancel_orders": [...],\n'
        '      "replace_orders": [ {...} ]\n'
        "    }\n"
        "  ],\n"
        '  "needs": [ {"symbol":"PAIR","timeframes":["1h"],"indicators":["ema100"]}, "news" ],\n'
        '  "next_run_minutes": float,\n'
        '  "next_run_time": "2025-01-01T10:30:00Z",\n'
        '  "notes": "optional"\n'
        "}\n"
        "If additional context is required, populate 'needs' and leave 'decisions' empty."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    hard_limit = TOKEN_LIMIT if TOKEN_LIMIT > 0 else None
    soft_limit = TOKEN_SOFT_LIMIT if TOKEN_SOFT_LIMIT > 0 else None
    per_cap = _current_request_token_cap()
    if per_cap:
        hard_limit = per_cap if hard_limit is None else min(hard_limit, per_cap)
        soft_limit = per_cap if soft_limit is None else min(soft_limit, per_cap)
    attempt = 0
    while True:
        token_estimate = estimate_tokens(messages, AI_MODEL)
        if hard_limit and token_estimate > hard_limit and attempt < 3:
            _shrink_bundle_for_tokens(payload["data"], max_bars=max(20, 60 - attempt * 15))
            messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
            attempt += 1
            continue
        if soft_limit and token_estimate > soft_limit and attempt < 3:
            _shrink_bundle_for_tokens(payload["data"], max_bars=max(30, 80 - attempt * 10))
            messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
            attempt += 1
            continue
        break
    if per_cap and token_estimate > per_cap:
        log(
            f"[AI] trade plan ({stage}): не удалось ужать запрос до {per_cap} токенов",
            Fore.YELLOW,
        )
        return None
    if not _ensure_token_budget(token_estimate, AI_MODEL, f"trade plan ({stage})"):
        return None
    _log_ai_request(AI_MODEL, token_estimate, f"trade plan ({stage})")
    attempt_count = 0
    max_attempts = max(1, TRADE_PLAN_MAX_ATTEMPTS)
    base_backoff = max(1.0, float(TRADE_PLAN_BACKOFF_SECONDS))
    last_error: Exception | None = None
    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            res = client.chat.completions.create(
                model=AI_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as exc:
            last_error = exc
            log(f"[ERROR] OpenAI trade plan attempt {attempt_count}: {exc}", Fore.RED)
            if attempt_count >= max_attempts:
                break
            delay = base_backoff * (attempt_count ** 2)
            time.sleep(delay + random.uniform(0, base_backoff))
            continue
        _register_ai_usage(AI_MODEL, getattr(res, "usage", None), f"trade plan ({stage})")
        content = res.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            log(f"[WARN] JSON decode (trade plan): {exc}", Fore.YELLOW)
            last_error = exc
            if attempt_count >= max_attempts:
                break
            delay = base_backoff * (attempt_count ** 2)
            time.sleep(delay + random.uniform(0, base_backoff))
            continue
    if last_error:
        log(
            f"[ERROR] OpenAI trade plan failed after {attempt_count} attempts: {last_error}",
            Fore.RED,
        )
    return None


def execute_symbol_decision(exchange, decision, positions_map, open_orders_cache, counts):
    if not decision:
        return 0, positions_map, open_orders_cache
    sym = decision.get("symbol")
    if not sym:
        return 0, positions_map, open_orders_cache
    action = (decision.get("action") or "skip").lower()
    counts[action] = counts.get(action, 0) + 1
    side = decision.get("side") or ""
    reason = decision.get("reason") or ""
    notional_pct = decision.get("notional_pct")
    log(f"{sym}: action={action} side={side} reason={reason}", Fore.LIGHTBLUE_EX)
    if notional_pct is not None:
        log(f"{sym}: notional_pct={notional_pct:.3f}", Fore.LIGHTBLACK_EX)
    current_position = positions_map.get(sym)
    open_orders_symbol = open_orders_cache.get(sym)
    if open_orders_symbol is None:
        try:
            open_orders_symbol = fetch_open_orders_for_symbol(exchange, sym)
        except Exception as exc:
            log(f"[WARN] fetch_open_orders {sym}: {exc}", Fore.YELLOW)
            open_orders_symbol = []
        open_orders_cache[sym] = open_orders_symbol

    extra_orders_raw = (
        decision.get("orders")
        or decision.get("adjustments")
        or decision.get("extra_orders")
        or []
    )
    if isinstance(extra_orders_raw, dict):
        extra_orders = [extra_orders_raw]
    elif isinstance(extra_orders_raw, list):
        extra_orders = list(extra_orders_raw)
    else:
        extra_orders = []

    def normalize_order_ids(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]
        result = []
        for item in items:
            if item in (None, ""):
                continue
            if isinstance(item, dict):
                oid = (
                    item.get("id")
                    or item.get("orderId")
                    or item.get("order_id")
                    or item.get("cancel")
                    or item.get("old")
                )
                if oid:
                    result.append(str(oid))
            else:
                result.append(str(item))
        return result

    cancel_candidates = normalize_order_ids(
        decision.get("cancel_orders")
        or decision.get("cancelOrders")
        or decision.get("cancel_order_ids")
    )
    replace_raw = (
        decision.get("replace_orders")
        or decision.get("replaceOrders")
        or decision.get("order_replacements")
    )
    replace_list = (
        replace_raw
        if isinstance(replace_raw, list)
        else ([replace_raw] if isinstance(replace_raw, dict) else [])
    )
    replacement_orders = []
    cancelled_ids = set()
    cancelled_success = []
    cancel_failures = []

    def try_cancel(order_id: str, source: str):
        oid = str(order_id)
        if not oid or oid in cancelled_ids:
            return
        success, err = cancel_order_by_id(exchange, sym, oid)
        if success:
            cancelled_ids.add(oid)
            cancelled_success.append((oid, source))
            log(f"?? —?'—?—?—?—?—? ——?——?—< {oid} {sym} (source={source})", Fore.LIGHTBLUE_EX)
        else:
            cancel_failures.append((oid, err))
            log(f"⚠️ Не удалось отменить ордер {oid} {sym}: {err}", Fore.YELLOW)

    for oid in cancel_candidates:
        try_cancel(oid, "cancel_orders")

    for entry in replace_list:
        if not isinstance(entry, dict):
            continue
        cancel_id = (
            entry.get("cancel")
            or entry.get("cancel_id")
            or entry.get("old")
            or entry.get("order_id")
        )
        if cancel_id:
            try_cancel(cancel_id, "replace_orders")
        new_order = entry.get("new") or entry.get("order") or entry.get("replacement")
        if new_order:
            replacement_orders.append(new_order)

    if cancel_failures:
        errs = "; ".join(f"{oid}: {err}" for oid, err in cancel_failures)
        send_tg(f"{sym}: не удалось отменить ордера: {errs}")

    if replacement_orders:
        extra_orders.extend(replacement_orders)

    if extra_orders:
        executed, actions_performed = execute_extra_orders(
            exchange,
            sym,
            extra_orders,
            current_position=current_position,
            open_orders=open_orders_symbol,
            max_limits_per_side=MAX_NON_REDUCE_LIMITS_PER_SIDE,
        )
        if executed:
            send_tg(f"{sym}: —?—?—?——? выполнил:\n- " + "\n- ".join(executed))
        if actions_performed:
            positions_map, _ = fetch_positions_snapshot(exchange, symbols_filter=[sym])
            current_position = positions_map.get(sym)
            try:
                open_orders_symbol = fetch_open_orders_for_symbol(exchange, sym)
            except Exception as exc:
                log(f"[WARN] fetch_open_orders {sym}: {exc}", Fore.YELLOW)
                open_orders_symbol = []
            open_orders_cache[sym] = open_orders_symbol

    return 1, positions_map, open_orders_cache


def load_environment():
    load_dotenv(".env")
    if os.path.exists(".env.local"):
        load_dotenv(".env.local", override=True)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return int(default)


def resolve_timezone(value: str):
    if not value:
        return None
    val = value.strip()
    if not val or val.lower() in ("local", "system"):
        return None
    if val.upper() == "UTC":
        return datetime.timezone.utc
    offset_match = re.fullmatch(r"(?:UTC)?([+-])(\d{1,2})(?::?(\d{2}))?", val, re.IGNORECASE)
    if not offset_match:
        offset_match = re.fullmatch(r"([+-])(\d{1,2})(?::?(\d{2}))?", val)
    if offset_match:
        sign = 1 if offset_match.group(1) == "+" else -1
        hours = int(offset_match.group(2))
        minutes = int(offset_match.group(3) or 0)
        delta = datetime.timedelta(hours=hours, minutes=minutes)
        delta *= sign
        name = f"UTC{offset_match.group(1)}{hours:02d}:{minutes:02d}"
        return datetime.timezone(delta, name=name)
    if ZoneInfo is not None:
        try:
            return ZoneInfo(val)
        except Exception:
            return None
    return None


def refresh_settings():
    load_environment()
    global PAIR_LIST, TIMEFRAME, LEVERAGE, RISK_PCT, SL_ATR, TP_ATR, TRAILING_ATR_MULT
    global DEFAULT_NEXT_RUN_MINUTES
    global MIN_NOTIONAL_USDT, AI_AFTER_NEEDS_BIAS, MAX_OPEN_POSITIONS
    global MIN_CONTEXT_30M, MIN_CONTEXT_4H, DEFAULT_CONTEXT_30M, DEFAULT_CONTEXT_4H
    global CONTEXT_STEP_30M, CONTEXT_STEP_4H
    global TG_TOKEN, TG_CHAT, TG_TOPIC_ID, TG_MIN_INTERVAL, TG_DUP_WINDOW, TG_RETRY_ATTEMPTS, TG_RETRY_BACKOFF
    global AI_MODEL, AI_KEY, AI_MODEL_PRIMARY, AI_MODEL_CHEAP, AI_MODEL_THRESHOLD, AI_TOKEN_BUDGET_CYCLE
    global NEWS_PROVIDER, NEWS_API_TOKEN, NEWS_ITEMS_LIMIT
    global POSITION_MODE, HEDGE_MODE, ORDER_MARGIN_UTILIZATION
    global LOG_TIMEZONE, LOG_TZINFO, _LOG_TZ_WARNING_EMITTED
    global PAIR_CANDIDATE_LIMIT, PAIR_PREFETCH_LIMIT
    global SUPPORT_CONTEXT_TIMEFRAMES, SUPPORT_CONTEXT_INDICATORS, SUPPORT_CONTEXT_LIMIT
    global LOW_CONFIDENCE_TIMEFRAMES, LOW_CONFIDENCE_INDICATORS, LOW_CONFIDENCE_SERIALIZE_LIMIT
    global NEEDS_MAX_TIMEFRAMES, NEEDS_MAX_INDICATORS, NEEDS_SERIALIZE_DEFAULT_LIMIT
    PAIR_LIST = os.getenv("PAIR_LIST", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,DOGE/USDT:USDT").split(",")
    TIMEFRAME = os.getenv("TIMEFRAME", "30m")
    LEVERAGE = int(os.getenv("LEVERAGE", 10))
    RISK_PCT = float(os.getenv("RISK_PCT", os.getenv("RISK_EQUITY_PCT", 0.015)))
    SL_ATR = float(os.getenv("SL_ATR", os.getenv("SL_ATR_MULT", 0.8)))
    TP_ATR = float(os.getenv("TP_ATR", os.getenv("TP_ATR_MULT", 1.6)))
    TRAILING_ATR_MULT = float(os.getenv("TRAILING_ATR_MULT", os.getenv("TRAILING_ATR", 0.0)))
    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", 5.0))
    AI_AFTER_NEEDS_BIAS = int(os.getenv("AI_AFTER_NEEDS_BIAS", 1))
    MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 0)
    env_default_next = os.getenv("DEFAULT_NEXT_RUN_MINUTES")
    if env_default_next:
        try:
            default_val = float(env_default_next)
            if math.isfinite(default_val) and default_val > 0:
                DEFAULT_NEXT_RUN_MINUTES = default_val
        except (TypeError, ValueError):
            pass
    DEFAULT_NEXT_RUN_MINUTES = max(1.0, DEFAULT_NEXT_RUN_MINUTES)

    TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")
    TG_TOPIC_ID = safe_int(os.getenv("TELEGRAM_TOPIC_ID"))
    try:
        TG_MIN_INTERVAL = float(os.getenv("TELEGRAM_MIN_INTERVAL", "1.5"))
    except (TypeError, ValueError):
        TG_MIN_INTERVAL = 1.5
    TG_MIN_INTERVAL = max(0.1, TG_MIN_INTERVAL)
    try:
        TG_DUP_WINDOW = float(os.getenv("TELEGRAM_DUP_WINDOW", "10"))
    except (TypeError, ValueError):
        TG_DUP_WINDOW = 10.0
    TG_DUP_WINDOW = max(0.0, TG_DUP_WINDOW)
    try:
        TG_RETRY_ATTEMPTS = int(os.getenv("TELEGRAM_RETRY_ATTEMPTS", "3"))
    except (TypeError, ValueError):
        TG_RETRY_ATTEMPTS = 3
    TG_RETRY_ATTEMPTS = max(1, TG_RETRY_ATTEMPTS)
    try:
        TG_RETRY_BACKOFF = float(os.getenv("TELEGRAM_RETRY_BACKOFF", "1.5"))
    except (TypeError, ValueError):
        TG_RETRY_BACKOFF = 1.5
    TG_RETRY_BACKOFF = max(0.5, TG_RETRY_BACKOFF)
    AI_MODEL_PRIMARY = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    AI_MODEL_CHEAP = os.getenv("OPENAI_MODEL_CHEAP", os.getenv("OPENAI_MODEL_BACKUP", "gpt-4o-mini"))
    try:
        AI_MODEL_THRESHOLD = int(os.getenv("OPENAI_MODEL_CHEAP_THRESHOLD", "5"))
    except (TypeError, ValueError):
        AI_MODEL_THRESHOLD = 5
    AI_MODEL_THRESHOLD = max(0, AI_MODEL_THRESHOLD)
    AI_MODEL = AI_MODEL_PRIMARY or AI_MODEL_CHEAP or "gpt-4.1-mini"
    try:
        AI_TOKEN_BUDGET_CYCLE = int(os.getenv("OPENAI_TOKEN_BUDGET_PER_CYCLE", "100000"))
    except (TypeError, ValueError):
        AI_TOKEN_BUDGET_CYCLE = 170_000
    AI_TOKEN_BUDGET_CYCLE = max(1000, AI_TOKEN_BUDGET_CYCLE)
    AI_KEY = os.getenv("OPENAI_API_KEY")

    global TOKEN_LIMIT, TOKEN_SOFT_LIMIT
    TOKEN_LIMIT = env_int("OPENAI_REQUEST_TOKEN_LIMIT", 12000)
    TOKEN_SOFT_LIMIT = env_int("OPENAI_REQUEST_TOKEN_SOFT_LIMIT", 8000)

    MIN_CONTEXT_30M = env_int("AI_CONTEXT_30M_MIN", 16)
    MIN_CONTEXT_4H = env_int("AI_CONTEXT_4H_MIN", 12)
    DEFAULT_CONTEXT_30M = max(MIN_CONTEXT_30M, env_int("AI_CONTEXT_30M", 40))
    DEFAULT_CONTEXT_4H = max(MIN_CONTEXT_4H, env_int("AI_CONTEXT_4H", 40))
    CONTEXT_STEP_30M = max(1, env_int("AI_CONTEXT_30M_STEP", 4))
    CONTEXT_STEP_4H = max(1, env_int("AI_CONTEXT_4H_STEP", 2))

    support_tf_env = os.getenv("AI_SUPPORT_TIMEFRAMES")
    if support_tf_env:
        parsed_support_tf = None
        try:
            parsed_support_tf = json.loads(support_tf_env)
        except json.JSONDecodeError:
            parsed_support_tf = None
        if isinstance(parsed_support_tf, (list, tuple, set)):
            SUPPORT_CONTEXT_TIMEFRAMES = [
                str(item).strip() for item in parsed_support_tf if str(item).strip()
            ]
        elif isinstance(parsed_support_tf, str):
            SUPPORT_CONTEXT_TIMEFRAMES = [parsed_support_tf.strip()] if parsed_support_tf.strip() else []
        else:
            SUPPORT_CONTEXT_TIMEFRAMES = [item.strip() for item in support_tf_env.split(",") if item.strip()]
    else:
        SUPPORT_CONTEXT_TIMEFRAMES = ["15m", "1h", "4h"]
    if not SUPPORT_CONTEXT_TIMEFRAMES:
        SUPPORT_CONTEXT_TIMEFRAMES = ["15m", "1h", "4h"]

    support_ind_env = os.getenv("AI_SUPPORT_INDICATORS")
    if support_ind_env:
        parsed_support_ind = None
        try:
            parsed_support_ind = json.loads(support_ind_env)
        except json.JSONDecodeError:
            parsed_support_ind = None
        if isinstance(parsed_support_ind, (list, tuple, set)):
            SUPPORT_CONTEXT_INDICATORS = [
                str(item).strip() for item in parsed_support_ind if str(item).strip()
            ]
        elif isinstance(parsed_support_ind, str):
            SUPPORT_CONTEXT_INDICATORS = [parsed_support_ind.strip()] if parsed_support_ind.strip() else []
        else:
            SUPPORT_CONTEXT_INDICATORS = [item.strip() for item in support_ind_env.split(",") if item.strip()]
    else:
        SUPPORT_CONTEXT_INDICATORS = ["ema20", "ema50", "ema100", "rsi14", "atr14"]
    if not SUPPORT_CONTEXT_INDICATORS:
        SUPPORT_CONTEXT_INDICATORS = ["ema20", "ema50", "ema100", "rsi14", "atr14"]

    try:
        SUPPORT_CONTEXT_LIMIT = int(os.getenv("AI_SUPPORT_CONTEXT_LIMIT", "48"))
    except (TypeError, ValueError):
        SUPPORT_CONTEXT_LIMIT = 48
    SUPPORT_CONTEXT_LIMIT = max(12, SUPPORT_CONTEXT_LIMIT)

    low_conf_tf_env = os.getenv("BYBITBOT_LOW_CONF_TIMEFRAMES")
    if low_conf_tf_env:
        parsed_low_conf_tf = None
        try:
            parsed_low_conf_tf = json.loads(low_conf_tf_env)
        except json.JSONDecodeError:
            parsed_low_conf_tf = None
        if isinstance(parsed_low_conf_tf, (list, tuple, set)):
            LOW_CONFIDENCE_TIMEFRAMES = [
                str(item).strip() for item in parsed_low_conf_tf if str(item).strip()
            ]
        elif isinstance(parsed_low_conf_tf, str):
            LOW_CONFIDENCE_TIMEFRAMES = [parsed_low_conf_tf.strip()] if parsed_low_conf_tf.strip() else []
        else:
            LOW_CONFIDENCE_TIMEFRAMES = [item.strip() for item in low_conf_tf_env.split(",") if item.strip()]
    else:
        LOW_CONFIDENCE_TIMEFRAMES = list(SUPPORT_CONTEXT_TIMEFRAMES)
    if not LOW_CONFIDENCE_TIMEFRAMES:
        LOW_CONFIDENCE_TIMEFRAMES = list(SUPPORT_CONTEXT_TIMEFRAMES)

    low_conf_ind_env = os.getenv("BYBITBOT_LOW_CONF_INDICATORS")
    if low_conf_ind_env:
        parsed_low_conf_ind = None
        try:
            parsed_low_conf_ind = json.loads(low_conf_ind_env)
        except json.JSONDecodeError:
            parsed_low_conf_ind = None
        if isinstance(parsed_low_conf_ind, (list, tuple, set)):
            LOW_CONFIDENCE_INDICATORS = [
                str(item).strip() for item in parsed_low_conf_ind if str(item).strip()
            ]
        elif isinstance(parsed_low_conf_ind, str):
            LOW_CONFIDENCE_INDICATORS = [parsed_low_conf_ind.strip()] if parsed_low_conf_ind.strip() else []
        else:
            LOW_CONFIDENCE_INDICATORS = [item.strip() for item in low_conf_ind_env.split(",") if item.strip()]
    else:
        LOW_CONFIDENCE_INDICATORS = list(SUPPORT_CONTEXT_INDICATORS)
    if not LOW_CONFIDENCE_INDICATORS:
        LOW_CONFIDENCE_INDICATORS = list(SUPPORT_CONTEXT_INDICATORS)

    try:
        LOW_CONFIDENCE_SERIALIZE_LIMIT = int(os.getenv("BYBITBOT_LOW_CONF_LIMIT", str(min(40, SUPPORT_CONTEXT_LIMIT))))
    except (TypeError, ValueError):
        LOW_CONFIDENCE_SERIALIZE_LIMIT = min(40, SUPPORT_CONTEXT_LIMIT)
    LOW_CONFIDENCE_SERIALIZE_LIMIT = max(10, min(LOW_CONFIDENCE_SERIALIZE_LIMIT, SUPPORT_CONTEXT_LIMIT))

    try:
        NEEDS_MAX_TIMEFRAMES = int(os.getenv("BYBITBOT_NEEDS_MAX_TF", "2"))
    except (TypeError, ValueError):
        NEEDS_MAX_TIMEFRAMES = 2
    NEEDS_MAX_TIMEFRAMES = max(1, NEEDS_MAX_TIMEFRAMES)

    try:
        NEEDS_MAX_INDICATORS = int(os.getenv("BYBITBOT_NEEDS_MAX_INDICATORS", "6"))
    except (TypeError, ValueError):
        NEEDS_MAX_INDICATORS = 6
    NEEDS_MAX_INDICATORS = max(1, NEEDS_MAX_INDICATORS)

    try:
        NEEDS_SERIALIZE_DEFAULT_LIMIT = int(os.getenv("BYBITBOT_NEEDS_BARS_LIMIT", "10"))
    except (TypeError, ValueError):
        NEEDS_SERIALIZE_DEFAULT_LIMIT = 10
    NEEDS_SERIALIZE_DEFAULT_LIMIT = max(5, NEEDS_SERIALIZE_DEFAULT_LIMIT)

    PAIR_CANDIDATE_LIMIT = env_int("PAIR_CANDIDATE_LIMIT", PAIR_CANDIDATE_LIMIT)
    PAIR_PREFETCH_LIMIT = env_int("PAIR_PREFETCH_LIMIT", PAIR_PREFETCH_LIMIT)

    NEWS_PROVIDER = (os.getenv("CRYPTO_NEWS_PROVIDER") or "cryptocompare").strip().lower()
    NEWS_API_TOKEN = os.getenv("CRYPTO_NEWS_TOKEN") or os.getenv("NEWS_API_TOKEN")
    NEWS_ITEMS_LIMIT = env_int("CRYPTO_NEWS_LIMIT", 5)
    POSITION_MODE = (os.getenv("BYBIT_POSITION_MODE") or "oneway").strip().lower()
    HEDGE_MODE = POSITION_MODE in ("hedge", "hedged", "dual", "dual_side", "dual-side")
    try:
        ORDER_MARGIN_UTILIZATION = float(os.getenv("ORDER_MARGIN_UTILIZATION", 0.95))
    except (TypeError, ValueError):
        ORDER_MARGIN_UTILIZATION = 0.95
    ORDER_MARGIN_UTILIZATION = max(0.1, min(ORDER_MARGIN_UTILIZATION, 1.0))
    MAX_NON_REDUCE_LIMITS_PER_SIDE = env_int("BYBITBOT_MAX_NON_REDUCE_LIMITS_PER_SIDE", 1)
    NON_REDUCE_PRICE_DECIMALS = env_int("BYBITBOT_NON_REDUCE_PRICE_DECIMALS", 4)
    TRADE_PLAN_MAX_ATTEMPTS = env_int("BYBITBOT_TRADE_PLAN_MAX_ATTEMPTS", 5)
    try:
        TRADE_PLAN_BACKOFF_SECONDS = float(os.getenv("BYBITBOT_TRADE_PLAN_BACKOFF", "5"))
    except (TypeError, ValueError):
        TRADE_PLAN_BACKOFF_SECONDS = 5.0
    TRADE_PLAN_BACKOFF_SECONDS = max(1.0, TRADE_PLAN_BACKOFF_SECONDS)
    try:
        AI_CONFIDENCE_THRESHOLD = float(os.getenv("BYBITBOT_CONFIDENCE_THRESHOLD", "0.55"))
    except (TypeError, ValueError):
        AI_CONFIDENCE_THRESHOLD = 0.55
    low_conf_needs_env = os.getenv("BYBITBOT_CONFIDENCE_NEEDS")
    if low_conf_needs_env:
        try:
            parsed_needs = json.loads(low_conf_needs_env)
            if isinstance(parsed_needs, (list, tuple)):
                LOW_CONFIDENCE_NEEDS = [str(item).strip() for item in parsed_needs if str(item).strip()]
            else:
                raise ValueError
        except Exception:
            LOW_CONFIDENCE_NEEDS = [item.strip() for item in low_conf_needs_env.split(',') if item.strip()]
    else:
        LOW_CONFIDENCE_NEEDS = ["funding", "open_interest", "news"]
    LOG_TIMEZONE = (os.getenv("LOG_TIMEZONE") or "").strip()
    parsed_tz = resolve_timezone(LOG_TIMEZONE)
    if LOG_TIMEZONE and parsed_tz is None:
        if not _LOG_TZ_WARNING_EMITTED:
            print(f"[WARN] LOG_TIMEZONE '{LOG_TIMEZONE}' не распознан, используется системное время.")
            _LOG_TZ_WARNING_EMITTED = True
        LOG_TZINFO = None
    else:
        LOG_TZINFO = parsed_tz
        _LOG_TZ_WARNING_EMITTED = False

RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
    "https://decrypt.co/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://u.today/rss",
]

NEWS_PROVIDER = "cryptocompare"
NEWS_API_TOKEN = ""
NEWS_ITEMS_LIMIT = 5



refresh_settings()

# --- Конфигурация ---
MIN_CONTEXT_30M = env_int("AI_CONTEXT_30M_MIN", 16)
MIN_CONTEXT_4H = env_int("AI_CONTEXT_4H_MIN", 12)
DEFAULT_CONTEXT_30M = max(MIN_CONTEXT_30M, env_int("AI_CONTEXT_30M", 40))
DEFAULT_CONTEXT_4H = max(MIN_CONTEXT_4H, env_int("AI_CONTEXT_4H", 40))
CONTEXT_STEP_30M = max(1, env_int("AI_CONTEXT_30M_STEP", 4))
CONTEXT_STEP_4H = max(1, env_int("AI_CONTEXT_4H_STEP", 2))
AI_LOG_FILE = "ai_decisions.log"
AI_ARCHIVE_FILE = "ai_decisions_archive.log"
AI_REQUESTS_LOG = "ai_requests.log"
if "ORDER_MARGIN_UTILIZATION" not in globals():
    ORDER_MARGIN_UTILIZATION = 0.95
ORDER_MARGIN_UTILIZATION = max(0.1, min(ORDER_MARGIN_UTILIZATION, 1.0))
if "MAX_NON_REDUCE_LIMITS_PER_SIDE" not in globals():
    MAX_NON_REDUCE_LIMITS_PER_SIDE = 1
if "NON_REDUCE_PRICE_DECIMALS" not in globals():
    NON_REDUCE_PRICE_DECIMALS = 4
if "TRADE_PLAN_MAX_ATTEMPTS" not in globals():
    TRADE_PLAN_MAX_ATTEMPTS = 5
if "TRADE_PLAN_BACKOFF_SECONDS" not in globals():
    TRADE_PLAN_BACKOFF_SECONDS = 5.0
if "AI_CONFIDENCE_THRESHOLD" not in globals():
    AI_CONFIDENCE_THRESHOLD = 0.55
if "LOW_CONFIDENCE_NEEDS" not in globals():
    LOW_CONFIDENCE_NEEDS = ["funding", "open_interest", "news"]
if "SUPPORT_CONTEXT_TIMEFRAMES" not in globals():
    SUPPORT_CONTEXT_TIMEFRAMES = ["15m", "1h", "4h"]
if "SUPPORT_CONTEXT_INDICATORS" not in globals():
    SUPPORT_CONTEXT_INDICATORS = ["ema20", "ema50", "ema100", "rsi14", "atr14"]
if "SUPPORT_CONTEXT_LIMIT" not in globals():
    SUPPORT_CONTEXT_LIMIT = 48
if "LOW_CONFIDENCE_TIMEFRAMES" not in globals():
    LOW_CONFIDENCE_TIMEFRAMES = list(SUPPORT_CONTEXT_TIMEFRAMES)
if "LOW_CONFIDENCE_INDICATORS" not in globals():
    LOW_CONFIDENCE_INDICATORS = list(SUPPORT_CONTEXT_INDICATORS)
if "LOW_CONFIDENCE_SERIALIZE_LIMIT" not in globals():
    LOW_CONFIDENCE_SERIALIZE_LIMIT = min(40, SUPPORT_CONTEXT_LIMIT)
if "NEEDS_SERIALIZE_DEFAULT_LIMIT" not in globals():
    NEEDS_SERIALIZE_DEFAULT_LIMIT = 10
if "NEEDS_MAX_TIMEFRAMES" not in globals():
    NEEDS_MAX_TIMEFRAMES = 2
if "NEEDS_MAX_INDICATORS" not in globals():
    NEEDS_MAX_INDICATORS = 6
if "SUMMARY_TIMEFRAME_SHORTLIST" not in globals():
    SUMMARY_TIMEFRAME_SHORTLIST = ["30m", "4h"]
if "SUMMARY_INDICATOR_SHORTLIST" not in globals():
    SUMMARY_INDICATOR_SHORTLIST = ["ema20", "ema50", "vol", "rsi14", "macd", "atr14"]
if "NEEDS_LONG_BARS_LIMIT" not in globals():
    NEEDS_LONG_BARS_LIMIT = 10
if "TG_MIN_INTERVAL" not in globals():
    TG_MIN_INTERVAL = 1.5
if "TG_DUP_WINDOW" not in globals():
    TG_DUP_WINDOW = 10.0
if "TG_RETRY_ATTEMPTS" not in globals():
    TG_RETRY_ATTEMPTS = 3
if "TG_RETRY_BACKOFF" not in globals():
    TG_RETRY_BACKOFF = 1.5

# --- AI token tracking ---
AI_TOKEN_BUDGET_CYCLE = 170_000
AI_TOKEN_USAGE_TOTAL = 0
AI_TOKEN_USAGE_BY_MODEL: dict[str, dict[str, int]] = {}
AI_SECONDARY_BUDGET_START = 80_000
AI_PER_REQUEST_TOKEN_CAP = 50_000
AI_HARD_STOP_BUDGET = 200_000
MAX_SYMBOLS_PER_CYCLE = 15
UNIVERSE_CACHE_DEFAULT = {
    "pairs": [],
    "global_timeframes": [],
    "global_indicators": [],
    "notes": None,
}


def load_universe_cache() -> dict:
    try:
        raw = UNIVERSE_CACHE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        log(f"[WARN] Failed to read universe_cache.json: {exc}", Fore.YELLOW)
    return dict(UNIVERSE_CACHE_DEFAULT)


def save_universe_cache(payload: dict) -> None:
    try:
        UNIVERSE_CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"[WARN] Failed to write universe_cache.json: {exc}", Fore.YELLOW)

# --- AI model selection state ---
AI_MODEL_PRIMARY = ""
AI_MODEL_CHEAP = ""
AI_MODEL_THRESHOLD = 5
# --- Вспомогательные функции ---
def _current_log_time():
    base = datetime.datetime.now(datetime.timezone.utc)
    if LOG_TZINFO is not None:
        return base.astimezone(LOG_TZINFO)
    return base.astimezone()


def _current_local_tz():
    try:
        return LOG_TZINFO or datetime.datetime.now().astimezone().tzinfo
    except Exception:
        return None


def _write_runtime_status(next_delay_minutes: Optional[float], next_run_dt: Optional[datetime.datetime], status: str) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    if next_delay_minutes is not None:
        try:
            payload["next_run_minutes"] = round(float(next_delay_minutes), 2)
        except (TypeError, ValueError):
            pass
    if next_run_dt is not None:
        try:
            payload["next_run_utc"] = next_run_dt.astimezone(datetime.timezone.utc).isoformat()
        except Exception:
            payload["next_run_utc"] = next_run_dt.isoformat()
        try:
            local_tz = _current_local_tz()
            payload["next_run_local"] = next_run_dt.astimezone(local_tz).isoformat() if local_tz else next_run_dt.isoformat()
        except Exception:
            pass
    try:
        RUNTIME_STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _format_tz_suffix(dt: datetime.datetime) -> str:
    parts = []
    tz_name = (dt.tzname() or "").strip()
    if tz_name and tz_name.upper() != "UTC":
        parts.append(tz_name)
    offset = dt.utcoffset()
    if offset is not None:
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        hours, minutes = divmod(total_minutes, 60)
        offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"
        if offset_str not in parts:
            parts.append(offset_str)
    return " ".join(parts).strip()


def log(msg: str, color=Fore.WHITE):
    now = _current_log_time()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    tz_suffix = _format_tz_suffix(now)
    if tz_suffix:
        stamp = f"{stamp} {tz_suffix}"
    print(color + f"[{stamp}] {msg}" + Style.RESET_ALL)

_TG_LAST_SEND_TS = 0.0
_TG_LAST_MESSAGE: str | None = None
_TG_LAST_MESSAGE_TS = 0.0


def send_tg(msg: str | Sequence[str], **extra):
    if not TG_TOKEN or not TG_CHAT:
        return None
    global _TG_LAST_SEND_TS, _TG_LAST_MESSAGE, _TG_LAST_MESSAGE_TS
    if isinstance(msg, (list, tuple, set)):
        message_text = "\n".join(str(part) for part in msg if part)
    else:
        message_text = str(msg)
    if not message_text:
        return None
    now_ts = time.time()
    if (
        _TG_LAST_MESSAGE == message_text
        and (now_ts - _TG_LAST_MESSAGE_TS) < TG_DUP_WINDOW
    ):
        return None
    delay_needed = TG_MIN_INTERVAL - (now_ts - _TG_LAST_SEND_TS)
    if delay_needed > 0:
        time.sleep(delay_needed)
        now_ts = time.time()
    base_payload = {"chat_id": TG_CHAT, "text": message_text}
    extra_payload = dict(extra) if extra else {}
    thread_override = extra_payload.pop("thread_id", None)
    if "message_thread_id" in extra_payload:
        base_payload.update(extra_payload)
    else:
        thread_candidate = thread_override if thread_override is not None else TG_TOPIC_ID
        thread_id_int = safe_int(thread_candidate) if thread_candidate is not None else None
        if thread_id_int is not None:
            base_payload["message_thread_id"] = thread_id_int
        base_payload.update(extra_payload)
    attempt = 0
    last_error = None
    while attempt < TG_RETRY_ATTEMPTS:
        attempt += 1
        payload = dict(base_payload)
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json=payload,
                timeout=5,
            )
        except Exception as exc:
            last_error = exc
            log(f"?? ?????? Telegram ({attempt}/{TG_RETRY_ATTEMPTS}): {exc}", Fore.YELLOW)
        else:
            try:
                data = response.json()
            except Exception:
                last_error = f"{response.status_code} {response.text}"
                log(f"?? Telegram: ?????????? ??? {last_error}", Fore.YELLOW)
            else:
                if isinstance(data, dict) and data.get("ok"):
                    result = data.get("result") or {}
                    message_id = result.get("message_id")
                    _TG_LAST_SEND_TS = time.time()
                    _TG_LAST_MESSAGE = message_text
                    _TG_LAST_MESSAGE_TS = _TG_LAST_SEND_TS
                    return message_id
                last_error = data
                log(f"?? Telegram API ???? ????: {data}", Fore.YELLOW)
                if isinstance(data, dict) and data.get("error_code") == 429:
                    retry_after = data.get("parameters", {}).get("retry_after")
                    sleep_for = float(retry_after or (TG_RETRY_BACKOFF * attempt))
                    time.sleep(max(TG_MIN_INTERVAL, sleep_for))
                    continue
                if (
                    isinstance(data, dict)
                    and data.get("error_code") == 400
                    and "thread not found" in (data.get("description") or "").lower()
                ):
                    base_payload.pop("message_thread_id", None)
                    log("[WARN] Telegram topic not found, retrying without thread.", Fore.YELLOW)
                    time.sleep(TG_RETRY_BACKOFF * attempt)
                    continue
        time.sleep(TG_RETRY_BACKOFF * attempt)
    log(f"?? Telegram send failed after {TG_RETRY_ATTEMPTS} attempts: {last_error}", Fore.RED)
    return None

def _load_changelog_state() -> dict:
    try:
        raw = CHANGELOG_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log(f"⚠️ Не удалось прочитать состояние changelog: {exc}", Fore.YELLOW)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log("⚠️ Повреждён файл changelog_state.json, начинаем заново.", Fore.YELLOW)
        return {}
    return data if isinstance(data, dict) else {}


def _save_changelog_state(state: dict) -> None:
    try:
        CHANGELOG_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"⚠️ Не удалось сохранить состояние changelog: {exc}", Fore.YELLOW)


def _build_tg_message_link(chat_id: str, message_id: int | None) -> Optional[str]:
    if not chat_id or not message_id:
        return None
    chat_id = str(chat_id).strip()
    if not chat_id:
        return None
    if chat_id.startswith("@"):
        username = chat_id.lstrip("@")
        if username:
            return f"https://t.me/{username}/{message_id}"
        return None
    try:
        numeric_id = int(chat_id)
    except ValueError:
        return None
    if numeric_id >= 0:
        return None
    channel_id = abs(numeric_id)
    if channel_id > 1000000000000:
        channel_id -= 1000000000000
    return f"https://t.me/c/{channel_id}/{message_id}"


def _current_changelog_signature() -> dict:
    digest = hashlib.sha256((BOT_CHANGELOG or "").encode("utf-8")).hexdigest()
    return {
        "version": BOT_VERSION,
        "commit": _LAST_COMMIT_HASH or "",
        "digest": digest,
    }


def ensure_changelog_announcement() -> dict:
    signature = _current_changelog_signature()
    state = _load_changelog_state()
    if (
        state.get("version") == signature["version"]
        and state.get("commit") == signature["commit"]
        and state.get("digest") == signature["digest"]
        and state.get("message_id")
    ):
        return state

    lines = [f"ℹ️ Версия {BOT_VERSION}"]
    if BOT_CHANGELOG:
        lines.append(BOT_CHANGELOG)
    message_text = "\n".join(lines)
    message_id = send_tg(message_text, disable_web_page_preview=True)
    if not message_id:
        return state
    link = _build_tg_message_link(TG_CHAT, message_id)
    new_state = {
        **signature,
        "message_id": message_id,
        "link": link,
        "sent_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _save_changelog_state(new_state)
    return new_state


def _restart_with_latest_code(reason: str) -> None:
    log(reason, Fore.LIGHTBLUE_EX)
    send_tg(reason)
    _write_runtime_status(None, None, "restarting")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    python_exec = sys.executable or "python"
    args = [python_exec, *sys.argv]
    try:
        os.execv(python_exec, args)
    except Exception as exc:
        err_msg = f"❌ Не удалось перезапустить процесс автоматически: {exc}"
        log(err_msg, Fore.RED)
        send_tg(err_msg)
        raise


def save_json_line(path, data):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"⚠️ Ошибка записи в {path}: {e}", Fore.YELLOW)

def extract_position_amount(position) -> float:
    candidates = [
        position.get("contracts"),
        position.get("contractSize"),
        position.get("positionAmt"),
        position.get("positionAmount"),
        position.get("size"),
        position.get("amount"),
    ]
    info = position.get("info") or {}
    candidates.extend([
        info.get("size"),
        info.get("position_q"),
        info.get("positionAmt"),
        info.get("position_value"),
    ])
    for val in candidates:
        if val in (None, "", 0):
            continue
        try:
            amount = float(val)
            if abs(amount) > 0:
                return amount
        except (TypeError, ValueError):
            continue
    return 0.0

def simplify_position(position):
    amount = extract_position_amount(position)
    if amount == 0:
        return None
    info = position.get("info") or {}
    side = position.get("side")
    if not side and amount != 0:
        side = "long" if amount > 0 else "short"
    entry_price = position.get("entryPrice") or position.get("average") or info.get("avgPrice")
    leverage = position.get("leverage") or info.get("leverage")
    unrealized = position.get("unrealizedPnl") or info.get("unrealisedPnl")
    liq_price = position.get("liquidationPrice") or info.get("liqPrice")
    return {
        "side": side,
        "amount": amount,
        "entryPrice": entry_price,
        "leverage": leverage,
        "unrealizedPnl": unrealized,
        "liquidationPrice": liq_price,
        "raw": info,
    }

def fetch_positions_snapshot(exchange, symbols_filter=None):
    try:
        positions = exchange.fetch_positions()
    except Exception as e:
        log(f"⚠️ Не удалось получить список позиций: {e}", Fore.YELLOW)
        return {}, None
    count = 0
    simplified = {}
    symbols_filter = set(symbols_filter) if symbols_filter else None
    for pos in positions or []:
        symbol = pos.get("symbol")
        if symbols_filter and symbol not in symbols_filter:
            continue
        simp = simplify_position(pos)
        if simp:
            simplified[symbol] = simp
            count += 1
    return simplified, count


def _compact_positions_snapshot(positions_map: dict[str, Any] | None) -> dict[str, Any]:
    """Strip heavy fields from positions before sending to the trade plan model."""
    snapshot: dict[str, Any] = {}
    if not isinstance(positions_map, dict):
        return snapshot
    for symbol, payload in positions_map.items():
        if not isinstance(payload, dict):
            continue
        filtered = {k: v for k, v in payload.items() if k != "raw"}
        if filtered:
            snapshot[symbol] = filtered
    return snapshot


def _compact_orders_snapshot(orders_map: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Return a lean view of pending orders grouped by symbol for trade planning."""
    snapshot: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(orders_map, dict):
        return snapshot
    keep_keys = {"id", "type", "side", "price", "stopPrice", "amount", "remaining", "reduceOnly", "status", "timestamp"}
    for symbol, bucket in orders_map.items():
        if not bucket:
            continue
        simplified: list[dict[str, Any]] = []
        for order in bucket:
            if not isinstance(order, dict):
                continue
            simplified.append({key: order.get(key) for key in keep_keys})
        if simplified:
            snapshot[symbol] = simplified
    return snapshot


def simplify_order(order):
    info = order.get("info") or {}

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return {
        "id": order.get("id") or info.get("orderId"),
        "type": order.get("type") or info.get("orderType"),
        "side": order.get("side"),
        "price": to_float(order.get("price") or info.get("price")),
        "stopPrice": to_float(order.get("stopPrice") or info.get("triggerPrice")),
        "amount": to_float(order.get("amount") or info.get("qty")),
        "filled": to_float(order.get("filled") or info.get("cumExecQty")),
        "remaining": to_float(order.get("remaining") or info.get("leavesQty")),
        "reduceOnly": order.get("reduceOnly") or info.get("reduceOnly"),
        "status": order.get("status") or info.get("orderStatus"),
        "timestamp": to_iso_utc(order.get("timestamp") or order.get("datetime") or info.get("createdTime")),
    }


def fetch_open_orders_for_symbol(exchange, symbol, limit: int | None = 50):
    has_attr = getattr(exchange, "has", {})
    if isinstance(has_attr, dict) and not has_attr.get("fetchOpenOrders", False):
        return []
    resolved_symbol = _resolve_symbol_alias(symbol) or symbol
    try:
        raw_orders = exchange.fetch_open_orders(resolved_symbol)
    except Exception as e:
        log(f"⚠️ Не удалось получить открытые ордера для {symbol}: {e}", Fore.YELLOW)
        return []
    simplified = []
    for order in raw_orders:
        simplified.append(simplify_order(order))
        if limit and len(simplified) >= limit:
            break
    return simplified


def fetch_all_open_orders_grouped(exchange, limit: int | None = None) -> dict[str, list]:
    has_attr = getattr(exchange, "has", {})
    if isinstance(has_attr, dict) and not has_attr.get("fetchOpenOrders", False):
        return {}
    try:
        raw_orders = exchange.fetch_open_orders()
    except Exception as e:
        log(f"[WARN] Failed to fetch global open orders: {e}", Fore.YELLOW)
        return {}
    grouped: dict[str, list] = {}
    for order in raw_orders or []:
        symbol = order.get("symbol")
        if not symbol:
            continue
        canonical_symbol = _resolve_symbol_alias(symbol) or symbol
        simplified = simplify_order(order)
        simplified["symbol"] = canonical_symbol
        bucket = grouped.setdefault(canonical_symbol, [])
        bucket.append(simplified)
        if limit and len(bucket) >= limit:
            continue
    return grouped


def cancel_order_by_id(exchange, symbol, order_id: str):
    resolved_symbol = _resolve_symbol_alias(symbol) or symbol
    try:
        exchange.cancel_order(order_id, resolved_symbol)
        return True, None
    except Exception as e:
        return False, str(e)


def _cancel_orders_batch(exchange, symbol, order_ids):
    cancelled = []
    failures = []
    for oid in order_ids:
        if not oid:
            continue
        success, err = cancel_order_by_id(exchange, symbol, oid)
        if success:
            cancelled.append(oid)
        else:
            failures.append((oid, err))
    return cancelled, failures


def _cleanup_excess_non_reduce_limits(exchange, symbol, open_orders, position_side, max_per_side=1):
    if not open_orders:
        return open_orders or []
    position_side = (position_side or "").lower()
    to_cancel: list[str] = []
    limit_groups: dict[tuple[str, float], list[dict]] = {}
    for order in open_orders or []:
        if not isinstance(order, dict):
            continue
        if _is_truthy_flag(order.get("reduceOnly")):
            continue
        order_type = (order.get("type") or "").lower()
        if order_type != "limit":
            continue
        price = safe_float(order.get("price"))
        if price is None or not math.isfinite(price) or price <= 0:
            continue
        side = (order.get("side") or "").lower()
        same_direction = (
            position_side in ("long", "buy") and side == "buy"
        ) or (
            position_side in ("short", "sell") and side == "sell"
        )
        oid = order.get("id")
        if same_direction and oid:
            to_cancel.append(str(oid))
            continue
        price_key = round(price, NON_REDUCE_PRICE_DECIMALS)
        key = (side, price_key)
        limit_groups.setdefault(key, []).append(order)
    for key, bucket in limit_groups.items():
        keep_count = max_per_side if max_per_side > 0 else 0
        if len(bucket) <= keep_count:
            continue
        sorted_bucket = sorted(
            bucket,
            key=lambda o: safe_float(o.get("timestamp") or 0) or 0,
        )
        for order in sorted_bucket[keep_count:]:
            oid = order.get("id")
            if oid:
                to_cancel.append(str(oid))
    if not to_cancel:
        return open_orders or []
    cancelled, failures = _cancel_orders_batch(exchange, symbol, to_cancel)
    if cancelled:
        summary = ", ".join(cancelled)
        log(f"[WARN] Cancelled excess non-reduce limits for {symbol}: {summary}", Fore.YELLOW)
        send_tg(f"[WARN] {symbol}: cancelled non-reduce limits: {summary}")
    if failures:
        details = "; ".join(f"{oid}: {err}" for oid, err in failures)
        log(f"[WARN] Failed to cancel some non-reduce limits for {symbol}: {details}", Fore.YELLOW)
    try:
        return fetch_open_orders_for_symbol(exchange, symbol)
    except Exception as exc:
        log(f"[WARN] fetch_open_orders {symbol}: {exc}", Fore.YELLOW)
        return open_orders or []


def get_position_idx(side: str | None) -> int | None:
    if HEDGE_MODE:
        if (side or "").lower() == "buy":
            return 1  # long position
        if (side or "").lower() == "sell":
            return 2  # short position
        return None
    # One-way mode
    return 0

def to_iso_utc(ts_value):
    if ts_value is None or ts_value == "":
        return None
    try:
        if isinstance(ts_value, str):
            ts_value = float(ts_value)
        if ts_value > 1e12:
            ts_value /= 1000.0
        dt = datetime.datetime.fromtimestamp(ts_value, tz=datetime.timezone.utc)
        return dt.isoformat()
    except Exception:
        try:
            return pd.to_datetime(ts_value, utc=True).isoformat()
        except Exception:
            return str(ts_value)

def estimate_tokens(messages, model) -> int:
    payload = json.dumps(messages, ensure_ascii=False)
    if tiktoken:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for msg in messages:
            total += 4  # chatml framing
            content = msg.get("content") or ""
            if content:
                total += len(encoding.encode(content))
            if msg.get("name"):
                total -= 1
        total += 2  # assistant priming
        return total
    # Fallback: rough estimate 4 chars per token
    return max(1, math.ceil(len(payload) / 4))

# --- Индикаторы ---
def ema(series, n): return series.ewm(span=n, adjust=False).mean()
def rsi(series, n=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up, ma_down = up.ewm(alpha=1/n, adjust=False).mean(), down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)
def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# --- Получение контекста ---
def get_higher_tf(exchange, symbol, tf="4h", limit=120):
    resolved_symbol = _resolve_symbol_alias(symbol) or symbol
    try:
        data = exchange.fetch_ohlcv(resolved_symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.drop(columns=["ts"], inplace=True)
        df["ema20"] = ema(df["close"],20)
        df["ema50"] = ema(df["close"],50)
        df["rsi"] = rsi(df["close"],14)
        return df.tail(60).to_dict(orient="records")
    except Exception as e:
        log(f"⚠️ Не удалось получить higher_tf {tf}: {e}", Fore.YELLOW)
        return []

def get_funding_rate(exchange, symbol):
    try:
        if hasattr(exchange,"fetchFundingRate"):
            fr = exchange.fetchFundingRate(symbol)
            funding_rate = fr.get("fundingRate")
            try:
                funding_rate = float(funding_rate) if funding_rate is not None else None
            except (TypeError, ValueError):
                funding_rate = None
            next_time = to_iso_utc(fr.get("nextFundingTime"))
            ts_iso = to_iso_utc(fr.get("timestamp"))
            mark_price = fr.get("markPrice")
            index_price = fr.get("indexPrice")
            summary_parts = []
            if funding_rate is not None:
                summary_parts.append(f"текущая ставка {funding_rate*100:.4f}%")
            if next_time:
                summary_parts.append(f"след. выплата {next_time}")
            if not summary_parts:
                summary_parts.append("данные получены, ставка отсутствует")
            return {
                "source": "fetchFundingRate",
                "fundingRate": funding_rate,
                "fundingRatePct": funding_rate*100 if funding_rate is not None else None,
                "timestamp": ts_iso,
                "nextFundingTime": next_time,
                "markPrice": mark_price,
                "indexPrice": index_price,
                "summary": ", ".join(summary_parts)
            }
        if hasattr(exchange, "fetchFundingRateHistory"):
            history = exchange.fetchFundingRateHistory(symbol, limit=1)
            if history:
                fr = history[-1]
                rate = fr.get("fundingRate")
                try:
                    rate = float(rate) if rate is not None else None
                except (TypeError, ValueError):
                    rate = None
                ts_iso = to_iso_utc(fr.get("timestamp") or fr.get("datetime"))
                summary = (
                    f"последняя ставка {rate*100:.4f}% на {ts_iso}"
                    if rate is not None and ts_iso else
                    f"последняя ставка {rate*100:.4f}%"
                    if rate is not None else
                    f"в истории найдена запись на {ts_iso}" if ts_iso else "история получена"
                )
                return {
                    "source": "fetchFundingRateHistory",
                    "fundingRate": rate,
                    "fundingRatePct": rate*100 if rate is not None else None,
                    "timestamp": ts_iso,
                    "summary": summary
                }
    except Exception as e:
        log(f"⚠️ Funding rate недоступен: {e}", Fore.YELLOW)
    return {}

def get_open_interest(exchange, symbol):
    try:
        if hasattr(exchange,"fetchOpenInterestHistory"):
            hist = exchange.fetchOpenInterestHistory(symbol, timeframe="1h", limit=72)
            cleaned = []
            for row in (hist or [])[-24:]:
                if isinstance(row, dict):
                    ts = row.get("timestamp") or row.get("datetime")
                    cleaned.append({
                        "timestamp": to_iso_utc(ts),
                        "openInterestAmount": row.get("openInterestAmount"),
                        "openInterestValue": row.get("openInterestValue"),
                        "symbol": row.get("symbol", symbol)
                    })
                else:
                    cleaned.append(row)
            return cleaned
    except Exception as e:
        log(f"⚠️ Open interest недоступен: {e}", Fore.YELLOW)
    return []


def get_news_from_rss(base_symbol: str, limit: int):
    if not RSS_FEEDS:
        return {"summary": "RSS источники не настроены", "items": []}
    if feedparser is None:
        return {"summary": "feedparser не установлен", "items": []}
    base_upper = (base_symbol or "").upper()
    symbol_articles = []
    general_articles = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            log(f"⚠️ RSS источник недоступен ({url}): {e}", Fore.YELLOW)
            continue
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            link = entry.get("link")
            summary_text = (
                entry.get("summary")
                or (entry.get("summary_detail") or {}).get("value")
                or entry.get("description")
                or ""
            )
            published_raw = entry.get("published", entry.get("updated"))
            item = {
                "title": title,
                "url": link,
                "source": entry.get("source", {}).get("title") if isinstance(entry.get("source"), dict) else entry.get("source"),
                "published_at": to_iso_utc(published_raw),
                "body": summary_text,
            }
            general_articles.append(item)
            if base_upper and base_upper in title.upper():
                symbol_articles.append(item)
            if len(symbol_articles) >= limit and len(general_articles) >= limit:
                break
        if len(symbol_articles) >= limit and len(general_articles) >= limit:
            break
    if symbol_articles:
        selected = symbol_articles[:limit]
        summary = f"RSS {len(selected)} записей по {base_upper}"
    else:
        selected = general_articles[:limit]
        summary = (
            f"RSS {len(selected)} общих новостей" if selected else "новости не найдены (RSS)"
        )
    return {"summary": summary, "items": selected, "asset": base_upper, "source": "rss"}


def get_news_from_cryptocompare(base_symbol: str, limit: int):
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {
        "lang": "EN",
        "sortOrder": "latest",
    }
    try:
        response = requests.get(url, params=params, timeout=6)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        log(f"[WARN] CryptoCompare news unavailable: {exc}", Fore.YELLOW)
        return {"summary": "CryptoCompare unavailable", "items": [], "asset": base_symbol.upper(), "source": "cryptocompare"}
    articles = payload.get("Data") or []
    base_upper = (base_symbol or "").upper()
    focused: list[dict[str, Any]] = []
    general: list[dict[str, Any]] = []
    for art in articles:
        title = art.get("title") or ""
        categories = (art.get("categories") or "").upper()
        entry = {
            "title": title,
            "body": art.get("body") or art.get("summary") or "",
            "url": art.get("url"),
            "source": (art.get("source_info") or {}).get("name") or art.get("source"),
            "published_at": to_iso_utc(art.get("published_on")),
        }
        if base_upper and (base_upper in title.upper() or base_upper in categories):
            focused.append(entry)
        else:
            general.append(entry)
        if len(focused) >= limit and len(general) >= limit:
            break
    selected = focused[:limit] if focused else general[:limit]
    summary = (
        f"CryptoCompare {len(selected)} articles for {base_upper}"
        if base_upper and selected
        else (f"CryptoCompare {len(selected)} latest articles" if selected else "CryptoCompare: no articles")
    )
    return {"summary": summary, "items": selected, "asset": base_upper, "source": "cryptocompare"}

def get_news(symbol):
    base = symbol.split("/")[0].split(":")[0].upper()
    limit = max(1, NEWS_ITEMS_LIMIT)
    provider_payload = None
    if NEWS_PROVIDER in ("cryptocompare", "cc", "crypto"):
        provider_payload = get_news_from_cryptocompare(base, limit)
        if provider_payload.get("items"):
            return provider_payload
    if NEWS_PROVIDER and NEWS_PROVIDER not in ("cryptocompare", "cc", "crypto"):
        log(f"[WARN] Unknown NEWS_PROVIDER '{NEWS_PROVIDER}', falling back to RSS.", Fore.YELLOW)
    return get_news_from_rss(base, limit)

# --- Подключение к бирже ---
def init_exchange():
    exchange = ccxt.bybit({
        "apiKey": os.getenv("BYBIT_API_KEY",""),
        "secret": os.getenv("BYBIT_API_SECRET",""),
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "recvWindow": 5000,
            "hedgeMode": HEDGE_MODE
        }
    })
    exchange.options["recvWindow"] = 5000
    return exchange

def fetch_df(exchange, symbol, tf):
    resolved_symbol = _resolve_symbol_alias(symbol) or symbol
    ohlcv = exchange.fetch_ohlcv(resolved_symbol, timeframe=tf, limit=200)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def ensure_position_mode(exchange):
    desired = "hedged" if HEDGE_MODE else "oneway"
    try:
        if hasattr(exchange, "set_position_mode"):
            symbols_available = set(getattr(exchange, "symbols", []) or [])
            target_symbol = None
            for raw_symbol in PAIR_LIST:
                if not raw_symbol:
                    continue
                if raw_symbol in symbols_available:
                    target_symbol = raw_symbol
                    break
                alias_symbol = SYMBOL_ALIASES.get(raw_symbol)
                if alias_symbol and alias_symbol in symbols_available:
                    target_symbol = alias_symbol
                    break
            if target_symbol is None and symbols_available:
                target_symbol = next(iter(symbols_available))
            if target_symbol:
                exchange.set_position_mode(HEDGE_MODE, target_symbol)
                log(f"⚙️ Position mode set: {desired}", Fore.LIGHTBLACK_EX)
    except Exception as e:
        code = get_bybit_retcode(e)
        if code == 110025:
            log(f"ℹ️ Position mode already set ({desired}, code {code})", Fore.LIGHTBLACK_EX)
        else:
            log(f"⚠️ Failed to set position mode ({desired}): {e}", Fore.YELLOW)


def get_bybit_retcode(error) -> int | None:
    text = str(error)
    match = re.search(r'"retCode"\s*:\s*(-?\d+)', text)
    if not match:
        match = re.search(r"'retCode'\s*:\s*(-?\d+)", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def fetch_usdt_equity(exchange):
    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        log(f"⚠️ Не удалось получить баланс: {e}", Fore.YELLOW)
        return 0.0, 0.0, {}
    usdt = balance.get("USDT") or balance.get("USDT:USDT") or {}

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    total_val = to_float(usdt.get("total") or usdt.get("equity") or usdt.get("walletBalance"))
    free_val = to_float(usdt.get("free") or usdt.get("available") or usdt.get("availableBalance"))
    if free_val is None:
        used_val = to_float(usdt.get("used"))
        if used_val is not None and total_val is not None:
            free_val = total_val - used_val
    if total_val is None and free_val is not None:
        total_val = free_val
    if free_val is None and total_val is not None:
        free_val = total_val
    total_val = float(total_val) if total_val is not None else 0.0
    free_val = float(free_val) if free_val is not None else 0.0
    if total_val < 0:
        total_val = 0.0
    if free_val < 0:
        free_val = 0.0
    return total_val, free_val, balance


def compute_order_amount(order, current_position):
    amount = order.get("amount") or order.get("qty") or order.get("quantity")
    if amount not in (None, "", 0):
        try:
            val = float(amount)
            if abs(val) > 0:
                return abs(val)
        except (TypeError, ValueError):
            pass
    percent = order.get("amountPercent") or order.get("percent")
    if percent and current_position:
        try:
            pct = float(percent)
            if pct <= 0:
                return None
            base = abs(float(current_position.get("amount") or 0))
            if base <= 0:
                return None
            return base * pct / 100.0
        except (TypeError, ValueError):
            return None
    return None


ORDER_TYPE_MAP = {
    "limit": "limit",
    "market": "market",
    "stop": "stop",
    "stop_limit": "stopLimit",
    "take_profit": "takeProfit",
    "stop_loss": "stopLoss",
    "trailing_stop": "trailingStop",
}
ORDER_TYPE_ALIASES = {
    "l": "limit",
    "m": "market",
    "stoplimit": "stop_limit",
    "stop_limit_order": "stop_limit",
    "stopmarket": "stop",
    "stop_market": "stop",
    "stop_market_order": "stop",
    "market_if_touched": "stop",
    "mit": "stop",
    "limit_if_touched": "limit",
    "lit": "limit",
    "takeprofit": "take_profit",
    "take_profit_order": "take_profit",
    "tp": "take_profit",
    "tp_order": "take_profit",
    "stoploss": "stop_loss",
    "stop_loss_order": "stop_loss",
    "sl": "stop_loss",
    "sl_order": "stop_loss",
    "trailingstop": "trailing_stop",
    "trailing": "trailing_stop",
    "trailing_stop_market": "trailing_stop",
    "trailing_stop_order": "trailing_stop",
    "partialclose": "partial_close",
    "partial_close_order": "partial_close",
}
VALID_ORDER_TYPES = set(ORDER_TYPE_MAP.values())


def normalize_order_type_key(raw_type):
    text = "" if raw_type is None else str(raw_type)
    if not text:
        return "limit"
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not key:
        return "limit"
    return ORDER_TYPE_ALIASES.get(key, key)


def get_trigger_direction_for_side(side: str) -> str:
    """Return trigger direction flag understood by ccxt/bybit for a closing order."""
    side_lower = (side or "").lower()
    if side_lower == "sell":
        return "descending"
    if side_lower == "buy":
        return "ascending"
    # Default to trigger on downside to avoid missing protection for long positions.
    return "descending"


def _resolve_ai_model_for_pairs(pair_count: Optional[int]) -> str:
    if AI_MODEL_CHEAP and AI_TOKEN_USAGE_TOTAL >= AI_SECONDARY_BUDGET_START:
        return AI_MODEL_CHEAP
    primary = AI_MODEL_PRIMARY or AI_MODEL
    if primary:
        return primary
    return AI_MODEL_CHEAP or AI_MODEL


def update_ai_model_for_analysis(pair_count: Optional[int]) -> str:
    """Switch active AI model based on number of symbols that need analysis."""
    global AI_MODEL
    desired = _resolve_ai_model_for_pairs(pair_count)
    if not desired:
        return AI_MODEL
    if AI_MODEL != desired:
        previous = AI_MODEL or "undefined"
        AI_MODEL = desired
        context = f"{pair_count} symbols" if pair_count is not None else "unknown workload"
        log(f"[AI] Model switched {previous} -> {desired} ({context})", Fore.LIGHTBLACK_EX)
    return AI_MODEL


def _init_ai_cycle_usage() -> None:
    global AI_TOKEN_USAGE_TOTAL, AI_TOKEN_USAGE_BY_MODEL
    AI_TOKEN_USAGE_TOTAL = 0
    AI_TOKEN_USAGE_BY_MODEL = {}


def _log_ai_request(model: str, estimate: int, context: str) -> None:
    budget = AI_HARD_STOP_BUDGET or AI_TOKEN_BUDGET_CYCLE
    cap = _current_request_token_cap()
    cap_text = f", кап {cap}" if cap else ""
    log(
        f"[AI] {context}: модель {model}, оценка {estimate} токенов (использовано {AI_TOKEN_USAGE_TOTAL}/{budget}{cap_text})",
        Fore.LIGHTBLACK_EX,
    )


def _ensure_token_budget(estimate: int, model: str, context: str) -> bool:
    hard_stop = AI_HARD_STOP_BUDGET or AI_TOKEN_BUDGET_CYCLE
    if hard_stop and AI_TOKEN_USAGE_TOTAL >= hard_stop:
        log(
            f"[AI] Пропуск {context}: достигнут жёсткий предел {hard_stop} токенов",
            Fore.YELLOW,
        )
        return False
    projected_total = AI_TOKEN_USAGE_TOTAL + (estimate or 0)
    if hard_stop and projected_total > hard_stop:
        log(
            f"[AI] Пропуск {context}: запрос превысит предел {hard_stop} токенов (будет {projected_total})",
            Fore.YELLOW,
        )
        return False
    return True


def _register_ai_usage(model: str, usage: Any, context: str) -> None:
    if not usage:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
    completion_tokens = getattr(usage, "completion_tokens", None) or 0
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    global AI_TOKEN_USAGE_TOTAL, AI_TOKEN_USAGE_BY_MODEL
    AI_TOKEN_USAGE_TOTAL += total_tokens
    stats = AI_TOKEN_USAGE_BY_MODEL.setdefault(
        model,
        {"prompt": 0, "completion": 0, "total": 0, "requests": 0},
    )
    stats["prompt"] += prompt_tokens
    stats["completion"] += completion_tokens
    stats["total"] += total_tokens
    stats["requests"] += 1
    log(
        f"[AI] {context}: {model} prompt={prompt_tokens} completion={completion_tokens} total={total_tokens} "
        f"(цикл {AI_TOKEN_USAGE_TOTAL}/{AI_TOKEN_BUDGET_CYCLE})",
        Fore.LIGHTBLACK_EX,
    )
    if AI_TOKEN_BUDGET_CYCLE and AI_TOKEN_USAGE_TOTAL > AI_TOKEN_BUDGET_CYCLE:
        log(
            f"[AI] Внимание: превышен лимит токенов {AI_HARD_STOP_BUDGET or AI_TOKEN_BUDGET_CYCLE} (использовано {AI_TOKEN_USAGE_TOTAL})",
            Fore.YELLOW,
        )
    _maybe_switch_model_after_usage()


def _is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _resolve_symbol_alias(symbol: str | None) -> str | None:
    if not symbol:
        return None
    sym = str(symbol).strip()
    if not sym:
        return None
    if sym in SYMBOL_ALIASES:
        sym = SYMBOL_ALIASES[sym]
    if "/" in sym and ":" in sym:
        return sym
    sym_upper = sym.upper()
    mapped = TICKER_TO_SYMBOL.get(sym_upper)
    if mapped:
        return mapped
    if sym_upper.endswith("USDT"):
        base = sym_upper[:-4]
        mapped = TICKER_TO_SYMBOL.get(base)
        if mapped:
            return mapped
        return f"{base}/USDT:USDT"
    mapped = TICKER_TO_SYMBOL.get(sym_upper)
    if mapped:
        return mapped
    return f"{sym_upper}/USDT:USDT"


def _canonical_decision_symbol(symbol: str | None) -> str | None:
    resolved = _resolve_symbol_alias(symbol)
    if resolved:
        return resolved
    if symbol:
        return str(symbol).strip()
    return None

def _order_allows_increase(order: dict) -> bool:
    if not isinstance(order, dict):
        return False
    flags = ["allowIncrease", "allow_increase", "increase", "increasePosition", "increase_position", "scaleIn", "scale_in"]
    for key in flags:
        if _is_truthy_flag(order.get(key)):
            return True
    intent = (order.get("intent") or order.get("action") or "").lower()
    if intent in {"increase", "scale_in", "add", "add_position"}:
        return True
    return False

def _maybe_switch_model_after_usage() -> None:
    global AI_MODEL
    if AI_MODEL_CHEAP and AI_TOKEN_USAGE_TOTAL >= AI_SECONDARY_BUDGET_START and AI_MODEL != AI_MODEL_CHEAP:
        previous = AI_MODEL
        AI_MODEL = AI_MODEL_CHEAP
        log(f"[AI] Switching to cheaper model {AI_MODEL_CHEAP} after {AI_TOKEN_USAGE_TOTAL} tokens (was {previous})", Fore.YELLOW)


def _current_request_token_cap() -> Optional[int]:
    if AI_HARD_STOP_BUDGET and AI_TOKEN_USAGE_TOTAL >= AI_HARD_STOP_BUDGET:
        return 0
    if False:
        return AI_PER_REQUEST_TOKEN_CAP
    return None


def _safe_round(value: Optional[float], digits: int = 8) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return round(float(value), digits)


def _extract_protection_orders(orders) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for order in orders or []:
        if not isinstance(order, dict):
            continue
        reduce_flag = _is_truthy_flag(order.get("reduceOnly"))
        close_on_trigger = _is_truthy_flag(order.get("closeOnTrigger"))
        order_type = (order.get("type") or "").lower()
        has_tp_sl_field = any(
            order.get(key) not in (None, "")
            for key in ("takeProfit", "stopLoss", "tp", "sl")
        )
        if not (
            reduce_flag
            or close_on_trigger
            or has_tp_sl_field
            or order_type in ("stop", "stop_limit", "stoploss", "takeprofit", "trailingstop")
            or _has_stop_flag(order)
            or _has_trailing_flag(order)
        ):
            continue
        extracted.append(order)
    return extracted


def _categorize_protection_orders(orders) -> dict[str, list[tuple]]:
    summary: dict[str, list[tuple]] = {"stop": [], "take_profit": [], "trailing": []}
    for order in _extract_protection_orders(orders):
        order_type = (order.get("type") or "").lower()
        amount_val = safe_float(
            order.get("remaining") or order.get("leavesQty") or order.get("amount")
        )
        amount_round = _safe_round(amount_val, 6)
        trailing_val = safe_float(order.get("trailingStop"))
        stop_price = safe_float(
            order.get("stopPrice") or order.get("triggerPrice") or order.get("stopLoss")
        )
        take_price = safe_float(
            order.get("price")
            or order.get("takeProfit")
            or order.get("tp")
        )
        if _has_trailing_flag(order):
            summary["trailing"].append((_safe_round(trailing_val, 6), amount_round))
        elif _has_stop_flag(order):
            summary["stop"].append((_safe_round(stop_price, 6), amount_round))
        else:
            if take_price is None and order_type not in ("takeprofit", "take_profit"):
                take_price = stop_price if stop_price is not None else trailing_val
            summary["take_profit"].append((_safe_round(take_price, 6), amount_round))
    for key in summary:
        summary[key].sort()
    return summary


def _describe_protection_changes(initial_orders, final_orders) -> list[str]:
    changes: list[str] = []
    initial_summary = _categorize_protection_orders(initial_orders)
    final_summary = _categorize_protection_orders(final_orders)

    spec = {
        "stop": {
            "label": "СЛ",
            "changed": "изменены СЛ",
            "added": "добавлено СЛ",
            "removed": "убрано СЛ",
        },
        "take_profit": {
            "label": "ТП",
            "changed": "изменены ТП",
            "added": "добавлено ТП",
            "removed": "убрано ТП",
        },
        "trailing": {
            "label": "трейлинг",
            "changed": "изменён трейлинг",
            "added": "добавлен трейлинг",
            "removed": "убран трейлинг",
            "added_plural": "добавлено трейлинг: {}",
            "removed_plural": "убрано трейлинг: {}",
            "changed_plural": "изменены трейлинги",
        },
    }

    for key, meta in spec.items():
        init_list = initial_summary.get(key, [])
        final_list = final_summary.get(key, [])
        diff = len(final_list) - len(init_list)
        if diff > 0:
            if key == "trailing":
                if diff == 1:
                    changes.append(meta["added"])
                else:
                    plural_text = meta.get("added_plural")
                    if plural_text:
                        changes.append(plural_text.format(diff))
                    else:
                        changes.append(f"добавлено {meta['label']}: {diff}")
            else:
                changes.append(f"{meta['added']}: {diff}")
        elif diff < 0:
            diff_abs = abs(diff)
            if key == "trailing":
                if diff_abs == 1:
                    changes.append(meta["removed"])
                else:
                    plural_text = meta.get("removed_plural")
                    if plural_text:
                        changes.append(plural_text.format(diff_abs))
                    else:
                        changes.append(f"убрано {meta['label']}: {diff_abs}")
            else:
                changes.append(f"{meta['removed']}: {diff_abs}")
        else:
            if init_list != final_list and init_list and final_list:
                if key == "trailing":
                    changes.append(meta["changed"] if len(init_list) == 1 else meta.get("changed_plural", meta["changed"]))
                else:
                    changes.append(meta["changed"])
    return changes


def _protection_orders_signature(orders) -> tuple:
    snapshot: list[tuple[Any, ...]] = []
    for order in _extract_protection_orders(orders):
        side = (order.get("side") or "").lower()
        order_type = (order.get("type") or "").lower()
        price = _safe_round(safe_float(order.get("price")))
        trigger = _safe_round(
            safe_float(
                order.get("stopPrice")
                or order.get("triggerPrice")
                or order.get("stopLoss")
            )
        )
        take_profit = _safe_round(safe_float(order.get("takeProfit")))
        amount = _safe_round(
            safe_float(
                order.get("remaining")
                or order.get("leavesQty")
                or order.get("amount")
            )
        )
        snapshot.append((side, order_type, price, trigger, take_profit, amount))
    snapshot.sort()
    return tuple(snapshot)


def get_current_commit_info() -> tuple[str | None, str | None, str | None]:
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(SCRIPT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None, None, None
    commit_timestamp = None
    try:
        commit_timestamp = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%cI"],
            cwd=str(SCRIPT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit_timestamp = None
    commit_message = None
    try:
        commit_message = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=str(SCRIPT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit_message = None
    return commit_hash or None, commit_message or None, commit_timestamp or None


def _sync_with_remote() -> None:
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        return
    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        working_tree_dirty = bool(status_proc.stdout.strip())
    except Exception as exc:
        log(f"[WARN] Git status failed before sync: {exc}", Fore.YELLOW)
        return
    try:
        fetch_proc = subprocess.run(
            ["git", "fetch", "--all", "--prune"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        fetch_output = (fetch_proc.stdout or "").strip()
        if fetch_output:
            log(f"[GIT] fetch: {fetch_output}", Fore.LIGHTBLACK_EX)
    except subprocess.CalledProcessError as exc:
        log(f"[WARN] Git fetch failed: {exc.stderr or exc.stdout or exc}", Fore.YELLOW)
        return
    if working_tree_dirty:
        log("[GIT] Skipping pull (working tree has local changes).", Fore.LIGHTBLACK_EX)
        return
    try:
        pull_proc = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        pull_output = (pull_proc.stdout or pull_proc.stderr or "").strip()
        if pull_output:
            log(f"[GIT] pull: {pull_output}", Fore.LIGHTBLACK_EX)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr or exc.stdout or str(exc)
        log(f"[WARN] Git pull failed: {details}", Fore.YELLOW)


def _cleanup_redundant_stop_orders(exchange, symbol, reduce_orders, protection_side, position_qty, is_long):
    """Remove surplus reduce-only stop orders that exceed current position coverage."""
    if (
        not reduce_orders
        or position_qty is None
        or not math.isfinite(position_qty)
        or position_qty <= 0
    ):
        return [], []

    stop_entries: list[dict[str, Any]] = []
    for order in reduce_orders:
        if not isinstance(order, dict):
            continue
        try:
            if order.get("reduceOnly") not in (True, "true", "1", 1):
                continue
            if (order.get("side") or "").lower() != protection_side:
                continue
        except AttributeError:
            continue
        order_type = (order.get("type") or "").lower()
        trigger_price = safe_float(
            order.get("stopPrice")
            or order.get("triggerPrice")
            or order.get("stopLoss")
        )
        if order_type not in ("stop", "stoploss", "stop_limit", "stoplimit") and trigger_price is None:
            continue
        order_id = order.get("id")
        if not order_id:
            continue
        remaining = safe_float(order.get("remaining") or order.get("leavesQty"))
        if remaining is None or remaining <= 0:
            remaining = safe_float(order.get("amount"))
        if remaining is None or remaining <= 0:
            continue
        if trigger_price is None or not math.isfinite(trigger_price):
            continue
        stop_entries.append(
            {
                "id": str(order_id),
                "trigger": trigger_price,
                "amount": remaining,
            }
        )

    if len(stop_entries) <= 1:
        return [], []

    stop_entries.sort(key=lambda item: item["trigger"], reverse=is_long)
    coverage = 0.0
    tolerance = max(position_qty * 1e-6, 1e-8)
    keep_ids: set[str] = set()
    for entry in stop_entries:
        keep_ids.add(entry["id"])
        coverage += entry["amount"]
        if coverage >= position_qty - tolerance:
            break

    cancelled_ids: list[str] = []
    cancel_errors: list[tuple[str, str]] = []
    for entry in stop_entries:
        if entry["id"] in keep_ids:
            continue
        success, err = cancel_order_by_id(exchange, symbol, entry["id"])
        if success:
            cancelled_ids.append(entry["id"])
        else:
            cancel_errors.append((entry["id"], err))
    return cancelled_ids, cancel_errors


def ensure_position_protection(exchange, symbol, position, df_primary, open_orders, config=None):
    cfg = config or {}
    position_amount = safe_float((position or {}).get("amount"))
    if position_amount is None or not math.isfinite(position_amount):
        position_amount = safe_float((position or {}).get("contracts"))
    if position_amount is None or not math.isfinite(position_amount):
        position_amount = safe_float((position or {}).get("size"))
    if position is None or position_amount is None or not math.isfinite(position_amount) or position_amount == 0:
        return open_orders or []

    position_side = (position.get("side") or "").lower()
    exchange_symbol = _resolve_symbol_alias(symbol) or symbol
    if position_side in ("sell", "short"):
        protection_side = "buy"
        is_long = False
    elif position_side in ("buy", "long"):
        protection_side = "sell"
        is_long = True
    else:
        is_long = position_amount > 0
        protection_side = "sell" if is_long else "buy"

    sl_mult = cfg.get("sl_atr", SL_ATR)
    tp_mult = cfg.get("tp_atr", TP_ATR)
    trailing_mult = cfg.get("trailing_atr_mult", TRAILING_ATR_MULT)
    reduce_orders_source = open_orders or []
    if not reduce_orders_source:
        reduce_orders_source = fetch_open_orders_for_symbol(exchange, symbol, limit=200)
    reduce_orders = [order for order in reduce_orders_source if isinstance(order, dict)]
    position_qty = abs(position_amount)
    cancelled_stop_ids, cancel_stop_errors = _cleanup_redundant_stop_orders(
        exchange,
        symbol,
        reduce_orders,
        protection_side,
        position_qty,
        is_long,
    )
    if cancelled_stop_ids:
        summary = ", ".join(cancelled_stop_ids)
        log(f"✅ {symbol}: удалены лишние стоп-ордера: {summary}", Fore.LIGHTBLUE_EX)
        send_tg(f"✅ {symbol}: удалены лишние стоп-ордера: {summary}")
    if cancel_stop_errors:
        details = "; ".join(f"{oid}: {err}" for oid, err in cancel_stop_errors)
        log(f"⚠️ {symbol}: не удалось удалить часть стоп-ордеров: {details}", Fore.YELLOW)
        send_tg(f"⚠️ {symbol}: ошибка при удалении стоп-ордеров: {details}")
    if cancelled_stop_ids or cancel_stop_errors:
        open_orders = fetch_open_orders_for_symbol(exchange, symbol, limit=200)
        reduce_orders = [order for order in (open_orders or []) if isinstance(order, dict)]

    has_stop = False
    has_take = False
    has_trailing = False
    for existing in reduce_orders:
        try:
            if existing.get("reduceOnly") not in (True, "true", "1", 1):
                continue
            if (existing.get("side") or "").lower() != protection_side:
                continue
            order_type = (existing.get("type") or "").lower()
        except AttributeError:
            continue
        stop_price_existing = safe_float(existing.get("stopPrice") or existing.get("triggerPrice") or existing.get("stopLoss"))
        trailing_flag = safe_float(existing.get("trailingStop")) if isinstance(existing.get("trailingStop"), (int, float, str)) else None
        if order_type in ("stop", "stoploss", "stop_limit", "stoplimit") or stop_price_existing is not None:
            has_stop = True
        elif order_type in ("takeprofit", "limit") and existing.get("price") is not None:
            has_take = True
        elif order_type == "trailingstop" or trailing_flag:
            has_trailing = True
    if has_stop and has_take and (trailing_mult <= 0 or has_trailing):
        return open_orders or []

    df_calc = df_primary.copy() if isinstance(df_primary, pd.DataFrame) and not df_primary.empty else None
    if df_calc is None:
        return open_orders or []
    if "atr" not in df_calc.columns:
        try:
            df_calc["atr"] = atr(df_calc, 14)
        except Exception as exc:
            log(f"⚠️ {symbol}: не удалось вычислить ATR для защиты позиции ({exc})", Fore.YELLOW)
            return open_orders or []
    last_row = df_calc.iloc[-1]
    price = safe_float(last_row.get("close"))
    atrv = safe_float(last_row.get("atr"))
    if not (math.isfinite(price) and math.isfinite(atrv) and atrv and atrv > 0):
        log(f"⚠️ {symbol}: нет валидных значений ATR/цены для защиты позиции", Fore.YELLOW)
        return open_orders or []

    if is_long:
        stop_price = price - sl_mult * atrv
        take_price = price + tp_mult * atrv
    else:
        stop_price = price + sl_mult * atrv
        take_price = price - tp_mult * atrv
    qty = position_qty
    if not math.isfinite(qty) or qty <= 0:
        return open_orders or []
    position_idx = get_position_idx(protection_side)
    base_params = {
        "reduceOnly": True,
    }
    if position_idx is not None:
        base_params["positionIdx"] = position_idx

    created_orders = []
    try:
        trigger_direction = get_trigger_direction_for_side(protection_side)
        stop_params = dict(base_params)
        stop_params.update(
            {
                "triggerPrice": stop_price,
                "triggerDirection": trigger_direction,
                "closeOnTrigger": True,
            }
        )
        exchange.create_order(exchange_symbol, "market", protection_side, qty, None, stop_params)
        created_orders.append(("stopLoss", stop_price))
    except Exception as exc:
        log(f"⚠️ {symbol}: не удалось выставить стоп-ордер защиты позиции: {exc}", Fore.YELLOW)

    try:
        tp_params = dict(base_params)
        tp_params["takeProfit"] = take_price
        tp_params.setdefault("timeInForce", "GTC")
        exchange.create_order(exchange_symbol, "limit", protection_side, qty, take_price, tp_params)
        created_orders.append(("takeProfit", take_price))
    except Exception as exc:
        log(f"⚠️ {symbol}: не удалось выставить тейк-профит позиции: {exc}", Fore.YELLOW)

    if trailing_mult > 0 and math.isfinite(trailing_mult):
        try:
            trailing_params = dict(base_params)
            trailing_params["trailingStop"] = trailing_mult * atrv
            exchange.create_order(exchange_symbol, "trailingStop", protection_side, qty, None, trailing_params)
            created_orders.append(("trailingStop", trailing_mult * atrv))
        except Exception as exc:
            log(f"⚠️ {symbol}: не удалось выставить трейлинг-стоп: {exc}", Fore.YELLOW)

    if created_orders:
        log(f"🛡️ {symbol}: обновлена защита позиции {created_orders}", Fore.LIGHTBLUE_EX)
        send_tg(
            f"🛡️ {symbol}: обновлена защита позиции\n"
            + "\n".join(f"- {typ} @ {price}" for typ, price in created_orders)
        )
    return fetch_open_orders_for_symbol(exchange, symbol)



def execute_extra_orders(
    exchange,
    symbol,
    orders,
    current_position=None,
    open_orders=None,
    available_margin: float | None = None,
    symbol_leverage: float | None = None,
    max_limits_per_side: int = 1,
):
    executed = []
    exchange_symbol = _resolve_symbol_alias(symbol) or symbol
    open_orders = open_orders or []
    position_side = ((current_position or {}).get("side") or "").lower()
    reduce_only_map: dict[str, list[dict]] = {}
    existing_non_reduce_limits: dict[tuple[str, float], int] = {}
    for existing in open_orders:
        if not isinstance(existing, dict):
            continue
        try:
            reduce_flag = _is_truthy_flag(existing.get("reduceOnly"))
        except AttributeError:
            continue
        side_key = (existing.get("side") or "").lower()
        if reduce_flag:
            reduce_only_map.setdefault(side_key, []).append(existing)
        else:
            order_type_existing = (existing.get("type") or "").lower()
            if order_type_existing == "limit":
                price_existing = safe_float(existing.get("price"))
                if price_existing is not None and math.isfinite(price_existing) and price_existing > 0:
                    price_key = round(price_existing, NON_REDUCE_PRICE_DECIMALS)
                    key = (side_key, price_key)
                    existing_non_reduce_limits[key] = existing_non_reduce_limits.get(key, 0) + 1
    position_amount = 0.0
    if current_position:
        try:
            position_amount = float(current_position.get("amount") or 0)
        except (TypeError, ValueError):
            position_amount = 0.0
    margin_buffer = None
    if available_margin is not None:
        try:
            margin_buffer = max(0.0, float(available_margin) * ORDER_MARGIN_UTILIZATION)
        except (TypeError, ValueError):
            margin_buffer = None
    leverage = symbol_leverage if symbol_leverage and symbol_leverage > 0 else 1.0
    max_limits_per_side = max(0, max_limits_per_side)
    if not isinstance(orders, (list, tuple)):
        log(f"[WARN] Invalid extra orders payload for {symbol}; skipping.", Fore.YELLOW)
        return executed, False
    cancelled_success = []
    cancel_errors = []
    for idx, order in enumerate(orders, 1):
        if not isinstance(order, dict):
            log(f"[WARN] Extra order #{idx} for {symbol} is not a dict; skipping.", Fore.YELLOW)
            continue
        if order.get("status") is not None and order.get("id"):
            log(f"[INFO] Retaining existing order {order.get('id')} for {symbol}; skipping extra placement.", Fore.LIGHTBLACK_EX)
            continue
        raw_type = (
            order.get("type")
            or order.get("orderType")
            or order.get("order_type")
            or order.get("ccxt_type")
        )
        order_type_key = normalize_order_type_key(raw_type)
        params = dict(order.get("params") or {})
        note = order.get("note") or order.get("comment") or ""
        reduce_only_flag = _is_truthy_flag(order.get("reduceOnly"))
        if reduce_only_flag:
            params["reduceOnly"] = True
        elif "reduceOnly" in params:
            params["reduceOnly"] = bool(params["reduceOnly"])
        side = (order.get("side") or "").lower()
        if not side:
            log(f"[WARN] Missing side in extra order #{idx} for {symbol}; skipping.", Fore.YELLOW)
            continue
        amount = compute_order_amount(order, current_position)
        if amount is None:
            log(f"[WARN] Unable to determine amount for extra order #{idx} for {symbol}; skipping.", Fore.YELLOW)
            continue
        try:
            amount = abs(float(amount))
        except (TypeError, ValueError):
            log(f"[WARN] Invalid amount in extra order #{idx} for {symbol}; skipping.", Fore.YELLOW)
            continue
        if amount <= 0:
            log(f"[WARN] Invalid amount in extra order #{idx} for {symbol}; skipping.", Fore.YELLOW)
            continue
        price = safe_float(order.get("price"))
        if price is not None and (not math.isfinite(price) or price <= 0):
            log(f"[WARN] Invalid price in extra order #{idx} for {symbol}; skipping.", Fore.YELLOW)
            continue
        price_key = round(price, NON_REDUCE_PRICE_DECIMALS) if price is not None else None
        is_reduce_only = _is_truthy_flag(params.get("reduceOnly"))
        if not is_reduce_only and position_side:
            same_direction = (
                (position_side in ("long", "buy") and side == "buy")
                or (position_side in ("short", "sell") and side == "sell")
            )
            if same_direction and not _order_allows_increase(order):
                log(
                    f"[WARN] Skipping additive extra order #{idx} for {symbol}: position side {position_side} vs order {side.upper()}",
                    Fore.YELLOW,
                )
                continue
        if (
            not is_reduce_only
            and order_type_key == "limit"
            and price is not None
            and max_limits_per_side > 0
        ):
            key = (side, price_key)
            if existing_non_reduce_limits.get(key, 0) >= max_limits_per_side:
                log(
                    f"[WARN] Skipping duplicate non-reduce limit for {symbol} {side.upper()} @ {price}",
                    Fore.YELLOW,
                )
                continue
        margin_required = None
        if not is_reduce_only and margin_buffer is not None:
            ref_price = price
            if ref_price is None:
                ref_price = safe_float(
                    order.get("triggerPrice")
                    or order.get("stopPrice")
                    or order.get("stop_price")
                    or params.get("triggerPrice")
                    or params.get("stopPrice")
                    or params.get("stop_price")
                )
            if ref_price is not None and math.isfinite(ref_price) and ref_price > 0:
                notional_estimate = amount * ref_price
                margin_required = notional_estimate / leverage if leverage else notional_estimate
                if margin_required > margin_buffer:
                    log(
                        f"[WARN] Skipping extra order #{idx} for {symbol}: margin required {margin_required:.2f} USDT exceeds available {margin_buffer:.2f} USDT",
                        Fore.YELLOW,
                    )
                    continue
        if is_reduce_only:
            if abs(position_amount) == 0:
                log(f"[INFO] Skipping reduce-only order for {symbol}: no active position", Fore.LIGHTBLACK_EX)
                send_tg(f"[INFO] {symbol}: reduce-only order skipped (flat position)")
                continue
            existing_list = reduce_only_map.get(side)
            if existing_list:
                for existing_order in existing_list:
                    oid = existing_order.get("id")
                    if not oid:
                        continue
                    success, err = cancel_order_by_id(exchange, symbol, str(oid))
                    if success:
                        cancelled_success.append(str(oid))
                        log(f"[INFO] Cancelled existing reduce-only order {oid} for {symbol}", Fore.LIGHTBLUE_EX)
                    else:
                        cancel_errors.append((oid, err))
                        log(f"[WARN] Failed to cancel reduce-only order {oid} for {symbol}: {err}", Fore.YELLOW)
                reduce_only_map[side] = []
        position_idx = order.get("positionIdx")
        if position_idx is None:
            params.setdefault("positionIdx", get_position_idx(side))
        else:
            params["positionIdx"] = position_idx
        if order_type_key == "partial_close":
            base_order_type_raw = (
                order.get("orderType")
                or order.get("order_type")
                or order.get("ccxt_type")
                or order.get("baseType")
            )
            base_order_type_key = normalize_order_type_key(base_order_type_raw or "market")
            ccxt_type = ORDER_TYPE_MAP.get(base_order_type_key, base_order_type_key)
            params.setdefault("reduceOnly", True)
            if not side and current_position:
                side = "sell" if (current_position.get("amount") or 0) > 0 else "buy"
        else:
            ccxt_type = ORDER_TYPE_MAP.get(order_type_key, order_type_key)
            if order_type_key == "stop_loss":
                trigger_price = (
                    order.get("triggerPrice")
                    or order.get("stopPrice")
                    or order.get("stop_price")
                    or params.get("triggerPrice")
                    or params.get("stopPrice")
                    or params.get("stop_price")
                    or price
                )
                trigger_price = safe_float(trigger_price)
                if trigger_price is None or not math.isfinite(trigger_price):
                    log(f"[WARN] Missing triggerPrice for extra order #{idx} {symbol}", Fore.YELLOW)
                    continue
                ccxt_type = "market"
                price = None
                params.setdefault("reduceOnly", True)
                params["triggerPrice"] = trigger_price
                params.setdefault("triggerDirection", get_trigger_direction_for_side(side))
                params.setdefault("closeOnTrigger", True)
                params.pop("stopLoss", None)
                params.pop("stopPrice", None)
                params.pop("stop_price", None)
        if ccxt_type not in VALID_ORDER_TYPES:
            fallback_type = "limit" if price is not None else "market"
            log(
                f"[WARN] Unsupported order type '{raw_type}' for extra order #{idx} {symbol}, falling back to {fallback_type}",
                Fore.YELLOW,
            )
            ccxt_type = fallback_type
        if ccxt_type in ("limit", "stopLimit", "takeProfit", "stopLoss") and price is None:
            log(f"[WARN] Missing price for extra order #{idx} ({ccxt_type}) {symbol}", Fore.YELLOW)
            continue
        try:
            order_id = exchange.create_order(exchange_symbol, ccxt_type, side, amount, price, params)
            display_type = "STOP-MARKET" if order_type_key == "stop_loss" else ccxt_type.upper()
            desc = f"{display_type} {side.upper()} {amount}"
            if price:
                desc += f" @ {price}"
            if note:
                desc += f" - {note}"
            executed.append(desc)
            log(f"[INFO] Extra order placed for {symbol}: {desc}", Fore.LIGHTBLUE_EX)
            if not is_reduce_only and order_type_key == "limit" and price is not None:
                key = (side, price_key if price_key is not None else round(price, NON_REDUCE_PRICE_DECIMALS))
                existing_non_reduce_limits[key] = existing_non_reduce_limits.get(key, 0) + 1
            if margin_buffer is not None and margin_required is not None:
                margin_buffer = max(0.0, margin_buffer - margin_required)
        except Exception as e:
            log(f"[ERROR] Extra order #{idx} for {symbol} failed: {e}", Fore.RED)
    if cancelled_success:
        send_tg(f"[INFO] {symbol}: cancelled reduce-only orders {', '.join(cancelled_success)}")
    if cancel_errors:
        errs = "; ".join(f"{oid}: {err}" for oid, err in cancel_errors)
        send_tg(f"[WARN] {symbol}: errors cancelling orders - {errs}")
    actions_performed = bool(executed or cancelled_success or cancel_errors)
    return executed, actions_performed

# --- Решение модели (2 прохода, русский лог) ---
def ai_decision(
    symbol,
    df_primary,
    equity,
    available_margin,
    exchange,
    current_position=None,
    open_orders=None,
    extra_context=None,
    target_meta=None,
    news_payload=None,
    initial_decision=None
):
    if not AI_KEY:
        log("❌ Не указан OPENAI_API_KEY", Fore.RED)
        return None

    df_30m = df_primary
    extra_context = extra_context or {}
    target_meta = target_meta or {}
    client = OpenAI(api_key=AI_KEY, timeout=30)
    df_30m["ema20"] = ema(df_30m["close"],20)
    df_30m["ema50"] = ema(df_30m["close"],50)
    df_30m["rsi"] = rsi(df_30m["close"],14)
    df_30m["atr"] = atr(df_30m,14)
    portfolio_guidance = dict(target_meta) if isinstance(target_meta, dict) else {}
    extra_context_payload = extra_context if isinstance(extra_context, dict) else {}
    news_payload_payload = news_payload if isinstance(news_payload, dict) else None
    higher_tf = get_higher_tf(exchange, symbol, "4h")

    higher_trend_bias = None
    higher_trend_label = "неопределён"
    if higher_tf:
        last_higher = higher_tf[-1]
        ema20_ht = last_higher.get("ema20")
        ema50_ht = last_higher.get("ema50")
        if ema20_ht is not None and ema50_ht is not None:
            if ema20_ht > ema50_ht:
                higher_trend_bias = "buy"
                higher_trend_label = "восходящий"
            elif ema20_ht < ema50_ht:
                higher_trend_bias = "sell"
                higher_trend_label = "нисходящий"

    context_counts = {
        "30m": min(DEFAULT_CONTEXT_30M, len(df_30m)),
        "4h": min(DEFAULT_CONTEXT_4H, len(higher_tf))
    }
    current_context = {}
    position_payload = None
    if current_position:
        position_payload = {
            "side": current_position.get("side"),
            "amount": current_position.get("amount"),
            "entryPrice": current_position.get("entryPrice"),
            "leverage": current_position.get("leverage"),
            "unrealizedPnl": current_position.get("unrealizedPnl"),
            "liquidationPrice": current_position.get("liquidationPrice")
        }
    open_orders = open_orders or []

    # >>>>>>>>>>>> ИСПРАВЛЕНО: system_msg как тройная строка без \u-escape <<<<<<<<<<<<
    system_msg = """Ты — ИИ-помощник по трейдингу в сбалансированном интрадей стиле. 
Работаешь по сценарию: (1) если тренды 30m и 4h совпадают и RSI не в экстремумах — входи по тренду; 
(2) если 30m показывает зарождающийся разворот против слабого тренда на 4h — допускается контртренд с короткой целью; 
(3) skip используется только при реальном конфликте сигналов или явной неопределённости. 
Обязательно анализируй EMA20/EMA50, RSI(14), ATR(14) на 30m и 4h, формируй понятный риск/идею. 
Если уверенность < 70% или сигналы расходятся — сначала запроси дополнительные данные через поле 'needs' 
(доступно: higher_tf:<tf>, funding, open_interest, news, а также {"timeframes":["1h"],"indicators":[{"indicator":"ema","length":55}, "atr14"]}), и только после доп. проверки выбирай конечное действие. 
Если позиция уже открыта, не открывай её заново: оцени необходимость частичного сокращения, закрытия или удержания. 
Если по символу есть активные лимитные/стоп-ордера (open_orders), не дублируй их без пересмотра. 
Для отмены/замены ордеров передавай cancel_orders и replace_orders. 
Для частичных закрытий, дополнительных лимитов/стопов, трейлингов и других операций используй массив 'orders', 
описывая ордера в стиле CCXT (type, side, amount/percent, price, params). 
Если выбираешь action="skip", обязательно укажи причину, опираясь на показания этих индикаторов. 
Ответ строго в формате JSON без текста.
Decide decisively. Always include a numeric "confidence" between 0 and 1 and target values >=0.70 when signals align. If confidence would fall below the threshold, request the missing context via "needs" with concrete items instead of hesitating. Make recommendations with clear reasoning."""
    # >>>>>>>>>>>> конец исправления <<<<<<<<<<<<

    ema_trend_bias = None
    ema_trend_label = "неопределён"
    ema_slope = None
    if not df_30m.empty:
        last_row = df_30m.iloc[-1]
        ema20_last = last_row.get("ema20")
        ema50_last = last_row.get("ema50")
        if pd.notna(ema20_last) and pd.notna(ema50_last):
            if ema20_last > ema50_last:
                ema_trend_bias = "buy"
                ema_trend_label = "восходящий"
            elif ema20_last < ema50_last:
                ema_trend_bias = "sell"
                ema_trend_label = "нисходящий"
        if len(df_30m) >= 3 and pd.notna(df_30m.iloc[-1].get("ema20")) and pd.notna(df_30m.iloc[-3].get("ema20")):
            ema_slope = df_30m.iloc[-1]["ema20"] - df_30m.iloc[-3]["ema20"]

    def build_context():
        tail_30m = df_30m.tail(context_counts["30m"]).reset_index()
        if not tail_30m.empty:
            tail_30m["timestamp"] = pd.to_datetime(tail_30m["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            tf_30m = tail_30m[["timestamp","open","high","low","close","volume","ema20","ema50","rsi","atr"]].to_dict(orient="records")
        else:
            tf_30m = []
        count_4h = min(context_counts["4h"], len(higher_tf))
        tf_4h = higher_tf[-count_4h:] if count_4h else []
        context = {"tf_30m": tf_30m, "tf_4h": tf_4h}
        if position_payload:
            context["position"] = position_payload
        if open_orders:
            context["open_orders"] = open_orders
        return context

    def build_prompt(extra=None, bias=False):
        news_desc = "CryptoCompare API (fallback: RSS feeds)" if NEWS_PROVIDER in ("cryptocompare", "cc", "crypto") else "RSS headlines for the asset"
        prompt = {
            "символ": symbol,
            "финансы": {
                "equity_total": equity,
                "available_margin": available_margin,
                "risk_pct": RISK_PCT,
                "configured_leverage": LEVERAGE,
                "min_notional_usdt": MIN_NOTIONAL_USDT,
                "sl_atr_mult": SL_ATR,
                "tp_atr_mult": TP_ATR
            },
            "капитал": equity,
            "индикаторы": current_context,
            "текущая_позиция": position_payload or {"статус": "нет позиции"},
            "открытые_ордера": open_orders,
            "как_создавать_ордеры": [
                "Возвращай массив 'orders', если нужно выставить дополнительные заявки.",
                "Пример: orders=[{\"type\":\"limit\",\"side\":\"sell\",\"amount\":0.001,\"price\":111500,\"reduceOnly\":true,\"note\":\"частичный тейк\"}].",
                "Для частичного закрытия можно указать amountPercent вместо amount; trailing_stop передавай через params (например, {'trailingStop':50}).",
                "Учитывай open_orders (см. поле 'open_orders'): избегай дублирования существующих лимитов/стопов.",
                "Для обновления тейков/стопов используй cancel_orders или replace_orders (сначала укажи id заявки, затем опиши новый ордер)."
            ],
            "сценарий": {
                "алгоритм": [
                    "1) Проверь тренды 30m/4h и RSI.",
                    "2) Если уверенность <70% или сигналы расходятся — запроси needs (higher_tf:<tf>, funding, open_interest, news).",
                    "3) После получения дополнительных данных выбери окончательное действие."
                ],
                "базовый_тренд": {
                    "рекомендуемое_действие": "open" if ema_trend_bias else "skip",
                    "сторона": ema_trend_bias,
                    "описание": (
                        f"Доминирующий тренд {ema_trend_label} по EMA20/EMA50 на 30m."
                        f" На 4h тренд {higher_trend_label}. При их совпадении отдавай предпочтение входу."
                    ),
                    "наклон_ema20": ema_slope
                },
                "контртренд": {
                    "условие": "RSI выходит из экстремума 30m, а ATR падает",
                    "напоминание": "если 4h тренд сильный, уменьши размер и ставь плотный SL"
                },
                "пороговые_значения": {
                    "long": {"rsi_30m": "<=55", "rsi_4h": "<=60"},
                    "short": {"rsi_30m": ">=45", "rsi_4h": ">=40"},
                    "atr": "избегай входа, если текущий ATR выше среднего за 14×1.8"
                },
                "skip": "используй только при конфликте трендов или резком росте ATR/новостях"
            },
            "доступные_данные": {
                "higher_tf": ["higher_tf:4h", "higher_tf:1h", "higher_tf:30m"],
                "funding": "fetchFundingRate",
                "open_interest": "fetchOpenInterestHistory",
                "news": news_desc
            },
            "если_неуверен": {
                "action": "open" if ema_trend_bias else "skip",
                "side": ema_trend_bias,
                "reason": (
                    "следуем доминирующему тренду с умеренным риском после запроса needs"
                    if ema_trend_bias else "недостаточно уверенности — запроси needs"
                ),
                "до_решения": "обязательно запроси дополнительные данные через needs перед финальным выбором",
                "sl_atr": SL_ATR,
                "tp_atr": TP_ATR
            }
        }
        if portfolio_guidance:
            prompt["portfolio_guidance"] = portfolio_guidance
        if extra_context_payload:
            prompt["requested_context"] = extra_context_payload
        if news_payload_payload:
            prompt["news_focus"] = news_payload_payload
        if extra: prompt["доп_контекст"] = extra
        if bias: prompt["режим"] = "чуть более уверенный после допконтекста"
        return json.dumps(prompt, ensure_ascii=False)

    def prepare_messages(stage="initial", extra=None, bias=False):
        nonlocal current_context
        prev_counts = context_counts.copy()
        trimmed = False
        tokens = 0
        attempts = 0
        messages = []
        user_payload = ""
        trim_sources = []
        hard_limit = TOKEN_LIMIT if TOKEN_LIMIT > 0 else None
        soft_limit = TOKEN_SOFT_LIMIT if TOKEN_SOFT_LIMIT > 0 else None
        while True:
            current_context = build_context()
            user_payload = build_prompt(extra, bias)
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_payload}
            ]
            tokens = estimate_tokens(messages, AI_MODEL)
            exceeded_hard = hard_limit is not None and tokens > hard_limit
            exceeded_soft = soft_limit is not None and tokens > soft_limit
            if not exceeded_hard and not exceeded_soft:
                break
            trimmed_step = False
            trim_reason = "hard" if exceeded_hard else "soft"
            if context_counts["30m"] > MIN_CONTEXT_30M:
                new_val = max(MIN_CONTEXT_30M, context_counts["30m"] - CONTEXT_STEP_30M)
                if new_val < context_counts["30m"]:
                    context_counts["30m"] = new_val
                    trimmed_step = True
            if not trimmed_step and context_counts["4h"] > MIN_CONTEXT_4H:
                new_val = max(MIN_CONTEXT_4H, context_counts["4h"] - CONTEXT_STEP_4H)
                if new_val < context_counts["4h"]:
                    context_counts["4h"] = new_val
                    trimmed_step = True
            if not trimmed_step:
                break
            trimmed = True
            trim_sources.append(trim_reason)
            attempts += 1
            if attempts > 50:
                log(f"⚠️ Обрезка контекста не укладывается в лимит ({stage}) для {symbol}", Fore.YELLOW)
                break
        trimmed = trimmed or (context_counts != prev_counts)
        if trimmed:
            reasons_text = "/".join(sorted(set(trim_sources))) if trim_sources else "unknown"
            trim_text = (
                f"✂️ контекст обрезан ({reasons_text}) до "
                f"{context_counts['30m']}×30m и {context_counts['4h']}×4h "
                f"из-за лимита ({tokens} токенов, этап: {stage}) для {symbol}"
            )
            log(trim_text, Fore.MAGENTA)
            send_tg(trim_text)
        if hard_limit and tokens > hard_limit:
            log(f"⚠️ Лимит токенов превышен даже после обрезки ({tokens}>{TOKEN_LIMIT}, этап: {stage}) для {symbol}", Fore.YELLOW)
        return messages, tokens, user_payload

    def ensure_skip_reason(decision_obj):
        action = (decision_obj.get("action") or "").lower()
        if action != "skip":
            return decision_obj
        reason = (decision_obj.get("reason") or "").strip()
        reason_lower = reason.lower()
        if any(key in reason_lower for key in ("ema", "rsi", "atr")):
            return decision_obj
        if df_30m.empty:
            details = "данные индикаторов отсутствуют"
        else:
            last_row = df_30m.iloc[-1]
            parts = []
            ema20_val = last_row.get("ema20")
            ema50_val = last_row.get("ema50")
            rsi_val = last_row.get("rsi")
            atr_val = last_row.get("atr")
            if pd.notna(ema20_val) and pd.notna(ema50_val):
                relation = "ниже" if ema20_val < ema50_val else "выше"
                parts.append(f"EMA20 {ema20_val:.2f} {relation} EMA50 {ema50_val:.2f}")
            elif pd.notna(ema20_val) or pd.notna(ema50_val):
                parts.append(f"EMA данные неполные (EMA20={ema20_val}, EMA50={ema50_val})")
            if pd.notna(rsi_val):
                parts.append(f"RSI14 {rsi_val:.1f}")
            if pd.notna(atr_val):
                parts.append(f"ATR14 {atr_val:.2f}")
            details = "; ".join(parts) if parts else "нет валидных значений EMA/RSI/ATR"
        decision_obj["reason"] = (reason + " — " if reason else "") + f"индикаторы: {details}"
        log(f"ℹ️ Причина skip дополнена индикаторами для {symbol}", Fore.LIGHTBLACK_EX)
        return decision_obj

    decision = (
        dict(initial_decision)
        if isinstance(initial_decision, dict)
        else initial_decision
    )
    confidence_value = None
    confidence_raw = None
    confidence_display = None
    low_confidence = False
    auto_needs_triggered = False
    auto_low_confidence_needs_triggered = False
    needs = []
    if decision is None:
        messages_init, tokens_init, _ = prepare_messages(stage="initial")
        log(f"ℹ️ Токены запроса (initial) для {symbol}: {tokens_init}", Fore.LIGHTBLACK_EX)
        per_cap_init = _current_request_token_cap()
        if per_cap_init and tokens_init > per_cap_init:
            log(f"⚠️ {symbol}: запрос initial превышает кап {per_cap_init} токенов", Fore.YELLOW)
            return {"symbol": symbol, "action": "skip", "reason": "token cap exceeded"}
        if not _ensure_token_budget(tokens_init, AI_MODEL, f"{symbol} initial decision"):
            log(f"⚠️ {symbol}: пропуск initial-запроса из-за лимита токенов", Fore.YELLOW)
            return {"symbol": symbol, "action": "skip", "reason": "token budget exceeded"}
        _log_ai_request(AI_MODEL, tokens_init, f"{symbol} initial decision")

        # --- Первый проход ---
        start_init = time.perf_counter()
        res = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0,
            response_format={"type":"json_object"},
            messages=messages_init
        )
        duration_init = time.perf_counter() - start_init
        log(f"⏱️ OpenAI initial запрос для {symbol}: {duration_init:.2f} c", Fore.LIGHTBLACK_EX)
        _register_ai_usage(AI_MODEL, getattr(res, "usage", None), f"{symbol} initial decision")
        msg = res.choices[0].message.content
        decision = json.loads(msg)
        needs = decision.get("needs", [])
        confidence_raw = decision.get("confidence")
        try:
            confidence_value = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        confidence_display = (
            f"{confidence_value:.3f}"
            if confidence_value is not None
            else (str(confidence_raw) if confidence_raw is not None else None)
        )
        low_confidence = (
            confidence_value is not None
            and confidence_value < AI_CONFIDENCE_THRESHOLD
        )
        action_initial = (decision.get("action") or "").lower()
        if not isinstance(needs, list):
            needs = list(needs) if isinstance(needs, (tuple, set)) else []
            decision["needs"] = needs
        if not needs and action_initial == "skip":
            reason_text = (decision.get("reason") or "").lower()
            keywords_auto_needs = ("запрос", "needs", "дополнитель", "подтвержд")
            if any(word in reason_text for word in keywords_auto_needs):
                auto_needs = ["funding", "open_interest", "news"]
                decision["needs"] = auto_needs
                needs = auto_needs
                auto_needs_triggered = True
        if low_confidence:
            additional_needs = [item for item in LOW_CONFIDENCE_NEEDS if item not in needs]
            if additional_needs:
                needs.extend(additional_needs)
                decision["needs"] = needs
                display = confidence_display or "n/a"
                msg_low = f"?? Low confidence ({display}) for {symbol}: requesting {', '.join(additional_needs)}"
                log(msg_low, Fore.LIGHTBLACK_EX)
                send_tg(msg_low)

        save_json_line(
            AI_REQUESTS_LOG,
            {
                "symbol": symbol,
                "stage": "initial",
                "tokens": tokens_init,
                "token_limit": TOKEN_LIMIT,
                "token_soft_limit": TOKEN_SOFT_LIMIT,
                "duration_sec": round(duration_init, 4),
                "context_counts": context_counts.copy(),
                "context": current_context,
                "needs": needs,
                "auto_needs": auto_needs_triggered,
                "low_confidence": low_confidence,
                "auto_low_confidence": auto_low_confidence_needs_triggered
            }
        )
    else:
        needs = decision.get("needs", []) if isinstance(decision, dict) else []
        confidence_raw = decision.get("confidence") if isinstance(decision, dict) else None
        try:
            confidence_value = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        confidence_display = (
            f"{confidence_value:.3f}"
            if confidence_value is not None
            else (str(confidence_raw) if confidence_raw is not None else None)
        )
        low_confidence = (
            confidence_value is not None
            and confidence_value < AI_CONFIDENCE_THRESHOLD
        )

    if not isinstance(needs, list):
        if isinstance(needs, (tuple, set)):
            needs = list(needs)
        else:
            needs = []
        decision["needs"] = needs

    if low_confidence:
        structured_timeframes: list[str] = []
        for tf in LOW_CONFIDENCE_TIMEFRAMES:
            tf_clean = str(tf).strip()
            if tf_clean and tf_clean not in structured_timeframes:
                structured_timeframes.append(tf_clean)
        if structured_timeframes and NEEDS_MAX_TIMEFRAMES:
            structured_timeframes = structured_timeframes[:NEEDS_MAX_TIMEFRAMES]
        structured_indicators: list[str] = []
        for ind in LOW_CONFIDENCE_INDICATORS:
            ind_clean = str(ind).strip()
            if ind_clean and ind_clean not in structured_indicators:
                structured_indicators.append(ind_clean)
        if structured_indicators and NEEDS_MAX_INDICATORS:
            structured_indicators = structured_indicators[:NEEDS_MAX_INDICATORS]
        existing_auto_struct = any(
            isinstance(item, dict) and item.get("_auto_low_confidence") for item in needs
        )
        if (structured_timeframes or structured_indicators) and not existing_auto_struct:
            structured_need = {"_auto_low_confidence": True}
            if structured_timeframes:
                structured_need["timeframes"] = structured_timeframes
            if structured_indicators:
                structured_need["indicators"] = structured_indicators
            structured_need["limit"] = LOW_CONFIDENCE_SERIALIZE_LIMIT
            needs.insert(0, structured_need)
            decision["needs"] = needs
            auto_low_confidence_needs_triggered = True
            display = confidence_display or "n/a"
            tf_text = ", ".join(structured_timeframes) if structured_timeframes else "none"
            ind_text = ", ".join(structured_indicators) if structured_indicators else "none"
            msg_auto = (
                f"?? Low confidence ({display}) for {symbol}: requesting extra TFs [{tf_text}] "
                f"and indicators [{ind_text}]"
            )
            log(msg_auto, Fore.LIGHTBLACK_EX)
            send_tg(msg_auto)

    # --- Если запрошен контекст ---
    if needs:
        if auto_low_confidence_needs_triggered:
            display = confidence_display or "n/a"
            msg_auto_low = f"?? Low-confidence auto context ({display}) for {symbol}: {needs}"
            log(msg_auto_low, Fore.CYAN)
            send_tg(msg_auto_low)
        elif auto_needs_triggered:
            msg_auto_needs = f"?? Auto-requested context after skip reason for {symbol}: {needs}"
            log(msg_auto_needs, Fore.CYAN)
            send_tg(msg_auto_needs)
        else:
            msg_manual = f"?? Model requested extra context for {symbol}: {needs}"
            log(msg_manual, Fore.CYAN)
            send_tg(msg_manual)
        extra = {}
        needs_followup = []
        indicator_extra = None
        for n in needs:
            if isinstance(n, dict):
                serialize_limit_override = safe_int(
                    n.get("limit") or n.get("depth") or n.get("bars")
                )
                if serialize_limit_override is None and n.get("_auto_low_confidence"):
                    serialize_limit_override = LOW_CONFIDENCE_SERIALIZE_LIMIT
                tf_values: list[str] = []
                indicators_requested: list[str] = []
                if "timeframe" in n and n.get("timeframe"):
                    mapped_tf = normalize_requested_timeframe(n.get("timeframe"), default=TIMEFRAME)
                    if mapped_tf and mapped_tf not in tf_values:
                        tf_values.append(mapped_tf)
                if isinstance(n.get("timeframes"), (list, tuple, set)):
                    for tf in n["timeframes"]:
                        mapped_tf = normalize_requested_timeframe(tf, default=TIMEFRAME)
                        if mapped_tf and mapped_tf not in tf_values:
                            tf_values.append(mapped_tf)
                higher_tf_raw = n.get("higher_tf")
                higher_tf_list: list[str] = []
                if higher_tf_raw:
                    if isinstance(higher_tf_raw, (list, tuple, set)):
                        higher_tf_list = [str(tf) for tf in higher_tf_raw if tf]
                    else:
                        higher_tf_list = [str(higher_tf_raw)]
                if higher_tf_list and NEEDS_MAX_TIMEFRAMES:
                    higher_tf_list = higher_tf_list[:NEEDS_MAX_TIMEFRAMES]
                for tf in higher_tf_list:
                    mapped_tf = normalize_requested_timeframe(tf, default="4h")
                    try:
                        higher_payload = get_higher_tf(exchange, symbol, mapped_tf or "1h")
                    except Exception as exc_ht:
                        dataset.setdefault("errors", []).append(f"needs higher_tf({mapped_tf}): {exc_ht}")
                        continue
                    extra.setdefault("higher_tf", {})[mapped_tf] = higher_payload
                    if mapped_tf not in tf_values:
                        tf_values.append(mapped_tf)
                raw_indicator_fields = []
                for key in ("indicator", "indicators", "indicator_set"):
                    if key in n:
                        raw_indicator_fields.append(n[key])
                shorthand_fields = {
                    k: v for k, v in n.items()
                    if isinstance(k, str) and k.lower() in ("ema", "sma", "rsi", "atr", "stoch")
                }
                for k, v in shorthand_fields.items():
                    raw_indicator_fields.append({"indicator": k, "length": v})
                if not raw_indicator_fields and n.get("type") in ("indicator", "indicators"):
                    raw_indicator_fields.append(n.get("config") or n.get("details"))
                for field in raw_indicator_fields:
                    indicators_requested.extend(_expand_indicator_entries(field))
                indicators_requested = [ind for ind in dict.fromkeys(indicators_requested) if ind]
                if not tf_values:
                    tf_values = [TIMEFRAME]
                if tf_values and NEEDS_MAX_TIMEFRAMES:
                    tf_values = tf_values[:NEEDS_MAX_TIMEFRAMES]
                if indicators_requested and NEEDS_MAX_INDICATORS:
                    indicators_requested = indicators_requested[:NEEDS_MAX_INDICATORS]
                if indicators_requested:
                    indicator_extra = extra.setdefault(
                        "indicator_extra",
                        {"requests": [], "timeframes": {}, "errors": []}
                    )
                    indicator_extra["requests"].append(
                        {
                            "source": n,
                            "timeframes": tf_values,
                            "indicators": indicators_requested
                        }
                    )
                    timeframe_store = extra.setdefault("timeframes", {})
                    for tf in tf_values:
                        try:
                            df_tf = fetch_df(exchange, symbol, tf)
                        except Exception as tf_exc:
                            indicator_extra.setdefault("errors", []).append(f"{tf}: {tf_exc}")
                            continue
                        applied_cols = []
                        for ind_name in indicators_requested:
                            col = _apply_indicator_to_df(df_tf, ind_name)
                            if col:
                                applied_cols.append(col)
                        effective_limit = serialize_limit_override
                        if effective_limit is None and tf in SUPPORT_CONTEXT_TIMEFRAMES:
                            effective_limit = SUPPORT_CONTEXT_LIMIT
                        base_limit = int(max(1, NEEDS_LONG_BARS_LIMIT or NEEDS_SERIALIZE_DEFAULT_LIMIT or 10))
                        bars_limit = base_limit
                        if effective_limit is not None:
                            bars_limit = max(1, min(int(effective_limit), base_limit))
                        elif NEEDS_SERIALIZE_DEFAULT_LIMIT:
                            bars_limit = max(1, min(int(NEEDS_SERIALIZE_DEFAULT_LIMIT), base_limit))
                        tf_payload = {"bars": _serialize_df(df_tf, limit=bars_limit)}
                        tf_payload["limit"] = bars_limit
                        if applied_cols:
                            indicator_values = {}
                            for col in applied_cols:
                                series = df_tf[col].dropna()
                                if series.empty:
                                    indicator_values[col] = None
                                else:
                                    val = series.iloc[-1]
                                    indicator_values[col] = float(val) if isinstance(val, numbers.Number) else val
                            tf_payload["indicators"] = indicator_values
                        timeframe_store.setdefault(tf, tf_payload)
                        indicator_extra["timeframes"][tf] = tf_payload
                elif not higher_tf_list and not any(n.get(flag) for flag in ("funding", "open_interest", "news")):
                    needs_followup.append(n)
                if n.get("funding"):
                    try:
                        funding_payload = get_funding_rate(exchange, symbol)
                        extra["funding"] = funding_payload
                        dataset["funding"] = funding_payload
                    except Exception as exc_fn:
                        dataset.setdefault("errors", []).append(f"needs funding: {exc_fn}")
                if n.get("open_interest"):
                    try:
                        oi_payload = get_open_interest(exchange, symbol)
                        extra["open_interest"] = oi_payload
                        dataset["open_interest"] = oi_payload
                    except Exception as exc_oi:
                        dataset.setdefault("errors", []).append(f"needs open_interest: {exc_oi}")
                if n.get("news"):
                    news_payload = None
                    if news_full and symbol in news_full:
                        news_payload = news_full[symbol]
                    elif news_cache:
                        news_payload = news_cache.get(symbol)
                    if news_payload is None:
                        try:
                            news_payload = get_news(symbol)
                        except Exception as exc_news:
                            dataset.setdefault("errors", []).append(f"needs news: {exc_news}")
                            news_payload = None
                    if news_payload is not None:
                        extra["news"] = news_payload
                        dataset["news"] = news_payload
                continue
            if not isinstance(n, str):
                needs_followup.append(n)
                continue
            if n.startswith("higher_tf"):
                mapped_tf = normalize_requested_timeframe(n, default="4h")
                extra.setdefault("higher_tf", {})[mapped_tf] = get_higher_tf(exchange, symbol, mapped_tf or "4h")
            elif n == "funding":
                extra["funding"] = get_funding_rate(exchange, symbol)
            elif n == "open_interest":
                extra["open_interest"] = get_open_interest(exchange, symbol)
            elif n == "news":
                extra["news"] = get_news(symbol)

        stats_report = []
        for key,val in extra.items():
            if isinstance(val, list):
                stats_report.append(f"{key}: {len(val)} записей")
            elif isinstance(val, dict) and val:
                if key == "higher_tf":
                    inner = {k: len(v) for k,v in val.items()}
                    stats_report.append(f"{key}: " + ", ".join(f"{tf}={cnt}" for tf,cnt in inner.items()))
                elif key == "indicator_extra":
                    tf_entries = []
                    for tf_name, payload in (val.get("timeframes") or {}).items():
                        indicators = payload.get("indicators") or {}
                        tf_entries.append(
                            f"{tf_name}: индикаторы {', '.join(indicators.keys()) or 'нет'}"
                        )
                    if tf_entries:
                        stats_report.append(f"{key}: " + "; ".join(tf_entries))
                    errors = val.get("errors") or []
                    if errors:
                        stats_report.append(f"{key}_errors: " + "; ".join(errors))
                else:
                    summary = val.get("summary")
                    if summary:
                        stats_report.append(f"{key}: {summary}")
                    else:
                        stats_report.append(f"{key}: найдено")
            else:
                stats_report.append(f"{key}: нет данных")

        log("✅ Контекст собран: " + ", ".join(stats_report), Fore.LIGHTBLACK_EX)
        send_tg("✅ Контекст собран для " + symbol + ":\n" + "\n".join(stats_report))

        if False:
            log(f"⚠️ {symbol}: пропуск допконтекста из-за достигнутого лимита токенов", Fore.YELLOW)
            decision["needs_followup"] = needs
            decision.pop("needs", None)
            return ensure_skip_reason(decision)
        bias_flag = bool(AI_AFTER_NEEDS_BIAS)
        messages_extra, tokens_extra, _ = prepare_messages(stage="extra", extra=extra, bias=bias_flag)
        log(f"ℹ️ Токены запроса (extra) для {symbol}: {tokens_extra}", Fore.LIGHTBLACK_EX)
        per_cap_extra = _current_request_token_cap()
        if per_cap_extra and tokens_extra > per_cap_extra:
            log(f"⚠️ {symbol}: запрос extra превышает кап {per_cap_extra} токенов", Fore.YELLOW)
            decision["needs_followup"] = needs
            decision.pop("needs", None)
            return ensure_skip_reason(decision)
        if not _ensure_token_budget(tokens_extra, AI_MODEL, f"{symbol} extra decision"):
            log(f"⚠️ {symbol}: пропуск extra-запроса из-за лимита токенов", Fore.YELLOW)
            decision["needs_followup"] = needs
            return ensure_skip_reason(decision)
        _log_ai_request(AI_MODEL, tokens_extra, f"{symbol} extra decision")
        start_extra = time.perf_counter()
        res2 = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0,
            response_format={"type":"json_object"},
            messages=messages_extra
        )
        duration_extra = time.perf_counter() - start_extra
        log(f"⏱️ OpenAI extra запрос для {symbol}: {duration_extra:.2f} c", Fore.LIGHTBLACK_EX)
        _register_ai_usage(AI_MODEL, getattr(res2, "usage", None), f"{symbol} extra decision")
        msg2 = res2.choices[0].message.content
        decision = json.loads(msg2)
        needs_followup = decision.get("needs", [])
        if needs_followup:
            log(f"ℹ️ После допконтекста модель все ещё запрашивает {needs_followup} для {symbol}", Fore.LIGHTBLACK_EX)
            decision["needs_followup"] = needs_followup
            decision.pop("needs", None)
        save_json_line(
            AI_REQUESTS_LOG,
            {
                "symbol": symbol,
                "stage": "extra",
                "tokens": tokens_extra,
                "token_limit": TOKEN_LIMIT,
                "token_soft_limit": TOKEN_SOFT_LIMIT,
                "duration_sec": round(duration_extra, 4),
                "context_counts": context_counts.copy(),
                "context": current_context,
                "extra": extra,
                "bias": bias_flag,
                "needs_followup": needs_followup
            }
        )
        log(f"🧩 Второй проход завершён для {symbol}", Fore.CYAN)
        send_tg(f"🧩 Второй проход завершён для {symbol}")

    decision = ensure_skip_reason(decision)
    return decision

# --- Основная логика ---
def run_cycle():
    _sync_with_remote()
    _write_runtime_status(None, None, "running")
    refresh_settings()
    _init_ai_cycle_usage()
    metadata_state = maybe_refresh_metadata()
    if isinstance(metadata_state, dict) and metadata_state.get("reload_required"):
        new_hash = metadata_state.get("current_hash")
        short_hash = (new_hash or "")[:8] if isinstance(new_hash, str) else "?"
        reason = f"♻️ Обнаружен новый коммит {short_hash}, перезапускаем бота для загрузки обновлений."
        _restart_with_latest_code(reason)
    ex = init_exchange()
    ex.load_markets()
    markets_set = set(ex.symbols or [])
    if not markets_set:
        markets_set = set((getattr(ex, "markets", {}) or {}).keys())

    symbol_alias_hits: dict[str, str] = {}
    missing_symbols: set[str] = set()

    def normalize_symbol(symbol: str | None, *, record_missing: bool = True) -> Optional[str]:
        if not symbol:
            return None
        sym = str(symbol).strip()
        if not sym:
            return None
        if sym in markets_set:
            return sym
        alias_target = SYMBOL_ALIASES.get(sym)
        if alias_target and alias_target in markets_set:
            symbol_alias_hits[sym] = alias_target
            return alias_target
        if record_missing:
            missing_symbols.add(sym)
        return None

    ensure_position_mode(ex)
    positions_map, open_positions = fetch_positions_snapshot(ex)
    base_max_positions = max(0, MAX_OPEN_POSITIONS or 0)
    max_positions_limit = base_max_positions
    if max_positions_limit > 0 and open_positions is None:
        log("⚠️ Не удалось определить количество открытых позиций — лимит по позициям отключён на этот цикл", Fore.YELLOW)
        open_positions = None
    equity, available_margin, _ = fetch_usdt_equity(ex)
    if equity <= 0:
        equity = 64.0
    if available_margin <= 0:
        available_margin = equity
    session_dt = _current_log_time()
    session_stamp = session_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    session_separator = "=" * 56
    start_banner = f"{session_separator} START SESSION {session_stamp} {session_separator}"
    log(start_banner, Fore.MAGENTA)
    send_tg(f"{session_separator}\nSTART SESSION {session_stamp}\n{session_separator}")
    source_label = os.getenv("BYBITBOT_SOURCE_LABEL")
    source_ref = os.getenv("BYBITBOT_SOURCE_REF")
    source_context = os.getenv("BYBITBOT_FALLBACK_CONTEXT")
    if source_label and source_ref:
        git_line = f"[GIT] {source_ref} — {source_label}"
        if source_context:
            git_line += f" ({source_context})"
        log(git_line, Fore.LIGHTBLACK_EX)
        send_tg(git_line)
    else:
        commit_hash, commit_message, commit_ts = get_current_commit_info()
        if commit_hash:
            short_hash = commit_hash[:8]
            message_text = commit_message or "no commit message"
            timestamp_text = commit_ts or "timestamp unavailable"
            git_line = f"[GIT] {short_hash} @ {timestamp_text} - {message_text} (version {BOT_VERSION})"
            log(git_line, Fore.LIGHTBLACK_EX)
            send_tg(git_line)
    last_equity = equity
    last_available_margin = available_margin
    log(f"🚀 Бот v{BOT_VERSION} запущен. Баланс: {equity:.2f} USDT, доступно {available_margin:.2f} USDT", Fore.GREEN)
    send_tg(f"🚀 Бот запущен. Баланс: {equity:.2f} USDT, доступно {available_margin:.2f} USDT")

    position_symbols: set[str] = set()
    for sym_pos, payload in positions_map.items():
        if not payload:
            continue
        try:
            amt = float(payload.get("amount") or 0)
        except (TypeError, ValueError):
            amt = 0.0
        if math.isfinite(amt) and abs(amt) > 0:
            position_symbols.add(sym_pos)

    normalized_pair_list: list[str] = []
    for raw_pair in PAIR_LIST:
        resolved_pair = normalize_symbol(raw_pair)
        if resolved_pair and resolved_pair not in normalized_pair_list:
            normalized_pair_list.append(resolved_pair)

    candidate_pairs_set: set[str] = set()

    def add_candidates(values, *, record_missing: bool = True):
        for value in values:
            resolved = normalize_symbol(value, record_missing=record_missing)
            if resolved:
                candidate_pairs_set.add(resolved)

    add_candidates(PAIR_LIST)
    add_candidates(BASE_PAIR_CANDIDATES)
    add_candidates(position_symbols, record_missing=False)

    universe_cache = load_universe_cache()
    news_headlines = _build_news_digest(sorted(candidate_pairs_set))
    selection_result = None
    universe_state = dict(universe_cache)
    news_requests: list[Any] = []
    updated_universe = ai_update_universe(
        exchange=ex,
        symbols=sorted(candidate_pairs_set),
        positions_map=positions_map,
        equity=equity,
        available_margin=available_margin,
        universe_cache=universe_cache,
        news_digest=news_headlines,
    )
    if updated_universe:
        selection_result, universe_state, news_requests = updated_universe
        universe_state = universe_state or {}
        universe_state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_universe_cache(universe_state)
    if universe_state.get("pairs"):
        for pair in universe_state.get("pairs", []):
            resolved_pair = normalize_symbol(pair, record_missing=False)
            if resolved_pair:
                candidate_pairs_set.add(resolved_pair)
    news_full_cache: dict[str, dict] = {}
    for req in news_requests or []:
        if isinstance(req, dict):
            sym_request = req.get("symbol")
        else:
            sym_request = req
        if not sym_request:
            continue
        try:
            news_full_cache[sym_request] = get_news(sym_request)
        except Exception as exc_news:
            log(f"[WARN] Failed to fetch detailed news for {sym_request}: {exc_news}", Fore.YELLOW)
    if selection_result is None:
        selection_result = None

    if selection_result:
        limit_candidates: list[Any] = []
        limits_block = selection_result.get("limits")
        if isinstance(limits_block, dict):
            limit_candidates.extend(
                limits_block.get(key)
                for key in (
                    "max_positions",
                    "maxPositions",
                    "max_open_positions",
                    "maxOpenPositions",
                )
            )
        limit_candidates.extend(
            selection_result.get(key)
            for key in (
                "max_positions",
                "maxPositions",
                "max_open_positions",
                "maxOpenPositions",
            )
        )
        for candidate in limit_candidates:
            val = safe_int(candidate)
            if val and val > 0:
                max_positions_limit = val
                break
        if max_positions_limit != base_max_positions and max_positions_limit > 0:
            log(f"[INFO] Model requested max open positions: {max_positions_limit}", Fore.LIGHTBLACK_EX)

    open_orders_prefetch: dict[str, list] = {}
    order_symbols: set[str] = set()
    order_symbols_non_reduce: set[str] = set()

    prefetch_candidates = list(candidate_pairs_set)
    for sym_candidate in prefetch_candidates[:PAIR_PREFETCH_LIMIT]:
        try:
            orders_snapshot = fetch_open_orders_for_symbol(ex, sym_candidate)
        except Exception as orders_exc:
            log(f"[WARN] {sym_candidate}: failed to fetch open orders: {orders_exc}", Fore.YELLOW)
            orders_snapshot = []
        open_orders_prefetch[sym_candidate] = orders_snapshot
        if orders_snapshot:
            order_symbols.add(sym_candidate)
            if any(isinstance(o, dict) and not _is_reduce_only(o) for o in orders_snapshot):
                order_symbols_non_reduce.add(sym_candidate)

    add_candidates(order_symbols, record_missing=False)
    news_pairs = _collect_news_pairs()
    news_priority: list[str] = []
    if news_pairs:
        for raw_pair in news_pairs:
            resolved_pair = normalize_symbol(raw_pair)
            if resolved_pair and resolved_pair not in news_priority:
                news_priority.append(resolved_pair)
                candidate_pairs_set.add(resolved_pair)

    for sym_candidate in sorted(candidate_pairs_set):
        if sym_candidate in open_orders_prefetch:
            continue
        if len(open_orders_prefetch) >= PAIR_PREFETCH_LIMIT:
            break
        try:
            orders_snapshot = fetch_open_orders_for_symbol(ex, sym_candidate)
        except Exception as orders_exc:
            log(f"[WARN] {sym_candidate}: failed to fetch open orders: {orders_exc}", Fore.YELLOW)
            orders_snapshot = []
        open_orders_prefetch[sym_candidate] = orders_snapshot
        if orders_snapshot:
            order_symbols.add(sym_candidate)

    def _append_unique(target_list: list[str], values, seen: set[str]):
        for val in values:
            if not val or val in seen:
                continue
            target_list.append(val)
            seen.add(val)
            if len(target_list) >= PAIR_CANDIDATE_LIMIT:
                break
        return target_list

    available_pairs: list[str] = []
    selected_symbols: list[str] = []
    seen_available: set[str] = set()

    news_sorted = sorted(news_priority)
    if max_positions_limit > 0 and open_positions is not None and open_positions >= max_positions_limit:
        _append_unique(available_pairs, sorted(position_symbols), seen_available)
        _append_unique(available_pairs, sorted(order_symbols), seen_available)
    else:
        _append_unique(available_pairs, news_sorted, seen_available)
        _append_unique(available_pairs, sorted(position_symbols), seen_available)
        _append_unique(available_pairs, sorted(order_symbols), seen_available)
        remaining_pairs = [p for p in sorted(candidate_pairs_set) if p not in seen_available]
        _append_unique(available_pairs, remaining_pairs, seen_available)

    selection_pairs = universe_state.get("pairs") or []
    selection_pairs_normalized: list[str] = []
    for raw_pair in selection_pairs:
        if not raw_pair:
            continue
        resolved_pair = normalize_symbol(raw_pair, record_missing=False)
        selection_pairs_normalized.append(resolved_pair or raw_pair)
    selection_pairs = [p for p in selection_pairs_normalized if p]

    if not available_pairs:
        available_pairs = normalized_pair_list[:PAIR_CANDIDATE_LIMIT]
        if not available_pairs:
            available_pairs = list(sorted(markets_set))[:PAIR_CANDIDATE_LIMIT]
    if selection_pairs:
        prioritized = [p for p in selection_pairs if p in available_pairs]
        prioritized += [p for p in available_pairs if p not in prioritized]
        available_pairs = prioritized

    if missing_symbols:
        missing_desc = ', '.join(sorted(missing_symbols))
        log(f"[WARN] Removed pairs not listed on Bybit: {missing_desc}", Fore.YELLOW)
        send_tg(f"[WARN] Pairs missing on Bybit: {missing_desc}")
        missing_symbols.clear()
    if symbol_alias_hits:
        alias_desc = ', '.join(f"{src}->{dst}" for src, dst in sorted(symbol_alias_hits.items()))
        log(f"[INFO] Using alias tickers: {alias_desc}", Fore.LIGHTBLACK_EX)
        symbol_alias_hits.clear()

    if len(available_pairs) > MAX_SYMBOLS_PER_CYCLE:
        available_pairs = available_pairs[:MAX_SYMBOLS_PER_CYCLE]
    log("[INFO] Candidates for analysis: " + ', '.join(available_pairs), Fore.LIGHTBLACK_EX)

    open_orders_cache = dict(open_orders_prefetch)

    if not selection_pairs:
        selection_pairs = available_pairs
    analysis_pool_count = len(selection_pairs) if selection_pairs else len(available_pairs)
    update_ai_model_for_analysis(analysis_pool_count)

    selection = selection_result
    global_timeframes = []
    global_indicators = []
    news_cache = {}
    target_map = {}
    selected_symbols = []
    selection_missing_symbols: list[str] = []
    selection_next_run = None
    selection_next_time = None

    if selection:
        global_timeframes = selection.get("global_timeframes") or []
        global_indicators = selection.get("global_indicators") or []
        news_cache = dict(selection.get("_news_digest") or {})
        if news_full_cache:
            news_cache.update(news_full_cache)
        normalized_targets: list[dict[str, Any]] = []
        for target in selection.get("targets") or []:
            raw_symbol = None
            target_payload = {}
            if isinstance(target, str):
                raw_symbol = target.strip()
            elif isinstance(target, dict):
                target_payload = dict(target)
                symbol_candidates = (
                    target_payload.get("symbol"),
                    target_payload.get("pair"),
                    target_payload.get("ticker"),
                )
                for candidate in symbol_candidates:
                    if isinstance(candidate, str) and candidate.strip():
                        raw_symbol = candidate.strip()
                        break
            else:
                log(f"[WARN] Ignoring unexpected target payload of type {type(target)!r}", Fore.YELLOW)
                continue
            if not raw_symbol:
                continue
            sym_sel = normalize_symbol(raw_symbol)
            if not sym_sel:
                selection_missing_symbols.append(raw_symbol)
                continue
            target_copy = dict(target_payload)
            target_copy["symbol"] = sym_sel
            target_copy.setdefault("raw_symbol", raw_symbol)
            target_map[sym_sel] = target_copy
            if sym_sel not in selected_symbols:
                selected_symbols.append(sym_sel)
            normalized_targets.append(target_copy)
        if isinstance(selection, dict):
            selection = dict(selection)
            selection["targets"] = normalized_targets
        selection_reason = selection.get("reason")
        if selection_reason:
            log(f"[INFO] Portfolio rationale: {selection_reason}", Fore.CYAN)
            send_tg(f"[INFO] Portfolio analysis: {selection_reason}")
        confidence = selection.get("confidence")
        if confidence is not None:
            conf_text = str(confidence)
            try:
                conf_val = float(confidence)
            except (TypeError, ValueError):
                conf_val = None
            else:
                conf_text = f"{conf_val:.3f}"
            confidence_msg = f"[AI] Portfolio selection confidence: {conf_text}"
            log(confidence_msg, Fore.LIGHTBLACK_EX)
            send_tg(confidence_msg)
    if selection:
        raw_next_run = selection.get("next_run_minutes")
        if raw_next_run is not None:
            try:
                selection_next_run = float(raw_next_run)
            except (TypeError, ValueError):
                selection_next_run = None
        selection_next_time = selection.get("next_run_time")

    decisions_map: dict[str, dict] = {}
    bundle = None
    trade_plan = None
    if selection:
        extra_symbols_candidates: list[str] = []
        for sym_extra in sorted(order_symbols_non_reduce):
            normalized_extra = normalize_symbol(sym_extra, record_missing=False)
            if not normalized_extra and sym_extra:
                sym_key = sym_extra.replace("/", "").replace(":", "").upper()
                mapped = TICKER_TO_SYMBOL.get(sym_key) or TICKER_TO_SYMBOL.get(sym_key.rstrip("USDT"))
                if mapped:
                    normalized_extra = normalize_symbol(mapped, record_missing=False) or mapped
            if normalized_extra and normalized_extra not in extra_symbols_candidates:
                extra_symbols_candidates.append(normalized_extra)
        bundle, bundle_orders = build_portfolio_bundle(
            ex,
            selection,
            positions_map,
            news_cache=news_cache,
            extra_symbols=extra_symbols_candidates,
        )
        bundle_meta = bundle.setdefault("meta", {})
        selection_needs_payload = (selection or {}).get("needs") or []
        if selection_needs_payload:
            bundle = augment_bundle_with_needs(
                ex,
                bundle,
                selection_needs_payload,
                news_cache=news_cache,
                news_full=news_full_cache,
            )
            bundle_meta.setdefault("selection_needs", selection_needs_payload)
        bundle_meta = bundle.setdefault("meta", bundle_meta if isinstance(bundle_meta, dict) else {})
        summary_timeframes = list(SUMMARY_TIMEFRAME_SHORTLIST or [])
        if NEEDS_MAX_TIMEFRAMES and summary_timeframes:
            summary_timeframes = summary_timeframes[:NEEDS_MAX_TIMEFRAMES]
        summary_indicators = list(SUMMARY_INDICATOR_SHORTLIST or [])
        if NEEDS_MAX_INDICATORS and summary_indicators:
            summary_indicators = summary_indicators[:NEEDS_MAX_INDICATORS]
        if summary_timeframes:
            bundle_meta["selection_timeframes"] = summary_timeframes
        if summary_indicators:
            bundle_meta["selection_indicators"] = summary_indicators
        if bundle_orders:
            open_orders_cache.update(bundle_orders)
        bundle_meta["active_symbols"] = sorted(position_symbols)
        bundle_meta["pending_symbols"] = sorted(order_symbols_non_reduce)
        trade_plan = ai_plan_trades(
            ex,
            bundle,
            equity,
            available_margin,
            positions_snapshot=positions_map,
            pending_orders=open_orders_prefetch,
            stage="initial",
        )
        if trade_plan:
            trade_next_minutes = trade_plan.get("next_run_minutes")
            if trade_next_minutes is not None:
                try:
                    selection_next_run = float(trade_next_minutes)
                except (TypeError, ValueError):
                    log("[WARN] Invalid next_run_minutes from trade plan.", Fore.YELLOW)
            trade_next_time = trade_plan.get("next_run_time")
            if trade_next_time:
                selection_next_time = trade_next_time
        if trade_plan:
            for decision in trade_plan.get("decisions") or []:
                sym_raw = decision.get("symbol")
                sym_dec = sym_raw.strip() if isinstance(sym_raw, str) else ""
                canonical_key = _canonical_decision_symbol(sym_dec) or sym_dec
                if canonical_key:
                    decisions_map[canonical_key] = decision
            additional_missing = trade_plan.get("missing_symbols")
            if additional_missing:
                selection_missing_symbols.extend(list(additional_missing))
    for sym_key in decisions_map.keys():
        if sym_key not in selected_symbols:
            selected_symbols.append(sym_key)
    if selection_missing_symbols:
        missing_from_ai = ", ".join(sorted(set(selection_missing_symbols)))
        log(f"[WARN] Model symbols missing on Bybit: {missing_from_ai}", Fore.YELLOW)
        send_tg(f"[WARN] Model symbols missing on Bybit: {missing_from_ai}")

    symbols_sequence: list[str] = []
    seen_symbols: set[str] = set()
    for sym_sel in selected_symbols:
        if sym_sel and sym_sel not in seen_symbols:
            symbols_sequence.append(sym_sel)
            seen_symbols.add(sym_sel)

    for sym_candidate in sorted(position_symbols):
        if sym_candidate not in seen_symbols:
            symbols_sequence.append(sym_candidate)
            seen_symbols.add(sym_candidate)

    for sym_candidate in sorted(order_symbols):
        if sym_candidate not in seen_symbols:
            symbols_sequence.append(sym_candidate)
            seen_symbols.add(sym_candidate)

    for sym_candidate in available_pairs:
        if sym_candidate not in seen_symbols:
            symbols_sequence.append(sym_candidate)
            seen_symbols.add(sym_candidate)

    priority_sequence = []
    seen_priority: set[str] = set()
    for sym_order in symbols_sequence:
        if sym_order in position_symbols and sym_order not in seen_priority:
            priority_sequence.append(sym_order)
            seen_priority.add(sym_order)
    for sym_order in symbols_sequence:
        if sym_order in order_symbols_non_reduce and sym_order not in seen_priority:
            priority_sequence.append(sym_order)
            seen_priority.add(sym_order)
    for sym_order in symbols_sequence:
        if sym_order not in seen_priority:
            priority_sequence.append(sym_order)
            seen_priority.add(sym_order)
    symbols_sequence = priority_sequence
    if len(symbols_sequence) > MAX_SYMBOLS_PER_CYCLE:
        symbols_sequence = symbols_sequence[:MAX_SYMBOLS_PER_CYCLE]
    if not symbols_sequence:
        symbols_sequence = available_pairs or list(PAIR_LIST)

    decisions_total = 0
    counts = {"open":0,"close":0,"skip":0}
    decisions_details: list[str] = []
    eligible_flat_symbols = 0
    skipped_plan_omitted_symbols = 0
    skipped_plan_unavailable_symbols = 0
    flat_skip_symbols: list[str] = []
    flat_unavailable_symbols: list[str] = []

    trade_plan_failed = trade_plan is None
    for i,sym in enumerate(symbols_sequence,1):
        if AI_HARD_STOP_BUDGET and AI_TOKEN_USAGE_TOTAL >= AI_HARD_STOP_BUDGET:
            log(f"⚠️ Достигнут лимит {AI_HARD_STOP_BUDGET} токенов — дальнейший анализ остановлен", Fore.YELLOW)
            break
        log(f"[{i}/{len(symbols_sequence)}] {sym}", Fore.LIGHTBLUE_EX)
        try:
            equity, available_margin, _ = fetch_usdt_equity(ex)
            if equity > 0:
                last_equity = equity
            else:
                equity = last_equity
            if available_margin > 0:
                last_available_margin = available_margin
            else:
                available_margin = last_available_margin
            symbol_meta = dict(target_map.get(sym, {}) or {})
            base_timeframes = list(SUMMARY_TIMEFRAME_SHORTLIST or []) or [TIMEFRAME, "4h"]
            requested_timeframes = list(dict.fromkeys(base_timeframes))
            if NEEDS_MAX_TIMEFRAMES and requested_timeframes:
                requested_timeframes = requested_timeframes[:NEEDS_MAX_TIMEFRAMES]
            base_indicators = list(SUMMARY_INDICATOR_SHORTLIST or [])
            if not base_indicators:
                base_indicators = ["ema20", "ema50", "vol", "rsi14", "macd", "atr14"]
            requested_indicators = list(dict.fromkeys(base_indicators))
            if NEEDS_MAX_INDICATORS and requested_indicators:
                requested_indicators = requested_indicators[:NEEDS_MAX_INDICATORS]
            timeframe_dfs = {}
            for tf in requested_timeframes:
                try:
                    tf_df = fetch_df(ex, sym, tf)
                except Exception as exc_fetch:
                    log(f"?? не удалось получить {tf} для {sym}: {exc_fetch}", Fore.YELLOW)
                    continue
                for ind_name in requested_indicators:
                    _apply_indicator_to_df(tf_df, ind_name)
                timeframe_dfs[tf] = tf_df
            primary_tf = TIMEFRAME if TIMEFRAME in timeframe_dfs else (requested_timeframes[0] if requested_timeframes else TIMEFRAME)
            if primary_tf not in timeframe_dfs:
                try:
                    tf_df = fetch_df(ex, sym, primary_tf)
                    timeframe_dfs[primary_tf] = tf_df
                except Exception as exc_fetch:
                    log(f"?? не удалось получить базовый таймфрейм {primary_tf} для {sym}: {exc_fetch}", Fore.RED)
                    continue
            df = timeframe_dfs[primary_tf].copy()
            timeframes_payload = {}
            for tf, tf_df in timeframe_dfs.items():
                if tf == primary_tf:
                    effective_limit = max(NEEDS_SERIALIZE_DEFAULT_LIMIT, DEFAULT_CONTEXT_30M)
                elif tf in SUPPORT_CONTEXT_TIMEFRAMES:
                    effective_limit = SUPPORT_CONTEXT_LIMIT
                else:
                    effective_limit = NEEDS_SERIALIZE_DEFAULT_LIMIT
                base_limit = int(max(1, NEEDS_LONG_BARS_LIMIT or NEEDS_SERIALIZE_DEFAULT_LIMIT or 10))
                bars_limit = base_limit
                if effective_limit:
                    bars_limit = max(1, min(int(effective_limit), base_limit))
                payload = {"bars": _serialize_df(tf_df, limit=bars_limit)}
                payload["limit"] = bars_limit
                timeframes_payload[tf] = payload
            extra_serialized = {
                "timeframes": timeframes_payload,
                "indicators": requested_indicators,
                "primary": primary_tf,
            }
            if symbol_meta.get("notional_pct") is not None:
                extra_serialized["recommended_notional_pct"] = symbol_meta.get("notional_pct")
            news_payload_symbol = news_cache.get(sym) if news_cache else None
            if not news_payload_symbol:
                try:
                    news_payload_symbol = get_news(sym)
                except Exception as news_exc:
                    log(f"⚠️ Не удалось получить новости для {sym}: {news_exc}", Fore.YELLOW)
                    news_payload_symbol = None
            current_position = positions_map.get(sym)
            initial_position_amount = safe_float(
                (current_position or {}).get("amount")
                or (current_position or {}).get("contracts")
            )
            if initial_position_amount is None or not math.isfinite(initial_position_amount):
                initial_position_amount = 0.0
            has_position = abs(initial_position_amount) > 0
            open_orders_symbol = open_orders_prefetch.get(sym)
            sym_confidence_text: str | None = None
            sym_confidence_value: float | None = None
            sym_confidence_tag: str | None = None
            if open_orders_symbol is None:
                try:
                    open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)
                except Exception as fetch_exc:
                    log(f"⚠️ Не удалось получить открытые ордера для {sym}: {fetch_exc}", Fore.YELLOW)
                    open_orders_symbol = []
                open_orders_prefetch[sym] = open_orders_symbol
            open_orders_symbol = _cleanup_excess_non_reduce_limits(
                ex,
                sym,
                open_orders_symbol,
                (current_position or {}).get("side"),
                MAX_NON_REDUCE_LIMITS_PER_SIDE,
            )
            open_orders_prefetch[sym] = open_orders_symbol
            initial_protection_orders = _extract_protection_orders(open_orders_symbol)
            initial_protection_signature = _protection_orders_signature(open_orders_symbol)
            canonical_lookup = _canonical_decision_symbol(sym)
            preloaded_decision = decisions_map.get(canonical_lookup)
            initial_payload = dict(preloaded_decision) if isinstance(preloaded_decision, dict) else None
            if initial_payload:
                initial_payload["symbol"] = sym

            dec = ai_decision(
                sym,
                df,
                equity,
                available_margin,
                ex,
                current_position=current_position,
                open_orders=open_orders_symbol,
                extra_context=extra_serialized,
                target_meta=symbol_meta,
                news_payload=news_payload_symbol,
                initial_decision=initial_payload,
            )

            if not dec:
                if initial_payload:
                    dec = initial_payload
                else:
                    default_action = "hold" if has_position else "skip"
                    exposure_notes = []
                    if has_position:
                        exposure_notes.append(f"open amount {initial_position_amount:.4f}")
                    pending_count = len(open_orders_symbol or [])
                    if pending_count:
                        exposure_notes.append(f"{pending_count} pending order(s)")
                    if not exposure_notes:
                        exposure_notes.append("no active exposure")
                    if trade_plan:
                        base_reason = "trade plan omitted symbol"
                    else:
                        base_reason = "trade plan unavailable"
                    detail_text = "; ".join(exposure_notes)
                    default_reason = f"{base_reason}; {detail_text}"
                    dec = {
                        "symbol": sym,
                        "action": default_action,
                        "reason": default_reason,
                    }
            decision_confidence_raw = dec.get("confidence")
            if decision_confidence_raw is not None:
                try:
                    sym_confidence_value = float(decision_confidence_raw)
                    sym_confidence_text = f"{sym_confidence_value:.3f}"
                except (TypeError, ValueError):
                    sym_confidence_value = None
                    sym_confidence_text = str(decision_confidence_raw)
                if sym_confidence_value is not None:
                    if sym_confidence_value >= 0.8:
                        sym_confidence_tag = "HIGH"
                    elif sym_confidence_value >= AI_CONFIDENCE_THRESHOLD:
                        sym_confidence_tag = "MEDIUM"
                    else:
                        sym_confidence_tag = "LOW"
                else:
                    sym_confidence_tag = "UNKNOWN"
                tag_display = f" ({sym_confidence_tag})" if sym_confidence_tag else ""
                confidence_msg = f"[AI] {sym} confidence: {sym_confidence_text}{tag_display}"
                log(confidence_msg, Fore.LIGHTBLACK_EX)
                send_tg(confidence_msg)
            else:
                sym_confidence_tag = None
            low_confidence_flag = (
                sym_confidence_value is not None
                and sym_confidence_value < AI_CONFIDENCE_THRESHOLD
            )
            needs_list = dec.get("needs")
            if not isinstance(needs_list, list):
                needs_list = []
                dec["needs"] = needs_list
            if low_confidence_flag:
                for item in LOW_CONFIDENCE_NEEDS:
                    if item not in needs_list:
                        needs_list.append(item)
            if symbol_meta.get("notional_pct") is not None and dec.get("notional_pct") is None:
                dec["notional_pct"] = symbol_meta.get("notional_pct")
            symbol_leverage = _resolve_symbol_leverage(dec, symbol_meta, current_position)
            dec["leverage"] = symbol_leverage
            _set_symbol_leverage(ex, sym, symbol_leverage, current_position)
            save_json_line(
                AI_LOG_FILE,
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "symbol": sym,
                    "decision": dec,
                },
            )
            action = (dec.get("action") or "skip").lower()
            side = dec.get("side") or ""
            reason = dec.get("reason") or ""
            if not has_position:
                eligible_flat_symbols += 1
                reason_lower = reason.lower()
                if action == "skip":
                    if (
                        "trade plan omitted symbol" in reason_lower
                        and "no active exposure" in reason_lower
                    ):
                        skipped_plan_omitted_symbols += 1
                        flat_skip_symbols.append(sym)
                    if "trade plan unavailable" in reason_lower:
                        skipped_plan_unavailable_symbols += 1
                        flat_unavailable_symbols.append(sym)
            counts[action] = counts.get(action,0)+1
            decisions_total += 1
            side_text = side.lower()
            detail_entry: str | None = None
            orders_activity = False

            extra_orders_raw = dec.get("orders") or dec.get("adjustments") or dec.get("extra_orders") or []
            if isinstance(extra_orders_raw, dict):
                item = dict(extra_orders_raw)
                if action in ("manage", "hold") and not _is_truthy_flag(item.get("reduceOnly")):
                    item["reduceOnly"] = True
                extra_orders = [item]
            elif isinstance(extra_orders_raw, list):
                extra_orders = []
                for item in extra_orders_raw:
                    if not isinstance(item, dict):
                        continue
                    item_copy = dict(item)
                    if action in ("manage", "hold") and not _is_truthy_flag(item_copy.get("reduceOnly")):
                        item_copy["reduceOnly"] = True
                    extra_orders.append(item_copy)
            else:
                extra_orders = []

            if extra_orders:
                filtered_orders: list[dict] = []
                dropped_orders: list[str] = []
                position_side_label = (current_position or {}).get("side")
                if position_side_label:
                    position_side = position_side_label.lower()
                elif initial_position_amount > 0:
                    position_side = "long"
                elif initial_position_amount < 0:
                    position_side = "short"
                else:
                    position_side = ""
                for order in extra_orders:
                    if not isinstance(order, dict):
                        continue
                    reduce_only_flag = _is_truthy_flag(order.get("reduceOnly"))
                    order_side = (order.get("side") or "").lower()
                    allow_order = True
                    if not reduce_only_flag:
                        allow_order = False
                        if action == "open" and not has_position:
                            allow_order = True
                        elif has_position and position_side:
                            same_direction = (
                                (position_side in ("long", "buy") and order_side == "buy")
                                or (position_side in ("short", "sell") and order_side == "sell")
                            )
                            if not same_direction:
                                allow_order = True
                    if allow_order:
                        filtered_orders.append(order)
                    else:
                        summary = f"{order_side.upper()} {order.get('type') or 'order'}"
                        if order.get("price") is not None:
                            summary += f" @{order.get('price')}"
                        dropped_orders.append(summary)
                if dropped_orders:
                    log(
                        f"[WARN] Skipping non-reduce extra orders for {sym}: {', '.join(dropped_orders)}",
                        Fore.YELLOW,
                    )
                extra_orders = filtered_orders

            def normalize_order_ids(value):
                if value is None:
                    return []
                if isinstance(value, (list, tuple, set)):
                    items = list(value)
                else:
                    items = [value]
                result = []
                for item in items:
                    if item in (None, ""):
                        continue
                    if isinstance(item, dict):
                        oid = (
                            item.get("id")
                            or item.get("orderId")
                            or item.get("order_id")
                            or item.get("cancel")
                            or item.get("old")
                        )
                        if oid:
                            result.append(str(oid))
                    else:
                        result.append(str(item))
                return result

            cancel_candidates = normalize_order_ids(
                dec.get("cancel_orders")
                or dec.get("cancelOrders")
                or dec.get("cancel_order_ids")
            )
            replace_raw = dec.get("replace_orders") or dec.get("replaceOrders") or dec.get("order_replacements")
            replace_list = replace_raw if isinstance(replace_raw, list) else ([replace_raw] if isinstance(replace_raw, dict) else [])
            replacement_orders = []
            cancelled_ids = set()
            cancelled_success = []
            cancel_failures = []
            orders_activity = False

            def try_cancel(order_id: str, source: str):
                nonlocal orders_activity
                oid = str(order_id)
                if not oid or oid in cancelled_ids:
                    return
                success, err = cancel_order_by_id(ex, sym, oid)
                if success:
                    cancelled_ids.add(oid)
                    cancelled_success.append((oid, source))
                    orders_activity = True
                    log(f"🗑️ Отменён ордер {oid} для {sym} (источник {source})", Fore.LIGHTBLUE_EX)
                else:
                    cancel_failures.append((oid, err))
                    log(f"⚠️ Не удалось отменить ордер {oid} для {sym}: {err}", Fore.YELLOW)

            for oid in cancel_candidates:
                try_cancel(oid, "cancel_orders")

            for entry in replace_list:
                if not isinstance(entry, dict):
                    continue
                cancel_id = (
                    entry.get("cancel")
                    or entry.get("id")
                    or entry.get("old")
                    or entry.get("orderId")
                )
                if cancel_id:
                    try_cancel(cancel_id, "replace_orders")
                new_spec = (
                    entry.get("order")
                    or entry.get("new")
                    or entry.get("replacement")
                )
                if new_spec:
                    if isinstance(new_spec, dict):
                        replacement_orders.append(new_spec)
                    elif isinstance(new_spec, list):
                        replacement_orders.extend([x for x in new_spec if isinstance(x, dict)])

            if replacement_orders:
                orders_activity = True
                extra_orders.extend(replacement_orders)

            if cancelled_success:
                summary = ", ".join(oid for oid, _ in cancelled_success)
                send_tg(f"🗑️ Отменены ордера по {sym}: {summary}")
                # Обновляем список открытых ордеров после отмены
                open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)
            if cancel_failures:
                errors = "; ".join(f"{oid}: {err}" for oid, err in cancel_failures)
                send_tg(f"⚠️ Не удалось отменить ордера по {sym}: {errors}")

            if action == "skip":
                log(f"⏸ Пропуск {sym} ({reason})", Fore.WHITE)
                send_tg(f"⏸ Пропуск {sym} — {reason or 'причина не указана'}")
            elif action == "close":
                if not current_position or abs(float(current_position.get("amount") or 0)) == 0:
                    log(f"⚠️ Позиция по {sym} отсутствует, нечего закрывать ({reason})", Fore.YELLOW)
                    send_tg(f"⚠️ {sym}: закрытие пропущено — нет открытой позиции")
                else:
                    close_side = "sell" if (current_position.get("amount") or 0) > 0 else "buy"
                    qty = abs(float(current_position.get("amount") or 0))
                    if qty == 0:
                        log(f"⚠️ Объём позиции {sym} равен нулю, пропускаем закрытие", Fore.YELLOW)
                    else:
                        params = {"reduceOnly": True}
                        position_idx = get_position_idx(close_side)
                        if position_idx is not None:
                            params["positionIdx"] = position_idx
                        try:
                            ex.create_order(sym, "market", close_side, qty, None, params)
                            log(f"🔻 Закрыть позицию {sym} ({reason})", Fore.YELLOW)
                            send_tg(f"🔻 Закрыт {sym} {close_side.upper()} {qty:.4f} — {reason or 'причина не указана'}")
                            positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=available_pairs)
                            current_position = positions_map.get(sym)
                        except Exception as e:
                            err_text = str(e)
                            log(f"❌ Ошибка закрытия {sym}: {err_text}", Fore.RED)
                            send_tg(f"❌ Ошибка закрытия для {sym}: {err_text}")
            elif action == "hold":
                log(f"⏳ Удерживаем {sym} ({reason})", Fore.BLUE)
                send_tg(f"⏳ {sym}: удерживаем позицию — {reason or 'причина не указана'}")
                if current_position and abs(float(current_position.get('amount') or 0)) > 0:
                    updated_orders = ensure_position_protection(ex, sym, current_position, df, open_orders_symbol)
                    if updated_orders is not None:
                        open_orders_symbol = updated_orders
                        open_orders_cache[sym] = updated_orders
            elif action == "open":
                if current_position and abs(float(current_position.get("amount") or 0)) > 0:
                    log(f"⚠️ Позиция по {sym} уже открыта (side={current_position.get('side')}, amount={current_position.get('amount')}), пропускаем повторное открытие", Fore.YELLOW)
                    send_tg(f"⚠️ {sym}: позиция уже открыта, сигнал open пропущен")
                elif max_positions_limit > 0 and open_positions is not None and open_positions >= max_positions_limit:
                    log(f"⛔ Лимит открытых позиций достигнут ({open_positions}/{max_positions_limit}), пропускаем {sym}", Fore.YELLOW)
                    send_tg(f"⛔ Лимит открытых позиций достигнут ({open_positions}/{max_positions_limit}), {sym} пропущен")
                else:
                    log(f"🟢 Сигнал {side.upper()} ({reason})", Fore.GREEN)
                    send_tg(f"🟢 {sym} {side.upper()} — {reason or 'причина не указана'}")
                    if df.empty:
                        log(f"⚠️ Нет данных 30m для {sym}, пропускаем открытие", Fore.YELLOW)
                        send_tg(f"⚠️ {sym}: недостаточно данных для открытия позиции")
                        continue
                    df["atr"] = atr(df,14)
                    last_row = df.iloc[-1]
                    price = float(last_row.get("close") or 0)
                    atrv = float(last_row.get("atr") or 0)
                    if not (math.isfinite(price) and math.isfinite(atrv) and atrv > 0):
                        log(f"⚠️ Не удалось рассчитать ATR/цену для {sym}, пропуск сигнала", Fore.YELLOW)
                        send_tg(f"⚠️ {sym}: нет валидных значений ATR для расчёта размера")
                        continue
                    sl = price - SL_ATR * atrv if side == "buy" else price + SL_ATR * atrv
                    tp = price + TP_ATR * atrv if side == "buy" else price - TP_ATR * atrv
                    duplicate_order = None
                    if price > 0:
                        side_lower = side.lower()
                        for existing in open_orders_symbol or []:
                            try:
                                existing_side = (existing.get("side") or "").lower()
                                existing_type = (existing.get("type") or "").lower()
                            except AttributeError:
                                continue
                            if existing_side != side_lower:
                                continue
                            if existing_type != "limit":
                                continue
                            if existing.get("reduceOnly") in (True, "true", "1", 1):
                                continue
                            existing_price = existing.get("price")
                            if not (isinstance(existing_price, (int, float)) and math.isfinite(existing_price)):
                                continue
                            if abs(existing_price - price) / price <= 0.0005:
                                duplicate_order = existing
                                break
                    if duplicate_order:
                        dup_price = duplicate_order.get("price")
                        log(f"ℹ️ Пропуск лимитного ордера {sym}: уже выставлен {side.upper()} @ {dup_price}", Fore.LIGHTBLACK_EX)
                        send_tg(f"ℹ️ {sym}: лимит {side.upper()} @ {dup_price} уже активен, новый ордер не размещён")
                        continue
                    explicit_qty, explicit_notional = _extract_decision_position_size(dec, symbol_meta)
                    qty = None
                    notional = None
                    if explicit_qty is not None or explicit_notional is not None:
                        if price is None or not math.isfinite(price) or price <= 0:
                            log(f"⚠️ Невозможно применить объём для {sym}: недопустимая цена", Fore.YELLOW)
                            send_tg(f"⚠️ {sym}: модель прислала объём, но цена недоступна — пропускаем сделку.")
                            continue
                        qty = explicit_qty if explicit_qty is not None else explicit_notional / price
                        notional = qty * price
                    else:
                        risk_distance = abs(price - sl)
                        if risk_distance <= 0 or not math.isfinite(risk_distance):
                            log(f"⚠️ Невозможно рассчитать риск для {sym}", Fore.YELLOW)
                            send_tg(f"⚠️ {sym}: не удалось оценить риск, сделка пропущена")
                            continue
                        risk_budget_base = max(0.0, min(equity, available_margin))
                        risk_capital = risk_budget_base * RISK_PCT
                        if risk_capital <= 0:
                            log(f"⚠️ Недостаточно бюджета риска для {sym} ({available_margin:.2f} USDT)", Fore.YELLOW)
                            send_tg(f"⚠️ {sym}: недостаточно свободного баланса ({available_margin:.2f} USDT)")
                            continue
                        qty = risk_capital / risk_distance
                        if not math.isfinite(qty) or qty <= 0:
                            log(f"⚠️ Расчёт объёма дал некорректное значение для {sym}", Fore.YELLOW)
                            continue
                        notional = qty * price
                        if not math.isfinite(notional) or notional <= 0:
                            log(f"⚠️ Невозможно определить нотионал для {sym}", Fore.YELLOW)
                            continue
                    if not math.isfinite(qty) or qty <= 0:
                        log(f"⚠️ Объём сделки некорректен для {sym}", Fore.YELLOW)
                        continue
                    if not math.isfinite(notional) or notional <= 0:
                        log(f"⚠️ Нотионал сделки некорректен для {sym}", Fore.YELLOW)
                        continue
                    if notional < MIN_NOTIONAL_USDT:
                        qty = MIN_NOTIONAL_USDT / price
                        notional = qty * price
                    effective_margin = max(0.0, available_margin * ORDER_MARGIN_UTILIZATION)
                    max_notional = effective_margin * max(1, symbol_leverage)
                    if max_notional <= 0:
                        log(f"⛔ Доступная маржа для {sym} исчерпана", Fore.YELLOW)
                        send_tg(f"⛔ {sym}: доступная маржа исчерпана")
                        continue
                    if max_notional < MIN_NOTIONAL_USDT:
                        log(f"⛔ Недостаточно маржи для минимального ордера {sym} (доступно {available_margin:.2f} USDT)", Fore.YELLOW)
                        send_tg(f"⛔ {sym}: маржа меньше минимального объёма (доступно {available_margin:.2f} USDT)")
                        continue
                    margin_required = notional / symbol_leverage if symbol_leverage else notional
                    if margin_required > effective_margin:
                        log(f'⚠️ {sym}: требуемая маржа {margin_required:.2f} USDT превышает доступную {effective_margin:.2f} USDT, ордер пропущен', Fore.YELLOW)
                        send_tg(f'⚠️ {sym}: требуемая маржа {margin_required:.2f} USDT больше доступной {effective_margin:.2f} USDT, ордер пропущен')
                        continue
                    if notional > max_notional:
                        qty = max_notional / price
                        notional = max_notional
                        log(f'ℹ️ Объём {sym} уменьшен до {qty:.4f} (~{notional:.2f} USDT) из-за лимита маржи', Fore.LIGHTBLACK_EX)
                    try:
                        qty = float(ex.amount_to_precision(sym, qty))
                    except Exception:
                        qty = float(round(qty, 8))
                    if qty <= 0:
                        log(f"⚠️ После округления объём стал ≤ 0 для {sym}", Fore.YELLOW)
                        continue
                    notional = qty * price
                    if notional < MIN_NOTIONAL_USDT:
                        log(f"⛔ После округления объём {sym} ниже минимального ({notional:.2f} USDT)", Fore.YELLOW)
                        send_tg(f"⛔ {sym}: объём после округления ниже минимума ({notional:.2f} USDT)")
                        continue
                    try:
                        position_idx = get_position_idx(side)
                        params = {"takeProfit": tp, "stopLoss": sl, "tpSlMode": "Full", "reduceOnly": False}
                        if position_idx is not None:
                            params["positionIdx"] = position_idx
                        margin_required = notional / symbol_leverage if symbol_leverage else notional
                        ex.create_order(sym, "limit", side, qty, price, params)
                        log(f"✅ Ордер {sym} {side.upper()} {qty:.4f}@{price:.2f} SL:{sl:.2f} TP:{tp:.2f}", Fore.GREEN)
                        send_tg(
                            f"✅ {sym} {side.upper()} @ {price:.2f}\n"
                            f"SL {sl:.2f} TP {tp:.2f}\n"
                            f"Объём {notional:.2f} USDT, маржа {margin_required:.2f} USDT, плечо x{symbol_leverage}"
                        )
                        positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=available_pairs)
                        current_position = positions_map.get(sym)
                    except Exception as e:
                        err_text = str(e)
                        log(f"❌ Ошибка ордера: {err_text}", Fore.RED)
                        send_tg(f"❌ Ошибка ордера для {sym}: {err_text}")
            else:
                if action not in ("hold", "manage", "none", "", None):
                    log(f"ℹ️ Неизвестное действие \"{action}\" для {sym}, обработка только дополнительных ордеров", Fore.YELLOW)

            current_amount_val = safe_float((current_position or {}).get("amount") or (current_position or {}).get("contracts"))
            limit_blocks_new_orders = (
                max_positions_limit > 0
                and open_positions is not None
                and open_positions >= max_positions_limit
                and (current_amount_val is None or abs(current_amount_val) == 0)
            )
            if limit_blocks_new_orders:
                extra_orders = []

            if extra_orders:
                executed, actions_performed = execute_extra_orders(
                    ex,
                    sym,
                    extra_orders,
                    current_position=current_position,
                    open_orders=open_orders_symbol,
                    available_margin=available_margin,
                    symbol_leverage=symbol_leverage,
                    max_limits_per_side=MAX_NON_REDUCE_LIMITS_PER_SIDE,
                )
                if executed:
                    orders_activity = True
                    send_tg("🛠️ " + sym + " доп. ордера:\n- " + "\n- ".join(executed))
                if actions_performed:
                    orders_activity = True
                    positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=available_pairs)
                    current_position = positions_map.get(sym)
                    open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)

            final_position_payload = positions_map.get(sym)
            final_position_amount = safe_float(
                (final_position_payload or {}).get("amount")
                or (final_position_payload or {}).get("contracts")
            )
            if final_position_amount is None or not math.isfinite(final_position_amount):
                final_position_amount = 0.0
            final_protection_orders = _extract_protection_orders(open_orders_symbol)
            final_protection_signature = _protection_orders_signature(open_orders_symbol)
            protection_changed = initial_protection_signature != final_protection_signature
            amount_diff = abs(final_position_amount - initial_position_amount)
            amount_tolerance = max(abs(initial_position_amount), abs(final_position_amount)) * 1e-6 + 1e-8
            position_changed = amount_diff > amount_tolerance
            if protection_changed:
                orders_activity = True

            if detail_entry is None:
                if action == "open":
                    direction = "лонг" if side_text in ("buy", "long") else "шорт" if side_text in ("sell", "short") else ""
                    detail_entry = f"[{sym}] - открыт {direction or 'позиция'} (плечо x{symbol_leverage})"
                elif action == "close":
                    direction = "лонг" if side_text in ("buy", "long") else "шорт" if side_text in ("sell", "short") else ""
                    detail_entry = f"[{sym}] - закрыт {direction or 'позиция'} (плечо x{symbol_leverage})"
                elif action == "manage":
                    detail_entry = f"[{sym}] - держим позицию ({'меняли ордера' if orders_activity else 'ордера без изменений'})"
                elif action in ("hold", "none"):
                    change_parts: list[str] = []
                    if position_changed:
                        if final_position_amount > initial_position_amount:
                            change_parts.append("объём увеличен")
                        elif final_position_amount < initial_position_amount:
                            change_parts.append("объём уменьшен")
                    protection_changes = _describe_protection_changes(
                        initial_protection_orders,
                        final_protection_orders,
                    )
                    change_parts.extend(protection_changes)
                    if not change_parts:
                        change_parts.append("без изменений")
                    detail_entry = f"[{sym}] - держим позицию ({', '.join(change_parts)})"
                elif action == "skip":
                    detail_entry = f"[{sym}] - пропуск" + (f" — {reason}" if reason else "")
                else:
                    detail_entry = f"[{sym}] - пропуск" + (f" — {reason}" if reason else "")
            if detail_entry and sym_confidence_text:
                tag_suffix = f" {sym_confidence_tag}" if sym_confidence_tag else ""
                detail_entry = f"{detail_entry} [conf {sym_confidence_text}{tag_suffix}]"
            if detail_entry:
                decisions_details.append(detail_entry)

        except Exception as e:
            log(f"Ошибка {sym}: {e}\n{traceback.format_exc()}", Fore.RED)

    global_open_orders = fetch_all_open_orders_grouped(ex, limit=200)
    cleanup_symbols = sorted(
        set(symbols_sequence)
        | set(order_symbols)
        | {sym for sym, orders in open_orders_cache.items() if orders}
        | set(global_open_orders.keys())
    )
    cleanup_cancelled = {}
    cleanup_failures = []
    if cleanup_symbols:
        refreshed_positions, refreshed_count = fetch_positions_snapshot(ex, symbols_filter=cleanup_symbols)
        if refreshed_count is None:
            combined_positions = dict(positions_map)
        else:
            combined_positions = refreshed_positions
        for sym_cleanup in cleanup_symbols:
            if not sym_cleanup:
                continue
            position_payload = combined_positions.get(sym_cleanup)
            amount_val = safe_float(
                (position_payload or {}).get("amount") or (position_payload or {}).get("contracts")
            )
            if amount_val is not None and math.isfinite(amount_val) and abs(amount_val) > 0:
                continue
            orders_snapshot = global_open_orders.get(sym_cleanup)
            if orders_snapshot is None:
                orders_snapshot = fetch_open_orders_for_symbol(ex, sym_cleanup, limit=200)
            if not orders_snapshot:
                continue
            has_non_reduce_orders = any(
                isinstance(order, dict) and not _is_reduce_only(order)
                for order in orders_snapshot
            )
            if has_non_reduce_orders:
                continue
            to_cancel_ids = []
            for order in orders_snapshot:
                if not isinstance(order, dict):
                    continue
                if not _is_reduce_only(order):
                    continue
                oid = order.get("id")
                if oid:
                    to_cancel_ids.append(str(oid))
            if not to_cancel_ids:
                continue
            cancelled_here = []
            for oid in to_cancel_ids:
                success, err = cancel_order_by_id(ex, sym_cleanup, oid)
                if success:
                    cancelled_here.append(oid)
                else:
                    cleanup_failures.append((sym_cleanup, oid, err))
            if cancelled_here:
                cleanup_cancelled[sym_cleanup] = cancelled_here
                updated_snapshot = fetch_open_orders_for_symbol(ex, sym_cleanup, limit=200)
                open_orders_cache[sym_cleanup] = updated_snapshot
                global_open_orders[sym_cleanup] = updated_snapshot
    if cleanup_cancelled:
        for sym_cleanup, ids in cleanup_cancelled.items():
            summary = ", ".join(ids)
            log(f"✅ Сняты reduce-only стоп-ордера по {sym_cleanup}: {summary}", Fore.LIGHTBLUE_EX)
            send_tg(f"✅ {sym_cleanup}: убраны reduce-only стопы (без позиции): {summary}")
    if cleanup_failures:
        details = "; ".join(f"{sym}:{oid} -> {err}" for sym, oid, err in cleanup_failures)
        log(f"⚠️ Не удалось отменить reduce-only стоп-ордера: {details}", Fore.YELLOW)
        send_tg(f"⚠️ Ошибка отмены reduce-only стоп-ордеров: {details}")

    if decisions_total>0:
        pct={k:(v/decisions_total)*100 for k,v in counts.items()}
        summary=f"📈 Итоги: открыто {counts['open']} ({pct['open']:.1f}%), " \
                f"закрыто {counts['close']} ({pct['close']:.1f}%), " \
                f"пропуск {counts['skip']} ({pct['skip']:.1f}%) — всего {decisions_total}"
        log(summary, Fore.CYAN)
        send_tg(summary)
        if decisions_details:
            detail_msg = "\n".join(decisions_details)
            log(detail_msg, Fore.LIGHTBLACK_EX)
            send_tg(detail_msg)

    final_positions_map, final_positions_count = fetch_positions_snapshot(ex)
    final_positions_available = final_positions_count is not None
    if not final_positions_available:
        final_positions_map = dict(positions_map)

    no_active_positions = final_positions_available and final_positions_count == 0
    flat_skipped_all = (
        no_active_positions
        and eligible_flat_symbols > 0
        and skipped_plan_omitted_symbols == eligible_flat_symbols
    )
    if flat_skipped_all:
        skipped_list = ", ".join(sorted(set(flat_skip_symbols))) or "n/a"
        issue_msg = (
            "[FAIL] Trade plan omitted every flat symbol "
            f"(omitted symbols: {skipped_list})"
        )
        log(issue_msg, Fore.RED)
        send_tg(issue_msg)
        raise RuntimeError("trade plan omitted every flat symbol; no active exposure")

    unavailable_all = (
        trade_plan_failed
        and eligible_flat_symbols > 0
        and skipped_plan_unavailable_symbols == eligible_flat_symbols
    )
    if unavailable_all:
        skipped_list = ", ".join(sorted(set(flat_unavailable_symbols))) or "n/a"
        issue_msg = (
            "[FAIL] Trade plan unavailable for all flat symbols "
            f"(symbols: {skipped_list})"
        )
        log(issue_msg, Fore.RED)
        send_tg(issue_msg)
        raise RuntimeError("trade plan unavailable for all flat symbols")

    unprotected_positions: list[tuple[str, float]] = []
    for sym_active, payload in final_positions_map.items():
        if not isinstance(payload, dict):
            continue
        amount_val = safe_float(payload.get("amount") or payload.get("contracts"))
        if amount_val is None or not math.isfinite(amount_val) or abs(amount_val) <= 1e-8:
            continue
        orders_snapshot = global_open_orders.get(sym_active) if 'global_open_orders' in locals() else None
        if orders_snapshot is None:
            orders_snapshot = fetch_open_orders_for_symbol(ex, sym_active)
        protective_orders = _extract_protection_orders(orders_snapshot)
        has_stop = any(_has_stop_flag(order) or _has_trailing_flag(order) for order in protective_orders)
        if not has_stop:
            unprotected_positions.append((sym_active, amount_val))

    unresolved_unprotected: list[str] = []
    for sym_unprotected, amount_unprotected in unprotected_positions:
        restored = False
        position_payload = final_positions_map.get(sym_unprotected) or {}
        try:
            df_attempt = fetch_df(ex, sym_unprotected, TIMEFRAME)
        except Exception as exc_fetch_df:
            log(f"[WARN] Failed to fetch primary timeframe for {sym_unprotected}: {exc_fetch_df}", Fore.YELLOW)
            df_attempt = None
        try:
            open_orders_attempt = fetch_open_orders_for_symbol(ex, sym_unprotected)
        except Exception as exc_fetch_orders:
            log(f"[WARN] Failed to refresh open orders for {sym_unprotected}: {exc_fetch_orders}", Fore.YELLOW)
            open_orders_attempt = []
        if df_attempt is not None and position_payload:
            try:
                updated_orders = ensure_position_protection(
                    ex,
                    sym_unprotected,
                    position_payload,
                    df_attempt,
                    open_orders_attempt,
                )
                if isinstance(updated_orders, list):
                    open_orders_attempt = updated_orders
            except Exception as exc_protect:
                log(f"[WARN] Failed to restore protection for {sym_unprotected}: {exc_protect}", Fore.YELLOW)
        try:
            refreshed_orders = fetch_open_orders_for_symbol(ex, sym_unprotected)
        except Exception as exc_refresh_orders:
            log(f"[WARN] Failed to refresh orders after protection attempt for {sym_unprotected}: {exc_refresh_orders}", Fore.YELLOW)
            refreshed_orders = open_orders_attempt
        protective_orders_after = _extract_protection_orders(refreshed_orders)
        has_stop_after = any(_has_stop_flag(order) or _has_trailing_flag(order) for order in protective_orders_after)
        if has_stop_after:
            restored = True
            continue

        close_side = "sell" if amount_unprotected > 0 else "buy"
        qty_close = abs(amount_unprotected)
        params_close = {"reduceOnly": True}
        position_idx = get_position_idx(close_side)
        if position_idx is not None:
            params_close["positionIdx"] = position_idx
        try:
            ex.create_order(sym_unprotected, "market", close_side, qty_close, None, params_close)
            log(f"[INFO] Closed position for {sym_unprotected} {close_side.upper()} {qty_close:.4f} due to missing protection", Fore.YELLOW)
            send_tg(f"[WARN] {sym_unprotected}: position closed due to missing protection")
            latest_positions, _ = fetch_positions_snapshot(ex, symbols_filter=[sym_unprotected])
            final_positions_map.update(latest_positions)
            restored = True
            continue
        except Exception as exc_close:
            log(f"[ERROR] Failed to close unprotected position {sym_unprotected}: {exc_close}", Fore.RED)
            send_tg(f"[ERROR] {sym_unprotected}: failed to close unprotected position - {exc_close}")

        latest_positions, _ = fetch_positions_snapshot(ex, symbols_filter=[sym_unprotected])
        pos_check = latest_positions.get(sym_unprotected)
        amt_check = safe_float((pos_check or {}).get("amount") or (pos_check or {}).get("contracts"))
        if amt_check is None or not math.isfinite(amt_check) or abs(amt_check) <= 1e-8:
            restored = True
        if not restored:
            unresolved_unprotected.append(sym_unprotected)

    if unresolved_unprotected:
        details = ", ".join(sorted(set(unresolved_unprotected)))
        alert_msg = f"[FAIL] Positions remain without protection or close-out: {details}"
        log(alert_msg, Fore.RED)
        send_tg(alert_msg)
        raise ProtectionMissingError("unprotected positions could not be safeguarded or closed")

    if AI_TOKEN_USAGE_BY_MODEL:
        usage_lines = [
            f"- модель {model}: {stats['total']} токенов (prompt {stats['prompt']}, completion {stats['completion']}, запросов {stats['requests']})"
            for model, stats in sorted(AI_TOKEN_USAGE_BY_MODEL.items())
        ]
        usage_report = "AI токены за цикл:\n" + "\n".join(usage_lines)
        log(usage_report, Fore.LIGHTBLACK_EX)
        send_tg(usage_report)

    next_delay_minutes = None
    next_run_dt = None
    if selection_next_run is not None:
        try:
            minutes_val = float(selection_next_run)
        except (TypeError, ValueError):
            minutes_val = None
        if minutes_val and minutes_val > 0:
            next_delay_minutes = minutes_val
            next_run_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=minutes_val)
            next_local = next_run_dt.astimezone(_current_local_tz() or datetime.datetime.now().astimezone().tzinfo)
            msg = f"Next cycle in {minutes_val:.1f} min (~{next_local.strftime('%Y-%m-%d %H:%M:%S %Z')})"
            log(msg, Fore.CYAN)
            send_tg(msg)
        else:
            log('Invalid next_run_minutes from model.', Fore.YELLOW)
    elif selection_next_time:
        candidate = selection_next_time.strip() if isinstance(selection_next_time, str) else ""
        if candidate:
            iso_candidate = candidate.replace("Z", "+00:00")
            try:
                target_dt = datetime.datetime.fromisoformat(iso_candidate)
                if target_dt.tzinfo is None:
                    target_dt = target_dt.replace(tzinfo=datetime.timezone.utc)
                delta = (target_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60.0
                if delta > 0:
                    next_delay_minutes = delta
                    next_run_dt = target_dt
                    next_local = target_dt.astimezone(_current_local_tz() or datetime.datetime.now().astimezone().tzinfo)
                    msg = f"Next run scheduled for {next_local.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                    log(msg, Fore.CYAN)
                    send_tg(msg)
                else:
                    log('next_run_time from model is in the past.', Fore.YELLOW)
            except Exception as exc:
                log(f"Failed to parse next_run_time '{selection_next_time}': {exc}", Fore.YELLOW)
    if next_delay_minutes is None:
        next_delay_minutes = DEFAULT_NEXT_RUN_MINUTES
        fallback_msg = f"Next cycle defaulting to {next_delay_minutes:.0f} minutes."
        log(fallback_msg, Fore.LIGHTBLACK_EX)
        send_tg(fallback_msg)
        next_run_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=next_delay_minutes)
    elif next_run_dt is None:
        next_run_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=next_delay_minutes)
    _write_runtime_status(next_delay_minutes, next_run_dt, "sleeping")
    send_tg("✅ Цикл завершён.")
    changelog_state = ensure_changelog_announcement()
    version_display = BOT_VERSION
    send_kwargs: dict[str, Any] = {}
    link = changelog_state.get("link") if isinstance(changelog_state, dict) else None
    changelog_message_id = changelog_state.get("message_id") if isinstance(changelog_state, dict) else None
    if link:
        version_display = f'<a href="{link}">{BOT_VERSION}</a>'
        send_kwargs["parse_mode"] = "HTML"
        send_kwargs["disable_web_page_preview"] = True
    if changelog_message_id:
        send_kwargs["reply_to_message_id"] = changelog_message_id
    if changelog_message_id or link:
        send_tg(f"ℹ️ Версия {version_display}", **send_kwargs)
    else:
        send_tg(f"ℹ️ Версия {BOT_VERSION}. {BOT_CHANGELOG}")
    try:
        equity_end, available_end, _ = fetch_usdt_equity(ex)
    except Exception as exc_equity:
        end_balance_text = f"⚠️ Не удалось обновить баланс: {exc_equity}"
        log(end_balance_text, Fore.YELLOW)
        send_tg(end_balance_text)
    else:
        end_balance_text = f"Баланс: {equity_end:.2f} USDT, доступно {available_end:.2f} USDT"
        log(f"🏁 Завершение сессии. {end_balance_text}", Fore.GREEN)
        send_tg(f"🏁 Завершение сессии. {end_balance_text}")
    end_dt = _current_log_time()
    end_stamp = end_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    end_banner = f"{session_separator} END SESSION {end_stamp} {session_separator}"
    log(end_banner, Fore.MAGENTA)
    send_tg(f"{session_separator}\nEND SESSION {end_stamp}\n{session_separator}")
    return next_delay_minutes

def main():
    ensure_version_backup()
    while True:
        try:
            delay_minutes = run_cycle()
        except KeyboardInterrupt:
            log("Interrupted by user.", Fore.YELLOW)
            _write_runtime_status(None, None, "stopped")
            break
        except Exception:
            _write_runtime_status(None, None, "error")
            raise
        if not delay_minutes or delay_minutes <= 0:
            delay_minutes = DEFAULT_NEXT_RUN_MINUTES

        try:
            remaining_seconds = max(0.0, float(delay_minutes) * 60.0)
        except (TypeError, ValueError):
            remaining_seconds = float(DEFAULT_NEXT_RUN_MINUTES) * 60.0

        if remaining_seconds <= 0:
            continue

        next_run_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=remaining_seconds)
        local_tz = _current_local_tz() or datetime.datetime.now().astimezone().tzinfo
        next_local = next_run_dt.astimezone(local_tz)
        eta_msg = (
            f"🕒 Следующая сессия запланирована на {next_local.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"(~{delay_minutes:.1f} мин)"
        )
        log(eta_msg, Fore.LIGHTBLACK_EX)
        send_tg(eta_msg)

        progress_enabled = remaining_seconds >= 180
        if progress_enabled:
            progress_interval = min(300.0, max(90.0, remaining_seconds / 4.0))
        else:
            progress_interval = remaining_seconds

        try:
            while remaining_seconds > 0:
                step = min(progress_interval, remaining_seconds)
                time.sleep(step)
                remaining_seconds -= step
                if remaining_seconds <= 0:
                    break
                if not progress_enabled:
                    continue
                minutes_left = remaining_seconds / 60.0
                eta_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=remaining_seconds)
                eta_local = eta_dt.astimezone(local_tz)
                progress_msg = (
                    f"⏳ Осталось ~{minutes_left:.1f} мин до следующей сессии "
                    f"({eta_local.strftime('%H:%M:%S %Z')})"
                )
                log(progress_msg, Fore.LIGHTBLACK_EX)
                send_tg(progress_msg)
        except KeyboardInterrupt:
            log("Interrupted during sleep.", Fore.YELLOW)
            _write_runtime_status(None, None, "stopped")
            break
    _write_runtime_status(None, None, "stopped")

if __name__ == "__main__":
    ensure_version_backup()
    try:
        main()
    except KeyboardInterrupt:
        log("Startup interrupted by user.", Fore.YELLOW)
        _write_runtime_status(None, None, "stopped")
        sys.exit(0)
    except Exception as exc:
        log(f"Startup failed for version {BOT_VERSION}: {exc}", Fore.RED)
        traceback.print_exc()
        _write_runtime_status(None, None, "error")
        try:
            current_script = Path(__file__).resolve()
        except (NameError, OSError):
            raise
        fallback = _find_previous_version_script(BOT_VERSION, current_script)
        if not fallback:
            log("No fallback version available. Exiting.", Fore.RED)
            raise
        fallback_version, fallback_path = fallback
        log(
            f"Attempting fallback version {fallback_version} from {fallback_path.name}",
            Fore.YELLOW
        )
        exit_code = _run_previous_version_script(fallback_path)
        if exit_code != 0:
            log(f"Fallback version exited with code {exit_code}", Fore.RED)
        sys.exit(exit_code)

