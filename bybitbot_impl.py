# -*- coding: utf-8 -*-
# Version: 12.3
"""
Bybit Intraday AI Trading Bot — 30m, 5 пар USDT Perpetual
Сбалансированный интрадей-бот с поддержкой OpenAI GPT, Telegram и расширенным контекстом.

"""

import os
import atexit
import shutil
import stat
import subprocess
import sys
import threading
import time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# --- Импорты ---
import math, time, json, traceback, datetime, random, warnings, re, numbers, hashlib, textwrap, types
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional, Tuple, Any, Sequence, Mapping
import pandas as pd
import ccxt
import requests
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from colorama import Fore, Style, init
from openai import OpenAI, RateLimitError
from dotenv import dotenv_values
import db_logger
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

# Ensure UTF-8 console on Windows/streams that support reconfigure
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Версия бота: обновляйте при каждом релизе/значимых изменениях
BOT_VERSION = "1.2.2"
BOT_CHANGELOG = (
    "Volatility-aware balance between news and technicals guides the AI to lean on catalysts in high ATR and on TA in calm markets."
)
SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_repo_root(script_dir: Path) -> Path:
    """Return the nearest parent containing .git (or script_dir if not found)."""
    current = script_dir
    for _ in range(6):
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return script_dir


REPO_ROOT = _resolve_repo_root(SCRIPT_DIR)
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.txt"
STATE_DIR = REPO_ROOT
db_logger.initialize()
def _configure_state_paths() -> None:
    global STATE_DIR
    global EQUITY_HISTORY_FILE
    global CYCLE_STATE_FILE
    global FALLBACK_HISTORY_FILE
    global RESULTS_STATE_FILE
    global RELEASE_STATE_FILE
    global BYBIT_CREDENTIALS_FILE
    global SUPPORT_SANDBOX_ROOT
    global SUPPORT_SANDBOX_STATE_FILE
    global COMMANDS_HELP_STATE_FILE
    global MASTER_DECISIONS_FILE
    state_dir_raw = os.getenv("BYBITBOT_STATE_DIR")
    try:
        STATE_DIR = (Path(state_dir_raw).expanduser().resolve() if state_dir_raw else REPO_ROOT)
    except Exception:
        STATE_DIR = SCRIPT_DIR
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    EQUITY_HISTORY_FILE = STATE_DIR / "equity_history.json"
    CYCLE_STATE_FILE = STATE_DIR / "cycle_state.json"
    FALLBACK_HISTORY_FILE = STATE_DIR / "fallback_history.json"
    RESULTS_STATE_FILE = STATE_DIR / "results_state.json"
    RELEASE_STATE_FILE = STATE_DIR / "release_state.json"
    BYBIT_CREDENTIALS_FILE = STATE_DIR / "bybit_credentials.json"
    SUPPORT_SANDBOX_ROOT = STATE_DIR / "support_sandboxes"
    SUPPORT_SANDBOX_STATE_FILE = STATE_DIR / "support_sandboxes.json"
    COMMANDS_HELP_STATE_FILE = STATE_DIR / "commands_help_state.json"
    MASTER_DECISIONS_FILE = STATE_DIR / "master_decisions.json"
    try:
        SUPPORT_SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Per-cycle storage for pseudo-trailing decisions.
_PREV_UNREALIZED_PNL: dict[str, float] = {}
_TRAIL_PROTECTION: dict[str, dict[str, float]] = {}
_CURRENT_CYCLE_NUMBER: int | None = None

MARKET_MODE_GUIDE: dict[str, Any] = {
    "market_modes": {
        "trend": {
            "open_position": "yes",
            "conditions": {
                "trend_alignment": "EMA20 > EMA50 для long или EMA20 < EMA50 для short; тренд 4h совпадает",
                "rsi": "не в экстремумах (long: RSI < 70; short: RSI > 30)",
                "atr": "не выше 1.8 × среднего ATR",
            },
            "risk": {
                "risk_multiplier": 1.0,
                "sl_atr": "1.0 × ATR",
                "tp_atr": "2.0 × ATR",
            },
            "confidence_required": 0.75,
            "needs_confirmation": {
                "required": False,
                "when": [
                    "если 30m и 4h частично расходятся",
                    "если RSI в зоне экстремума",
                    "если ATR резко вырос",
                ],
                "confirm_with": [
                    "higher_tf:4h",
                    "funding",
                    "open_interest",
                    "news",
                ],
            },
        },
        "countertrend": {
            "open_position": "conditional",
            "conditions": {
                "rsi": "выход из экстремума (long: RSI < 30 → вверх; short: RSI > 70 → вниз)",
                "atr": "снижается относительно среднего",
                "trend_strength": "4h тренд слабый, нет усиления",
            },
            "risk": {
                "risk_multiplier": 0.5,
                "sl_atr": "0.7 × ATR",
                "tp_atr": "1.0 × ATR",
            },
            "confidence_required": 0.80,
            "needs_confirmation": {
                "required": True,
                "confirm_with": [
                    "higher_tf:30m",
                    "higher_tf:1h",
                    "funding",
                    "open_interest",
                    "news",
                ],
            },
        },
        "range": {
            "open_position": "only_at_boundaries",
            "conditions": {
                "price_location": "у верхней или нижней границы диапазона",
                "rsi": "должен подтверждать разворот от границы",
                "atr": "ниже среднего, рынок маловолатилен",
            },
            "risk": {
                "risk_multiplier": 0.25,
                "sl_atr": "0.6–0.8 × ATR",
                "tp_target": "до противоположной границы диапазона",
            },
            "confidence_required": 0.70,
            "needs_confirmation": {
                "required": True,
                "confirm_with": [
                    "higher_tf:30m",
                    "higher_tf:1h",
                    "atr_low_vol",
                    "news_neutral",
                    "funding≈0",
                ],
            },
        },
    }
}

MARKET_MODE_CONFIG: dict[str, dict[str, float]] = {
    "trend": {
        "risk_multiplier": 1.0,
        "sl_scale": 1.0,
        "tp_ratio": 2.0,
        "confidence_required": 0.75,
    },
    "countertrend": {
        "risk_multiplier": 0.5,
        "sl_scale": 0.7,
        "tp_ratio": 1.4286,
        "confidence_required": 0.80,
    },
    "range": {
        "risk_multiplier": 0.25,
        "sl_scale": 0.7,
        "tp_ratio": 1.0,
        "confidence_required": 0.70,
    },
}

SYMBOL_MARKET_MODE_HINTS: dict[str, dict[str, Any]] = {}

_LOG_HISTORY: deque[str] = deque(maxlen=200)
LOG_EXTRA_SETTLE_POSITIONS = str(os.getenv("LOG_EXTRA_SETTLE_POSITIONS", "")).strip().lower() in {"1", "true", "yes", "on"}
GRAPH_OUTPUT_DIR = SCRIPT_DIR / "assets" / "graphs"
GRAPH_SEND_INTERVAL_DEFAULT = 60

MASTER_PROMPT_POSITION_CACHE: dict[str, dict[str, Any]] = {}
MASTER_PROMPT_SHARE = os.getenv("MASTER_PROMPT_SHARE", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_CONTEXT_30M = int(os.getenv("AI_DEFAULT_CONTEXT_30M", "120"))
DEFAULT_CONTEXT_4H = int(os.getenv("AI_DEFAULT_CONTEXT_4H", "48"))
MIN_CONTEXT_30M = int(os.getenv("AI_MIN_CONTEXT_30M", "16"))
MIN_CONTEXT_4H = int(os.getenv("AI_MIN_CONTEXT_4H", "8"))
CONTEXT_STEP_30M = max(1, int(os.getenv("AI_CONTEXT_STEP_30M", "8")))
CONTEXT_STEP_4H = max(1, int(os.getenv("AI_CONTEXT_STEP_4H", "4")))
MIN_INDICATORS_PER_TF = max(1, int(os.getenv("AI_MIN_INDICATORS_PER_TF", "3")))
MASTER_DECISIONS_SHARE = os.getenv("MASTER_DECISIONS_SHARE", "1").strip().lower() not in {"0", "false", "no"}
MASTER_PROMPT_MASTER_ID = os.getenv("MASTER_PROMPT_MASTER_ID")
MASTER_DECISIONS_FILE: Optional[Path] = None
MASTER_DECISION_CACHE: dict[str, Any] = {}
MASTER_DECISION_META: dict[str, Any] = {}
AI_REQUESTS_FULL_CONTEXT = str(os.getenv("AI_REQUESTS_FULL_CONTEXT", "0")).strip().lower() in {"1", "true", "yes", "on"}
AI_REQUESTS_MAX_CONTEXT_BARS = max(0, int(os.getenv("AI_REQUESTS_MAX_CONTEXT_BARS", "0")))

# --- AI provider / fallback configuration ---
# Primary provider is OpenAI; on rate-limit (429) we can switch to a secondary
# provider (DeepSeek-compatible endpoint) if configured via environment.
AI_PROVIDER_PRIMARY = "openai"
AI_PROVIDER_SECONDARY = "deepseek"
AI_PROVIDER_CURRENT = "openai"
DEEPSEEK_API_KEY: str | None = None
DEEPSEEK_API_BASE: str | None = None
DEEPSEEK_MODEL: str | None = None
AI_OFFLINE_CANCEL_ENTRIES = True
_AI_OFFLINE_NOTICE_EMITTED_CYCLE: int | None = None
_AI_OFFLINE_ACTIVE_CYCLE: int | None = None
OFFLINE_TRADING_ENABLED = False
OFFLINE_PAIR_LIMIT = 8
OFFLINE_MAX_NEW_POSITIONS = 1
OFFLINE_NEWS_BIAS_ENABLED = True


def _bytes_from_env(env_name: str, default_mb: float) -> int:
    raw_value = os.getenv(env_name)
    if raw_value:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = float(default_mb)
    else:
        value = float(default_mb)
    return max(0, int(value * 1024 * 1024))


def _float_from_env(env_name: str, default_value: float) -> float:
    raw_value = os.getenv(env_name)
    if raw_value:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(default_value)
    return float(default_value)


def _parse_settle_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    parts: list[str] = []
    for token in raw.split(","):
        cleaned = token.strip()
        if cleaned:
            parts.append(cleaned.upper())
    return parts or list(default)


DEFAULT_USER_LOG_MAX_MB = 2.5
USER_LOG_MAX_BYTES = _bytes_from_env("BYBIT_USER_LOG_MAX_MB", DEFAULT_USER_LOG_MAX_MB)
USER_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_USER_LOG_BACKUPS", "3")))
DEFAULT_AI_LOG_MAX_MB = 16.0
AI_LOG_MAX_BYTES = _bytes_from_env("BYBIT_AI_LOG_MAX_MB", DEFAULT_AI_LOG_MAX_MB)
AI_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_AI_LOG_BACKUPS", "5")))
DEFAULT_MAIN_LOG_MAX_MB = 12.0
_main_log_path_raw = os.getenv("BYBIT_MAIN_LOG")
if _main_log_path_raw:
    try:
        MAIN_LOG_PATH = Path(_main_log_path_raw).expanduser()
    except Exception:
        MAIN_LOG_PATH = REPO_ROOT / "bybit.log"
else:
    MAIN_LOG_PATH = REPO_ROOT / "bybit.log"
MAIN_LOG_MAX_BYTES = _bytes_from_env("BYBIT_MAIN_LOG_MAX_MB", DEFAULT_MAIN_LOG_MAX_MB)
MAIN_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_MAIN_LOG_BACKUPS", "5")))
MAIN_LOG_ENABLED = str(os.getenv("BYBIT_MAIN_LOG_DISABLE", "0")).lower() not in {"1", "true", "yes"}
DEFAULT_ERROR_LOG_MAX_MB = 8.0
_error_log_path_raw = os.getenv("BYBIT_ERROR_LOG")
if _error_log_path_raw:
    try:
        ERROR_LOG_PATH = Path(_error_log_path_raw).expanduser()
    except Exception:
        ERROR_LOG_PATH = REPO_ROOT / "assets" / "error.log"
else:
    ERROR_LOG_PATH = REPO_ROOT / "assets" / "error.log"
ERROR_LOG_MAX_BYTES = _bytes_from_env("BYBIT_ERROR_LOG_MAX_MB", DEFAULT_ERROR_LOG_MAX_MB)
ERROR_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_ERROR_LOG_BACKUPS", "5")))
ERROR_LOG_ENABLED = str(os.getenv("BYBIT_ERROR_LOG_DISABLE", "0")).lower() not in {"1", "true", "yes"}
DEFAULT_ATR_GUARD_MAX_RATIO = 0.055
DEFAULT_ATR_GUARD_MIN_RATIO = 0.018
DEFAULT_ATR_GUARD_LOW_BOOST = 1.15
VOL_GUARD_ENABLED = str(os.getenv("ATR_GUARD_ENABLED", "1")).lower() not in {"0", "false", "no"}
ATR_GUARD_MAX_RATIO = max(0.0, _float_from_env("ATR_GUARD_MAX_RATIO", DEFAULT_ATR_GUARD_MAX_RATIO))
ATR_GUARD_MIN_RATIO = max(0.0, _float_from_env("ATR_GUARD_MIN_RATIO", DEFAULT_ATR_GUARD_MIN_RATIO))
if ATR_GUARD_MAX_RATIO > 0 and ATR_GUARD_MIN_RATIO > ATR_GUARD_MAX_RATIO:
    ATR_GUARD_MIN_RATIO = ATR_GUARD_MAX_RATIO * 0.75
ATR_GUARD_LOW_BOOST = max(1.0, _float_from_env("ATR_GUARD_LOW_BOOST", DEFAULT_ATR_GUARD_LOW_BOOST))
DEFAULT_EXTRA_POSITION_SETTLES = ["USDC"]
EXTRA_POSITION_SETTLES: list[str] = list(DEFAULT_EXTRA_POSITION_SETTLES)
DRAWDOWN_CONTROL_ENABLED: bool = True
DRAWDOWN_WINDOW_HOURS: float = 24.0 * 7.0
DRAWDOWN_RULES: tuple[tuple[float, float, float, float], ...] = (
    (50.0, 0.35, 0.55, 0.45),
    (40.0, 0.50, 0.65, 0.60),
    (30.0, 0.70, 0.75, 0.80),
)
DEFAULT_VOLATILITY_NEWS_PRIORITY_ATR = 0.02
DEFAULT_VOLATILITY_TA_PRIORITY_ATR = 0.012
VOLATILITY_NEWS_PRIORITY_ATR = max(
    0.0,
    _float_from_env("VOLATILITY_NEWS_PRIORITY_ATR", DEFAULT_VOLATILITY_NEWS_PRIORITY_ATR),
)
VOLATILITY_TA_PRIORITY_ATR = max(
    0.0,
    _float_from_env("VOLATILITY_TA_PRIORITY_ATR", DEFAULT_VOLATILITY_TA_PRIORITY_ATR),
)
if 0 < VOLATILITY_NEWS_PRIORITY_ATR < VOLATILITY_TA_PRIORITY_ATR:
    VOLATILITY_TA_PRIORITY_ATR = VOLATILITY_NEWS_PRIORITY_ATR * 0.75


def detect_regime(
    df_primary: pd.DataFrame | None,
    higher_tf_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    """
    Classify current market regime as TREND_UP / TREND_DOWN / FLAT / COUNTER.

    Returns (mode, metrics) where mode is one of the strings above and
    metrics contains helper values (ema spread, atr ratio, rsi, etc.).
    """
    mode = "FLAT"
    metrics: dict[str, Any] = {}

    if df_primary is None or df_primary.empty:
        return mode, metrics

    last_row = df_primary.iloc[-1]
    close_val = safe_float(last_row.get("close"))
    ema20_val = safe_float(last_row.get("ema20"))
    ema50_val = safe_float(last_row.get("ema50"))
    rsi_val = safe_float(last_row.get("rsi"))
    atr_val = safe_float(last_row.get("atr"))

    if not (math.isfinite(close_val or 0) and math.isfinite(ema20_val or 0) and math.isfinite(ema50_val or 0)):
        return mode, metrics

    ema_spread = (ema20_val - ema50_val) if ema20_val is not None and ema50_val is not None else 0.0
    ema_spread_pct = abs(ema_spread) / close_val if close_val else 0.0
    atr_ratio = (atr_val or 0.0) / close_val if close_val and atr_val is not None else 0.0

    metrics.update(
        {
            "close": close_val,
            "ema20": ema20_val,
            "ema50": ema50_val,
            "ema_spread_pct": ema_spread_pct,
            "atr_ratio": atr_ratio,
            "rsi": rsi_val,
        }
    )

    higher_bias = None
    if higher_tf_rows:
        last_higher = higher_tf_rows[-1]
        ema20_ht = safe_float(last_higher.get("ema20"))
        ema50_ht = safe_float(last_higher.get("ema50"))
        if ema20_ht is not None and ema50_ht is not None:
            if ema20_ht > ema50_ht:
                higher_bias = "up"
            elif ema20_ht < ema50_ht:
                higher_bias = "down"
        metrics["higher_ema20"] = ema20_ht
        metrics["higher_ema50"] = ema50_ht

    flat_spread_threshold = 0.0025  # 0.25%
    flat_atr_threshold = 0.01      # 1% дневной волатильности

    if ema_spread_pct < flat_spread_threshold and atr_ratio < flat_atr_threshold:
        mode = "FLAT"
    else:
        local_trend_up = ema_spread > 0
        if higher_bias == "up" and local_trend_up:
            mode = "TREND_UP"
        elif higher_bias == "down" and not local_trend_up:
            mode = "TREND_DOWN"
        else:
            mode = "COUNTER"

    metrics["mode"] = mode
    metrics["higher_bias"] = higher_bias
    return mode, metrics


def _regime_to_market_mode(regime_label: str | None) -> str:
    """Map TREND/COUNTER/FLAT regimes into prompt-friendly market modes."""
    if regime_label in {"TREND_UP", "TREND_DOWN"}:
        return "trend"
    if regime_label == "COUNTER":
        return "countertrend"
    if regime_label == "FLAT":
        return "range"
    return "trend"


def _resolve_analysis_balance(atr_ratio: float | None) -> dict[str, Any]:
    """Determine how much weight to assign to news vs technicals based on ATR/price."""
    ratio = float(atr_ratio or 0.0)
    news_weight = 0.5
    ta_weight = 0.5
    mode = "balanced"
    guidance = "Use news and technicals evenly."
    if VOLATILITY_NEWS_PRIORITY_ATR and ratio >= VOLATILITY_NEWS_PRIORITY_ATR:
        mode = "news_priority"
        news_weight = 0.65
        ta_weight = 0.35
        guidance = (
            "Volatile conditions: prioritize news/sentiment catalysts; still confirm with key indicators."
        )
    elif VOLATILITY_TA_PRIORITY_ATR and 0 < ratio <= VOLATILITY_TA_PRIORITY_ATR:
        mode = "technical_priority"
        news_weight = 0.35
        ta_weight = 0.65
        guidance = (
            "Calm market: lean on technical structure/indicators; treat news as a secondary tiebreaker."
        )
    return {
        "mode": mode,
        "atr_ratio": ratio,
        "news_weight": round(news_weight, 2),
        "technical_weight": round(ta_weight, 2),
        "guidance": guidance,
    }


def _parse_env_list(raw: str | None, default: Sequence[str] | None) -> list[str]:
    if not raw:
        return list(default or [])
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_env_map(raw: str | None, default: Mapping[str, Any] | None) -> dict[str, int]:
    base = {str(key).strip(): int(value) for key, value in (default or {}).items() if value is not None}
    if not raw:
        return base
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            numeric = safe_int(value) if value is not None else None
            if numeric is not None and numeric > 0:
                base[normalized_key] = numeric
        return base
    for part in str(raw).split(","):
        chunk = part.strip()
        if ":" not in chunk:
            continue
        key, val = chunk.split(":", 1)
        normalized_key = key.strip()
        numeric = safe_int(val.strip())
        if normalized_key and numeric is not None and numeric > 0:
            base[normalized_key] = numeric
    return base


def _format_tz_suffix(dt: datetime.datetime) -> str:
    if not LOG_TZINFO:
        return ""
    tz = dt.tzinfo
    if tz is None:
        return ""
    offset = tz.utcoffset(dt)
    if offset is None:
        return ""
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"

def _current_log_time() -> datetime.datetime:
    now = datetime.datetime.now(datetime.timezone.utc)
    if LOG_TZINFO is not None:
        return now.astimezone(LOG_TZINFO)
    return now

def log(msg: str, color=Fore.WHITE):
    now = _current_log_time()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    tz_suffix = _format_tz_suffix(now)
    if tz_suffix:
        stamp = f"{stamp} {tz_suffix}"
    record = f"[{stamp}] {msg}"
    _LOG_HISTORY.append(record)
    print(color + record + Style.RESET_ALL)
    _append_main_log(record)
    if (
        color in (Fore.YELLOW, Fore.RED)
        or msg.startswith("[WARN]")
        or msg.startswith("[ERROR]")
        or msg.startswith("[FAIL]")
        or "[EX] fail" in msg
        or "Traceback" in msg
    ):
        _append_error_log(record)
    if TELEGRAM_FORWARD_LOGS:
        _enqueue_tg_log(record)


def _maybe_rotate_file(path: Path, max_bytes: int, backups: int) -> None:
    if max_bytes <= 0:
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
    # Keep all rotated files; do not prune older archives.
    # External tooling (e.g., logrotate) can be used if cleanup is desired.


def _append_main_log(text: str) -> None:
    if not MAIN_LOG_ENABLED:
        return
    path = MAIN_LOG_PATH
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _maybe_rotate_file(path, MAIN_LOG_MAX_BYTES, MAIN_LOG_BACKUPS)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")
    except Exception:
        pass


def _append_error_log(text: str) -> None:
    if not ERROR_LOG_ENABLED:
        return
    path = ERROR_LOG_PATH
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _maybe_rotate_file(path, ERROR_LOG_MAX_BYTES, ERROR_LOG_BACKUPS)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")
    except Exception:
        pass
    try:
        severity = "INFO"
        if "[ERROR]" in text or text.startswith("[ERROR]") or "[FAIL]" in text:
            severity = "ERROR"
        elif "[WARN]" in text:
            severity = "WARN"
        db_logger.log_error_event(text, severity=severity)
    except Exception:
        pass


class _StdTee:
    """Mirror stdout/stderr into the main rotating log."""
    def __init__(self, original_stream, prefix: str) -> None:
        self._orig = original_stream
        self._prefix = prefix

    def write(self, data) -> None:
        if data is None:
            return
        try:
            self._orig.write(data)
        except Exception:
            pass
        try:
            for line in str(data).splitlines():
                if not line:
                    continue
                _append_main_log(f"{self._prefix} {line}")
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass


_STDIO_TEE_ENABLED = False


def enable_stdio_logging() -> None:
    global _STDIO_TEE_ENABLED
    if _STDIO_TEE_ENABLED:
        return
    try:
        sys.stdout = _StdTee(sys.stdout, "[STDOUT]")
        sys.stderr = _StdTee(sys.stderr, "[STDERR]")
        _STDIO_TEE_ENABLED = True
    except Exception:
        pass


def _append_user_bybit_log(user_id: str | None, text: str) -> None:
    if not user_id:
        return
    try:
        p = Path("runtime") / str(user_id) / "bybit.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        _maybe_rotate_file(p, USER_LOG_MAX_BYTES, USER_LOG_BACKUPS)
        with p.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")
    except Exception:
        pass

def _select_prompt_position(symbol: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    if not MASTER_PROMPT_SHARE:
        return summary or {"has_position": False}
    cached = MASTER_PROMPT_POSITION_CACHE.get(symbol)
    if cached:
        return cached
    value = summary or {"has_position": False}
    MASTER_PROMPT_POSITION_CACHE[symbol] = value
    return value


def _load_master_decision_store() -> dict[str, Any]:
    if not (MASTER_DECISIONS_SHARE and MASTER_DECISIONS_FILE):
        return {}
    try:
        with MASTER_DECISIONS_FILE.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log(f"[WARN] Failed to read {MASTER_DECISIONS_FILE}: {exc}", Fore.YELLOW)
        return {}
    decisions = payload.get("decisions")
    if isinstance(decisions, dict):
        return decisions
    return {}


def _save_master_decision_store(decisions: Mapping[str, Any], *, meta: Mapping[str, Any] | None = None) -> None:
    if not (MASTER_DECISIONS_SHARE and MASTER_DECISIONS_FILE):
        return
    payload: dict[str, Any] = {"decisions": decisions}
    if meta:
        payload.update(meta)
    payload.setdefault("updated_at", datetime.datetime.utcnow().isoformat() + "Z")
    try:
        MASTER_DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with MASTER_DECISIONS_FILE.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        log(f"[WARN] Failed to persist master decision cache: {exc}", Fore.YELLOW)


def _cache_master_decision(symbol: str, decision: Mapping[str, Any]) -> None:
    if not MASTER_DECISIONS_SHARE:
        return
    key = _canonical_decision_symbol(symbol) or symbol
    if not key:
        return
    MASTER_DECISION_CACHE[key] = json.loads(json.dumps(decision))


def _persist_master_decisions(meta: Mapping[str, Any] | None = None) -> None:
    if not (MASTER_DECISIONS_SHARE and MASTER_DECISION_CACHE):
        return
    _save_master_decision_store(MASTER_DECISION_CACHE, meta=meta)


def _load_master_decisions_runtime() -> dict[str, Any]:
    if MASTER_DECISION_CACHE:
        return MASTER_DECISION_CACHE
    loaded = _load_master_decision_store()
    if loaded:
        MASTER_DECISION_CACHE.update(loaded)
    return loaded


def _pull_master_decision(symbol: str) -> dict[str, Any] | None:
    if not MASTER_DECISIONS_SHARE:
        return None
    key = _canonical_decision_symbol(symbol) or symbol
    if not key:
        return None
    cache = _load_master_decisions_runtime()
    entry = cache.get(key)
    if not entry:
        return None
    return json.loads(json.dumps(entry))

LIMIT_ORDER_FALLBACK_SECONDS = float(os.getenv("LIMIT_ORDER_FALLBACK_SECONDS", "30"))
LIMIT_ORDER_PENDING: dict[tuple[str, str], dict[str, Any]] = {}



def record_pending_entry(symbol: str, qty_value: float, side_value: str, user_id: str | None = None) -> None:
    try:
        if not (qty_value and float(qty_value) > 0):
            return
    except Exception:
        return
    key = _pending_entry_key(symbol, user_id)
    LIMIT_ORDER_PENDING[key] = {
        "ts": time.time(),
        "qty": float(qty_value),
        "side": (side_value or "buy").lower(),
    }

def clear_pending_entry(symbol: str, user_id: str | None = None) -> None:
    LIMIT_ORDER_PENDING.pop(_pending_entry_key(symbol, user_id), None)


def _pending_entry_key(symbol: str, user_id: str | None = None) -> tuple[str, str]:
    user_key = user_id or os.getenv("BYBITBOT_USER_ID") or "default"
    return (user_key, symbol)
CYCLE_FALLBACK_INTERVAL = 5
PNL_LOOKBACK_HOURS = 168
SPARKLINE_BLOCKS = "▁▂▃▄▅▆▇█"
RESULTS_CLOSED_ORDER_DISPLAY_LIMIT = 10
REQUIRE_TAKE_PROFIT = True
DEFAULT_PARTIAL_TP_SCHEME = [(0.33, 1.2), (0.33, 2.0), (0.34, 3.0)]
DEFAULT_ENTRY_LADDER_SCHEME = [(0.6, 0.0), (0.4, 0.6)]
PARTIAL_TP_SCHEME = list(DEFAULT_PARTIAL_TP_SCHEME)
ENTRY_LADDER_SCHEME = list(DEFAULT_ENTRY_LADDER_SCHEME)
DEFAULT_AUTO_MIN_NOTIONAL: bool = True
DEFAULT_AUTO_MARGIN_SCALE: bool = True
DEFAULT_AUTO_MARGIN_SCALE_RATIO: float = 0.75
DEFAULT_AUTO_MARGIN_CONFIDENCE_MULT: float = 1.15
AUTO_MIN_NOTIONAL: bool = DEFAULT_AUTO_MIN_NOTIONAL
AUTO_MARGIN_SCALE: bool = DEFAULT_AUTO_MARGIN_SCALE
AUTO_MARGIN_SCALE_RATIO: float = DEFAULT_AUTO_MARGIN_SCALE_RATIO
AUTO_MARGIN_CONFIDENCE_MULT: float = DEFAULT_AUTO_MARGIN_CONFIDENCE_MULT
AUTO_DIRECTION_ADJUST_ENABLED: bool = str(os.getenv("AUTO_DIRECTION_ADJUST_ENABLED", "1")).strip().lower() not in {"0", "false", "no"}
AUTO_DIRECTION_REDUCE_FACTOR: float = max(
    0.0, min(1.0, _float_from_env("AUTO_DIRECTION_REDUCE_FACTOR", 0.5))
)
AUTO_DIRECTION_SCALE_FACTOR: float = max(0.0, _float_from_env("AUTO_DIRECTION_SCALE_FACTOR", 0.25))
AUTO_DIRECTION_MIN_QTY: float = max(0.0, _float_from_env("AUTO_DIRECTION_MIN_QTY", 0.0))
AUTO_DIRECTION_MIN_CONFIDENCE: float = max(
    0.0,
    min(1.0, _float_from_env("AUTO_DIRECTION_MIN_CONFIDENCE", 0.65)),
)
DEFAULT_OPEN_MIN_CONFIDENCE: float = 0.7
OPEN_MIN_CONFIDENCE: float = DEFAULT_OPEN_MIN_CONFIDENCE

TELEGRAM_DECISIONS_VERBOSE = False
_LAST_COMMIT_HASH: Optional[str] = None
SYMBOL_RULES_CACHE: dict[str, dict[str, float | None]] = {}
DYNAMIC_SYMBOL_ALIASES: dict[str, str] = {}
CURRENT_RISK_PCT: float = 0.0
DYNAMIC_RISK_ENABLED: bool = True
MIN_DYNAMIC_RISK_PCT: float = 0.0
MAX_DYNAMIC_RISK_PCT: float = 0.0
BREAKEVEN_ENABLED: bool = True
BREAKEVEN_ATR_MULT: float = 0.6
BREAKEVEN_BUFFER_ATR: float = 0.15
TRAILING_DYNAMIC_TRIGGER_ATR: float = 1.4
TRAILING_DYNAMIC_FACTOR: float = 0.65
TRAILING_DYNAMIC_MIN_ATR: float = 0.35
PSEUDOTRAIL_MIN_IMPROVE_ATR: float = 0.35
PSEUDOTRAIL_STOP_LOCK_FACTOR: float = 0.35
PSEUDOTRAIL_TP_EXTEND_FACTOR: float = 0.25
MIN_NEXT_RUN_MINUTES: float = 5.0
MAX_NEXT_RUN_FROM_START_MINUTES: float = 45.0
ONLINE_MIN_NEXT_RUN_MINUTES: float | None = None
ONLINE_MAX_NEXT_RUN_MINUTES: float | None = None
OFFLINE_MIN_NEXT_RUN_MINUTES: float | None = None
OFFLINE_MAX_NEXT_RUN_MINUTES: float | None = None
BACKOFF_MIN_NEXT_RUN_MINUTES: float | None = None
BACKOFF_MAX_NEXT_RUN_MINUTES: float | None = None
PSEUDOTRAIL_MAX_TAKE_EXTENDS: int = 2
PSEUDOTRAIL_MAX_TAKE_SHIFT_ATR_MULT: float = 0.5
PSEUDOTRAIL_POSITION_STALE_PCT: float = 0.03
PSEUDOTRAIL_POSITION_SIZE_STALE_RATIO: float = 0.6
PROTECTION_MAX_PRICE_RATIO: float = 10.0
PROTECTION_MIN_PRICE_RATIO: float = 0.05
IMMEDIATE_CLOSE_ON_BREACH: bool = False
TELEGRAM_FORWARD_LOGS: bool = False
TELEGRAM_LOG_BATCH_SIZE: int = 12
TELEGRAM_LOG_FLUSH_INTERVAL: float = 5.0
TELEGRAM_LOG_RATE_LIMIT_WINDOW: float = 60.0
TELEGRAM_LOG_MAX_MESSAGES_PER_WINDOW: int = 18
TELEGRAM_LOG_THREAD_ID: int | None = None
TELEGRAM_WEBHOOK_URL: str = ""
TELEGRAM_WEBHOOK_HOST: str = "127.0.0.1"
TELEGRAM_WEBHOOK_PORT: int = 0
TELEGRAM_WEBHOOK_PATH: str = "/telegram"
TELEGRAM_WEBHOOK_SECRET: str | None = None
TELEGRAM_ALLOWED_CHAT_IDS: set[int] = set()
TELEGRAM_COMMANDS_LIST: list[dict[str, str]] = []
TELEGRAM_DEFAULT_COMMANDS: list[tuple[str, str]] = [
    ("start", "Приветствие и доступные команды"),
    ("help", "Список доступных команд"),
    ("status", "Текущий статус бота"),
    ("positions", "Открытые позиции"),
    ("risk", "Текущий риск-профиль"),
    ("logs", "Последние события"),
    ("logmode", "Переключить режим Telegram-логов"),
    ("schedule", "Запланировать следующую сессию"),
    ("tokens", "Лимиты OpenAI токенов"),
    ("bybitkey", "Установить BYBIT_API_KEY/BYBIT_API_SECRET"),
    ("adduser", "Создать нового юзер-бота (DM)"),
    ("config", "Настройки бота и окружения"),
    ("sandbox", "Управление песочницами"),
    ("version", "Текущая версия и changelog"),
    ("ai", "Диагностика AI payload"),
]
COMMANDS_HELP_SECTIONS = [
    {
        "key": "core",
        "title": "Основные команды",
        "lines": [
            "/status — текущий статус цикла, equity и расписания",
            "/positions - активные позиции и защитные ордера",
            "/risk - действующие параметры риска и плеча",
            "/logs [N|symbol minutes] - последние логи или фильтр по тикеру (пример: /logs BTC 60)",
            "/logmode [brief|verbose|toggle] - управлять режимом Telegram-логов",
        ],
    },
    {
        "key": "ops",
        "title": "Торговля и диагностика",
        "lines": [
            "/schedule <in|at> — задать вручную следующий запуск",
            "/tokens — бюджет токенов OpenAI и текущий расход",
            "/ai payload [universe|trade] — показать последний запрос/ответ модели",
            "/sandbox — управление песочницами",
        ],
    },
    {
        "key": "admin",
        "title": "Администрирование",
        "lines": [
            "/bybitkey <key> <secret> — заменить API ключи (сохраняются в secrets.env)",
            "/adduser <id> <telegram_id> — добавить нового трейдера",
            "/config — вывести активные переменные окружения",
            "/version — показать changelog текущей версии",
        ],
    },
]
TELEGRAM_RELEASE_THREAD_ID: int | None = None
TELEGRAM_COMMAND_THREAD_ID: int | None = None
TELEGRAM_INPROGRESS_THREAD_ID: int | None = None
TELEGRAM_RESULTS_THREAD_ID: int | None = None
TELEGRAM_STATUS_THREAD_ID: int | None = None
TELEGRAM_TRADE_THREAD_ID: int | None = None
TELEGRAM_SUPPORT_THREAD_ID: int | None = None
TELEGRAM_MESSAGE_PREFIX: str = ""
USER_ID: str = "default"
USER_LABEL: str = "redfaraon"
AI_SUPPORT_MODEL: str = ""
AI_LAST_EXCHANGE: dict[str, Any] = {}
SUPPORT_MAX_CONTEXT_BYTES: int = 4096
SUPPORT_CONTEXT_PATTERNS: tuple[str, ...] = ("*.py", "*.md", "*.txt", "*.yaml", "*.yml")
SUPPORT_CONTEXT_SKIP_DIRS: set[str] = {
    ".git",
    "__pycache__",
    "backups",
    "support_sandboxes",
    "users",
    ".idea",
    ".venv",
}
INPROGRESS_WIP_ENABLED: bool = False
_LAST_INPROGRESS_MESSAGE: str | None = None
USERS_DIR = REPO_ROOT / "users"
USERS_CONFIG_FILE = USERS_DIR / "users.json"
USERS_DEFAULT_SECRET = "secrets.env"
USERS_PUBLIC_ENV_FILE = "public.env"
MAIN_OWNER_CHAT_ID = 775747028
USERBOT_OWNERS: dict[str, int] = {}
USERBOT_DEFAULTS: dict[str, Any] = {
    "timezone": "UTC+3",
    "order_margin": 0.75,
    "risk_pct": 0.005,
    "leverage": 9,
    "min_notional": 0.1,
    "position_mode": "oneway",
    "max_positions": 3,
    "default_next_run": 30.0,
}
DEFAULT_RISK_PCT = float(USERBOT_DEFAULTS.get("risk_pct") or 0.005)
ACTIVE_POSITION_MODE: str = "oneway"
ACTIVE_HEDGE_MODE: bool = False
POSITION_MODE_MISMATCH_STATE: bool | None = None


def _sparkline_from_values(values: Sequence[float]) -> str | None:
    filtered = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not filtered:
        return None
    vmin = min(filtered)
    vmax = max(filtered)
    if math.isclose(vmax, vmin, rel_tol=1e-9, abs_tol=1e-9):
        idx = min(len(SPARKLINE_BLOCKS) // 2, len(SPARKLINE_BLOCKS) - 1)
        return SPARKLINE_BLOCKS[idx] * len(filtered)
    span = vmax - vmin or 1.0
    scale = len(SPARKLINE_BLOCKS) - 1
    spark_chars: list[str] = []
    for value in filtered:
        norm = (value - vmin) / span if span else 0.0
        idx = int(round(norm * scale))
        idx = max(0, min(scale, idx))
        spark_chars.append(SPARKLINE_BLOCKS[idx])
    return "".join(spark_chars)


def _load_bybit_credentials() -> tuple[str | None, str | None]:
    try:
        raw = BYBIT_CREDENTIALS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except Exception as exc:
        log(f"[BYBIT] Не удалось прочитать сохранённые ключи: {exc}", Fore.YELLOW)
        return None, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log("[BYBIT] Файл bybit_credentials.json повреждён, игнорируем.", Fore.YELLOW)
        return None, None
    if not isinstance(data, dict):
        return None, None
    key = data.get("apiKey")
    secret = data.get("apiSecret")
    return (str(key).strip() or None) if key else None, (str(secret).strip() or None) if secret else None


def _store_bybit_credentials(api_key: str | None, api_secret: str | None) -> None:
    if not api_key or not api_secret:
        try:
            BYBIT_CREDENTIALS_FILE.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            log(f"[BYBIT] Не удалось удалить сохранённые ключи: {exc}", Fore.YELLOW)
        os.environ.pop("BYBIT_API_KEY", None)
        os.environ.pop("BYBIT_API_SECRET", None)
        return
    data = {"apiKey": api_key, "apiSecret": api_secret}
    try:
        BYBIT_CREDENTIALS_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(BYBIT_CREDENTIALS_FILE, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass
    except Exception as exc:
        log(f"[BYBIT] Не удалось сохранить ключи: {exc}", Fore.YELLOW)
        return
    os.environ["BYBIT_API_KEY"] = api_key
    os.environ["BYBIT_API_SECRET"] = api_secret


def _mask_api_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return value[0] + "*" * (len(value) - 1)
    return f"{value[:3]}***{value[-3:]}"


def _mask_sensitive(value: str) -> str:
    if not value:
        return ""
    stripped = value.strip()
    if len(stripped) <= 6:
        return stripped[0] + "*" * max(0, len(stripped) - 1)
    return f"{stripped[:3]}***{stripped[-3:]}"


def _load_users_config() -> dict[str, Any]:
    if not USERS_CONFIG_FILE.exists():
        return {"users": []}
    try:
        raw = USERS_CONFIG_FILE.read_text(encoding="utf-8")
    except Exception as exc:
        log(f"[USERS] Не удалось прочитать {USERS_CONFIG_FILE}: {exc}", Fore.YELLOW)
        return {"users": []}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log(f"[USERS] Некорректный JSON в {USERS_CONFIG_FILE}: {exc}", Fore.YELLOW)
        return {"users": []}
    return data if isinstance(data, dict) else {"users": []}


def _save_users_config(config: dict[str, Any]) -> None:
    try:
        USERS_DIR.mkdir(parents=True, exist_ok=True)
        USERS_CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"[USERS] Не удалось сохранить {USERS_CONFIG_FILE}: {exc}", Fore.YELLOW)


def _ensure_user_entry(
    user_id: str,
    label: str,
    *,
    state_dir: str,
    public_env_file: str,
    owner_id: Optional[int] = None,
) -> None:
    config = _load_users_config()
    users_list = config.setdefault("users", [])
    if not isinstance(users_list, list):
        users_list = []
        config["users"] = users_list
    entry = None
    for existing in users_list:
        if isinstance(existing, dict) and existing.get("id") == user_id:
            entry = existing
            break
    rel_public_path = str(Path(public_env_file))
    if entry is None:
        entry = {
            "id": user_id,
            "label": label,
            "enabled": True,
            "state_dir": state_dir,
            "env": {},
            "env_files": [rel_public_path],
        }
        users_list.append(entry)
    else:
        entry["label"] = label
        entry["enabled"] = True
        entry["state_dir"] = state_dir
        env_files = entry.get("env_files")
        if not isinstance(env_files, list):
            env_files = []
        if rel_public_path not in env_files:
            env_files.append(rel_public_path)
        entry["env_files"] = env_files
        env_overrides = entry.get("env")
        if not isinstance(env_overrides, dict):
            entry["env"] = {}
    if owner_id is not None:
        entry["owner_id"] = int(owner_id)
    _save_users_config(config)
    # Ensure bybit.log exists for this user
    try:
        if user_id:
            _ensure_user_bybit_log_for_all()
    except Exception:
        pass


def _write_user_secrets(user_id: str, api_key: str, api_secret: str) -> Path:
    user_dir = USERS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    secrets_path = user_dir / USERS_DEFAULT_SECRET
    content = f"BYBIT_API_KEY={api_key}\nBYBIT_API_SECRET={api_secret}\n"
    try:
        secrets_path.write_text(content, encoding="utf-8")
        try:
            os.chmod(secrets_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass
    except Exception as exc:
        raise RuntimeError(f"Не удалось сохранить secrets.env: {exc}") from exc
    public_env_path = user_dir / USERS_PUBLIC_ENV_FILE
    if not public_env_path.exists():
        try:
            public_env_path.write_text("# Дополнительные настройки для пользователя\n", encoding="utf-8")
        except Exception:
            pass
    return secrets_path


def _refresh_userbot_owners() -> None:
    global USERBOT_OWNERS
    owners: dict[str, int] = {}
    config = _load_users_config()
    for entry in config.get("users") or []:
        if not isinstance(entry, dict):
            continue
        bot_id = str(entry.get("id") or "").strip()
        if not bot_id:
            continue
        owner_val = safe_int(entry.get("owner_id"))
        if owner_val is None:
            continue
        owners[bot_id] = owner_val
    if MAIN_OWNER_CHAT_ID is not None:
        owners.setdefault("default", MAIN_OWNER_CHAT_ID)
    USERBOT_OWNERS = owners

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

BASE_INDICATOR_MIN_COUNT = 6
BASE_INDICATOR_MAX_COUNT = 8
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

AI_INITIAL_TIMEFRAMES: list[str] = []
AI_INITIAL_TF_DEPTHS: dict[str, int] = {}
AI_INITIAL_INDICATOR_POOL: list[str] = list(BASE_INDICATOR_CANDIDATES)
AI_INITIAL_EMA_COUNT: int = 2
AI_INITIAL_EXTRA_INDICATOR_COUNT: int = 5
AI_INITIAL_NEWS_PROVIDER: str = ""
AI_INITIAL_NEWS_LIMIT: int = 0
AI_NEEDS_TF_DEPTHS: dict[str, int] = {}

BASE_TIMEFRAME_CANDIDATES = ["5m", "15m", "30m", "1h", "2h", "4h", "1d"]
PAIR_TICKER_MAP = {pair: pair.split("/")[0].split(":")[0].upper() for pair in BASE_PAIR_CANDIDATES}
RECOGNIZED_STABLE_QUOTES = ("USDT", "USDC")
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
    "SHIB/USDT:USDT": "SHIB1000/USDT:USDT",
    "SHIB/USDT": "SHIB1000/USDT:USDT",
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


def _build_dynamic_symbol_aliases(markets: dict[str, Any]) -> dict[str, str]:
    dynamic: dict[str, str] = {}
    suffixes = ("1000", "100", "10000")
    for symbol, market in (markets or {}).items():
        if not isinstance(symbol, str):
            continue
        market = market or {}
        base = str(market.get("base") or symbol.split("/")[0]).strip()
        quote = str(market.get("quote") or "USDT").strip()
        settlement = None
        if "/" in symbol:
            right = symbol.split("/", 1)[1]
            if ":" in right:
                parts = [part for part in right.split(":") if part]
                if parts:
                    quote = parts[0]
                if len(parts) >= 2:
                    settlement = parts[1]
            else:
                quote = right
        base_upper = base.upper()
        quote_upper = quote.upper()
        settlement_upper = settlement.upper() if settlement else None
        alias_bases: set[str] = set()
        for suffix in suffixes:
            if base_upper.endswith(suffix) and len(base_upper) > len(suffix) + 1:
                alias_bases.add(base_upper[:-len(suffix)])
        if base_upper.endswith("PERP") and len(base_upper) > 4:
            alias_bases.add(base_upper[:-4])
        if base_upper.startswith("1000") and len(base_upper) > 4:
            alias_bases.add(base_upper[4:])
        for alias_base in alias_bases:
            if not alias_base or alias_base == base_upper:
                continue
            alias_forms = set()
            core = f"{alias_base}/{quote_upper}"
            alias_forms.add(core)
            if settlement_upper:
                alias_forms.add(f"{core}:{settlement_upper}")
            if quote_upper != "USDT":
                alias_forms.add(f"{alias_base}/{quote_upper}:USDT")
            alias_forms.add(f"{alias_base}/USDT:USDT")
            alias_forms.add(f"{alias_base}/USDT")
            for form in list(alias_forms):
                if form.endswith(":USDT"):
                    alias_forms.add(form[:-6])
            for alias in alias_forms:
                if alias and alias != symbol:
                    dynamic.setdefault(alias, symbol)
                    dynamic.setdefault(alias.upper(), symbol)
    return dynamic


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
        log(f"[INFO] Created backup {backup_path.name}", Fore.LIGHTBLACK_EX)
    except Exception as exc:
        log(f"[WARN] Failed to create backup '{backup_path.name}': {exc}", Fore.YELLOW)


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


def _round_qty_up(value: float, step: float) -> float:
    if step is None or step <= 0:
        return value
    if value <= 0:
        return 0.0
    steps = math.ceil(value / step)
    return max(0.0, steps * step)


def _round_qty_down(value: float, step: float) -> float:
    if step is None or step <= 0:
        return value
    if value <= 0:
        return 0.0
    steps = math.floor(value / step)
    return max(0.0, steps * step)


def _apply_qty_rules(qty: float, *, min_qty: float = 0.0, qty_step: float = 0.0) -> float:
    result = max(0.0, qty)
    if min_qty and result < min_qty:
        result = min_qty
    if qty_step and qty_step > 0:
        result = _round_qty_up(result, qty_step)
    return result


def _compute_vwma(df: pd.DataFrame, length: int) -> pd.Series:
    if df.empty or length <= 0:
        return pd.Series(dtype=float, index=df.index)
    weighted = df["close"] * df["volume"]
    numerator = weighted.rolling(length).sum()
    denominator = df["volume"].rolling(length).sum()
    vwma_series = numerator / denominator
    return vwma_series


def _compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    if df.empty or period <= 0 or len(df) < period:
        return pd.Series(dtype=float, index=df.index)
    atr_series = atr(df, period)
    if atr_series is None:
        return pd.Series(dtype=float, index=df.index)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper_band = hl2 + multiplier * atr_series
    lower_band = hl2 - multiplier * atr_series
    values: list[float] = []
    prev_value: float | None = None
    trend_up = True
    closes = df["close"].tolist()
    for idx, close_price in enumerate(closes):
        ub = upper_band.iloc[idx] if idx < len(upper_band) else math.nan
        lb = lower_band.iloc[idx] if idx < len(lower_band) else math.nan
        if math.isnan(ub) or math.isnan(lb):
            values.append(prev_value if prev_value is not None else math.nan)
            continue
        if prev_value is None:
            prev_value = ub
            trend_up = close_price >= lb
            values.append(prev_value)
            continue
        if close_price > prev_value:
            trend_up = True
        elif close_price < prev_value:
            trend_up = False
        if trend_up:
            candidate = lb if prev_value is None else max(lb, prev_value)
        else:
            candidate = ub if prev_value is None else min(ub, prev_value)
        prev_value = candidate
        values.append(candidate)
    return pd.Series(values, index=df.index)


def _clamp_qty_to_max_notional(
    qty: float,
    price: float,
    max_notional: float,
    qty_step: float = 0.0,
) -> float:
    if price <= 0 or max_notional <= 0 or qty <= 0:
        return 0.0 if max_notional <= 0 else max(0.0, min(qty, max_notional / price if price > 0 else qty))
    max_qty = max_notional / price
    if qty <= max_qty + NOTIONAL_EPSILON:
        return qty
    if qty_step and qty_step > 0:
        clamped = _round_qty_down(max_qty, qty_step)
        return clamped
    return max(0.0, max_qty)


def is_main_owner(user_id: Optional[int]) -> bool:
    return user_id == MAIN_OWNER_CHAT_ID if user_id is not None else False


def is_bot_owner(user_id: Optional[int], bot_id: str) -> bool:
    if user_id is None:
        return False
    if is_main_owner(user_id):
        return True
    return USERBOT_OWNERS.get(bot_id) == user_id


def _normalize_order_side(side: str | None, amount: float | None = None) -> tuple[str, bool]:
    normalized = (side or "").strip().lower()
    autodetected = False
    if normalized not in {"buy", "sell"}:
        if amount is not None and math.isfinite(amount):
            normalized = "buy" if amount >= 0 else "sell"
        else:
            normalized = "buy"
        autodetected = True
    return normalized, autodetected


def get_current_branch_name() -> str | None:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(SCRIPT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    if not branch or branch == "HEAD":
        return None
    return branch


def _format_commit_timestamp(iso_text: str | None) -> str | None:
    if not iso_text:
        return None
    iso_clean = iso_text.replace("Z", "+00:00")
    try:
        dt_obj = datetime.datetime.fromisoformat(iso_clean)
    except (TypeError, ValueError):
        return iso_text
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
    local_tz = _current_local_tz() or datetime.datetime.now().astimezone().tzinfo
    local_dt = dt_obj.astimezone(local_tz)
    return local_dt.strftime("%Y-%m-%d %H:%M %Z")


def _load_cycle_state() -> dict[str, Any]:
    try:
        raw = CYCLE_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"total_cycles": 0, "fallback_cycles": 0}
    except Exception:
        return {"total_cycles": 0, "fallback_cycles": 0}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"total_cycles": 0, "fallback_cycles": 0}
    if not isinstance(data, dict):
        return {"total_cycles": 0, "fallback_cycles": 0}
    total_cycles = safe_int(data.get("total_cycles")) or 0
    fallback_cycles = safe_int(data.get("fallback_cycles")) or 0
    data["total_cycles"] = total_cycles
    data["fallback_cycles"] = fallback_cycles
    return data


def _save_cycle_state(state: dict[str, Any]) -> None:
    try:
        CYCLE_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _sync_fallback_history_counters(
    total_cycles: int,
    fallback_cycles: int,
) -> None:
    try:
        raw = FALLBACK_HISTORY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        history = {}
    except Exception:
        return
    else:
        try:
            history = json.loads(raw)
        except json.JSONDecodeError:
            history = {}
    if not isinstance(history, dict):
        history = {}
    changed = False
    if history.get("routine_counter") != total_cycles:
        history["routine_counter"] = total_cycles
        changed = True
    if bool(history.get("fallback_active")):
        if history.get("fallback_cycles") != fallback_cycles:
            history["fallback_cycles"] = fallback_cycles
            changed = True
    if changed:
        try:
            FALLBACK_HISTORY_FILE.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def _record_cycle_completion(
    base_state: dict[str, Any],
    *,
    cycle_kind: str | None,
    cycle_mode: str | None,
    branch_name: str | None,
    commit_hash: str | None,
    commit_timestamp: str | None,
) -> dict[str, Any]:
    state = dict(base_state or {})
    total_cycles = safe_int(state.get("total_cycles")) or 0
    total_cycles += 1
    state["total_cycles"] = total_cycles
    kind_normalized = (cycle_kind or "normal").strip().lower()
    mode_normalized = (cycle_mode or "last").strip().lower()
    if kind_normalized != "normal":
        fallback_cycles = safe_int(state.get("fallback_cycles")) or 0
        fallback_cycles += 1
        state["fallback_cycles"] = fallback_cycles
    else:
        state["fallback_cycles"] = 0
    state["last_cycle"] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kind": kind_normalized or "normal",
        "mode": mode_normalized or "last",
        "branch": branch_name or None,
        "commit": commit_hash or None,
        "commit_timestamp": commit_timestamp or None,
    }
    state.pop("next_cycle_prepared", None)
    _save_cycle_state(state)
    _sync_fallback_history_counters(total_cycles, state.get("fallback_cycles") or 0)
    return state


def _pnl_window_label() -> str:
    hours = float(PNL_LOOKBACK_HOURS)
    if hours % 24 == 0 and hours >= 24:
        days = int(hours // 24)
        return f"{days}d"
    if hours.is_integer():
        return f"{int(hours)}h"
    return f"{hours:.1f}h"


def _emit_realized_pnl_message(stage_label: str, value: float | None, fill_count: int | None) -> None:
    if value is None or not math.isfinite(value):
        return
    fills_suffix = ""
    if fill_count:
        fills_suffix = f" ({fill_count} fills)"
    stage_clean = (stage_label or "").strip()
    window_label = _pnl_window_label()
    if stage_clean:
        label_text = f"{window_label} closed, {stage_clean}"
    else:
        label_text = f"{window_label} closed"
    message = f"PnL ({label_text}): {value:+.2f} USDT{fills_suffix}"
    colour = Fore.CYAN if value >= 0 else Fore.YELLOW
    log(message, colour)
    try:
        send_tg(message)
    except Exception:
        pass


def _emit_unrealized_pnl_message(stage_label: str, value: float, positions_count: int) -> None:
    stage_clean = (stage_label or "").strip()
    label_text = "open unrealized"
    if stage_clean:
        label_text = f"{label_text}, {stage_clean}"
    positions_suffix = ""
    if positions_count:
        suffix_word = "position" if positions_count == 1 else "positions"
        positions_suffix = f" ({positions_count} {suffix_word})"
    message = f"PnL ({label_text}): {value:+.2f} USDT{positions_suffix}"
    colour = Fore.CYAN if value >= 0 else Fore.YELLOW
    log(message, colour)
    try:
        send_tg(message)
    except Exception:
        pass
def _pick_positive_float(*values) -> float | None:
    for value in values:
        candidate = safe_float(value)
        if candidate is not None and candidate > 0:
            return float(candidate)
    return None


def _get_symbol_trade_rules(exchange, symbol: str) -> dict[str, float | None]:
    canonical_symbol = _resolve_symbol_alias(symbol) or symbol
    cache_key = canonical_symbol
    cached = SYMBOL_RULES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    symbols_to_try = []
    for candidate in (symbol, canonical_symbol):
        if candidate and candidate not in symbols_to_try:
            symbols_to_try.append(candidate)
            if isinstance(candidate, str) and candidate.upper().endswith(":USDT"):
                without_settle = candidate.split(":")[0]
                if without_settle and without_settle not in symbols_to_try:
                    symbols_to_try.append(without_settle)
    markets = getattr(exchange, "markets", {}) or {}
    market = None
    for candidate in symbols_to_try:
        try:
            market = exchange.market(candidate)
        except Exception:
            market = None
        if isinstance(market, dict):
            break
        market = markets.get(candidate)
        if isinstance(market, dict):
            break
    if not isinstance(market, dict):
        for candidate in symbols_to_try:
            match = next((markets[key] for key in markets if key and key.upper() == candidate.upper()), None)
            if isinstance(match, dict):
                market = match
                break
    rules: dict[str, float | None] = {
        "min_qty": None,
        "min_notional": None,
        "qty_step": None,
        "price_step": None,
        "contract_size": None,
    }
    if isinstance(market, dict):
        limits = market.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        cost_limits = limits.get("cost") or {}
        price_limits = limits.get("price") or {}
        precision = market.get("precision") or {}
        info = market.get("info") or {}
        lot_filter = info.get("lotSizeFilter") or info.get("lot_size_filter") or {}
        price_filter = info.get("priceFilter") or info.get("price_filter") or {}
        min_qty = _pick_positive_float(
            amount_limits.get("min"),
            market.get("minAmount"),
            market.get("lotSize"),
            precision.get("amount"),
            lot_filter.get("minOrderQty"),
            lot_filter.get("minTradingQty"),
            info.get("minOrderQty"),
            info.get("minTradingQty"),
            info.get("minQty"),
            info.get("min_qty"),
            info.get("minLimitOrderQty"),
            info.get("min_limit_order_qty"),
            info.get("minTradeQty"),
        )
        qty_step = _pick_positive_float(
            amount_limits.get("step"),
            precision.get("amount"),
            lot_filter.get("qtyStep"),
            lot_filter.get("stepSize"),
            info.get("qtyStep"),
            info.get("stepSize"),
            info.get("step_size"),
            info.get("qty_step"),
        )
        min_notional = _pick_positive_float(
            cost_limits.get("min"),
            info.get("minNotional"),
            lot_filter.get("minOrderValue"),
            price_filter.get("minOrderValue"),
            info.get("minOrderValue"),
        )
        price_step = _pick_positive_float(
            price_limits.get("min"),
            price_limits.get("step"),
            price_limits.get("tickSize"),
            precision.get("price"),
            price_filter.get("tickSize"),
            info.get("tickSize"),
        )
        contract_size = _pick_positive_float(
            market.get("contractSize"),
            market.get("contract_size"),
            info.get("contractSize"),
            info.get("lotSize"),
        )
        if min_qty:
            rules["min_qty"] = float(min_qty)
        if qty_step:
            rules["qty_step"] = float(qty_step)
        if min_notional:
            rules["min_notional"] = float(min_notional)
        if price_step:
            rules["price_step"] = float(price_step)
        if contract_size:
            rules["contract_size"] = float(contract_size)
    SYMBOL_RULES_CACHE[cache_key] = rules
    if cache_key != symbol:
        SYMBOL_RULES_CACHE[symbol] = rules
    return rules


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
    market = None
    try:
        market = exchange.market(symbol)
    except Exception:
        market = None
    if isinstance(market, dict):
        market_type = market.get("type") or ("linear" if market.get("linear") else "inverse" if market.get("inverse") else None)
        if market_type not in ("swap", "future", "linear", "inverse"):
            return
    current_lev = safe_float((current_position or {}).get("leverage") if isinstance(current_position, dict) else None)
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


def _extract_decision_pct(decision, symbol_meta=None):
    """Extract a percentage size reference from AI payload (0-1)."""
    def normalize(raw):
        value = safe_float(raw)
        if value is None:
            return None
        if value > 1 and value <= 100:
            value = value / 100.0
        if value <= 0:
            return None
        return min(value, 1.0)

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
        for key in ("size_pct", "notional_pct", "pct"):
            pct = normalize(source.get(key))
            if pct is not None:
                return pct
    return None

def _select_risk_budget_base(equity: float, available_margin: float) -> float:
    """Pick a sane risk budget base even if one of the inputs is zero or missing."""
    eq = float(equity) if equity and math.isfinite(equity) else 0.0
    margin = float(available_margin) if available_margin and math.isfinite(available_margin) else 0.0
    if eq > 0 and margin > 0:
        return min(eq, margin)
    return margin if margin > 0 else eq

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
    latest_block = None
    for block in blocks:
        first_line = block.splitlines()[0].strip()
        if first_line and first_line[0].isdigit():
            latest_block = block
            break
    if latest_block is None:
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


def _resolve_commit_metadata(ref: str | None) -> tuple[str | None, str | None, str | None]:
    target = (ref or "").strip()
    if not target:
        return None, None, None
    cmd = ["git", "show", "-s", "--format=%H%x1f%s%x1f%cI", target]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=SCRIPT_DIR,
        )
    except Exception:
        return None, None, None
    raw = result.stdout.strip()
    if not raw:
        return None, None, None
    parts = raw.split("\x1f")
    commit_hash = parts[0].strip() if parts else ""
    commit_msg = parts[1].strip() if len(parts) > 1 else ""
    commit_ts = parts[2].strip() if len(parts) > 2 else ""
    return (
        commit_hash or None,
        _sanitize_commit_subject(commit_msg),
        commit_ts or None,
    )


def _sanitize_commit_subject(message: str | None) -> Optional[str]:
    if not message:
        return None
    return message.strip().splitlines()[0].strip() or None


def _notify_release_event(
    event_type: str,
    commit_hash: str | None,
    commit_message: str | None,
    commit_timestamp: str | None,
) -> None:
    release_state = _load_release_state()
    if event_type == "release":
        last_version = release_state.get("last_version")
        last_release_hash = release_state.get("last_release_hash")
        if BOT_VERSION and last_version == BOT_VERSION:
            if not commit_hash or not last_release_hash or commit_hash == last_release_hash:
                return
    elif event_type == "commit":
        last_commit_hash = release_state.get("last_commit_hash")
        if commit_hash and commit_hash == last_commit_hash:
            return

    thread_target: Optional[int] = TELEGRAM_RELEASE_THREAD_ID
    if thread_target is None:
        thread_target = TG_GIT_TOPIC_ID if TG_GIT_TOPIC_ID is not None else TG_TOPIC_ID
    header = "🆕 Новый релиз" if event_type == "release" else "🆕 Новый коммит"
    lines = [header]
    formatted_ts = _format_commit_timestamp(commit_timestamp) if commit_timestamp else None
    if formatted_ts:
        lines.append(f"Дата: {formatted_ts}")
    elif commit_timestamp:
        lines.append(f"Дата (UTC): {commit_timestamp}")
    if commit_hash:
        lines.append(f"Коммит: {commit_hash[:8]} ({commit_hash})")
    if commit_message:
        lines.append(f"Сообщение: {commit_message}")
    lines.append(f"Версия: {BOT_VERSION}")
    if BOT_CHANGELOG:
        lines.append("Changelog:")
        lines.append(BOT_CHANGELOG)
    send_tg(
        "\n".join(lines),
        thread_id=thread_target,
        no_log_forward=True,
        no_prefix=True,
    )
    release_state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if event_type == "release":
        release_state["last_version"] = BOT_VERSION
        if commit_hash:
            release_state["last_release_hash"] = commit_hash
    elif event_type == "commit" and commit_hash:
        release_state["last_commit_hash"] = commit_hash
    _save_release_state(release_state)


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
    commit_hash_meta, commit_message_meta, commit_timestamp_meta = _resolve_commit_metadata(head)
    if version_changed:
        ensure_version_backup()
        log(f"🆕 Обнаружена новая версия: {previous_version} > {BOT_VERSION}", Fore.LIGHTBLUE_EX)
        release_thread = TELEGRAM_RELEASE_THREAD_ID if TELEGRAM_RELEASE_THREAD_ID is not None else TG_TOPIC_ID
        send_tg(
            f"🆕 Обновлена версия до {BOT_VERSION}",
            thread_id=release_thread,
            no_log_forward=True,
            no_prefix=True,
        )
        if head:
            _notify_release_event("release", commit_hash_meta, commit_message_meta, commit_timestamp_meta)
    elif metadata_changed:
        log("🆕 Обновлён changelog без изменения версии.", Fore.LIGHTBLACK_EX)
        release_thread = TELEGRAM_RELEASE_THREAD_ID if TELEGRAM_RELEASE_THREAD_ID is not None else TG_TOPIC_ID
        send_tg(
            "🆕 Обновлён changelog без изменения версии.",
            thread_id=release_thread,
            no_log_forward=True,
            no_prefix=True,
        )
    elif commit_changed and previous_hash is not None:
        log("🆕 Обновлена HEAD коммита без изменения changelog.", Fore.LIGHTBLACK_EX)
        if head:
            _notify_release_event("commit", commit_hash_meta, commit_message_meta, commit_timestamp_meta)

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


def _log_news_digest(digest: dict[str, Any] | None) -> None:
    if not isinstance(digest, dict) or not digest:
        log("[NEWS] No headlines fetched for this cycle.", Fore.LIGHTBLACK_EX)
        return
    parts = []
    for sym, payload in digest.items():
        items = payload.get("items") if isinstance(payload, dict) else None
        title = None
        if items and isinstance(items, list) and items:
            title = (items[0] or {}).get("title")
        parts.append(f"{sym}: {len(items or [])} items" + (f" — {title}" if title else ""))
    if parts:
        log("[NEWS] Headlines used in prompts: " + " | ".join(parts[:12]), Fore.LIGHTBLACK_EX)


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
        if base == "macd":
            fast = 12
            slow = 26
            signal = 9
            fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
            slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            hist_col = "macd"
            df["macd_line"] = macd_line
            df["macd_signal"] = signal_line
            df[hist_col] = macd_line - signal_line
            return hist_col
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
        if base == "vwma":
            period = length or 20
            col = f"vwma{period}"
            df[col] = _compute_vwma(df, period)
            return col
        if base == "vol":
            col = "vol"
            df[col] = df["volume"]
            return col
        if base == "bbands":
            period = length or 20
            col = f"bbands{period}"
            middle = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[col] = middle
            upper = middle + std * 2
            lower = middle - std * 2
            df[f"{col}_upper"] = upper
            df[f"{col}_lower"] = lower
            return col
        if base == "supertrend":
            period = length or 10
            col = f"supertrend{period}"
            df[col] = _compute_supertrend(df, period=period, multiplier=3.0)
            return col
        if base in ("stochrsi", "stochrs", "stochr"):
            period = length or 14
            rsi_series = rsi(df["close"], period)
            col = f"stochrsi{period}"
            rsi_min = rsi_series.rolling(period).min()
            rsi_max = rsi_series.rolling(period).max()
            denom = (rsi_max - rsi_min).replace(0, pd.NA)
            df[col] = 100 * (rsi_series - rsi_min) / denom
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


def _serialize_df(df: pd.DataFrame, limit: int = 60):
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


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        df = df.reset_index()
        if "timestamp" not in df.columns and df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception:
                pass
    return df


def _ensure_initial_indicators(
    df: pd.DataFrame,
    indicator_pool: Sequence[str],
    ema_count: int,
    extra_count: int,
) -> list[str]:
    selected: list[str] = []
    remaining = []
    for candidate in indicator_pool or []:
        normalized = str(candidate).strip()
        if not normalized:
            continue
        normalized_lower = normalized.lower()
        if normalized_lower.startswith("ema"):
            if len(selected) < ema_count:
                col = _apply_indicator_to_df(df, normalized_lower)
                if col and col not in selected:
                    selected.append(col)
            else:
                remaining.append(normalized_lower)
        else:
            remaining.append(normalized_lower)
    fallback_emas = ["ema20", "ema50", "ema100"]
    for fallback in fallback_emas:
        if len(selected) >= ema_count:
            break
        if fallback not in selected:
            col = _apply_indicator_to_df(df, fallback)
            if col and col not in selected:
                selected.append(col)
    extras: list[str] = []
    for candidate in remaining:
        if len(extras) >= extra_count:
            break
        normalized_lower = candidate.lower()
        if normalized_lower in {"volume", "vol"}:
            continue
        col = _apply_indicator_to_df(df, normalized_lower)
        if col and col not in selected and col not in extras:
            extras.append(col)
    return selected + extras


def _load_initial_dataframe(
    exchange,
    symbol: str,
    tf: str,
    df_primary: pd.DataFrame | None,
    primary_frame: str,
):
    if tf == primary_frame and isinstance(df_primary, pd.DataFrame) and not df_primary.empty:
        df_used = df_primary.copy()
    else:
        try:
            df_used = fetch_df(exchange, symbol, tf)
        except Exception as exc:
            log(f"[WARN] Failed to fetch {tf} for initial context: {exc}", Fore.YELLOW)
            return None
    df_used = df_used.copy()
    return _ensure_timestamp_column(df_used)


def ai_update_universe(exchange, symbols, positions_map, equity, available_margin, universe_cache, news_digest=None):
    client = _create_ai_client(timeout=30, context="universe update")
    if client is None:
        _emit_ai_offline_notice("client init failed (universe update)")
        return None
    if not news_digest:
        news_digest = _build_news_digest(symbols)
    _log_news_digest(news_digest)
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
        "candidate_pairs": [],
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
        "session_timezone": timestamp_now.tzname() or "UTC",
        "universe_cache": None,
    }
    system_msg = (
        "You are an AI portfolio strategist for Bybit. Build the trading universe for the next cycle using ONLY:"
        " (1) the current session date/time/timezone, (2) the supplied news headlines, and (3) any open exposure listed."
        " Do NOT reuse previous universes or cached pairs; every run must produce a fresh shortlist."
        " Respond strictly in JSON with keys:\n"
        '  "pairs": list of up to 8 symbols to analyze this cycle (mix bullish/bearish narratives based on news),\n'
        '  "timeframes": list of exactly two short timeframes (e.g., "30m","4h") to apply globally, plus optional "initial_timeframes" for the first pass,\n'
        '  "indicators": list containing six to eight items ({"indicator":"ema","length":20}, {"indicator":"ema","length":50}, "volume", "rsi14", "macd", plus one or two additional momentum/volatility indicators) that will drive the primary data bundles,\n'
        '  "initial_timeframes": list of up to two primary timeframes to inspect first (e.g., ["30m","4h"]),\n'
        '  "aggression": risk posture label (e.g., conservative, balanced, optimal, aggressive),\n'
        '  "style": trading style label (e.g., balanced_intraday, momentum, risk-off) so the follow-up trade prompt can adopt a matching voice,\n'
        '  "trade_horizon": trading horizon label (e.g., scalping, intraday, short-term, midterm),\n'
        '  "max_positions": integer cap for concurrently open symbols (factor in current exposure + liquidity),\n'
        '  "next_run_time": ISO timestamp for the next cycle expressed in UTC+03:00 (include "+03:00" or "Z"); must land between 5 and 40 minutes from the cycle start,\n'
        '  "next_run_minutes": REQUIRED float fallback delay strictly between 5 and 40 minutes (drive toward 5 when volatility/news/aggression is high, stretch toward 40 when markets are calm). Always return this even if next_run_time is present,\n'
        '  "notes": optional rationale describing how the news/indicators shaped this universe.'
        " Also include optional field 'news_requests' (symbols needing full news text)."
        " All of the returned metadata (timeframes, indicators, aggression, style, horizon, max positions, next run timing) will be applied directly to downstream initial requests and scheduling."
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
    context_label = "universe update"
    context_key = "universe_update"
    if not _ensure_token_budget(token_estimate, AI_MODEL, context_label):
        return None
    _log_ai_request(AI_MODEL, token_estimate, context_label)
    try:
        res = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except RateLimitError as exc:
        # Universe update is not critical for safety; if rate-limited, enter offline mode.
        if _current_ai_provider() == "openai":
            _switch_ai_provider_to_fallback("rate limit 429 (universe update)")
            retry_client = _create_ai_client(timeout=30, context="universe update (retry)")
            if retry_client is not None:
                try:
                    res = retry_client.chat.completions.create(
                        model=AI_MODEL,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=messages,
                    )
                except Exception:
                    _emit_ai_offline_notice("rate limit 429 (universe update)")
                    return None
            else:
                _emit_ai_offline_notice("rate limit 429 (universe update)")
                return None
        else:
            _emit_ai_offline_notice("rate limit 429 (universe update)")
            return None
    except Exception as exc:
        log(f"[ERROR] AI universe update: {exc}", Fore.RED)
        _emit_ai_offline_notice(f"universe update failed: {type(exc).__name__}")
        _record_ai_exchange(
            context_key,
            label=context_label,
            model=AI_MODEL,
            request=messages,
            error=exc,
            token_estimate=token_estimate,
        )
        return None
    usage = getattr(res, "usage", None)
    _register_ai_usage(AI_MODEL, usage, context_label)
    try:
        payload = res.choices[0].message.content
    except Exception:
        _record_ai_exchange(
            context_key,
            label=context_label,
            model=AI_MODEL,
            request=messages,
            response=None,
            error="empty response",
            usage=usage,
            token_estimate=token_estimate,
        )
        return None
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        log(f"[WARN] JSON decode (universe update): {exc}", Fore.YELLOW)
        _record_ai_exchange(
            context_key,
            label=context_label,
            model=AI_MODEL,
            request=messages,
            response=payload,
            error=f"JSON decode: {exc}",
            usage=usage,
            token_estimate=token_estimate,
        )
        return None
    # Normalize the response into a simple dict
    universe_payload = {}
    universe_payload["pairs"] = (result.get("pairs") or result.get("symbols") or [])[:8]
    universe_payload["timeframes"] = (result.get("timeframes") or ["30m","4h"])[:2]
    raw_indicators = result.get("indicators") or []
    normalized_indicators = []
    for item in raw_indicators:
        if isinstance(item, dict):
            indicator_name = item.get("indicator") or item.get("name")
            length = item.get("length") or item.get("period") or item.get("window")
            if indicator_name:
                normalized_indicators.append({"indicator": str(indicator_name).lower(), "length": safe_int(length)})
        else:
            normalized_indicators.append(str(item).lower())
    normalized_indicators = [ind for ind in normalized_indicators if ind]
    if not normalized_indicators:
        normalized_indicators = [
            {"indicator": "ema", "length": 20},
            {"indicator": "ema", "length": 50},
            {"indicator": "ema", "length": 100},
            "volume",
            "rsi14",
            "macd",
            "atr14",
            "stoch14",
            "supertrend",
        ]
    indicator_cap = max(
        BASE_INDICATOR_MIN_COUNT,
        min(BASE_INDICATOR_MAX_COUNT, len(normalized_indicators)),
    )
    selected_indicators = list(normalized_indicators[:indicator_cap])
    if len(selected_indicators) < BASE_INDICATOR_MIN_COUNT:
        for candidate in BASE_INDICATOR_CANDIDATES:
            if len(selected_indicators) >= BASE_INDICATOR_MIN_COUNT:
                break
            if candidate not in selected_indicators:
                selected_indicators.append(candidate)
    universe_payload["indicators"] = selected_indicators[:BASE_INDICATOR_MAX_COUNT]
    universe_payload["next_run_minutes"] = result.get("next_run_minutes")
    universe_payload["notes"] = result.get("notes")
    raw_initial_timeframes = (
        result.get("initial_timeframes")
        or result.get("initialTimeframes")
        or result.get("initial_tf")
        or result.get("initialTf")
    )
    initial_timeframes: list[str] = []
    if isinstance(raw_initial_timeframes, (list, tuple, set)):
        items = list(raw_initial_timeframes)
    elif isinstance(raw_initial_timeframes, str):
        items = [part.strip() for part in raw_initial_timeframes.split(",")]
    else:
        items = []
    for item in items:
        if not item:
            continue
        normalized_tf = normalize_requested_timeframe(item, default="")
        if normalized_tf and normalized_tf not in initial_timeframes:
            initial_timeframes.append(normalized_tf)
        if len(initial_timeframes) >= 2:
            break
    if not initial_timeframes and universe_payload["timeframes"]:
        initial_timeframes = list(universe_payload["timeframes"][:2])
    universe_payload["initial_timeframes"] = initial_timeframes[:2]
    aggression_level = result.get("aggression") or result.get("riskProfile") or result.get("risk_profile")
    trade_horizon = result.get("trade_horizon") or result.get("tradeHorizon") or result.get("horizon")
    style_label = (
        result.get("style")
        or result.get("trade_style")
        or result.get("tradeStyle")
        or result.get("strategy_style")
        or result.get("strategyStyle")
        or result.get("strategy")
    )
    if aggression_level:
        universe_payload["aggression"] = str(aggression_level).strip()
    if trade_horizon:
        universe_payload["trade_horizon"] = str(trade_horizon).strip()
    if style_label:
        universe_payload["style"] = str(style_label).strip()
    next_run_time_value = (
        result.get("next_run_time")
        or result.get("nextRunTime")
        or result.get("next_run")
    )
    if next_run_time_value:
        universe_payload["next_run_time"] = str(next_run_time_value).strip()
    max_positions_value = None
    limit_sources: list[Any] = []
    limits_block = result.get("limits")
    if isinstance(limits_block, dict):
        limit_sources.extend(
            limits_block.get(key)
            for key in (
                "max_positions",
                "maxPositions",
                "max_open_positions",
                "maxOpenPositions",
            )
        )
    limit_sources.extend(
        result.get(key)
        for key in (
            "max_positions",
            "maxPositions",
            "max_open_positions",
            "maxOpenPositions",
        )
    )
    for candidate in limit_sources:
        val = safe_int(candidate)
        if val and val > 0:
            max_positions_value = val
            break
    if max_positions_value:
        universe_payload["max_positions"] = max_positions_value
    news_requests = result.get("news_requests") or []
    selection_result = {
        "pairs": universe_payload["pairs"],
        "global_timeframes": universe_payload["timeframes"],
        "global_indicators": universe_payload["indicators"],
        "next_run_minutes": universe_payload.get("next_run_minutes"),
        "reason": universe_payload.get("notes"),
    }
    if universe_payload.get("initial_timeframes"):
        selection_result["initial_timeframes"] = list(universe_payload["initial_timeframes"])
        selection_result["global_timeframes"] = list(universe_payload["initial_timeframes"])
    if aggression_level:
        selection_result["aggression"] = str(aggression_level).strip()
    if trade_horizon:
        selection_result["trade_horizon"] = str(trade_horizon).strip()
    if style_label:
        selection_result["style"] = str(style_label).strip()
    if universe_payload.get("next_run_time"):
        selection_result["next_run_time"] = universe_payload["next_run_time"]
        style_label = selection_result.get("style")
        if style_label:
            selection_style = str(style_label).strip()
    if style_label:
        selection_result["style"] = str(style_label).strip()
    if max_positions_value:
        selection_result["limits"] = {"max_positions": max_positions_value}
        selection_result["max_positions"] = max_positions_value
    selection_result["_news_digest"] = news_digest
    universe_state = {
        "pairs": universe_payload["pairs"],
        "global_timeframes": universe_payload["timeframes"],
        "global_indicators": universe_payload["indicators"],
    }
    if universe_payload.get("initial_timeframes"):
        universe_state["initial_timeframes"] = list(universe_payload["initial_timeframes"])
        universe_state["global_timeframes"] = list(universe_payload["initial_timeframes"])
    if aggression_level:
        universe_state["aggression"] = str(aggression_level).strip()
    if trade_horizon:
        universe_state["trade_horizon"] = str(trade_horizon).strip()
    if style_label:
        universe_state["style"] = str(style_label).strip()
    if max_positions_value:
        universe_state["max_positions"] = max_positions_value
    _record_ai_exchange(
        context_key,
        label=context_label,
        model=AI_MODEL,
        request=messages,
        response={"raw": payload, "parsed": result},
        usage=usage,
        token_estimate=token_estimate,
        extra={
            "pairs": len(universe_payload.get("pairs") or []),
            "news_requests": len(news_requests or []),
        },
    )
    return selection_result, universe_state, news_requests


def build_portfolio_bundle(exchange, selection_result, positions_map, news_cache=None, extra_symbols=None):
    targets = (selection_result or {}).get("targets") or []
    global_timeframes = set((selection_result or {}).get("global_timeframes") or [])
    raw_global_indicators = (selection_result or {}).get("global_indicators") or []
    global_indicators = _dedupe_preserve_order(raw_global_indicators)
    bundle = {"symbols": [], "meta": {}}
    open_orders_cache = {}
    processed_symbols: set[str] = set()
    for target in targets:
        symbol = target.get("symbol")
        if not symbol:
            continue
        base_timeframes = list(global_timeframes)
        timeframes = list(dict.fromkeys(base_timeframes + (target.get("timeframes") or [])))
        baseline_source = list(global_indicators) if global_indicators else list(BASE_INDICATOR_CANDIDATES)
        baseline_cap = max(BASE_INDICATOR_MIN_COUNT, min(BASE_INDICATOR_MAX_COUNT, len(baseline_source)))
        baseline_indicators = baseline_source[:baseline_cap]
        indicator_candidates = list(target.get("indicators") or [])
        indicators = _dedupe_preserve_order(baseline_indicators + indicator_candidates + global_indicators)
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
    context_label = f"trade plan ({stage})"
    context_key = f"trade_plan_{stage}".strip().lower() if stage else "trade_plan"
    client = _create_ai_client(timeout=40, context=context_label)
    if client is None:
        _emit_ai_offline_notice(f"client init failed ({context_label})")
        log(f"❌ AI client unavailable for {context_label}", Fore.RED)
        return None
    positions_payload = _compact_positions_snapshot(positions_snapshot)
    pending_orders_payload = _compact_orders_snapshot(pending_orders)
    payload = {
        "allocations": {"spot_pct": SPOT_ALLOCATION_PCT, "derivatives_pct": (1.0 - SPOT_ALLOCATION_PCT)},
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
        '      "market": "spot|linear|inverse|derivatives",\n'
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
        '  "next_run_time": "2025-01-01T10:30:00+03:00",\n'
        '  "next_run_minutes": float (REQUIRED, 5..40; drive toward 5 when volatility/news/aggression is high, toward 40 when calm),\n'
        '  "notes": "optional"\n'
        "}\n"
        "For each decision choose the execution market: 'spot' for cash trades, or 'linear'/'inverse'/'derivatives' for perpetuals. "
        "If unsure, use 'derivatives'. If additional context is required, populate 'needs' and leave 'decisions' empty."
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
    if not _ensure_token_budget(token_estimate, AI_MODEL, context_label):
        return None
    _log_ai_request(AI_MODEL, token_estimate, context_label)
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
        except RateLimitError as exc:
            last_error = exc
            if _current_ai_provider() == "openai":
                _switch_ai_provider_to_fallback("rate limit 429 (trade plan)")
                retry_client = _create_ai_client(timeout=40, context=f"{context_label} (retry)")
                if retry_client is not None:
                    client = retry_client
                    continue
            _emit_ai_offline_notice("rate limit 429 (trade plan)")
            if attempt_count >= max_attempts:
                break
            delay = base_backoff * (attempt_count ** 2)
            time.sleep(delay + random.uniform(0, base_backoff))
            _record_ai_exchange(
                context_key,
                label=context_label,
                model=AI_MODEL,
                request=messages,
                error=f"attempt {attempt_count}: {exc}",
                token_estimate=token_estimate,
                extra={"stage": stage, "attempt": attempt_count},
            )
            continue
        except Exception as exc:
            last_error = exc
            log(f"[ERROR] OpenAI trade plan attempt {attempt_count}: {exc}", Fore.RED)
            if attempt_count >= max_attempts:
                break
            delay = base_backoff * (attempt_count ** 2)
            time.sleep(delay + random.uniform(0, base_backoff))
            _record_ai_exchange(
                context_key,
                label=context_label,
                model=AI_MODEL,
                request=messages,
                error=f"attempt {attempt_count}: {exc}",
                token_estimate=token_estimate,
                extra={"stage": stage, "attempt": attempt_count},
            )
            continue
        usage_obj = getattr(res, "usage", None)
        _register_ai_usage(AI_MODEL, usage_obj, context_label)
        content = res.choices[0].message.content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            log(f"[WARN] JSON decode (trade plan): {exc}", Fore.YELLOW)
            last_error = exc
            if attempt_count >= max_attempts:
                break
            delay = base_backoff * (attempt_count ** 2)
            time.sleep(delay + random.uniform(0, base_backoff))
            _record_ai_exchange(
                context_key,
                label=context_label,
                model=AI_MODEL,
                request=messages,
                response=content,
                error=f"JSON decode: {exc}",
                usage=usage_obj,
                token_estimate=token_estimate,
                extra={"stage": stage, "attempt": attempt_count},
            )
            continue
        _record_ai_exchange(
            context_key,
            label=context_label,
            model=AI_MODEL,
            request=messages,
            response={"raw": content, "parsed": parsed},
            usage=usage_obj,
            token_estimate=token_estimate,
            extra={"stage": stage, "attempt": attempt_count},
        )
        return parsed
    if last_error:
        log(
            f"[ERROR] OpenAI trade plan failed after {attempt_count} attempts: {last_error}",
            Fore.RED,
        )
        _record_ai_exchange(
            context_key,
            label=context_label,
            model=AI_MODEL,
            request=messages,
            error=f"failed after {attempt_count} attempts: {last_error}",
            token_estimate=token_estimate,
            extra={"stage": stage, "attempts": attempt_count},
        )
    return None


def _normalize_trade_side(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in {"buy", "long"}:
        return "buy"
    if normalized in {"sell", "short"}:
        return "sell"
    return None


def _extract_decision_confidence(decision: dict[str, Any]) -> float:
    for key in ("confidence", "confidence_value", "confidenceValue"):
        raw = decision.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return 0.0


def _auto_directional_adjustment(
    symbol: str,
    decision: dict[str, Any],
    current_position: dict[str, Any] | None,
    extra_orders: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not AUTO_DIRECTION_ADJUST_ENABLED:
        return extra_orders
    position_qty = abs(
        safe_float((current_position or {}).get("amount"))
        or safe_float((current_position or {}).get("contracts"))
        or 0.0
    )
    if not position_qty or position_qty <= 0:
        return extra_orders
    current_side = _normalize_trade_side((current_position or {}).get("side"))
    if not current_side:
        amt = (
            safe_float((current_position or {}).get("contracts"))
            or safe_float((current_position or {}).get("amount"))
            or 0.0
        )
        if amt > 0:
            current_side = "buy"
        elif amt < 0:
            current_side = "sell"
    decision_side = _normalize_trade_side(decision.get("side"))
    if not decision_side:
        return extra_orders
    action = (decision.get("action") or "").lower()
    if action not in {"open", "manage", "close"}:
        return extra_orders
    confidence_value = _extract_decision_confidence(decision)
    if confidence_value < AUTO_DIRECTION_MIN_CONFIDENCE:
        return extra_orders
    if any(str(order.get("note") or "").startswith("auto_direction") for order in extra_orders):
        return extra_orders
    if current_side and current_side != decision_side:
        reduce_qty = min(position_qty, position_qty * AUTO_DIRECTION_REDUCE_FACTOR)
        reduce_qty = max(reduce_qty, AUTO_DIRECTION_MIN_QTY)
        if reduce_qty > 0:
            extra_orders.insert(
                0,
                {
                    "type": "market",
                    "side": decision_side,
                    "amount": reduce_qty,
                    "reduceOnly": True,
                    "note": "auto_direction_reduce",
                },
            )
            log(
                f"{symbol}: auto-direction reduce {reduce_qty:.6f} due to mismatch "
                f"(position={current_side}, decision={decision_side})",
                Fore.LIGHTBLUE_EX,
            )
    elif current_side == decision_side:
        scale_qty = position_qty * AUTO_DIRECTION_SCALE_FACTOR
        scale_qty = max(scale_qty, AUTO_DIRECTION_MIN_QTY)
        if scale_qty > 0:
            extra_orders.append(
                {
                    "type": "market",
                    "side": decision_side,
                    "amount": scale_qty,
                    "reduceOnly": False,
                    "note": "auto_direction_scale",
                }
            )
            log(
                f"{symbol}: auto-direction scale-in {scale_qty:.6f} (position aligned with {decision_side})",
                Fore.LIGHTBLUE_EX,
            )
    return extra_orders


def execute_symbol_decision(exchange, decision, positions_map, open_orders_cache, counts):
    if not decision:
        return 0, positions_map, open_orders_cache
    sym = decision.get("symbol")
    if not sym:
        return 0, positions_map, open_orders_cache
    # Apply market selection from AI (spot vs derivatives)
    market_choice = _normalize_market_value(
        decision.get("market") or decision.get("venue") or decision.get("category")
    )
    effective_sym = _apply_category_to_symbol(sym, market_choice) if market_choice else sym
    if effective_sym != sym:
        log(f"{sym}: market={market_choice or 'default'} -> using {effective_sym}", Fore.LIGHTBLACK_EX)
        sym = effective_sym
    action_raw = (decision.get("action") or "skip").lower()
    action_aliases = {
        "replace_orders": "manage",
        "refresh_orders": "manage",
        "update_orders": "manage",
        "maintain": "hold",
        "maintain_position": "hold",
    }
    action = action_aliases.get(action_raw, action_raw)
    if action != action_raw:
        log(f"{sym}: normalized action {action_raw!r} > {action!r}", Fore.LIGHTBLACK_EX)
    counts[action] = counts.get(action, 0) + 1
    side = decision.get("side") or ""
    side_lower = side.lower()
    reason = decision.get("reason") or ""
    notional_pct = decision.get("notional_pct")
    current_position = positions_map.get(sym)
    position_category = str((current_position or {}).get("category") or "").lower()
    is_spot_position = position_category == "spot"
    if not is_spot_position and sym.upper().endswith(":SPOT"):
        is_spot_position = True
    spot_reduce_order: dict[str, Any] | None = None
    if (
        is_spot_position
        and side_lower == "sell"
        and action not in {"reduce", "skip", "hold", "none"}
    ):
        counts[action] = max(0, counts.get(action, 0) - 1)
        action = "reduce"
        counts[action] = counts.get(action, 0) + 1
        position_amount = safe_float(
            (current_position or {}).get("amount")
            or (current_position or {}).get("contracts")
            or (current_position or {}).get("size")
        )
        position_amount = abs(position_amount) if position_amount is not None else 0.0
        requested_qty = safe_float(
            decision.get("qty")
            or decision.get("amount")
            or decision.get("volume")
            or decision.get("size")
        )
        reduce_qty = (
            min(position_amount, abs(requested_qty))
            if requested_qty is not None and position_amount
            else position_amount
        )
        if reduce_qty and reduce_qty > 0:
            spot_reduce_order = {
                "type": "market",
                "side": "sell",
                "amount": reduce_qty,
                "note": "spot reduce",
                "reduceOnly": True,
                "spotReduce": True,
            }
            log(f"{sym}: spot SELL converted to reduce amount={reduce_qty:.6f}", Fore.LIGHTBLUE_EX)
    mode_hint = SYMBOL_MARKET_MODE_HINTS.get(sym) or {}
    analysis_balance = mode_hint.get("analysis_balance") or {}
    def _fmt_weight(value: float | None) -> str:
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "n/a"
    balance_mode = analysis_balance.get("mode") or "balanced"
    balance_summary = (
        f"{balance_mode} (news={_fmt_weight(analysis_balance.get('news_weight'))} "
        f"ta={_fmt_weight(analysis_balance.get('technical_weight'))})"
    )
    reason_text = reason or "entry signal"
    regime = decision.get("regime")
    if regime:
        reason_text = f"[{regime}] {reason_text}"
    config_block = decision.get("config") or {}
    sl_atr_conf = safe_float(config_block.get("sl_atr") or SL_ATR)
    tp_atr_conf = safe_float(config_block.get("tp_atr") or TP_ATR)
    sl_ratio = f"{(sl_atr_conf / SL_ATR):.2f}x" if SL_ATR else "n/a"
    tp_ratio = f"{(tp_atr_conf / TP_ATR):.2f}x" if TP_ATR else "n/a"
    decision_log_suffix = f"mode={mode_hint.get('mode') or 'n/a'} balance={balance_summary}"
    log(
        f"{sym}: action={action} side={side} reason={reason_text} {decision_log_suffix}",
        Fore.LIGHTBLUE_EX,
    )
    if action == "open":
        logic_note = analysis_balance.get("guidance")
        open_parts = [
            f"reason={reason_text}",
            f"sl_atr={sl_atr_conf:.2f} ({sl_ratio})",
            f"tp_atr={tp_atr_conf:.2f} ({tp_ratio})",
            f"notional_pct={_format_notional_pct(notional_pct) if notional_pct is not None else 'n/a'}",
            f"balance={balance_summary}",
        ]
        if logic_note:
            open_parts.append(f"logic={logic_note}")
        log(
            f"[OPEN] {sym}: " + "; ".join(open_parts),
            Fore.LIGHTGREEN_EX,
        )
    else:
        if action in {"skip", "hold"}:
            needs_payload = decision.get("needs")
            if isinstance(needs_payload, list):
                needs_text = ", ".join(str(item) for item in needs_payload if item)
            else:
                needs_text = str(needs_payload) if needs_payload else "none"
        log(
            f"[SKIP] {sym}: {reason_text}; balance={balance_summary}; needs={needs_text}",
            Fore.LIGHTYELLOW_EX,
        )
    if notional_pct is not None:
        log(f"{sym}: notional_pct={_format_notional_pct(notional_pct)}", Fore.LIGHTBLACK_EX)
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
    if spot_reduce_order:
        extra_orders.insert(0, spot_reduce_order)

    extra_orders = _auto_directional_adjustment(sym, decision, current_position, extra_orders)

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
            log(f"🔷 —?'—?—?—?—?—? ——?——?—< {oid} {sym} (source={source})", Fore.LIGHTBLUE_EX)
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

    executed_orders: list[str] = []
    if extra_orders:
        previous_position_snapshot = current_position
        executed, actions_performed = execute_extra_orders(
            exchange,
            sym,
            extra_orders,
            equity=equity,
            current_position=current_position,
            open_orders=open_orders_symbol,
            max_limits_per_side=MAX_NON_REDUCE_LIMITS_PER_SIDE,
        )
        executed_orders = list(executed) if executed else []
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
            if action == "open" and not has_position and executed_orders:
                prev_amount_val = safe_float(
                    (previous_position_snapshot or {}).get("amount")
                    or (previous_position_snapshot or {}).get("contracts")
                    or (previous_position_snapshot or {}).get("size")
                )
                curr_amount_val = safe_float(
                    (current_position or {}).get("amount")
                    or (current_position or {}).get("contracts")
                    or (current_position or {}).get("size")
                )
                prev_has_pos = (
                    prev_amount_val is not None and math.isfinite(prev_amount_val) and abs(prev_amount_val) > 0
                )
                curr_has_pos = (
                    curr_amount_val is not None and math.isfinite(curr_amount_val) and abs(curr_amount_val) > 0
                )
                if not prev_has_pos and not curr_has_pos:
                    pending_orders = open_orders_symbol or []
                    pending_descriptions = [
                        _summarize_order_spec(order)
                        for order in pending_orders[:3]
                        if isinstance(order, dict)
                    ]
                    pending_text = (
                        "; ".join(pending_descriptions) if pending_descriptions else "awaiting exchange confirmation"
                    )
                    log(
                        f"[INFO] {sym}: entry orders submitted, waiting for fill ({pending_text}).",
                        Fore.LIGHTBLACK_EX,
                    )

    return 1, positions_map, open_orders_cache


def load_environment():
    env_paths = [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR / ".env.local",
    ]
    merged: dict[str, str] = {}
    for path in env_paths:
        if not path.exists():
            continue
        values = dotenv_values(path)
        for key, value in values.items():
            if value is None:
                continue
            merged[key] = value
    for key, value in merged.items():
        os.environ[key] = value


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


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return bool(default)
    normalized = value.strip().lower()
    return normalized in ("1", "true", "yes", "on", "y")


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


def _parse_ratio_scheme(value: str | None, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not value:
        return list(default)
    result: list[tuple[float, float]] = []
    for chunk in value.split(","):
        candidate = chunk.strip()
        if not candidate:
            continue
        if "@" in candidate:
            ratio_part, mult_part = candidate.split("@", 1)
        else:
            ratio_part, mult_part = candidate, "1"
        try:
            ratio = float(ratio_part.strip())
            multiplier = float(mult_part.strip())
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ratio) or ratio <= 0:
            continue
        if not math.isfinite(multiplier):
            continue
        result.append((ratio, multiplier))
    return result if result else list(default)


def _parse_telegram_command_list(raw: str | None) -> list[dict[str, str]]:
    if not raw:
        return []
    commands: list[dict[str, str]] = []
    for chunk in raw.split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        if ":" in piece:
            command_part, description_part = piece.split(":", 1)
        else:
            command_part, description_part = piece, piece
        command_clean = command_part.strip().lstrip("/")
        description_clean = description_part.strip() or command_clean
        if not command_clean:
            continue
        commands.append({
            "command": command_clean[:32],
            "description": description_clean[:256],
        })
    return commands


def refresh_settings():
    _configure_state_paths()
    load_environment()
    stored_api_key, stored_api_secret = _load_bybit_credentials()
    if stored_api_key and not os.getenv("BYBIT_API_KEY"):
        os.environ["BYBIT_API_KEY"] = stored_api_key
    if stored_api_secret and not os.getenv("BYBIT_API_SECRET"):
        os.environ["BYBIT_API_SECRET"] = stored_api_secret
    global PAIR_LIST, TIMEFRAME, LEVERAGE, RISK_PCT, SL_ATR, TP_ATR, TRAILING_ATR_MULT
    global DEFAULT_NEXT_RUN_MINUTES
    global MIN_NOTIONAL_USDT, AI_AFTER_NEEDS_BIAS, MAX_OPEN_POSITIONS, MAX_POSITIONS_PER_BASE
    global CURRENT_RISK_PCT, DYNAMIC_RISK_ENABLED, MIN_DYNAMIC_RISK_PCT, MAX_DYNAMIC_RISK_PCT
    global BREAKEVEN_ENABLED, BREAKEVEN_ATR_MULT, BREAKEVEN_BUFFER_ATR
    global MIN_CONTEXT_30M, MIN_CONTEXT_4H, DEFAULT_CONTEXT_30M, DEFAULT_CONTEXT_4H
    global CONTEXT_STEP_30M, CONTEXT_STEP_4H
    global AI_INITIAL_TIMEFRAMES, AI_INITIAL_TF_DEPTHS, AI_INITIAL_INDICATOR_POOL
    global AI_INITIAL_EMA_COUNT, AI_INITIAL_EXTRA_INDICATOR_COUNT
    global AI_INITIAL_NEWS_PROVIDER, AI_INITIAL_NEWS_LIMIT, AI_NEEDS_TF_DEPTHS
    global TG_TOKEN, TG_CHAT, TG_TOPIC_ID, TG_GIT_TOPIC_ID, TG_MIN_INTERVAL, TG_DUP_WINDOW, TG_RETRY_ATTEMPTS, TG_RETRY_BACKOFF
    global AI_MODEL, AI_KEY, AI_MODEL_PRIMARY, AI_MODEL_CHEAP, AI_MODEL_THRESHOLD, AI_TOKEN_BUDGET_CYCLE
    global AI_SECONDARY_BUDGET_START, AI_HARD_STOP_BUDGET
    global AI_PROVIDER_PRIMARY, AI_PROVIDER_SECONDARY, AI_PROVIDER_CURRENT
    global DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
    global AI_OFFLINE_CANCEL_ENTRIES
    global OFFLINE_TRADING_ENABLED, OFFLINE_PAIR_LIMIT, OFFLINE_MAX_NEW_POSITIONS, OFFLINE_NEWS_BIAS_ENABLED
    global NEWS_PROVIDER, NEWS_API_TOKEN, NEWS_ITEMS_LIMIT
    global SPOT_ALLOCATION_PCT, DERIV_ALLOCATION_PCT, CURRENT_MARKET_ALLOCATIONS
    global POSITION_MODE, HEDGE_MODE, ACTIVE_POSITION_MODE, ACTIVE_HEDGE_MODE, POSITION_MODE_MISMATCH_STATE, ORDER_MARGIN_UTILIZATION
    global USER_LOG_MAX_BYTES, USER_LOG_BACKUPS, DRAWDOWN_CONTROL_ENABLED, DRAWDOWN_WINDOW_HOURS
    global AI_LOG_MAX_BYTES, AI_LOG_BACKUPS
    global EXTRA_POSITION_SETTLES
    global MAIN_LOG_PATH, MAIN_LOG_MAX_BYTES, MAIN_LOG_BACKUPS, MAIN_LOG_ENABLED
    global LOG_TIMEZONE, LOG_TZINFO, _LOG_TZ_WARNING_EMITTED
    global PAIR_CANDIDATE_LIMIT, PAIR_PREFETCH_LIMIT
    global SUPPORT_CONTEXT_TIMEFRAMES, SUPPORT_CONTEXT_INDICATORS, SUPPORT_CONTEXT_LIMIT
    global LOW_CONFIDENCE_TIMEFRAMES, LOW_CONFIDENCE_INDICATORS, LOW_CONFIDENCE_SERIALIZE_LIMIT
    global NEEDS_MAX_TIMEFRAMES, NEEDS_MAX_INDICATORS, NEEDS_SERIALIZE_DEFAULT_LIMIT
    global PARTIAL_TP_SCHEME, ENTRY_LADDER_SCHEME
    global AUTO_MIN_NOTIONAL, AUTO_MARGIN_SCALE, AUTO_MARGIN_SCALE_RATIO
    global VOL_GUARD_ENABLED, ATR_GUARD_MAX_RATIO, ATR_GUARD_MIN_RATIO, ATR_GUARD_LOW_BOOST
    global TELEGRAM_FORWARD_LOGS, TELEGRAM_LOG_BATCH_SIZE, TELEGRAM_LOG_FLUSH_INTERVAL, TELEGRAM_LOG_RATE_LIMIT_WINDOW, TELEGRAM_LOG_MAX_MESSAGES_PER_WINDOW, TELEGRAM_LOG_THREAD_ID
    global TELEGRAM_DECISIONS_VERBOSE
    global TELEGRAM_WEBHOOK_URL, TELEGRAM_WEBHOOK_HOST, TELEGRAM_WEBHOOK_PORT, TELEGRAM_WEBHOOK_PATH, TELEGRAM_WEBHOOK_SECRET
    global TELEGRAM_ALLOWED_CHAT_IDS, TELEGRAM_COMMANDS_LIST, TELEGRAM_RELEASE_THREAD_ID, TELEGRAM_COMMAND_THREAD_ID
    global TELEGRAM_INPROGRESS_THREAD_ID, TELEGRAM_RESULTS_THREAD_ID, TELEGRAM_STATUS_THREAD_ID, TELEGRAM_TRADE_THREAD_ID, TELEGRAM_SUPPORT_THREAD_ID
    global TRAILING_DYNAMIC_TRIGGER_ATR, TRAILING_DYNAMIC_FACTOR, TRAILING_DYNAMIC_MIN_ATR
    global PSEUDOTRAIL_MIN_IMPROVE_ATR, PSEUDOTRAIL_STOP_LOCK_FACTOR, PSEUDOTRAIL_TP_EXTEND_FACTOR
    global USER_ID, USER_LABEL, TELEGRAM_MESSAGE_PREFIX, TG_TOPIC_ID, TG_GIT_TOPIC_ID
    global AI_SUPPORT_MODEL, SUPPORT_MAX_CONTEXT_BYTES, INPROGRESS_WIP_ENABLED
    global IMMEDIATE_CLOSE_ON_BREACH, _TRAIL_PROTECTION
    _configure_state_paths()
    USER_ID = os.getenv("BYBITBOT_USER_ID") or USER_ID or "shared"
    USER_LABEL = os.getenv("BYBITBOT_USER_LABEL") or USER_LABEL or "redfaraon"
    prefix_override = os.getenv("TELEGRAM_MESSAGE_PREFIX")
    if prefix_override is not None:
        TELEGRAM_MESSAGE_PREFIX = prefix_override.strip()
    else:
        TELEGRAM_MESSAGE_PREFIX = USER_LABEL.strip()
    PAIR_LIST = os.getenv("PAIR_LIST", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,DOGE/USDT:USDT").split(",")
    TIMEFRAME = os.getenv("TIMEFRAME", "30m")
    LEVERAGE = int(os.getenv("LEVERAGE", 10))
    risk_env_value = os.getenv("RISK_PCT")
    if risk_env_value is None:
        risk_env_value = os.getenv("RISK_EQUITY_PCT")
    try:
        RISK_PCT = float(risk_env_value) if risk_env_value is not None else DEFAULT_RISK_PCT
    except (TypeError, ValueError):
        log(f"[WARN] Invalid RISK_PCT value '{risk_env_value}', using default {DEFAULT_RISK_PCT:.4f}", Fore.YELLOW)
        RISK_PCT = DEFAULT_RISK_PCT
    if not math.isfinite(RISK_PCT) or RISK_PCT <= 0:
        log(f"[WARN] RISK_PCT={RISK_PCT} is not positive, using default {DEFAULT_RISK_PCT:.4f}", Fore.YELLOW)
        RISK_PCT = DEFAULT_RISK_PCT
    DYNAMIC_RISK_ENABLED = env_int("RISK_DYNAMIC_ENABLED", 1) != 0
    base_min_default = max(0.0005, RISK_PCT * 0.5)
    base_max_default = max(RISK_PCT, RISK_PCT * 1.8)
    try:
        MIN_DYNAMIC_RISK_PCT = float(os.getenv("MIN_DYNAMIC_RISK_PCT", str(base_min_default)))
    except (TypeError, ValueError):
        MIN_DYNAMIC_RISK_PCT = base_min_default
    try:
        MAX_DYNAMIC_RISK_PCT = float(os.getenv("MAX_DYNAMIC_RISK_PCT", str(base_max_default)))
    except (TypeError, ValueError):
        MAX_DYNAMIC_RISK_PCT = base_max_default
    MIN_DYNAMIC_RISK_PCT = max(1e-5, min(MIN_DYNAMIC_RISK_PCT, RISK_PCT))
    MAX_DYNAMIC_RISK_PCT = max(RISK_PCT, max(MIN_DYNAMIC_RISK_PCT, MAX_DYNAMIC_RISK_PCT))
    # Market allocation defaults (overridable by model via trade plan 'allocations')
    try:
        SPOT_ALLOCATION_PCT = float(os.getenv("SPOT_ALLOCATION_PCT", "0.0"))
    except (TypeError, ValueError):
        SPOT_ALLOCATION_PCT = 0.0
    SPOT_ALLOCATION_PCT = max(0.0, min(1.0, SPOT_ALLOCATION_PCT))
    DERIV_ALLOCATION_PCT = max(0.0, min(1.0, 1.0 - SPOT_ALLOCATION_PCT))
    CURRENT_MARKET_ALLOCATIONS = {
        "spot": SPOT_ALLOCATION_PCT,
        "derivatives": DERIV_ALLOCATION_PCT,
        "linear": DERIV_ALLOCATION_PCT,
        "inverse": DERIV_ALLOCATION_PCT,
    }
    CURRENT_RISK_PCT = min(MAX_DYNAMIC_RISK_PCT, max(MIN_DYNAMIC_RISK_PCT, CURRENT_RISK_PCT if CURRENT_RISK_PCT > 0 else RISK_PCT))
    try:
        BREAKEVEN_ENABLED = env_int("BREAKEVEN_ENABLED", env_int("MOVE_STOP_TO_BREAKEVEN", 1)) != 0
    except Exception:
        BREAKEVEN_ENABLED = True
    try:
        BREAKEVEN_ATR_MULT = float(os.getenv("BREAKEVEN_ATR_MULT", str(BREAKEVEN_ATR_MULT)))
    except (TypeError, ValueError):
        BREAKEVEN_ATR_MULT = 0.6
    try:
        BREAKEVEN_BUFFER_ATR = float(os.getenv("BREAKEVEN_BUFFER_ATR", str(BREAKEVEN_BUFFER_ATR)))
    except (TypeError, ValueError):
        BREAKEVEN_BUFFER_ATR = 0.15
    BREAKEVEN_ATR_MULT = max(0.0, BREAKEVEN_ATR_MULT)
    BREAKEVEN_BUFFER_ATR = max(0.0, BREAKEVEN_BUFFER_ATR)
    SL_ATR = float(os.getenv("SL_ATR", os.getenv("SL_ATR_MULT", 0.8)))
    TP_ATR = float(os.getenv("TP_ATR", os.getenv("TP_ATR_MULT", 1.6)))
    TRAILING_ATR_MULT = float(os.getenv("TRAILING_ATR_MULT", os.getenv("TRAILING_ATR", "1.0")))
    TRAILING_ATR_MULT = max(0.0, TRAILING_ATR_MULT)
    try:
        TRAILING_DYNAMIC_TRIGGER_ATR = float(os.getenv("TRAILING_DYNAMIC_TRIGGER_ATR", str(TRAILING_DYNAMIC_TRIGGER_ATR)))
    except (TypeError, ValueError):
        TRAILING_DYNAMIC_TRIGGER_ATR = 1.4
    try:
        TRAILING_DYNAMIC_FACTOR = float(os.getenv("TRAILING_DYNAMIC_FACTOR", str(TRAILING_DYNAMIC_FACTOR)))
    except (TypeError, ValueError):
        TRAILING_DYNAMIC_FACTOR = 0.65
    try:
        TRAILING_DYNAMIC_MIN_ATR = float(os.getenv("TRAILING_DYNAMIC_MIN_ATR", str(TRAILING_DYNAMIC_MIN_ATR)))
    except (TypeError, ValueError):
        TRAILING_DYNAMIC_MIN_ATR = 0.35
    TRAILING_DYNAMIC_TRIGGER_ATR = max(0.0, TRAILING_DYNAMIC_TRIGGER_ATR)
    TRAILING_DYNAMIC_FACTOR = max(0.1, TRAILING_DYNAMIC_FACTOR)
    TRAILING_DYNAMIC_MIN_ATR = max(0.05, TRAILING_DYNAMIC_MIN_ATR)
    try:
        PSEUDOTRAIL_MIN_IMPROVE_ATR = float(os.getenv("PSEUDOTRAIL_MIN_IMPROVE_ATR", str(PSEUDOTRAIL_MIN_IMPROVE_ATR)))
    except (TypeError, ValueError):
        PSEUDOTRAIL_MIN_IMPROVE_ATR = 0.35
    try:
        PSEUDOTRAIL_STOP_LOCK_FACTOR = float(os.getenv("PSEUDOTRAIL_STOP_LOCK_FACTOR", str(PSEUDOTRAIL_STOP_LOCK_FACTOR)))
    except (TypeError, ValueError):
        PSEUDOTRAIL_STOP_LOCK_FACTOR = 0.35
    try:
        PSEUDOTRAIL_TP_EXTEND_FACTOR = float(os.getenv("PSEUDOTRAIL_TP_EXTEND_FACTOR", str(PSEUDOTRAIL_TP_EXTEND_FACTOR)))
    except (TypeError, ValueError):
        PSEUDOTRAIL_TP_EXTEND_FACTOR = 0.25
    PSEUDOTRAIL_MIN_IMPROVE_ATR = max(0.0, PSEUDOTRAIL_MIN_IMPROVE_ATR)
    PSEUDOTRAIL_STOP_LOCK_FACTOR = max(0.0, PSEUDOTRAIL_STOP_LOCK_FACTOR)
    PSEUDOTRAIL_TP_EXTEND_FACTOR = max(0.0, PSEUDOTRAIL_TP_EXTEND_FACTOR)
    # Scheduling bounds (online/offline) from .env
    global MIN_NEXT_RUN_MINUTES, MAX_NEXT_RUN_FROM_START_MINUTES
    global ONLINE_MIN_NEXT_RUN_MINUTES, ONLINE_MAX_NEXT_RUN_MINUTES
    global OFFLINE_MIN_NEXT_RUN_MINUTES, OFFLINE_MAX_NEXT_RUN_MINUTES
    global BACKOFF_MIN_NEXT_RUN_MINUTES, BACKOFF_MAX_NEXT_RUN_MINUTES
    try:
        MIN_NEXT_RUN_MINUTES = float(os.getenv("MIN_NEXT_RUN_MINUTES", str(MIN_NEXT_RUN_MINUTES)))
    except (TypeError, ValueError):
        MIN_NEXT_RUN_MINUTES = 5.0
    try:
        MAX_NEXT_RUN_FROM_START_MINUTES = float(os.getenv("MAX_NEXT_RUN_FROM_START_MINUTES", str(MAX_NEXT_RUN_FROM_START_MINUTES)))
    except (TypeError, ValueError):
        MAX_NEXT_RUN_FROM_START_MINUTES = 45.0
    MIN_NEXT_RUN_MINUTES = max(1.0, MIN_NEXT_RUN_MINUTES)
    MAX_NEXT_RUN_FROM_START_MINUTES = max(MIN_NEXT_RUN_MINUTES, MAX_NEXT_RUN_FROM_START_MINUTES)
    def _env_float(name: str) -> float | None:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    ONLINE_MIN_NEXT_RUN_MINUTES = _env_float("ONLINE_MIN_NEXT_RUN_MINUTES") or 10.0
    ONLINE_MAX_NEXT_RUN_MINUTES = _env_float("ONLINE_MAX_NEXT_RUN_MINUTES") or 45.0
    OFFLINE_MIN_NEXT_RUN_MINUTES = _env_float("OFFLINE_MIN_NEXT_RUN_MINUTES") or 5.0
    OFFLINE_MAX_NEXT_RUN_MINUTES = _env_float("OFFLINE_MAX_NEXT_RUN_MINUTES") or 35.0
    BACKOFF_MIN_NEXT_RUN_MINUTES = _env_float("BACKOFF_MIN_NEXT_RUN_MINUTES") or 25.0
    BACKOFF_MAX_NEXT_RUN_MINUTES = _env_float("BACKOFF_MAX_NEXT_RUN_MINUTES") or 55.0
    IMMEDIATE_CLOSE_ON_BREACH = env_int("IMMEDIATE_CLOSE_ON_BREACH", 0) != 0
    if not isinstance(_TRAIL_PROTECTION, dict):
        _TRAIL_PROTECTION = {}
    TELEGRAM_FORWARD_LOGS = env_int("TELEGRAM_FORWARD_LOGS", int(TELEGRAM_FORWARD_LOGS)) != 0
    TELEGRAM_LOG_BATCH_SIZE = max(1, env_int("TELEGRAM_LOG_BATCH_SIZE", TELEGRAM_LOG_BATCH_SIZE))
    try:
        TELEGRAM_LOG_FLUSH_INTERVAL = float(os.getenv("TELEGRAM_LOG_FLUSH_INTERVAL", str(TELEGRAM_LOG_FLUSH_INTERVAL)))
    except (TypeError, ValueError):
        TELEGRAM_LOG_FLUSH_INTERVAL = 5.0
    TELEGRAM_LOG_FLUSH_INTERVAL = max(1.0, TELEGRAM_LOG_FLUSH_INTERVAL)
    try:
        TELEGRAM_LOG_RATE_LIMIT_WINDOW = float(os.getenv("TELEGRAM_LOG_RATE_WINDOW", str(TELEGRAM_LOG_RATE_LIMIT_WINDOW)))
    except (TypeError, ValueError):
        TELEGRAM_LOG_RATE_LIMIT_WINDOW = 60.0
    TELEGRAM_LOG_RATE_LIMIT_WINDOW = max(5.0, TELEGRAM_LOG_RATE_LIMIT_WINDOW)
    TELEGRAM_LOG_MAX_MESSAGES_PER_WINDOW = max(
        1,
        env_int("TELEGRAM_LOG_RATE_LIMIT", TELEGRAM_LOG_MAX_MESSAGES_PER_WINDOW),
    )
    TELEGRAM_LOG_THREAD_ID = safe_int(os.getenv("TELEGRAM_LOG_THREAD_ID"))
    TELEGRAM_WEBHOOK_URL = (os.getenv("TELEGRAM_WEBHOOK_URL") or "").strip()
    TELEGRAM_WEBHOOK_HOST = (os.getenv("TELEGRAM_WEBHOOK_HOST") or "0.0.0.0").strip()
    TELEGRAM_WEBHOOK_PORT = env_int("TELEGRAM_WEBHOOK_PORT", TELEGRAM_WEBHOOK_PORT)
    TELEGRAM_WEBHOOK_PATH = (os.getenv("TELEGRAM_WEBHOOK_PATH") or "/telegram").strip()
    secret_value = os.getenv("TELEGRAM_WEBHOOK_SECRET")
    TELEGRAM_WEBHOOK_SECRET = secret_value.strip() if secret_value else None
    allowed_chats_raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS")
    if allowed_chats_raw:
        allowed_set: set[int] = set()
        for item in allowed_chats_raw.split(","):
            entry = item.strip()
            if not entry:
                continue
            value = safe_int(entry)
            if value is not None:
                allowed_set.add(value)
        TELEGRAM_ALLOWED_CHAT_IDS = allowed_set
    else:
        TELEGRAM_ALLOWED_CHAT_IDS = set()
    commands_raw = os.getenv("TELEGRAM_COMMANDS")
    TELEGRAM_COMMANDS_LIST = _parse_telegram_command_list(commands_raw)
    release_topic_raw = os.getenv("TELEGRAM_RELEASE_TOPIC_ID") or os.getenv("TELEGRAM_RELEASE_THREAD_ID")
    TELEGRAM_RELEASE_THREAD_ID = safe_int(release_topic_raw) if release_topic_raw else 7
    command_topic_raw = os.getenv("TELEGRAM_COMMAND_TOPIC_ID") or os.getenv("TELEGRAM_COMMAND_THREAD_ID")
    TELEGRAM_COMMAND_THREAD_ID = safe_int(command_topic_raw) if command_topic_raw else TELEGRAM_COMMAND_THREAD_ID
    inprogress_topic_raw = os.getenv("TELEGRAM_INPROGRESS_TOPIC_ID") or os.getenv("TELEGRAM_INPROGRESS_THREAD_ID")
    TELEGRAM_INPROGRESS_THREAD_ID = safe_int(inprogress_topic_raw) if inprogress_topic_raw else TELEGRAM_INPROGRESS_THREAD_ID
    results_topic_raw = os.getenv("TELEGRAM_RESULTS_TOPIC_ID") or os.getenv("TELEGRAM_RESULTS_THREAD_ID")
    TELEGRAM_RESULTS_THREAD_ID = safe_int(results_topic_raw) if results_topic_raw else TELEGRAM_RESULTS_THREAD_ID
    status_topic_raw = os.getenv("TELEGRAM_STATUS_TOPIC_ID") or os.getenv("TELEGRAM_STATUS_THREAD_ID")
    TELEGRAM_STATUS_THREAD_ID = safe_int(status_topic_raw) if status_topic_raw else TELEGRAM_STATUS_THREAD_ID
    support_topic_raw = os.getenv("TELEGRAM_SUPPORT_TOPIC_ID") or os.getenv("TELEGRAM_SUPPORT_THREAD_ID")
    if support_topic_raw:
        TELEGRAM_SUPPORT_THREAD_ID = safe_int(support_topic_raw)
    elif TELEGRAM_SUPPORT_THREAD_ID is None:
        TELEGRAM_SUPPORT_THREAD_ID = 6
    PARTIAL_TP_SCHEME = _parse_ratio_scheme(os.getenv("PARTIAL_TP_SCHEME"), DEFAULT_PARTIAL_TP_SCHEME)
    ENTRY_LADDER_SCHEME = _parse_ratio_scheme(os.getenv("ENTRY_LADDER_SCHEME"), DEFAULT_ENTRY_LADDER_SCHEME)
    AUTO_MIN_NOTIONAL = env_bool("AUTO_MIN_NOTIONAL", DEFAULT_AUTO_MIN_NOTIONAL)
    AUTO_MARGIN_SCALE = env_bool("AUTO_MARGIN_SCALE", DEFAULT_AUTO_MARGIN_SCALE)
    ratio_candidate = os.getenv("AUTO_MARGIN_SCALE_RATIO")
    if ratio_candidate is not None and ratio_candidate != "":
        try:
            ratio_candidate_val = float(ratio_candidate)
        except (ValueError, TypeError):
            ratio_candidate_val = DEFAULT_AUTO_MARGIN_SCALE_RATIO
    else:
        ratio_candidate_val = DEFAULT_AUTO_MARGIN_SCALE_RATIO
    AUTO_MARGIN_SCALE_RATIO = max(0.0, min(1.0, ratio_candidate_val))
    try:
        AUTO_MARGIN_CONFIDENCE_MULT = float(
            os.getenv("AUTO_MARGIN_CONFIDENCE_MULT", str(DEFAULT_AUTO_MARGIN_CONFIDENCE_MULT))
        )
    except (TypeError, ValueError):
        AUTO_MARGIN_CONFIDENCE_MULT = DEFAULT_AUTO_MARGIN_CONFIDENCE_MULT
    if not math.isfinite(AUTO_MARGIN_CONFIDENCE_MULT) or AUTO_MARGIN_CONFIDENCE_MULT < 1.0:
        AUTO_MARGIN_CONFIDENCE_MULT = DEFAULT_AUTO_MARGIN_CONFIDENCE_MULT
    AUTO_MARGIN_CONFIDENCE_MULT = float(os.getenv("AUTO_MARGIN_CONFIDENCE_MULT") or DEFAULT_AUTO_MARGIN_CONFIDENCE_MULT)
    USER_LOG_MAX_BYTES = _bytes_from_env("BYBIT_USER_LOG_MAX_MB", DEFAULT_USER_LOG_MAX_MB)
    USER_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_USER_LOG_BACKUPS", str(USER_LOG_BACKUPS))))
    main_log_override = os.getenv("BYBIT_MAIN_LOG")
    if main_log_override:
        try:
            MAIN_LOG_PATH = Path(main_log_override).expanduser()
        except Exception:
            MAIN_LOG_PATH = REPO_ROOT / "bybit.log"
    else:
        MAIN_LOG_PATH = REPO_ROOT / "bybit.log"
    MAIN_LOG_MAX_BYTES = _bytes_from_env("BYBIT_MAIN_LOG_MAX_MB", DEFAULT_MAIN_LOG_MAX_MB)
    MAIN_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_MAIN_LOG_BACKUPS", str(MAIN_LOG_BACKUPS))))
    MAIN_LOG_ENABLED = not env_bool("BYBIT_MAIN_LOG_DISABLE", False)
    VOL_GUARD_ENABLED = env_bool("ATR_GUARD_ENABLED", VOL_GUARD_ENABLED)
    ATR_GUARD_MAX_RATIO = max(
        0.0,
        _float_from_env(
            "ATR_GUARD_MAX_RATIO",
            ATR_GUARD_MAX_RATIO if ATR_GUARD_MAX_RATIO > 0 else DEFAULT_ATR_GUARD_MAX_RATIO,
        ),
    )
    ATR_GUARD_MIN_RATIO = max(
        0.0,
        _float_from_env(
            "ATR_GUARD_MIN_RATIO",
            ATR_GUARD_MIN_RATIO if ATR_GUARD_MIN_RATIO > 0 else DEFAULT_ATR_GUARD_MIN_RATIO,
        ),
    )
    if ATR_GUARD_MAX_RATIO > 0 and ATR_GUARD_MIN_RATIO > ATR_GUARD_MAX_RATIO:
        ATR_GUARD_MIN_RATIO = ATR_GUARD_MAX_RATIO * 0.75
    ATR_GUARD_LOW_BOOST = max(
        1.0,
        _float_from_env(
            "ATR_GUARD_LOW_BOOST",
            ATR_GUARD_LOW_BOOST if ATR_GUARD_LOW_BOOST > 0 else DEFAULT_ATR_GUARD_LOW_BOOST,
        ),
    )
    DRAWDOWN_CONTROL_ENABLED = env_bool("DRAWDOWN_CONTROL_ENABLED", DRAWDOWN_CONTROL_ENABLED)
    try:
        DRAWDOWN_WINDOW_HOURS = float(os.getenv("DRAWDOWN_WINDOW_HOURS", str(DRAWDOWN_WINDOW_HOURS)))
    except (TypeError, ValueError):
        DRAWDOWN_WINDOW_HOURS = 24.0 * 7.0
    if not math.isfinite(AUTO_MARGIN_CONFIDENCE_MULT) or AUTO_MARGIN_CONFIDENCE_MULT < 1.0:
        AUTO_MARGIN_CONFIDENCE_MULT = DEFAULT_AUTO_MARGIN_CONFIDENCE_MULT
    AI_LOG_MAX_BYTES = _bytes_from_env("BYBIT_AI_LOG_MAX_MB", DEFAULT_AI_LOG_MAX_MB)
    AI_LOG_BACKUPS = max(1, int(os.getenv("BYBIT_AI_LOG_BACKUPS", str(AI_LOG_BACKUPS))))
    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", 5.0))
    EXTRA_POSITION_SETTLES = _parse_settle_list(os.getenv("BYBIT_EXTRA_POSITION_SETTLES"), DEFAULT_EXTRA_POSITION_SETTLES)
    global NOTIONAL_EPSILON
    NOTIONAL_EPSILON = float(os.getenv("NOTIONAL_TOLERANCE", "1e-6"))
    AI_AFTER_NEEDS_BIAS = int(os.getenv("AI_AFTER_NEEDS_BIAS", 1))
    MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 15)
    MAX_POSITIONS_PER_BASE = max(0, env_int("MAX_POSITIONS_PER_BASE", MAX_POSITIONS_PER_BASE))
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
    topic_raw = os.getenv("TELEGRAM_TOPIC_ID") or os.getenv("TG_TOPIC_ID")
    trade_topic_raw = os.getenv("TELEGRAM_TRADE_TOPIC_ID") or os.getenv("TELEGRAM_TRADE_THREAD_ID")
    TELEGRAM_TRADE_THREAD_ID = safe_int(trade_topic_raw) if trade_topic_raw else None
    default_topic = safe_int(topic_raw)
    TG_TOPIC_ID = TELEGRAM_TRADE_THREAD_ID if TELEGRAM_TRADE_THREAD_ID is not None else default_topic
    TG_GIT_TOPIC_ID = safe_int(os.getenv("TELEGRAM_GIT_TOPIC_ID", "581"))
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
    TELEGRAM_DECISIONS_VERBOSE = str(os.getenv("TELEGRAM_DECISIONS_VERBOSE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    primary_env = (os.getenv("OPENAI_MODEL_PRIMARY") or os.getenv("OPENAI_MODEL"))
    AI_MODEL_PRIMARY = primary_env.strip() if isinstance(primary_env, str) and primary_env.strip() else "gpt-4.1-mini"
    cheap_env = (os.getenv("OPENAI_MODEL_CHEAP") or os.getenv("OPENAI_MODEL_BACKUP"))
    AI_MODEL_CHEAP = cheap_env.strip() if isinstance(cheap_env, str) and cheap_env.strip() else "gpt-4o-mini"
    if AI_MODEL_PRIMARY == AI_MODEL_CHEAP:
        if AI_MODEL_PRIMARY.lower().startswith("gpt-4o"):
            AI_MODEL_PRIMARY = "gpt-4.1-mini"
        elif AI_MODEL_PRIMARY.lower().startswith("gpt-4.1"):
            AI_MODEL_CHEAP = "gpt-4o-mini"
    try:
        AI_MODEL_THRESHOLD = int(os.getenv("OPENAI_MODEL_CHEAP_THRESHOLD", "5"))
    except (TypeError, ValueError):
        AI_MODEL_THRESHOLD = 5
    AI_MODEL_THRESHOLD = max(0, AI_MODEL_THRESHOLD)
    AI_MODEL = AI_MODEL_PRIMARY or AI_MODEL_CHEAP or "gpt-4.1-mini"
    ai_support_model_env = os.getenv("AI_SUPPORT_MODEL")
    if isinstance(ai_support_model_env, str) and ai_support_model_env.strip():
        AI_SUPPORT_MODEL = ai_support_model_env.strip()
    elif not AI_SUPPORT_MODEL:
        AI_SUPPORT_MODEL = AI_MODEL
    context_bytes_env = os.getenv("AI_SUPPORT_CONTEXT_BYTES")
    if context_bytes_env:
        try:
            SUPPORT_MAX_CONTEXT_BYTES = max(1024, min(20000, int(str(context_bytes_env).strip())))
        except (TypeError, ValueError):
            pass
    INPROGRESS_WIP_ENABLED = str(os.getenv("INPROGRESS_WIP_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    _refresh_userbot_owners()
    default_budget = globals().get("AI_TOKEN_BUDGET_CYCLE", 170_000)
    try:
        raw_budget = os.getenv("OPENAI_TOKEN_BUDGET_PER_CYCLE", str(default_budget))
        AI_TOKEN_BUDGET_CYCLE = int(raw_budget)
    except (TypeError, ValueError):
        AI_TOKEN_BUDGET_CYCLE = max(1000, int(default_budget) if isinstance(default_budget, (int, float)) else 170_000)
    else:
        AI_TOKEN_BUDGET_CYCLE = max(1000, AI_TOKEN_BUDGET_CYCLE)
    default_secondary = globals().get("AI_SECONDARY_BUDGET_START", 70_000)
    try:
        raw_secondary = os.getenv("OPENAI_SECONDARY_BUDGET_START", str(default_secondary))
        AI_SECONDARY_BUDGET_START = int(raw_secondary)
    except (TypeError, ValueError):
        AI_SECONDARY_BUDGET_START = max(0, int(default_secondary) if isinstance(default_secondary, (int, float)) else 70_000)
    else:
        AI_SECONDARY_BUDGET_START = max(0, AI_SECONDARY_BUDGET_START)
    hard_stop_raw = os.getenv("OPENAI_HARD_STOP_BUDGET")
    if hard_stop_raw is not None:
        hard_stop_clean = hard_stop_raw.strip()
        if not hard_stop_clean:
            AI_HARD_STOP_BUDGET = 0
        else:
            try:
                AI_HARD_STOP_BUDGET = max(0, int(float(hard_stop_clean)))
            except (TypeError, ValueError):
                fallback_hard = globals().get("AI_HARD_STOP_BUDGET", 0)
                AI_HARD_STOP_BUDGET = max(0, fallback_hard if isinstance(fallback_hard, (int, float)) else 0)
    AI_KEY = os.getenv("OPENAI_API_KEY")

    # Configure optional secondary provider (DeepSeek-compatible OpenAI API).
    # Defaults for provider names are code-level (not trading/risk) and can be
    # overridden via environment if needed.
    AI_PROVIDER_PRIMARY = (os.getenv("AI_PROVIDER_PRIMARY") or "openai").strip().lower() or "openai"
    AI_PROVIDER_SECONDARY = (os.getenv("AI_PROVIDER_SECONDARY") or "deepseek").strip().lower() or "deepseek"
    # Runtime-active provider; normally primary until we hit a rate limit.
    AI_PROVIDER_CURRENT = AI_PROVIDER_PRIMARY
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or None
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
    deepseek_model_env = os.getenv("DEEPSEEK_MODEL")
    if isinstance(deepseek_model_env, str) and deepseek_model_env.strip():
        DEEPSEEK_MODEL = deepseek_model_env.strip()
    else:
        DEEPSEEK_MODEL = None
    AI_OFFLINE_CANCEL_ENTRIES = str(os.getenv("AI_OFFLINE_CANCEL_ENTRIES", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    OFFLINE_TRADING_ENABLED = str(os.getenv("OFFLINE_TRADING_ENABLED", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    OFFLINE_NEWS_BIAS_ENABLED = str(os.getenv("OFFLINE_NEWS_BIAS_ENABLED", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        OFFLINE_PAIR_LIMIT = int(float(os.getenv("OFFLINE_PAIR_LIMIT", "8")))
    except (TypeError, ValueError):
        OFFLINE_PAIR_LIMIT = 8
    OFFLINE_PAIR_LIMIT = max(2, min(30, OFFLINE_PAIR_LIMIT))
    try:
        OFFLINE_MAX_NEW_POSITIONS = int(float(os.getenv("OFFLINE_MAX_NEW_POSITIONS", "1")))
    except (TypeError, ValueError):
        OFFLINE_MAX_NEW_POSITIONS = 1
    OFFLINE_MAX_NEW_POSITIONS = max(0, min(10, OFFLINE_MAX_NEW_POSITIONS))

    global TOKEN_LIMIT, TOKEN_SOFT_LIMIT
    TOKEN_LIMIT = env_int("OPENAI_REQUEST_TOKEN_LIMIT", 12000)
    TOKEN_SOFT_LIMIT = env_int("OPENAI_REQUEST_TOKEN_SOFT_LIMIT", 8000)

    MIN_CONTEXT_30M = env_int("AI_CONTEXT_30M_MIN", 16)
    MIN_CONTEXT_4H = env_int("AI_CONTEXT_4H_MIN", 12)
    DEFAULT_CONTEXT_30M = max(MIN_CONTEXT_30M, env_int("AI_CONTEXT_30M", 40))
    DEFAULT_CONTEXT_4H = max(MIN_CONTEXT_4H, env_int("AI_CONTEXT_4H", 40))
    CONTEXT_STEP_30M = max(1, env_int("AI_CONTEXT_30M_STEP", 4))
    CONTEXT_STEP_4H = max(1, env_int("AI_CONTEXT_4H_STEP", 2))

    base_tf = normalize_requested_timeframe(TIMEFRAME or "30m")
    fallback_initial = [base_tf]
    if "4h" not in fallback_initial:
        fallback_initial.append("4h")
    parsed_initial = []
    for item in _parse_env_list(os.getenv("AI_INITIAL_TIMEFRAMES"), fallback_initial):
        tf_norm = normalize_requested_timeframe(item, default=base_tf)
        if tf_norm and tf_norm not in parsed_initial:
            parsed_initial.append(tf_norm)
    AI_INITIAL_TIMEFRAMES = parsed_initial or fallback_initial[:]
    default_depths: dict[str, int] = {}
    for tf in AI_INITIAL_TIMEFRAMES:
        if tf == base_tf:
            default_depths[tf] = int(DEFAULT_CONTEXT_30M)
        elif tf == "4h":
            default_depths[tf] = int(DEFAULT_CONTEXT_4H)
        else:
            default_depths[tf] = int(DEFAULT_CONTEXT_30M)
    AI_INITIAL_TF_DEPTHS = _parse_env_map(os.getenv("AI_INITIAL_TF_DEPTHS"), default_depths)
    pool_override = _parse_env_list(os.getenv("AI_INITIAL_INDICATOR_POOL"), BASE_INDICATOR_CANDIDATES)
    AI_INITIAL_INDICATOR_POOL = [item.lower() for item in pool_override]
    AI_INITIAL_EMA_COUNT = max(1, env_int("AI_INITIAL_EMA_COUNT", 2))
    AI_INITIAL_EXTRA_INDICATOR_COUNT = max(0, env_int("AI_INITIAL_EXTRA_INDICATOR_COUNT", 5))

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
        NEEDS_SERIALIZE_DEFAULT_LIMIT = int(os.getenv("BYBITBOT_NEEDS_BARS_LIMIT", "8"))
    except (TypeError, ValueError):
        NEEDS_SERIALIZE_DEFAULT_LIMIT = 8
    NEEDS_SERIALIZE_DEFAULT_LIMIT = max(5, NEEDS_SERIALIZE_DEFAULT_LIMIT)

    default_needs_depths = {
        "15m": NEEDS_SERIALIZE_DEFAULT_LIMIT,
        "1h": NEEDS_SERIALIZE_DEFAULT_LIMIT,
        "4h": NEEDS_SERIALIZE_DEFAULT_LIMIT,
    }
    AI_NEEDS_TF_DEPTHS = _parse_env_map(os.getenv("AI_NEEDS_TF_DEPTHS"), default_needs_depths)

    PAIR_CANDIDATE_LIMIT = env_int("PAIR_CANDIDATE_LIMIT", PAIR_CANDIDATE_LIMIT)
    PAIR_PREFETCH_LIMIT = env_int("PAIR_PREFETCH_LIMIT", PAIR_PREFETCH_LIMIT)
    try:
        NEW_IDEAS_LIMIT = int(os.getenv("BYBITBOT_NEW_IDEAS_LIMIT", "6"))
    except (TypeError, ValueError):
        NEW_IDEAS_LIMIT = 6
    NEW_IDEAS_LIMIT = max(0, min(NEW_IDEAS_LIMIT, 12))

    news_provider_env = os.getenv("CRYPTO_NEWS_PROVIDER") or os.getenv("NEWS_PROVIDER") or "hybrid"
    NEWS_PROVIDER = news_provider_env.strip().lower() or "hybrid"
    NEWS_API_TOKEN = os.getenv("CRYPTO_NEWS_TOKEN") or os.getenv("NEWS_API_TOKEN")
    NEWS_ITEMS_LIMIT = env_int("CRYPTO_NEWS_LIMIT", 5)
    AI_INITIAL_NEWS_PROVIDER = os.getenv("AI_INITIAL_NEWS_PROVIDER") or NEWS_PROVIDER
    AI_INITIAL_NEWS_LIMIT = max(1, env_int("AI_INITIAL_NEWS_LIMIT", NEWS_ITEMS_LIMIT or 5))
    POSITION_MODE = (os.getenv("BYBIT_POSITION_MODE") or "oneway").strip().lower()
    HEDGE_MODE = POSITION_MODE in ("hedge", "hedged", "dual", "dual_side", "dual-side")
    ACTIVE_POSITION_MODE = POSITION_MODE
    ACTIVE_HEDGE_MODE = HEDGE_MODE
    POSITION_MODE_MISMATCH_STATE = None
    try:
        ORDER_MARGIN_UTILIZATION = float(os.getenv("ORDER_MARGIN_UTILIZATION", 0.95))
    except (TypeError, ValueError):
        ORDER_MARGIN_UTILIZATION = 0.95
    ORDER_MARGIN_UTILIZATION = max(0.1, min(ORDER_MARGIN_UTILIZATION, 1.0))
    MAX_NON_REDUCE_LIMITS_PER_SIDE = env_int("BYBITBOT_MAX_NON_REDUCE_LIMITS_PER_SIDE", 2)
    NON_REDUCE_PRICE_DECIMALS = env_int("BYBITBOT_NON_REDUCE_PRICE_DECIMALS", 4)
    TRADE_PLAN_MAX_ATTEMPTS = env_int("BYBITBOT_TRADE_PLAN_MAX_ATTEMPTS", 5)
    try:
        TRADE_PLAN_BACKOFF_SECONDS = float(os.getenv("BYBITBOT_TRADE_PLAN_BACKOFF", "5"))
    except (TypeError, ValueError):
        TRADE_PLAN_BACKOFF_SECONDS = 5.0
    TRADE_PLAN_BACKOFF_SECONDS = max(1.0, TRADE_PLAN_BACKOFF_SECONDS)
    try:
        AI_CONFIDENCE_THRESHOLD = float(os.getenv("BYBITBOT_CONFIDENCE_THRESHOLD", "0.65"))
    except (TypeError, ValueError):
        AI_CONFIDENCE_THRESHOLD = 0.65
    try:
        OPEN_MIN_CONFIDENCE = float(os.getenv("OPEN_MIN_CONFIDENCE", str(DEFAULT_OPEN_MIN_CONFIDENCE)))
    except (TypeError, ValueError):
        OPEN_MIN_CONFIDENCE = DEFAULT_OPEN_MIN_CONFIDENCE
    if not math.isfinite(OPEN_MIN_CONFIDENCE) or OPEN_MIN_CONFIDENCE <= 0:
        OPEN_MIN_CONFIDENCE = DEFAULT_OPEN_MIN_CONFIDENCE
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
            log(f"[WARN] LOG_TIMEZONE '{LOG_TIMEZONE}' не распознан, используется системное время.", Fore.YELLOW)
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
    "https://cryptoslate.com/feed",
    "https://www.theblock.co/rss",
]

NEWS_PROVIDER_ALIAS_CC = {"cryptocompare", "cc", "crypto"}
NEWS_PROVIDER_ALIAS_RSS = {"rss", "feed", "feeds"}
NEWS_PROVIDER_ALIAS_HYBRID = {"hybrid", "mixed", "multi", "combined", "all", "default"}

NEWS_PROVIDER = "hybrid"
NEWS_API_TOKEN = ""
NEWS_ITEMS_LIMIT = 5


if "MAX_POSITIONS_PER_BASE" not in globals():
    MAX_POSITIONS_PER_BASE = 2
try:
    MAX_POSITIONS_PER_BASE = max(0, int(os.getenv("MAX_POSITIONS_PER_BASE", str(MAX_POSITIONS_PER_BASE))))
except (TypeError, ValueError):
    MAX_POSITIONS_PER_BASE = max(0, MAX_POSITIONS_PER_BASE)

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
if "NOTIONAL_EPSILON" not in globals():
    NOTIONAL_EPSILON = 1e-6
if "MAX_NON_REDUCE_LIMITS_PER_SIDE" not in globals():
    MAX_NON_REDUCE_LIMITS_PER_SIDE = 2
if "NON_REDUCE_PRICE_DECIMALS" not in globals():
    NON_REDUCE_PRICE_DECIMALS = 4
if "TRADE_PLAN_MAX_ATTEMPTS" not in globals():
    TRADE_PLAN_MAX_ATTEMPTS = 5
if "TRADE_PLAN_BACKOFF_SECONDS" not in globals():
    TRADE_PLAN_BACKOFF_SECONDS = 5.0
if "AI_CONFIDENCE_THRESHOLD" not in globals():
    AI_CONFIDENCE_THRESHOLD = 0.65
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
    NEEDS_SERIALIZE_DEFAULT_LIMIT = 8
if "NEEDS_MAX_TIMEFRAMES" not in globals():
    NEEDS_MAX_TIMEFRAMES = 2
if "NEEDS_MAX_INDICATORS" not in globals():
    NEEDS_MAX_INDICATORS = 6
if "AI_EXTRA_PASSES_MAX" not in globals():
    AI_EXTRA_PASSES_MAX = 2
if "SUMMARY_TIMEFRAME_SHORTLIST" not in globals():
    SUMMARY_TIMEFRAME_SHORTLIST = ["30m", "4h"]
if "SUMMARY_INDICATOR_SHORTLIST" not in globals():
    SUMMARY_INDICATOR_SHORTLIST = ["ema20", "ema50", "ema200", "rsi14", "atr14", "macd", "vwma20", "supertrend", "vol"]
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
if "TG_GIT_TOPIC_ID" not in globals():
    TG_GIT_TOPIC_ID = 581
if "TELEGRAM_COMMAND_THREAD_ID" not in globals():
    TELEGRAM_COMMAND_THREAD_ID = None
if "TELEGRAM_INPROGRESS_THREAD_ID" not in globals():
    TELEGRAM_INPROGRESS_THREAD_ID = None
if "TELEGRAM_RESULTS_THREAD_ID" not in globals():
    TELEGRAM_RESULTS_THREAD_ID = None
if "TELEGRAM_STATUS_THREAD_ID" not in globals():
    TELEGRAM_STATUS_THREAD_ID = None
if "TELEGRAM_SUPPORT_THREAD_ID" not in globals():
    TELEGRAM_SUPPORT_THREAD_ID = None
if "NEW_IDEAS_LIMIT" not in globals():
    NEW_IDEAS_LIMIT = 6
NEW_IDEAS_LIMIT = max(0, min(NEW_IDEAS_LIMIT, 12))
if "AI_SUPPORT_MODEL" not in globals():
    AI_SUPPORT_MODEL = ""
if "SUPPORT_MAX_CONTEXT_BYTES" not in globals():
    SUPPORT_MAX_CONTEXT_BYTES = 4096
if "INPROGRESS_WIP_ENABLED" not in globals():
    INPROGRESS_WIP_ENABLED = False
if "_LAST_INPROGRESS_MESSAGE" not in globals():
    _LAST_INPROGRESS_MESSAGE = None

# --- AI token tracking ---
AI_TOKEN_BUDGET_CYCLE = 170_000
AI_TOKEN_USAGE_TOTAL = 0
AI_TOKEN_USAGE_BY_MODEL: dict[str, dict[str, int]] = {}
try:
    AI_SECONDARY_BUDGET_START = int(os.getenv("OPENAI_SECONDARY_BUDGET_START", "70000"))
except (TypeError, ValueError):
    AI_SECONDARY_BUDGET_START = 70_000
AI_SECONDARY_BUDGET_START = max(0, AI_SECONDARY_BUDGET_START)
AI_PER_REQUEST_TOKEN_CAP = 50_000
AI_HARD_STOP_BUDGET = 200_000
MAX_SYMBOLS_PER_CYCLE = 12
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


def _current_ai_provider() -> str:
    """Return the active AI provider label ('openai' / 'deepseek' / other)."""
    provider = (AI_PROVIDER_CURRENT or AI_PROVIDER_PRIMARY or "openai") if "AI_PROVIDER_CURRENT" in globals() else "openai"
    return str(provider).strip().lower() or "openai"


def _create_ai_client(timeout: float = 30.0, *, context: str = ""):
    """
    Construct an OpenAI-compatible client for the current provider.

    For the primary provider ('openai') this uses OPENAI_API_KEY (and optional OPENAI_BASE_URL).
    For the secondary provider ('deepseek') this uses DEEPSEEK_API_KEY / DEEPSEEK_API_BASE.
    """
    provider = _current_ai_provider()
    api_key: str | None = None
    base_url: str | None = None
    if provider == "deepseek":
        api_key = DEEPSEEK_API_KEY
        base_url = DEEPSEEK_API_BASE
        if not api_key:
            # DeepSeek not configured – fall back to OpenAI without changing the global provider.
            provider = "openai"
    if provider == "openai":
        api_key = AI_KEY
        base_url = os.getenv("OPENAI_BASE_URL") or None
    if not api_key:
        label = provider.upper()
        ctx = context or "AI request"
        log(f"[AI] {ctx}: missing API key for provider {label}", Fore.RED)
        return None
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs)
    except Exception as exc:
        ctx = context or "AI request"
        label = provider.upper()
        log(f"[ERROR] Failed to init AI client for {label} ({ctx}): {exc}", Fore.RED)
        return None


def _switch_ai_provider_to_fallback(reason: str) -> None:
    """
    Switch from the primary AI provider to the configured secondary (DeepSeek)
    after a rate-limit / quota error, if possible.
    """
    global AI_PROVIDER_CURRENT, AI_MODEL
    current = _current_ai_provider()
    target = (AI_PROVIDER_SECONDARY or "deepseek").strip().lower() or "deepseek"
    if current == target:
        return
    if target == "deepseek" and not DEEPSEEK_API_KEY:
        log(
            "[AI] Rate limit hit but DEEPSEEK_API_KEY is not configured; staying on OpenAI.",
            Fore.YELLOW,
        )
        return
    previous_model = AI_MODEL
    # Prefer explicit DeepSeek model if provided, otherwise keep current logical model.
    new_model = DEEPSEEK_MODEL or AI_MODEL
    AI_PROVIDER_CURRENT = target
    if new_model and new_model != AI_MODEL:
        AI_MODEL = new_model
    old_label = current.upper()
    new_label = target.upper()
    model_suffix = ""
    try:
        if previous_model != AI_MODEL:
            model_suffix = f" (model {previous_model} -> {AI_MODEL})"
    except Exception:
        model_suffix = ""
    msg = f"[AI] Switching provider {old_label} -> {new_label}{model_suffix} due to {reason}"
    log(msg, Fore.YELLOW)
    try:
        send_tg(msg)
    except Exception:
        pass


def _emit_ai_offline_notice(reason: str) -> None:
    """Log/notify once per cycle when we enter AI-offline fallback mode."""
    global _AI_OFFLINE_NOTICE_EMITTED_CYCLE, _AI_OFFLINE_ACTIVE_CYCLE
    cycle_no = safe_int(globals().get("_CURRENT_CYCLE_NUMBER")) or 0
    if cycle_no:
        _AI_OFFLINE_ACTIVE_CYCLE = cycle_no
    if cycle_no and _AI_OFFLINE_NOTICE_EMITTED_CYCLE == cycle_no:
        return
    if cycle_no:
        _AI_OFFLINE_NOTICE_EMITTED_CYCLE = cycle_no
    provider = _current_ai_provider().upper()
    cancel_entries = "on" if AI_OFFLINE_CANCEL_ENTRIES else "off"
    msg = (
        f"[AI] OFFLINE fallback active ({provider} unavailable: {reason}). "
        f"Strategy: no new entries; cancel non-reduce orders={cancel_entries}; manage existing positions with local protection."
    )
    log(msg, Fore.YELLOW)
    try:
        send_tg(msg)
    except Exception:
        pass


def _offline_news_score(news_item: Any) -> float:
    """
    Very simple headline-based sentiment score in [-1..+1].
    Uses only text already fetched by the bot (no external paid signals).
    """
    if not news_item:
        return 0.0
    texts: list[str] = []
    if isinstance(news_item, dict):
        headline = news_item.get("headline") or news_item.get("title") or ""
        summary = news_item.get("summary") or ""
        texts.extend([str(headline), str(summary)])
        items = news_item.get("items") or []
        if isinstance(items, list):
            for it in items[:5]:
                if isinstance(it, dict):
                    texts.append(str(it.get("title") or it.get("headline") or ""))
                    texts.append(str(it.get("summary") or ""))
                elif it:
                    texts.append(str(it))
    elif isinstance(news_item, list):
        for it in news_item[:5]:
            if isinstance(it, dict):
                texts.append(str(it.get("title") or it.get("headline") or ""))
                texts.append(str(it.get("summary") or ""))
            elif it:
                texts.append(str(it))
    else:
        texts.append(str(news_item))

    blob = " ".join(t for t in texts if t).lower()
    if not blob.strip():
        return 0.0

    pos_words = (
        "surge",
        "rally",
        "breakout",
        "bull",
        "bullish",
        "record",
        "approval",
        "etf",
        "adoption",
        "partnership",
        "upgrade",
        "support",
        "rebound",
        "beats",
    )
    neg_words = (
        "dump",
        "crash",
        "breakdown",
        "bear",
        "bearish",
        "hack",
        "exploit",
        "lawsuit",
        "ban",
        "regulator",
        "downgrade",
        "liquidation",
        "outflow",
        "rejection",
        "halt",
        "default",
        "investigation",
    )
    pos = sum(1 for w in pos_words if w in blob)
    neg = sum(1 for w in neg_words if w in blob)
    if pos == 0 and neg == 0:
        return 0.0
    raw = (pos - neg) / max(1, pos + neg)
    return float(max(-1.0, min(1.0, raw)))


def _offline_select_pairs(
    exchange,
    candidate_pairs: Sequence[str],
    *,
    news_digest: dict[str, Any] | None,
    required: set[str] | None = None,
    limit: int = 8,
) -> list[str]:
    """
    Replacement for AI universe selection when AI is unavailable.
    Scores symbols by a simple mix of 24h move magnitude and news intensity.
    """
    required = set(required or set())
    limit = max(2, min(30, int(limit or 8)))
    pairs = [p for p in candidate_pairs if p]
    if not pairs:
        return sorted(required)[:limit]

    # Limit API load: score only a subset, but always include required symbols.
    pool: list[str] = []
    seen: set[str] = set()
    for sym in list(required) + pairs:
        if sym in seen:
            continue
        seen.add(sym)
        pool.append(sym)
        if len(pool) >= max(limit * 4, 20):
            break

    scored: list[tuple[float, str]] = []
    for sym in pool:
        pct_move = 0.0
        try:
            t = exchange.fetch_ticker(sym)
            if isinstance(t, dict):
                pct_move = abs(safe_float(t.get("percentage")) or 0.0) / 100.0
        except Exception:
            pct_move = 0.0
        news_score = _offline_news_score((news_digest or {}).get(sym))
        news_strength = abs(news_score) if OFFLINE_NEWS_BIAS_ENABLED else 0.0
        # Heavier weight on news: keep volatility as a filter but prioritize strong headlines.
        score = (pct_move * 0.4) + (news_strength * 0.6)
        scored.append((score, sym))

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    result: list[str] = []
    for sym in sorted(required):
        if sym not in result:
            result.append(sym)
    for _score, sym in scored:
        if sym in result:
            continue
        result.append(sym)
        if len(result) >= limit:
            break
    return result[:limit]


def _offline_decision_for_symbol(
    symbol: str,
    df_primary: pd.DataFrame,
    *,
    current_position: dict[str, Any] | None,
    news_score: float = 0.0,
    max_new_positions_left: int = 0,
) -> dict[str, Any]:
    """
    Replacement for per-symbol AI initial decision when AI is unavailable.

    Conservative rules:
    - Never add to positions, only open new ones when OFFLINE_TRADING_ENABLED=1 and capacity allows.
    - Trend-following when EMAs align, mean-reversion only at RSI extremes.
    """
    amt = safe_float((current_position or {}).get("amount") or (current_position or {}).get("contracts")) or 0.0
    has_position = abs(amt) > 0

    if not OFFLINE_TRADING_ENABLED or max_new_positions_left <= 0:
        return {
            "symbol": symbol,
            "action": "skip",
            "reason": "AI offline: entries disabled",
            "ai_unavailable": True,
            "confidence": 0.0,
        }

    if df_primary is None or df_primary.empty:
        return {
            "symbol": symbol,
            "action": "skip",
            "reason": "AI offline: insufficient market data",
            "ai_unavailable": True,
            "confidence": 0.0,
        }

    df_local = df_primary.copy()
    try:
        if "ema20" not in df_local.columns:
            df_local["ema20"] = ema(df_local["close"], 20)
        if "ema50" not in df_local.columns:
            df_local["ema50"] = ema(df_local["close"], 50)
        if "ema100" not in df_local.columns:
            df_local["ema100"] = ema(df_local["close"], 100)
        if "rsi14" not in df_local.columns:
            df_local["rsi14"] = rsi(df_local["close"], 14)
        if "atr14" not in df_local.columns:
            df_local["atr14"] = atr(df_local, 14)
        if not {"macd_line", "macd_signal", "macd_hist"} <= set(df_local.columns):
            fast = df_local["close"].ewm(span=12, adjust=False).mean()
            slow = df_local["close"].ewm(span=26, adjust=False).mean()
            macd_line = fast - slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df_local["macd_line"] = macd_line
            df_local["macd_signal"] = signal_line
            df_local["macd_hist"] = macd_line - signal_line
    except Exception as exc:
        return {
            "symbol": symbol,
            "action": "skip",
            "reason": f"AI offline: indicator calc failed ({type(exc).__name__})",
            "ai_unavailable": True,
            "confidence": 0.0,
        }

    last = df_local.iloc[-1]
    close = safe_float(last.get("close")) or 0.0
    ema20_val = safe_float(last.get("ema20"))
    ema50_val = safe_float(last.get("ema50"))
    ema100_val = safe_float(last.get("ema100"))
    rsi_val = safe_float(last.get("rsi14"))
    atr_val = safe_float(last.get("atr14")) or safe_float(last.get("atr"))
    macd_line_val = safe_float(last.get("macd_line"))
    macd_signal_val = safe_float(last.get("macd_signal"))
    macd_hist_val = safe_float(last.get("macd_hist") or last.get("macd"))

    if not (close and math.isfinite(close) and close > 0 and ema20_val and ema50_val and rsi_val is not None):
        return {
            "symbol": symbol,
            "action": "skip",
            "reason": "AI offline: missing indicators",
            "ai_unavailable": True,
            "confidence": 0.0,
        }

    # Bias direction based on news (avoid trading against strong negative/positive headlines).
    allow_long = True
    allow_short = True
    if OFFLINE_NEWS_BIAS_ENABLED and math.isfinite(news_score):
        if news_score <= -0.5:
            allow_long = False
        elif news_score >= 0.5:
            allow_short = False

    bull = ema20_val > ema50_val and close > ema50_val
    bear = ema20_val < ema50_val and close < ema50_val
    long_trend_allowed = bull and (ema100_val is None or close >= ema100_val)
    short_trend_allowed = bear and (ema100_val is None or close <= ema100_val)
    macd_bull = macd_hist_val is None or macd_hist_val >= 0
    macd_bear = macd_hist_val is None or macd_hist_val <= 0

    # "Range" hint: low EMA separation relative to ATR (avoid trend trades).
    ema_sep = abs(ema20_val - ema50_val)
    atr_ratio = (atr_val / close) if close > 0 and atr_val else 0.0
    range_hint = False
    if atr_val and math.isfinite(atr_val) and atr_val > 0:
        range_hint = (ema_sep / atr_val) < 0.35 or atr_ratio <= 0.008

    if has_position:
        side_raw = str((current_position or {}).get("side") or "").lower()
        pos_side = "buy" if side_raw in {"buy", "long"} or amt > 0 else "sell"
        close_side = "sell" if pos_side == "buy" else "buy"

        # Conservative exit rules (do not overtrade):
        # - Exit if trend clearly flipped against the position AND RSI confirms weakness/strength.
        # - Exit if strong opposite news bias (|news_score|>=0.7) and RSI is already unfavorable.
        flipped_against = (pos_side == "buy" and bear) or (pos_side == "sell" and bull)
        rsi_unfavorable = (pos_side == "buy" and rsi_val <= 40) or (pos_side == "sell" and rsi_val >= 60)
        macd_flip = False
        if macd_hist_val is not None and macd_signal_val is not None:
            macd_flip = (pos_side == "buy" and macd_hist_val < 0 and macd_line_val is not None and macd_line_val < macd_signal_val) or (
                pos_side == "sell" and macd_hist_val > 0 and macd_line_val is not None and macd_line_val > macd_signal_val
            )
        strong_news_against = False
        if OFFLINE_NEWS_BIAS_ENABLED and math.isfinite(news_score):
            if pos_side == "buy" and news_score <= -0.7:
                strong_news_against = True
            if pos_side == "sell" and news_score >= 0.7:
                strong_news_against = True

        if (flipped_against and rsi_unfavorable) or (macd_flip and rsi_unfavorable) or (strong_news_against and rsi_unfavorable):
            reason_bits = []
            if flipped_against:
                reason_bits.append("trend_flip")
            if macd_flip:
                reason_bits.append("macd_flip")
            if strong_news_against:
                reason_bits.append(f"news_against={news_score:+.2f}")
            reason_bits.append(f"rsi={rsi_val:.1f}")
            return {
                "symbol": symbol,
                "action": "close",
                "side": close_side,
                "reason": "AI offline rules: exit (" + ", ".join(reason_bits) + ")",
                "ai_unavailable": True,
                "confidence": 0.0,
            }

        return {
            "symbol": symbol,
            "action": "manage",
            "side": pos_side,
            "reason": f"AI offline: manage existing position (rsi={rsi_val:.1f}, range={range_hint})",
            "regime": "trend" if bull or bear else "counter",
            "ai_unavailable": True,
            "confidence": 0.0,
            "config": {"sl_atr": SL_ATR, "tp_atr": TP_ATR},
        }

    chosen_side: str | None = None
    reason_bits: list[str] = []

    strong_news = OFFLINE_NEWS_BIAS_ENABLED and math.isfinite(news_score) and abs(news_score) >= 0.6

    regime = "flat" if range_hint else ("trend" if bull or bear else "counter")
    reason_bits.append(f"regime={regime}")

    if not range_hint:
        # Trend-following (trend)
        if long_trend_allowed and allow_long and 45 <= rsi_val <= 70 and macd_bull:
            chosen_side = "buy"
            reason_bits.append("trend bull (ema20>ema50>ema100)")
            if macd_hist_val is not None:
                reason_bits.append(f"macd_hist={macd_hist_val:+.3f}")
        elif short_trend_allowed and allow_short and 30 <= rsi_val <= 55 and macd_bear:
            chosen_side = "sell"
            reason_bits.append("trend bear (ema20<ema50<ema100)")
            if macd_hist_val is not None:
                reason_bits.append(f"macd_hist={macd_hist_val:+.3f}")

        # Counter-trend fallback (outside range)
        if chosen_side is None:
            if allow_long and rsi_val <= 32:
                chosen_side = "buy"
                reason_bits.append("countertrend long (rsi<=32)")
            elif allow_short and rsi_val >= 68:
                chosen_side = "sell"
                reason_bits.append("countertrend short (rsi>=68)")
    else:
        # Flat / range regime
        quiet_news = not (OFFLINE_NEWS_BIAS_ENABLED and abs(news_score) >= 0.4)
        mild_trend_long = allow_long and bull and macd_bull and 40 <= rsi_val <= 62
        mild_trend_short = allow_short and bear and macd_bear and 38 <= rsi_val <= 60
        drift_long = allow_long and macd_bull and 42 <= rsi_val <= 65
        drift_short = allow_short and macd_bear and 35 <= rsi_val <= 58

        if strong_news and allow_long and bull and macd_bull and 40 <= rsi_val <= 65:
            chosen_side = "buy"
            reason_bits.append("range+news long (ema20>ema50, macd>=0)")
        elif strong_news and allow_short and bear and macd_bear and 35 <= rsi_val <= 60:
            chosen_side = "sell"
            reason_bits.append("range+news short (ema20<ema50, macd<=0)")
        elif mild_trend_long:
            chosen_side = "buy"
            reason_bits.append("range trend long (ema20>ema50, macd>=0, rsi 40-62)")
        elif mild_trend_short:
            chosen_side = "sell"
            reason_bits.append("range trend short (ema20<ema50, macd<=0, rsi 38-60)")
        elif drift_long:
            chosen_side = "buy"
            reason_bits.append("range drift long (macd>=0, rsi 42-65)")
        elif drift_short:
            chosen_side = "sell"
            reason_bits.append("range drift short (macd<=0, rsi 35-58)")
        elif quiet_news and allow_long and rsi_val <= 35 and macd_bull:
            chosen_side = "buy"
            reason_bits.append("range mean-reversion (rsi<=35)")
        elif quiet_news and allow_short and rsi_val >= 65 and macd_bear:
            chosen_side = "sell"
            reason_bits.append("range mean-reversion (rsi>=65)")

    if chosen_side is None:
        return {
            "symbol": symbol,
            "action": "skip",
            "reason": f"AI offline: no signal (rsi={rsi_val:.1f}, range={range_hint})",
            "regime": regime,
            "ai_unavailable": True,
            "confidence": 0.0,
        }

    if OFFLINE_NEWS_BIAS_ENABLED and math.isfinite(news_score):
        reason_bits.append(f"news_bias={news_score:+.2f}")
    reason_bits.append(f"rsi={rsi_val:.1f}")

    # Adjust position size based on news, regime and volatility.
    base_notional = CURRENT_RISK_PCT or RISK_PCT or 0.01
    notional = float(base_notional)

    # 1) News bias: side-aware, плавный шкалирующий коэффициент.
    if OFFLINE_NEWS_BIAS_ENABLED and math.isfinite(news_score):
        try:
            news_val = float(news_score)
        except (TypeError, ValueError):
            news_val = 0.0
        news_val = max(-1.0, min(1.0, news_val))
        max_boost = 0.3  # до ±30 % по одной только новостной компоненте
        if chosen_side == "buy":
            news_factor = 1.0 + max_boost * news_val
        else:  # sell / short
            news_factor = 1.0 - max_boost * news_val
        news_factor = max(0.5, min(1.5, news_factor))
        notional *= news_factor

    # 2) Режим: тренд / контртренд. Для флета основной даунскейл идёт через range_hint ниже.
    regime_factor = 1.0
    if not range_hint:
        if regime == "trend":
            regime_factor = 1.15  # немного агрессивнее по тренду
        elif regime == "counter":
            regime_factor = 0.75  # аккуратнее в контртренде
    else:
        # Флет/рейндж: сохраняем отдельный коэффициент ниже через range_hint.
        regime_factor = 1.0
    notional *= regime_factor

    # 3) Рейндж: общая шкала риска в боковике.
    if range_hint:
        notional *= 0.6 if not strong_news else 0.75
    if atr_ratio and atr_ratio > 0.025:
        notional *= 0.8
    if MAX_DYNAMIC_RISK_PCT:
        notional = min(notional, float(MAX_DYNAMIC_RISK_PCT))
    if MIN_DYNAMIC_RISK_PCT:
        notional = max(notional, float(MIN_DYNAMIC_RISK_PCT))
    notional = max(0.0005, notional)

    return {
        "symbol": symbol,
        "action": "open",
        "side": chosen_side,
        "reason": "AI offline rules: " + "; ".join(reason_bits),
        "regime": regime,
        "ai_unavailable": True,
        "confidence": 0.82,
        "config": {"sl_atr": SL_ATR, "tp_atr": TP_ATR},
        "notional_pct": notional,
    }

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


def _update_runtime_status_field(key: str, value: Any) -> None:
    state = _read_runtime_status()
    if not isinstance(state, dict):
        state = {}
    state[key] = value
    try:
        RUNTIME_STATUS_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
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
    record = f"[{stamp}] {msg}"
    _LOG_HISTORY.append(record)
    print(color + record + Style.RESET_ALL)
    if TELEGRAM_FORWARD_LOGS:
        _enqueue_tg_log(record)


def _append_user_log(user_id: int | None, text: str) -> None:
    """Append a line to the shared chat log (runtime/chat.log)."""
    if user_id is None:
        return
    try:
        root = Path("runtime")
        root.mkdir(parents=True, exist_ok=True)
        chat_log = root / "chat.log"
        now = _current_log_time().strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(chat_log, "a", encoding="utf-8") as fh:
            fh.write(f"[{now}] user={user_id} {text}\n")
    except Exception:
        pass


def _append_user_bybit_log(user_id: str | int | None, text: str) -> None:
    """Append a trading-history style message to a bybit.log per user.

    File location: runtime/<user_id>/bybit.log
    """
    if user_id is None:
        return
    try:
        uid = str(user_id)
        user_dir = Path("runtime") / uid
        user_dir.mkdir(parents=True, exist_ok=True)
        bybit_log = user_dir / "bybit.log"
        now = _current_log_time().strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(bybit_log, "a", encoding="utf-8") as fh:
            fh.write(f"[{now}] {text}\n")
    except Exception:
        # keep non-fatal
        pass


def _append_trade_log(text: str) -> None:
    """Append a trading line to the shared runtime/trades.log."""
    try:
        root = Path("runtime")
        root.mkdir(parents=True, exist_ok=True)
        trade_log = root / "trades.log"
        now = _current_log_time().strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(trade_log, "a", encoding="utf-8") as fh:
            fh.write(f"[{now}] {text}\n")
    except Exception:
        pass


def _ensure_user_bybit_log_for_all() -> None:
    """Ensure every user configured has a bybit.log file in runtime/USERID/bybit.log

    This is called at startup – if missing, it creates the folder and the file.
    """
    try:
        cfg = _load_users_config()
        users_list = cfg.get("users") or []
        for entry in users_list:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("id")
            if not uid:
                continue
            user_dir = Path("runtime") / str(uid)
            user_dir.mkdir(parents=True, exist_ok=True)
            p = user_dir / "bybit.log"
            if not p.exists():
                try:
                    p.write_text("", encoding="utf-8")
                except Exception:
                    pass
    except Exception as exc:
        log(f"[WARN] Failed to ensure per-user bybit logs: {exc}", Fore.YELLOW)

_TG_LAST_SEND_TS = 0.0
_TG_LAST_MESSAGE: str | None = None
_TG_LAST_MESSAGE_TS = 0.0
_TG_LOG_BUFFER: deque[str] = deque()
_TG_LOG_LAST_FLUSH = 0.0
_TG_LOG_RATE_WINDOW: deque[float] = deque()
_TG_IN_SEND = 0
_TG_LOCK = threading.RLock()
_LOG_TS_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_LOG_TZ_PATTERN = re.compile(r"UTC([+-])(\d{2}):(\d{2})")
_TELEGRAM_CONFIGURED = False
_TELEGRAM_WEBHOOK_THREAD: threading.Thread | None = None
_TELEGRAM_WEBHOOK_SERVER: ThreadingHTTPServer | None = None
_TELEGRAM_COMMAND_SIGNATURE: tuple[tuple[str, str], ...] = ()
_TELEGRAM_WEBHOOK_SIGNATURE: tuple[str, str | None] | None = None
_TELEGRAM_LONG_POLL_THREAD: threading.Thread | None = None
_TELEGRAM_LONG_POLL_STOP: threading.Event | None = None
_LOG_HISTORY: deque[str] = deque(maxlen=200)
LATEST_STATUS: dict[str, Any] = {}
PENDING_USERBOT_CREATION: dict[int, dict[str, Any]] = {}
_SCHEDULE_EVENT = threading.Event()
_SCHEDULE_OVERRIDE_LOCK = threading.RLock()
_SCHEDULE_OVERRIDE: dict[str, Any] | None = None
_LAST_WORKTREE_STATE_DIRTY: bool = False
_LAST_WORKTREE_STATE_HASH: str = ""
_LAST_GIT_NOTIFICATION: tuple[str, float] | None = None

def _send_git_notification(message: str):
    global _LAST_GIT_NOTIFICATION
    now_ts = time.time()
    if _LAST_GIT_NOTIFICATION and _LAST_GIT_NOTIFICATION[0] == message:
        # Skip duplicate notifications if they match the previous payload.
        if now_ts - _LAST_GIT_NOTIFICATION[1] < 300:
            return
    _LAST_GIT_NOTIFICATION = (message, now_ts)
    log(message, Fore.LIGHTBLACK_EX)
    thread_target = TG_GIT_TOPIC_ID if TG_GIT_TOPIC_ID is not None else TG_TOPIC_ID
    send_tg(message, thread_id=thread_target, no_prefix=True)


def _send_inprogress_notification(message: str):
    global _LAST_INPROGRESS_MESSAGE
    if not INPROGRESS_WIP_ENABLED:
        return
    if message == _LAST_INPROGRESS_MESSAGE:
        return
    _LAST_INPROGRESS_MESSAGE = message
    log(message, Fore.LIGHTBLACK_EX)
    thread_target = TELEGRAM_INPROGRESS_THREAD_ID if TELEGRAM_INPROGRESS_THREAD_ID is not None else TG_TOPIC_ID
    send_tg(message, thread_id=thread_target, no_log_forward=True)


def _send_results_notification(message: str):
    log(message, Fore.CYAN)
    thread_target = TELEGRAM_RESULTS_THREAD_ID if TELEGRAM_RESULTS_THREAD_ID is not None else TG_TOPIC_ID
    send_tg(message, thread_id=thread_target, no_log_forward=True)


def _send_status_notification(message: str):
    log(message, Fore.LIGHTBLUE_EX)
    thread_target = TELEGRAM_STATUS_THREAD_ID if TELEGRAM_STATUS_THREAD_ID is not None else TG_TOPIC_ID
    send_tg(message, thread_id=thread_target, no_log_forward=True)


def _split_message(text: str, chunk_limit: int = 3800) -> list[str]:
    if len(text) <= chunk_limit:
        return [text]
    lines = text.splitlines()
    if not lines:
        return [text[:chunk_limit]]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        candidate_len = len(line)
        if current and current_len + candidate_len + 1 > chunk_limit:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += candidate_len + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [text[:chunk_limit]]


def _wait_for_log_slot() -> None:
    if TELEGRAM_LOG_MAX_MESSAGES_PER_WINDOW <= 0:
        return
    while True:
        now = time.time()
        with _TG_LOCK:
            window = TELEGRAM_LOG_RATE_LIMIT_WINDOW
            while _TG_LOG_RATE_WINDOW and now - _TG_LOG_RATE_WINDOW[0] > window:
                _TG_LOG_RATE_WINDOW.popleft()
            if len(_TG_LOG_RATE_WINDOW) < TELEGRAM_LOG_MAX_MESSAGES_PER_WINDOW:
                return
            oldest = _TG_LOG_RATE_WINDOW[0]
        sleep_for = max(TG_MIN_INTERVAL, (oldest + TELEGRAM_LOG_RATE_LIMIT_WINDOW) - now + 0.05)
        time.sleep(sleep_for)


def _record_log_slot() -> None:
    if TELEGRAM_LOG_MAX_MESSAGES_PER_WINDOW <= 0:
        return
    with _TG_LOCK:
        _TG_LOG_RATE_WINDOW.append(time.time())


def _format_tg_log_batch(batch: Sequence[str]) -> str:
    if not batch:
        return ""
    first_ts = (batch[0].split("]", 1)[0] if batch[0].startswith("[") else "").lstrip("[")
    last_ts = (batch[-1].split("]", 1)[0] if batch[-1].startswith("[") else "").lstrip("[")
    if first_ts and last_ts and first_ts != last_ts:
        header = f"ℹ️ Logs x{len(batch)} ({first_ts} > {last_ts})"
    elif first_ts:
        header = f"🪵 Logs x{len(batch)} ({first_ts})"
    else:
        header = f"🪵 Logs x{len(batch)}"
    body = "\n".join(batch)
    return f"{header}\n```\n{body}\n```"


def _enqueue_tg_log(record: str) -> None:
    if not TELEGRAM_FORWARD_LOGS:
        return
    should_flush = False
    with _TG_LOCK:
        if not TELEGRAM_FORWARD_LOGS:
            return
        _TG_LOG_BUFFER.append(record)
        now = time.time()
        if _TG_IN_SEND == 0 and (
            len(_TG_LOG_BUFFER) >= TELEGRAM_LOG_BATCH_SIZE
            or (now - _TG_LOG_LAST_FLUSH) >= TELEGRAM_LOG_FLUSH_INTERVAL
        ):
            should_flush = True
    if should_flush:
        _flush_tg_log_buffer()


def _flush_tg_log_buffer(force: bool = False) -> None:
    global _TG_LOG_LAST_FLUSH
    if not TELEGRAM_FORWARD_LOGS:
        with _TG_LOCK:
            _TG_LOG_BUFFER.clear()
        return
    with _TG_LOCK:
        if not TELEGRAM_FORWARD_LOGS:
            _TG_LOG_BUFFER.clear()
            return
        if not _TG_LOG_BUFFER:
            return
        if _TG_IN_SEND > 0 and not force:
            return
        now = time.time()
        if (
            not force
            and len(_TG_LOG_BUFFER) < TELEGRAM_LOG_BATCH_SIZE
            and (now - _TG_LOG_LAST_FLUSH) < TELEGRAM_LOG_FLUSH_INTERVAL
        ):
            return
        snapshot = list(_TG_LOG_BUFFER)
        _TG_LOG_BUFFER.clear()
        _TG_LOG_LAST_FLUSH = now
    pending_batches: list[list[str]] = []
    chunk: list[str] = []
    chunk_len = 0
    for entry in snapshot:
        entry_len = len(entry)
        if chunk and (
            len(chunk) >= TELEGRAM_LOG_BATCH_SIZE
            or chunk_len + entry_len + 1 > 3500
        ):
            pending_batches.append(chunk)
            chunk = []
            chunk_len = 0
        chunk.append(entry)
        chunk_len += entry_len + 1
    if chunk:
        pending_batches.append(chunk)
    for batch in pending_batches:
        payload = _format_tg_log_batch(batch)
        if not payload:
            continue
        _wait_for_log_slot()
        message_id = send_tg(
            payload,
            thread_id=TELEGRAM_LOG_THREAD_ID,
            no_log_forward=True,
        )
        if message_id is None:
            with _TG_LOCK:
                for entry in reversed(batch):
                    _TG_LOG_BUFFER.appendleft(entry)
            break
        _record_log_slot()
atexit.register(_flush_tg_log_buffer, True)


def send_tg(msg: str | Sequence[str], **extra):
    if not TG_TOKEN or not TG_CHAT:
        return None
    global _TG_LAST_SEND_TS, _TG_LAST_MESSAGE, _TG_LAST_MESSAGE_TS, _TG_IN_SEND
    if isinstance(msg, (list, tuple, set)):
        message_text = "\\n".join(str(part) for part in msg if part)
    else:
        message_text = str(msg)
    if not message_text:
        return None
    extra_payload = dict(extra) if extra else {}
    skip_prefix = bool(extra_payload.pop('no_prefix', False))
    _ = extra_payload.pop('no_log_forward', None)
    if TELEGRAM_MESSAGE_PREFIX and not skip_prefix:
        message_text = f"[{TELEGRAM_MESSAGE_PREFIX}] {message_text}"
    chat_override = extra_payload.pop('chat_id_override', None)
    thread_override = extra_payload.pop('thread_id', None)
    target_chat = chat_override if chat_override is not None else TG_CHAT
    thread_candidate = thread_override if thread_override is not None else TG_TOPIC_ID
    thread_id_int = safe_int(thread_candidate) if thread_candidate is not None else None

    chunks = _split_message(message_text)
    now_ts = time.time()
    with _TG_LOCK:
        if (
            len(chunks) == 1
            and _TG_LAST_MESSAGE == chunks[0]
            and (now_ts - _TG_LAST_MESSAGE_TS) < TG_DUP_WINDOW
        ):
            return None
        _TG_IN_SEND += 1

    last_message_id = None
    try:
        for chunk_text in chunks:
            attempt = 0
            last_error = None
            while attempt < TG_RETRY_ATTEMPTS:
                attempt += 1
                with _TG_LOCK:
                    last_send_ts = _TG_LAST_SEND_TS
                delay_needed = TG_MIN_INTERVAL - (time.time() - last_send_ts)
                if delay_needed > 0:
                    time.sleep(delay_needed)
                payload = {'chat_id': target_chat, 'text': chunk_text}
                if thread_id_int is not None and 'message_thread_id' not in extra_payload:
                    payload['message_thread_id'] = thread_id_int
                payload.update(extra_payload)
                try:
                    response = requests.post(
                        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                        json=payload,
                        timeout=5,
                    )
                except Exception as exc:
                    last_error = exc
                    log(f"⚠️ Telegram ({attempt}/{TG_RETRY_ATTEMPTS}): {exc}", Fore.YELLOW)
                else:
                    try:
                        data = response.json()
                    except Exception:
                        last_error = f"{response.status_code} {response.text}"
                        log(f"⚠️ Telegram: декодирование ответа не удалось — {last_error}", Fore.YELLOW)
                    else:
                        if isinstance(data, dict) and data.get('ok'):
                            result = data.get('result') or {}
                            message_id = result.get('message_id')
                            sent_ts = time.time()
                            with _TG_LOCK:
                                _TG_LAST_SEND_TS = sent_ts
                                _TG_LAST_MESSAGE = chunk_text
                                _TG_LAST_MESSAGE_TS = sent_ts
                            last_message_id = message_id
                            break
                        last_error = data
                        log(f"⚠️ Telegram API ответил ошибкой: {data}", Fore.YELLOW)
                        if isinstance(data, dict) and data.get('error_code') == 429:
                            retry_after = data.get('parameters', {}).get('retry_after')
                            sleep_for = float(retry_after or (TG_RETRY_BACKOFF * attempt))
                            time.sleep(max(TG_MIN_INTERVAL, sleep_for))
                            continue
                        if (
                            isinstance(data, dict)
                            and data.get('error_code') == 400
                            and 'thread not found' in (data.get('description') or '').lower()
                        ):
                            thread_id_int = None
                            log('⚠️ Telegram topic not found, retrying without thread.', Fore.YELLOW)
                            time.sleep(TG_RETRY_BACKOFF * attempt)
                            continue
                time.sleep(TG_RETRY_BACKOFF * attempt)
            else:
                log(f"❌ Telegram send failed after {TG_RETRY_ATTEMPTS} attempts: {last_error}", Fore.RED)
                return last_message_id
    finally:
        should_flush = False
        with _TG_LOCK:
            _TG_IN_SEND = max(0, _TG_IN_SEND - 1)
            if TELEGRAM_FORWARD_LOGS and _TG_IN_SEND == 0 and _TG_LOG_BUFFER:
                should_flush = True
        if should_flush:
            _flush_tg_log_buffer()
    return last_message_id


def send_tg_decision(msg: str | Sequence[str], **extra) -> None:
    """Gate verbose Telegram messages behind TELEGRAM_DECISIONS_VERBOSE."""
    if TELEGRAM_DECISIONS_VERBOSE:
        send_tg(msg, **extra)


def send_tg_photo(
    path: Path,
    *,
    caption: str | None = None,
    chat_id_override: int | None = None,
    thread_id: int | None = None,
) -> int | None:
    if not TG_TOKEN:
        return None
    target_chat = chat_id_override if chat_id_override is not None else TG_CHAT
    if target_chat is None:
        return None
    payload = {"chat_id": target_chat}
    thread_candidate = thread_id if thread_id is not None else TELEGRAM_LOG_THREAD_ID
    thread_id_int = safe_int(thread_candidate) if thread_candidate is not None else None
    if thread_id_int is not None:
        payload["message_thread_id"] = thread_id_int
    if caption:
        payload["caption"] = caption
    try:
        with path.open("rb") as fh:
            response = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                data=payload,
                files={"photo": fh},
                timeout=10,
            )
        data = response.json()
    except Exception as exc:
        log(f"[WARN] Telegram photo send failed: {exc}", Fore.YELLOW)
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        log(f"[WARN] Telegram photo API error: {data}", Fore.YELLOW)
        return None
    return data.get("result", {}).get("message_id")


def _delete_tg_message(chat_id: int, message_id: int) -> bool:
    if not TG_TOKEN or not chat_id or not message_id:
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/deleteMessage",
            json={
                "chat_id": chat_id,
                "message_id": message_id,
            },
            timeout=5,
        )
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if isinstance(data, dict) and data.get("ok"):
            return True
    except Exception as exc:
        log(f"[WARN] Failed to delete Telegram message {message_id} in {chat_id}: {exc}", Fore.YELLOW)
    return False


def _pin_tg_message(chat_id: int, message_id: int) -> None:
    if not TG_TOKEN or not chat_id or not message_id:
        return
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "disable_notification": True,
    }
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/pinChatMessage",
            json=payload,
            timeout=5,
        )
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if not (isinstance(data, dict) and data.get("ok")):
            log(f"[WARN] pinChatMessage failed: {data}", Fore.YELLOW)
    except Exception as exc:
        log(f"[WARN] Не удалось закрепить сообщение {message_id}: {exc}", Fore.YELLOW)
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
        log("🆕 Повреждён файл changelog_state.json, начинаем заново.", Fore.YELLOW)
        return {}
    return data if isinstance(data, dict) else {}


def _save_changelog_state(state: dict) -> None:
    try:
        CHANGELOG_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"⚠️ Не удалось сохранить состояние changelog: {exc}", Fore.YELLOW)


def _load_results_state() -> dict:
    try:
        raw = RESULTS_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log(f"[RESULTS] Не удалось прочитать results_state.json: {exc}", Fore.YELLOW)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log("[RESULTS] Файл results_state.json повреждён, начинаем заново.", Fore.YELLOW)
        return {}
    return data if isinstance(data, dict) else {}


def _save_results_state(state: dict) -> None:
    try:
        RESULTS_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"[RESULTS] Не удалось сохранить results_state.json: {exc}", Fore.YELLOW)


def _results_order_key(detail: dict[str, Any]) -> str:
    order_id = str(detail.get("id") or "").strip()
    if order_id:
        return order_id
    symbol = str(detail.get("symbol") or "").upper()
    timestamp = detail.get("timestamp") or ""
    pnl = detail.get("pnl")
    return f"{symbol}|{timestamp}|{pnl}"


def _format_closed_order_line(detail: dict[str, Any]) -> str:
    symbol = str(detail.get("symbol") or "").upper() or "?"
    side = (detail.get("side") or "").upper()
    pnl_val = safe_float(detail.get("pnl"))
    amount_val = safe_float(detail.get("amount"))
    price_val = safe_float(detail.get("price"))
    ts_iso = detail.get("timestamp")
    time_label = ""
    if ts_iso:
        try:
            dt = datetime.datetime.fromisoformat(ts_iso)
            local_tz = _current_local_tz()
            if local_tz:
                dt = dt.astimezone(local_tz)
            time_label = dt.strftime("%H:%M")
        except Exception:
            time_label = ts_iso[:16]
    amount_text = f"{amount_val:.4f}" if isinstance(amount_val, (int, float)) and math.isfinite(amount_val) else "-"
    price_text = f"@ {price_val:.4f}" if isinstance(price_val, (int, float)) and math.isfinite(price_val) else ""
    pnl_text = f"{pnl_val:+.2f} USDT" if pnl_val is not None and math.isfinite(pnl_val) else "n/a"
    time_text = f" [{time_label}]" if time_label else ""
    return f"- {symbol} {side or '?'} {amount_text} {price_text} (PnL {pnl_text}){time_text}"


def _update_daily_closed_pnl_map(daily_map: dict[str, float], orders: Sequence[dict[str, Any]]) -> bool:
    if not isinstance(daily_map, dict):
        return False
    tzinfo = _current_local_tz() or datetime.timezone.utc
    changed = False
    for detail in orders:
        pnl_val = safe_float(detail.get("pnl"))
        ts_iso = detail.get("timestamp")
        if pnl_val is None or ts_iso is None:
            continue
        try:
            dt = datetime.datetime.fromisoformat(ts_iso)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        day_key = dt.astimezone(tzinfo).date().isoformat()
        prev = safe_float(daily_map.get(day_key)) or 0.0
        daily_map[day_key] = prev + float(pnl_val)
        changed = True
    if changed:
        max_days = 30
        keys_sorted = sorted(daily_map.keys())
        if len(keys_sorted) > max_days:
            for outdated in keys_sorted[:-max_days]:
                daily_map.pop(outdated, None)
    return changed


def _build_daily_pnl_percent_chart(
    history: Sequence[dict[str, Any]],
    *,
    latest_point: tuple[datetime.datetime, float, float | None] | tuple[datetime.datetime, float] | None = None,
    days: int = 7,
) -> tuple[str | None, list[tuple[datetime.date, float]]]:
    if not history and not latest_point:
        return None, []
    entries: list[tuple[datetime.datetime, float, float | None]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        ts_raw = entry.get("timestamp")
        equity_val = entry.get("equity")
        if not isinstance(ts_raw, str) or not ts_raw:
            continue
        try:
            equity_float = float(equity_val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(equity_float):
            continue
        ts_iso = ts_raw.replace("Z", "+00:00")
        try:
            ts = datetime.datetime.fromisoformat(ts_iso)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        realized_val = entry.get("realized")
        try:
            realized_float = float(realized_val) if realized_val is not None else None
        except (TypeError, ValueError):
            realized_float = None
        if realized_float is not None and not math.isfinite(realized_float):
            realized_float = None
        entries.append((ts, equity_float, realized_float))
    if latest_point:
        if len(latest_point) == 3:
            ts_latest, equity_latest, realized_latest = latest_point
        else:
            ts_latest, equity_latest = latest_point  # type: ignore[misc]
            realized_latest = None
        if isinstance(ts_latest, datetime.datetime) and math.isfinite(equity_latest):
            if ts_latest.tzinfo is None:
                ts_latest = ts_latest.replace(tzinfo=datetime.timezone.utc)
            realized_final = None
            if realized_latest is not None:
                try:
                    realized_val = float(realized_latest)
                except (TypeError, ValueError):
                    realized_val = None
                else:
                    realized_final = realized_val if math.isfinite(realized_val) else None
            entries.append((ts_latest, float(equity_latest), realized_final))
    if len(entries) < 2:
        return None, []
    entries.sort(key=lambda item: item[0])
    daily_closes: dict[datetime.date, tuple[datetime.datetime, float, float | None]] = {}
    for ts, equity_val, realized_val in entries:
        day_key = ts.date()
        prev = daily_closes.get(day_key)
        if prev is None or ts >= prev[0]:
            daily_closes[day_key] = (ts, equity_val, realized_val)
    sorted_days = sorted(daily_closes.keys())
    if len(sorted_days) < 2:
        return None, []
    changes: list[tuple[datetime.date, float]] = []
    prev_equity: float | None = None
    prev_realized: float | None = None
    for day in sorted_days:
        _, close_equity, close_realized = daily_closes[day]
        if prev_equity is not None:
            change_value: float | None = None
            if prev_realized is not None and close_realized is not None:
                change_value = close_realized - prev_realized
            else:
                change_value = close_equity - prev_equity
            base_equity = prev_equity if prev_equity and math.isfinite(prev_equity) else None
            if base_equity is None or abs(base_equity) < 1e-8:
                base_equity = 1.0
            if change_value is not None and math.isfinite(change_value):
                pct_change = (change_value / base_equity) * 100.0
                changes.append((day, pct_change))
        prev_equity = close_equity
        prev_realized = close_realized if close_realized is not None and math.isfinite(close_realized) else prev_realized
    if not changes:
        return None, []
    tail = changes[-days:]
    sparkline = _sparkline_from_values([pct for _, pct in tail])
    if not sparkline:
        return None, []
    return sparkline, tail


def _build_daily_closed_pnl_chart(
    daily_closed_map: Mapping[str, Any] | None,
    history: Sequence[dict[str, Any]],
    *,
    days: int = 7,
) -> tuple[str | None, list[tuple[datetime.date, float]]]:
    if not isinstance(daily_closed_map, Mapping) or not daily_closed_map:
        return None, []
    tzinfo = _current_local_tz() or datetime.timezone.utc
    equity_closes: dict[datetime.date, tuple[datetime.datetime, float]] = {}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        ts_raw = entry.get("timestamp")
        equity_val = safe_float(entry.get("equity"))
        if not isinstance(ts_raw, str) or equity_val is None or not math.isfinite(equity_val):
            continue
        ts = ts_raw.replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(ts)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        local_dt = dt.astimezone(tzinfo)
        day_key = local_dt.date()
        prev = equity_closes.get(day_key)
        if prev is None or local_dt >= prev[0]:
            equity_closes[day_key] = (local_dt, equity_val)
    if len(equity_closes) < 2:
        return None, []
    ordered_equity = sorted(equity_closes.items())
    changes: list[tuple[datetime.date, float]] = []
    for day_key_str, pnl_value in sorted(daily_closed_map.items()):
        pnl_float = safe_float(pnl_value)
        if pnl_float is None or not math.isfinite(pnl_float):
            continue
        try:
            day = datetime.date.fromisoformat(day_key_str)
        except ValueError:
            continue
        prev_equity: float | None = None
        for eq_day, (_, eq_value) in ordered_equity:
            if eq_day < day:
                prev_equity = eq_value
            else:
                break
        if prev_equity is None or abs(prev_equity) < 1e-8:
            continue
        pct = (pnl_float / prev_equity) * 100.0
        changes.append((day, pct))
    if not changes:
        return None, []
    tail = changes[-days:]
    sparkline = _sparkline_from_values([pct for _, pct in tail])
    if not sparkline:
        return None, []
    return sparkline, tail


def _load_release_state() -> dict:
    try:
        raw = RELEASE_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log(f"[RELEASE] Не удалось прочитать release_state.json: {exc}", Fore.YELLOW)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log("[RELEASE] Файл release_state.json повреждён, начинаем заново.", Fore.YELLOW)
        return {}
    return data if isinstance(data, dict) else {}


def _save_release_state(state: dict) -> None:
    try:
        RELEASE_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"[RELEASE] Не удалось сохранить release_state.json: {exc}", Fore.YELLOW)


def _infer_market_category(symbol: str, market_info: dict | None) -> str | None:
    category = None
    if isinstance(market_info, dict):
        base_info = market_info.get("info") if isinstance(market_info.get("info"), dict) else {}
        market_type = str(market_info.get("type") or base_info.get("category") or "").lower()
        if market_type in {"spot"}:
            return "spot"
        contract_type = str(
            market_info.get("contractType")
            or (base_info.get("contractType") if isinstance(base_info, dict) else "")
        ).lower()
        if _is_truthy_flag(market_info.get("linear")) or "linear" in contract_type:
            category = "linear"
        elif _is_truthy_flag(market_info.get("inverse")) or "inverse" in contract_type:
            category = "inverse"
        else:
            if market_type in {"linear", "inverse"}:
                category = market_type
            elif market_type in {"swap", "future"}:
                settle_coin = str(
                    market_info.get("settle")
                    or (base_info.get("settleCoin") if isinstance(base_info, dict) else "")
                    or (base_info.get("settle") if isinstance(base_info, dict) else "")
                ).upper()
                quote_coin = str(market_info.get("quote") or "").upper()
                if settle_coin and quote_coin:
                    category = "linear" if settle_coin == quote_coin else "inverse"
                else:
                    category = "linear" if market_type == "swap" else None
    if not category:
        sym = str(symbol or "")
        if sym.upper().endswith(":SPOT") or sym.upper().endswith(":SP"):
            return "spot"
        if ":" in sym:
            return "linear"
        if "/" in sym:
            return "spot"
    return category


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


def _default_command_payload() -> list[dict[str, str]]:
    return [{"command": cmd, "description": desc} for cmd, desc in TELEGRAM_DEFAULT_COMMANDS]


def configure_telegram_bot() -> None:
    global _TELEGRAM_CONFIGURED, _TELEGRAM_COMMAND_SIGNATURE, _TELEGRAM_WEBHOOK_SIGNATURE
    if not TG_TOKEN:
        return
    commands_payload = TELEGRAM_COMMANDS_LIST or _default_command_payload()
    commands_signature = tuple((item["command"], item["description"]) for item in commands_payload)
    if commands_signature != _TELEGRAM_COMMAND_SIGNATURE:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/setMyCommands",
                json={"commands": commands_payload},
                timeout=5,
            )
            data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            if not isinstance(data, dict) or not data.get("ok"):
                log(f"⚠️ Telegram setMyCommands failed: {data or response.text}", Fore.YELLOW)
            else:
                log("ℹ️ Telegram commands updated", Fore.LIGHTBLACK_EX)
                _TELEGRAM_COMMAND_SIGNATURE = commands_signature
        except Exception as exc:
            log(f"⚠️ Telegram setMyCommands error: {exc}", Fore.YELLOW)

    if TELEGRAM_WEBHOOK_URL:
        webhook_signature = (TELEGRAM_WEBHOOK_URL, TELEGRAM_WEBHOOK_SECRET)
        if webhook_signature != _TELEGRAM_WEBHOOK_SIGNATURE:
            payload = {
                "url": TELEGRAM_WEBHOOK_URL,
                "allowed_updates": ["message", "channel_post", "callback_query"],
                "drop_pending_updates": True,
            }
            if TELEGRAM_WEBHOOK_SECRET:
                payload["secret_token"] = TELEGRAM_WEBHOOK_SECRET
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{TG_TOKEN}/setWebhook",
                    json=payload,
                    timeout=5,
                )
                data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                if not isinstance(data, dict) or not data.get("ok"):
                    log(f"⚠️ Telegram setWebhook failed: {data or response.text}", Fore.YELLOW)
                else:
                    log(f"ℹ️ Telegram webhook set to {TELEGRAM_WEBHOOK_URL}", Fore.LIGHTBLACK_EX)
                    _TELEGRAM_WEBHOOK_SIGNATURE = webhook_signature
            except Exception as exc:
                log(f"⚠️ Telegram setWebhook error: {exc}", Fore.YELLOW)
    elif _TELEGRAM_WEBHOOK_SIGNATURE is not None:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/deleteWebhook",
                json={"drop_pending_updates": True},
                timeout=5,
            )
            log("ℹ️ Telegram webhook cleared", Fore.LIGHTBLACK_EX)
        except Exception as exc:
            log(f"⚠️ Telegram deleteWebhook error: {exc}", Fore.YELLOW)
        finally:
            _TELEGRAM_WEBHOOK_SIGNATURE = None
    _TELEGRAM_CONFIGURED = True


def _is_chat_allowed(chat_id: int) -> bool:
    if not TELEGRAM_ALLOWED_CHAT_IDS:
        return True
    return chat_id in TELEGRAM_ALLOWED_CHAT_IDS


def process_telegram_update(update: dict) -> None:
    message = update.get("message") or update.get("channel_post")
    if not isinstance(message, dict):
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            message = callback.get("message")
            data = callback.get("data")
            if isinstance(message, dict) and isinstance(data, str) and data.startswith("/"):
                chat_id = message.get("chat", {}).get("id")
                thread_id = message.get("message_thread_id")
                if isinstance(chat_id, int) and _is_chat_allowed(chat_id):
                    handle_telegram_command(chat_id, data, thread_id=thread_id)
        return
    chat = message.get("chat") or {}
    user_payload = message.get("from") or {}
    from_user_id = safe_int(user_payload.get("id"))
    chat_id = chat.get("id")
    if not isinstance(chat_id, int):
        return
    if not _is_chat_allowed(chat_id):
        log(f"ℹ️ Telegram update ignored from chat {chat_id}", Fore.LIGHTBLACK_EX)
        return
    text = message.get("text") or ""
    if not isinstance(text, str):
        return
    entities = message.get("entities") or []
    is_command = text.startswith("/") or any((isinstance(ent, dict) and ent.get("type") == "bot_command") for ent in entities)
    thread_id = message.get("message_thread_id")
    log(
        f"[TG] update chat={chat_id} thread={thread_id} from={from_user_id} command={is_command} text={text[:64]!r}",
        Fore.LIGHTBLACK_EX,
    )
    if not is_command:
        if from_user_id is not None and _process_pending_userbot_message(from_user_id, chat_id, message):
            return
        if (
            TELEGRAM_SUPPORT_THREAD_ID is not None
            and thread_id == TELEGRAM_SUPPORT_THREAD_ID
        ):
            handle_support_message(chat_id, text, thread_id=thread_id, message=message)
        # Log user personal chats to per-user log
        if from_user_id is not None and chat_id == from_user_id:
            _append_user_log(from_user_id, f"IN: {text}")
        return
    handle_telegram_command(chat_id, text, thread_id=thread_id, user_id=from_user_id)


class _TelegramWebhookHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        if TELEGRAM_WEBHOOK_PATH and not self.path.startswith(TELEGRAM_WEBHOOK_PATH):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        if TELEGRAM_WEBHOOK_SECRET:
            secret_header = self.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if secret_header != TELEGRAM_WEBHOOK_SECRET:
                self.send_response(403)
                self.end_headers()
                return
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        try:
            process_telegram_update(payload)
        except Exception as exc:  # pylint: disable=broad-except
            log(f"⚠️ Telegram webhook handler error: {exc}", Fore.YELLOW)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def start_telegram_webhook_server() -> None:
    global _TELEGRAM_WEBHOOK_THREAD, _TELEGRAM_WEBHOOK_SERVER, TELEGRAM_WEBHOOK_PATH
    if _TELEGRAM_WEBHOOK_THREAD is not None:
        return
    if TELEGRAM_WEBHOOK_PORT <= 0:
        return
    if not TG_TOKEN:
        return
    host = TELEGRAM_WEBHOOK_HOST or "0.0.0.0"
    path = TELEGRAM_WEBHOOK_PATH or "/telegram"
    if not path.startswith("/"):
        path = f"/{path}"
    TELEGRAM_WEBHOOK_PATH = path
    try:
        server = ThreadingHTTPServer((host, TELEGRAM_WEBHOOK_PORT), _TelegramWebhookHandler)
    except Exception as exc:
        log(f"⚠️ Failed to start Telegram webhook server: {exc}", Fore.YELLOW)
        return

    def _serve() -> None:
        log(f"ℹ️ Telegram webhook server listening on http://{host}:{TELEGRAM_WEBHOOK_PORT}{path}", Fore.LIGHTBLACK_EX)
        try:
            server.serve_forever()
        except Exception as exc:  # pylint: disable=broad-except
            log(f"⚠️ Telegram webhook server stopped: {exc}", Fore.YELLOW)

    thread = threading.Thread(target=_serve, daemon=True)
    _TELEGRAM_WEBHOOK_SERVER = server
    _TELEGRAM_WEBHOOK_THREAD = thread
    thread.start()


def stop_telegram_webhook_server() -> None:
    global _TELEGRAM_WEBHOOK_THREAD, _TELEGRAM_WEBHOOK_SERVER
    server = _TELEGRAM_WEBHOOK_SERVER
    if server:
        try:
            server.shutdown()
        except Exception:
            pass
        _TELEGRAM_WEBHOOK_SERVER = None
    thread = _TELEGRAM_WEBHOOK_THREAD
    if thread:
        if thread.is_alive():
            thread.join(timeout=5)
        _TELEGRAM_WEBHOOK_THREAD = None


def _collect_live_status_snapshot() -> dict[str, Any]:
    exchange = init_exchange()
    try:
        try:
            ensure_position_mode(exchange)
        except Exception:
            pass
        equity, available, balance_payload = fetch_usdt_equity(exchange)
        positions_map, _ = fetch_positions_snapshot(exchange)
        open_orders = fetch_all_open_orders_grouped(exchange, limit=200)
    finally:
        try:
            exchange.close()
        except Exception:
            pass
    total_unrealized = 0.0
    for payload in positions_map.values():
        total_unrealized += safe_float(payload.get("unrealizedPnl")) or 0.0
    orders_count = sum(len(bucket) for bucket in open_orders.values())
    realized = None
    if isinstance(balance_payload, dict):
        realized = safe_float(balance_payload.get("_realizedPnl"))
    stable_label = _stable_currency_label(balance_payload)
    return {
        "equity": equity,
        "available": available,
        "positions": positions_map,
        "positions_unrealized": total_unrealized,
        "orders": open_orders,
        "orders_count": orders_count,
        "stable_label": stable_label,
        "realized": realized,
    }


def _summarize_positions_lines(positions_map: dict[str, Any], limit: int = 10) -> tuple[list[str], float, int]:
    if not positions_map:
        return [], 0.0, 0
    lines: list[str] = []
    total_unrealized = 0.0
    items = sorted(positions_map.items())
    for idx, (symbol, payload) in enumerate(items, start=1):
        amount = safe_float(payload.get("amount"))
        if amount is None:
            amount = 0.0
        side = payload.get("side") or ("long" if amount >= 0 else "short")
        entry_price = safe_float(payload.get("entryPrice"))
        unreal = safe_float(payload.get("unrealizedPnl")) or 0.0
        total_unrealized += unreal
        amount_text = f"{abs(amount):.4f}"
        entry_txt = f" @ {entry_price:.4f}" if entry_price is not None and math.isfinite(entry_price) else ""
        lines.append(f"{symbol} — {side.upper()} {amount_text}{entry_txt} (PnL {unreal:+.2f} USDT)")
        if len(lines) >= limit:
            break
    overflow = max(0, len(items) - limit)
    return lines, total_unrealized, overflow


def start_telegram_long_polling() -> None:
    global _TELEGRAM_LONG_POLL_THREAD, _TELEGRAM_LONG_POLL_STOP
    if TELEGRAM_WEBHOOK_URL:
        return
    if _TELEGRAM_LONG_POLL_THREAD is not None:
        return
    if not TG_TOKEN:
        return
    stop_event = threading.Event()
    _TELEGRAM_LONG_POLL_STOP = stop_event
    # Ensure no webhook is active and clear pending updates to avoid 409 conflicts
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/deleteWebhook",
            json={"drop_pending_updates": True},
            timeout=5,
        )
    except Exception:
        pass

    def _poll_updates() -> None:
        nonlocal stop_event
        offset = 0
        log("ℹ️ Telegram long polling started", Fore.LIGHTBLACK_EX)
        while not stop_event.is_set():
            try:
                response = requests.get(
                    f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                    params={
                        "timeout": 25,
                        "offset": offset,
                        "allowed_updates": ["message", "channel_post", "callback_query"],
                    },
                    timeout=30,
                )
                data = response.json()
                if not isinstance(data, dict) or not data.get("ok"):
                    raise RuntimeError(data)
                updates = data.get("result") or []
                for update in updates:
                    try:
                        offset = max(offset, int(update.get("update_id", 0)) + 1)
                    except (TypeError, ValueError):
                        pass
                    try:
                        process_telegram_update(update)
                    except Exception as exc:  # pylint: disable=broad-except
                        log(f"[WARN] Telegram update processing error: {exc}", Fore.YELLOW)
                if not updates:
                    time.sleep(1.0)
            except Exception as exc:  # pylint: disable=broad-except
                if stop_event.is_set():
                    break
                log(f"[WARN] Telegram polling error: {exc}", Fore.YELLOW)
                time.sleep(5.0)

    thread = threading.Thread(target=_poll_updates, daemon=True)
    _TELEGRAM_LONG_POLL_THREAD = thread
    thread.start()


def stop_telegram_long_polling() -> None:
    global _TELEGRAM_LONG_POLL_THREAD, _TELEGRAM_LONG_POLL_STOP
    stop_event = _TELEGRAM_LONG_POLL_STOP
    thread = _TELEGRAM_LONG_POLL_THREAD
    if stop_event:
        stop_event.set()
    if thread:
        if thread.is_alive():
            thread.join(timeout=5)
    _TELEGRAM_LONG_POLL_THREAD = None
    _TELEGRAM_LONG_POLL_STOP = None


def shutdown_telegram_services() -> None:
    stop_telegram_long_polling()
    stop_telegram_webhook_server()


def _graph_interval_minutes() -> int:
    try:
        dynamic_default = int(max(1.0, DEFAULT_NEXT_RUN_MINUTES))
    except Exception:
        dynamic_default = GRAPH_SEND_INTERVAL_DEFAULT
    return max(0, env_int("GRAPH_SEND_INTERVAL_MINUTES", dynamic_default))


def _graph_send_each_cycle() -> bool:
    raw = os.getenv("GRAPH_SEND_EACH_CYCLE")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _graph_thread_id() -> int | None:
    candidate = TELEGRAM_LOG_THREAD_ID if TELEGRAM_LOG_THREAD_ID is not None else TELEGRAM_COMMAND_THREAD_ID
    return safe_int(candidate) if candidate is not None else None


def _generate_graphs() -> None:
    python_exec = sys.executable or "python"
    script = SCRIPT_DIR / "analyze_graphs.py"
    if not script.exists():
        log("[GRAPH] analyze_graphs.py not found; skipping generation.", Fore.YELLOW)
        return
    try:
        subprocess.run(
            [
                python_exec,
                str(script),
                "--state-dir",
                str(SCRIPT_DIR / "assets"),
                "--output",
                str(GRAPH_OUTPUT_DIR),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log("[GRAPH] Diagnostic plots regenerated.", Fore.LIGHTBLACK_EX)
    except Exception as exc:
        log(f"[GRAPH] Failed to regenerate plots: {exc}", Fore.YELLOW)


def _send_graph_photos() -> list[str]:
    sent_labels: list[str] = []
    if not TG_TOKEN or not TG_CHAT:
        log("[GRAPH] Telegram credentials missing; skipping graph delivery.", Fore.YELLOW)
        return sent_labels
    thread_id = _graph_thread_id()
    now_txt = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    captions = {
        "equity": f"Equity / Available margin ({now_txt})",
        "pnl": f"Closed / Unrealized PnL ({now_txt})",
        "signals": f"Signal distribution ({now_txt})",
    }
    for base, caption in captions.items():
        preferred = [
            GRAPH_OUTPUT_DIR / f"{base}.jpg",
            GRAPH_OUTPUT_DIR / f"{base}.png",
        ]
        path = next((p for p in preferred if p.exists()), None)
        if not path:
            log(f"[GRAPH] Plot {base} (jpg/png) missing; skipping send.", Fore.LIGHTBLACK_EX)
            continue
        msg_id = send_tg_photo(path, caption=caption, thread_id=thread_id)
        if msg_id:
            sent_labels.append(path.name)
            log(f"[GRAPH] Sent {path.name} to Telegram (msg {msg_id}).", Fore.LIGHTBLACK_EX)
        else:
            log(f"[GRAPH] Failed to send {path.name} to Telegram.", Fore.YELLOW)
    if not sent_labels:
        log("[GRAPH] No plots were sent (files missing or Telegram send failed).", Fore.YELLOW)
    return sent_labels


def maybe_send_graphs() -> None:
    interval = _graph_interval_minutes()
    force_each_cycle = _graph_send_each_cycle() or interval <= 0
    status = _read_runtime_status()
    last_sent = status.get("last_graph_sent")
    last_dt = None
    if last_sent:
        try:
            last_dt = datetime.datetime.fromisoformat(last_sent)
        except Exception:
            last_dt = None
    now = datetime.datetime.now(datetime.timezone.utc)
    if not force_each_cycle and last_dt and (now - last_dt).total_seconds() < interval * 60:
        return
    if force_each_cycle:
        log("[GRAPH] Forced graph push for this cycle.", Fore.LIGHTBLACK_EX)
    _generate_graphs()
    sent = _send_graph_photos()
    if sent:
        _update_runtime_status_field("last_graph_sent", now.isoformat())


def _build_help_message() -> str:
    commands = TELEGRAM_COMMANDS_LIST or _default_command_payload()
    lines = ["Команды закреплены в разделе Commands. Краткий список:"]
    for entry in commands:
        lines.append(f"/{entry['command']} — {entry['description']}")
    return "\n".join(lines)


def _build_start_keyboard() -> dict[str, Any]:
    commands = TELEGRAM_COMMANDS_LIST or _default_command_payload()
    buttons: list[list[dict[str, str]]] = []
    row: list[dict[str, str]] = []
    for entry in commands:
        cmd = entry.get("command")
        if not cmd:
            continue
        row.append({"text": f"/{cmd}"})
        if len(row) >= 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return {
        "keyboard": buttons or [[{"text": "/help"}, {"text": "/status"}]],
        "resize_keyboard": True,
        "one_time_keyboard": False,
    }


def _support_context_targets() -> list[Path]:
    base_targets = [
        SCRIPT_DIR / "README.md",
        SCRIPT_DIR / "CHANGELOG.txt",
        SCRIPT_DIR / "bybitbot_impl.py",
        SCRIPT_DIR / "bybitbot.py",
        SCRIPT_DIR / "manage_update.py",
        REPO_ROOT / "users" / "README.md",
    ]
    seen: set[str] = set()
    targets: list[Path] = []

    def add(path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        targets.append(path)

    for item in base_targets:
        add(item)
    try:
        for pattern in SUPPORT_CONTEXT_PATTERNS:
            for path in REPO_ROOT.rglob(pattern):
                if any(part in SUPPORT_CONTEXT_SKIP_DIRS for part in path.parts):
                    continue
                add(path)
    except Exception:
        pass
    return targets


def _extract_support_keywords(question: str | None) -> list[str]:
    if not question:
        return []
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9_/]{3,}", question.lower())
    aliases = {
        "песочниц": "sandbox",
        "песочницa": "sandbox",
        "sandbox": "sandbox",
        "sandboxe": "sandbox",
        "model": "model",
        "modely": "model",
        "модель": "model",
        "модели": "model",
        "моделях": "model",
        "ai": "ai",
        "gpt": "gpt",
        "логика": "logic",
        "логике": "logic",
        "logika": "logic",
        "logic": "logic",
        "strategy": "strategy",
        "стратегия": "strategy",
        "стратегии": "strategy",
        "торговля": "trade",
        "торговли": "trade",
        "торговый": "trade",
        "торговые": "trade",
        "решения": "decision",
        "решение": "decision",
        "decision": "decision",
        "order": "order",
        "ордер": "order",
        "ордера": "order",
    }
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        base = aliases.get(token, token)
        if len(base) < 3:
            continue
        if base in seen:
            continue
        seen.add(base)
        keywords.append(base)
    if not keywords:
        keywords = ["trade", "run_cycle", "decision"]
    return keywords


def _load_support_context_snippet(question: str | None = None) -> str:
    if SUPPORT_MAX_CONTEXT_BYTES <= 0:
        return ""
    keywords = _extract_support_keywords(question)
    targets = _support_context_targets()
    per_file_budget = max(512, SUPPORT_MAX_CONTEXT_BYTES // len(targets))
    snippets: list[str] = []
    for path in targets:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not raw:
            continue
        lower = raw.lower()
        collected: list[str] = []
        if keywords:
            for kw in keywords:
                idx = lower.find(kw)
                attempts = 0
                while idx != -1 and attempts < 5:
                    start = max(0, idx - 400)
                    end = min(len(raw), idx + 400)
                    chunk = raw[start:end].strip()
                    if chunk:
                        collected.append(chunk)
                    if len(collected) * 200 >= per_file_budget:
                        break
                    idx = lower.find(kw, idx + len(kw))
                    attempts += 1
        if not collected:
            collected = [raw[:per_file_budget].strip()]
        snippet_text = "\n---\n".join(collected)[:per_file_budget]
        snippets.append(f"### {path.name}\n{snippet_text}")
    combined = "\n\n".join(snippets)
    return combined[:SUPPORT_MAX_CONTEXT_BYTES]


def _handle_support_question(text: str, *, thread_id: Optional[int], reply_to: Optional[int]) -> None:
    question = text.strip()
    if not question:
        return
    if not AI_KEY:
        send_tg(
            "ℹ️ Не могу ответить автоматически: отсутствует OpenAI ключ.",
            thread_id=thread_id,
            reply_to_message_id=reply_to,
        )
        return
    support_model = AI_SUPPORT_MODEL or AI_MODEL
    context_blob = _load_support_context_snippet(question)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — технический помощник по торговому боту ByBit. "
                "Отвечай кратко на русском языке, опираясь на предоставленный фрагмент кода и документацию. "
                "Если информации недостаточно, скажи, что нужно уточнение."
            ),
        },
        {
            "role": "user",
            "content": f"Контекст:\n{context_blob}\n\nВопрос: {question}",
        },
    ]
    token_estimate = estimate_tokens(messages, support_model)
    if not _ensure_token_budget(token_estimate, support_model, "support reply"):
        send_tg(
            "ℹ️ Лимит токенов достигнут, не могу ответить автоматически прямо сейчас.",
            thread_id=thread_id,
            reply_to_message_id=reply_to,
        )
        return
    _log_ai_request(support_model, token_estimate, "support reply")
    client = OpenAI(api_key=AI_KEY, timeout=20)
    try:
        response = client.chat.completions.create(
            model=support_model,
            messages=messages,
            temperature=0.4,
        )
    except Exception as exc:
        log(f"[WARN] Support reply failed: {exc}", Fore.YELLOW)
        send_tg(
            "ℹ️ Не удалось получить ответ от модели, перешлите вопрос вручную.",
            thread_id=thread_id,
            reply_to_message_id=reply_to,
        )
        return
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = "ℹ️ Модель не дала ответа. Нужна дополнительная информация."
    _register_ai_usage(support_model, getattr(response, "usage", None), "support reply")
    send_tg(answer, thread_id=thread_id, reply_to_message_id=reply_to)


def _load_sandbox_state() -> dict[str, Any]:
    try:
        raw = SUPPORT_SANDBOX_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"sandboxes": []}
    except Exception:
        return {"sandboxes": []}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"sandboxes": []}
    if not isinstance(data, dict):
        return {"sandboxes": []}
    if not isinstance(data.get("sandboxes"), list):
        data["sandboxes"] = []
    return data


def _save_sandbox_state(state: dict[str, Any]) -> None:
    try:
        SUPPORT_SANDBOX_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"[WARN] Не удалось сохранить состояние песочниц: {exc}", Fore.YELLOW)


def _get_user_sandboxes(user_id: int) -> list[dict[str, Any]]:
    state = _load_sandbox_state()
    entries = state.get("sandboxes") or []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and entry.get("user_id") == user_id]


def _find_sandbox_entry(sandbox_id: str) -> dict[str, Any] | None:
    state = _load_sandbox_state()
    for entry in state.get("sandboxes") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == sandbox_id:
            return entry
    return None


def _register_sandbox_entry(
    *,
    user_id: int,
    bot_id: str,
    path: Path,
    promotable: bool,
    origin: str,
) -> dict[str, Any]:
    state = _load_sandbox_state()
    entries = state.setdefault("sandboxes", [])
    if not isinstance(entries, list):
        entries = []
        state["sandboxes"] = entries
    entry_id = path.name
    entry = {
        "id": entry_id,
        "user_id": user_id,
        "bot_id": bot_id,
        "path": str(path),
        "promotable": bool(promotable),
        "origin": origin[:200],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    entries = [e for e in entries if not (isinstance(e, dict) and e.get("id") == entry_id)]
    entries.append(entry)
    state["sandboxes"] = entries
    _save_sandbox_state(state)
    return entry


def _remove_sandbox_entry(sandbox_id: str) -> None:
    state = _load_sandbox_state()
    entries = state.get("sandboxes") or []
    if not isinstance(entries, list):
        entries = []
    entries = [entry for entry in entries if not (isinstance(entry, dict) and entry.get("id") == sandbox_id)]
    state["sandboxes"] = entries
    _save_sandbox_state(state)


def _sanitize_sandbox_secrets(root: Path, allowed_bot_ids: set[str], allow_root_env: bool) -> None:
    users_root = root / "users"
    if users_root.exists():
        for secrets_file in users_root.glob("*/secrets.env"):
            bot_name = secrets_file.parent.name
            if bot_name in allowed_bot_ids:
                continue
            try:
                secrets_file.write_text("# secrets hidden in sandbox\n", encoding="utf-8")
            except Exception:
                pass
    env_path = root / ".env"
    if env_path.exists() and not allow_root_env:
        try:
            env_path.write_text("# credentials hidden in sandbox\nSUPPORT_SANDBOX=1\n", encoding="utf-8")
        except Exception:
            pass


def _ensure_sandbox_user_config(root: Path, bot_id: str) -> None:
    users_dir = root / "users"
    try:
        users_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    users_json = users_dir / "users.json"
    if users_json.exists():
        try:
            if users_json.stat().st_size > 0:
                return
        except Exception:
            return
    sample_json = users_dir / "users.example.json"
    if sample_json.exists():
        try:
            shutil.copy2(sample_json, users_json)
            return
        except Exception:
            pass
    bot_label = bot_id or "default"
    payload = {
        "users": [
            {
                "id": bot_label,
                "label": bot_label,
                "enabled": True,
                "state_dir": f"runtime/{bot_label}",
            }
        ]
    }
    try:
        users_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _prepare_support_sandbox(
    request_text: str,
    *,
    requester_id: Optional[int],
    bot_id: str,
    promotable: bool,
) -> tuple[Path | None, str, str | None]:
    global SUPPORT_SANDBOX_ROOT
    if SUPPORT_SANDBOX_ROOT is None:
        return None, "support sandbox directory unavailable", None
    requester_id_int = int(requester_id) if requester_id is not None else 0
    user_limit = 10 if promotable else 3
    current_sandboxes = _get_user_sandboxes(requester_id_int) if requester_id is not None else []
    if len(current_sandboxes) >= user_limit:
        return None, f"достигнут лимит песочниц ({user_limit}). Удалите старые песочницы командой /sandbox delete <id>.", None
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", request_text.lower()).strip("-")[:24] or "request"
    sandbox_dir = SUPPORT_SANDBOX_ROOT / f"{timestamp}_{slug}"
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.log",
        "support_sandboxes",
    )
    try:
        shutil.copytree(REPO_ROOT, sandbox_dir, ignore=ignore)
    except Exception as exc:
        log(f"[WARN] Support sandbox copy failed: {exc}", Fore.YELLOW)
        return None, f"copy failed: {exc}", None
    allow_root_env = promotable and bot_id == "default"
    allowed_bots = {bot_id} if promotable else set()
    _sanitize_sandbox_secrets(sandbox_dir, allowed_bots, allow_root_env)
    try:
        request_file = sandbox_dir / "SUPPORT_REQUEST.txt"
        request_file.write_text(
            f"Original request:\n{textwrap.dedent(request_text).strip()}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        log(f"[WARN] Support sandbox note write failed: {exc}", Fore.YELLOW)
    _ensure_sandbox_user_config(sandbox_dir, bot_id)
    sim_output = ""
    simulation_log = sandbox_dir / "simulation.log"
    try:
        result = subprocess.run(
            [sys.executable, "bybitbot.py", "--list-users"],
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        sim_output = (result.stdout or "").strip()
        simulation_log.write_text(
            (result.stdout or "") + ("\n" + (result.stderr or "") if result.stderr else ""),
            encoding="utf-8",
        )
    except Exception as exc:
        sim_output = f"simulation failed: {exc}"
        try:
            simulation_log.write_text(sim_output, encoding="utf-8")
        except Exception:
            pass
    if requester_id is not None:
        entry = _register_sandbox_entry(
            user_id=requester_id_int,
            bot_id=bot_id,
            path=sandbox_dir,
            promotable=promotable,
            origin=request_text,
        )
        entry_id = entry.get("id")
    else:
        entry_id = sandbox_dir.name
    return sandbox_dir, sim_output, entry_id


def handle_support_message(chat_id: int, text: str, *, thread_id: Optional[int], message: dict) -> None:
    content = (text or "").strip()
    if not content:
        return
    lower = content.lower()
    words_wish = ("хочу", "нужно", "сделай", "добавь", "улучши", "пусть", "надо", "please", "feature")
    is_question = "?" in content or lower.startswith(("почему", "как", "что", "когда", "где"))
    is_wish = any(trigger in lower for trigger in words_wish) or not is_question
    reply_to = message.get("message_id") if isinstance(message.get("message_id"), int) else None
    log(
        f"[SUPPORT] chat={chat_id} thread={thread_id} question={is_question and not is_wish} text={content[:80]!r}",
        Fore.LIGHTBLACK_EX,
    )
    if is_question and not is_wish:
        send_tg(
            "🧠 Вопрос принят, формирую ответ…",
            thread_id=thread_id,
            reply_to_message_id=reply_to,
        )
        _handle_support_question(content, thread_id=thread_id, reply_to=reply_to)
        return
    current_bot_id = USER_ID
    requester_id = safe_int(message.get("from", {}).get("id"))
    is_owner = is_bot_owner(requester_id, current_bot_id) if requester_id is not None else False
    send_tg(
        "🧪 Получил пожелание, поднимаю тестовую песочницу…",
        thread_id=thread_id,
        reply_to_message_id=reply_to,
        chat_id_override=chat_id,
    )
    sandbox_dir, sim_excerpt, sandbox_id = _prepare_support_sandbox(
        content,
        requester_id=requester_id,
        bot_id=current_bot_id,
        promotable=is_owner,
    )
    if sandbox_dir is None:
        send_tg(
            f"ℹ️ Не удалось подготовить тестовое окружение: {sim_excerpt}",
            thread_id=thread_id,
            reply_to_message_id=reply_to,
        )
        return
    path_display = str(sandbox_dir).replace("`", "'")
    summary_lines = [
        "🧪 Подготовлена песочница для проверки пожелания.",
        f"Каталог: `{path_display}`",
    ]
    if sandbox_id:
        summary_lines.append(f"ID: `{sandbox_id}`")
    if sim_excerpt:
        summary_lines.append(f"Симуляция: {sim_excerpt[:200]}")
    send_tg(
        summary_lines,
        thread_id=thread_id,
        reply_to_message_id=reply_to,
        no_prefix=True,
        parse_mode="Markdown",
        chat_id_override=chat_id,
    )


def _format_positions_message(limit: int = 10, *, live: bool = False) -> str:
    if live:
        try:
            snapshot = _collect_live_status_snapshot()
        except Exception as exc:
            log(f"[WARN] Не удалось получить live-позиции: {exc}", Fore.YELLOW)
        else:
            lines, total_unrealized, overflow = _summarize_positions_lines(snapshot.get("positions") or {}, limit)
            if not lines:
                return "Открытых позиций нет."
            header = ["Открытые позиции (live):"]
            header.extend(lines)
            if overflow > 0:
                header.append(f"… ещё {overflow}")
            header.append(f"? PnL: {total_unrealized:+.2f} USDT")
            return "\n".join(header)
    positions = LATEST_STATUS.get("positions") or []
    if not positions:
        return "Открытых позиций нет."
    lines = ["Открытые позиции (последний цикл):"]
    total_unrealized = 0.0
    for pos in positions[:limit]:
        entry_price = pos.get("entry")
        entry_txt = f" @ {entry_price:.4f}" if entry_price and math.isfinite(entry_price) else ""
        unreal_val = safe_float(pos.get("unrealized", 0.0)) or 0.0
        total_unrealized += unreal_val
        lines.append(
            f"{pos.get('symbol')} — {pos.get('side')} {pos.get('amount'):.4f}{entry_txt} (PnL {unreal_val:+.2f} USDT)"
        )
    if len(positions) > limit:
        lines.append(f"… ещё {len(positions) - limit}")
    lines.append(f"? PnL: {total_unrealized:+.2f} USDT")
    return "\n".join(lines)


def _format_status_message(live: bool = False) -> str:
    if live:
        try:
            snapshot = _collect_live_status_snapshot()
        except Exception as exc:
            log(f"[WARN] Не удалось получить live-статус: {exc}", Fore.YELLOW)
        else:
            stable_label = snapshot.get("stable_label") or "USDT"
            lines = [
                "Статус (live):",
                f"Баланс: {snapshot.get('equity', 0.0):.2f} {stable_label}, доступно {snapshot.get('available', 0.0):.2f} {stable_label}",
                f"Открытых позиций: {len(snapshot.get('positions') or [])}",
                f"Открытых ордеров: {snapshot.get('orders_count', 0)}",
                f"? PnL по позициям: {snapshot.get('positions_unrealized', 0.0):+.2f} USDT",
            ]
            realized = snapshot.get("realized")
            if realized is not None:
                lines.append(f"Реализованный PnL (Bybit): {realized:+.2f} USDT")
            return "\n".join(lines)
    if not LATEST_STATUS:
        return "Статус пока недоступен."
    stable_label = LATEST_STATUS.get("stable_label") or "USDT"
    lines = [
        f"Цикл #{LATEST_STATUS.get('cycle', '?')} ({LATEST_STATUS.get('cycle_kind', 'normal')}/{LATEST_STATUS.get('cycle_mode', 'last')})",
    ]
    if LATEST_STATUS.get("timestamp"):
        lines.append(f"Обновлено: {LATEST_STATUS['timestamp']}")
    if LATEST_STATUS.get("equity_end") is not None:
        lines.append(
            f"Баланс: {LATEST_STATUS.get('equity_end', 0.0):.2f} {stable_label}, доступно {LATEST_STATUS.get('available_end', 0.0):.2f} {stable_label}"
        )
    elif LATEST_STATUS.get("equity_start") is not None:
        lines.append(
            f"Баланс: {LATEST_STATUS.get('equity_start', 0.0):.2f} {stable_label}, доступно {LATEST_STATUS.get('available_start', 0.0):.2f} {stable_label}"
        )
    if LATEST_STATUS.get("closed_pnl") is not None:
        lines.append(f"PnL ({_pnl_window_label()} closed): {LATEST_STATUS['closed_pnl']:+.2f} USDT")
    if LATEST_STATUS.get("unrealized") is not None:
        lines.append(f"PnL (open unrealized): {LATEST_STATUS['unrealized']:+.2f} USDT")
    lines.append(f"Открытых позиций: {len(LATEST_STATUS.get('positions') or [])}")
    return "\n".join(lines)


def _format_risk_message() -> str:
    return (
        "Риск-профиль:\n"
        f"Базовый риск: {RISK_PCT:.4f}\n"
        f"Текущий риск: {CURRENT_RISK_PCT:.4f}\n"
        f"Диапазон: {MIN_DYNAMIC_RISK_PCT:.4f} – {MAX_DYNAMIC_RISK_PCT:.4f}\n"
        f"Плечо (env): {LEVERAGE}x"
    )


def _format_log_history_message(lines: int = 12) -> str:
    return _render_log_block(list(_LOG_HISTORY)[-max(1, min(200, int(lines))):])


def _render_log_block(records: Sequence[str], header: str | None = None) -> str:
    if not records:
        return "Логов пока нет."
    title = header or f"Последние {len(records)} записей лога:"
    body = "\n".join(records)
    return f"{title}\n```\n{body}\n```"


def _parse_log_timestamp(line: str) -> datetime.datetime | None:
    match = _LOG_TS_PATTERN.search(line)
    if not match:
        return None
    dt_str = match.group(1)
    try:
        base_dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    tz_match = _LOG_TZ_PATTERN.search(line)
    if tz_match:
        sign = 1 if tz_match.group(1) == "+" else -1
        hours = int(tz_match.group(2))
        minutes = int(tz_match.group(3))
        offset = datetime.timedelta(hours=hours, minutes=minutes)
        tzinfo = datetime.timezone(sign * offset)
    else:
        tzinfo = datetime.timezone.utc
    return base_dt.replace(tzinfo=tzinfo)


def _filter_log_history(*, symbol: str | None = None, minutes: int | None = None, limit: int = 50) -> list[str]:
    if not _LOG_HISTORY:
        return []
    limit = max(1, min(200, int(limit)))
    symbol_norm = symbol.upper() if symbol else None
    threshold: datetime.datetime | None = None
    if minutes and minutes > 0:
        threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    matched: list[str] = []
    for line in reversed(_LOG_HISTORY):
        if symbol_norm and symbol_norm not in line.upper():
            continue
        ts = _parse_log_timestamp(line)
        if threshold and ts:
            ts_utc = ts.astimezone(datetime.timezone.utc)
            if ts_utc < threshold:
                if symbol_norm:
                    break
                if len(matched) >= limit:
                    break
                continue
        matched.append(line)
        if len(matched) >= limit:
            break
    matched.reverse()
    return matched


def _handle_logs_command(args: list[str]) -> str:
    if not _LOG_HISTORY:
        return "Логов пока нет."
    symbol = None
    minutes: int | None = None
    count = 20
    for raw in args:
        token = raw.strip()
        if not token:
            continue
        lower = token.lower()
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"symbol", "pair"} and value:
                symbol = value
                continue
            if key in {"minutes", "window"}:
                val = safe_int(value)
                if val and val > 0:
                    minutes = val
                continue
            if key in {"count", "lines"}:
                val = safe_int(value)
                if val and val > 0:
                    count = val
                continue
            continue
        if token.isdigit():
            count = int(token)
            continue
        if lower.endswith("m") and lower[:-1].isdigit():
            minutes = int(lower[:-1])
            continue
        if lower.endswith("h") and lower[:-1].isdigit():
            minutes = int(lower[:-1]) * 60
            continue
        if symbol is None:
            symbol = token
        elif minutes is None:
            extra = safe_int(token)
            if extra and extra > 0:
                minutes = extra
    if symbol and minutes is None:
        minutes = 60
    # Allow larger windows but keep a hard cap to avoid Telegram limits.
    count = max(5, min(500, count))
    filtered = _filter_log_history(symbol=symbol, minutes=minutes, limit=count)
    if not filtered:
        detail = f" по {symbol}" if symbol else ""
        return f"Нет логов за указанный интервал{detail}."
    if symbol and minutes:
        header = f"Логи для {symbol.upper()} за последние {minutes} мин ({len(filtered)} записей)"
    elif symbol:
        header = f"Логи для {symbol.upper()} ({len(filtered)} записей)"
    elif minutes:
        header = f"Логи за последние {minutes} мин ({len(filtered)} записей)"
    else:
        header = None
    return _render_log_block(filtered, header)


def _shorten_text(value: str | None, limit: int = 60) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_summary_list(label: str, entries: list[dict[str, Any]] | None, limit: int = 3) -> str | None:
    if not entries:
        return None
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = entry.get("item")
        count = entry.get("count")
        if not item or count is None:
            continue
        normalized.append(f"{item} ({count})")
        if len(normalized) >= limit:
            break
    if not normalized:
        return None
    return f"{label}: " + ", ".join(normalized)


def _format_last_cycle_summary() -> str | None:
    summary_path = _resolve_log_path("ai_decision_summary.json")
    if summary_path is None or not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    lines: list[str] = []
    entries = payload.get("entries")
    window = payload.get("window_hours")
    if entries is not None or window is not None:
        window_text = f"{window:.1f}" if isinstance(window, (int, float)) else "n/a"
        entries_text = str(entries) if entries is not None else "n/a"
        lines.append(f"Решений: {entries_text} (окно {window_text} ч)")
    for label, key in (
        ("Действия", "actions"),
        ("Символы", "symbols"),
        ("Skip", "skip_reasons"),
        ("Стороны", "sides"),
    ):
        formatted = _format_summary_list(label, payload.get(key), limit=4)
        if formatted:
            lines.append(formatted)
    recent = payload.get("recent") or []
    if recent:
        lines.append("Последние решения:")
        for entry in recent[:3]:
            symbol = entry.get("symbol") or "n/a"
            action = (entry.get("action") or "n/a").upper()
            reason = _shorten_text(entry.get("reason"), limit=50)
            confidence = entry.get("confidence")
            suffix = ""
            if reason:
                suffix += f" ({reason})"
            if isinstance(confidence, (int, float)):
                suffix += f" conf={confidence:.3f}"
            lines.append(f"- {symbol}: {action}{suffix}")
    return "\n".join(lines) if lines else None


def _handle_logmode_command(args: list[str]) -> str:
    global TELEGRAM_DECISIONS_VERBOSE

    def _summary_response(base: str) -> str:
        summary = _format_last_cycle_summary()
        if summary:
            return f"{base}\nПоследний цикл:\n{summary}"
        return f"{base}\nПока нет данных о последнем цикле."

    if not args:
        mode = "подробный" if TELEGRAM_DECISIONS_VERBOSE else "сводный"
        return f"Текущий режим логов: {mode}. Используйте /logmode [brief|verbose|toggle]."
    option = args[0].lower()
    if option in {"verbose", "detail", "full", "on", "1", "true"}:
        if TELEGRAM_DECISIONS_VERBOSE:
            return _summary_response("Подробные логи уже включены.")
        TELEGRAM_DECISIONS_VERBOSE = True
        return _summary_response("Подробные логи включены.")
    if option in {"brief", "summary", "off", "short", "0", "false"}:
        if not TELEGRAM_DECISIONS_VERBOSE:
            return "Сводный режим уже активен."
        TELEGRAM_DECISIONS_VERBOSE = False
        return "Сводный режим логов включён."
    if option in {"toggle", "switch"}:
        TELEGRAM_DECISIONS_VERBOSE = not TELEGRAM_DECISIONS_VERBOSE
        if TELEGRAM_DECISIONS_VERBOSE:
            return _summary_response("Режим логов переключён на подробный.")
        return "Режим логов переключён на сводный."
    return "Использование: /logmode [brief|verbose|toggle]"


def _read_runtime_status() -> dict[str, Any]:
    try:
        raw = RUNTIME_STATUS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _format_local_dt(value: datetime.datetime | None) -> str:
    if not value:
        return "не запланировано"
    try:
        local_tz = _current_local_tz() or datetime.datetime.now().astimezone().tzinfo
        local_dt = value.astimezone(local_tz)
    except Exception:
        local_dt = value
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def _ceil_datetime_to_step(value: datetime.datetime, step_minutes: int) -> datetime.datetime:
    if step_minutes <= 0:
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    step_seconds = step_minutes * 60
    ts = value.timestamp()
    ceil_ts = math.ceil(ts / step_seconds) * step_seconds
    return datetime.datetime.fromtimestamp(ceil_ts, tz=value.tzinfo)


def _floor_datetime_to_step(value: datetime.datetime, step_minutes: int) -> datetime.datetime:
    if step_minutes <= 0:
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    step_seconds = step_minutes * 60
    ts = value.timestamp()
    floor_ts = math.floor(ts / step_seconds) * step_seconds
    return datetime.datetime.fromtimestamp(floor_ts, tz=value.tzinfo)


def _round_datetime_to_step(value: datetime.datetime, step_minutes: int) -> datetime.datetime:
    if step_minutes <= 0:
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    step_seconds = step_minutes * 60
    ts = value.timestamp()
    round_ts = round(ts / step_seconds) * step_seconds
    return datetime.datetime.fromtimestamp(round_ts, tz=value.tzinfo)


def _align_next_run_to_step(
    *,
    target_dt: datetime.datetime,
    now_utc: datetime.datetime,
    min_delay_minutes: float,
    max_delay_minutes: float,
    step_minutes: int,
) -> tuple[datetime.datetime, float, Optional[str]]:
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=datetime.timezone.utc)
    base_delay = max(0.0, (target_dt - now_utc).total_seconds() / 60.0)
    max_limit = max_delay_minutes if max_delay_minutes > 0 else None

    def in_bounds(delay: float) -> bool:
        if delay < max(0.0, float(min_delay_minutes)) - 1e-9:
            return False
        if max_limit is not None and delay > float(max_limit) + 1e-9:
            return False
        return True

    candidates: list[tuple[str, datetime.datetime, float]] = []

    round_dt = _round_datetime_to_step(target_dt, step_minutes)
    round_delay = max(0.0, (round_dt - now_utc).total_seconds() / 60.0)
    if round_dt != target_dt and in_bounds(round_delay):
        candidates.append(("round", round_dt, round_delay))

    ceil_dt = _ceil_datetime_to_step(target_dt, step_minutes)
    ceil_delay = max(0.0, (ceil_dt - now_utc).total_seconds() / 60.0)
    if ceil_dt != target_dt and in_bounds(ceil_delay):
        candidates.append(("ceil", ceil_dt, ceil_delay))

    floor_dt = _floor_datetime_to_step(target_dt, step_minutes)
    floor_delay = max(0.0, (floor_dt - now_utc).total_seconds() / 60.0)
    if floor_dt != target_dt and in_bounds(floor_delay):
        candidates.append(("floor", floor_dt, floor_delay))

    if candidates:
        best = min(candidates, key=lambda item: abs((item[1] - target_dt).total_seconds()))
        label, dt_value, delay_value = best
        return dt_value, delay_value, label

    return target_dt, base_delay, None


def _format_schedule_overview() -> str:
    status = _read_runtime_status()
    lines = []
    if status:
        lines.append(f"Текущий статус: {status.get('status', 'unknown')}")
        next_local = status.get("next_run_local")
        next_utc = status.get("next_run_utc")
        if next_local:
            lines.append(f"Следующий запуск: {next_local}")
        elif next_utc:
            lines.append(f"Следующий запуск (UTC): {next_utc}")
        else:
            lines.append("Следующий запуск: не запланирован.")
        if status.get("next_run_minutes") is not None:
            lines.append(f"Оставшееся время (мин): {status['next_run_minutes']}")
    else:
        lines.append("Статус цикла недоступен.")
    with _SCHEDULE_OVERRIDE_LOCK:
        override = dict(_SCHEDULE_OVERRIDE) if _SCHEDULE_OVERRIDE else None
    if override:
        target_dt = override.get("target")
        delay = override.get("delay")
        origin = override.get("note") or "ручное"
        lines.append(
            "Ручное расписание: "
            f"{_format_local_dt(target_dt)} (~{delay:.1f} мин), источник: {origin}"
        )
    else:
        lines.append("Ручное расписание не активно.")
    lines.append("")
    lines.append("Команды: /schedule now, /schedule in 30, /schedule at 23:15, /schedule cancel")
    return "\n".join(lines)


def _parse_minutes_argument(token: str) -> Optional[float]:
    pattern = re.fullmatch(r"\s*(\d+(?:[\.,]\d+)?)([a-zA-Z]*)\s*", token)
    if not pattern:
        return None
    value = float(pattern.group(1).replace(",", "."))
    unit = pattern.group(2).lower()
    if unit in ("", "m", "min", "mins", "minute", "minutes"):
        return value
    if unit in ("h", "hr", "hrs", "hour", "hours"):
        return value * 60.0
    return None


def _parse_schedule_datetime(expression: str) -> Optional[datetime.datetime]:
    expr = expression.strip()
    if not expr:
        return None
    if expr.lower() in {"now", "сейчас"}:
        return datetime.datetime.now(datetime.timezone.utc)
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", expr):
        local_tz = _current_local_tz() or datetime.datetime.now().astimezone().tzinfo
        now_local = datetime.datetime.now(local_tz)
        parts = expr.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) > 2 else 0
        candidate = now_local.replace(hour=hour, minute=minute, second=second, microsecond=0)
        if candidate <= now_local:
            candidate += datetime.timedelta(days=1)
        return candidate.astimezone(datetime.timezone.utc)
    cleaned = expr.replace("T", " ").replace("Z", "+00:00")
    try:
        parsed = datetime.datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        local_tz = _current_local_tz() or datetime.datetime.now().astimezone().tzinfo
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(datetime.timezone.utc)


def _set_manual_schedule(
    *,
    delay_minutes: Optional[float],
    target_dt: Optional[datetime.datetime],
    note: str,
) -> str:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if target_dt is not None:
        target = target_dt
        if target.tzinfo is None:
            target = target.replace(tzinfo=datetime.timezone.utc)
        delta = (target - now_utc).total_seconds() / 60.0
        delay = max(0.0, delta)
    elif delay_minutes is not None:
        delay = max(0.0, float(delay_minutes))
        target = now_utc + datetime.timedelta(minutes=delay)
    else:
        delay = 0.0
        target = now_utc
    with _SCHEDULE_OVERRIDE_LOCK:
        global _SCHEDULE_OVERRIDE
        _SCHEDULE_OVERRIDE = {
            "delay": delay,
            "target": target,
            "note": note,
            "set_at": now_utc,
        }
    _SCHEDULE_EVENT.set()
    _write_runtime_status(delay, target, "scheduled")
    if delay <= 0.01:
        return "⏱ Следующая сессия будет запущена немедленно."
    return f"⏱ Следующая сессия запланирована через {delay:.1f} мин ({_format_local_dt(target)})."


def _clear_manual_schedule() -> str:
    with _SCHEDULE_OVERRIDE_LOCK:
        global _SCHEDULE_OVERRIDE
        had_override = _SCHEDULE_OVERRIDE is not None
        _SCHEDULE_OVERRIDE = None
    _SCHEDULE_EVENT.set()
    _write_runtime_status(None, None, "running")
    if had_override:
        return "⏱ Ручное расписание отменено, возвращаемся к автоматическому режиму."
    return "⏱ Ручное расписание не активно."


def _consume_schedule_override(default_delay: Optional[float]) -> tuple[float, Optional[datetime.datetime], bool]:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    with _SCHEDULE_OVERRIDE_LOCK:
        global _SCHEDULE_OVERRIDE
        override = _SCHEDULE_OVERRIDE
        if override:
            _SCHEDULE_OVERRIDE = None
    if override:
        target = override.get("target")
        if isinstance(target, datetime.datetime):
            target_dt = target if target.tzinfo else target.replace(tzinfo=datetime.timezone.utc)
            delay_minutes = max(0.0, (target_dt - now_utc).total_seconds() / 60.0)
        else:
            try:
                delay_minutes = max(0.0, float(override.get("delay") or 0.0))
            except (TypeError, ValueError):
                delay_minutes = max(0.0, float(default_delay or DEFAULT_NEXT_RUN_MINUTES))
            target_dt = now_utc + datetime.timedelta(minutes=delay_minutes)
        return delay_minutes, target_dt, True

    status = _read_runtime_status()
    next_utc = status.get("next_run_utc") if isinstance(status, dict) else None
    if isinstance(next_utc, str) and next_utc.strip():
        iso_candidate = next_utc.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.datetime.fromisoformat(iso_candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            parsed = parsed.astimezone(datetime.timezone.utc)
            delay_minutes = max(0.0, (parsed - now_utc).total_seconds() / 60.0)
            if delay_minutes > 0:
                return delay_minutes, parsed, False
        except Exception:
            pass

    base_delay = default_delay if (default_delay is not None and default_delay > 0) else DEFAULT_NEXT_RUN_MINUTES
    target_dt = now_utc + datetime.timedelta(minutes=base_delay)
    return base_delay, target_dt, False


def _persist_env_values(updates: dict[str, str | None]) -> bool:
    if not updates:
        return False
    path = SCRIPT_DIR / ".env"
    try:
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        existing_lines = []
    except Exception:
        existing_lines = []
    seen: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _, _ = line.partition("=")
        key = key.strip()
        if key in updates:
            value = updates[key]
            if value is None:
                new_lines.append(f"{key}=")
            else:
                new_lines.append(f"{key}={value}")
            seen.add(key)
        else:
            new_lines.append(line)
    for key, value in updates.items():
        if key in seen:
            continue
        if value is None:
            new_lines.append(f"{key}=")
        else:
            new_lines.append(f"{key}={value}")
    try:
        path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


def _format_token_usage_message() -> str:
    hard_stop = AI_HARD_STOP_BUDGET or 0
    secondary = AI_SECONDARY_BUDGET_START or 0
    lines = [
        f"Использовано токенов: {AI_TOKEN_USAGE_TOTAL}",
        f"Лимит цикла: {AI_TOKEN_BUDGET_CYCLE}",
        f"Порог дешёвой модели: {secondary if secondary else 'отключён'}",
        f"Жёсткий стоп: {hard_stop if hard_stop else 'отключён'}",
    ]
    if AI_TOKEN_USAGE_BY_MODEL:
        lines.append("Статистика по моделям:")
        for model, stats in sorted(AI_TOKEN_USAGE_BY_MODEL.items()):
            prompt = stats.get("prompt", 0)
            completion = stats.get("completion", 0)
            total = stats.get("total", prompt + completion)
            lines.append(f"- {model}: {total} (prompt {prompt}, completion {completion})")
    return "\n".join(lines)


def _handle_schedule_command(args: list[str]) -> str:
    if not args:
        return _format_schedule_overview()
    first = args[0].lower()
    if first in {"cancel", "clear", "reset", "stop"}:
        return _clear_manual_schedule()
    if first in {"now", "run", "start"}:
        return _set_manual_schedule(delay_minutes=0.0, target_dt=None, note="manual")
    if first in {"in", "через"} and len(args) > 1:
        minutes = _parse_minutes_argument(args[1])
        if minutes is None:
            return "⏱ Не удалось разобрать интервал. Пример: /schedule in 45"
        return _set_manual_schedule(delay_minutes=minutes, target_dt=None, note="manual")
    if first in {"at", "в"} and len(args) > 1:
        target = _parse_schedule_datetime(" ".join(args[1:]))
        if target is None:
            return "⏱ Не удалось разобрать время запуска. Пример: /schedule at 23:15"
        return _set_manual_schedule(delay_minutes=None, target_dt=target, note="manual")
    minutes = _parse_minutes_argument(args[0])
    if minutes is not None:
        return _set_manual_schedule(delay_minutes=minutes, target_dt=None, note="manual")
    target = _parse_schedule_datetime(" ".join(args))
    if target is not None:
        return _set_manual_schedule(delay_minutes=None, target_dt=target, note="manual")
    return (
        "⏱ Использование: /schedule 30 (в минутах), "
        "/schedule at 23:15, /schedule 2025-01-01 12:00, "
        "/schedule cancel"
    )


def _handle_ai_command(args: list[str]) -> str:
    if not args:
        return "ℹ️ ℹ️ℹ️ℹ️ℹ️? подкоманду. Пример: /ai payload [universe|trade]"
    sub = args[0].lower()
    if sub == "payload":
        context_hint = args[1] if len(args) > 1 else None
        return _format_ai_payload(context_hint)
    return "ℹ️ Неизвестная подкоманда /ai. Доступно: payload"



def handle_telegram_command(chat_id: int, text: str, *, thread_id: Optional[int] = None, user_id: Optional[int] = None) -> None:
    if not text:
        return
    command_thread = TELEGRAM_COMMAND_THREAD_ID
    response_thread = thread_id if thread_id is not None else command_thread
    parts = text.strip().split()
    if not parts:
        return
    command = parts[0].lstrip("/")
    if "@" in command:
        command = command.split("@", 1)[0]
    command = command.lower()
    args = parts[1:]
    if command == "start":
        reply = _build_help_message()
        send_tg(
            reply,
            chat_id_override=chat_id,
            thread_id=response_thread,
            no_log_forward=True,
            reply_markup=_build_start_keyboard(),
        )
        return
    if command == "help":
        _ensure_commands_help_messages()
        reply = _build_help_message()
    elif command == "status":
        reply = _format_status_message(live=True)
    elif command == "positions":
        reply = _format_positions_message(live=True)
    elif command == "risk":
        reply = _format_risk_message()
    elif command in {"logs", "logtail", "log"}:
        reply = _handle_logs_command(args)
    elif command == "logmode":
        reply = _handle_logmode_command(args)
    elif command in {"schedule", "next"}:
        reply = _handle_schedule_command(args)
    elif command in {"tokens", "token"}:
        reply = _handle_tokens_command(args)
    elif command in {"bybitkey", "bybit"}:
        dm_context = user_id is not None and chat_id == user_id
        if not (is_bot_owner(user_id, USER_ID) or dm_context):
            reply = "🚫 Команда /bybitkey доступна только владельцу процесса или в личном чате после /adduser."
        else:
            reply = _handle_bybit_key_command(args)
    elif command == "adduser":
        reply = _handle_add_user_command(args, user_id=user_id, origin_chat=chat_id, origin_thread=response_thread)
        if reply is None:
            return
    elif command == "config":
        reply = _handle_config_command(args, user_id=user_id, bot_id=USER_ID)
        if reply is None:
            return
    elif command == "sandbox":
        reply = _handle_sandbox_command(args, user_id=user_id)
        if reply is None:
            return
    elif command == "version":
        reply = f"Версия {BOT_VERSION}\n{BOT_CHANGELOG}"
        _tmp = _handle_version_command(args, user_id=user_id, chat_id=chat_id, thread_id=response_thread)
        if _tmp is None:
            return
        reply = _tmp
    elif command == "ai":
        reply = _handle_ai_command(args)
    else:
        reply = "Неизвестная команда. Используйте /help."
    send_tg(
        reply,
        chat_id_override=chat_id,
        thread_id=response_thread,
        no_log_forward=True,
    )
    # Append per-user log for DM replies
    try:
        if user_id is not None and chat_id == user_id:
            _append_user_log(user_id, f"OUT: {reply}")
    except Exception:
        pass
    # Forward replies from a direct message (private chat) to the main owner chat
    try:
        if user_id is not None and chat_id == user_id and MAIN_OWNER_CHAT_ID is not None and MAIN_OWNER_CHAT_ID != chat_id:
            fwd_text = f"[DM from {user_id}] Command: {text.strip()}\nReply:\n{reply}"
            log(f"[DEBUG] Forwarding DM reply to owner {MAIN_OWNER_CHAT_ID}: {fwd_text[:200]}", Fore.LIGHTBLACK_EX)
            send_tg(fwd_text, chat_id_override=MAIN_OWNER_CHAT_ID)
    except Exception as exc:  # keep the command reply stable even if forwarding fails
        log(f"⚠️ Failed to forward DM reply to owner: {exc}", Fore.YELLOW)


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
    release_thread = TELEGRAM_RELEASE_THREAD_ID if TELEGRAM_RELEASE_THREAD_ID is not None else TG_TOPIC_ID
    message_id = send_tg(message_text, disable_web_page_preview=True, thread_id=release_thread)
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


def _load_commands_help_state() -> dict[str, Any]:
    if not COMMANDS_HELP_STATE_FILE:
        return {}
    try:
        raw = COMMANDS_HELP_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_commands_help_state(state: dict[str, Any]) -> None:
    if not COMMANDS_HELP_STATE_FILE:
        return
    try:
        COMMANDS_HELP_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _build_command_help_sections() -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for entry in COMMANDS_HELP_SECTIONS:
        title = entry.get("title") or ""
        lines = entry.get("lines") or []
        body = "\n".join(f"- {line}" for line in lines)
        text = f"{title}\n{body}"
        sections.append({"key": entry.get("key") or title.lower(), "text": text})
    return sections


def _ensure_commands_help_messages() -> None:
    if not TG_TOKEN or not TG_CHAT:
        return
    thread_id = TELEGRAM_COMMAND_THREAD_ID if TELEGRAM_COMMAND_THREAD_ID is not None else TG_TOPIC_ID
    if thread_id is None:
        return
    sections = _build_command_help_sections()
    if not sections:
        return
    state = _load_commands_help_state()
    stored = state.get("messages") or {}
    updated: dict[str, dict[str, Any]] = {}
    for section in sections:
        key = str(section["key"])
        text = section["text"]
        signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = stored.get(key) if isinstance(stored, dict) else None
        reuse = entry and entry.get("signature") == signature
        message_id = entry.get("message_id") if reuse else None
        if not message_id:
            message_id = send_tg(
                text,
                thread_id=thread_id,
                no_log_forward=True,
                disable_notification=True,
            )
        if message_id:
            _pin_tg_message(TG_CHAT, message_id)
            updated[key] = {"message_id": message_id, "signature": signature}
    if updated:
        payload = {
            "messages": updated,
            "chat_id": TG_CHAT,
            "thread_id": thread_id,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    _save_commands_help_state(payload)


def _find_release_scripts(limit: int = 10) -> list[tuple[str, Path]]:
    """Scan repo root for release scripts like 2025.11.06*.py and return sorted list.
    Returns list of tuples (version_name, path), newest first by filename.
    """
    items: list[tuple[str, Path]] = []
    try:
        candidates = list(REPO_ROOT.glob("20*.py"))
    except Exception:
        candidates = []
    pattern = re.compile(r"^20\d{2}\.\d{2}\.\d{2}(\..+)?\.py$")
    for p in candidates:
        name = p.name
        if pattern.match(name):
            version_name = name[:-3]
            items.append((version_name, p))
    items.sort(key=lambda x: x[0], reverse=True)
    if limit > 0:
        items = items[:limit]
    return items


def _handle_version_command(args: list[str], *, user_id: Optional[int], chat_id: int, thread_id: Optional[int]) -> Optional[str]:
    """Implements /version list|set|latest with inline buttons and auto-restart."""
    sub = (args[0].lower() if args else "list")
    header = f"Версия {BOT_VERSION}"
    if BOT_CHANGELOG:
        header += f"\n{BOT_CHANGELOG}"

    def _send_with_keyboard(versions: list[tuple[str, Path]]) -> None:
        lines = [header, "", "Доступные релизы:"]
        for ver, _ in versions:
            lines.append(f"- {ver}")
        text = "\n".join(lines)
        buttons: list[list[dict[str, str]]] = []
        buttons.append([{"text": "Latest HEAD", "callback_data": "/version latest"}])
        row: list[dict[str, str]] = []
        for ver, _ in versions:
            row.append({"text": ver, "callback_data": f"/version set {ver}"})
            if len(row) >= 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        reply_markup = {"inline_keyboard": buttons}
        send_tg(text, chat_id_override=chat_id, thread_id=thread_id, no_log_forward=True, reply_markup=reply_markup)

    if sub in {"list", "ls"}:
        versions = _find_release_scripts(limit=10)
        _send_with_keyboard(versions)
        return None

    if sub == "latest":
        if not is_bot_owner(user_id, USER_ID):
            return "Только владелец бота может переключать версию."
        target_path = _get_bot_config_path(USER_ID, REPO_ROOT)
        updated = _persist_env_file(target_path, {"TARGET_VERSION": None})
        if not updated:
            return "Не удалось обновить TARGET_VERSION."
        reason = "[RESTART] /version latest -> switching to HEAD"
        _restart_with_latest_code(reason)
        return "Переключаюсь на HEAD и перезапускаюсь…"

    if sub == "set":
        if len(args) < 2:
            return "Использование: /version set <YYYY.MM.DD[.N]>"
        if not is_bot_owner(user_id, USER_ID):
            return "Только владелец бота может переключать версию."
        ver = args[1].strip()
        target_path = _get_bot_config_path(USER_ID, REPO_ROOT)
        updated = _persist_env_file(target_path, {"TARGET_VERSION": ver})
        if not updated:
            return "Не удалось обновить TARGET_VERSION."
        reason = f"[RESTART] /version set {ver}"
        _restart_with_latest_code(reason)
        return f"Переключаюсь на {ver} и перезапускаюсь…"

    return "Использование: /version list | /version latest | /version set <YYYY.MM.DD[.N]>"
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
        err_msg = f"ℹ️ Не удалось перезапустить процесс автоматически: {exc}"
        log(err_msg, Fore.RED)
        send_tg(err_msg)
        raise


def save_json_line(path, data):
    try:
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        ai_names = {
            Path(AI_LOG_FILE).name if isinstance(AI_LOG_FILE, str) else str(AI_LOG_FILE),
            Path(AI_ARCHIVE_FILE).name if isinstance(AI_ARCHIVE_FILE, str) else str(AI_ARCHIVE_FILE),
            Path(AI_REQUESTS_LOG).name if isinstance(AI_REQUESTS_LOG, str) else str(AI_REQUESTS_LOG),
        }
        if p.name in ai_names:
            _maybe_rotate_file(p, AI_LOG_MAX_BYTES, AI_LOG_BACKUPS)
        entry = dict(data)
        entry.setdefault("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
        db_logger.log_ai_decision(entry)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
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
    global DYNAMIC_SYMBOL_ALIASES
    snapshots: list[dict[str, Any]] = []

    def _log_raw_positions(payload: list[dict], settle_hint: Optional[str] = None) -> None:
        try:
            entries: list[str] = []
            total = len(payload or [])
            for pos in payload or []:
                symbol_raw = pos.get("symbol")
                amount_raw = safe_float(
                    pos.get("contracts")
                    or pos.get("positionAmt")
                    or pos.get("size")
                    or pos.get("amount")
                )
                display_amount = (
                    f"{amount_raw:.4f}"
                    if amount_raw is not None and math.isfinite(amount_raw)
                    else "0"
                )
                if symbol_raw:
                    if len(entries) < 20:
                        entries.append(f"{symbol_raw}:{display_amount}")
            label = f" ({settle_hint})" if settle_hint else ""
            if entries:
                log(
                    f"[DEBUG] fetch_positions raw snapshot{label} ({total} total): "
                    + ", ".join(entries),
                    Fore.LIGHTBLACK_EX,
                )
        except Exception:
            pass

    try:
        positions = exchange.fetch_positions()
        snapshots.extend(positions or [])
        _log_raw_positions(positions, None)
    except Exception as e:
        log(f"⚠️ Не удалось получить список позиций: {e}", Fore.YELLOW)
        return {}, None
    def _extract_settle_coin(position: dict[str, Any]) -> str | None:
        settle_coin = position.get("settle")
        if not settle_coin:
            info_payload = position.get("info")
            if isinstance(info_payload, dict):
                settle_coin = info_payload.get("settleCoin") or info_payload.get("settle")
        if not settle_coin:
            symbol_text = str(position.get("symbol") or "")
            if ":" in symbol_text:
                settle_coin = symbol_text.split(":", 1)[1]
        if settle_coin:
            settle_coin = str(settle_coin).strip().upper()
        return settle_coin or None

    extra_settles = [settle for settle in EXTRA_POSITION_SETTLES if settle and settle.upper() != "USDT"]
    extra_settle_summary: list[str] = []
    if extra_settles:
        def _fetch_extra_positions(settle_hint: str) -> list[dict[str, Any]] | None:
            candidates = [
                {"settleCoin": settle_hint, "type": "linear"},
                {"settleCoin": settle_hint},
                {"category": "linear", "settle": settle_hint},
                {"settle": settle_hint},
            ]
            last_exc: Exception | None = None
            for params in candidates:
                try:
                    # Pass extra parameters via the params argument so ccxt
                    # can apply them correctly (e.g. settleCoin/settle).
                    result = exchange.fetch_positions(None, dict(params))
                except Exception as exc:
                    last_exc = exc
                    continue
                if result:
                    return result
            if last_exc:
                raise last_exc
            return None

        for settle in extra_settles:
            extra_positions = None
            try:
                extra_positions = _fetch_extra_positions(settle)
                filtered_positions: list[dict[str, Any]] = []
                for pos in extra_positions or []:
                    settle_coin = _extract_settle_coin(pos)
                    if settle_coin is not None and settle_coin != settle.upper():
                        continue
                    filtered_positions.append(pos)
                if filtered_positions:
                    snapshots.extend(filtered_positions)
                _log_raw_positions(filtered_positions, settle)
                summary_items: list[str] = []
                for pos in filtered_positions:
                    symbol_raw = pos.get("symbol") or settle
                    amt_val = safe_float(
                        pos.get("contracts")
                        or pos.get("positionAmt")
                        or pos.get("size")
                        or pos.get("amount")
                    )
                    if amt_val is not None and math.isfinite(amt_val):
                        summary_items.append(f"{symbol_raw}:{amt_val:.4f}")
                    if len(summary_items) >= 5:
                        break
                if summary_items:
                    extra_settle_summary.append(f"{settle}: " + ", ".join(summary_items))
            except Exception as exc_extra:
                log(
                    f"[WARN] Failed to fetch positions for settle {settle}: {exc_extra}",
                    Fore.YELLOW,
                )
    if extra_settle_summary and LOG_EXTRA_SETTLE_POSITIONS:
        log(
            "[DEBUG] extra-settle positions: " + " | ".join(extra_settle_summary),
            Fore.LIGHTBLACK_EX,
        )
    count = 0
    simplified = {}
    symbols_filter = set(symbols_filter) if symbols_filter else None
    for pos in snapshots or []:
        source_symbol = pos.get("symbol")
        canonical_symbol = _resolve_symbol_alias(source_symbol) or source_symbol
        if not canonical_symbol:
            continue
        if symbols_filter and canonical_symbol not in symbols_filter and source_symbol not in symbols_filter:
            continue
        simp = simplify_position(pos)
        if not simp:
            continue
        # Avoid double-counting the same canonical symbol across multiple settles
        if canonical_symbol not in simplified:
            simplified[canonical_symbol] = simp
            if source_symbol and source_symbol != canonical_symbol:
                DYNAMIC_SYMBOL_ALIASES[source_symbol] = canonical_symbol
            count += 1
        else:
            # Prefer keeping the first snapshot as the primary view; we only update aliases once.
            pass
    return simplified, count


def fetch_spot_position_symbols(exchange, *, min_value_usd: float = 0.05) -> dict[str, dict[str, Any]]:
    """
    Return a mapping of spot symbols that currently have a balance >= min_value_usd.
    Each entry contains a synthetic position payload so downstream logic can treat
    spot holdings as active exposure.
    """
    positions: dict[str, dict[str, Any]] = {}
    try:
        balances = exchange.fetch_balance({"type": "spot"})
    except Exception as exc:
        log(f"[WARN] Failed to fetch spot balances: {exc}", Fore.YELLOW)
        return positions
    totals = balances.get("total")
    assets: dict[str, float] = {}
    if isinstance(totals, dict):
        for asset, value in totals.items():
            qty = safe_float(value)
            if qty is None:
                continue
            assets[asset.upper()] = max(qty, assets.get(asset.upper(), 0.0))
    skip_keys = {"info", "free", "used", "total", "timestamp", "datetime"}
    for asset, payload in balances.items():
        if asset in skip_keys:
            continue
        asset_name = str(asset or "").upper()
        if not asset_name:
            continue
        if isinstance(payload, dict):
            amount_val = payload.get("total") if payload.get("total") is not None else payload.get("free")
        else:
            amount_val = payload
        qty = safe_float(amount_val)
        if qty is None:
            continue
        assets[asset_name] = max(qty, assets.get(asset_name, 0.0))
    stable_skip = {"USD", "BUSD", "USDT", "USDC"}
    stable_skip.update(quote.upper() for quote in RECOGNIZED_STABLE_QUOTES)
    markets: dict[str, Any] = getattr(exchange, "markets", {}) or {}
    if not markets:
        try:
            exchange.load_markets()
            markets = getattr(exchange, "markets", {}) or {}
        except Exception:
            markets = {}
    market_symbols = set(markets.keys() or [])
    if not market_symbols:
        market_symbols = set(getattr(exchange, "symbols", []) or [])
    price_cache = getattr(exchange, "_spot_price_cache", None)
    if not isinstance(price_cache, dict):
        price_cache = {}
        setattr(exchange, "_spot_price_cache", price_cache)

    def candidate_symbols(asset_name: str) -> list[str]:
        base = asset_name.upper()
        candidates: list[str] = []
        mapped = TICKER_TO_SYMBOL.get(base)
        if mapped:
            candidates.append(mapped)
        candidates.extend(
            [
                f"{base}/USDT",
                f"{base}/USDT:SPOT",
                f"{base}/USDT:USDT",
            ]
        )
        alias_target = SYMBOL_ALIASES.get(f"{base}/USDT:USDT")
        if alias_target:
            candidates.append(alias_target)
        seen: set[str] = set()
        ordered: list[str] = []
        for entry in candidates:
            cleaned = entry.replace("//", "/")
            if cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered

    def lookup_price(symbol_name: str) -> Optional[float]:
        price = price_cache.get(symbol_name)
        if price:
            return price
        market = markets.get(symbol_name) or {}
        info = market.get("info") if isinstance(market.get("info"), dict) else {}
        for source in (market, info):
            if not isinstance(source, dict):
                continue
            for key in ("last", "close", "markPrice", "indexPrice", "lastPrice", "avgPrice"):
                price = safe_float(source.get(key))
                if price:
                    break
            if price:
                break
        if not price and symbol_name in market_symbols:
            try:
                ticker = exchange.fetch_ticker(symbol_name)
            except Exception:
                ticker = None
            if isinstance(ticker, dict):
                for key in ("last", "close", "info"):
                    value = ticker.get(key)
                    if key == "info" and isinstance(value, dict):
                        for nested in ("lastPrice", "close", "avgPrice"):
                            price = safe_float(value.get(nested))
                            if price:
                                break
                    else:
                        price = safe_float(value)
                    if price:
                        break
        if price:
            price_cache[symbol_name] = price
        return price

    detected: list[str] = []
    for asset_name, qty in assets.items():
        if asset_name in stable_skip:
            continue
        amount = safe_float(qty)
        if amount is None or amount <= 0:
            continue
        resolved_symbol = None
        unit_price = None
        for cand in candidate_symbols(asset_name):
            if market_symbols and cand not in market_symbols:
                continue
            unit_price = lookup_price(cand)
            if unit_price:
                resolved_symbol = cand
                break
        if not resolved_symbol or not unit_price:
            continue
        usd_value = unit_price * amount
        if usd_value < min_value_usd:
            continue
        payload = {
            "symbol": resolved_symbol,
            "amount": amount,
            "side": "long",
            "entryPrice": unit_price,
            "category": "spot",
            "info": {
                "source": "spot_balance",
                "usd_value": usd_value,
            },
        }
        positions[resolved_symbol] = payload
        detected.append(f"{resolved_symbol}≈{usd_value:.2f} USD")
    if detected:
        log(f"[DEBUG] Spot holdings detected: {', '.join(sorted(detected))}", Fore.LIGHTBLACK_EX)
    return positions


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
    except TypeError as e:
        try:
            exchange.cancel_order(order_id, resolved_symbol, {})
            return True, None
        except Exception as e2:
            tb = traceback.format_exc()
            log(f"[DEBUG] cancel_order_by_id TypeError fallback failed for {order_id} {symbol}: {e} | {e2}", Fore.YELLOW)
            log(tb, Fore.LIGHTBLACK_EX)
            # Treat already-cancelled / not-exists as benign
            text = str(e2)
            if "OrderNotFound" in text or 'retCode":110001' in text or 'order not exists' in text.lower():
                return True, None
            return False, f"{e} | fallback: {e2}"
    except Exception as e:
        text = str(e)
        # Consider Bybit 110001 and ccxt OrderNotFound benign during cleanups
        if "OrderNotFound" in text or 'retCode":110001' in text or 'order not exists' in text.lower():
            log(f"[INFO] cancel noop for {order_id} {symbol}: already cancelled/filled", Fore.LIGHTBLACK_EX)
            return True, None
        tb = traceback.format_exc()
        log(f"[DEBUG] cancel_order_by_id failed for {order_id} {symbol}: {e}", Fore.YELLOW)
        log(tb, Fore.LIGHTBLACK_EX)
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
        order_intent_raw = (
            order.get("intent")
            or order.get("tag")
            or order.get("note")
            or order.get("comment")
            or ""
        )
        order_intent = str(order_intent_raw).lower()
        if _is_truthy_flag(order.get("ladder")) or any(
            keyword in order_intent for keyword in ("ladder", "scale", "step", "stagger", "laddering")
        ):
            continue
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


def _has_active_limit_at_price(open_orders, side: str, price: float, tolerance: float = 5e-4) -> tuple[bool, float]:
    if price is None or not math.isfinite(price) or price <= 0:
        return False, 0.0
    side_lower = (side or '').lower()
    if not side_lower:
        return False, 0.0
    for existing in open_orders or []:
        if not isinstance(existing, dict):
            continue
        try:
            existing_side = (existing.get('side') or '').lower()
            existing_type = (existing.get('type') or '').lower()
        except AttributeError:
            continue
        if existing_side != side_lower:
            continue
        if existing_type != 'limit':
            continue
        if existing.get('reduceOnly') in (True, 'true', '1', 1):
            continue
        existing_price = safe_float(existing.get('price'))
        if not (existing_price and math.isfinite(existing_price) and existing_price > 0):
            continue
        if abs(existing_price - price) / price <= tolerance:
            amount_val = safe_float(existing.get("amount") or existing.get("qty") or existing.get("quantity"))
            if amount_val is None or not math.isfinite(amount_val) or amount_val < 0:
                amount_val = 0.0
            return True, float(amount_val)
    return False, 0.0

def get_position_idx(side: str | None) -> int | None:
    if ACTIVE_HEDGE_MODE:
        side_lower = (side or "").lower()
        if side_lower == "buy":
            return 1  # long position in hedged mode
        if side_lower == "sell":
            return 2  # short position in hedged mode
        return None
    # One-way mode uses index 0; keep submitting zero until hedge mode is enabled.
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
        # Use derivative symbol format for Bybit (e.g., BTC/USDT:USDT)
        resolved_symbol = _resolve_symbol_alias(symbol) or symbol
        params = {}
        try:
            market_info = exchange.market(resolved_symbol)
        except Exception:
            market_info = None
        category = _infer_market_category(resolved_symbol, market_info)
        if category:
            params["category"] = category
        if hasattr(exchange, "fetchFundingRate"):
            fr = exchange.fetchFundingRate(resolved_symbol, params)
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
            history = exchange.fetchFundingRateHistory(resolved_symbol, limit=1, params=params)
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
        resolved_symbol = _resolve_symbol_alias(symbol) or symbol
        params = {}
        try:
            market_info = exchange.market(resolved_symbol)
        except Exception:
            market_info = None
        category = _infer_market_category(resolved_symbol, market_info)
        if category == "spot":
            return []
        if category:
            params["category"] = category
        if hasattr(exchange, "fetchOpenInterestHistory"):
            hist = exchange.fetchOpenInterestHistory(resolved_symbol, timeframe="1h", limit=72, params=params)
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


# --- Helpers for mixed spot/derivatives trading ---
def _sanitize_order_params_for_category(params: dict[str, Any] | None, category: str | None) -> dict[str, Any]:
    p: dict[str, Any] = dict(params or {})
    cat = (category or "").lower()
    if cat == "spot":
        for key in (
            "reduceOnly",
            "positionIdx",
            "triggerPrice",
            "triggerDirection",
            "closeOnTrigger",
            "stopLoss",
            "takeProfit",
            "tpSlMode",
            "trailingStop",
            "stopLossPrice",
            "takeProfitPrice",
        ):
            p.pop(key, None)
        p["category"] = "spot"
    else:
        p.setdefault("category", "linear")
    return p


def _spot_funds_sufficient(exchange, symbol: str, side: str, amount: float | None, price: float | None) -> tuple[bool, str | None]:
    try:
        balance = exchange.fetch_balance()
    except Exception as exc:
        return False, f"fetch_balance failed: {exc}"
    side_lower = (side or "").lower()
    base = str(symbol).split("/")[0].split(":")[0]
    quote = str(symbol).split("/")[1].split(":")[0] if "/" in str(symbol) else "USDT"
    if side_lower == "sell":
        bucket = balance.get(base) or {}
        free = bucket.get("free") if isinstance(bucket, dict) else None
        try:
            free_val = float(free)
        except Exception:
            free_val = None
        if free_val is None or amount is None or free_val + 1e-12 < float(amount):
            return False, f"spot sell {base}: insufficient free balance (have {free_val}, need {amount})"
    elif side_lower == "buy":
        if amount is None or price is None:
            return False, "spot buy requires amount and price"
        need = float(amount) * float(price) * 1.001
        bucket = balance.get(quote) or {}
        free = bucket.get("free") if isinstance(bucket, dict) else None
        try:
            free_val = float(free)
        except Exception:
            free_val = None
        if free_val is None or free_val + 1e-8 < need:
            return False, f"spot buy {base}: insufficient {quote} (have {free_val}, need ~{need:.2f})"
    return True, None


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


def _merge_news_payloads(base_symbol: str, payloads: Sequence[dict[str, Any]], limit: int) -> dict[str, Any]:
    base_upper = (base_symbol or "").upper()
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    for payload in payloads:
        items = payload.get("items") or []
        source_name = (payload.get("source") or "news").lower()
        source_counts[source_name] = source_counts.get(source_name, 0) + len(items)
        for item in items:
            title = (item.get("title") or "").strip()
            url = item.get("url") or item.get("link")
            key = f"{title}|{url}"
            if key in seen:
                continue
            seen.add(key)
            combined.append(
                {
                    "title": title,
                    "body": item.get("body") or item.get("summary") or "",
                    "url": url,
                    "source": item.get("source") or source_name,
                    "published_at": item.get("published_at") or to_iso_utc(item.get("published_at_raw") or item.get("published_on")),
                }
            )
    combined.sort(key=lambda entry: entry.get("published_at") or "", reverse=True)
    if source_counts:
        counts_text = ", ".join(f"{src}={cnt}" for src, cnt in source_counts.items())
        summary = f"news counts: {counts_text}"
    else:
        summary = "news unavailable"
    return {
        "summary": summary or "Mixed news",
        "items": combined[:limit],
        "asset": base_upper,
        "source": "hybrid",
        "counts": source_counts,
    }

def get_news(symbol):
    base = symbol.split("/")[0].split(":")[0].upper()
    limit = max(1, AI_INITIAL_NEWS_LIMIT or NEWS_ITEMS_LIMIT)
    provider_source = AI_INITIAL_NEWS_PROVIDER or NEWS_PROVIDER
    normalized_provider = (provider_source or "hybrid").strip().lower()
    payloads: list[dict[str, Any]] = []
    def _fetch_cc():
        return get_news_from_cryptocompare(base, limit)
    def _fetch_rss():
        return get_news_from_rss(base, limit)

    if normalized_provider in NEWS_PROVIDER_ALIAS_HYBRID:
        payloads.extend([_fetch_cc(), _fetch_rss()])
    elif normalized_provider in NEWS_PROVIDER_ALIAS_CC:
        cc_payload = _fetch_cc()
        payloads.append(cc_payload)
        if not cc_payload.get("items"):
            payloads.append(_fetch_rss())
    elif normalized_provider in NEWS_PROVIDER_ALIAS_RSS:
        payloads.append(_fetch_rss())
    else:
        log(f"[WARN] Unknown NEWS_PROVIDER '{NEWS_PROVIDER}', using hybrid feeds.", Fore.YELLOW)
        payloads.extend([_fetch_cc(), _fetch_rss()])
    payloads = [payload for payload in payloads if payload]
    if not payloads:
        return {"summary": "News unavailable", "items": [], "asset": base, "source": normalized_provider or "hybrid"}
    if len(payloads) == 1:
        return payloads[0]
    return _merge_news_payloads(base, payloads, limit)

# --- Подключение к бирже ---
def init_exchange():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        stored_key, stored_secret = _load_bybit_credentials()
        if stored_key and stored_secret:
            api_key = api_key or stored_key
            api_secret = api_secret or stored_secret
    if not api_key or not api_secret:
        raise RuntimeError(
            "Требуется указать BYBIT_API_KEY и BYBIT_API_SECRET. "
            "Используйте /bybitkey <apiKey> <apiSecret> или добавьте их в users/<id>/secrets.env."
        )
    exchange = ccxt.bybit({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "recvWindow": 5000,
            "hedgeMode": HEDGE_MODE
        }
    })
    exchange.options["recvWindow"] = 5000
    return exchange

# --- Patch: robust init_exchange with adjustable recvWindow and time sync ---
def _init_exchange_enhanced() -> Any:
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        stored_key, stored_secret = _load_bybit_credentials()
        if stored_key and stored_secret:
            api_key = api_key or stored_key
            api_secret = api_secret or stored_secret
    if not api_key or not api_secret:
        raise RuntimeError(
            "ℹ️ℹ️ℹ️ ℹ️ℹ️? BYBIT_API_KEY ? BYBIT_API_SECRET. "
            "ℹ️ℹ️ℹ️ /bybitkey <apiKey> <apiSecret> ℹ️? ℹ️ℹ️ℹ️? ℹ️ ? users/<id>/secrets.env."
        )

    def _env_int(name: str, default: int) -> int:
        try:
            v = int(str(os.getenv(name, str(default))).strip())
        except Exception:
            v = default
        return max(1000, min(60000, v))

    recv_window_ms = _env_int("BYBIT_RECV_WINDOW_MS", 15000)

    exchange = ccxt.bybit({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "recvWindow": recv_window_ms,
            "adjustForTimeDifference": True,
            "hedgeMode": HEDGE_MODE,
        },
    })
    try:
        exchange.options["recvWindow"] = recv_window_ms
        exchange.options["adjustForTimeDifference"] = True
    except Exception:
        pass
    try:
        if getattr(exchange, "has", {}).get("fetchTime"):
            exchange.load_time_difference()
    except Exception:
        pass
    try:
        _enable_exchange_logging(exchange)
    except Exception as exc:
        log(f"[WARN] Не удалось включить расширенное логирование ордеров: {exc}", Fore.YELLOW)
    return exchange

# Replace default init_exchange with enhanced version
try:
    init_exchange = _init_exchange_enhanced  # type: ignore
except Exception:
    # Should not happen, but prefer not to crash at import time
    pass



def _load_equity_history() -> list[dict]:
    try:
        raw = EQUITY_HISTORY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except Exception:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    result: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ts = entry.get("timestamp")
        equity_val = entry.get("equity")
        try:
            equity_float = float(equity_val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(equity_float):
            continue
        if not isinstance(ts, str) or not ts:
            continue
        realized_val = entry.get("realized")
        try:
            realized_float = float(realized_val)
        except (TypeError, ValueError):
            realized_float = None
        payload = {"timestamp": ts, "equity": equity_float}
        if realized_float is not None and math.isfinite(realized_float):
            payload["realized"] = realized_float
        result.append(payload)
    return result


def _save_equity_history(history: list[dict]) -> None:
    try:
        EQUITY_HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _parse_timestamp(value: str | None) -> Optional[datetime.datetime]:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _compute_drawdown_pct(history: list[dict], window_hours: float) -> float:
    if not history:
        return 0.0
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=window_hours)
    window: list[tuple[datetime.datetime, float]] = []
    for entry in history:
        ts = _parse_timestamp(entry.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        equity_val = entry.get("equity")
        try:
            equity_float = float(equity_val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(equity_float) or equity_float <= 0:
            continue
        window.append((ts, equity_float))
    if len(window) < 2:
        return 0.0
    window.sort(key=lambda item: item[0])
    peak = window[0][1]
    max_drawdown = 0.0
    for _, equity_val in window:
        if equity_val > peak:
            peak = equity_val
        if peak <= 0:
            continue
        drawdown = (equity_val - peak) / peak * 100.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def _maybe_apply_drawdown_controls(
    history: list[dict],
    base_margin_utilization: float,
    base_auto_margin_ratio: float,
) -> tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    if not DRAWDOWN_CONTROL_ENABLED or not history:
        return None, None, None, None
    drawdown_pct = _compute_drawdown_pct(history, DRAWDOWN_WINDOW_HOURS)
    if drawdown_pct >= 0:
        return None, None, None, None
    severity = abs(drawdown_pct)
    selected_rule: tuple[float, float, float, float] | None = None
    for threshold, risk_mult, margin_cap, auto_ratio in DRAWDOWN_RULES:
        if severity >= threshold:
            selected_rule = (threshold, risk_mult, margin_cap, auto_ratio)
            break
    if not selected_rule:
        return None, None, None, None
    threshold, risk_mult, margin_cap, auto_ratio = selected_rule
    margin_override = min(base_margin_utilization, margin_cap)
    auto_ratio_override = min(base_auto_margin_ratio, auto_ratio)
    risk_override = max(MIN_DYNAMIC_RISK_PCT, RISK_PCT * risk_mult)
    message = (
        f"[RISK] Drawdown {drawdown_pct:.1f}% >= {threshold:.0f}% -> "
        f"risk {risk_override:.4f}, margin_util {margin_override:.2f}, ladder ratio {auto_ratio_override:.2f}"
    )
    return message, margin_override, auto_ratio_override, risk_override


def _update_equity_history(
    history: list[dict],
    timestamp: datetime.datetime,
    equity: float,
    realized: float | None = None,
) -> None:
    if not math.isfinite(equity):
        return
    ts = timestamp.astimezone(datetime.timezone.utc)
    entry: dict[str, float | str] = {"timestamp": ts.isoformat(), "equity": float(equity)}
    if realized is not None and math.isfinite(realized):
        entry["realized"] = float(realized)
    history.append(entry)
    cutoff = ts - datetime.timedelta(hours=max(PNL_LOOKBACK_HOURS * 6, 72))
    pruned: list[dict] = []
    for entry in history:
        entry_ts_raw = entry.get("timestamp")
        if not isinstance(entry_ts_raw, str):
            continue
        try:
            entry_ts = datetime.datetime.fromisoformat(entry_ts_raw)
        except ValueError:
            continue
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=datetime.timezone.utc)
        if entry_ts >= cutoff:
            payload = {
                "timestamp": entry_ts.isoformat(),
                "equity": float(entry.get("equity", 0.0)),
            }
            realized_val = entry.get("realized")
            try:
                realized_float = float(realized_val)
            except (TypeError, ValueError):
                realized_float = None
            if realized_float is not None and math.isfinite(realized_float):
                payload["realized"] = realized_float
            pruned.append(payload)
    history[:] = pruned
    _save_equity_history(history)


def _extract_realized_pnl(balance: dict | None) -> float | None:
    if not isinstance(balance, dict):
        return None

    def to_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    priority_map = {
        "cumrealisedpnl": 1,
        "cumrealizedpnl": 1,
        "totalrealisedpnl": 2,
        "totalrealizedpnl": 2,
        "totalrpl": 3,
        "sessionrpl": 4,
        "realisedpnl": 5,
        "realizedpnl": 5,
        "rpl": 6,
    }
    best: tuple[int, float] | None = None

    def visit(obj):
        nonlocal best
        if isinstance(obj, dict):
            for key, value in obj.items():
                lower_key = str(key).lower()
                if isinstance(value, (dict, list, tuple, set)):
                    visit(value)
                    continue
                if lower_key in priority_map or any(hint in lower_key for hint in priority_map):
                    numeric = to_float(value)
                    if numeric is None:
                        continue
                    priority = priority_map.get(lower_key)
                    if priority is None:
                        for hint, rank in priority_map.items():
                            if hint in lower_key:
                                priority = rank
                                break
                    if priority is None:
                        priority = 50
                    if best is None or priority < best[0]:
                        best = (priority, numeric)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                visit(item)

    visit(balance)
    return best[1] if best else None


def _extract_base_asset(symbol: str | None) -> str | None:
    if not symbol:
        return None
    sym = str(symbol).strip()
    if not sym:
        return None
    base = sym.split('/', 1)[0]
    if ':' in base:
        base = base.split(':', 1)[0]
    base_norm = re.sub(r"[^A-Z0-9]", "", base.upper())
    if not base_norm:
        return None
    base_norm = re.sub(r"\d+$", "", base_norm) or base_norm
    return base_norm


def _build_base_exposure_map(positions_map) -> dict[str, int]:
    exposures: defaultdict[str, int] = defaultdict(int)
    for sym, payload in (positions_map or {}).items():
        if not payload:
            continue
        amount_val = safe_float((payload or {}).get("amount") or (payload or {}).get("contracts"))
        if amount_val is None or not math.isfinite(amount_val) or abs(amount_val) <= 1e-8:
            continue
        base_key = _extract_base_asset(sym)
        if base_key:
            exposures[base_key] += 1
    return dict(exposures)


def _extract_closed_pnl_from_payload(payload: Any) -> float | None:
    """
    Try to find a numeric closed/realized PnL field inside an order/trade payload.
    Ignores unrealized/floating keys.
    """
    def gather(obj: Any, depth: int = 0) -> list[tuple[float, int]]:
        if depth > 4:
            return []
        candidates_local: list[tuple[float, int]] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                lower_key = str(key).lower()
                if isinstance(value, (dict, list, tuple, set)):
                    candidates_local.extend(gather(value, depth + 1))
                    continue
                if "pnl" not in lower_key and "rpl" not in lower_key:
                    continue
                if any(term in lower_key for term in ("unreal", "floating", "u_pnl")):
                    continue
                if not any(
                    term in lower_key
                    for term in (
                        "closed",
                        "close",
                        "realised",
                        "realized",
                        "settled",
                        "settle",
                        "order",
                        "realisedpnl",
                        "realizedpnl",
                        "realise",
                    )
                ) and lower_key not in {"pnl", "rpl"}:
                    continue
                numeric = safe_float(value)
                if numeric is None:
                    continue
                score = 1
                if "realised" in lower_key or "realized" in lower_key or "settle" in lower_key:
                    score += 3
                elif "closed" in lower_key or "close" in lower_key:
                    score += 2
                elif lower_key in {"pnl", "rpl"}:
                    score += 1
                candidates_local.append((numeric, score))
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                candidates_local.extend(gather(item, depth + 1))
        return candidates_local

    candidates = gather(payload, 0)
    if not candidates:
        return None
    best_val, _ = max(candidates, key=lambda item: (item[1], abs(item[0])))
    return best_val


def _collect_bybit_closed_pnl_v5(
    exchange,
    target_symbols: set[str],
    start_ms: int,
    end_ms: int | None,
    limit_per_symbol: int,
) -> tuple[float, int, list[str], list[dict[str, Any]]]:
    method = getattr(exchange, "privateGetV5PositionClosedPnl", None)
    if not callable(method):
        return 0.0, 0, [], []
    warnings: list[str] = []
    total_pnl = 0.0
    details: list[dict[str, Any]] = []
    markets = getattr(exchange, "markets", {}) or {}
    normalized_targets: set[str] = set()
    for sym in target_symbols:
        if not sym:
            continue
        sym_upper = sym.upper()
        normalized_targets.add(sym_upper)
        normalized_targets.add(sym_upper.replace(":USDT", ""))
        normalized_targets.add(sym_upper.replace("/", ""))
        normalized_targets.add(sym_upper.replace(":", ""))
    target_symbols = normalized_targets
    categories: set[str] = set()
    for sym in target_symbols:
        market = markets.get(sym)
        category = _infer_market_category(sym, market)
        if category and category != "spot":
            categories.add(category)
    if not categories:
        warnings.append("[PnL] bybit closed-pnl skipped: only spot symbols provided")
        return 0.0, 0, warnings, details
    max_rows = max(limit_per_symbol * max(1, len(target_symbols) or 1), limit_per_symbol)
    for category in categories:
        cursor = None
        fetched = 0
        while fetched < max_rows:
            params = {
                "category": category,
                "startTime": start_ms,
                "limit": min(200, max_rows - fetched),
            }
            if end_ms is not None:
                params["endTime"] = end_ms
            if cursor:
                params["cursor"] = cursor
            try:
                response = method(params)
            except Exception as exc:
                warnings.append(f"[PnL] bybit closed-pnl ({category}) failed: {exc}")
                break
            result = response.get("result") if isinstance(response, dict) else None
            rows = result.get("list") if isinstance(result, dict) else None
            if not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol_id = row.get("symbol")
                symbol = None
                if symbol_id:
                    try:
                        symbol = exchange.safe_symbol(symbol_id, None)
                    except Exception:
                        symbol = symbol_id
                symbol_key = (symbol or symbol_id or "").upper()
                symbol_compact = symbol_key.replace(":USDT", "").replace("/", "").replace(":", "")
                if target_symbols and symbol_key not in target_symbols and symbol_compact not in target_symbols:
                    continue
                pnl_val = safe_float(row.get("closedPnl"))
                if pnl_val is None:
                    continue
                ts_raw = row.get("updatedTime") or row.get("createdTime") or row.get("closedTime")
                ts_iso = None
                if ts_raw is not None:
                    try:
                        ts_int = int(ts_raw)
                        ts_iso = datetime.datetime.fromtimestamp(ts_int / 1000, tz=datetime.timezone.utc).isoformat()
                    except (TypeError, ValueError, OverflowError):
                        ts_iso = None
                amount_val = safe_float(row.get("closedSize") or row.get("qty") or row.get("size"))
                price_val = safe_float(row.get("avgExitPrice") or row.get("avgPrice") or row.get("exitPrice"))
                side = (row.get("side") or "").upper()
                detail = {
                    "id": str(row.get("execId") or row.get("orderId") or row.get("positionIdx") or f"{symbol_id}:{ts_raw}"),
                    "symbol": symbol or symbol_id,
                    "side": side,
                    "amount": amount_val,
                    "price": price_val,
                    "pnl": float(pnl_val),
                    "timestamp": ts_iso,
                }
                details.append(detail)
                total_pnl += float(pnl_val)
                fetched += 1
                if fetched >= max_rows:
                    break
            cursor = result.get("nextPageCursor") if isinstance(result, dict) else None
            if not cursor:
                break
    if details:
        details.sort(key=lambda item: item.get("timestamp") or "")
    return total_pnl, len(details), warnings, details


def _collect_recent_closed_pnl(
    exchange,
    symbols: Sequence[str],
    window_start: datetime.datetime,
    window_end: datetime.datetime | None = None,
    *,
    limit_per_symbol: int = 200,
) -> tuple[float | None, int, list[str], list[dict[str, Any]]]:
    """
    Sum realized PnL from closed orders (and, if needed, trades) within [window_start, window_end].
    Returns (total_pnl, fill_count, warnings, details).
    """
    if not symbols:
        return None, 0, [], []
    warnings: list[str] = []
    since_ms = int(window_start.timestamp() * 1000)
    until_ms = int(window_end.timestamp() * 1000) if window_end else None
    markets_available = set(getattr(exchange, "markets", {}) or {})
    exchange_id = getattr(exchange, "id", "") or ""
    bybit_cache: dict[str, dict[str, str]] = {}
    total_pnl = 0.0
    fill_count = 0
    seen_order_ids: set[str] = set()
    order_details: list[dict[str, Any]] = []
    normalized_symbols = []
    for sym in symbols:
        if not sym:
            continue
        if markets_available and sym not in markets_available:
            continue
        normalized_symbols.append(sym)
    symbol_filter = {sym for sym in symbols if sym}
    if exchange_id.lower() == "bybit":
        bybit_total, bybit_count, bybit_warnings, bybit_details = _collect_bybit_closed_pnl_v5(
            exchange,
            symbol_filter,
            since_ms,
            until_ms,
            limit_per_symbol,
        )
        warnings.extend(bybit_warnings)
        if bybit_count:
            return bybit_total, bybit_count, warnings, bybit_details
    # Prefer closed orders first
    spot_trade_pool: dict[str, list[dict[str, Any]]] = {}
    for symbol in normalized_symbols:
        params = {}
        if exchange_id.lower() == "bybit":
            params = bybit_cache.get(symbol, {})
            if not params:
                market_info = None
                try:
                    market_info = exchange.market(symbol)
                except Exception:
                    market_info = None
                category = _infer_market_category(symbol, market_info)
                settle_coin = ""
                if isinstance(market_info, dict):
                    settle_coin = str(
                        market_info.get("settle")
                        or ((market_info.get("info") or {}).get("settleCoin") if isinstance(market_info.get("info"), dict) else "")
                        or ((market_info.get("info") or {}).get("settle") if isinstance(market_info.get("info"), dict) else "")
                    )
                params = {}
                if category:
                    params["category"] = category
                if settle_coin:
                    params["settleCoin"] = settle_coin.upper()
                bybit_cache[symbol] = params
        try:
            orders = exchange.fetch_closed_orders(symbol, since=since_ms, limit=limit_per_symbol, params=params or {})
        except Exception as exc:
            warnings.append(f"[PnL] fetch_closed_orders failed for {symbol}: {exc}")
            continue
        if not orders:
            continue
        for order in orders:
            order_ts = order.get("timestamp") or order.get("lastTradeTimestamp")
            if order_ts is not None:
                try:
                    order_ts = int(order_ts)
                except (TypeError, ValueError):
                    order_ts = None
            if order_ts is not None and order_ts < since_ms:
                continue
            if until_ms is not None and order_ts is not None and order_ts > until_ms:
                continue
            order_id = order.get("id")
            if order_id and order_id in seen_order_ids:
                continue
            pnl_val = _extract_closed_pnl_from_payload(order)
            if pnl_val is None:
                pnl_val = _extract_closed_pnl_from_payload(order.get("info"))
            if pnl_val is None:
                continue
            if order_id:
                seen_order_ids.add(order_id)
            total_pnl += pnl_val
            fill_count += 1
            detail = {
                "id": order_id or "",
                "symbol": symbol,
                "side": str(order.get("side") or "").upper() or "?",
                "amount": safe_float(
                    order.get("amount")
                    or order.get("filled")
                    or order.get("qty")
                    or order.get("reduce_only_size")
                ),
                "price": safe_float(order.get("average") or order.get("price")),
                "pnl": float(pnl_val),
                "timestamp": (
                    datetime.datetime.fromtimestamp(order_ts / 1000, tz=datetime.timezone.utc).isoformat()
                    if order_ts is not None
                    else None
                ),
            }
            order_details.append(detail)
    if fill_count > 0:
        return total_pnl, fill_count, warnings, order_details
    # Fallback to trade history if orders did not expose realised PnL
    seen_trade_ids: set[str] = set()
    for symbol in normalized_symbols:
        params = {}
        if exchange_id.lower() == "bybit":
            params = bybit_cache.get(symbol, {})
            if not params:
                market_info = None
                try:
                    market_info = exchange.market(symbol)
                except Exception:
                    market_info = None
                category = _infer_market_category(symbol, market_info)
                settle_coin = ""
                if isinstance(market_info, dict):
                    settle_coin = str(
                        market_info.get("settle")
                        or ((market_info.get("info") or {}).get("settleCoin") if isinstance(market_info.get("info"), dict) else "")
                        or ((market_info.get("info") or {}).get("settle") if isinstance(market_info.get("info"), dict) else "")
                    )
                params = {}
                if category:
                    params["category"] = category
                if settle_coin:
                    params["settleCoin"] = settle_coin.upper()
                bybit_cache[symbol] = params
        try:
            trades = exchange.fetch_my_trades(symbol, since=since_ms, limit=limit_per_symbol, params=params or {})
        except Exception as exc:
            warnings.append(f"[PnL] fetch_my_trades failed for {symbol}: {exc}")
            continue
        if not trades:
            continue
        for trade in trades:
            trade_ts = trade.get("timestamp")
            if trade_ts is not None:
                try:
                    trade_ts = int(trade_ts)
                except (TypeError, ValueError):
                    trade_ts = None
            if trade_ts is not None and trade_ts < since_ms:
                continue
            if until_ms is not None and trade_ts is not None and trade_ts > until_ms:
                continue
            trade_id = trade.get("id")
            if trade_id and trade_id in seen_trade_ids:
                continue
            pnl_val = _extract_closed_pnl_from_payload(trade)
            if pnl_val is None:
                pnl_val = _extract_closed_pnl_from_payload(trade.get("info"))
            amount_val = safe_float(
                trade.get("amount")
                or trade.get("amountFilled")
                or trade.get("filled")
                or trade.get("contracts")
                or trade.get("qty")
            )
            price_val = safe_float(trade.get("price") or trade.get("average") or trade.get("cost"))
            fee_cost = None
            fee_currency = None
            fee = trade.get("fee")
            if isinstance(fee, dict):
                fee_cost = safe_float(fee.get("cost"))
                fee_currency = str(fee.get("currency") or "").upper()
            if pnl_val is None:
                category = params.get("category") if isinstance(params, dict) else None
                if not category:
                    market_info = None
                    try:
                        market_info = exchange.market(symbol)
                    except Exception:
                        market_info = None
                    category = _infer_market_category(symbol, market_info)
                if category == "spot" and amount_val is not None and price_val is not None:
                    spot_trade_pool.setdefault(symbol, []).append(
                        {
                            "id": trade_id or "",
                            "timestamp": trade_ts,
                            "side": trade.get("side"),
                            "amount": amount_val,
                            "price": price_val,
                            "fee_cost": fee_cost,
                            "fee_currency": fee_currency,
                        }
                    )
                continue
            if trade_id:
                seen_trade_ids.add(trade_id)
            total_pnl += pnl_val
            if fee_cost is not None and fee_currency in {"USDT", "USDC", "USD"}:
                total_pnl -= fee_cost
            fill_count += 1
            detail = {
                "id": trade_id or "",
                "symbol": symbol,
                "side": str(trade.get("side") or "").upper() or "?",
                "amount": amount_val,
                "price": price_val,
                "pnl": float(pnl_val),
                "timestamp": (
                    datetime.datetime.fromtimestamp(trade_ts / 1000, tz=datetime.timezone.utc).isoformat()
                    if trade_ts is not None
                    else None
                ),
            }
            order_details.append(detail)
    for symbol, trades in spot_trade_pool.items():
        pnl_spot, count_spot, details_spot = _compute_spot_realized_pnl(symbol, trades)
        if count_spot:
            total_pnl += pnl_spot
            fill_count += count_spot
            order_details.extend(details_spot)
    if fill_count > 0:
        return total_pnl, fill_count, warnings, order_details
    return None, 0, warnings, order_details


def _sum_unrealized_pnl(positions_map: dict[str, Any] | None) -> tuple[float, int]:
    total = 0.0
    count = 0
    if not isinstance(positions_map, dict):
        return 0.0, 0
    for payload in positions_map.values():
        if not isinstance(payload, dict):
            continue
        raw_value = payload.get("unrealizedPnl")
        if raw_value is None:
            raw_info = payload.get("raw")
            if isinstance(raw_info, dict):
                raw_value = raw_info.get("unrealisedPnl") or raw_info.get("unrealizedPnl")
        value = safe_float(raw_value)
        if value is None or not math.isfinite(value):
            continue
        total += value
        if abs(value) > 1e-8:
            count += 1
    return total, count


def _adjust_dynamic_risk_from_pnl(
    pnl_value: float | None,
    basis_label: str,
    sample_count: int,
) -> str | None:
    global CURRENT_RISK_PCT
    if not DYNAMIC_RISK_ENABLED or pnl_value is None or not math.isfinite(pnl_value):
        return None
    current = CURRENT_RISK_PCT if CURRENT_RISK_PCT > 0 and math.isfinite(CURRENT_RISK_PCT) else RISK_PCT
    target_multiplier = 1.0
    magnitude = pnl_value
    if basis_label == "closed":
        if magnitude <= -6:
            target_multiplier = 0.4
        elif magnitude <= -3:
            target_multiplier = 0.6
        elif magnitude <= -1:
            target_multiplier = 0.8
        elif magnitude >= 7:
            target_multiplier = 1.45
        elif magnitude >= 3:
            target_multiplier = 1.25
        elif magnitude >= 1:
            target_multiplier = 1.1
    else:
        if magnitude <= -6:
            target_multiplier = 0.45
        elif magnitude <= -3:
            target_multiplier = 0.65
        elif magnitude <= -1:
            target_multiplier = 0.85
        elif magnitude >= 6:
            target_multiplier = 1.35
        elif magnitude >= 2.5:
            target_multiplier = 1.15
    desired_pct = min(
        MAX_DYNAMIC_RISK_PCT,
        max(MIN_DYNAMIC_RISK_PCT, RISK_PCT * target_multiplier),
    )
    if not math.isfinite(desired_pct):
        desired_pct = RISK_PCT
    smoothing = 0.6 if desired_pct < current else 0.45
    blended = current + (desired_pct - current) * smoothing
    blended = min(MAX_DYNAMIC_RISK_PCT, max(MIN_DYNAMIC_RISK_PCT, blended))
    threshold = max(1e-5, RISK_PCT * 0.03)
    if abs(blended - current) < threshold:
        return None
    CURRENT_RISK_PCT = blended
    delta = (CURRENT_RISK_PCT / RISK_PCT) if RISK_PCT > 0 else 1.0
    label = basis_label or "equity"
    return (
        f"[RISK] {label} pnl {pnl_value:+.2f} -> risk {CURRENT_RISK_PCT:.4f} "
        f"(base {RISK_PCT:.4f}, x{delta:.2f})"
    )


def _resolve_log_path(filename: str) -> Path | None:
    if not filename:
        return None
    candidates = [
        Path(filename),
        STATE_DIR / filename,
        SCRIPT_DIR / filename,
        SCRIPT_DIR / "assets" / filename,
        REPO_ROOT / "assets" / filename,
    ]
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            if resolved.exists():
                return resolved
        except Exception:
            continue
    return None


def _generate_ai_decision_summary(log_path: Path, output_path: Path, window_hours: float = 24.0) -> None:
    try:
        raw_text = log_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return
    except Exception as exc:
        log(f"[WARN] Failed to read AI decision log {log_path}: {exc}", Fore.YELLOW)
        return
    if not raw_text.strip():
        return
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now_utc - datetime.timedelta(hours=window_hours)
    actions_counter = Counter()
    symbols_counter = Counter()
    skip_reasons_counter = Counter()
    side_counter = Counter()
    recent_entries: deque[dict[str, Any]] = deque(maxlen=8)
    total_considered = 0
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        ts_raw = payload.get("timestamp") or payload.get("time")
        if not ts_raw:
            continue
        try:
            ts_obj = datetime.datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if ts_obj.tzinfo is None:
            ts_obj = ts_obj.replace(tzinfo=datetime.timezone.utc)
        ts_utc = ts_obj.astimezone(datetime.timezone.utc)
        if window_hours and ts_utc < cutoff:
            continue
        decision = payload.get("decision") or {}
        action = str(decision.get("action") or payload.get("action") or "").strip().lower()
        symbol = str(payload.get("symbol") or decision.get("symbol") or "").strip()
        total_considered += 1
        if action:
            actions_counter[action] += 1
        if symbol:
            symbols_counter[symbol] += 1
        if action == "skip":
            reason_text = str(decision.get("reason") or "").strip()
            if reason_text:
                skip_reasons_counter[reason_text] += 1
        side_val = str(decision.get("side") or "").strip().lower()
        if side_val:
            side_counter[side_val] += 1
        recent_entries.append({
            "timestamp": ts_obj.isoformat(),
            "symbol": symbol,
            "action": action or None,
            "side": side_val or None,
            "reason": decision.get("reason"),
            "confidence": decision.get("confidence"),
        })
    if total_considered == 0:
        return
    def _top(counter: Counter, limit: int = 5):
        return [{"item": key, "count": value} for key, value in counter.most_common(limit)]
    summary_payload = {
        "generated_at": now_utc.isoformat(),
        "window_hours": window_hours,
        "entries": total_considered,
        "actions": _top(actions_counter, 10),
        "symbols": _top(symbols_counter, 10),
        "skip_reasons": _top(skip_reasons_counter, 10),
        "sides": _top(side_counter, 5),
        "recent": list(recent_entries),
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as exc:
        log(f"[WARN] Failed to write AI decision summary {output_path}: {exc}", Fore.YELLOW)


def _compute_recent_pnl(
    history: list[dict],
    timestamp: datetime.datetime,
    current_equity: float,
    current_realized: float | None = None,
    hours: float = PNL_LOOKBACK_HOURS,
) -> tuple[float | None, float | None, str | None]:
    if not history or not math.isfinite(current_equity) or hours <= 0:
        return None, None, None
    ts_now = timestamp.astimezone(datetime.timezone.utc)
    cutoff = ts_now - datetime.timedelta(hours=hours)
    candidates: list[tuple[datetime.datetime, float, float | None]] = []
    for entry in history:
        entry_ts_raw = entry.get("timestamp")
        if not isinstance(entry_ts_raw, str):
            continue
        try:
            entry_ts = datetime.datetime.fromisoformat(entry_ts_raw)
        except ValueError:
            continue
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=datetime.timezone.utc)
        if entry_ts > ts_now:
            continue
        try:
            equity_val = float(entry.get("equity"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(equity_val):
            continue
        realized_val = entry.get("realized")
        try:
            realized_val = float(realized_val)
        except (TypeError, ValueError):
            realized_val = None
        if realized_val is not None and not math.isfinite(realized_val):
            realized_val = None
        candidates.append((entry_ts, equity_val, realized_val))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda item: item[0])
    reference_equity = None
    reference_realized = None
    for entry_ts, equity_val, realized_val in candidates:
        if entry_ts >= cutoff:
            reference_equity = equity_val
            reference_realized = realized_val
            break
    if reference_equity is None:
        reference_equity = candidates[0][1]
        reference_realized = candidates[0][2]
    if reference_equity is None or not math.isfinite(reference_equity):
        return None, None, None
    if (
        current_realized is not None
        and reference_realized is not None
        and math.isfinite(current_realized)
        and math.isfinite(reference_realized)
    ):
        pnl_value = current_realized - reference_realized
        return pnl_value, reference_realized, "realized"
    pnl_value = current_equity - reference_equity
    return pnl_value, reference_equity, "equity"

def fetch_df(exchange, symbol, tf):
    resolved_symbol = _resolve_symbol_alias(symbol) or symbol

    def _symbol_candidates(base_symbol: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def push(value: str | None) -> None:
            if not value:
                return
            if value in seen:
                return
            seen.add(value)
            candidates.append(value)

        push(base_symbol)
        upper_symbol = (base_symbol or "").upper()
        if ":" in base_symbol:
            raw = base_symbol.split(":", 1)[0]
            push(raw)
            push(f"{raw}:SPOT")
            if raw.endswith("/USDT"):
                push(raw.replace("/USDT", "/USDT:USDT"))
        else:
            push(f"{upper_symbol}:USDT")
            push(f"{upper_symbol}:SPOT")
        return candidates

    last_exc: Exception | None = None
    for candidate in _symbol_candidates(resolved_symbol):
        try:
            ohlcv = exchange.fetch_ohlcv(candidate, timeframe=tf, limit=200)
        except Exception as exc:
            last_exc = exc
            continue
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Unable to fetch OHLCV for {symbol}")


def _detect_position_mode(exchange, fallback_hedge: bool) -> bool:
    try:
        positions = exchange.fetch_positions()
    except Exception:
        return fallback_hedge
    for payload in positions or []:
        if not isinstance(payload, dict):
            continue
        idx = payload.get("positionIdx")
        if idx is None:
            info = payload.get("info") or {}
            idx = info.get("positionIdx")
        idx_val = safe_int(idx)
        if idx_val in (1, 2):
            return True
    return False


def ensure_position_mode(exchange):
    desired = "hedged" if HEDGE_MODE else "oneway"
    actual_hedge: bool | None = None
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
                actual_hedge = HEDGE_MODE
                log(f"[INFO] Position mode set: {desired}", Fore.LIGHTBLACK_EX)
    except Exception as exc:
        code = get_bybit_retcode(exc)
        if code in {110025, 110024, 110028}:
            log(
                f"[INFO] Unable to switch position mode to {desired} (code {code}); using exchange-reported mode.",
                Fore.LIGHTBLACK_EX,
            )
            actual_hedge = _detect_position_mode(exchange, HEDGE_MODE)
        else:
            log(f"[WARN] Failed to set position mode ({desired}): {exc}", Fore.YELLOW)
            actual_hedge = _detect_position_mode(exchange, HEDGE_MODE)
    if actual_hedge is None:
        actual_hedge = _detect_position_mode(exchange, HEDGE_MODE)
    global ACTIVE_POSITION_MODE, ACTIVE_HEDGE_MODE, POSITION_MODE_MISMATCH_STATE
    previous_mode = ACTIVE_POSITION_MODE
    ACTIVE_HEDGE_MODE = bool(actual_hedge)
    ACTIVE_POSITION_MODE = "hedged" if ACTIVE_HEDGE_MODE else "oneway"
    if ACTIVE_POSITION_MODE != previous_mode:
        log(f"[INFO] Active position mode now {ACTIVE_POSITION_MODE}", Fore.LIGHTBLACK_EX)
    mismatch = ACTIVE_HEDGE_MODE != HEDGE_MODE
    if POSITION_MODE_MISMATCH_STATE != mismatch:
        POSITION_MODE_MISMATCH_STATE = mismatch
        if mismatch:
            log(
                f"[INFO] Exchange reports {ACTIVE_POSITION_MODE} mode; desired mode is {desired}. Will retry when eligible.",
                Fore.LIGHTBLACK_EX,
            )
        else:
            log("[INFO] Exchange position mode now matches the configured preference.", Fore.LIGHTBLACK_EX)


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


def _stable_wallet_for_symbol(balance: dict | None, symbol: str) -> dict[str, Any] | None:
    if not isinstance(balance, dict):
        return None
    symbol_upper = symbol.upper()
    for key, wallet in balance.items():
        if not isinstance(key, str):
            continue
        if not isinstance(wallet, dict):
            continue
        head = key.upper().split(":", 1)[0]
        if head == symbol_upper:
            return wallet
    return None


def _extract_wallet_totals(wallet: dict | None) -> tuple[float, float]:
    if not isinstance(wallet, dict):
        wallet = {}

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    total_val = to_float(wallet.get("total") or wallet.get("equity") or wallet.get("walletBalance"))
    free_val = to_float(wallet.get("free") or wallet.get("available") or wallet.get("availableBalance"))
    if free_val is None:
        used_val = to_float(wallet.get("used"))
        if used_val is not None and total_val is not None:
            free_val = total_val - used_val
    if total_val is None and free_val is not None:
        total_val = free_val
    if free_val is None and total_val is not None:
        free_val = total_val
    total_val = float(total_val) if total_val is not None else 0.0
    free_val = float(free_val) if free_val is not None else 0.0
    return total_val, free_val


def _stable_symbols_from_balance(balance: dict | None) -> list[str]:
    if not isinstance(balance, dict):
        return []
    summary = balance.get("_stable_summary")
    if isinstance(summary, dict):
        detected = [
            symbol
            for symbol in RECOGNIZED_STABLE_QUOTES
            if isinstance(summary.get(symbol), dict) and summary[symbol].get("present")
        ]
        if detected:
            return detected
    detected: list[str] = []
    for symbol in RECOGNIZED_STABLE_QUOTES:
        wallet = _stable_wallet_for_symbol(balance, symbol)
        if wallet is not None:
            detected.append(symbol)
    return detected


def _stable_currency_label(balance: dict | None) -> str:
    detected = _stable_symbols_from_balance(balance)
    if not detected:
        return RECOGNIZED_STABLE_QUOTES[0]
    return "/".join(detected)


def fetch_usdt_equity(exchange):
    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        log(f"⚠️ Не удалось получить баланс: {e}", Fore.YELLOW)
        return 0.0, 0.0, {}
    if not isinstance(balance, dict):
        return 0.0, 0.0, balance

    total_val = 0.0
    free_val = 0.0
    stable_summary: dict[str, dict[str, Any]] = {}
    stable_symbols: list[str] = []
    for symbol in RECOGNIZED_STABLE_QUOTES:
        wallet = _stable_wallet_for_symbol(balance, symbol)
        present = wallet is not None
        wallet_totals = _extract_wallet_totals(wallet)
        stable_summary[symbol] = {
            "total": wallet_totals[0],
            "free": wallet_totals[1],
            "present": present,
        }
        if present:
            stable_symbols.append(symbol)
        total_val += wallet_totals[0]
        free_val += wallet_totals[1]

    balance["_stable_summary"] = stable_summary
    balance["_stable_symbols"] = stable_symbols or [RECOGNIZED_STABLE_QUOTES[0]]
    balance["_stable_label"] = _stable_currency_label(balance)
    balance["_stable_equity_total"] = total_val
    balance["_stable_available_total"] = free_val
    if total_val < 0:
        total_val = 0.0
    if free_val < 0:
        free_val = 0.0
    realized_val = _extract_realized_pnl(balance)
    balance["_realizedPnl"] = realized_val
    return total_val, free_val, balance


def _parse_timestamp_any(value) -> Optional[datetime.datetime]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    if isinstance(value, (int, float)):
        val = int(value)
        if val > 1_000_000_000_000:
            return datetime.datetime.fromtimestamp(val / 1000, tz=datetime.timezone.utc)
        if val > 1_000_000_000:
            return datetime.datetime.fromtimestamp(val, tz=datetime.timezone.utc)
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        if trimmed.isdigit():
            return _parse_timestamp_any(int(trimmed))
        try:
            return datetime.datetime.fromisoformat(trimmed.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def fetch_unified_cash_flows(exchange, *, limit: int = 20) -> dict[str, Any]:
    """Fetch Unified-Funding transfers for the past 24h."""
    summary: dict[str, Any] = {
        "records": [],
        "totals": {"deposit": {}, "withdraw": {}},
        "warnings": [],
    }

    if not hasattr(exchange, "privateGetV5AssetTransferQueryInterTransferList"):
        summary["warnings"].append("[STATUS] Endpoint inter-transfer list недоступен; операции Unified-Funding не будут показаны.")
        return summary

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    since_utc = datetime.datetime.combine(
        now_utc.date(),
        datetime.time.min,
        tzinfo=datetime.timezone.utc,
    )
    since_ms = int(since_utc.timestamp() * 1000)
    until_ms = int(now_utc.timestamp() * 1000)

    def _normalize_account(value: str | None) -> str:
        val = (value or "").upper()
        replacements = {
            "UNIFIED_TRADING": "UNIFIED",
            "UNIFIEDTRADE": "UNIFIED",
            "UNIFIEDTRADEACCOUNT": "UNIFIED",
            "UNIFIED_ACCOUNT": "UNIFIED",
            "CONTRACT": "DERIVATIVES",
        }
        return replacements.get(val, val)

    def _record(kind: str, amount: float, coin: str, timestamp: Optional[datetime.datetime], status: str, raw: dict[str, Any]) -> None:
        entry = {
            "type": kind,
            "coin": coin,
            "amount": float(amount),
            "timestamp": timestamp,
            "status": status,
            "raw": raw,
        }
        summary["records"].append(entry)
        totals = summary["totals"].setdefault(kind, {})
        totals[coin] = totals.get(coin, 0.0) + float(amount)

    params_transfer = {
        "limit": limit,
        "startTime": since_ms,
        "endTime": until_ms,
    }
    try:
        resp = exchange.privateGetV5AssetTransferQueryInterTransferList(params_transfer)
        rows = ((((resp or {}).get("result") or {}).get("list")) or [])
    except Exception as exc:
        rows = []
        summary["warnings"].append(f"[STATUS] Не удалось получить внутренние переводы: {exc}")
    for row in rows:
        amount = safe_float(row.get("amount") or row.get("qty"))
        if amount is None or amount <= 0:
            continue
        coin = str(row.get("coin") or row.get("currency") or "USDT").upper()
        ts = _parse_timestamp_any(row.get("timestamp") or row.get("updatedTime") or row.get("createdTime"))
        if ts and ts < since_utc:
            continue
        from_acc = _normalize_account(row.get("fromAccountType") or row.get("fromAccount"))
        to_acc = _normalize_account(row.get("toAccountType") or row.get("toAccount"))
        accounts = {from_acc, to_acc}
        if accounts != {"UNIFIED", "FUNDING"}:
            continue
        status = str(row.get("status") or row.get("transferStatus") or "").upper()
        if to_acc == "UNIFIED":
            _record("deposit", float(amount), coin, ts, status, row)
        else:
            _record("withdraw", float(amount), coin, ts, status, row)

    summary["records"].sort(
        key=lambda item: item.get("timestamp") or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
        reverse=True,
    )
    return summary


def compute_order_amount(order, current_position):
    if not isinstance(order, dict):
        return None

    def _extract_number(keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = order.get(key)
            if value in (None, "", 0):
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and abs(candidate) > 0:
                return candidate
        return None

    amount_keys = ("amount", "qty", "quantity", "size", "contracts", "volume")
    amount_value = _extract_number(amount_keys)
    if amount_value is not None:
        return abs(amount_value)

    percent_keys = (
        "amountPercent",
        "amount_percent",
        "amount_pct",
        "percent",
        "sizePercent",
        "size_percent",
        "size_pct",
    )
    percent_value = _extract_number(percent_keys)
    if percent_value is not None and current_position:
        if percent_value <= 0:
            return None
        base_amount = safe_float(
            (current_position or {}).get("amount")
            or (current_position or {}).get("contracts")
            or (current_position or {}).get("size")
        )
        if base_amount is None or not math.isfinite(base_amount) or base_amount == 0:
            return None
        return abs(base_amount) * percent_value / 100.0
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


def get_trigger_direction_for_side(
    side: str,
    trigger_price: float | None = None,
    reference_price: float | None = None,
) -> str:
    """Return trigger direction flag understood by ccxt/bybit for a closing order."""
    trigger = safe_float(trigger_price)
    reference = safe_float(reference_price)
    if trigger is not None and reference is not None and math.isfinite(trigger) and math.isfinite(reference):
        return "above" if trigger >= reference else "below"
    side_lower = (side or "").lower()
    if side_lower in {"sell", "short"}:
        return "below"
    if side_lower in {"buy", "long"}:
        return "above"
    return "below"


def _resolve_ai_model_for_pairs(pair_count: Optional[int]) -> str:
    if AI_MODEL_CHEAP:
        if pair_count is not None and AI_MODEL_THRESHOLD and pair_count >= AI_MODEL_THRESHOLD:
            return AI_MODEL_CHEAP
        if AI_TOKEN_USAGE_TOTAL >= AI_SECONDARY_BUDGET_START:
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


def _json_excerpt(value: Any, limit: int = 1800) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            text = repr(value)
        text = text.strip()
    if len(text) <= limit:
        return text
    truncated = text[: limit - 3].rstrip()
    leftover = len(text) - len(truncated)
    return f"{truncated}... (+{leftover} chars)"


def _record_ai_exchange(
    context: str,
    *,
    model: str,
    request: Any = None,
    response: Any = None,
    error: Any = None,
    usage: Any = None,
    token_estimate: Optional[int] = None,
    label: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    key = (context or "default").strip().lower() or "default"
    entry: dict[str, Any] = {
        "key": key,
        "label": label or context or "default",
        "model": model,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if token_estimate is not None:
        entry["token_estimate"] = token_estimate
    if request is not None:
        entry["request_excerpt"] = _json_excerpt(request)
    if response is not None:
        entry["response_excerpt"] = _json_excerpt(response)
    if error is not None:
        entry["error"] = str(error)
    usage_dict: dict[str, Any] | None = None
    if usage:
        if isinstance(usage, dict):
            usage_dict = {
                "prompt": usage.get("prompt_tokens") or usage.get("prompt"),
                "completion": usage.get("completion_tokens") or usage.get("completion"),
                "total": usage.get("total_tokens") or usage.get("total"),
            }
        else:
            usage_dict = {
                "prompt": getattr(usage, "prompt_tokens", None),
                "completion": getattr(usage, "completion_tokens", None),
                "total": getattr(usage, "total_tokens", None),
            }
    if usage_dict:
        entry["usage"] = usage_dict
    if extra:
        entry["meta"] = extra
    AI_LAST_EXCHANGE[key] = entry


def _format_exchange_params_blob(payload: Any) -> str:
    if not payload:
        return "-"
    if isinstance(payload, (dict, list, tuple)):
        return _json_excerpt(payload, limit=320)
    return str(payload)


def _enable_exchange_logging(exchange: Any) -> Any:
    if not exchange or getattr(exchange, "_bybitbot_exchange_logging", False):
        return exchange
    try:
        original_create_order = exchange.create_order
    except AttributeError:
        original_create_order = None
    if callable(original_create_order):
        def logged_create_order(self, symbol, order_type, side, amount, price=None, params=None):
            param_text = _format_exchange_params_blob(params)
            log(f"[EX] create {symbol} {side}/{order_type} qty={amount} price={price or 'market'} params={param_text}", Fore.LIGHTBLACK_EX)
            start = time.time()
            try:
                result = original_create_order(symbol, order_type, side, amount, price, params)
                info = result if isinstance(result, dict) else {}
                order_id = info.get("id") or info.get("orderId") or info.get("clientOrderId")
                status = info.get("status") or info.get("state")
                filled = info.get("filled") or info.get("amount") or info.get("cumExecQty")
                duration = time.time() - start
                log(
                    f"[EX] done {symbol} id={order_id or '?'} status={status or '?'} filled={filled} ({duration:.2f}s)",
                    Fore.LIGHTBLACK_EX,
                )
                return result
            except Exception as exc:
                log(f"[EX] fail {symbol} {side}/{order_type}: {exc}", Fore.YELLOW)
                raise
        exchange.create_order = types.MethodType(logged_create_order, exchange)
    original_cancel_order = getattr(exchange, "cancel_order", None)
    if callable(original_cancel_order):
        def logged_cancel_order(self, order_id, symbol=None, params=None):
            param_text = _format_exchange_params_blob(params)
            safe_params = params if params else {}
            log(f"[EX] cancel {symbol or '?'} #{order_id} params={param_text}", Fore.LIGHTBLACK_EX)
            start = time.time()
            try:
                result = original_cancel_order(order_id, symbol, safe_params)
                duration = time.time() - start
                log(f"[EX] cancel ok {symbol or '?'} #{order_id} ({duration:.2f}s)", Fore.LIGHTBLACK_EX)
                return result
            except TypeError as exc:
                # Defensive: some exchange adapters raise TypeError on unexpected param shapes.
                # Retry with empty params to see if that helps, and log traceback for debugging.
                tb = traceback.format_exc()
                log(f"[EX] cancel TypeError {symbol or '?'} #{order_id}: {exc} — retrying with empty params", Fore.YELLOW)
                log(tb, Fore.LIGHTBLACK_EX)
                try:
                    result = original_cancel_order(order_id, symbol, {})
                    duration = time.time() - start
                    log(f"[EX] cancel ok (fallback) {symbol or '?'} #{order_id} ({duration:.2f}s)", Fore.LIGHTBLACK_EX)
                    return result
                except Exception as exc2:
                    tb2 = traceback.format_exc()
                    log(f"[EX] cancel fail (fallback) {symbol or '?'} #{order_id}: {exc2}", Fore.YELLOW)
                    log(tb2, Fore.LIGHTBLACK_EX)
                    raise
            except Exception as exc:
                text = str(exc)
                if "OrderNotFound" in text or 'retCode":110001' in text or 'order not exists' in text.lower():
                    duration = time.time() - start
                    log(f"[EX] cancel noop {symbol or '?'} #{order_id} ({duration:.2f}s) — already cancelled/filled", Fore.LIGHTBLACK_EX)
                    return None
                log(f"[EX] cancel fail {symbol or '?'} #{order_id}: {exc}", Fore.YELLOW)
                tb = traceback.format_exc()
                log(tb, Fore.LIGHTBLACK_EX)
                raise
        exchange.cancel_order = types.MethodType(logged_cancel_order, exchange)
    setattr(exchange, "_bybitbot_exchange_logging", True)
    return exchange


def _select_ai_exchange(context_hint: Optional[str]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    if not AI_LAST_EXCHANGE:
        return None, None
    hint = (context_hint or "").strip().lower()
    if hint:
        if hint in AI_LAST_EXCHANGE:
            return hint, AI_LAST_EXCHANGE[hint]
        for key, entry in AI_LAST_EXCHANGE.items():
            label = (entry.get("label") or "").lower()
            if hint in key or (label and hint in label):
                return key, entry
        return None, None
    latest_key = max(
        AI_LAST_EXCHANGE.keys(),
        key=lambda key: AI_LAST_EXCHANGE[key].get("captured_at") or "",
    )
    return latest_key, AI_LAST_EXCHANGE[latest_key]


def _format_ai_payload(context_hint: Optional[str] = None) -> str:
    key, entry = _select_ai_exchange(context_hint)
    if not entry:
        if AI_LAST_EXCHANGE:
            options = ", ".join(
                entry.get("label") or ctx
                for ctx, entry in sorted(AI_LAST_EXCHANGE.items())
            )
            return f"ℹ️ ������ AI payload '{context_hint}'. Доступно: {options}"
        return "ℹ️ Пока нет сохранённых AI-запросов — дождитесь следующего вызова модели."
    label = entry.get("label") or key or "payload"
    lines = [
        f"[AI] Последний запрос ({label})",
        f"- Время: {entry.get('captured_at') or 'n/a'}",
        f"- Модель: {entry.get('model') or 'n/a'}",
    ]
    if entry.get("token_estimate") is not None:
        lines.append(f"- Оценка токенов: {entry['token_estimate']}")
    usage = entry.get("usage")
    if usage:
        prompt = usage.get("prompt")
        completion = usage.get("completion")
        total = usage.get("total")
        usage_parts = []
        if prompt is not None:
            usage_parts.append(f"prompt {prompt}")
        if completion is not None:
            usage_parts.append(f"completion {completion}")
        if total is not None:
            usage_parts.append(f"total {total}")
        if usage_parts:
            lines.append(f"- Факт токенов: {', '.join(usage_parts)}")
    if entry.get("error"):
        lines.append(f"- Ошибка: {entry['error']}")
    if entry.get("meta"):
        for key_name, value in entry["meta"].items():
            lines.append(f"- {key_name}: {value}")
    request_excerpt = entry.get("request_excerpt")
    if request_excerpt:
        lines.append("")
        lines.append("---- Запрос ----")
        lines.append(request_excerpt)
    response_excerpt = entry.get("response_excerpt")
    if response_excerpt:
        lines.append("")
        lines.append("---- Ответ ----")
        lines.append(response_excerpt)
    return "\n".join(lines)


def _is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _make_dedupe_key(value: Any) -> str:
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return repr(sorted(value.items()))
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return repr(value)
    return str(value)


def _dedupe_preserve_order(values: Sequence[Any] | None) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for item in values or []:
        key = _make_dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _resolve_symbol_alias(symbol: str | None) -> str | None:
    if not symbol:
        return None
    sym = str(symbol).strip()
    if not sym:
        return None
    alias_target = SYMBOL_ALIASES.get(sym) or SYMBOL_ALIASES.get(sym.upper())
    dynamic_alias = DYNAMIC_SYMBOL_ALIASES.get(sym) or DYNAMIC_SYMBOL_ALIASES.get(sym.upper())
    if dynamic_alias:
        sym = dynamic_alias
    if alias_target:
        sym = alias_target
    sym_upper = sym.upper()
    # Explicit spot directive: e.g., BTC/USDT:SPOT -> BTC/USDT
    if sym_upper.endswith(":SPOT") or sym_upper.endswith(":SP"):
        base_part = sym_upper.split("/")[0]
        quote_part = sym_upper.split("/")[1].split(":")[0] if "/" in sym_upper else "USDT"
        return f"{base_part}/{quote_part}"
    if ":" in sym_upper:
        return sym_upper
    mapped = TICKER_TO_SYMBOL.get(sym_upper)
    if mapped:
        return mapped
    if "/" in sym_upper:
        base, quote = sym_upper.split("/", 1)
        if ":" in quote:
            return f"{base}/{quote}"
        quote = quote or "USDT"
        if quote == "USDT":
            # Default to derivatives unless explicitly marked spot via :SPOT
            return f"{base}/USDT:USDT"
        return f"{base}/{quote}"
    for suffix in RECOGNIZED_STABLE_QUOTES:
        if sym_upper.endswith(suffix):
            base = sym_upper[: -len(suffix)]
            if not base:
                continue
            mapped = TICKER_TO_SYMBOL.get(base)
            if mapped:
                return mapped
            return f"{base}/{suffix}:{suffix}"
    return f"{sym_upper}/USDT:USDT"



def _normalize_market_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        s = str(value).strip().lower()
    except Exception:
        return None
    if not s:
        return None
    if s in {"spot", "cash", "spot_market"}:
        return "spot"
    if s in {"derivatives", "futures", "perp", "perps", "contract", "contracts"}:
        return "derivatives"
    if s in {"linear", "inverse"}:
        return s
    return None


def _apply_category_to_symbol(symbol: str, category: Optional[str]) -> str:
    cat = (category or "").lower()
    sym = str(symbol).strip()
    if cat == "spot":
        # Force plain spot symbol without settle suffix
        if ":" in sym:
            left, right = sym.split("/", 1) if "/" in sym else (sym, "USDT")
            quote = right.split(":")[0]
            return f"{left}/{quote}"
        if "/" in sym:
            return sym
        return f"{sym}/USDT"
    # For derivatives default to USDT settle if not specified
    if ":" in sym:
        return sym
    if "/" in sym:
        base, quote = sym.split("/", 1)
        if ":" in quote:
            return sym
        if quote.upper() == "USDT":
            return f"{base}/USDT:USDT"
        return sym
    return f"{sym}/USDT:USDT"

def _canonical_decision_symbol(symbol: str | None) -> str | None:
    resolved = _resolve_symbol_alias(symbol)
    if resolved:
        return resolved
    if symbol:
        return str(symbol).strip()
    return None

def _order_allows_increase(order: dict) -> bool:
    if not isinstance(order, dict):
        return True
    flags = ["allowIncrease", "allow_increase", "increase", "increasePosition", "increase_position", "scaleIn", "scale_in"]
    for key in flags:
        if _is_truthy_flag(order.get(key)):
            return True
    intent = (order.get("intent") or order.get("action") or "").lower()
    if intent in {"increase", "scale_in", "add", "add_position"}:
        return True
    return True

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


def _fallback_momentum_decision(
    symbol: str,
    df_30m: pd.DataFrame,
    higher_trend_bias: str | None,
    current_position: dict | None,
) -> dict[str, Any] | None:
    if df_30m is None or df_30m.empty:
        return None
    try:
        last_row = df_30m.iloc[-1]
    except Exception:
        return None
    ema20_val = safe_float(last_row.get("ema20"))
    ema50_val = safe_float(last_row.get("ema50"))
    rsi_val = safe_float(last_row.get("rsi"))
    close_price = safe_float(last_row.get("close"))
    if ema20_val is None or ema50_val is None or close_price is None:
        return None

    bias: str | None = None
    if ema20_val > ema50_val * (1 + 1e-6):
        bias = "buy"
    elif ema50_val > ema20_val * (1 + 1e-6):
        bias = "sell"

    position_amount = safe_float(
        (current_position or {}).get("amount")
        or (current_position or {}).get("contracts")
    ) or 0.0
    position_side_raw = (current_position or {}).get("side") or ""
    position_side = position_side_raw.lower()
    if not position_side and position_amount != 0:
        position_side = "buy" if position_amount > 0 else "sell"

    def confidence_bonus(side: str) -> float:
        if higher_trend_bias and higher_trend_bias == side:
            return 0.08
        if higher_trend_bias:
            return 0.02
        return 0.04

    if position_amount != 0:
        if position_side in ("buy", "long"):
            if bias == "sell" or (rsi_val is not None and rsi_val < 45):
                return {
                    "symbol": symbol,
                    "action": "close",
                    "confidence": min(0.9, 0.72 + confidence_bonus("sell")),
                    "reason": "fallback: exit long as EMA trend flipped",
                    "needs": [],
                    "source": "fallback",
                }
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": min(0.9, 0.70 + confidence_bonus("buy")),
                "reason": "fallback: long bias intact; keep protections fresh",
                "needs": [],
                "source": "fallback",
            }
        if position_side in ("sell", "short"):
            if bias == "buy" or (rsi_val is not None and rsi_val > 55):
                return {
                    "symbol": symbol,
                    "action": "close",
                    "confidence": min(0.9, 0.72 + confidence_bonus("buy")),
                    "reason": "fallback: exit short as EMA trend flipped",
                    "needs": [],
                    "source": "fallback",
                }
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": min(0.9, 0.70 + confidence_bonus("sell")),
                "reason": "fallback: short bias intact; keep protections fresh",
                "needs": [],
                "source": "fallback",
            }
        return None

    if bias == "buy":
        if rsi_val is None or 48 <= rsi_val <= 70:
            return {
                "symbol": symbol,
                "action": "open",
                "side": "buy",
                "confidence": min(0.9, 0.68 + confidence_bonus("buy")),
                "reason": "fallback: EMA20>EMA50 with supportive RSI momentum",
                "needs": [],
                "source": "fallback",
                "notes": {"fallback_strategy": "ema_rsi_momentum"},
            }
    elif bias == "sell":
        if rsi_val is None or 30 <= rsi_val <= 52:
            return {
                "symbol": symbol,
                "action": "open",
                "side": "sell",
                "confidence": min(0.9, 0.68 + confidence_bonus("sell")),
                "reason": "fallback: EMA20<EMA50 with RSI confirming weakness",
                "needs": [],
                "source": "fallback",
                "notes": {"fallback_strategy": "ema_rsi_momentum"},
            }
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
                continue
            summary["take_profit"].append((_safe_round(take_price, 6), amount_round))
    for key in summary:
        summary[key].sort()
    return summary


def _evaluate_position_protection(
    position_payload: dict[str, Any] | None,
    orders,
    price_hint: float | None = None,
) -> tuple[bool, bool, dict[str, list[tuple[float | None, float | None]]]]:
    """
    Return (has_stop_loss, has_take_profit) for a position given its protective orders.

    For LONG positions, only stops *below* entry price are treated as stop-loss protection.
    For SHORT positions, only stops *above* entry price are treated as stop-loss protection.
    Stops on the profitable side are treated as take-profit equivalents.
    """
    position_side_raw = str((position_payload or {}).get("side") or "").lower()
    amount_val = safe_float(
        (position_payload or {}).get("amount") or (position_payload or {}).get("contracts")
    )
    if position_side_raw in {"sell", "short"}:
        is_long = False
    elif position_side_raw in {"buy", "long"}:
        is_long = True
    else:
        is_long = False if amount_val is not None and amount_val < 0 else True

    entry_price = safe_float(
        (position_payload or {}).get("entryPrice")
        or (position_payload or {}).get("average")
        or (position_payload or {}).get("avgEntryPrice")
    )
    mark_price = safe_float(
        (position_payload or {}).get("markPrice")
        or (position_payload or {}).get("mark_price")
        or (position_payload or {}).get("lastPrice")
    )
    live_price = safe_float(
        (position_payload or {}).get("last_price")
        or (position_payload or {}).get("price")
        or ((position_payload or {}).get("raw") or {}).get("lastPrice")
    )
    guard_price = price_hint if price_hint is not None and math.isfinite(price_hint) else None
    if guard_price is None or not math.isfinite(guard_price):
        guard_price = mark_price
    if guard_price is None or not math.isfinite(guard_price):
        guard_price = live_price
    if guard_price is None or not math.isfinite(guard_price):
        guard_price = entry_price

    categorized = _categorize_protection_orders(orders)
    # Filter unreasonable take levels (e.g. 43k for ETH) to avoid false positives.
    filtered_takes: list[tuple[float | None, float | None]] = []
    for price_val, amt_val in categorized["take_profit"]:
        if price_val is None or not math.isfinite(price_val):
            continue
        if guard_price is not None and math.isfinite(guard_price):
            ratio = abs(price_val) / max(abs(guard_price), 1e-9)
            if ratio > PROTECTION_MAX_PRICE_RATIO or ratio < PROTECTION_MIN_PRICE_RATIO:
                continue
        filtered_takes.append((price_val, amt_val))
    categorized["take_profit"] = filtered_takes
    has_take_profit = bool(filtered_takes)
    has_trailing = bool(categorized["trailing"])

    expected_stop_side = "sell" if is_long else "buy"
    has_stop_loss = False
    for order in _extract_protection_orders(orders):
        if not (_has_stop_flag(order) or _has_trailing_flag(order)):
            continue
        order_side = str(order.get("side") or "").lower()
        if order_side and order_side != expected_stop_side:
            continue
        if _has_trailing_flag(order):
            has_stop_loss = True
            continue
        stop_val = safe_float(order.get("stopPrice") or order.get("triggerPrice") or order.get("stopLoss"))
        if stop_val is None or not math.isfinite(stop_val):
            continue
        ref_price = guard_price if guard_price is not None and math.isfinite(guard_price) else entry_price
        tol = max(abs(ref_price or 0.0) * 1e-4, 1e-3)
        if ref_price is not None and math.isfinite(ref_price):
            if is_long:
                if stop_val <= ref_price - tol:
                    has_stop_loss = True
            else:
                if stop_val >= ref_price + tol:
                    has_stop_loss = True
        else:
            has_stop_loss = True
    if not has_stop_loss and categorized.get("stop"):
        # If stops were detected but didn't pass strict directional/threshold checks, treat as protective to avoid false "missing protection" closes.
        has_stop_loss = True

    return has_stop_loss, has_take_profit, categorized


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


def _format_protection_snapshot(orders, position_payload: dict | None = None) -> str:
    stops: list[float] = []
    takes: list[float] = []
    for order in _extract_protection_orders(orders):
        stop_val = safe_float(
            order.get("stopPrice")
            or order.get("triggerPrice")
            or order.get("stopLoss")
        )
        if stop_val is not None and math.isfinite(stop_val) and abs(float(stop_val)) > 1e-12:
            stops.append(float(stop_val))
        take_val = safe_float(
            order.get("takeProfit")
            or order.get("tpPrice")
            or order.get("price")
        )
        if take_val is not None and math.isfinite(take_val) and abs(float(take_val)) > 1e-12:
            takes.append(float(take_val))
    if position_payload and isinstance(position_payload, dict):
        pos_stop = safe_float(
            position_payload.get("stopLoss")
            or (position_payload.get("raw") or {}).get("stopLoss")
            or (position_payload.get("raw") or {}).get("sl")
        )
        if pos_stop is not None and math.isfinite(pos_stop) and abs(float(pos_stop)) > 1e-12:
            stops.append(float(pos_stop))
        pos_take = safe_float(
            position_payload.get("takeProfit")
            or (position_payload.get("raw") or {}).get("takeProfit")
            or (position_payload.get("raw") or {}).get("tp")
        )
        if pos_take is not None and math.isfinite(pos_take) and abs(float(pos_take)) > 1e-12:
            takes.append(float(pos_take))
    stops.sort()
    takes.sort()
    def _format_list(values: list[float]) -> str:
        if not values:
            return "n/a"
        return ",".join(f"{value:.4f}" for value in values[:5])

    return f"stop={_format_list(stops)}; take={_format_list(takes)}"


def _format_position_snapshot(amount: float | None) -> str:
    if amount is None or not math.isfinite(amount) or abs(amount) < 1e-12:
        return "flat"
    direction = "LONG" if amount > 0 else "SHORT"
    return f"{direction} {abs(amount):.4f}"


def _describe_size_change(initial_amount: float, final_amount: float, tolerance: float) -> str | None:
    if tolerance is None:
        return None
    if abs(initial_amount - final_amount) <= tolerance:
        return None
    before = _format_position_snapshot(initial_amount)
    after = _format_position_snapshot(final_amount)
    return f"size {before} -> {after}"


def _get_position_reference_price(payload: dict | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload.get("entryPrice"),
        payload.get("avgEntryPrice"),
        payload.get("avgPrice"),
        payload.get("markPrice"),
        payload.get("lastPrice"),
    ]
    for candidate in candidates:
        value = safe_float(candidate)
        if value is not None and math.isfinite(value):
            return float(value)
    return None


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
    global _LAST_INPROGRESS_MESSAGE
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
        status_snapshot = status_proc.stdout or ""
    except Exception as exc:
        log(f"[WARN] Git status failed before sync: {exc}", Fore.YELLOW)
        return
    global _LAST_WORKTREE_STATE_DIRTY, _LAST_WORKTREE_STATE_HASH
    if working_tree_dirty:
        current_hash = hashlib.sha256(status_snapshot.encode("utf-8")).hexdigest()
        if (not _LAST_WORKTREE_STATE_DIRTY) or (current_hash != _LAST_WORKTREE_STATE_HASH):
            lines = status_snapshot.strip().splitlines()
            preview_lines = lines[:10]
            preview = "\n".join(f"- {line}" for line in preview_lines) if preview_lines else "- изменения без подробностей"
            if len(lines) > len(preview_lines):
                preview += f"\n… и ещё {len(lines) - len(preview_lines)} файлов"
            if INPROGRESS_WIP_ENABLED:
                _send_inprogress_notification(f"[WIP] Есть незакоммиченные изменения:\n{preview}")
        _LAST_WORKTREE_STATE_DIRTY = True
        _LAST_WORKTREE_STATE_HASH = current_hash
    else:
        if _LAST_WORKTREE_STATE_DIRTY:
            if INPROGRESS_WIP_ENABLED:
                _send_inprogress_notification("[WIP] Рабочее дерево очищено.")
            _LAST_INPROGRESS_MESSAGE = None
        _LAST_WORKTREE_STATE_DIRTY = False
        _LAST_WORKTREE_STATE_HASH = ""
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
            _send_git_notification(f"[GIT] fetch: {fetch_output}")
    except subprocess.CalledProcessError as exc:
        log(f"[WARN] Git fetch failed: {exc.stderr or exc.stdout or exc}", Fore.YELLOW)
        return
    branch_name = get_current_branch_name()
    pull_cmd = ["git", "pull", "--ff-only"]
    if branch_name:
        pull_cmd.extend(["origin", branch_name])
    if working_tree_dirty:
        _send_git_notification("[GIT] Working tree dirty, pulling with --autostash.")
        pull_cmd.append("--autostash")
    try:
        pull_proc = subprocess.run(
            pull_cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        pull_output = (pull_proc.stdout or pull_proc.stderr or "").strip()
        if pull_output:
            _send_git_notification(f"[GIT] pull: {pull_output}")
    except subprocess.CalledProcessError as exc:
        details = exc.stderr or exc.stdout or str(exc)
        lower_details = (details or "").lower()
        if ("fast-forward" in lower_details or "divergent" in lower_details or "need to specify how to reconcile" in lower_details) and not working_tree_dirty:
            branch_name = get_current_branch_name() or "auto"
            log(f"[GIT] Pull not possible (history diverged); resetting to origin/{branch_name}.", Fore.YELLOW)
            try:
                reset_proc = subprocess.run(
                    ["git", "reset", "--hard", f"origin/{branch_name}"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as reset_exc:
                reset_details = reset_exc.stderr or reset_exc.stdout or str(reset_exc)
                log(f"[WARN] git reset --hard origin/{branch_name} failed: {reset_details}", Fore.RED)
            else:
                reset_output = (reset_proc.stdout or reset_proc.stderr or "").strip()
                if reset_output:
                    _send_git_notification(f"[GIT] reset --hard origin/{branch_name}: {reset_output}")
                else:
                    _send_git_notification(f"[GIT] reset --hard origin/{branch_name}")
                return
        else:
            if working_tree_dirty:
                log("[WARN] Git pull failed and рабочее дерево грязное; пропускаем автосинх.", Fore.YELLOW)
            log(f"[WARN] Git pull failed: {details}", Fore.YELLOW)


def _cleanup_redundant_stop_orders(
    exchange,
    symbol,
    reduce_orders,
    protection_side,
    position_qty,
    is_long,
    keep_ids_preferred: set[str] | None = None,
):
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
        summary_text = _summarize_order_spec(order)
        summary_repr = f"{order_id}: {summary_text}" if summary_text else str(order_id)
        stop_entries.append(
            {
                "id": str(order_id),
                "trigger": trigger_price,
                "amount": remaining,
                "summary": summary_repr,
            }
        )

    if len(stop_entries) <= 1:
        return [], []

    stop_entries.sort(key=lambda item: item["trigger"], reverse=is_long)
    coverage = 0.0
    tolerance = max(position_qty * 1e-6, 1e-8)
    keep_ids: set[str] = set()
    preferred_ids: set[str] = {str(val) for val in (keep_ids_preferred or set())}
    # Always keep newly placed stops first to avoid cancelling fresh protection.
    for entry in stop_entries:
        if entry["id"] in preferred_ids:
            keep_ids.add(entry["id"])
            coverage += entry["amount"]
    for entry in stop_entries:
        if entry["id"] in keep_ids:
            continue
        keep_ids.add(entry["id"])
        coverage += entry["amount"]
        if coverage >= position_qty - tolerance:
            break

    cancelled_entries: list[str] = []
    cancel_errors: list[tuple[str, str]] = []
    for entry in stop_entries:
        if entry["id"] in keep_ids:
            continue
        success, err = cancel_order_by_id(exchange, symbol, entry["id"])
        if success:
            cancelled_entries.append(entry.get("summary") or entry["id"])
        else:
            descriptor = entry.get("summary") or entry["id"]
            cancel_errors.append((descriptor, err))
    return cancelled_entries, cancel_errors


def _close_position_now(
    exchange, symbol, qty, close_side, *, position_idx: int | None = None
) -> None:
    params = {"reduceOnly": True}
    if position_idx is not None:
        params["positionIdx"] = position_idx
    try:
        exchange.create_order(symbol, "market", close_side, qty, None, params)
        log(f"[INFO] {symbol}: immediate close {close_side.upper()} {qty:.6f} due to protection breach", Fore.YELLOW)
    except Exception as exc:
        log(f"[WARN] Failed to close {symbol} during protection check: {exc}", Fore.YELLOW)


def _trail_state_matches_position(
    trail_state: dict[str, Any] | None,
    *,
    is_long: bool,
    entry_price: float | None,
    position_qty: float | None,
) -> bool:
    if not isinstance(trail_state, dict):
        return False
    recorded_side = (trail_state.get("position_side") or "").lower()
    if recorded_side:
        if is_long and recorded_side != "long":
            return False
        if not is_long and recorded_side != "short":
            return False
    recorded_qty = safe_float(trail_state.get("position_qty"))
    if (
        position_qty is not None
        and math.isfinite(position_qty)
        and recorded_qty is not None
        and math.isfinite(recorded_qty)
        and recorded_qty > 0
        and position_qty > 0
    ):
        denom = max(abs(recorded_qty), abs(position_qty), 1e-9)
        ratio = abs(position_qty - recorded_qty) / denom
        if ratio > float(PSEUDOTRAIL_POSITION_SIZE_STALE_RATIO):
            return False
    if entry_price is not None and math.isfinite(entry_price):
        base_price = safe_float(trail_state.get("base_price"))
        if base_price is not None and math.isfinite(base_price):
            diff = abs(entry_price - base_price)
            threshold = max(abs(base_price), abs(entry_price), 1.0) * float(PSEUDOTRAIL_POSITION_STALE_PCT)
            if diff > threshold:
                return False
    return True



def ensure_position_protection(exchange, symbol, position, df_primary, open_orders, config=None):
    global _PREV_UNREALIZED_PNL
    global _TRAIL_PROTECTION
    global PSEUDOTRAIL_MIN_IMPROVE_ATR, PSEUDOTRAIL_STOP_LOCK_FACTOR, PSEUDOTRAIL_TP_EXTEND_FACTOR
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
    reduce_orders_source = open_orders or []
    if not reduce_orders_source:
        reduce_orders_source = fetch_open_orders_for_symbol(exchange, symbol, limit=200)
    reduce_orders = [order for order in reduce_orders_source if isinstance(order, dict)]
    position_qty = abs(position_amount)
    # Defer cleanup of redundant stops until AFTER new protection is placed,
    # to avoid leaving the position unprotected if new orders fail.
    # We'll refresh and clean up near the end of this function.

    target_spec = cfg.get("target") if isinstance(cfg.get("target"), dict) else {}
    trailing_requested = any(
        key in cfg
        for key in (
            "trailing_atr_mult",
            "trailing_atr",
            "trailing",
            "trailing_stop",
        )
    ) or any(
        target_spec.get(key) not in (None, "")
        for key in (
            "trailingStop",
            "trailing_stop",
            "trailingPercent",
            "trailing_percent",
            "trailingCallback",
            "trailing_callback",
        )
    )
    trailing_mult = cfg.get("trailing_atr_mult", 0.0 if not trailing_requested else TRAILING_ATR_MULT)

    has_stop = False
    has_take = False
    has_trailing = False
    existing_stop_prices: list[float] = []
    existing_stop_best: float | None = None
    existing_take_prices: list[float] = []
    existing_take_count = 0
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
            if stop_price_existing is not None and math.isfinite(stop_price_existing):
                existing_stop_prices.append(float(stop_price_existing))
        elif order_type in ("takeprofit", "limit") and existing.get("price") is not None:
            take_px = safe_float(existing.get("price") or existing.get("takeProfit") or existing.get("take_profit"))
            if take_px is not None and math.isfinite(take_px):
                existing_take_prices.append(float(take_px))
        elif order_type == "trailingstop" or trailing_flag:
            has_trailing = True

    df_calc = df_primary.copy() if isinstance(df_primary, pd.DataFrame) and not df_primary.empty else None
    if df_calc is None:
        log(f"[WARN] {symbol}: пропуск обновления защиты — нет актуальных свечей для расчёта", Fore.LIGHTBLACK_EX)
        return open_orders or []
    if "atr" not in df_calc.columns:
        try:
            df_calc["atr"] = atr(df_calc, 14)
        except Exception as exc:
            log(f"⚠️ {symbol}: не удалось вычислить ATR для защиты позиции ({exc})", Fore.YELLOW)
            return open_orders or []
    last_row = df_calc.iloc[-1]
    # Use live market/mark price when possible; candle close can be stale enough to create invalid triggers
    # (e.g. stop trigger <= current price) and leave positions unprotected.
    atrv = safe_float(last_row.get("atr"))
    live_price = None
    try:
        ticker = exchange.fetch_ticker(exchange_symbol)
        if isinstance(ticker, dict):
            live_price = safe_float(
                ticker.get("last")
                or ticker.get("close")
                or (ticker.get("info") or {}).get("lastPrice")
                or (ticker.get("info") or {}).get("price")
            )
    except Exception:
        live_price = None
    raw_mark_price = safe_float(
        position.get("markPrice")
        or position.get("mark_price")
        or position.get("lastPrice")
        or position.get("last_price")
        or (position.get("raw") or {}).get("markPrice")
        or (position.get("raw") or {}).get("lastPrice")
    )
    close_price = safe_float(last_row.get("close"))
    price_candidates = [live_price, raw_mark_price, close_price]
    price = next((p for p in price_candidates if p is not None and math.isfinite(p)), close_price)
    # If mark price deviates слишком сильно от последней свечи (устаревший снимок), используем close.
    if (
        price is not None
        and math.isfinite(price)
        and close_price is not None
        and math.isfinite(close_price)
        and atrv is not None
        and math.isfinite(atrv)
        and atrv > 0
        and raw_mark_price is not None
        and math.isfinite(raw_mark_price)
        and abs(raw_mark_price - close_price) > (5.0 * atrv)
    ):
        price = close_price
    if not (math.isfinite(price) and math.isfinite(atrv) and atrv and atrv > 0):
        log(f"⚠️ {symbol}: нет валидных значений ATR/цены для защиты позиции", Fore.YELLOW)
        return open_orders or []

    explicit_entry = safe_float(target_spec.get("entryPrice") or target_spec.get("entry_price"))
    entry_price = safe_float(position.get("entryPrice") or position.get("avgEntryPrice") or position.get("entry_price"))
    reference_price = price
    if explicit_entry and math.isfinite(explicit_entry):
        reference_price = explicit_entry
    elif entry_price and math.isfinite(entry_price):
        reference_price = entry_price

    trail_state = _TRAIL_PROTECTION.get(symbol) if isinstance(_TRAIL_PROTECTION, dict) else None
    if trail_state and not _trail_state_matches_position(
        trail_state,
        is_long=is_long,
        entry_price=entry_price,
        position_qty=position_qty,
    ):
        trail_state = None
        try:
            del _TRAIL_PROTECTION[symbol]
        except KeyError:
            pass
    trail_take_extensions = max(0, safe_int((trail_state or {}).get("take_extensions")) or 0)
    tightening_count = max(0, safe_int((trail_state or {}).get("tightening_count")) or 0)
    take_last_extended_cycle = safe_int((trail_state or {}).get("take_last_extended_cycle"))

    if is_long:
        stop_price = price - sl_mult * atrv
        take_price = price + tp_mult * atrv
    else:
        stop_price = price + sl_mult * atrv
        take_price = price - tp_mult * atrv

    market_ref = live_price if (live_price is not None and math.isfinite(live_price)) else raw_mark_price
    if market_ref is None or not math.isfinite(market_ref):
        market_ref = price

    # If we saw existing stop triggers, treat only those on the loss side as stop-loss protection
    # (long: below market; short: above market). Ignore mis-sided triggers so they won't poison stop_price.
    if existing_stop_prices:
        tol = max(abs(market_ref) * 1e-4, 1e-3)
        if is_long:
            valid = [p for p in existing_stop_prices if p < market_ref - tol]
            if valid:
                has_stop = True
                existing_stop_best = max(valid)
                stop_price = max(stop_price, existing_stop_best)
        else:
            valid = [p for p in existing_stop_prices if p > market_ref + tol]
            if valid:
                has_stop = True
                existing_stop_best = min(valid)
                stop_price = min(stop_price, existing_stop_best)

    if existing_take_prices:
        tol = max(abs(market_ref) * 1e-4, 1e-6)
        if is_long:
            valid_takes = [p for p in existing_take_prices if p > market_ref + tol]
        else:
            valid_takes = [p for p in existing_take_prices if p < market_ref - tol]
        if valid_takes:
            has_take = True
            existing_take_count = len(valid_takes)

    already_protected = has_stop and has_take and (trailing_mult <= 0 or has_trailing)
    # If we have trailing state from the previous cycle, use it as a baseline to avoid losing prior tightening.
    trail_activated_cycle = safe_int((trail_state or {}).get("activated_cycle"))
    trail_active = trail_activated_cycle is not None and trail_activated_cycle >= 0
    cycles_since_activation = None
    if trail_active:
        try:
            cycles_since_activation = max(0, int((_CURRENT_CYCLE_NUMBER or trail_activated_cycle) - trail_activated_cycle))
        except Exception:
            cycles_since_activation = 0

    def _trail_status_tag(*, activation_event: bool = False) -> str:
        if activation_event:
            return ", trail=activated"
        if trail_active:
            if cycles_since_activation is None:
                return ", trail=active"
            return f", trail=active({cycles_since_activation}c)"
        return ", trail=inactive"

    stored_stop = safe_float((trail_state or {}).get("stop"))
    stored_take = safe_float((trail_state or {}).get("take"))
    if stored_stop is not None and math.isfinite(stored_stop):
        stop_price = stored_stop
    if stored_take is not None and math.isfinite(stored_take):
        # Only reuse stored take if it is on the profitable side of the current market.
        tol = max(abs(market_ref) * 1e-4, 1e-6)
        if (is_long and stored_take > market_ref + tol) or ((not is_long) and stored_take < market_ref - tol):
            take_price = stored_take
    # Ensure stop is on the correct side of the current price.
    if is_long and stop_price is not None and math.isfinite(stop_price) and market_ref is not None and math.isfinite(market_ref):
        if stop_price >= market_ref:
            stop_price = market_ref - sl_mult * atrv
    elif not is_long and stop_price is not None and math.isfinite(stop_price) and market_ref is not None and math.isfinite(market_ref):
        if stop_price <= market_ref:
            stop_price = market_ref + sl_mult * atrv

    # Sanity: take should be on the profitable side (and never behind stop).
    if take_price is not None and math.isfinite(take_price) and stop_price is not None and math.isfinite(stop_price):
        tol = max(abs(market_ref) * 1e-4, 1e-6)
        if is_long:
            if take_price <= market_ref + tol or take_price <= stop_price + tol:
                take_price = market_ref + tp_mult * atrv
        else:
            if take_price >= market_ref - tol or take_price >= stop_price - tol:
                take_price = market_ref - tp_mult * atrv

    breakeven_note = None
    profit_distance = 0.0
    if entry_price and math.isfinite(entry_price) and math.isfinite(price):
        if is_long:
            profit_distance = max(0.0, price - entry_price)
        else:
            profit_distance = max(0.0, entry_price - price)

    # Pseudo-trailing across cycles: if unrealized PnL improved since the previous cycle, gently
    # tighten the stop and let the take-profit breathe a bit further. If PnL worsened, leave
    # protection unchanged to avoid expanding risk.
    initial_stop_price = stop_price
    initial_take_price = take_price
    pseudo_timeframe = str(cfg.get("timeframe") or TIMEFRAME or "n/a")
    pseudo_ctx = f"[PSEUDOTRAIL] {symbol}@{pseudo_timeframe}"
    current_unreal = safe_float(position.get("unrealizedPnl") or (position.get("raw") or {}).get("unrealisedPnl"))
    prev_unreal = _PREV_UNREALIZED_PNL.get(symbol) if isinstance(_PREV_UNREALIZED_PNL, dict) else None
    current_valid = current_unreal is not None and math.isfinite(current_unreal)
    prev_valid = prev_unreal is not None and isinstance(prev_unreal, (int, float)) and math.isfinite(prev_unreal)
    delta_unreal: float | None = None
    delta_price_equiv: float | None = None
    tightened_applied = False
    pre_tightening_count = tightening_count
    if current_valid and prev_valid and atrv is not None and math.isfinite(atrv) and atrv > 0 and position_qty and math.isfinite(position_qty) and position_qty > 0:
        delta_unreal = current_unreal - prev_unreal
        # Normalize unrealized PnL delta into an approximate price movement so the trigger is position-size invariant.
        # For linear USDT contracts / spot this matches: ΔPnL ≈ ΔPrice * qty.
        delta_price_equiv = float(delta_unreal) / float(position_qty)
        improve_threshold = atrv * PSEUDOTRAIL_MIN_IMPROVE_ATR
        tol = max(improve_threshold * 1e-6, 1e-9)
        if delta_price_equiv + tol >= improve_threshold and improve_threshold > 0:
            was_active = trail_active
            lock_distance = delta_price_equiv * PSEUDOTRAIL_STOP_LOCK_FACTOR
            if lock_distance > 0:
                if is_long:
                    candidate_stop = price - lock_distance
                    if math.isfinite(candidate_stop) and candidate_stop > stop_price:
                        stop_price = candidate_stop
                else:
                    candidate_stop = price + lock_distance
                    if math.isfinite(candidate_stop) and candidate_stop < stop_price:
                        stop_price = candidate_stop
            extend = delta_price_equiv * PSEUDOTRAIL_TP_EXTEND_FACTOR
            take_extension_delta = extend if extend is not None else 0.0
            if take_extension_delta and take_extension_delta > 0 and atrv is not None and math.isfinite(atrv):
                cap = float(PSEUDOTRAIL_MAX_TAKE_SHIFT_ATR_MULT) * atrv
                if math.isfinite(cap) and cap > 0:
                    take_extension_delta = min(take_extension_delta, cap)
            should_extend_take = (
                take_extension_delta > 0
                and pre_tightening_count >= 1
                and trail_take_extensions < PSEUDOTRAIL_MAX_TAKE_EXTENDS
            )
            if should_extend_take:
                if is_long:
                    candidate_take = take_price + take_extension_delta
                else:
                    candidate_take = take_price - take_extension_delta
                if candidate_take is not None and math.isfinite(candidate_take):
                    take_price = candidate_take
                    trail_take_extensions += 1
                    if _CURRENT_CYCLE_NUMBER is not None:
                        take_last_extended_cycle = _CURRENT_CYCLE_NUMBER
            def _fmt_px(value: float | None) -> str:
                return f"{value:.4f}" if value is not None and math.isfinite(value) else "n/a"

            stop_transition_text = ""
            if (
                initial_stop_price is not None
                and math.isfinite(initial_stop_price)
                and stop_price is not None
                and math.isfinite(stop_price)
            ):
                stop_transition_text = f", stop {_fmt_px(initial_stop_price)} -> {_fmt_px(stop_price)}"

            take_transition_text = ""
            if (
                initial_take_price is not None
                and math.isfinite(initial_take_price)
                and take_price is not None
                and math.isfinite(take_price)
            ):
                take_transition_text = f", take {_fmt_px(initial_take_price)} -> {_fmt_px(take_price)}"

            px_text = f", ΔPx≈{delta_price_equiv:.4f}, ATR={atrv:.4f}, triggerPx={improve_threshold:.4f}, qty={position_qty:.6f}"
            keep_tp_note = f", keep_tp={existing_take_count}" if has_take else ""
            log(
                f"{pseudo_ctx}: tightened{keep_tp_note} ΔPnL={delta_unreal:.4f}{px_text}{stop_transition_text}{take_transition_text}"
                f"{_trail_status_tag(activation_event=(not was_active))}",
                Fore.LIGHTBLUE_EX,
            )
            tightening_count = pre_tightening_count + 1
            tightened_applied = True
        else:
            def _fmt_px(value: float | None) -> str:
                return f"{value:.4f}" if value is not None and math.isfinite(value) else "n/a"
            stop_text = _fmt_px(initial_stop_price)
            take_text = _fmt_px(initial_take_price)
            log(
                f"{pseudo_ctx}: skipped (ΔPnL={delta_unreal:.4f}, ΔPx≈{delta_price_equiv:.4f}, triggerPx={improve_threshold:.4f}, "
                f"stop={stop_text}, take={take_text}){_trail_status_tag()}",
                Fore.LIGHTBLACK_EX,
            )
    else:
        missing_reasons: list[str] = []
        if not current_valid:
            missing_reasons.append("current PnL unavailable")
        if not prev_valid:
            missing_reasons.append("previous PnL unavailable")
        if position_qty is None or not math.isfinite(position_qty) or position_qty <= 0:
            missing_reasons.append("position qty unavailable")
        if atrv is None or not math.isfinite(atrv) or atrv <= 0:
            missing_reasons.append("ATR unavailable")
        if not missing_reasons:
            missing_reasons.append("PnL not improved")
        if missing_reasons:
            log(f"{pseudo_ctx}: not applied ({'; '.join(missing_reasons)}){_trail_status_tag()}", Fore.LIGHTBLACK_EX)
    # Persist/restore trailing levels across cycles.
    if tightened_applied:
        base_stop = safe_float((trail_state or {}).get("base_stop")) or initial_stop_price
        base_take = safe_float((trail_state or {}).get("base_take")) or initial_take_price
        base_unreal = safe_float((trail_state or {}).get("base_unreal")) or prev_unreal
        base_price = safe_float((trail_state or {}).get("base_price")) or reference_price
        activated_cycle = safe_int((trail_state or {}).get("activated_cycle")) or (_CURRENT_CYCLE_NUMBER or 0)
        take_total_shift = safe_float((trail_state or {}).get("take_shift_total")) or 0.0
        if (
            base_stop is not None
            and math.isfinite(base_stop)
            and stop_price is not None
            and math.isfinite(stop_price)
            and base_unreal is not None
            and math.isfinite(base_unreal)
            and current_unreal is not None
            and math.isfinite(current_unreal)
            ):
            total_pnl_delta = current_unreal - base_unreal
            stop_total_shift = stop_price - base_stop
            take_total_shift = (
                (take_price - base_take)
                if (take_price is not None and math.isfinite(take_price) and base_take is not None and math.isfinite(base_take))
                else 0.0
            )
            cycles_ago = (_CURRENT_CYCLE_NUMBER or activated_cycle) - activated_cycle
            log(
                f"{pseudo_ctx}: TRAIL state active for {cycles_ago} cycles; "
                f"ΔPnL_total={total_pnl_delta:.4f}, stop_total={stop_total_shift:+.4f}, take_total={take_total_shift:+.4f}",
                Fore.LIGHTBLACK_EX,
            )
        _TRAIL_PROTECTION[symbol] = {
            "stop": float(stop_price) if stop_price is not None and math.isfinite(stop_price) else None,
            "take": float(take_price) if take_price is not None and math.isfinite(take_price) else None,
            "base_stop": float(base_stop) if base_stop is not None and math.isfinite(base_stop) else None,
            "base_take": float(base_take) if base_take is not None and math.isfinite(base_take) else None,
            "base_unreal": float(base_unreal) if base_unreal is not None and math.isfinite(base_unreal) else None,
            "base_price": float(base_price) if base_price is not None and math.isfinite(base_price) else None,
            "take_shift_total": float(take_total_shift) if math.isfinite(take_total_shift) else None,
            "activated_cycle": int(activated_cycle),
            "take_extensions": int(trail_take_extensions),
            "take_last_extended_cycle": int(take_last_extended_cycle) if take_last_extended_cycle is not None else None,
            "tightening_count": int(tightening_count),
            "position_side": "long" if is_long else "short",
            "position_qty": float(position_qty) if position_qty is not None and math.isfinite(position_qty) else None,
        }
    elif trail_state:
        activated_cycle = safe_int(trail_state.get("activated_cycle"))
        base_stop = safe_float(trail_state.get("base_stop"))
        base_take = safe_float(trail_state.get("base_take"))
        base_unreal = safe_float(trail_state.get("base_unreal"))
        take_total_shift = safe_float(trail_state.get("take_shift_total")) or 0.0
        if activated_cycle is not None and base_stop is not None and math.isfinite(base_stop):
            cycles_ago = (_CURRENT_CYCLE_NUMBER or activated_cycle) - activated_cycle
            total_pnl_delta = (current_unreal - base_unreal) if (current_unreal is not None and math.isfinite(current_unreal) and base_unreal is not None and math.isfinite(base_unreal)) else 0.0
            stop_total_shift = (stop_price - base_stop) if (stop_price is not None and math.isfinite(stop_price)) else 0.0
            take_total_shift = (
                (take_price - base_take)
                if (take_price is not None and math.isfinite(take_price) and base_take is not None and math.isfinite(base_take))
                else 0.0
            )
            log(
                f"{pseudo_ctx}: TRAIL state still active (skip) for {cycles_ago} cycles; "
                f"ΔPnL_total={total_pnl_delta:.4f}, stop_total={stop_total_shift:+.4f}, take_total={take_total_shift:+.4f}",
                Fore.LIGHTBLACK_EX,
            )
        _TRAIL_PROTECTION[symbol] = {
            "stop": float(stop_price) if stop_price is not None and math.isfinite(stop_price) else float(trail_state.get("stop")) if trail_state.get("stop") is not None else None,
            "take": float(take_price) if take_price is not None and math.isfinite(take_price) else float(trail_state.get("take")) if trail_state.get("take") is not None else None,
            "base_stop": float(base_stop) if base_stop is not None and math.isfinite(base_stop) else None,
            "base_take": float(base_take) if base_take is not None and math.isfinite(base_take) else None,
            "base_unreal": float(base_unreal) if base_unreal is not None and math.isfinite(base_unreal) else None,
            "base_price": float(trail_state.get("base_price")) if trail_state.get("base_price") is not None and math.isfinite(trail_state.get("base_price")) else None,
            "take_shift_total": float(take_total_shift) if math.isfinite(take_total_shift) else None,
            "activated_cycle": int(activated_cycle) if activated_cycle is not None else None,
            "take_extensions": int(trail_take_extensions),
            "take_last_extended_cycle": int(take_last_extended_cycle) if take_last_extended_cycle is not None else None,
            "tightening_count": int(tightening_count),
            "position_side": "long" if is_long else "short",
            "position_qty": float(position_qty) if position_qty is not None and math.isfinite(position_qty) else None,
        }
    if BREAKEVEN_ENABLED and entry_price and math.isfinite(entry_price):
        breakeven_trigger = atrv * BREAKEVEN_ATR_MULT
        breakeven_buffer = atrv * BREAKEVEN_BUFFER_ATR
        if is_long and breakeven_trigger > 0 and price - entry_price >= breakeven_trigger:
            breakeven_stop = entry_price + breakeven_buffer
            if math.isfinite(breakeven_stop):
                adjusted_stop = max(stop_price, breakeven_stop)
                if adjusted_stop > stop_price:
                    stop_price = adjusted_stop
                    breakeven_note = f"break-even {breakeven_stop:.4f}"
        elif not is_long and breakeven_trigger > 0 and entry_price - price >= breakeven_trigger:
            breakeven_stop = entry_price - breakeven_buffer
            if math.isfinite(breakeven_stop):
                adjusted_stop = min(stop_price, breakeven_stop)
                if adjusted_stop < stop_price:
                    stop_price = adjusted_stop
                    breakeven_note = f"break-even {breakeven_stop:.4f}"

    explicit_stop = safe_float(target_spec.get("stopLoss") or target_spec.get("stop_loss"))
    explicit_take = safe_float(target_spec.get("takeProfit") or target_spec.get("take_profit"))
    if explicit_stop is not None and math.isfinite(explicit_stop):
        stop_price = explicit_stop
    if explicit_take is not None and math.isfinite(explicit_take):
        take_price = explicit_take
    if existing_stop_best is not None and math.isfinite(existing_stop_best) and stop_price is not None and math.isfinite(stop_price):
        stop_price = max(stop_price, existing_stop_best) if is_long else min(stop_price, existing_stop_best)

    # -- immediate exit check --
    if IMMEDIATE_CLOSE_ON_BREACH:
        def _stop_breached(curr_price: float | None, target: float | None) -> bool:
            if curr_price is None or target is None or not math.isfinite(curr_price) or not math.isfinite(target):
                return False
            if is_long:
                return curr_price <= target + 1e-9
            return curr_price >= target - 1e-9

        def _take_reached(curr_price: float | None, target: float | None) -> bool:
            if curr_price is None or target is None or not math.isfinite(curr_price) or not math.isfinite(target):
                return False
            if is_long:
                return curr_price >= target - 1e-9
            return curr_price <= target + 1e-9

        if price is not None and math.isfinite(price) and entry_price is not None and math.isfinite(entry_price):
            close_side = "sell" if is_long else "buy"
            position_idx = get_position_idx(close_side)
            if _stop_breached(price, stop_price):
                _close_position_now(exchange, symbol, position_qty, close_side, position_idx=position_idx)
                return open_orders or []
            if _take_reached(price, take_price):
                _close_position_now(exchange, symbol, position_qty, close_side, position_idx=position_idx)
                return open_orders or []

    trailing_offset = None
    if trailing_requested:
        explicit_trailing = safe_float(target_spec.get("trailingStop") or target_spec.get("trailing_stop"))
        if explicit_trailing is not None and math.isfinite(explicit_trailing):
            trailing_offset = abs(explicit_trailing)
        if trailing_offset is None:
            trailing_percent = safe_float(target_spec.get("trailingPercent") or target_spec.get("trailing_percent"))
            if trailing_percent is not None and math.isfinite(trailing_percent) and trailing_percent > 0:
                base_price = reference_price if reference_price and math.isfinite(reference_price) else price
                trailing_offset = abs(base_price) * (trailing_percent / 100.0) if base_price else None
        if trailing_offset is None:
            trailing_callback = safe_float(target_spec.get("trailingCallback") or target_spec.get("trailing_callback"))
            if trailing_callback is not None and math.isfinite(trailing_callback) and trailing_callback > 0:
                trailing_offset = trailing_callback
        if trailing_offset is None and trailing_mult > 0 and math.isfinite(trailing_mult):
            trailing_offset = trailing_mult * atrv
        if trailing_mult > 0 and atrv and TRAILING_DYNAMIC_TRIGGER_ATR > 0 and profit_distance > 0:
            trigger_distance = atrv * TRAILING_DYNAMIC_TRIGGER_ATR
            if trigger_distance > 0 and profit_distance >= trigger_distance:
                dynamic_offset = atrv * TRAILING_DYNAMIC_FACTOR
                dynamic_offset = max(dynamic_offset, atrv * TRAILING_DYNAMIC_MIN_ATR)
                if dynamic_offset > 0 and (trailing_offset is None or dynamic_offset < trailing_offset - 1e-9):
                    trailing_offset = dynamic_offset
                    log(f"🔷 {symbol}: tightened trailing offset to {trailing_offset:.4f} (profit distance {profit_distance:.4f})", Fore.LIGHTBLUE_EX)
        if trailing_offset is not None and trailing_offset <= 0:
            trailing_offset = None
    refresh_takeprofits = tightened_applied or not has_take
    refresh_trailing = bool(trailing_requested) and not has_trailing
    should_place_stop = not has_stop
    if (
        has_stop
        and existing_stop_best is not None
        and math.isfinite(existing_stop_best)
        and stop_price is not None
        and math.isfinite(stop_price)
    ):
        if is_long:
            should_place_stop = stop_price > existing_stop_best + 1e-9
        else:
            should_place_stop = stop_price < existing_stop_best - 1e-9
    if already_protected and not refresh_takeprofits and not refresh_trailing and not should_place_stop:
        return open_orders or []
    qty = position_qty
    if not math.isfinite(qty) or qty <= 0:
        return open_orders or []
    position_idx = get_position_idx(protection_side)
    base_params = {
        "reduceOnly": True,
    }
    if position_idx is not None:
        base_params["positionIdx"] = position_idx
    stop_ids_preferred: set[str] = set()

    market_info = None
    try:
        market_info = exchange.market(exchange_symbol)
    except Exception:
        market_info = None
    position_category = _infer_market_category(exchange_symbol, market_info)
    min_amount = None
    min_notional = None
    min_qty_step = None
    if isinstance(market_info, dict):
        limits = market_info.get("limits")
        if isinstance(limits, dict):
            amount_limits = limits.get("amount")
            if isinstance(amount_limits, dict):
                min_amount = safe_float(amount_limits.get("min"))
            notional_limits = limits.get("cost")
            if isinstance(notional_limits, dict):
                min_notional = safe_float(notional_limits.get("min"))
        info_payload = market_info.get("info") if isinstance(market_info.get("info"), dict) else None
        lot_filter = info_payload.get("lotSizeFilter") if isinstance(info_payload, dict) else None
        if isinstance(lot_filter, dict):
            min_qty_step = safe_float(lot_filter.get("qtyStep")) or min_qty_step
            min_order_qty = safe_float(lot_filter.get("minOrderQty"))
            if min_order_qty is not None:
                min_amount = max(min_amount or 0.0, min_order_qty)

    created_log_parts: list[str] = []
    if should_place_stop:
        try:
            trigger_direction = get_trigger_direction_for_side(
                protection_side,
                trigger_price=stop_price,
                reference_price=price,
            )
            stop_params = dict(base_params)
            stop_params.update(
                {
                    "triggerPrice": stop_price,
                    "triggerDirection": trigger_direction,
                    "closeOnTrigger": True,
                }
            )
            stop_order = exchange.create_order(
                exchange_symbol,
                "market",
                protection_side,
                qty,
                None,
                stop_params,
            )
            try:
                if isinstance(stop_order, dict):
                    stop_order_id = (
                        stop_order.get("id")
                        or (stop_order.get("info") or {}).get("orderId")
                        or (stop_order.get("info") or {}).get("orderID")
                    )
                    if stop_order_id:
                        stop_ids_preferred.add(str(stop_order_id))
            except Exception:
                pass
            created_log_parts.append(f"stopLoss @ {stop_price:.2f}")
            if breakeven_note:
                created_log_parts.append(breakeven_note)
        except Exception as exc:
            log(f"⚠️ {symbol}: не удалось выставить стоп-ордер защиты позиции: {exc}", Fore.YELLOW)

    take_orders_success = bool(has_take)
    fallback_take_price = None
    fallback_limit_success = False
    remaining_qty = qty
    fallback_qty_target = qty
    take_created: list[str] = []
    min_qty_violation = False
    min_notional_violation = False
    take_limit_errors: list[str] = []

    if refresh_takeprofits:
        tp_scheme_override = target_spec.get("takeProfitLevels") or target_spec.get("take_profit_levels")
        # When trailing protection is active and we have a stored take-profit, prefer restoring that exact level
        # instead of regenerating ATR-based tiers, to avoid drifting away from the tightened TP.
        if trail_active and stored_take is not None and math.isfinite(stored_take):
            tp_scheme_override = [{"ratio": 1.0, "price": float(stored_take)}]
        normalized_scheme: list[tuple[float, str, float]] = []
        if isinstance(tp_scheme_override, list):
            for item in tp_scheme_override:
                ratio_val = None
                multiplier_val = None
                if isinstance(item, dict):
                    ratio_val = safe_float(item.get("ratio") or item.get("share") or item.get("size") or item.get("qty"))
                    explicit_price = safe_float(item.get("price"))
                    if explicit_price is not None and math.isfinite(explicit_price):
                        ratio_clean = float(ratio_val) if ratio_val is not None and math.isfinite(ratio_val) and ratio_val > 0 else 0.0
                        normalized_scheme.append((ratio_clean, "price", float(explicit_price)))
                        continue
                    multiplier_val = safe_float(item.get("atr") or item.get("atr_mult") or item.get("multiplier") or item.get("distance"))
                elif isinstance(item, (int, float)):
                    multiplier_val = float(item)
                    ratio_val = 1.0
                if ratio_val is None or not math.isfinite(ratio_val) or ratio_val <= 0:
                    ratio_val = 0.0
                if multiplier_val is not None and math.isfinite(multiplier_val):
                    normalized_scheme.append((float(ratio_val), "atr", float(multiplier_val)))
        if not normalized_scheme:
            if PARTIAL_TP_SCHEME:
                normalized_scheme = [(float(r), "atr", float(m)) for r, m in PARTIAL_TP_SCHEME]
            else:
                normalized_scheme = [(1.0, "atr", float(tp_mult or 1.0))]

        filtered_scheme: list[tuple[float, str, float]] = []
        for ratio_val, kind_val, val in normalized_scheme:
            ratio_clean = float(ratio_val) if ratio_val is not None and math.isfinite(ratio_val) else 0.0
            if ratio_clean <= 0:
                continue
            if kind_val not in ("atr", "price"):
                continue
            clean_val = float(val) if val is not None and math.isfinite(val) else None
            if clean_val is None:
                continue
            if clean_val <= 0:
                continue
            filtered_scheme.append((ratio_clean, kind_val, clean_val))
        if not filtered_scheme:
            filtered_scheme = [(1.0, "atr", float(tp_mult or 1.0))]

        ratio_total = sum(ratio for ratio, _, _ in filtered_scheme) or 1.0
        max_take_distance_atr_mult = safe_float(cfg.get("max_take_distance_atr_mult")) or 20.0
        max_tp_distance = float(max_take_distance_atr_mult) * float(atrv) if atrv is not None and math.isfinite(atrv) and atrv > 0 else None

        for idx, (ratio_val, kind_val, value_val) in enumerate(filtered_scheme):
            share = ratio_val / ratio_total if ratio_total else 0.0
            target_qty = qty * share if idx < len(filtered_scheme) - 1 else remaining_qty
            target_qty = min(target_qty, remaining_qty)
            if target_qty <= 0:
                continue
            try:
                target_qty_precise = float(exchange.amount_to_precision(exchange_symbol, target_qty))
            except Exception:
                target_qty_precise = float(round(target_qty, 8))
            if target_qty_precise <= 0:
                continue
            if min_amount and target_qty_precise + 1e-12 < min_amount:
                min_qty_violation = True
                continue
            if kind_val == "price":
                tp_target_price = float(value_val)
            else:
                multiplier_val = float(value_val)
                if explicit_take is not None and math.isfinite(explicit_take):
                    if idx == 0:
                        tp_target_price = explicit_take
                    else:
                        tp_target_price = explicit_take + (multiplier_val * atrv if is_long else -multiplier_val * atrv)
                else:
                    tp_target_price = reference_price + (multiplier_val * atrv if is_long else -multiplier_val * atrv)
            if tp_target_price is None or not math.isfinite(tp_target_price) or tp_target_price <= 0:
                continue
            tol = max(abs(market_ref) * 1e-4, 1e-6)
            if is_long:
                if tp_target_price <= market_ref + tol or (stop_price is not None and math.isfinite(stop_price) and tp_target_price <= stop_price + tol):
                    take_limit_errors.append(f"invalid-tp@{tp_target_price:.6f}")
                    continue
            else:
                if tp_target_price >= market_ref - tol or (stop_price is not None and math.isfinite(stop_price) and tp_target_price >= stop_price - tol):
                    take_limit_errors.append(f"invalid-tp@{tp_target_price:.6f}")
                    continue
            if max_tp_distance is not None and math.isfinite(max_tp_distance) and max_tp_distance > 0:
                if abs(tp_target_price - market_ref) > max_tp_distance:
                    take_limit_errors.append(f"tp-too-far@{tp_target_price:.6f}")
                    continue
            layer_notional = target_qty_precise * tp_target_price
            if layer_notional < MIN_NOTIONAL_USDT * 0.5:
                min_notional_violation = True
                continue
            tp_params = dict(base_params)
            tp_params["takeProfit"] = tp_target_price
            tp_params.setdefault("timeInForce", "GTC")
            try:
                exchange.create_order(
                    exchange_symbol,
                    "limit",
                    protection_side,
                    target_qty_precise,
                    tp_target_price,
                    tp_params,
                )
            except Exception as exc:
                log(f"⚠️ {symbol}: не удалось выставить тейк-профит ({target_qty_precise:.4f}@{tp_target_price:.2f}): {exc}", Fore.YELLOW)
                continue
            remaining_qty = max(0.0, remaining_qty - target_qty_precise)
            take_created.append(f"takeProfit {target_qty_precise:.4f} @ {tp_target_price:.2f}")

        take_orders_success = False
        if take_created:
            created_log_parts.extend(take_created)
            take_orders_success = True

        if not take_orders_success and take_price is not None and math.isfinite(take_price) and take_price > 0:
            fallback_take_price = float(take_price)

        fallback_qty_target = max(remaining_qty, 0.0)
        if fallback_qty_target <= 1e-9 or fallback_qty_target > qty + 1e-9:
            fallback_qty_target = qty
        if not take_orders_success and fallback_take_price:
            try:
                fallback_qty_precise = float(exchange.amount_to_precision(exchange_symbol, fallback_qty_target))
            except Exception:
                fallback_qty_precise = float(round(fallback_qty_target, 8))
            if fallback_qty_precise <= 0:
                fallback_qty_precise = fallback_qty_target
            if min_amount and fallback_qty_precise + 1e-12 < min_amount:
                min_qty_violation = True
            elif min_qty_step and fallback_qty_precise + 1e-12 < min_qty_step:
                min_qty_violation = True
            else:
                fallback_params = dict(base_params)
                fallback_params["takeProfit"] = fallback_take_price
                fallback_params.setdefault("timeInForce", "GTC")
                try:
                    exchange.create_order(
                        exchange_symbol,
                        "limit",
                        protection_side,
                        fallback_qty_precise,
                        fallback_take_price,
                        fallback_params,
                    )
                except Exception as exc:
                    take_limit_errors.append(str(exc))
                else:
                    created_log_parts.append(f"takeProfit {fallback_qty_precise:.4f} @ {fallback_take_price:.2f}")
                    take_orders_success = True
                    fallback_limit_success = True

    trailing_amount = abs(trailing_offset) if trailing_offset is not None else None
    if position_category == "spot":
        trailing_amount = None
    trailing_set = False
    take_set_via_trading_stop = False
    trailing_errors: list[str] = []
    take_trading_stop_errors: list[str] = []

    set_trading_stop_callable = getattr(exchange, "set_trading_stop", None)
    if callable(set_trading_stop_callable) and (trailing_amount or fallback_take_price):
        if position_category:
            trading_stop_params = {
                "category": position_category,
                "side": "Sell" if is_long else "Buy",
                "symbol": exchange_symbol,
            }
            if position_idx is not None:
                trading_stop_params["positionIdx"] = position_idx
            if reference_price and math.isfinite(reference_price):
                trading_stop_params["triggerPrice"] = reference_price
            if trailing_amount:
                trailing_text = f"{trailing_amount:.8f}"
                trading_stop_params["trailingStop"] = trailing_text
                trading_stop_params["trailingAmount"] = trailing_text
            if fallback_take_price:
                trading_stop_params["takeProfit"] = fallback_take_price
            try:
                set_trading_stop_callable(exchange_symbol, trading_stop_params)
                if trailing_amount:
                    created_log_parts.append(f"tradingStop trailing {trailing_amount:.4f}")
                    trailing_set = True
                if fallback_take_price:
                    created_log_parts.append(f"takeProfit set_trading_stop @ {fallback_take_price:.2f}")
                    take_set_via_trading_stop = True
                    take_orders_success = True
            except Exception as exc:
                if trailing_amount:
                    trailing_errors.append(str(exc))
                if fallback_take_price:
                    take_trading_stop_errors.append(str(exc))
        else:
            if trailing_amount:
                trailing_errors.append("market category unknown for set_trading_stop")
            if fallback_take_price:
                take_trading_stop_errors.append("market category unknown for set_trading_stop")
    elif trailing_amount or fallback_take_price:
        if trailing_amount:
            trailing_errors.append("set_trading_stop not supported by exchange")
        if fallback_take_price:
            take_trading_stop_errors.append("set_trading_stop not supported by exchange")

    if trailing_amount and not trailing_set:
        if not trailing_errors:
            trailing_errors.append("trailing stop unsupported on this market")
        combined = "; ".join(trailing_errors)
        if "set_trading_stop not supported" in combined.lower():
            log(f"ℹ️ {symbol}: trailing stop unavailable on this market (likely spot).", Fore.LIGHTBLACK_EX)
        else:
            log(f"[WARN] {symbol}: trailing stop setup failed ({combined})", Fore.YELLOW)

    forced_actions: list[str] = []
    forced_actions: list[str] = []
    forced_errors: list[str] = []
    if not take_orders_success:
        details_parts = []
        if take_trading_stop_errors:
            details_parts.append("; ".join(take_trading_stop_errors))
        if take_limit_errors:
            details_parts.append("; ".join(take_limit_errors))
        if min_qty_violation:
            details_parts.append("amount below min precision")
        if min_notional_violation:
            details_parts.append("notional below MIN_NOTIONAL_USDT")
        details = "; ".join(details_parts) if details_parts else f"scheme={filtered_scheme}"
        log(f"[WARN] {symbol}: take-profit orders were not placed ({details})", Fore.YELLOW)

        force_price = None
        for candidate in (fallback_take_price, take_price, reference_price, price):
            if candidate is not None and math.isfinite(candidate) and candidate > 0:
                force_price = float(candidate)
                break
        force_qty_target = fallback_qty_target if fallback_qty_target > 0 else qty
        if force_qty_target <= 0 and qty > 0:
            force_qty_target = qty
        if force_qty_target > 0 and force_price and math.isfinite(force_price) and force_price > 0:
            try:
                force_qty_precise = float(exchange.amount_to_precision(exchange_symbol, force_qty_target))
            except Exception:
                force_qty_precise = float(round(force_qty_target, 8))
            if force_qty_precise <= 0:
                force_qty_precise = force_qty_target
            if force_qty_precise > 0:
                force_params = dict(base_params)
                force_params.setdefault("timeInForce", "GTC")
                try:
                    exchange.create_order(
                        exchange_symbol,
                        "limit",
                        protection_side,
                        force_qty_precise,
                        force_price,
                        force_params,
                    )
                except Exception as exc_force_limit:
                    forced_errors.append(f"limit {force_qty_precise:.4f}@{force_price:.4f}: {exc_force_limit}")
                else:
                    created_log_parts.append(f"forced takeProfit {force_qty_precise:.4f} @ {force_price:.2f}")
                    forced_actions.append(f"limit {force_qty_precise:.4f}@{force_price:.2f}")
                    take_orders_success = True
        if not take_orders_success and force_qty_target > 0 and not has_stop:
            try:
                force_qty_precise = float(exchange.amount_to_precision(exchange_symbol, force_qty_target))
            except Exception:
                force_qty_precise = float(round(force_qty_target, 8))
            if force_qty_precise <= 0:
                force_qty_precise = force_qty_target
            market_params = dict(base_params)
            market_params.pop("takeProfit", None)
            market_params.setdefault("closeOnTrigger", True)
            try:
                exchange.create_order(
                    exchange_symbol,
                    "market",
                    protection_side,
                    force_qty_precise,
                    None,
                    market_params,
                )
            except Exception as exc_force_market:
                forced_errors.append(f"market {force_qty_precise:.4f}: {exc_force_market}")
            else:
                created_log_parts.append(f"forced MARKET takeProfit {force_qty_precise:.4f}")
                forced_actions.append(f"market {force_qty_precise:.4f}")
                take_orders_success = True

    if forced_actions:
        log(f"🔷 {symbol}: fallback take-profit executed ({', '.join(forced_actions)})", Fore.LIGHTBLUE_EX)
        send_tg(
            f"ℹ️ {symbol}: fallback take-profit executed\n"
            + "\n".join(f"- {entry}" for entry in forced_actions)
        )
    if forced_errors:
        log(f"[WARN] {symbol}: fallback take-profit errors ({'; '.join(forced_errors)})", Fore.YELLOW)

    # Now that new protection is set (or attempted), clean up redundant reduce-only orders
    try:
        refreshed_open = fetch_open_orders_for_symbol(exchange, symbol, limit=200)
        refreshed_reduce = [order for order in (refreshed_open or []) if isinstance(order, dict)]
        cancelled_stop_entries, cancel_stop_errors = _cleanup_redundant_stop_orders(
            exchange,
            symbol,
            refreshed_reduce,
            protection_side,
            position_qty,
            is_long,
            keep_ids_preferred=stop_ids_preferred,
        )
        if cancelled_stop_entries:
            summary = "; ".join(cancelled_stop_entries)
            log(f"📈 {symbol}: удалены лишние стоп-ордера: {summary}", Fore.LIGHTBLUE_EX)
            send_tg(f"📈 {symbol}: удалены лишние стоп-ордера: {summary}")
        if cancel_stop_errors:
            details = "; ".join(f"{descriptor} -> {err}" for descriptor, err in cancel_stop_errors)
            log(f"⚠️ {symbol}: не удалось удалить часть стоп-ордеров: {details}", Fore.YELLOW)
            send_tg(f"ℹ️ {symbol}: ошибка при удалении стоп-ордеров: {details}")
    except Exception as exc_cleanup:
        log(f"[WARN] {symbol}: cleanup of redundant stops failed: {exc_cleanup}", Fore.YELLOW)

    if created_log_parts:
        log(f"🔷 {symbol}: обновлена защита позиции {created_log_parts}", Fore.LIGHTBLUE_EX)
        send_tg(
            f"ℹ️ {symbol}: обновлена защита позиции\n"
            + "\n".join(f"- {entry}" for entry in created_log_parts)
        )
    final_open_orders = fetch_open_orders_for_symbol(exchange, symbol)
    protective_after = _extract_protection_orders(final_open_orders)
    has_stop_after = any(
        _has_stop_flag(order) or _has_trailing_flag(order) for order in protective_after
    )
    if not has_stop_after and stop_price and math.isfinite(stop_price):
        log(f"[WARN] {symbol}: стоп-ордера не обнаружены после очистки, повторная установка", Fore.YELLOW)
        fallback_params = dict(base_params)
        fallback_params.update(
            {
                "triggerPrice": stop_price,
                "triggerDirection": get_trigger_direction_for_side(
                    protection_side, trigger_price=stop_price, reference_price=price
                ),
                "closeOnTrigger": True,
            }
        )
        try:
            retry_order = exchange.create_order(
                exchange_symbol, "market", protection_side, qty, None, fallback_params
            )
            if isinstance(retry_order, dict):
                retry_id = (
                    retry_order.get("id")
                    or (retry_order.get("info") or {}).get("orderId")
                    or (retry_order.get("info") or {}).get("orderID")
                )
                if retry_id:
                    stop_ids_preferred.add(str(retry_id))
            log(f"🔁 {symbol}: стоп-ордер восстановлен повторно @ {stop_price:.2f}", Fore.LIGHTBLUE_EX)
        except Exception as exc_retry:
            log(f"⚠️ {symbol}: повторная установка стопа не удалась: {exc_retry}", Fore.YELLOW)
        final_open_orders = fetch_open_orders_for_symbol(exchange, symbol)
    return final_open_orders




def _prepare_protection_dataframe(
    df_candidate: pd.DataFrame | None,
    df_primary: pd.DataFrame | None,
    symbol: str,
) -> pd.DataFrame | None:
    candidate = None
    if isinstance(df_candidate, pd.DataFrame) and not df_candidate.empty:
        candidate = df_candidate.copy()
    elif isinstance(df_primary, pd.DataFrame) and not df_primary.empty:
        candidate = df_primary.copy()
    if candidate is None:
        return None
    if "atr" not in candidate.columns:
        try:
            candidate["atr"] = atr(candidate, 14)
        except Exception as exc:
            log(
                f"[WARN] {symbol}: failed to prepare ATR for protection refresh: {exc}",
                Fore.YELLOW,
            )
            return None
    return candidate


def _refresh_position_protection_if_possible(
    exchange,
    symbol: str,
    position: dict[str, Any] | None,
    df_candidate: pd.DataFrame | None,
    df_primary: pd.DataFrame | None,
    open_orders,
    symbol_meta: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]] | None, bool]:
    if position is None:
        return None, False
    protection_df = _prepare_protection_dataframe(df_candidate, df_primary, symbol)
    if protection_df is None:
        return None, False
    updated_orders = ensure_position_protection(
        exchange,
        symbol,
        position,
        protection_df,
        open_orders,
        config=symbol_meta,
    )
    has_stop = False
    has_take = False
    if updated_orders:
        has_stop, has_take, _ = _evaluate_position_protection(position, updated_orders or [])
    if not has_take:
        log(f"[WARN] {symbol}: protection refresh left position without take-profit, retrying once", Fore.YELLOW)
        updated_orders = ensure_position_protection(
            exchange,
            symbol,
            position,
            protection_df,
            updated_orders,
            config=symbol_meta,
        )
        _, has_take, _ = _evaluate_position_protection(position, updated_orders or [])
        if not has_take:
            log(f"[WARN] {symbol}: still no take-profit after retry; monitor manually", Fore.YELLOW)
    return updated_orders, True


def _format_decimal(value: numbers.Real, precision: int = 6) -> str:
    try:
        text = f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)
    text = text.rstrip("0").rstrip(".")
    if text in {"", "-"}:
        return "0"
    if text == "-0":
        return "0"
    return text


def _format_notional_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(value)
    if 0.0 <= numeric <= 1.0:
        return f"{numeric * 100:.1f}%"
    return f"{numeric:.3f}"


def _summarize_order_spec(order_dict: dict[str, Any] | None) -> str:
    if not isinstance(order_dict, dict):
        return str(order_dict)
    side_label = (order_dict.get("side") or "?").upper()
    order_type = (
        order_dict.get("type")
        or order_dict.get("orderType")
        or order_dict.get("ordType")
        or order_dict.get("category")
        or "order"
    )
    order_type_label = str(order_type).upper()
    quantity_val: Any = None
    for key in ("amount", "contracts", "qty", "quantity", "size", "volume"):
        val = order_dict.get(key)
        if val not in (None, ""):
            quantity_val = val
            break
    if isinstance(quantity_val, numbers.Real):
        qty_text = _format_decimal(quantity_val, precision=6)
    else:
        qty_text = str(quantity_val) if quantity_val not in (None, "") else ""
    price_val: Any = None
    for key in ("price", "triggerPrice", "stopPrice", "stop_price", "takeProfit", "stopLoss"):
        val = order_dict.get(key)
        if val not in (None, ""):
            price_val = val
            break
    if isinstance(price_val, numbers.Real):
        price_text = _format_decimal(price_val, precision=6)
    else:
        price_text = str(price_val) if price_val not in (None, "") else ""
    flags: list[str] = []
    if _is_truthy_flag(order_dict.get("reduceOnly")):
        flags.append("reduce")
    if _is_truthy_flag(order_dict.get("closePosition")):
        flags.append("close")
    if _is_truthy_flag(order_dict.get("scaleIn") or order_dict.get("ladder")):
        flags.append("scale")
    if _is_truthy_flag(order_dict.get("postOnly")):
        flags.append("post")
    if _is_truthy_flag(order_dict.get("hidden")):
        flags.append("hidden")
    intent_val = order_dict.get("intent") or order_dict.get("tag") or order_dict.get("note") or order_dict.get("comment")
    parts: list[str] = []
    header = f"{side_label} {order_type_label}".strip()
    if header:
        parts.append(header)
    if qty_text:
        parts.append(qty_text)
    if price_text:
        parts.append(f"@ {price_text}")
    if flags:
        parts.append(f"[{' '.join(flags)}]")
    if intent_val:
        parts.append(f"({intent_val})")
    return " ".join(parts) if parts else str(order_dict)


def execute_extra_orders(
    exchange,
    symbol,
    orders,
    equity: float | None = None,
    current_position=None,
    open_orders=None,
    available_margin: float | None = None,
    symbol_leverage: float | None = None,
    max_limits_per_side: int = 1,
):
    executed = []
    exchange_symbol = _resolve_symbol_alias(symbol) or symbol
    open_orders = open_orders or []

    equity_value = safe_float(equity)
    available_margin_value = safe_float(available_margin)
    margin_ratio = (
        (available_margin_value / equity_value)
        if equity_value and available_margin_value and equity_value != 0
        else None
    )

    def _summarize_position(payload):
        if not payload:
            return None
        size_val = safe_float(payload.get("amount"))
        entry_val = safe_float(payload.get("entryPrice"))
        notional = None
        if size_val is not None and entry_val is not None:
            notional = abs(size_val * entry_val)
        size_pct = (
            (notional / equity_value)
            if notional is not None and equity_value and equity_value != 0
            else None
        )
        return {
            "side": payload.get("side"),
            "size_pct": size_pct,
            "entry_price": entry_val,
            "unrealized_pnl": payload.get("unrealizedPnl"),
            "has_position": bool(size_val),
        }

    raw_position_payload = current_position if isinstance(current_position, dict) else None
    position_summary = _summarize_position(raw_position_payload)
    position_side = ((current_position or {}).get("side") or "").lower()
    # Determine market category (spot/linear/inverse)
    try:
        _market_info = exchange.market(exchange_symbol)
    except Exception:
        _market_info = None
    category = _infer_market_category(exchange_symbol, _market_info) or "linear"
    set_trading_stop_callable = getattr(exchange, "set_trading_stop", None)
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
    order_errors: list[str] = []

    last_price_snapshot: float | None = None
    last_price_checked = False

    def _resolve_market_price() -> float | None:
        nonlocal last_price_snapshot, last_price_checked
        if last_price_snapshot is not None or last_price_checked:
            return last_price_snapshot
        last_price_checked = True
        try:
            ticker = exchange.fetch_ticker(exchange_symbol)
        except Exception:
            ticker = None
        if isinstance(ticker, dict):
            last_price = ticker.get("last") or ticker.get("close")
            if last_price is None:
                info = ticker.get("info")
                if isinstance(info, dict):
                    last_price = info.get("lastPrice") or info.get("markPrice")
            last_price_snapshot = safe_float(last_price)
        return last_price_snapshot

    def resolve_reference_price(order_dict, fallback_price):
        candidates = [
            fallback_price,
            order_dict.get("referencePrice"),
            order_dict.get("currentPrice"),
            order_dict.get("lastPrice"),
            order_dict.get("triggerReferencePrice"),
            (current_position or {}).get("entryPrice"),
            (current_position or {}).get("avgEntryPrice"),
            (current_position or {}).get("markPrice"),
            (current_position or {}).get("lastPrice"),
        ]
        for candidate in candidates:
            ref = safe_float(candidate)
            if ref is not None and math.isfinite(ref):
                return ref
        market_price = _resolve_market_price()
        if market_price is not None and math.isfinite(market_price):
            return market_price
        return None

    for idx, order in enumerate(orders, 1):
        if not isinstance(order, dict):
            log(f"[WARN] Extra order #{idx} for {symbol} is not a dict; skipping.", Fore.YELLOW)
            continue
        status_value = order.get("status")
        order_id_value = order.get("id")
        if status_value is not None and order_id_value:
            order = dict(order)
            existing_oid = str(order_id_value)
            success, err = cancel_order_by_id(exchange, symbol, existing_oid)
            order.pop("status", None)
            order.pop("id", None)
            if success:
                log(f"[INFO] Cancelled existing order {existing_oid} for {symbol} (AI replacement)", Fore.LIGHTBLUE_EX)
            else:
                log(f"[WARN] Failed to cancel existing order {existing_oid} for {symbol}: {err}", Fore.YELLOW)
                continue
        raw_type = (
            order.get("type")
            or order.get("orderType")
            or order.get("order_type")
            or order.get("ccxt_type")
        )
        order_type_key = normalize_order_type_key(raw_type)
        params = dict(order.get("params") or {})
        for cleanup_key in ("orderType", "order_type", "type", "ccxt_type"):
            params.pop(cleanup_key, None)
        note = order.get("note") or order.get("comment") or ""
        reduce_only_flag = _is_truthy_flag(order.get("reduceOnly"))
        if reduce_only_flag:
            params["reduceOnly"] = True
        elif "reduceOnly" in params:
            params["reduceOnly"] = bool(params["reduceOnly"])
        is_reduce_only = _is_truthy_flag(params.get("reduceOnly"))
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
        side_raw = order.get("side")
        reduce_direction = None
        if is_reduce_only:
            position_side_field = str((current_position or {}).get("side") or "").lower()
            if position_side_field in {"long", "buy"}:
                reduce_direction = "sell"
            elif position_side_field in {"short", "sell"}:
                reduce_direction = "buy"
            if reduce_direction is None:
                pos_amount_val = safe_float(
                    (current_position or {}).get("amount")
                    or (current_position or {}).get("contracts")
                    or (current_position or {}).get("size")
                )
                if pos_amount_val is not None and math.isfinite(pos_amount_val) and abs(pos_amount_val) > 0:
                    reduce_direction = "sell" if pos_amount_val > 0 else "buy"
        if reduce_direction:
            side_raw = reduce_direction
        side, autodetected_side = _normalize_order_side(side_raw, amount)
        if autodetected_side:
            log(
                f"[INFO] Normalized side for extra order #{idx} {symbol} to {side.upper()} (source={side_raw!r})",
                Fore.LIGHTBLACK_EX,
            )
        if side not in {"buy", "sell"}:
            log(f"[WARN] Missing valid side in extra order #{idx} for {symbol}; skipping.", Fore.YELLOW)
            continue
        price = safe_float(order.get("price"))
        # Allow MARKET orders without a valid price; strip price instead of skipping
        is_market_order = (order_type_key == "market") or (str(params.get("orderType") or "").lower() == "market")
        if price is not None and (not math.isfinite(price) or price <= 0):
            if is_market_order:
                # Remove price for MARKET; Bybit/ccxt ignores it for market orders
                order.pop("price", None)
                params.pop("price", None)
                price = None
            else:
                log(f"[WARN] Invalid price in extra order #{idx} for {symbol}; skipping.", Fore.YELLOW)
                continue
        price_key = round(price, NON_REDUCE_PRICE_DECIMALS) if price is not None else None
        trigger_price = safe_float(
            order.get("triggerPrice")
            or order.get("stopPrice")
            or order.get("stop_price")
            or params.get("triggerPrice")
            or params.get("stopPrice")
            or params.get("stop_price")
        )
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
        pending_reduce_cancels: list[dict[str, Any]] = []
        if is_reduce_only:
            if abs(position_amount) == 0:
                log(f"[INFO] Skipping reduce-only order for {symbol}: no active position", Fore.LIGHTBLACK_EX)
                send_tg(f"[INFO] {symbol}: reduce-only order skipped (flat position)")
                continue
            existing_list = reduce_only_map.get(side)
            if existing_list:
                pending_reduce_cancels = list(existing_list)
                reduce_only_map[side] = []
        position_idx = order.get("positionIdx")
        if position_idx is None:
            position_idx = get_position_idx(side)
        if position_idx is not None:
            params["positionIdx"] = position_idx
        else:
            params.pop("positionIdx", None)
        if order_type_key == "trailing_stop":
            if abs(position_amount) == 0:
                log(f"[INFO] Skipping trailing-stop request for {symbol}: no active position", Fore.LIGHTBLACK_EX)
                continue
            if category == "spot":
                log(f"[INFO] Skipping trailing-stop request for {symbol}: spot markets do not support exchange trailing.", Fore.LIGHTBLACK_EX)
                continue
            if not callable(set_trading_stop_callable):
                log(f"[WARN] Trailing stop requested for {symbol}, but exchange adapter lacks set_trading_stop.", Fore.YELLOW)
                continue
            trailing_amount = safe_float(
                order.get("trailingStop")
                or order.get("trailingAmount")
                or order.get("trailing_stop")
                or params.get("trailingStop")
                or params.get("trailingAmount")
            )
            trailing_percent = safe_float(order.get("trailingPercent") or order.get("trailing_percent"))
            trailing_callback = safe_float(order.get("trailingCallback") or order.get("trailing_callback"))
            reference_price = resolve_reference_price(order, price)
            trigger_ref = trigger_price if trigger_price is not None else reference_price
            if trailing_amount is None and trailing_percent is not None and math.isfinite(trailing_percent):
                ref_for_percent = reference_price if reference_price and math.isfinite(reference_price) else trigger_price
                if ref_for_percent and math.isfinite(ref_for_percent):
                    trailing_amount = abs(ref_for_percent) * (trailing_percent / 100.0)
            if trailing_amount is None and trailing_callback is not None and math.isfinite(trailing_callback):
                trailing_amount = abs(trailing_callback)
            take_profit_value = safe_float(
                order.get("takeProfit")
                or order.get("tp")
                or order.get("take_profit")
                or params.get("takeProfit")
                or params.get("tp")
            )
            if trailing_amount is None and take_profit_value is None:
                log(f"[WARN] Trailing-stop request for {symbol} missing trailing offset/take-profit; skipping.", Fore.YELLOW)
                continue
            trading_stop_params = {
                "category": category,
                "side": "Sell" if side == "sell" else "Buy",
                "symbol": exchange_symbol,
            }
            if position_idx is not None:
                trading_stop_params["positionIdx"] = position_idx
            if trigger_ref is not None and math.isfinite(trigger_ref):
                trading_stop_params["triggerPrice"] = trigger_ref
            if trailing_amount is not None and math.isfinite(trailing_amount) and trailing_amount > 0:
                trailing_str = f"{abs(trailing_amount):.8f}"
                trading_stop_params["trailingStop"] = trailing_str
                trading_stop_params["trailingAmount"] = trailing_str
            elif trailing_amount is not None:
                log(f"[WARN] Trailing-stop request for {symbol} has invalid trailing amount ({trailing_amount}); skipping.", Fore.YELLOW)
                continue
            if take_profit_value is not None and math.isfinite(take_profit_value) and take_profit_value > 0:
                trading_stop_params["takeProfit"] = take_profit_value
            try:
                set_trading_stop_callable(exchange_symbol, trading_stop_params)
            except Exception as exc:
                log(f"[WARN] Failed to set trailing stop for {symbol}: {exc}", Fore.YELLOW)
                continue
            desc_parts = ["TRADING-STOP", side.upper()]
            if "trailingStop" in trading_stop_params:
                desc_parts.append(f"trail={trading_stop_params['trailingStop']}")
            if "takeProfit" in trading_stop_params:
                desc_parts.append(f"tp={trading_stop_params['takeProfit']}")
            executed.append(" ".join(desc_parts))
            log(f"[INFO] Applied trailing stop for {symbol}: {' '.join(desc_parts[1:])}", Fore.LIGHTBLUE_EX)
            continue
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
            if order_type_key == "take_profit":
                params.setdefault("reduceOnly", True)
                params.setdefault("timeInForce", params.get("timeInForce") or "GTC")
                params.pop("takeProfit", None)
                params.pop("take_profit", None)
                params.pop("tp", None)
                ccxt_type = "limit"
            elif order_type_key in {"stop_loss", "stop"}:
                if trigger_price is None or not math.isfinite(trigger_price):
                    log(f"[WARN] Missing triggerPrice for extra order #{idx} {symbol}", Fore.YELLOW)
                    continue
                ccxt_type = "market"
                price = None
                params.setdefault("reduceOnly", True)
                params["triggerPrice"] = trigger_price
                reference_price = _resolve_market_price() if is_reduce_only else resolve_reference_price(order, price)
                params["triggerDirection"] = get_trigger_direction_for_side(
                    side,
                    trigger_price=trigger_price,
                    reference_price=reference_price,
                )
                params.setdefault("closeOnTrigger", True)
                params.pop("stopLoss", None)
                params.pop("stopPrice", None)
                params.pop("stop_price", None)
            elif order_type_key == "stop_limit":
                if trigger_price is None or not math.isfinite(trigger_price):
                    log(f"[WARN] Missing triggerPrice for extra order #{idx} {symbol}", Fore.YELLOW)
                    continue
                if price is None:
                    log(f"[WARN] Missing price for extra order #{idx} (stop-limit) {symbol}", Fore.YELLOW)
                    continue
                ccxt_type = "limit"
                params.setdefault("reduceOnly", True)
                params["triggerPrice"] = trigger_price
                reference_price = _resolve_market_price() if is_reduce_only else resolve_reference_price(order, price)
                params["triggerDirection"] = get_trigger_direction_for_side(
                    side,
                    trigger_price=trigger_price,
                    reference_price=reference_price,
                )
                params.pop("stopLoss", None)
                params.pop("stopPrice", None)
                params.pop("stop_price", None)
            elif trigger_price is not None and math.isfinite(trigger_price):
                params.setdefault("triggerPrice", trigger_price)
                reference_price = _resolve_market_price() if is_reduce_only else resolve_reference_price(order, price)
                params["triggerDirection"] = get_trigger_direction_for_side(
                    side,
                    trigger_price=trigger_price,
                    reference_price=reference_price,
                )
        if ccxt_type not in VALID_ORDER_TYPES:
            fallback_type = "limit" if price is not None else "market"
            log(
                f"[WARN] Unsupported order type '{raw_type}' for extra order #{idx} {symbol}, falling back to {fallback_type}",
                Fore.YELLOW,
            )
            ccxt_type = fallback_type
        allowed_types = {"limit", "market", "trailingStop"}
        if category == "spot" and "trailingStop" in allowed_types:
            allowed_types.remove("trailingStop")
        if ccxt_type not in allowed_types:
            fallback_type = "limit" if price is not None else "market"
            log(
                f"[WARN] Adjusting unsupported order type '{ccxt_type}' for extra order #{idx} {symbol} to {fallback_type}",
                Fore.YELLOW,
            )
            ccxt_type = fallback_type
        if ccxt_type == "limit" and price is None:
            log(f"[WARN] Missing price for extra order #{idx} ({ccxt_type}) {symbol}", Fore.YELLOW)
            continue
        if ccxt_type == "market":
            price = None
        if ccxt_type in {"limit", "market"}:
            params["orderType"] = ccxt_type.capitalize()
        # Sanitize params for category and set category for CCXT/Bybit v5
        params = _sanitize_order_params_for_category(params, category)
        if category == "spot":
            # Prevent spot short attempts and check balances
            if side == "sell":
                ok, err = _spot_funds_sufficient(exchange, exchange_symbol, side, amount, price)
                if not ok:
                    log(f"[WARN] Skipping spot SELL for {symbol}: {err}", Fore.YELLOW)
                    continue
            elif side == "buy" and price is not None:
                ok, err = _spot_funds_sufficient(exchange, exchange_symbol, side, amount, price)
                if not ok:
                    log(f"[WARN] Skipping spot BUY for {symbol}: {err}", Fore.YELLOW)
                    continue
        if (
            is_reduce_only
            and ccxt_type == "market"
            and order_type_key not in {"partial_close"}
        ):
            effective_trigger = safe_float(
                params.get("triggerPrice")
                or params.get("stopLossPrice")
                or params.get("takeProfitPrice")
            )
            if effective_trigger is None:
                log(
                    f"[WARN] Skipping reduce-only market order without trigger for {symbol} (extra #{idx})",
                    Fore.YELLOW,
                )
                continue
        order_created = False
        try:
            order_id = exchange.create_order(exchange_symbol, ccxt_type, side, amount, price, params)
            order_created = True
            if order_type_key in {"stop_loss", "stop"}:
                display_type = "STOP-MARKET"
            elif order_type_key == "stop_limit":
                display_type = "STOP-LIMIT"
            elif order_type_key == "take_profit":
                display_type = "TAKE-PROFIT"
            else:
                display_type = ccxt_type.upper()
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
            err_text = str(e)
            text_lower = err_text.lower()
            if "current position is zero" in text_lower or "110017" in text_lower:
                log(
                    f"[INFO] Extra order #{idx} for {symbol} failed due to zero position; skipping reduce-only placement.",
                    Fore.LIGHTBLACK_EX,
                )
            else:
                order_errors.append(err_text)
                log(f"[ERROR] Extra order #{idx} for {symbol} failed: {err_text}", Fore.RED)
        else:
            if order_created and pending_reduce_cancels:
                for existing_order in pending_reduce_cancels:
                    oid = existing_order.get("id")
                    if not oid:
                        continue
                    order_summary = _summarize_order_spec(existing_order)
                    summary_suffix = f": {order_summary}" if order_summary else ""
                    success, err = cancel_order_by_id(exchange, symbol, str(oid))
                    if success:
                        cancelled_entry = f"{oid}{summary_suffix}"
                        cancelled_success.append(cancelled_entry)
                        trigger_val = safe_float(
                            (existing_order or {}).get("stopPrice")
                            or (existing_order or {}).get("triggerPrice")
                            or (existing_order or {}).get("stopLoss")
                        )
                        price_val = safe_float((existing_order or {}).get("price"))
                        tp_val = safe_float((existing_order or {}).get("takeProfit") or (existing_order or {}).get("tp"))
                        level_bits: list[str] = []
                        if trigger_val is not None and math.isfinite(trigger_val):
                            level_bits.append(f"trigger={trigger_val:.6f}")
                        if tp_val is not None and math.isfinite(tp_val):
                            level_bits.append(f"tp={tp_val:.6f}")
                        if price_val is not None and math.isfinite(price_val):
                            level_bits.append(f"price={price_val:.6f}")
                        level_suffix = f" ({', '.join(level_bits)})" if level_bits else ""
                        log(
                            f"[INFO] Cancelled existing reduce-only order {oid} for {symbol}{summary_suffix}{level_suffix}",
                            Fore.LIGHTBLUE_EX,
                        )
                    else:
                        cancel_errors.append((oid, err))
                        log(f"[WARN] Failed to cancel reduce-only order {oid} for {symbol}{summary_suffix}: {err}", Fore.YELLOW)
    if cancelled_success:
        send_tg(f"[INFO] {symbol}: cancelled reduce-only orders {', '.join(cancelled_success)}")
    if cancel_errors:
        errs = "; ".join(f"{oid}: {err}" for oid, err in cancel_errors)
        send_tg(f"[WARN] {symbol}: errors cancelling orders - {errs}")
    actions_performed = bool(executed or cancelled_success or cancel_errors)
    return executed, actions_performed, order_errors

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
    initial_decision=None,
    *,
    priority_symbol: bool = False,
):
    df_30m = df_primary
    extra_context = extra_context or {}
    target_meta = target_meta or {}
    client = _create_ai_client(timeout=30, context=f"{symbol} decision")
    if client is None:
        _emit_ai_offline_notice(f"client init failed ({symbol} decision)")
        log(f"❌ Не удалось создать AI-клиент для {symbol}", Fore.RED)
        return None
    df_30m["ema20"] = ema(df_30m["close"],20)
    df_30m["ema50"] = ema(df_30m["close"],50)
    df_30m["rsi"] = rsi(df_30m["close"],14)
    df_30m["atr"] = atr(df_30m,14)
    portfolio_guidance = dict(target_meta) if isinstance(target_meta, dict) else {}
    extra_context_payload = extra_context if isinstance(extra_context, dict) else {}
    news_payload_payload = news_payload if isinstance(news_payload, dict) else None

    primary_initial_tf = (
        normalize_requested_timeframe(AI_INITIAL_TIMEFRAMES[0])
        if AI_INITIAL_TIMEFRAMES
        else normalize_requested_timeframe(TIMEFRAME or "30m")
    )
    normalized_initial_tfs: list[str] = []
    for tf in AI_INITIAL_TIMEFRAMES:
        normalized_tf = normalize_requested_timeframe(tf)
        if normalized_tf and normalized_tf not in normalized_initial_tfs:
            normalized_initial_tfs.append(normalized_tf)
    if not normalized_initial_tfs and primary_initial_tf:
        normalized_initial_tfs.append(primary_initial_tf)
    initial_frames_data: dict[str, list[dict[str, Any]]] = {}
    indicator_columns_by_tf: dict[str, list[str]] = {}
    for tf in normalized_initial_tfs:
        df_tf = _load_initial_dataframe(exchange, symbol, tf, df_primary, primary_initial_tf or tf)
        if df_tf is None or df_tf.empty:
            continue
        indicator_columns = _ensure_initial_indicators(
            df_tf,
            AI_INITIAL_INDICATOR_POOL,
            AI_INITIAL_EMA_COUNT,
            AI_INITIAL_EXTRA_INDICATOR_COUNT,
        )
        indicator_columns_by_tf[tf] = indicator_columns
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in indicator_columns:
            if col not in columns:
                columns.append(col)
        depth = max(
            1,
            int(
                AI_INITIAL_TF_DEPTHS.get(
                    tf,
                    AI_INITIAL_TF_DEPTHS.get(
                        primary_initial_tf or tf,
                        int(DEFAULT_CONTEXT_30M),
                    ),
                )
            ),
        )
        trimmed = df_tf.tail(depth)
        selected_columns = [col for col in columns if col in trimmed.columns]
        if not selected_columns:
            continue
        initial_frames_data[tf] = _serialize_df(trimmed[selected_columns])
    if not initial_frames_data and isinstance(df_30m, pd.DataFrame) and not df_30m.empty:
        trimmed = df_30m.tail(int(DEFAULT_CONTEXT_30M))
        selected = [col for col in ["timestamp", "open", "high", "low", "close", "volume"] if col in trimmed.columns]
        initial_frames_data[primary_initial_tf] = _serialize_df(trimmed[selected])
    higher_tf = initial_frames_data.get("4h", [])
    missing_initial_tfs = [tf for tf in normalized_initial_tfs if tf not in initial_frames_data]
    if missing_initial_tfs:
        log(
            f"[WARN] {symbol}: initial timeframe(s) unavailable {','.join(missing_initial_tfs)}",
            Fore.YELLOW,
        )
    context_counts = {
        tf: len(initial_frames_data.get(tf, []))
        for tf in normalized_initial_tfs
    }
    if not context_counts and primary_initial_tf:
        context_counts[primary_initial_tf] = len(initial_frames_data.get(primary_initial_tf, []))

    trim_order = list(normalized_initial_tfs)
    if not trim_order and primary_initial_tf:
        trim_order.append(primary_initial_tf)
    min_context_by_tf: dict[str, int] = {}
    step_context_by_tf: dict[str, int] = {}
    if trim_order:
        first_tf = trim_order[0]
        min_context_by_tf[first_tf] = MIN_CONTEXT_30M
        step_context_by_tf[first_tf] = CONTEXT_STEP_30M
        if len(trim_order) > 1:
            second_tf = trim_order[1]
            min_context_by_tf[second_tf] = MIN_CONTEXT_4H
            step_context_by_tf[second_tf] = CONTEXT_STEP_4H
        for extra_tf in trim_order[2:]:
            min_context_by_tf.setdefault(extra_tf, MIN_CONTEXT_4H)
            step_context_by_tf.setdefault(extra_tf, CONTEXT_STEP_4H)

    for tf_name in trim_order:
        context_counts.setdefault(tf_name, len(initial_frames_data.get(tf_name, [])))

    active_indicator_columns_by_tf: dict[str, list[str]] = {
        tf_name: list(cols) for tf_name, cols in indicator_columns_by_tf.items()
    }
    indicator_priority: list[str] = list(indicator_columns_by_tf.get(primary_initial_tf, []))
    if not indicator_priority:
        for cols in indicator_columns_by_tf.values():
            if cols:
                indicator_priority = list(cols)
                break
    indicator_trim_index = len(indicator_priority)
    MIN_INDICATORS_PER_TF = 2

    base_rows = initial_frames_data.get(primary_initial_tf) or higher_tf
    latest_row = base_rows[-1] if base_rows else None
    regime_mode, regime_metrics = detect_regime(df_30m, higher_tf)
    market_mode_label = _regime_to_market_mode(regime_mode)
    regime_metrics["market_mode"] = market_mode_label
    analysis_balance = _resolve_analysis_balance(regime_metrics.get("atr_ratio"))
    regime_metrics["analysis_balance"] = analysis_balance
    mode_config = MARKET_MODE_CONFIG.get(market_mode_label, {})
    sl_mult_local = max(0.05, SL_ATR * (mode_config.get("sl_scale") or 1.0))
    tp_mult_local = TP_ATR
    tp_ratio = mode_config.get("tp_ratio")
    if tp_ratio and tp_ratio > 0:
        tp_mult_local = max(0.05, sl_mult_local * tp_ratio)
    notional_scale = max(0.05, mode_config.get("risk_multiplier", 1.0))
    mode_confidence_required = mode_config.get("confidence_required")

    cfg_local = target_meta.setdefault("config", {}) if isinstance(target_meta, dict) else {}
    cfg_local["sl_atr"] = sl_mult_local
    cfg_local["tp_atr"] = tp_mult_local
    cfg_local["market_mode"] = market_mode_label
    if mode_confidence_required:
        cfg_local["confidence_required"] = mode_confidence_required

    base_notional = (
        target_meta.get("notional_pct")
        or (target_meta.get("target") or {}).get("notional_pct")
    )
    if base_notional is not None:
        try:
            scaled_notional = float(base_notional) * notional_scale
            if math.isfinite(scaled_notional) and scaled_notional > 0:
                target_meta["notional_pct"] = scaled_notional
        except (TypeError, ValueError):
            pass

    if latest_row:
        columns_to_log = ["open", "high", "low", "close", "volume"]
        columns_to_log.extend(indicator_columns_by_tf.get(primary_initial_tf, []))
        def _fmt(val):
            try:
                return f"{float(val):.4f}"
            except (TypeError, ValueError):
                return "n/a"
        entries = ", ".join(
            f"{col}={_fmt(latest_row.get(col))}"
            for col in columns_to_log
            if col in latest_row
        )
        log(
            f"[INFO] {symbol}: initial context {primary_initial_tf} {entries}; "
            f"regime={regime_mode} mode={market_mode_label} sl_atr={sl_mult_local:.2f} tp_atr={tp_mult_local:.2f} "
            f"notional_scale={notional_scale:.2f} analysis={analysis_balance.get('mode')} "
            f"(news={analysis_balance.get('news_weight')} ta={analysis_balance.get('technical_weight')})",
            Fore.LIGHTBLACK_EX,
        )

    guide_modes = (MARKET_MODE_GUIDE.get("market_modes") or {}) if isinstance(MARKET_MODE_GUIDE, dict) else {}
    SYMBOL_MARKET_MODE_HINTS[symbol] = {
        "mode": market_mode_label,
        "confidence_required": mode_confidence_required,
        "needs_confirmation": (guide_modes.get(market_mode_label, {}) or {}).get("needs_confirmation"),
        "analysis_balance": analysis_balance,
    }
    active_tf_indicators = {
        tf: active_indicator_columns_by_tf.get(tf, [])
        for tf in (trim_order or [])[:2]
    }
    if active_tf_indicators:
        parts = []
        for tf, cols in active_tf_indicators.items():
            if cols:
                parts.append(f"{tf}:[{','.join(cols)}]")
        if parts:
            log(
                f"[INFO] {symbol}: prompt indicators {', '.join(parts)}",
                Fore.LIGHTBLACK_EX,
            )

    higher_trend_bias = None
    higher_trend_label = "неопределён"
    if regime_metrics.get("higher_ema20") is not None and regime_metrics.get("higher_ema50") is not None:
        ema20_ht = regime_metrics.get("higher_ema20")
        ema50_ht = regime_metrics.get("higher_ema50")
        if ema20_ht > ema50_ht:
            higher_trend_bias = "buy"
            higher_trend_label = "восходящий"
        elif ema20_ht < ema50_ht:
            higher_trend_bias = "sell"
            higher_trend_label = "нисходящий"
    def fallback_due_to(trigger: str, extra_notes: dict[str, Any] | None = None) -> dict[str, Any] | None:
        fallback_decision = _fallback_momentum_decision(symbol, df_30m, higher_trend_bias, current_position)
        if not fallback_decision:
            return None
        base_reason = fallback_decision.get("reason") or "fallback strategy engaged"
        fallback_decision["reason"] = f"{base_reason} (fallback trigger: {trigger})"
        notes_payload: dict[str, Any] = {}
        existing_notes = fallback_decision.get("notes")
        if isinstance(existing_notes, dict):
            notes_payload.update(existing_notes)
        notes_payload.setdefault("fallback_trigger", trigger)
        if extra_notes:
            for key, value in extra_notes.items():
                if key not in notes_payload:
                    notes_payload[key] = value
        fallback_decision["notes"] = notes_payload
        return fallback_decision
    current_context = {}
    raw_position_payload = None
    equity_value = safe_float(equity)
    available_margin_value = safe_float(available_margin)
    margin_ratio = (
        (available_margin_value / equity_value)
        if equity_value and available_margin_value and equity_value != 0
        else None
    )
    if current_position:
        raw_position_payload = {
            "side": current_position.get("side"),
            "amount": current_position.get("amount"),
            "entryPrice": current_position.get("entryPrice"),
            "leverage": current_position.get("leverage"),
            "unrealizedPnl": current_position.get("unrealizedPnl"),
            "liquidationPrice": current_position.get("liquidationPrice"),
        }
    open_orders = open_orders or []
    def _summarize_position_for_prompt(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not payload:
            return None
        size_val = safe_float(payload.get("amount"))
        entry_val = safe_float(payload.get("entryPrice"))
        notional = None
        if size_val is not None and entry_val is not None:
            notional = abs(size_val * entry_val)
        size_pct = (
            (notional / equity_value)
            if notional is not None and equity_value and equity_value != 0
            else None
        )
        return {
            "side": payload.get("side"),
            "size_pct": size_pct,
            "entry_price": entry_val,
            "unrealized_pnl": payload.get("unrealizedPnl"),
            "has_position": bool(size_val),
        }
    position_summary = _summarize_position_for_prompt(raw_position_payload)

    # >>>>>>>>>>>> system_msg обновлён <<<<<<<<<<<<
    system_msg = (
        "You are an intraday multi-symbol trading assistant. "
        "You receive structured context only (OHLCV per timeframe, derived indicators, sentiment, regime hints, risk settings, open orders, anonymized position summary, and optional news items). "
        "Decide whether to open, close, manage, or skip positions and describe any protective orders required. "
        "You trade perpetual futures and may hold long or short exposure per symbol. If you expect a coin to fall, favor short entries; if you expect it to rise, favor longs. When you already hold a long and expect further upside, you may add to the position; when holding a short and expecting a bounce, reduce or close it (or flip to long). Mirror this logic when holding a long but expecting weakness—trim, close, or flip to short. "
        "When more data is necessary, reply with a needs array listing exact gaps (timeframes, indicators, news categories, etc.). "
        "Always answer with a strict JSON object that follows the declared response_format schema."
    )


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

    selected_indicator_names = indicator_columns_by_tf.get(primary_initial_tf, [])
    log(
        f"[INFO] {symbol}: regime={regime_mode} mode={market_mode_label} indicators={','.join(selected_indicator_names) or 'none'}",
        Fore.LIGHTBLACK_EX,
    )

    primary_context_key = trim_order[0] if trim_order else (primary_initial_tf or None)
    secondary_context_key = (
        trim_order[1] if len(trim_order) > 1 else ("4h" if "4h" in initial_frames_data else None)
    )
    if primary_context_key and primary_context_key not in context_counts:
        context_counts[primary_context_key] = len(initial_frames_data.get(primary_context_key, []))
    if secondary_context_key and secondary_context_key not in context_counts:
        context_counts[secondary_context_key] = len(initial_frames_data.get(secondary_context_key, []))

    def build_context(position_summary=position_summary):
        def _slice_context(tf_name: str, limit: int) -> list[dict[str, Any]]:
            if not tf_name or limit <= 0:
                return []
            entries = initial_frames_data.get(tf_name, [])
            if not entries:
                return []
            active_cols = {
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }
            active_cols.update(
                active_indicator_columns_by_tf.get(
                    tf_name, indicator_columns_by_tf.get(tf_name, [])
                )
                or []
            )
            sliced = entries[-limit:] if limit > 0 else entries
            trimmed_rows: list[dict[str, Any]] = []
            for row in sliced:
                if not isinstance(row, dict):
                    trimmed_rows.append(row)
                else:
                    trimmed_rows.append({k: v for k, v in row.items() if k in active_cols})
            return trimmed_rows

        tf_30m = _slice_context(primary_context_key, context_counts.get(primary_context_key, 0)) if primary_context_key else []
        tf_4h = _slice_context(secondary_context_key, context_counts.get(secondary_context_key, 0)) if secondary_context_key else []
        context = {"tf_30m": tf_30m, "tf_4h": tf_4h}
        initial_payload: dict[str, list[dict[str, Any]]] = {}
        frame_keys = normalized_initial_tfs or ([primary_initial_tf] if primary_initial_tf else [])
        for tf_name in frame_keys:
            limit = context_counts.get(tf_name, len(initial_frames_data.get(tf_name, [])))
            initial_payload[tf_name] = _slice_context(tf_name, limit)
        context["initial_timeframes"] = initial_payload
        context["regime"] = regime_metrics
        context["market_mode"] = market_mode_label
        context["analysis_balance"] = analysis_balance
        if position_summary:
            context["position"] = position_summary
        if open_orders:
            context["open_orders"] = open_orders
        return context

    def build_prompt(extra=None, bias=False, position_summary=position_summary):
        provider_mode = (NEWS_PROVIDER or "hybrid").strip().lower()
        if provider_mode in NEWS_PROVIDER_ALIAS_CC:
            news_desc = "CryptoCompare API (fallback: RSS feeds)"
        elif provider_mode in NEWS_PROVIDER_ALIAS_RSS:
            news_desc = "RSS headlines for the asset"
        else:
            news_desc = "CryptoCompare + RSS headlines"
        news_focus = news_payload_payload.get("focus") if isinstance(news_payload_payload, dict) else None
        if isinstance(news_focus, list) and news_focus:
            log(
                f"[INFO] {symbol}: news focus requested for prompt: {', '.join(map(str, news_focus))}",
                Fore.LIGHTBLACK_EX,
            )

        # Resolve account style from extra context or portfolio guidance; default to balanced intraday.
        style_value: str | None = None
        if isinstance(extra, dict):
            meta_candidate = extra.get("meta") or extra.get("bundle_meta")
            if isinstance(meta_candidate, dict):
                raw_style = meta_candidate.get("style")
                if isinstance(raw_style, str) and raw_style.strip():
                    style_value = raw_style.strip()
        if style_value is None and isinstance(portfolio_guidance, dict):
            raw_style = (
                portfolio_guidance.get("style")
                or (portfolio_guidance.get("target") or {}).get("style")
            )
            if isinstance(raw_style, str) and raw_style.strip():
                style_value = raw_style.strip()
        account_style_value = style_value or "balanced_intraday"
        account_payload = {
            "style": account_style_value,
            "risk_pct": RISK_PCT,
            "configured_leverage": LEVERAGE,
            "sl_atr_mult": sl_mult_local,
            "tp_atr_mult": tp_mult_local,
            "notional_scale": notional_scale,
            "account_scale": 1.0,
        }
        account_payload["market_mode"] = market_mode_label
        account_payload["analysis_balance"] = analysis_balance
        if mode_confidence_required:
            account_payload["confidence_floor"] = mode_confidence_required
        if margin_ratio is not None:
            account_payload["available_margin_pct"] = margin_ratio

        position_for_prompt = _select_prompt_position(symbol, position_summary)
        higher_tf_hints = [f"higher_tf:{tf}" for tf in (normalized_initial_tfs or [])]
        lower_tf_hints = [
            f"lower_tf:{tf}"
            for tf in AI_NEEDS_TF_DEPTHS.keys()
            if tf and tf not in (normalized_initial_tfs or [])
        ]
        additional_indicator_hints = [
            "ema10",
            "ema75",
            "ema100",
            "ema150",
            "ema300",
            "sma20",
            "sma50",
            "sma100",
            "sma200",
            "rsi7",
            "rsi21",
            "stoch21",
            "macd_signal",
            "macd_hist",
            "atr7",
            "atr21",
            "bbands50",
            "bbands100",
            "vwma50",
            "vwma100",
            "supertrend10",
            "supertrend20",
            "adx14",
            "adx20",
            "cci20",
            "cci50",
            "obv",
            "mfi14",
            "roc10",
            "tema20",
            "keltner20",
            "dmi14",
        ]
        indicator_hints = sorted(
            set((AI_INITIAL_INDICATOR_POOL or []) + additional_indicator_hints)
        )
        prompt = {
            "symbol": symbol,
            "account": account_payload,
            "context": current_context,
            "position": position_for_prompt,
            "open_orders": open_orders,
            "regime": regime_metrics,
            "market_mode": market_mode_label,
            "instructions": {
                "response_format": {
                    "action": "open|close|manage|skip",
                    "side": "buy|sell|flat",
                    "confidence": "0..1",
                    "reason": "short explanation in Russian or English",
                    "orders": "optional list of CCXT-style orders (type, side, amount/percent, price, params)",
                    "needs": "optional list of extra context requests",
                },
                "needs_rules": [
                    "Request only new context that is not already provided in context.initial_timeframes.",
                    "Specify concrete items (e.g., \"higher_tf:2h\", \"indicator:rsi21\", \"news:macro\").",
                ],
                "notes": [
                    "Make decisions based solely on the provided data; no strategy templates are pre-baked.",
                    "Default to trading with the dominant regime: TREND_UP/TREND_DOWN setups have the highest priority, FLAT/range ideas are secondary, and COUNTER plays are lowest priority and require stronger confirmation.",
                    "Treat countertrend entries as lower priority and only when multiple signals support them; otherwise prefer skip/manage or ask for more context.",
                    "Act in the direction the data supports: buy/long when regime and momentum are bullish, sell/short when they are bearish; there is no built-in bias toward either side.",
                    "If signals conflict or no position exists to manage, prefer skip/manage (or request extra context) rather than forcing a new entry.",
                    "Prefer amountPercent when sizing orders; engine will scale to each account.",
                    "Use reduceOnly=true when closing or trimming existing positions.",
                ],
            },
            "available_data": {
                "higher_tf": higher_tf_hints,
                "lower_tf": lower_tf_hints,
                "indicators": indicator_hints,
                "funding": "fetchFundingRate",
                "open_interest": "fetchOpenInterestHistory",
                "news": news_desc,
            },
        }
        instructions_block = prompt["instructions"]
        guide_modes = MARKET_MODE_GUIDE.get("market_modes") if isinstance(MARKET_MODE_GUIDE, dict) else None
        if guide_modes:
            instructions_block["market_modes"] = guide_modes
        if market_mode_label:
            instructions_block["active_market_mode"] = market_mode_label
        if mode_confidence_required:
            instructions_block["confidence_required"] = mode_confidence_required
        instructions_block["analysis_balance"] = analysis_balance
        balance_note = (
            "Adjust reliance on news vs technicals according to analysis_balance "
            "(news_priority = heed catalysts first; technical_priority = rely on structure/indicators)."
        )
        if balance_note not in instructions_block["notes"]:
            instructions_block["notes"].append(balance_note)
        prompt["instructions"] = instructions_block

        if portfolio_guidance:
            prompt["portfolio_guidance"] = portfolio_guidance
        if extra_context_payload:
            prompt["requested_context"] = extra_context_payload
        if news_payload_payload:
            prompt["news_focus"] = news_payload_payload
        if extra:
            prompt["extra_notes"] = extra
        if bias:
            prompt["bias_hint"] = "maintain existing bias to reduce flip-flops"
        return json.dumps(prompt, ensure_ascii=False)

    def prepare_messages(stage="initial", extra=None, bias=False):
        nonlocal indicator_trim_index
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
            for tf_name in trim_order:
                current_limit = context_counts.get(tf_name, 0)
                min_limit = min_context_by_tf.get(tf_name, 1)
                step_value = step_context_by_tf.get(tf_name, 1)
                if current_limit > min_limit:
                    new_val = max(min_limit, current_limit - step_value)
                    if new_val < current_limit:
                        context_counts[tf_name] = new_val
                        trimmed_step = True
                        break
            # If depth trimming is exhausted, progressively drop lowest-priority indicators.
            if not trimmed_step and indicator_trim_index > MIN_INDICATORS_PER_TF:
                removed_indicator = None
                while indicator_trim_index > MIN_INDICATORS_PER_TF and not removed_indicator:
                    candidate = indicator_priority[indicator_trim_index - 1]
                    indicator_trim_index -= 1
                    if not candidate:
                        continue
                    removed_somewhere = False
                    for tf_name, cols in active_indicator_columns_by_tf.items():
                        if candidate in cols and len(cols) > MIN_INDICATORS_PER_TF:
                            cols.remove(candidate)
                            removed_somewhere = True
                    if removed_somewhere:
                        removed_indicator = candidate
                        trim_sources.append(f"ind:{candidate}")
                if removed_indicator:
                    trimmed_step = True
            if not trimmed_step:
                break
            trimmed = True
            trim_sources.append(trim_reason)
            attempts += 1
            if attempts > 50:
                log(f"⛔ Обрезка контекста не укладывается в лимит ({stage}) для {symbol}", Fore.YELLOW)
                break
        trimmed = trimmed or (context_counts != prev_counts)
        if trimmed:
            reasons_text = "/".join(sorted(set(trim_sources))) if trim_sources else "unknown"
            counts_display = ", ".join(
                f"{context_counts.get(tf, 0)}?{tf}" for tf in trim_order if tf
            )
            if not counts_display and primary_context_key:
                counts_display = f"{context_counts.get(primary_context_key, 0)}?{primary_context_key}"
            trim_text = (
                f"Context trimmed ({reasons_text}) to "
                f"{counts_display or 'n/a'} "
                f"due to token limit ({tokens} tokens, stage: {stage}) for {symbol}"
            )
            log(trim_text, Fore.MAGENTA)
            send_tg(trim_text)
        if hard_limit and tokens > hard_limit:
            log(f"⛔ Лимит токенов превышен даже после обрезки ({tokens}>{TOKEN_LIMIT}, этап: {stage}) для {symbol}", Fore.YELLOW)
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
    # Tag regime for logging (trend/flat/counter) based on AI hints.
    ai_regime = None
    try:
        ai_regime = (decision.get("regime") if isinstance(decision, dict) else None) or None
    except Exception:
        ai_regime = None
    confidence_value = None
    confidence_raw = None
    confidence_display = None
    low_confidence = False
    auto_needs_triggered = False
    auto_low_confidence_needs_triggered = False
    needs = []
    if decision is None:
        messages_init, tokens_init, initial_payload = prepare_messages(stage="initial")
        initial_summary = (initial_payload or "").replace("\n", " ")[:200]
        log(f"[AI REQUEST] initial {symbol}: tokens={tokens_init}, prompt={initial_summary}", Fore.LIGHTBLACK_EX)
        per_cap_init = _current_request_token_cap()
        if per_cap_init and tokens_init > per_cap_init:
            log(f"⛔ {symbol}: запрос initial превышает кап {per_cap_init} токенов", Fore.YELLOW)
            fallback_option = fallback_due_to("token cap exceeded (initial)")
            if fallback_option:
                return fallback_option
            return {"symbol": symbol, "action": "skip", "reason": "token cap exceeded"}
        budget_ok = _ensure_token_budget(tokens_init, AI_MODEL, f"{symbol} initial decision")
        if not budget_ok and priority_symbol:
            log(f"⚠ {symbol}: превышен лимит токенов, продолжаем из-за открытой позиции", Fore.YELLOW)
        elif not budget_ok:
            log(f"⛔ {symbol}: пропуск initial-запроса из-за лимита токенов", Fore.YELLOW)
            fallback_option = fallback_due_to("token budget exhausted (initial)")
            if fallback_option:
                return fallback_option
            return {"symbol": symbol, "action": "skip", "reason": "token budget exceeded"}
        _log_ai_request(AI_MODEL, tokens_init, f"{symbol} initial decision")

        # --- Первый проход ---
        start_init = time.perf_counter()
        try:
            res = client.chat.completions.create(
                model=AI_MODEL,
                temperature=0,
                response_format={"type":"json_object"},
                messages=messages_init
            )
            duration_init = time.perf_counter() - start_init
            log(f"ℹ️ OpenAI initial запрос для {symbol}: {duration_init:.2f} c", Fore.LIGHTBLACK_EX)
            _register_ai_usage(AI_MODEL, getattr(res, "usage", None), f"{symbol} initial decision")
            msg = res.choices[0].message.content
            decision = json.loads(msg)
            if isinstance(decision, dict) and decision.get("regime"):
                ai_regime = ai_regime or decision.get("regime")
            if isinstance(decision, dict) and decision.get("regime"):
                ai_regime = ai_regime or decision.get("regime")
        except RateLimitError:
            raise
        except Exception as exc:
            _emit_ai_offline_notice(f"decision initial failed: {type(exc).__name__}")
            log(f"[WARN] {symbol}: AI initial decision failed: {exc}", Fore.YELLOW)
            fallback = dict(initial_decision) if isinstance(initial_decision, dict) else {}
            fallback.setdefault("symbol", symbol)
            current_amt = safe_float((current_position or {}).get("amount") or (current_position or {}).get("contracts")) or 0.0
            has_pos = abs(current_amt) > 0
            fallback["action"] = "hold" if has_pos else "skip"
            fallback["reason"] = f"AI unavailable (initial): {type(exc).__name__}"
            fallback["ai_unavailable"] = True
            return ensure_skip_reason(fallback)
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
                msg_low = f"ℹ️ Low confidence ({display}) for {symbol}: requesting {', '.join(additional_needs)}"
                log(msg_low, Fore.LIGHTBLACK_EX)
                send_tg_decision(msg_low)

        response_action = (decision.get("action") or "").lower() or "skip"
        response_reason = (decision.get("reason") or "").replace("\n", " ")[:200]
        response_regime = decision.get("regime") or ai_regime or ""
        regime_tag_init = f"[{response_regime}] " if response_regime else ""
        log(f"[AI RESPONSE] initial {symbol}: {regime_tag_init}action={response_action} reason={response_reason}", Fore.LIGHTBLACK_EX)

        context_payload = None
        if AI_REQUESTS_FULL_CONTEXT:
            context_payload = current_context
        elif AI_REQUESTS_MAX_CONTEXT_BARS > 0 and isinstance(current_context, dict):
            trimmed: dict[str, Any] = {}
            for tf_key, tf_bars in current_context.items():
                if isinstance(tf_bars, list) and tf_bars:
                    trimmed[tf_key] = tf_bars[-AI_REQUESTS_MAX_CONTEXT_BARS :]
            context_payload = trimmed

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
                "context": context_payload,
                "needs": needs,
                "auto_needs": auto_needs_triggered,
                "low_confidence": low_confidence,
                "auto_low_confidence": auto_low_confidence_needs_triggered,
                "prompt_summary": initial_summary,
                "response_action": response_action,
                "response_reason": response_reason,
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
        initial_tfs_set = set(normalized_initial_tfs or ([primary_initial_tf] if primary_initial_tf else []))
        structured_timeframes: list[str] = []
        for tf in LOW_CONFIDENCE_TIMEFRAMES:
            tf_clean = str(tf).strip()
            if not tf_clean or tf_clean in initial_tfs_set:
                continue
            if tf_clean not in structured_timeframes:
                structured_timeframes.append(tf_clean)
        if structured_timeframes and NEEDS_MAX_TIMEFRAMES:
            structured_timeframes = structured_timeframes[:NEEDS_MAX_TIMEFRAMES]
        provided_initial_indicators = set(indicator_columns_by_tf.get(primary_initial_tf, []))
        structured_indicators: list[str] = []
        for ind in LOW_CONFIDENCE_INDICATORS:
            ind_clean = str(ind).strip()
            if not ind_clean or ind_clean in provided_initial_indicators:
                continue
            if ind_clean not in structured_indicators:
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
            selected_limits = [
                AI_NEEDS_TF_DEPTHS.get(tf, NEEDS_SERIALIZE_DEFAULT_LIMIT)
                for tf in (structured_timeframes or [])
            ]
            fallback_limit = max(
                selected_limits
                or [NEEDS_SERIALIZE_DEFAULT_LIMIT or LOW_CONFIDENCE_SERIALIZE_LIMIT or 8]
            )
            structured_need["limit"] = max(1, int(fallback_limit))
            needs.insert(0, structured_need)
            decision["needs"] = needs
            auto_low_confidence_needs_triggered = True
            display = confidence_display or "n/a"
            tf_text = ", ".join(structured_timeframes) if structured_timeframes else "none"
            ind_text = ", ".join(structured_indicators) if structured_indicators else "none"
            msg_auto = (
                f"ℹ️ Low confidence ({display}) for {symbol}: requesting extra TFs [{tf_text}] "
                f"and indicators [{ind_text}]"
            )
            log(msg_auto, Fore.LIGHTBLACK_EX)
            send_tg_decision(msg_auto)

    # --- Если запрошен контекст ---
    if needs:
        if auto_low_confidence_needs_triggered:
            display = confidence_display or "n/a"
            msg_auto_low = f"ℹ️ Low-confidence auto context ({display}) for {symbol}: {needs}"
            log(msg_auto_low, Fore.CYAN)
            send_tg_decision(msg_auto_low)
        elif auto_needs_triggered:
            msg_auto_needs = f"ℹ️ Auto-requested context after skip reason for {symbol}: {needs}"
            log(msg_auto_needs, Fore.CYAN)
            send_tg_decision(msg_auto_needs)
        else:
            msg_manual = f"ℹ️ Model requested extra context for {symbol}: {needs}"
            log(msg_manual, Fore.CYAN)
            send_tg_decision(msg_manual)
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
                    if isinstance(k, str) and k.lower() in ("ema", "sma", "rsi", "atr", "stoch", "stochrsi")
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

        log("ℹ️ Контекст собран: " + ", ".join(stats_report), Fore.LIGHTBLACK_EX)
        send_tg("ℹ️ Контекст собран для " + symbol + ":\n" + "\n".join(stats_report))

        if False:
            log(f"⛔ {symbol}: пропуск допконтекста из-за достигнутого лимита токенов", Fore.YELLOW)
            decision["needs_followup"] = needs
            decision.pop("needs", None)
            return ensure_skip_reason(decision)
        bias_flag = bool(AI_AFTER_NEEDS_BIAS)
        messages_extra, tokens_extra, extra_payload = prepare_messages(stage="extra", extra=extra, bias=bias_flag)
        log(f"ℹ️ Токены запроса (extra) для {symbol}: {tokens_extra}", Fore.LIGHTBLACK_EX)
        extra_summary = (extra_payload or "").replace("\n", " ")[:200]
        log(f"[AI REQUEST] extra {symbol}: tokens={tokens_extra}, prompt={extra_summary}", Fore.LIGHTBLACK_EX)
        per_cap_extra = _current_request_token_cap()
        if per_cap_extra and tokens_extra > per_cap_extra:
            log(f"⛔ {symbol}: запрос extra превышает кап {per_cap_extra} токенов", Fore.YELLOW)
            decision["needs_followup"] = needs
            decision.pop("needs", None)
            fallback_option = fallback_due_to(
                "token cap exceeded (extra)",
                {"needs_carry_over": needs, "fallback_stage": "extra"},
            )
            if fallback_option:
                return fallback_option
            return ensure_skip_reason(decision)
        budget_ok_extra = _ensure_token_budget(tokens_extra, AI_MODEL, f"{symbol} extra decision")
        if not budget_ok_extra and priority_symbol:
            log(f"⚠ {symbol}: превышен лимит токенов на extra-запросе, продолжаем из-за открытой позиции", Fore.YELLOW)
        elif not budget_ok_extra:
            log(f"⛔ {symbol}: пропуск extra-запроса из-за лимита токенов", Fore.YELLOW)
            decision["needs_followup"] = needs
            fallback_option = fallback_due_to(
                "token budget exhausted (extra)",
                {"needs_carry_over": needs, "fallback_stage": "extra"},
            )
            if fallback_option:
                return fallback_option
            return ensure_skip_reason(decision)
        _log_ai_request(AI_MODEL, tokens_extra, f"{symbol} extra decision")
        start_extra = time.perf_counter()
        try:
            res2 = client.chat.completions.create(
                model=AI_MODEL,
                temperature=0,
                response_format={"type":"json_object"},
                messages=messages_extra
            )
            duration_extra = time.perf_counter() - start_extra
            log(f"ℹ️ OpenAI extra запрос для {symbol}: {duration_extra:.2f} c", Fore.LIGHTBLACK_EX)
            _register_ai_usage(AI_MODEL, getattr(res2, "usage", None), f"{symbol} extra decision")
            msg2 = res2.choices[0].message.content
            decision = json.loads(msg2)
            if isinstance(decision, dict) and decision.get("regime"):
                ai_regime = ai_regime or decision.get("regime")
        except RateLimitError:
            raise
        except Exception as exc:
            _emit_ai_offline_notice(f"decision extra failed: {type(exc).__name__}")
            log(f"[WARN] {symbol}: AI extra decision failed: {exc}", Fore.YELLOW)
            fallback = dict(decision) if isinstance(decision, dict) else {}
            fallback.setdefault("symbol", symbol)
            current_amt = safe_float((current_position or {}).get("amount") or (current_position or {}).get("contracts")) or 0.0
            has_pos = abs(current_amt) > 0
            fallback["action"] = "hold" if has_pos else "skip"
            fallback["reason"] = f"AI unavailable (extra): {type(exc).__name__}"
            fallback["ai_unavailable"] = True
            return ensure_skip_reason(fallback)
        needs_followup = decision.get("needs", [])
        if needs_followup:
            log(f"ℹ️ После допконтекста модель все ещё запрашивает {needs_followup} для {symbol}", Fore.LIGHTBLACK_EX)
            decision["needs_followup"] = needs_followup
            decision.pop("needs", None)
        response_action_extra = (decision.get("action") or "").lower() or "skip"
        response_reason_extra = (decision.get("reason") or "").replace("\n", " ")[:200]
        response_regime_extra = decision.get("regime") or ai_regime or ""
        regime_tag = f"[{response_regime_extra}] " if response_regime_extra else ""
        log(f"[AI RESPONSE] extra {symbol}: {regime_tag}action={response_action_extra} reason={response_reason_extra} needs_followup={needs_followup}", Fore.LIGHTBLACK_EX)
        context_payload_extra = None
        if AI_REQUESTS_FULL_CONTEXT:
            context_payload_extra = current_context
        elif AI_REQUESTS_MAX_CONTEXT_BARS > 0 and isinstance(current_context, dict):
            trimmed_extra: dict[str, Any] = {}
            for tf_key, tf_bars in current_context.items():
                if isinstance(tf_bars, list) and tf_bars:
                    trimmed_extra[tf_key] = tf_bars[-AI_REQUESTS_MAX_CONTEXT_BARS :]
            context_payload_extra = trimmed_extra

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
                "context": context_payload_extra,
                "extra": extra,
                "bias": bias_flag,
                "needs_followup": needs_followup,
                "prompt_summary": extra_summary,
                "response_action": response_action_extra,
                "response_reason": response_reason_extra
            }
        )
        log(f"🔷 Второй проход завершён для {symbol}", Fore.CYAN)
        send_tg(f"ℹ️ Второй проход завершён для {symbol}")

    decision = ensure_skip_reason(decision)
    return decision

# --- Основная логика ---
def build_trade_plan_snapshot(
    exchange,
    *,
    symbol_candidates: Sequence[str] | None = None,
    positions_map: Mapping[str, dict] | None = None,
    equity: float = 0.0,
    available_margin: float = 0.0,
    news_digest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbols = sorted({sym for sym in (symbol_candidates or []) if sym})
    positions_map = dict(positions_map or {})
    universe_cache = load_universe_cache()
    news_payload = news_digest if news_digest is not None else _build_news_digest(symbols)
    updated = ai_update_universe(
        exchange=exchange,
        symbols=symbols,
        positions_map=positions_map,
        equity=equity,
        available_margin=available_margin,
        universe_cache=universe_cache,
        news_digest=news_payload,
    )
    if not updated:
        raise RuntimeError("Engine: ai_update_universe returned no result")
    selection_result, universe_state, news_requests = updated
    universe_state = universe_state or {}
    universe_state["engine_generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_universe_cache(universe_state)
    bundle, open_orders_cache = build_portfolio_bundle(
        exchange,
        selection_result or {},
        positions_map,
        news_cache=None,
    )
    trade_plan = ai_plan_trades(
        exchange,
        bundle,
        equity,
        available_margin,
        positions_snapshot=positions_map,
        pending_orders=open_orders_cache,
        stage="initial",
    )
    return {
        "selection": selection_result,
        "universe": universe_state,
        "news_requests": news_requests,
        "bundle": bundle,
        "trade_plan": trade_plan,
        "meta": {
            "symbol_candidates": symbols,
            "has_news_digest": bool(news_payload),
        },
    }


def apply_trade_plan_snapshot(
    exchange,
    snapshot,
    *,
    user_id: str | None = None,
    dry_run: bool = False,
):
    trade_plan_payload = (
        snapshot.get("trade_plan")
        or snapshot.get("selection")
        or snapshot
    )
    if not trade_plan_payload:
        raise RuntimeError("No trade_plan payload in snapshot")
    decisions = (
        (trade_plan_payload or {}).get("trade_plan")
        or trade_plan_payload.get("decisions")
    )
    if not decisions and isinstance(snapshot.get("trade_plan"), dict):
        decisions = (
            snapshot["trade_plan"].get("trade_plan")
            or snapshot["trade_plan"].get("decisions")
        )
    if not decisions:
        decisions = []

    equity, available_margin, _ = fetch_usdt_equity(exchange)
    user_key = user_id or "default"
    user_tag = f"[user={user_key}]"
    open_skip_notes: defaultdict[str, list[str]] = defaultdict(list)
    symbols_with_position_seen: set[str] = set()

    def log_user(msg: str) -> None:
        tagged = f"{user_tag} {msg}"
        log(tagged, Fore.LIGHTBLACK_EX)
        if user_id:
            _append_user_bybit_log(user_id, tagged)

    def log_open_skip(symbol: str, reason: str) -> None:
        log_user(f"OPEN SKIP {symbol}: {reason}")
        try:
            open_skip_notes[symbol].append(reason)
        except Exception:
            pass

    def _record_pending_entry(symbol: str, qty_value: float, side_value: str) -> None:
        record_pending_entry(symbol, qty_value, side_value, user_key)

    def _clear_pending_entry(symbol: str) -> None:
        clear_pending_entry(symbol, user_key)

    def _execute_limit_fallback(symbol: str, pending_info: dict[str, Any], open_orders_list: list[dict[str, Any]] | None) -> bool:
        fallback_qty = pending_info.get("qty") or 0.0
        fallback_side = pending_info.get("side")
        if fallback_qty <= 0 or not fallback_side:
            _clear_pending_entry(symbol)
            return False
        orders_to_cancel = open_orders_list or []
        for order in orders_to_cancel:
            oid = order.get("id")
            if oid:
                cancel_order_by_id(exchange, symbol, str(oid))
        try:
            res = exchange.create_order(symbol, "market", fallback_side, fallback_qty, None, {})
            log_user(f"FALLBACK market order after {LIMIT_ORDER_FALLBACK_SECONDS:.0f}s: {symbol} {fallback_side} {fallback_qty:.6f}")
            log(
                f"{user_tag} FALLBACK {symbol} {fallback_side} {fallback_qty:.6f} -> {res}",
                Fore.LIGHTBLACK_EX,
            )
            send_tg(f"[FALLBACK] {user_tag} {symbol}: market {fallback_side.upper()} {fallback_qty:.6f} executed after limit not filled")
            return True
        except Exception as exc:
            log_user(f"FALLBACK ORDER FAIL {symbol}: {exc}")
            log(f"{user_tag} FALLBACK ORDER FAIL {symbol}: {exc}", Fore.YELLOW)
            return False
        finally:
            _clear_pending_entry(symbol)

    log_user(f"USERBOT starting run: equity={equity} available={available_margin}")

    for dec in decisions:
        try:
            symbol_raw = dec.get("symbol") or dec.get("pair") or dec.get("ticker")
            if not symbol_raw:
                continue
            symbol = str(symbol_raw).strip()
            if not symbol:
                continue
            action = (
                dec.get("action")
                or dec.get("side")
                or dec.get("direction")
                or "buy"
            )
            action = str(action).lower()
            order_type = (
                dec.get("type")
                or dec.get("order_type")
                or "market"
            )
            order_type = str(order_type).lower()
            notional = None
            for key in ("notional_usdt", "notional", "notionalUsd", "quote", "size_usdt"):
                val = dec.get(key)
                if val is None:
                    continue
                try:
                    notional = float(val)
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
                    if pct_v > 1:
                        pct_v = pct_v / 100.0
                    notional = max(0.0, float(equity) * float(pct_v))
            if notional is None:
                notional = min(50.0, float(equity) * 0.01 if equity and equity > 0 else 50.0)

            price = None
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker.get("last") or ticker.get("close") or ticker.get("info", {}).get("lastPrice")
            except Exception:
                price = None
            if not price:
                try:
                    mkt = exchange.market(symbol)
                except Exception:
                    mkt = None
                price = (mkt or {}).get("info", {}).get("lastPrice") if mkt else None
            if not price:
                log_user(f"SKIP {symbol}: cannot determine price for decision {dec}")
                continue

            qty = float(notional) / float(price) if price and float(price) > 0 else 0.0
            try:
                rules = _get_symbol_trade_rules(exchange, symbol)
                step = rules.get("qty_step") or rules.get("min_qty") or None
                if step and step > 0:
                    qty = max(step, (int(qty / step) * step))
            except Exception:
                pass
            qty = max(0.0, float(qty))
            if qty <= 0:
                log_user(f"SKIP {symbol}: computed qty 0 for notional {notional}")
                continue
            if dry_run:
                log_user(
                    f"DRY RUN {symbol} {action} {order_type} "
                    f"qty={qty:.6f} price={price} notional={notional}"
                )
                log(f"{user_tag} DRY RUN {symbol} {action} {order_type} qty={qty:.6f} price={price} notional={notional}", Fore.LIGHTBLACK_EX)
                continue
            params = dict(dec.get("params") or {})
            price_for_order = dec.get("price") or dec.get("limit") or price
            try:
                if order_type == "market":
                    res = exchange.create_order(symbol, "market", action, qty, None, params)
                    order_price_text = "market"
                else:
                    res = exchange.create_order(symbol, "limit", action, qty, float(price_for_order), params)
                    order_price_text = price_for_order or price
                log_user(
                    f"ORDER {symbol} {action} {order_type} qty={qty:.6f} "
                    f"price={order_price_text} -> {res}"
                )
                log(
                    f"{user_tag} ORDER {symbol} {action} {order_type} qty={qty:.6f} price={order_price_text} -> {res}",
                    Fore.LIGHTBLACK_EX,
                )
            except Exception as exc:
                log_user(f"ORDER FAIL {symbol}: {exc}")
                log(f"{user_tag} ORDER FAIL {symbol}: {exc}", Fore.YELLOW)
        except Exception as exc_outer:
            log_user(f"Decision processing error: {exc_outer}")

    return decisions


# --- ℹ️ℹ️ℹ️ ℹ️ℹ️ℹ️ ---
def run_cycle():


    global DYNAMIC_SYMBOL_ALIASES
    global SYMBOL_RULES_CACHE
    global CURRENT_RISK_PCT
    global ORDER_MARGIN_UTILIZATION
    global AUTO_MARGIN_SCALE
    global AUTO_MARGIN_SCALE_RATIO
    global AUTO_MARGIN_CONFIDENCE_MULT
    global _PREV_UNREALIZED_PNL, _TRAIL_PROTECTION, _CURRENT_CYCLE_NUMBER
    cycle_start_utc = datetime.datetime.now(datetime.timezone.utc)
    enable_stdio_logging()
    active_user_id = os.getenv("BYBITBOT_USER_ID") or "default"
    is_master_user = (
        (not MASTER_DECISIONS_SHARE)
        or (MASTER_PROMPT_MASTER_ID is None)
        or (str(active_user_id) == str(MASTER_PROMPT_MASTER_ID))
    )
    user_tag = f"[user={active_user_id}]"

    symbols_with_position_seen: set[str] = set()

    def log_user(msg: str, *, color: str = Fore.LIGHTBLACK_EX) -> None:
        tagged = f"{user_tag} {msg}"
        log(tagged, color)
        if active_user_id:
            _append_user_bybit_log(active_user_id, tagged)

    def log_open_skip(symbol: str, reason: str) -> None:
        log_user(f"OPEN SKIP {symbol}: {reason}")
    _sync_with_remote()
    _write_runtime_status(None, None, "running")
    refresh_settings()
    # Ensure each user has a bybit.log for per-user trading history
    try:
        _ensure_user_bybit_log_for_all()
    except Exception:
        pass
    configure_telegram_bot()
    start_telegram_webhook_server()
    start_telegram_long_polling()
    _init_ai_cycle_usage()
    metadata_state = maybe_refresh_metadata()
    if isinstance(metadata_state, dict) and metadata_state.get("reload_required"):
        new_hash = metadata_state.get("current_hash")
        short_hash = (new_hash or "")[:8] if isinstance(new_hash, str) else "?"
        reason = f"✅ Обнаружен новый коммит {short_hash}, перезапускаем бота для загрузки обновлений."
        _restart_with_latest_code(reason)
    ex = init_exchange()
    if MASTER_PROMPT_SHARE:
        MASTER_PROMPT_POSITION_CACHE.clear()
    if MASTER_DECISIONS_SHARE:
        MASTER_DECISION_CACHE.clear()
        if not is_master_user:
            _load_master_decisions_runtime()
    
    def _execute_limit_fallback(symbol: str, pending_info: dict[str, Any] | None, open_orders_list: list[dict[str, Any]] | None) -> bool:
        """Execute a market fallback if limit entry wasn't filled within timeout.

        Cancels current open entry limits for the symbol and places a market order
        with the original side/qty from pending_info. Returns True if a market
        order was successfully sent (or there is nothing to do), False otherwise.
        """
        try:
            qty = float((pending_info or {}).get("qty") or 0.0)
        except Exception:
            qty = 0.0
        side_val = (pending_info or {}).get("side")
        if qty <= 0 or not side_val:
            # Clear stale pending marker
            try:
                clear_pending_entry(symbol, active_user_id)
            except Exception:
                pass
            return False
        # Cancel existing entry limits
        for order in (open_orders_list or []):
            try:
                oid = order.get("id")
            except AttributeError:
                oid = None
            if oid:
                try:
                    cancel_order_by_id(ex, symbol, str(oid))
                except Exception:
                    pass
        try:
            res = ex.create_order(symbol, "market", str(side_val).lower(), qty, None, {})
            log_user(
                f"FALLBACK market order after {LIMIT_ORDER_FALLBACK_SECONDS:.0f}s: {symbol} {str(side_val).lower()} {qty:.6f}"
            )
            log(f"{user_tag} FALLBACK {symbol} {str(side_val).lower()} {qty:.6f} -> {res}", Fore.LIGHTBLACK_EX)
            try:
                send_tg(
                    f"[FALLBACK] {user_tag} {symbol}: market {str(side_val).upper()} {qty:.6f} executed after limit not filled"
                )
            except Exception:
                pass
            try:
                clear_pending_entry(symbol, active_user_id)
            except Exception:
                pass
            return True
        except Exception as _exc:
            # Keep pending to retry later; log the error
            log(f"[WARN] {user_tag} {symbol}: fallback market failed: {_exc}", Fore.YELLOW)
            return False
    SYMBOL_RULES_CACHE.clear()
    ex.load_markets()
    try:
        dynamic_aliases = _build_dynamic_symbol_aliases(getattr(ex, "markets", {}))
        DYNAMIC_SYMBOL_ALIASES.clear()
        DYNAMIC_SYMBOL_ALIASES.update(dynamic_aliases)
    except Exception as exc_alias:
        log(f"[WARN] Failed to build dynamic symbol aliases: {exc_alias}", Fore.YELLOW)
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
        sym_upper = sym.upper()
        if sym_upper in markets_set:
            if sym_upper != sym:
                symbol_alias_hits[sym] = sym_upper
            return sym_upper
        alias_target = SYMBOL_ALIASES.get(sym) or SYMBOL_ALIASES.get(sym_upper)
        if alias_target and alias_target in markets_set:
            symbol_alias_hits[sym] = alias_target
            return alias_target
        dynamic_target = DYNAMIC_SYMBOL_ALIASES.get(sym) or DYNAMIC_SYMBOL_ALIASES.get(sym_upper)
        if dynamic_target and dynamic_target in markets_set:
            symbol_alias_hits[sym] = dynamic_target
            return dynamic_target
        resolved = _resolve_symbol_alias(sym)
        if resolved and resolved in markets_set:
            if resolved != sym:
                symbol_alias_hits[sym] = resolved
            return resolved
        if record_missing:
            missing_symbols.add(sym)
        return None

    ensure_position_mode(ex)
    positions_map, open_positions = fetch_positions_snapshot(ex)
    spot_position_payloads = {}
    try:
        spot_position_payloads = fetch_spot_position_symbols(ex, min_value_usd=0.05)
    except Exception:
        spot_position_payloads = {}
    if spot_position_payloads:
        for sym_spot, payload in spot_position_payloads.items():
            if sym_spot not in positions_map:
                positions_map[sym_spot] = payload
    base_exposure_counts = _build_base_exposure_map(positions_map)
    pending_base_allocations: defaultdict[str, int] = defaultdict(int)
    base_max_positions = max(0, MAX_OPEN_POSITIONS or 0)
    max_positions_limit = base_max_positions
    if max_positions_limit > 0 and open_positions is None:
        log("⚠️ Не удалось определить количество открытых позиций — лимит по позициям отключён на этот цикл", Fore.YELLOW)
        open_positions = None
    equity, available_margin, balance_snapshot_start = fetch_usdt_equity(ex)
    realized_start = None
    if isinstance(balance_snapshot_start, dict):
        realized_start = balance_snapshot_start.get("_realizedPnl")
    start_stable_label = _stable_currency_label(balance_snapshot_start)
    if equity <= 0:
        equity = 64.0
    if available_margin <= 0:
        available_margin = equity
    current_risk_baseline = RISK_PCT
    if CURRENT_RISK_PCT <= 0 or not math.isfinite(CURRENT_RISK_PCT):
        CURRENT_RISK_PCT = current_risk_baseline
    else:
        CURRENT_RISK_PCT = min(MAX_DYNAMIC_RISK_PCT, max(MIN_DYNAMIC_RISK_PCT, CURRENT_RISK_PCT))
    initial_cycle_risk_pct = CURRENT_RISK_PCT
    try:
        history_bootstrap = _load_equity_history()
        _update_equity_history(
            history_bootstrap,
            datetime.datetime.now(datetime.timezone.utc),
            equity,
            realized_start,
        )
        drawdown_msg, margin_override, auto_ratio_override, risk_override = _maybe_apply_drawdown_controls(
            history_bootstrap,
            base_margin_utilization,
            base_auto_margin_ratio,
        )
        if margin_override is not None:
            ORDER_MARGIN_UTILIZATION = margin_override
        if auto_ratio_override is not None:
            AUTO_MARGIN_SCALE_RATIO = auto_ratio_override
        if risk_override is not None:
            CURRENT_RISK_PCT = min(CURRENT_RISK_PCT, risk_override)
        if drawdown_msg:
            log(drawdown_msg, Fore.YELLOW)
            try:
                send_tg(drawdown_msg)
            except Exception:
                pass
    except Exception:
        pass
    source_label = (os.getenv("BYBITBOT_SOURCE_LABEL") or "").strip()
    if source_label == "HEAD":
        os.environ["BYBITBOT_CYCLE_KIND"] = "normal"
        os.environ["BYBITBOT_CYCLE_MODE"] = "last"
    cycle_kind = (os.getenv("BYBITBOT_CYCLE_KIND") or "").strip()
    cycle_mode = (os.getenv("BYBITBOT_CYCLE_MODE") or "").strip()
    cycle_state = _load_cycle_state()
    rate_limit_backoff = bool((cycle_state or {}).get("rate_limit_backoff"))
    # Preserve previous per-symbol unrealized PnL for pseudo-trailing decisions this cycle.
    prev_unreal_map = cycle_state.get("positions_unrealized") if isinstance(cycle_state, dict) else {}
    try:
        _PREV_UNREALIZED_PNL = {str(k): float(v) for k, v in (prev_unreal_map or {}).items() if v is not None}
    except Exception:
        _PREV_UNREALIZED_PNL = {}
    # Preserve last trailing-protection levels (stop/take) across cycles.
    # Only load trailing state for symbols that were actually open in the previous cycle,
    # otherwise a symbol that was flat can incorrectly keep "active" trailing state for many cycles.
    prev_trail_map = cycle_state.get("positions_trailing") if isinstance(cycle_state, dict) else {}
    try:
        prev_unreal_keys = set((prev_unreal_map or {}).keys()) if isinstance(prev_unreal_map, dict) else set()
        _TRAIL_PROTECTION = {
            str(sym): {
                "stop": float(vals.get("stop")) if isinstance(vals, dict) and vals.get("stop") is not None else None,
                "take": float(vals.get("take")) if isinstance(vals, dict) and vals.get("take") is not None else None,
                "base_stop": float(vals.get("base_stop")) if isinstance(vals, dict) and vals.get("base_stop") is not None else None,
                "base_take": float(vals.get("base_take")) if isinstance(vals, dict) and vals.get("base_take") is not None else None,
                "base_unreal": float(vals.get("base_unreal")) if isinstance(vals, dict) and vals.get("base_unreal") is not None else None,
                "base_price": float(vals.get("base_price")) if isinstance(vals, dict) and vals.get("base_price") is not None else None,
                "activated_cycle": safe_int(vals.get("activated_cycle")) if isinstance(vals, dict) else None,
                "take_shift_total": float(vals.get("take_shift_total")) if isinstance(vals, dict) and vals.get("take_shift_total") is not None else None,
                "take_extensions": safe_int(vals.get("take_extensions")) if isinstance(vals, dict) else None,
                "take_last_extended_cycle": safe_int(vals.get("take_last_extended_cycle")) if isinstance(vals, dict) else None,
                "tightening_count": safe_int(vals.get("tightening_count")) if isinstance(vals, dict) else None,
                "position_side": (vals.get("position_side") if isinstance(vals, dict) else None),
                "position_qty": float(vals.get("position_qty")) if isinstance(vals, dict) and vals.get("position_qty") is not None else None,
            }
            for sym, vals in (prev_trail_map or {}).items()
            if str(sym) in prev_unreal_keys
        }
    except Exception:
        _TRAIL_PROTECTION = {}
    real_cycles_completed = safe_int(cycle_state.get("total_cycles")) or 0
    next_cycle_number = real_cycles_completed + 1
    cycle_counter = str(next_cycle_number)
    os.environ["BYBITBOT_CYCLE_COUNTER"] = cycle_counter
    _CURRENT_CYCLE_NUMBER = next_cycle_number

    session_dt = _current_log_time()
    session_stamp = session_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    session_separator = "=" * 56
    cycle_number_display = f"Cycle #{next_cycle_number}"
    start_banner = f"{session_separator} START SESSION {cycle_number_display} {session_stamp} {session_separator}"
    log(start_banner, Fore.MAGENTA)
    send_tg(f"{session_separator}\nSTART SESSION {cycle_number_display}\n{session_stamp}\n{session_separator}")
    cycle_kind_display = cycle_kind or "normal"
    cycle_mode_display = cycle_mode or "last"
    log(f"[CYCLE] {cycle_number_display} ({cycle_kind_display}/{cycle_mode_display})", Fore.LIGHTBLACK_EX)
    LATEST_STATUS.update({
        "cycle": next_cycle_number,
        "cycle_kind": cycle_kind_display,
        "cycle_mode": cycle_mode_display,
        "timestamp": session_dt.isoformat(),
        "equity_start": equity,
        "available_start": available_margin,
        "stable_label": start_stable_label,
    })
    if DYNAMIC_RISK_ENABLED and abs(CURRENT_RISK_PCT - current_risk_baseline) > max(1e-5, current_risk_baseline * 0.01):
        risk_state_msg = (
            f"[RISK] Cycle risk pct {CURRENT_RISK_PCT:.4f} (base {RISK_PCT:.4f}, range {MIN_DYNAMIC_RISK_PCT:.4f}-{MAX_DYNAMIC_RISK_PCT:.4f})"
        )
        log(risk_state_msg, Fore.LIGHTBLACK_EX)
        try:
            send_tg(risk_state_msg)
        except Exception:
            pass
    source_label = os.getenv("BYBITBOT_SOURCE_LABEL")
    source_ref = os.getenv("BYBITBOT_SOURCE_REF")
    source_hash = os.getenv("BYBITBOT_SOURCE_HASH")
    source_timestamp_env = os.getenv("BYBITBOT_SOURCE_TIMESTAMP")
    source_message_raw = os.getenv("BYBITBOT_SOURCE_MESSAGE")
    source_message = (source_message_raw or "").splitlines()[0].strip() if source_message_raw else ""
    failure_hash = os.getenv("BYBITBOT_FAILURE_HASH")
    failure_message_raw = os.getenv("BYBITBOT_FAILURE_MESSAGE")
    failure_message = (failure_message_raw or "").splitlines()[0].strip() if failure_message_raw else ""
    failure_timestamp = os.getenv("BYBITBOT_FAILURE_TIMESTAMP")
    source_context = os.getenv("BYBITBOT_FALLBACK_CONTEXT")
    cycle_descriptor = ""
    head_commit_hash, head_commit_message, head_commit_timestamp = get_current_commit_info()
    commit_descriptor = ""
    commit_short = ""
    commit_timestamp_for_display = head_commit_timestamp
    branch_name = get_current_branch_name()
    if source_label and source_ref:
        cycle_label_parts: list[str] = []
        if cycle_kind or cycle_mode:
            kind = cycle_kind or "normal"
            mode = cycle_mode or "last"
            cycle_label_parts.append(f"{kind}/{mode}")
        if cycle_counter:
            if cycle_label_parts:
                cycle_label_parts[-1] = f"{cycle_label_parts[-1]}#{cycle_counter}"
            else:
                cycle_label_parts.append(f"#{cycle_counter}")
        cycle_segment = f"[{cycle_label_parts[0]}] " if cycle_label_parts else ""
        cycle_descriptor = cycle_segment.strip()

        def _build_commit_segment(
            role: str,
            commit_hash_value: str | None,
            timestamp_value: str | None,
            message_value: str | None,
            label_value: str | None = None,
        ) -> str:
            if not (commit_hash_value or message_value or label_value):
                return ""
            segment = role
            if commit_hash_value:
                segment += f" {commit_hash_value[:8]}"
            if label_value:
                label_clean = label_value.strip()
                if label_clean:
                    commit_fragment = (commit_hash_value or "")[:8].lower()
                    if not commit_fragment or commit_fragment not in label_clean.lower():
                        segment += f" ({label_clean})"
            if timestamp_value:
                formatted = _format_commit_timestamp(timestamp_value)
                segment += f" @ {formatted if formatted else timestamp_value}"
            if message_value:
                segment += f" - {message_value}"
            return segment

        failure_segment = _build_commit_segment("fail", failure_hash, failure_timestamp, failure_message)
        backup_segment = _build_commit_segment(
            "backup",
            source_hash or source_ref,
            source_timestamp_env,
            source_message,
            source_label,
        )
        segments = [segment for segment in (failure_segment, backup_segment) if segment]
        if segments:
            git_line = f"[GIT] {cycle_segment}{' -> '.join(segments)}"
        else:
            fallback_line = f"{source_ref} - {source_label}"
            if source_message:
                fallback_line += f": {source_message}"
            git_line = f"[GIT] {cycle_segment}{fallback_line}"
        if source_context:
            git_line += f" ({source_context})"
        if branch_name:
            git_line += f" | branch {branch_name}"
        _send_git_notification(git_line)
        effective_hash = source_hash or source_ref or ""
        commit_short = effective_hash[:8]
        descriptor_base = commit_short or (source_ref or source_label or "")
        commit_descriptor = descriptor_base.strip()
        if source_message:
            commit_descriptor = f"{commit_descriptor} {source_message}".strip()
        commit_timestamp_for_display = source_timestamp_env or commit_timestamp_for_display
    else:
        if head_commit_hash:
            short_hash = head_commit_hash[:8]
            message_text = (head_commit_message or "no commit message").splitlines()[0]
            timestamp_text = head_commit_timestamp or "timestamp unavailable"
            formatted_ts = _format_commit_timestamp(head_commit_timestamp)
            if formatted_ts:
                timestamp_display = formatted_ts
            else:
                timestamp_display = timestamp_text
            git_line = f"[GIT] {short_hash} @ {timestamp_display} - {message_text} (version {BOT_VERSION})"
            if branch_name:
                git_line += f" | branch {branch_name}"
            _send_git_notification(git_line)
            commit_short = short_hash
            commit_descriptor = f"{short_hash} {message_text}"
    commit_descriptor = commit_descriptor.strip()
    commit_time_display = (
        _format_commit_timestamp(commit_timestamp_for_display)
        if commit_timestamp_for_display
        else None
    )
    last_equity = equity
    last_available_margin = available_margin
    start_balance_text = f"Баланс: {equity:.2f} {start_stable_label}, доступно {available_margin:.2f} {start_stable_label}"
    log(f"✅ Бот v{BOT_VERSION} запущен. {start_balance_text}", Fore.GREEN)
    send_tg(f"✅ Бот запущен. {start_balance_text}")

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
    spot_position_symbols = set(spot_position_payloads.keys())
    if spot_position_symbols:
        position_symbols.update(spot_position_symbols)

    # Defer position limit evaluation until after universe/model overrides.
    position_limit_reached = False

    normalized_pair_list: list[str] = []
    for raw_pair in PAIR_LIST:
        resolved_pair = normalize_symbol(raw_pair)
        if resolved_pair and resolved_pair not in normalized_pair_list:
            normalized_pair_list.append(resolved_pair)

    candidate_pairs_set: set[str] = set()

    def add_candidates(values, *, record_missing: bool = True, require_capacity: bool = False):
        if position_limit_reached and require_capacity:
            return
        for value in values:
            resolved = normalize_symbol(value, record_missing=record_missing)
            if resolved:
                candidate_pairs_set.add(resolved)

    add_candidates(PAIR_LIST, require_capacity=True)
    add_candidates(BASE_PAIR_CANDIDATES, require_capacity=True)
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

    selection_style: str | None = None
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
        initial_timeframes = selection_result.get("initial_timeframes") or selection_result.get("global_timeframes") or []
        aggression_level = selection_result.get("aggression")
        trade_horizon = selection_result.get("trade_horizon")
        style_value = selection_result.get("style")
        if isinstance(style_value, str):
            selection_style = style_value.strip() or None
        overview_lines: list[str] = []
        if initial_timeframes:
            overview_lines.append(f"TF: {', '.join(initial_timeframes[:2])}")
        if aggression_level:
            overview_lines.append(f"Aggression: {aggression_level}")
        if trade_horizon:
            overview_lines.append(f"Horizon: {trade_horizon}")
        if selection_style:
            overview_lines.append(f"Style: {selection_style}")
        if max_positions_limit > 0:
            overview_lines.append(f"Max positions: {max_positions_limit}")
        preview_pairs = selection_result.get("pairs") or []
        if preview_pairs:
            overview_lines.append("Pairs: " + ", ".join(preview_pairs[:6]))
        if branch_name:
            overview_lines.append(f"Branch: {branch_name}")
        if commit_time_display:
            overview_lines.append(f"Commit date: {commit_time_display}")
        commit_display = (commit_descriptor or commit_short).strip()
        if commit_display:
            if len(commit_display) > 80:
                commit_display = commit_display[:77] + "..."
            overview_lines.append(f"Commit: {commit_display}")
        header_tokens = ["[UNIVERSE]"]
        cycle_display = (cycle_descriptor or "").strip()
        if cycle_display:
            header_tokens.append(cycle_display)
        if overview_lines:
            header = " ".join(header_tokens)
            send_tg(header + "\n" + "\n".join(f"- {line}" for line in overview_lines))

    # Recalculate position limit after applying model-requested max_positions_limit.
    position_limit_reached = (
        max_positions_limit > 0
        and open_positions is not None
        and open_positions >= max_positions_limit
    )

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
    if news_pairs and not position_limit_reached:
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

    if (
        updated_universe is None
        and _AI_OFFLINE_ACTIVE_CYCLE is not None
        and safe_int(_AI_OFFLINE_ACTIVE_CYCLE) == safe_int(_CURRENT_CYCLE_NUMBER)
    ):
        required_symbols = set(position_symbols) | set(order_symbols) | set(order_symbols_non_reduce)
        offline_pairs = _offline_select_pairs(
            ex,
            sorted(candidate_pairs_set),
            news_digest=news_headlines,
            required=required_symbols,
            limit=OFFLINE_PAIR_LIMIT,
        )
        universe_state = dict(universe_state or {})
        universe_state["pairs"] = offline_pairs
        universe_state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        universe_state["ai_offline"] = True
        save_universe_cache(universe_state)
        log(
            f"[AI OFFLINE] Universe selection: {', '.join(offline_pairs)} (limit={OFFLINE_PAIR_LIMIT})",
            Fore.YELLOW,
        )

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

    raw_universe_pairs = universe_state.get("pairs") or []
    if position_limit_reached:
        allowed_when_capped = set(position_symbols) | order_symbols | order_symbols_non_reduce
        raw_universe_pairs = [pair for pair in raw_universe_pairs if pair in allowed_when_capped]
    selection_pairs = raw_universe_pairs
    selection_pairs_normalized: list[str] = []
    for raw_pair in selection_pairs:
        if not raw_pair:
            continue
        resolved_pair = normalize_symbol(raw_pair, record_missing=False)
        selection_pairs_normalized.append(resolved_pair or raw_pair)
    selection_pairs = [p for p in selection_pairs_normalized if p]

    if position_limit_reached:
        new_universe_candidates: list[str] = []
        new_universe_set: set[str] = set()
    else:
        new_universe_candidates = [p for p in selection_pairs if p not in position_symbols][:NEW_IDEAS_LIMIT]
        new_universe_set = set(new_universe_candidates)

    news_sorted = sorted(news_priority)
    if position_symbols:
        prioritized_positions = sorted(
            position_symbols,
            key=lambda sym: (0 if sym in spot_position_symbols else 1, sym),
        )
        _append_unique(available_pairs, prioritized_positions, seen_available)
    _append_unique(available_pairs, sorted(order_symbols), seen_available)
    if new_universe_candidates:
        _append_unique(available_pairs, new_universe_candidates, seen_available)
    if news_sorted:
        _append_unique(available_pairs, news_sorted, seen_available)
    remaining_pairs = [p for p in sorted(candidate_pairs_set) if p not in seen_available]
    if position_limit_reached:
        allowed_when_capped = set(position_symbols) | order_symbols | order_symbols_non_reduce
        remaining_pairs = [p for p in remaining_pairs if p in allowed_when_capped]
    _append_unique(available_pairs, remaining_pairs, seen_available)

    if not available_pairs:
        available_pairs = normalized_pair_list[:PAIR_CANDIDATE_LIMIT]
        if not available_pairs:
            available_pairs = list(sorted(markets_set))[:PAIR_CANDIDATE_LIMIT]
    if selection_pairs:
        prioritized = [p for p in selection_pairs if p in available_pairs]
        prioritized += [p for p in available_pairs if p not in prioritized]
        available_pairs = prioritized

    symbol_processing_limit = max(
        MAX_SYMBOLS_PER_CYCLE,
        len(selection_pairs),
        len(raw_universe_pairs),
        len(position_symbols) + len(new_universe_candidates),
        len(position_symbols) + len(order_symbols_non_reduce),
    )
    if rate_limit_backoff:
        rate_cap = max(1, MAX_SYMBOLS_PER_CYCLE // 2)
        symbol_processing_limit = min(symbol_processing_limit, rate_cap)
    if missing_symbols:
        missing_desc = ', '.join(sorted(missing_symbols))
        log(f"[WARN] Removed pairs not listed on Bybit: {missing_desc}", Fore.YELLOW)
        send_tg(f"[WARN] Pairs missing on Bybit: {missing_desc}")
        missing_symbols.clear()
    if symbol_alias_hits:
        alias_desc = ', '.join(f"{src}->{dst}" for src, dst in sorted(symbol_alias_hits.items()))
        log(f"[INFO] Using alias tickers: {alias_desc}", Fore.LIGHTBLACK_EX)
        symbol_alias_hits.clear()

    if len(available_pairs) > symbol_processing_limit:
        priority_pairs = [sym for sym in available_pairs if sym in position_symbols]
        trimmed_list = priority_pairs[:]
        allowed_extras = symbol_processing_limit
        for sym in available_pairs:
            if sym in position_symbols:
                continue
            if len(trimmed_list) >= len(priority_pairs) + allowed_extras:
                break
            trimmed_list.append(sym)
        available_pairs = trimmed_list
    if position_limit_reached:
        log(
            f"[INFO] Position cap reached ({open_positions}/{max_positions_limit}); restricting analysis to existing exposure.",
            Fore.LIGHTBLACK_EX,
        )
    annotated = []
    for _sym in available_pairs:
        exch_sym = _resolve_symbol_alias(_sym) or _sym
        try:
            mi = ex.market(exch_sym)
        except Exception:
            mi = None
        try:
            cat = _infer_market_category(exch_sym, mi)
        except Exception:
            cat = None
        if cat in ("spot", "linear", "inverse"):
            annotated.append(f"{_sym} [{cat}]")
        else:
            annotated.append(_sym)
    log("[INFO] Candidates for analysis: " + ', '.join(annotated), Fore.LIGHTBLACK_EX)

    start_pnl_symbols: set[str] = set(available_pairs)
    start_pnl_symbols.update(position_symbols)
    start_pnl_symbols.update(order_symbols)
    start_pnl_symbols.update(order_symbols_non_reduce)
    start_pnl_symbols.update(open_orders_prefetch.keys())
    start_pnl_list = sorted(sym for sym in start_pnl_symbols if sym)
    start_closed_pnl_value: float | None = None
    start_closed_pnl_count = 0
    if start_pnl_list:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        window_start = now_utc - datetime.timedelta(hours=PNL_LOOKBACK_HOURS)
        start_closed_pnl_value, start_closed_pnl_count, start_warnings, _ = _collect_recent_closed_pnl(
            ex,
            start_pnl_list,
            window_start,
            now_utc,
        )
        for warning_msg in start_warnings[:3]:
            log(warning_msg, Fore.LIGHTBLACK_EX)
        if len(start_warnings) > 3:
            log(f"[PnL] Suppressed {len(start_warnings) - 3} additional warnings (start snapshot).", Fore.LIGHTBLACK_EX)
    start_unreal_total, start_unreal_count = _sum_unrealized_pnl(positions_map)
    if start_closed_pnl_value is not None:
        _emit_realized_pnl_message("start", start_closed_pnl_value, start_closed_pnl_count)
    else:
        message_na = f"PnL ({_pnl_window_label()} closed, start): n/a"
        log(message_na, Fore.LIGHTBLACK_EX)
        try:
            send_tg(message_na)
        except Exception:
            pass
    _emit_unrealized_pnl_message("start", start_unreal_total, start_unreal_count)

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
    selection_style: str | None = None

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
        if selection_style:
            bundle_meta["style"] = selection_style
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
            # Optional model-driven market allocations
            alloc = trade_plan.get("allocations") if isinstance(trade_plan, dict) else None
            if isinstance(alloc, dict):
                spot_val = alloc.get("spot_pct") if isinstance(alloc.get("spot_pct"), (int, float)) else None
                deriv_val = alloc.get("derivatives_pct") if isinstance(alloc.get("derivatives_pct"), (int, float)) else None
                try:
                    if spot_val is not None:
                        spot_val = float(spot_val)
                except Exception:
                    spot_val = None
                try:
                    if deriv_val is not None:
                        deriv_val = float(deriv_val)
                except Exception:
                    deriv_val = None
                if spot_val is not None or deriv_val is not None:
                    if spot_val is None and deriv_val is not None:
                        spot_val = max(0.0, min(1.0, 1.0 - deriv_val))
                    if deriv_val is None and spot_val is not None:
                        deriv_val = max(0.0, min(1.0, 1.0 - spot_val))
                    if spot_val is not None and deriv_val is not None:
                        total = spot_val + deriv_val
                        if total > 0:
                            spot_val /= total
                            deriv_val /= total
                        CURRENT_MARKET_ALLOCATIONS["spot"] = spot_val
                        for key in ("derivatives", "linear", "inverse"):
                            CURRENT_MARKET_ALLOCATIONS[key] = deriv_val
                        log(
                            f"[AI] Market allocations: spot={spot_val:.2%}, derivatives={deriv_val:.2%}",
                            Fore.LIGHTBLACK_EX,
                        )
            trade_next_minutes = trade_plan.get("next_run_minutes")
            if trade_next_minutes is not None and selection_next_time is None:
                try:
                    selection_next_run = float(trade_next_minutes)
                except (TypeError, ValueError):
                    log("[WARN] Invalid next_run_minutes from trade plan.", Fore.YELLOW)
            trade_next_time = trade_plan.get("next_run_time")
            if trade_next_time and selection_next_time is None:
                selection_next_time = trade_next_time
        if trade_plan:
            for decision in (trade_plan.get("decisions") or []):
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
    missing_from_ai: str | None = None
    if selection_missing_symbols:
        missing_from_ai = ", ".join(sorted(set(selection_missing_symbols)))
        log(f"[WARN] Model symbols missing on Bybit: {missing_from_ai}", Fore.YELLOW)
    if selection_next_time or selection_next_run is not None:
        time_hint = selection_next_time if selection_next_time else "n/a"
        run_hint = f"{selection_next_run:.2f}" if (selection_next_run is not None and math.isfinite(selection_next_run)) else "n/a"
        log(f"[SCHED] model scheduling hints: next_run_time={time_hint}, next_run_minutes={run_hint}", Fore.LIGHTBLACK_EX)
        if missing_from_ai:
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
        if sym_order in new_universe_set and sym_order not in seen_priority:
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
    if len(symbols_sequence) > symbol_processing_limit:
        protected_symbols = [sym for sym in symbols_sequence if sym in position_symbols]
        trimmed_sequence = protected_symbols[:]
        allowed_slots = symbol_processing_limit
        for sym in symbols_sequence:
            if sym in position_symbols:
                continue
            if len(trimmed_sequence) >= len(protected_symbols) + allowed_slots:
                break
            trimmed_sequence.append(sym)
        symbols_sequence = trimmed_sequence
    if not symbols_sequence:
        symbols_sequence = available_pairs or list(PAIR_LIST)

    SYMBOL_MARKET_MODE_HINTS.clear()
    decisions_total = 0
    counts = {"open":0,"close":0,"skip":0}
    decisions_details: list[str] = []
    eligible_flat_symbols = 0
    skipped_plan_omitted_symbols = 0
    skipped_plan_unavailable_symbols = 0
    flat_skip_symbols: list[str] = []
    flat_unavailable_symbols: list[str] = []

    trade_plan_failed = trade_plan is None
    for i, sym in enumerate(symbols_sequence, 1):
        has_priority_exposure = sym in position_symbols
        # Snapshot current state for debugging: price, side, size, protection
        current_position = positions_map.get(sym)
        initial_position_amount = safe_float(
            (current_position or {}).get("amount")
            or (current_position or {}).get("contracts")
        )
        if initial_position_amount is None or not math.isfinite(initial_position_amount):
            initial_position_amount = 0.0
        px_ref = None
        try:
            px_ref = _get_position_reference_price(current_position)
        except Exception:
            px_ref = None
        side_label = "LONG" if initial_position_amount > 0 else "SHORT" if initial_position_amount < 0 else "FLAT"
        px_text = f"{px_ref:.4f}" if isinstance(px_ref, (int, float)) and math.isfinite(px_ref or 0) else "n/a"
        open_orders_symbol_snapshot = open_orders_prefetch.get(sym) or []
        prot_orders_snapshot = _extract_protection_orders(open_orders_symbol_snapshot)
        stop_levels: list[float] = []
        take_levels: list[float] = []
        for order in prot_orders_snapshot:
            if not isinstance(order, dict):
                continue
            stop_price = safe_float(order.get("stopPrice") or order.get("triggerPrice"))
            take_price = safe_float(order.get("takeProfitPrice") or order.get("price"))
            if stop_price and math.isfinite(stop_price) and stop_price > 0:
                stop_levels.append(stop_price)
            if take_price and math.isfinite(take_price) and take_price > 0:
                take_levels.append(take_price)
        stop_text = ",".join(f"{lv:.4f}" for lv in sorted(stop_levels)) if stop_levels else "n/a"
        take_text = ",".join(f"{lv:.4f}" for lv in sorted(take_levels)) if take_levels else "n/a"
        log(
            f"[{i}/{len(symbols_sequence)}] {sym}: px={px_text} side={side_label} qty={initial_position_amount:.4f} stop={stop_text} take={take_text}",
            Fore.LIGHTBLACK_EX,
        )
        if AI_HARD_STOP_BUDGET and AI_TOKEN_USAGE_TOTAL >= AI_HARD_STOP_BUDGET and not has_priority_exposure:
            log(f"⛔ Достигнут лимит {AI_HARD_STOP_BUDGET} токенов — дальнейший анализ остановлен", Fore.YELLOW)
            break
        if AI_HARD_STOP_BUDGET and AI_TOKEN_USAGE_TOTAL >= AI_HARD_STOP_BUDGET and has_priority_exposure:
            log(f"[{sym}] превышен лимит токенов, продолжаем из-за открытой позиции", Fore.YELLOW)
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
            protection_refreshed = False
            canonical_lookup = _canonical_decision_symbol(sym)
            preloaded_decision = decisions_map.get(canonical_lookup)
            decision_target = preloaded_decision.get("target") if isinstance(preloaded_decision, dict) else {}
            if isinstance(decision_target, dict) and decision_target:
                merged_target = dict(symbol_meta.get("target") or {})
                merged_target.update(decision_target)
                symbol_meta["target"] = merged_target
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
                    log(f"⚠️ не удалось получить {tf} для {sym}: {exc_fetch}", Fore.YELLOW)
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
                    log(f"❌ не удалось получить базовый таймфрейм {primary_tf} для {sym}: {exc_fetch}", Fore.RED)
                    continue
            df_primary = timeframe_dfs[primary_tf]
            df = df_primary.copy()
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
                "risk": {
                    "base_pct": RISK_PCT,
                    "current_pct": CURRENT_RISK_PCT,
                    "min_pct": MIN_DYNAMIC_RISK_PCT,
                    "max_pct": MAX_DYNAMIC_RISK_PCT,
                    "dynamic_enabled": bool(DYNAMIC_RISK_ENABLED),
                },
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
            if has_position:
                clear_pending_entry(sym, active_user_id)
                symbols_with_position_seen.add(sym)
                symbols_with_position_seen.add(sym)
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
            pending_info = LIMIT_ORDER_PENDING.get(_pending_entry_key(sym, active_user_id))
            if (
                pending_info
                and not has_position
                and open_orders_symbol
                and time.time() - pending_info.get("ts", 0.0) >= LIMIT_ORDER_FALLBACK_SECONDS
            ):
                if _execute_limit_fallback(sym, pending_info, open_orders_symbol):
                    continue
            canonical_lookup = _canonical_decision_symbol(sym)
            preloaded_decision = decisions_map.get(canonical_lookup)
            initial_payload = dict(preloaded_decision) if isinstance(preloaded_decision, dict) else None
            if initial_payload:
                initial_payload["symbol"] = sym

            dec = None
            master_decision_used = False
            rate_limit_error_hit = False
            ai_offline_mode = (
                _AI_OFFLINE_ACTIVE_CYCLE is not None
                and safe_int(_AI_OFFLINE_ACTIVE_CYCLE) == safe_int(_CURRENT_CYCLE_NUMBER)
            )
            if ai_offline_mode and AI_OFFLINE_CANCEL_ENTRIES and open_orders_symbol:
                cancelled_offline: list[str] = []
                for order in open_orders_symbol:
                    if not isinstance(order, dict):
                        continue
                    if _is_reduce_only(order):
                        continue
                    oid = order.get("id")
                    if not oid:
                        continue
                    order_summary = _summarize_order_spec(order)
                    summary_suffix = f": {order_summary}" if order_summary else ""
                    success, err = cancel_order_by_id(ex, sym, str(oid))
                    if success:
                        cancelled_offline.append(f"{oid}{summary_suffix}")
                if cancelled_offline:
                    msg = f"[AI OFFLINE] {sym}: cancelled non-reduce orders: {', '.join(cancelled_offline[:8])}"
                    log(msg, Fore.YELLOW)
                    try:
                        send_tg(msg)
                    except Exception:
                        pass
            if MASTER_DECISIONS_SHARE and not is_master_user:
                dec = _pull_master_decision(sym)
                if dec:
                    master_decision_used = True
                    log(
                        f"[AI SHARE] {sym}: using master decision action={dec.get('action')} reason={(dec.get('reason') or '')[:120]}",
                        Fore.LIGHTBLACK_EX,
                    )
                else:
                    log(
                        f"[WARN] {sym}: master decision unavailable; skipping AI call for follower user {active_user_id}",
                        Fore.YELLOW,
                    )
                    dec = {
                        "symbol": sym,
                        "action": "skip",
                        "reason": "master decision unavailable for follower run",
                    }
            if dec is None:
                if ai_offline_mode:
                    # Offline algorithmic replacement for per-symbol initial decision.
                    news_score = _offline_news_score(news_payload_symbol)
                    try:
                        max_new_left = max(
                            0,
                            min(
                                OFFLINE_MAX_NEW_POSITIONS,
                                max(0, (max_positions_limit or 0) - int(open_positions or 0)) if max_positions_limit else OFFLINE_MAX_NEW_POSITIONS,
                            ),
                        )
                    except Exception:
                        max_new_left = OFFLINE_MAX_NEW_POSITIONS
                    dec = _offline_decision_for_symbol(
                        sym,
                        df,
                        current_position=current_position,
                        news_score=news_score,
                        max_new_positions_left=max_new_left,
                    )
                    log(
                        f"[AI OFFLINE] {sym}: action={dec.get('action')} side={dec.get('side') or 'n/a'} reason={(dec.get('reason') or '')[:140]}",
                        Fore.YELLOW,
                    )
                else:
                    try:
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
                            priority_symbol=has_priority_exposure,
                        )
                    except RateLimitError:
                        rate_limit_backoff = True
                        rate_limit_error_hit = True
                        current_provider = _current_ai_provider()
                        provider_label = current_provider.upper()
                        if current_provider == "openai":
                            _switch_ai_provider_to_fallback("rate limit 429 (decision loop)")
                            msg = (
                                "[WARN] OpenAI rate limit 429: switching to DeepSeek, "
                                "halving symbol cap and deferring next run to 25-55m window."
                            )
                        else:
                            msg = (
                                f"[WARN] {provider_label} rate limit 429: "
                                "halving symbol cap and deferring next run to 25-55m window."
                            )
                        log(msg, Fore.YELLOW)
                        try:
                            send_tg(msg)
                        except Exception:
                            pass
                        break

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
            if MASTER_DECISIONS_SHARE and is_master_user and not master_decision_used and dec:
                _cache_master_decision(sym, dec)
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
                send_tg_decision(confidence_msg)
            else:
                sym_confidence_tag = None
            dec["confidence_value"] = sym_confidence_value or 0.0
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
            action_raw = (dec.get("action") or "skip").strip().lower()
            action_aliases = {
                "replace_orders": "manage",
                "refresh_orders": "manage",
                "update_orders": "manage",
                "maintain": "hold",
                "maintain_position": "hold",
            }
            action = action_aliases.get(action_raw, action_raw)
            if action != action_raw:
                log(f"[AI] {sym}: normalized action {action_raw!r} -> {action!r}", Fore.LIGHTBLACK_EX)
            dec["action"] = action
            side = (dec.get("side") or "").strip().lower()
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
            counted_action = action
            counts[action] = counts.get(action,0)+1
            decisions_total += 1
            side_text = side.lower()
            detail_entry: str | None = None
            orders_activity = False
            open_error: str | None = None
            preallocated_base_asset: str | None = None
            open_executed = False
            entry_errors: list[str] = []
            skip_conf_gate = bool(dec.get("ai_unavailable"))
            if action == "open" and not has_position and not skip_conf_gate:
                base_asset_key = _extract_base_asset(sym)
                exposure_cap = MAX_POSITIONS_PER_BASE
                if base_asset_key and exposure_cap > 0:
                    current_exposure = base_exposure_counts.get(base_asset_key, 0) + pending_base_allocations.get(base_asset_key, 0)
                    if current_exposure >= exposure_cap:
                        limit_reason = f"base exposure limit reached for {base_asset_key} ({current_exposure}/{exposure_cap})"
                        combined_reason = f"{reason}; {limit_reason}" if reason else limit_reason
                        dec["action"] = "skip"
                        dec["reason"] = combined_reason
                        action = "skip"
                        reason = combined_reason
                        log(f"[INFO] {sym}: skipping open - {limit_reason}", Fore.LIGHTBLACK_EX)
                        send_tg(f"[INFO] {sym}: skip open - {limit_reason}")
                        log_open_skip(sym, limit_reason)
                    else:
                        pending_base_allocations[base_asset_key] += 1
                        preallocated_base_asset = base_asset_key

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
                        same_direction = (
                            (position_side in ("long", "buy") and order_side == "buy")
                            or (position_side in ("short", "sell") and order_side == "sell")
                        )
                        scale_flag = _is_truthy_flag(order.get("scaleIn") or order.get("ladder"))
                        order_hint = (order.get("intent") or order.get("tag") or order.get("note") or "").lower()
                        scale_intent = any(keyword in order_hint for keyword in ("scale", "ladder", "step", "stagger"))
                        allow_scale_in = has_position and same_direction and (scale_flag or scale_intent)
                        if allow_scale_in:
                            allow_order = True
                        elif action == "open" and not has_position:
                            allow_order = True
                        elif has_position and position_side and not same_direction:
                            allow_order = True
                    if allow_order:
                        if reduce_only_flag and position_side:
                            desired_close_side = "buy" if position_side in ("short", "sell") else "sell"
                            if not order_side or order_side == position_side:
                                order["side"] = desired_close_side
                            elif order_side != desired_close_side:
                                order["side"] = desired_close_side
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

            parsed_replacements: list[tuple[str | None, list[dict[str, Any]]]] = []
            replace_summaries: list[str] = []
            for entry in replace_list:
                if not isinstance(entry, dict):
                    continue
                cancel_id = (
                    entry.get("cancel")
                    or entry.get("id")
                    or entry.get("old")
                    or entry.get("orderId")
                )
                new_spec = (
                    entry.get("order")
                    or entry.get("new")
                    or entry.get("replacement")
                )
                new_orders: list[dict[str, Any]] = []
                if isinstance(new_spec, dict):
                    new_orders.append(new_spec)
                elif isinstance(new_spec, list):
                    for item in new_spec:
                        if isinstance(item, dict):
                            new_orders.append(item)
                parsed_replacements.append((cancel_id, new_orders))
                summary_parts: list[str] = []
                if cancel_id:
                    summary_parts.append(f"cancel {cancel_id}")
                if new_orders:
                    order_desc = "; ".join(_summarize_order_spec(item) for item in new_orders)
                    summary_parts.append(f"-> {order_desc}")
                if summary_parts:
                    replace_summaries.append(" ".join(summary_parts))
            replacement_orders = []
            cancelled_ids = set()
            cancelled_success = []
            cancel_failures = []
            orders_activity = False

            decision_meta_parts: list[str] = [f"action={action.upper()}"]
            if side:
                decision_meta_parts.append(f"side={side.upper()}")
            if reason:
                decision_meta_parts.append(f"reason={reason}")
            notional_pct_val = dec.get("notional_pct")
            if notional_pct_val is not None:
                try:
                    decision_meta_parts.append(f"notional_pct={_format_notional_pct(notional_pct_val)}")
                except (TypeError, ValueError):
                    decision_meta_parts.append(f"notional_pct={notional_pct_val}")
            if symbol_leverage:
                decision_meta_parts.append(f"lev={symbol_leverage:g}x")
            if sym_confidence_text:
                conf_part = f"conf={sym_confidence_text}"
                if sym_confidence_tag:
                    conf_part += f" ({sym_confidence_tag})"
                decision_meta_parts.append(conf_part)
            log(f"[AI] {sym} decision: " + "; ".join(decision_meta_parts), Fore.CYAN)
            if cancel_candidates:
                log(f"[AI] {sym} cancel_orders: {', '.join(cancel_candidates)}", Fore.LIGHTBLACK_EX)
            if replace_summaries:
                log(f"[AI] {sym} replace_orders: {', '.join(replace_summaries)}", Fore.LIGHTBLACK_EX)
            planned_orders_for_log: list[dict[str, Any]] = list(extra_orders)
            for _, new_orders in parsed_replacements:
                planned_orders_for_log.extend(new_orders)
            if planned_orders_for_log:
                order_summaries = [_summarize_order_spec(item) for item in planned_orders_for_log[:5]]
                log(f"[AI] {sym} orders: {'; '.join(order_summaries)}", Fore.LIGHTBLACK_EX)
                if len(planned_orders_for_log) > 5:
                    log(f"[AI] {sym}: ... +{len(planned_orders_for_log) - 5} more order(s)", Fore.LIGHTBLACK_EX)

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
                    log(f"🔷 Отменён ордер {oid} для {sym} (источник {source})", Fore.LIGHTBLUE_EX)
                else:
                    cancel_failures.append((oid, err))
                    log(f"⚠️ Не удалось отменить ордер {oid} для {sym}: {err}", Fore.YELLOW)

            for oid in cancel_candidates:
                try_cancel(oid, "cancel_orders")

            for cancel_id, new_orders in parsed_replacements:
                if cancel_id:
                    try_cancel(cancel_id, "replace_orders")
                if new_orders:
                    replacement_orders.extend(new_orders)

            if replacement_orders:
                orders_activity = True
                extra_orders.extend(replacement_orders)

            if cancelled_success:
                summary = ", ".join(oid for oid, _ in cancelled_success)
                send_tg(f"📈 Отменены ордера по {sym}: {summary}")
                # Обновляем список открытых ордеров после отмены
                open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)
            if cancel_failures:
                errors = "; ".join(f"{oid}: {err}" for oid, err in cancel_failures)
                send_tg(f"ℹ️ Не удалось отменить ордера по {sym}: {errors}")

            # Downgrade low-confidence opens to skip before handling branches
            mode_hint = SYMBOL_MARKET_MODE_HINTS.get(sym) or {}
            mode_threshold = mode_hint.get("confidence_required")
            mode_label = mode_hint.get("mode")
            open_conf_threshold = OPEN_MIN_CONFIDENCE
            if isinstance(mode_threshold, (int, float)) and mode_threshold > open_conf_threshold:
                open_conf_threshold = mode_threshold
            # Downgrade low-confidence opens to skip before handling branches (skip for offline/manual decisions)
            if action == "open" and not has_position and not dec.get("ai_unavailable"):
                try:
                    conf_val = float(sym_confidence_value) if sym_confidence_value is not None else 0.0
                except Exception:
                    conf_val = 0.0
                if conf_val < open_conf_threshold:
                    extra_note = f" (mode {mode_label})" if mode_label else ""
                    log(
                        f"ℹ️ Пропуск {sym}: confidence {conf_val:.3f} ниже порога {open_conf_threshold:.3f}{extra_note}",
                        Fore.WHITE,
                    )
                    send_tg_decision(
                        f"ℹ️ {sym}: сигнал OPEN пропущен — confidence {conf_val:.3f} ниже порога {open_conf_threshold:.3f}{extra_note}"
                    )
                    action = "skip"
                    dec["action"] = "skip"

            if action == "skip":
                log(f"ℹ️ Пропуск {sym} ({reason})", Fore.WHITE)
                send_tg_decision(f"ℹ️ Пропуск {sym} — {reason or 'причина не указана'}")
            elif action == "close":
                if not current_position or abs(float(current_position.get("amount") or 0)) == 0:
                    log(f"⚠️ Позиция по {sym} отсутствует, нечего закрывать ({reason})", Fore.YELLOW)
                    send_tg_decision(f"ℹ️ {sym}: закрытие пропущено — нет открытой позиции")
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
                            log(f"⚠️ Закрыть позицию {sym} ({reason})", Fore.YELLOW)
                            send_tg(f"ℹ️ Закрыт {sym} {close_side.upper()} {qty:.4f} — {reason or 'причина не указана'}")
                            positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=available_pairs)
                            base_asset_after_close = _extract_base_asset(sym)
                            if base_asset_after_close:
                                existing = base_exposure_counts.get(base_asset_after_close, 0)
                                if existing > 0:
                                    base_exposure_counts[base_asset_after_close] = existing - 1
                            base_exposure_counts = _build_base_exposure_map(positions_map)
                            current_position = positions_map.get(sym)
                        except Exception as e:
                            err_text = str(e)
                            log(f"❌ Ошибка закрытия {sym}: {err_text}", Fore.RED)
                            send_tg(f"ℹ️ Ошибка закрытия для {sym}: {err_text}")
            elif action == "hold":
                log(f"🔷 Удерживаем {sym} ({reason})", Fore.BLUE)
                send_tg_decision(f"ℹ️ {sym}: удерживаем позицию — {reason or 'причина не указана'}")
                if current_position and abs(float(current_position.get('amount') or 0)) > 0:
                    updated_orders, refreshed = _refresh_position_protection_if_possible(
                        ex,
                        sym,
                        current_position,
                        df,
                        df_primary,
                        open_orders_symbol,
                        symbol_meta,
                    )
                    if refreshed:
                        protection_refreshed = True
                    if refreshed and isinstance(updated_orders, list):
                        open_orders_symbol = updated_orders
                        open_orders_cache[sym] = updated_orders
            elif action == "manage":
                log(f"🔧 Управляем {sym} ({reason})", Fore.BLUE)
                send_tg_decision(f"ℹ️ {sym}: управление позицией — {reason or 'причина не указана'}")
                if current_position and abs(float(current_position.get('amount') or 0)) > 0:
                    updated_orders, refreshed = _refresh_position_protection_if_possible(
                        ex,
                        sym,
                        current_position,
                        df,
                        df_primary,
                        open_orders_symbol,
                        symbol_meta,
                    )
                    if refreshed:
                        protection_refreshed = True
                    if refreshed and isinstance(updated_orders, list):
                        open_orders_symbol = updated_orders
                        open_orders_cache[sym] = updated_orders
            elif action == "needs":
                requested_context = dec.get("requested_context")
                context_parts: list[str] = []
                if isinstance(requested_context, dict):
                    for key, value in requested_context.items():
                        context_parts.append(f"{key}={value}")
                elif isinstance(requested_context, (list, tuple)):
                    context_parts.extend(str(item) for item in requested_context if item)
                elif requested_context:
                    context_parts.append(str(requested_context))
                context_desc = ", ".join(context_parts) if context_parts else "context not specified"
                log(f"ℹ️ Needs data for {sym}: {reason} (requested {context_desc})", Fore.WHITE)
                send_tg(
                    f"ℹ️ {sym}: needs additional data — {reason or 'reason not provided'} (requested {context_desc})"
                )
                continue
            elif action == "open":
                if current_position and abs(float(current_position.get("amount") or 0)) > 0:
                    log(f"⚠️ Позиция по {sym} уже открыта (side={current_position.get('side')}, amount={current_position.get('amount')}), пропускаем повторное открытие", Fore.YELLOW)
                    send_tg_decision(f"ℹ️ {sym}: позиция уже открыта, сигнал open пропущен")
                    updated_orders, refreshed = _refresh_position_protection_if_possible(
                        ex,
                        sym,
                        current_position,
                        df,
                        df_primary,
                        open_orders_symbol,
                        symbol_meta,
                    )
                    if refreshed:
                        protection_refreshed = True
                    if refreshed and isinstance(updated_orders, list):
                        open_orders_symbol = updated_orders
                        open_orders_cache[sym] = updated_orders
                    continue
                elif max_positions_limit > 0 and open_positions is not None and open_positions >= max_positions_limit:
                    log(f"⛔ Лимит открытых позиций достигнут ({open_positions}/{max_positions_limit}), пропускаем {sym}", Fore.YELLOW)
                    send_tg(f"⛔ Лимит открытых позиций достигнут ({open_positions}/{max_positions_limit}), {sym} пропущен")
                else:
                    log(f"✅ Сигнал {side.upper()} ({reason})", Fore.GREEN)
                    send_tg(f"ℹ️ {sym} {side.upper()} — {reason or 'причина не указана'}")
                    if df.empty:
                        log(f"⚠️ Нет данных 30m для {sym}, пропускаем открытие", Fore.YELLOW)
                        send_tg(f"ℹ️ {sym}: недостаточно данных для открытия позиции")
                        continue
                    df["atr"] = atr(df,14)
                    trade_rules = _get_symbol_trade_rules(ex, sym)
                    min_qty_rule = float(trade_rules.get("min_qty") or 0.0)
                    qty_step_rule = float(trade_rules.get("qty_step") or 0.0)
                    if qty_step_rule > 0 and (min_qty_rule <= 0 or qty_step_rule > min_qty_rule):
                        min_qty_rule = max(min_qty_rule, qty_step_rule)
                    exchange_min_notional = trade_rules.get("min_notional") or 0.0
                    min_notional_rule = trade_rules.get("min_notional") or 0.0
                    env_min_notional = float(MIN_NOTIONAL_USDT or 0.0)
                    effective_min_notional = max(exchange_min_notional, env_min_notional)
                    min_notional_required = max(effective_min_notional, min_notional_rule or 0.0)
                    log(
                        f"[INFO] {sym}: exchange min {exchange_min_notional:.2f} USDT; "
                        f"env min {env_min_notional:.2f} USDT; effective min {effective_min_notional:.2f} USDT",
                        Fore.LIGHTBLACK_EX,
                    )
                    last_row = df.iloc[-1]
                    price = float(last_row.get("close") or 0)
                    atrv = float(last_row.get("atr") or 0)
                    if not (math.isfinite(price) and math.isfinite(atrv) and atrv > 0):
                        log(f"⚠️ Не удалось рассчитать ATR/цену для {sym}, пропуск сигнала", Fore.YELLOW)
                        send_tg(f"ℹ️ {sym}: нет валидных значений ATR для расчёта размера")
                        continue
                    atr_ratio = atrv / price if price > 0 else 0.0
                    low_vol_multiplier = 1.0
                    if VOL_GUARD_ENABLED and price > 0:
                        if ATR_GUARD_MAX_RATIO > 0 and atr_ratio >= ATR_GUARD_MAX_RATIO:
                            guard_msg = (
                                f"[RISK] {sym}: ATR/price {atr_ratio:.2%} ≥ guard {ATR_GUARD_MAX_RATIO:.2%}, skip entry"
                            )
                            log(guard_msg, Fore.YELLOW)
                            send_tg(guard_msg)
                            log_open_skip(sym, f"atr ratio {atr_ratio:.2%} above guard")
                            continue
                        if (
                            ATR_GUARD_LOW_BOOST > 1.0
                            and ATR_GUARD_MIN_RATIO > 0
                            and 0 < atr_ratio <= ATR_GUARD_MIN_RATIO
                        ):
                            low_vol_multiplier = ATR_GUARD_LOW_BOOST
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
                        log(f"[INFO] {user_tag} {sym}: existing {side.upper()} @ {dup_price} still active; skipping new entry", Fore.LIGHTBLACK_EX)
                        send_tg(f"[INFO] {user_tag} {sym}: existing {side.upper()} @ {dup_price} still active, new order skipped")
                        log_open_skip(sym, f"duplicate limit @ {dup_price}")
                        continue
                    explicit_qty, explicit_notional = _extract_decision_position_size(dec, symbol_meta)
                    qty = None
                    notional = None
                    pct = _extract_decision_pct(dec, symbol_meta)
                    if pct is not None and equity and equity > 0:
                        notional = equity * pct
                        if notional is None or not math.isfinite(notional) or notional <= 0:
                            log(f"[WARN] {user_tag} {sym}: invalid notional from percentage {pct:.2%}", Fore.YELLOW)
                            send_tg(f"[WARN] {user_tag} {sym}: invalid percentage size {pct:.2%}")
                            log_open_skip(sym, "percentage notional invalid")
                            continue
                        if price is None or not math.isfinite(price) or price <= 0:
                            log(f"[WARN] {user_tag} {sym}: price invalid for percentage sizing", Fore.YELLOW)
                            send_tg(f"[WARN] {user_tag} {sym}: cannot size percentage order without price")
                            log_open_skip(sym, "percentage price invalid")
                            continue
                        qty = notional / price
                        log(f"[INFO] {user_tag} {sym}: applying percentage {pct:.2%} -> notional {notional:.2f} USDT", Fore.LIGHTBLACK_EX)
                    elif explicit_qty is not None or explicit_notional is not None:
                        if price is None or not math.isfinite(price) or price <= 0:
                            log(f"[WARN] {user_tag} {sym}: invalid price for explicit size", Fore.YELLOW)
                            send_tg(f"[WARN] {user_tag} {sym}: model sent explicit size but no usable price - skipping trade")
                            log_open_skip(sym, "explicit price invalid")
                            continue
                        qty = explicit_qty if explicit_qty is not None else explicit_notional / price
                        notional = qty * price
                    else:
                        risk_distance = abs(price - sl)
                        if risk_distance <= 0 or not math.isfinite(risk_distance):
                            log(f"[WARN] {user_tag} {sym}: unable to compute risk distance", Fore.YELLOW)
                            send_tg(f"[WARN] {user_tag} {sym}: failed to compute stop-based risk, skipping")
                            log_open_skip(sym, "risk distance invalid")
                            continue
                        risk_budget_base = _select_risk_budget_base(equity, available_margin)
                        try:
                            market_snapshot = ex.market(sym)
                        except Exception:
                            market_snapshot = None
                        category = _infer_market_category(sym, market_snapshot) or "derivatives"
                        alloc = CURRENT_MARKET_ALLOCATIONS.get(
                            category,
                            CURRENT_MARKET_ALLOCATIONS.get("derivatives", 1.0),
                        )
                        try:
                            alloc = float(alloc)
                        except (TypeError, ValueError):
                            alloc = 1.0
                        if not math.isfinite(alloc) or alloc <= 0:
                            alloc = 1.0
                        alloc = max(0.0, min(1.0, alloc))
                        risk_budget_base *= alloc
                        # Используем динамический риск, если он включён и валиден,
                        # иначе возвращаемся к базовому RISK_PCT.
                        raw_risk_pct = (
                            CURRENT_RISK_PCT
                            if DYNAMIC_RISK_ENABLED
                            and CURRENT_RISK_PCT
                            and math.isfinite(CURRENT_RISK_PCT)
                            else RISK_PCT
                        )
                        position_risk_pct = min(
                            MAX_DYNAMIC_RISK_PCT,
                            max(MIN_DYNAMIC_RISK_PCT, raw_risk_pct),
                        )
                        if position_risk_pct <= 0 or not math.isfinite(position_risk_pct):
                            position_risk_pct = RISK_PCT
                        risk_capital = risk_budget_base * position_risk_pct
                        if risk_capital <= 0:
                            log(f"[WARN] {user_tag} {sym}: risk budget is zero (available margin {available_margin:.2f} USDT)", Fore.YELLOW)
                            send_tg(f"[WARN] {user_tag} {sym}: insufficient free margin ({available_margin:.2f} USDT)")
                            log_open_skip(sym, "risk budget zero")
                            continue
                        qty = risk_capital / risk_distance
                    qty = _apply_qty_rules(qty, min_qty=min_qty_rule, qty_step=qty_step_rule)
                    if not math.isfinite(qty) or qty <= 0:
                        log(f"[WARN] {user_tag} {sym}: computed quantity is invalid", Fore.YELLOW)
                        log_open_skip(sym, "quantity invalid")
                        continue
                    notional = qty * price
                    if not math.isfinite(notional) or notional <= 0:
                        log(f"[WARN] {user_tag} {sym}: computed notional is invalid", Fore.YELLOW)
                        log_open_skip(sym, "notional invalid")
                        continue
                    log_user(
                        f"OPEN PLAN {sym}: side={side or '?'} qty={qty:.6f} notional={notional:.2f} sl={sl:.2f} tp={tp:.2f}"
                    )
                    if low_vol_multiplier > 1.0:
                        boosted_qty = qty * low_vol_multiplier
                        boosted_qty = _apply_qty_rules(boosted_qty, min_qty=min_qty_rule, qty_step=qty_step_rule)
                        if boosted_qty > qty * (1.0 + 1e-6):
                            qty = boosted_qty
                            notional = qty * price
                            log_user(
                                f"OPEN ADJUST {sym}: low-vol boost x{low_vol_multiplier:.2f} -> qty={qty:.6f}, notional={notional:.2f}"
                            )
                    if notional + NOTIONAL_EPSILON < min_notional_required:
                        if AUTO_MIN_NOTIONAL:
                            min_qty_from_notional = (
                                min_notional_required / price if price > 0 else min_notional_required
                            )
                            target_qty = (
                                max(min_qty_rule, min_qty_from_notional)
                                if min_qty_rule
                                else min_qty_from_notional
                            )
                            qty = _apply_qty_rules(target_qty, min_qty=min_qty_rule, qty_step=qty_step_rule)
                            notional = qty * price
                            log_user(
                                f"OPEN ADJUST {sym}: increasing qty to meet min notional {min_notional_required:.2f} USDT -> qty={qty:.6f}, notional={notional:.2f}"
                            )
                        else:
                            log_open_skip(
                                sym,
                                f"notional {notional:.2f} USDT below minimum {min_notional_required:.2f} USDT",
                            )
                            continue
                    effective_margin = max(0.0, available_margin * ORDER_MARGIN_UTILIZATION)
                    max_notional = effective_margin * max(1, symbol_leverage)
                    if max_notional <= 0:
                        log(f"[WARN] {user_tag} {sym}: usable margin exhausted", Fore.YELLOW)
                        send_tg(f"[WARN] {user_tag} {sym}: usable margin exhausted")
                        log_open_skip(sym, "usable margin exhausted")
                        continue
                    if (
                        AUTO_MARGIN_SCALE
                        and AUTO_MARGIN_SCALE_RATIO > 0
                        and max_notional > 0
                        and price
                        and price > 0
                    ):
                        ratio_to_use = AUTO_MARGIN_SCALE_RATIO
                        confidence_value = dec.get("confidence_value") or 0.0
                        if (
                            confidence_value >= AI_CONFIDENCE_THRESHOLD
                            and AUTO_MARGIN_CONFIDENCE_MULT > 1.0
                        ):
                            ratio_to_use = min(1.0, ratio_to_use * AUTO_MARGIN_CONFIDENCE_MULT)
                        desired_notional = max_notional * ratio_to_use
                        if desired_notional > notional:
                            scaled_notional = min(max_notional, max(desired_notional, notional))
                            qty = scaled_notional / price
                            notional = qty * price
                            log_user(
                                f"OPEN ADJUST {sym}: scaling to margin {notional:.2f} USDT (ratio {ratio_to_use:.2f}) -> qty={qty:.6f}"
                            )
                    if notional > max_notional + NOTIONAL_EPSILON:
                        qty = _clamp_qty_to_max_notional(qty, price, max_notional, qty_step_rule)
                        if qty <= 0:
                            log(f"[WARN] {user_tag} {sym}: usable margin cannot satisfy minimum trade size", Fore.YELLOW)
                            send_tg(f"[WARN] {user_tag} {sym}: margin too small for minimum order")
                            log_open_skip(sym, "margin cannot satisfy minimum size")
                            continue
                        qty = _apply_qty_rules(qty, min_qty=min_qty_rule, qty_step=qty_step_rule)
                        notional = qty * price
                    if notional + NOTIONAL_EPSILON < min_notional_required:
                        log(f"[WARN] {user_tag} {sym}: margin {available_margin:.2f} USDT below exchange minimum order size", Fore.YELLOW)
                        send_tg(f"[WARN] {user_tag} {sym}: margin {available_margin:.2f} USDT below minimum order size")
                        log_open_skip(sym, "margin below exchange minimum order size")
                        continue
                    margin_required = notional / symbol_leverage if symbol_leverage else notional
                    if margin_required > effective_margin:
                        log(f'[WARN] {user_tag} {sym}: required margin {margin_required:.2f} USDT exceeds usable {effective_margin:.2f} USDT (total {available_margin:.2f} USDT, ORDER_MARGIN_UTILIZATION={ORDER_MARGIN_UTILIZATION}), skipping order', Fore.YELLOW)
                        send_tg(f'[WARN] {user_tag} {sym}: required margin {margin_required:.2f} USDT exceeds usable {effective_margin:.2f} USDT, skipping')
                        log_open_skip(sym, f"required margin {margin_required:.2f} > usable {effective_margin:.2f}")
                        continue
                    qty = _apply_qty_rules(qty, min_qty=min_qty_rule, qty_step=qty_step_rule)
                    try:
                        qty = float(ex.amount_to_precision(sym, qty))
                    except Exception:
                        qty = float(round(qty, 8))
                    if qty <= 0:
                        log(f"⚠️ После округления объём стал ? 0 для {sym}", Fore.YELLOW)
                        continue
                    notional = qty * price
                    if notional + NOTIONAL_EPSILON < min_notional_required:
                        log(f"[WARN] {user_tag} {sym}: notional {notional:.2f} USDT below minimum {min_notional_required:.2f} USDT, skipping", Fore.YELLOW)
                        send_tg(f"[WARN] {user_tag} {sym}: size {notional:.2f} USDT below exchange minimum {min_notional_required:.2f} USDT")
                        continue
                    try:
                        # Determine category (spot/derivatives) and prepare base params
                        try:
                            _mi = ex.market(sym)
                        except Exception:
                            _mi = None
                        category = _infer_market_category(sym, _mi) or "linear"
                        position_idx = get_position_idx(side)
                        base_params = {"takeProfit": tp, "stopLoss": sl, "tpSlMode": "Full", "reduceOnly": False}
                        if position_idx is not None and category != "spot":
                            base_params["positionIdx"] = position_idx
                        # Sanitize for spot and set category for Bybit v5
                        base_params = _sanitize_order_params_for_category(base_params, category)
                        if category == "spot" and side.lower() == "sell":
                            ok, err = _spot_funds_sufficient(ex, sym, side, qty, price)
                            if not ok:
                                log(f"[WARN] Skipping spot SELL for {sym}: {err}", Fore.YELLOW)
                                send_tg(f"[WARN] {sym}: spot sell skipped — {err}")
                                continue
                        scheme = ENTRY_LADDER_SCHEME if ENTRY_LADDER_SCHEME else [(1.0, 0.0)]
                        normalized_entries: list[tuple[float, float]] = []
                        for share, offset in scheme:
                            share_val = float(share) if isinstance(share, (int, float)) else 0.0
                            offset_val = float(offset) if isinstance(offset, (int, float)) else 0.0
                            if not math.isfinite(share_val) or share_val <= 0:
                                continue
                            if not math.isfinite(offset_val) or offset_val < 0:
                                offset_val = 0.0
                            normalized_entries.append((share_val, offset_val))
                        if not normalized_entries:
                            normalized_entries = [(1.0, 0.0)]
                        if min_qty_rule and len(normalized_entries) > 1:
                            min_total_layers = min_qty_rule * len(normalized_entries)
                            if qty < min_total_layers:
                                normalized_entries = [(1.0, 0.0)]
                        # Compute sum of shares to normalize per-layer weights
                        ratio_total = sum(item[0] for item in normalized_entries) or 1.0
                        log(f"[DEBUG] {sym} - normalized_entries={normalized_entries}, ratio_total={ratio_total}", Fore.LIGHTBLACK_EX)
                        log(f"[DEBUG] {sym} {side.upper()} - qty={qty:.4f}, notional={notional:.2f}, margin_required={notional/symbol_leverage if symbol_leverage else notional:.2f}, effective_margin={effective_margin:.2f}, sl={sl:.2f}, tp={tp:.2f}", Fore.LIGHTBLACK_EX)
                        # Ensure fallback_price is defined for fallback path
                        fallback_price = price
                        remaining_qty = qty
                        entry_summaries: list[str] = []
                        entry_created = 0
                        total_margin_used = 0.0
                        side_lower = side.lower()
                        total_layers = len(normalized_entries)
                        for idx, (share_val, offset_val) in enumerate(normalized_entries):
                            try:
                                weight = share_val / ratio_total if ratio_total else 0.0
                            except Exception as exc:
                                log(f"[ERROR] {sym}: failed to compute weight for entry layer: {exc}; normalized_entries={normalized_entries}, ratio_total={ratio_total}", Fore.RED)
                                weight = share_val / (sum(it[0] for it in normalized_entries) or 1.0)
                            target_qty = qty * weight if idx < total_layers - 1 else remaining_qty
                            target_qty = min(target_qty, remaining_qty)
                            if target_qty <= 0:
                                continue
                            layer_price = price - offset_val * atrv if side_lower == "buy" else price + offset_val * atrv
                            if layer_price is None or not math.isfinite(layer_price) or layer_price <= 0:
                                continue
                            duplicate_match, duplicate_qty = _has_active_limit_at_price(open_orders_symbol, side_lower, layer_price)
                            if duplicate_match:
                                consumed_qty = target_qty if duplicate_qty <= 0 else min(target_qty, max(duplicate_qty, 0.0))
                                remaining_qty = max(0.0, remaining_qty - consumed_qty)
                                # Do not count duplicates as created entries
                                entry_summaries.append(f"existing limit @ {layer_price:.2f} (qty~{consumed_qty:.4f})")
                                log_user(
                                    f"OPEN DUPLICATE {sym}: existing limit @ {layer_price:.2f} (qty~{consumed_qty:.4f})",
                                )
                                continue
                            try:
                                precise_qty = float(ex.amount_to_precision(sym, target_qty))
                            except Exception:
                                precise_qty = float(round(target_qty, 8))
                            if precise_qty <= 0:
                                continue
                            if min_qty_rule and precise_qty < min_qty_rule:
                                adjusted_min_qty = min(min_qty_rule, max(remaining_qty, 0.0))
                                if adjusted_min_qty <= 0:
                                    continue
                                try:
                                    precise_qty = float(ex.amount_to_precision(sym, adjusted_min_qty))
                                except Exception:
                                    precise_qty = float(round(adjusted_min_qty, 8))
                                if precise_qty < min_qty_rule:
                                    continue
                            layer_notional = precise_qty * layer_price
                            order_price = layer_price
                            order_notional = layer_notional
                            fallback_used = False
                            if layer_notional + NOTIONAL_EPSILON < min_notional_required:
                                min_qty_needed = min_notional_required / layer_price if layer_price > 0 else min_notional_required
                                min_qty_target = _apply_qty_rules(min_qty_needed, min_qty=min_qty_rule, qty_step=qty_step_rule)
                                min_qty_target = min(min_qty_target, remaining_qty)
                                if min_qty_target <= 0:
                                    raise RuntimeError("no entry orders placed")
                                try:
                                    precise_qty = float(ex.amount_to_precision(sym, min_qty_target))
                                except Exception:
                                    precise_qty = float(round(min_qty_target, 8))
                                if precise_qty <= 0:
                                    raise RuntimeError("no entry orders placed")
                                if min_qty_rule and precise_qty < min_qty_rule - 1e-8:
                                    raise RuntimeError("no entry orders placed")
                                layer_notional = precise_qty * layer_price
                                if layer_notional + NOTIONAL_EPSILON < min_notional_required:
                                    raise RuntimeError("no entry orders placed")
                            fallback_notional = precise_qty * fallback_price
                            if fallback_notional + NOTIONAL_EPSILON < min_notional_required:
                                min_qty_needed = min_notional_required / fallback_price if fallback_price > 0 else min_notional_required
                                min_qty_target = _apply_qty_rules(min_qty_needed, min_qty=min_qty_rule, qty_step=qty_step_rule)
                                min_qty_target = min(min_qty_target, qty)
                                if min_qty_target <= 0:
                                    raise RuntimeError("no entry orders placed")
                                try:
                                    precise_qty = float(ex.amount_to_precision(sym, min_qty_target))
                                except Exception:
                                    precise_qty = float(round(min_qty_target, 8))
                                if precise_qty <= 0:
                                    raise RuntimeError("no entry orders placed")
                                if min_qty_rule and precise_qty < min_qty_rule - 1e-8:
                                    raise RuntimeError("no entry orders placed")
                                fallback_notional = precise_qty * fallback_price
                                if fallback_notional + NOTIONAL_EPSILON < min_notional_required:
                                    raise RuntimeError("no entry orders placed")
                                fallback_used = True
                                order_price = fallback_price
                                order_notional = fallback_notional
                            if not fallback_used:
                                order_price = layer_price
                                order_notional = precise_qty * order_price
                            layer_params = dict(base_params)
                            layer_params = _sanitize_order_params_for_category(layer_params, category)
                            log(
                                f"[EX] create {sym} {side_lower}/limit qty={precise_qty:.6f} price={order_price:.4f} params={layer_params}",
                                Fore.LIGHTBLACK_EX,
                            )
                            order_result = ex.create_order(sym, "limit", side, precise_qty, order_price, layer_params)
                            log(f"[EX] ok {sym} {side_lower}/limit -> {order_result}", Fore.LIGHTBLACK_EX)
                            try:
                                _append_user_bybit_log(
                                    USER_ID,
                                    f"ORDER: {sym} {side.upper()} {precise_qty:.6f}@{order_price:.4f} -> {order_result}",
                                )
                            except Exception:
                                pass
                            _append_trade_log(
                                f"{user_tag} ORDER {sym} {side.upper()} qty={precise_qty:.6f} price={order_price:.4f} -> {order_result}"
                            )
                            open_executed = True
                            entry_created = 1
                            remaining_qty = max(0.0, qty - precise_qty)
                            total_margin_used = order_notional / symbol_leverage if symbol_leverage else order_notional
                            entry_summaries.append(
                                f"{precise_qty:.4f} @ {order_price:.2f} ({'fallback' if fallback_used else 'limit'}, margin {total_margin_used:.2f} USDT)"
                            )
                        log(f"✅ Ордеры {sym} {side.upper()} ({entry_created}) SL:{sl:.2f} TP:{tp:.2f}", Fore.GREEN)
                        send_tg(
                            f"ℹ️ {sym} {side.upper()} входы:\n"
                            + "\n".join(f"- {summary}" for summary in entry_summaries)
                            + f"\nSL {sl:.2f} TP {tp:.2f}\nМаржа {total_margin_used:.2f} USDT, плечо x{symbol_leverage}"
                        )
                        open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)
                        positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=available_pairs)
                        current_position = positions_map.get(sym)
                        base_exposure_counts = _build_base_exposure_map(positions_map)
                        if open_executed and qty and qty > 0 and side:
                            record_pending_entry(sym, qty, side, active_user_id)

                    except Exception as e:
                        err_text = str(e)
                        entry_errors.append(err_text)
                        log_open_skip(sym, err_text)
                        open_error = err_text
                        log(f"❌ Ошибка ордера: {err_text}", Fore.RED)
                        send_tg(f"ℹ️ Ошибка ордера для {sym}: {err_text}")
                if preallocated_base_asset:
                    pending_val = pending_base_allocations.get(preallocated_base_asset, 0)
                    if pending_val > 0:
                        pending_base_allocations[preallocated_base_asset] = pending_val - 1
                    if not open_error and open_executed:
                        base_exposure_counts[preallocated_base_asset] = base_exposure_counts.get(preallocated_base_asset, 0) + 1
            else:
                if action not in ("hold", "manage", "none", "", None):
                    log(f"⚠️ Неизвестное действие \"{action}\" для {sym}, обработка только дополнительных ордеров", Fore.YELLOW)

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
                executed, actions_performed, order_errors = execute_extra_orders(
                    ex,
                    sym,
                    extra_orders,
                    equity=equity,
                    current_position=current_position,
                    open_orders=open_orders_symbol,
                    available_margin=available_margin,
                    symbol_leverage=symbol_leverage,
                    max_limits_per_side=MAX_NON_REDUCE_LIMITS_PER_SIDE,
                )
                if executed:
                    orders_activity = True
                    send_tg("🟢 " + sym + " доп. ордера:\n- " + "\n- ".join(executed))
                if actions_performed:
                    orders_activity = True
                    positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=available_pairs)
                    current_position = positions_map.get(sym)
                    open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)

            if (
                not protection_refreshed
                and current_position
                and abs(float(current_position.get("amount") or 0)) > 0
            ):
                updated_orders, refreshed = _refresh_position_protection_if_possible(
                    ex,
                    sym,
                    current_position,
                    df,
                    df_primary,
                    open_orders_symbol,
                    symbol_meta,
                )
                if refreshed:
                    protection_refreshed = True
                if refreshed and isinstance(updated_orders, list):
                    open_orders_symbol = updated_orders
                    open_orders_cache[sym] = updated_orders

            final_position_payload = positions_map.get(sym)
            final_position_amount = safe_float(
                (final_position_payload or {}).get("amount")
                or (final_position_payload or {}).get("contracts")
            )
            if final_position_amount is None or not math.isfinite(final_position_amount):
                final_position_amount = 0.0
            if abs(final_position_amount) > 0:
                symbols_with_position_seen.add(sym)
            if abs(final_position_amount) > 0:
                symbols_with_position_seen.add(sym)
            final_protection_orders = _extract_protection_orders(open_orders_symbol)
            final_protection_signature = _protection_orders_signature(open_orders_symbol)
            protection_changed = initial_protection_signature != final_protection_signature
            amount_diff = abs(final_position_amount - initial_position_amount)
            amount_tolerance = max(abs(initial_position_amount), abs(final_position_amount)) * 1e-6 + 1e-8
            position_changed = amount_diff > amount_tolerance
            initial_orders_snapshot = _format_protection_snapshot(initial_protection_orders, current_position)
            final_orders_snapshot = _format_protection_snapshot(final_protection_orders, final_position_payload)
            size_change_label = _describe_size_change(
                initial_position_amount, final_position_amount, amount_tolerance
            )
            if protection_changed:
                orders_activity = True

            open_success = (
                action == "open"
                and not open_error
                and (position_changed or (final_position_amount is not None and abs(final_position_amount) > 0))
            )
            open_pending = action == "open" and open_executed and not open_success and not open_error

            # Enrich logs with regime and entry type when available.
            regime = (dec.get("regime") if isinstance(dec, dict) else None) or symbol_meta.get("regime") or "n/a"
            entry_kind = None
            if side_text in ("buy", "long"):
                entry_kind = "buy_limit"
            elif side_text in ("sell", "short"):
                entry_kind = "sell_limit"
            # Helper to decorate base message with regime/entry meta.
            def _with_meta(base: str) -> str:
                extra_bits: list[str] = []
                if regime and regime not in {"n/a", ""}:
                    extra_bits.append(f"regime={regime}")
                if entry_kind and action == "open":
                    extra_bits.append(f"entry={entry_kind}")
                if extra_bits:
                    return f"{base} [{'; '.join(extra_bits)}]"
                return base

            if detail_entry is None:
                if action == "open":
                    if open_error:
                        detail_entry = _with_meta(f"[{sym}] - failed to open position (error: {open_error})")
                    elif open_success:
                        direction = "LONG" if side_text in ("buy", "long") else "SHORT" if side_text in ("sell", "short") else ""
                        entry_price = _get_position_reference_price(final_position_payload)
                        entry_text = f" @ {entry_price:.4f}" if entry_price is not None else ""
                        detail_entry = _with_meta(
                            f"[{sym}] - opened {direction or 'position'} {abs(final_position_amount):.4f}{entry_text} "
                            f"(lev x{symbol_leverage}); orders {final_orders_snapshot}"
                        )
                    elif open_pending:
                        direction = "LONG" if side_text in ("buy", "long") else "SHORT" if side_text in ("sell", "short") else ""
                        pending_parts = ["waiting fill"]
                        if entry_created:
                            plural = "layer" if entry_created == 1 else "layers"
                            pending_parts.append(f"{entry_created} {plural}")
                        pending_desc = ", ".join(pending_parts)
                        detail_entry = _with_meta(f"[{sym}] - entry orders placed ({pending_desc}) {direction or ''} (lev x{symbol_leverage})")
                        if entry_errors:
                            detail_entry += f" (last error: {entry_errors[-1]})"
                    else:
                        # open_skip_notes is available in apply_trade_plan_snapshot; in the main run_cycle
                        # summary it may be undefined, so fall back to a local empty mapping.
                        combined_reasons: list[str] = []
                        seen_reasons: set[str] = set()
                        try:
                            existing_reasons = open_skip_notes.get(sym) or []  # type: ignore[name-defined]
                        except NameError:
                            existing_reasons = []
                        for reason in existing_reasons:
                            if reason and reason not in seen_reasons:
                                combined_reasons.append(reason)
                                seen_reasons.add(reason)
                        for err in entry_errors:
                            if err and err not in seen_reasons:
                                combined_reasons.append(err)
                                seen_reasons.add(err)
                        if combined_reasons:
                            detail_entry = _with_meta(f"[{sym}] - open request skipped ({'; '.join(combined_reasons[-3:])})")
                        else:
                            detail_entry = _with_meta(f"[{sym}] - open request skipped")
                elif action == "close":
                    direction = "LONG" if side_text in ("buy", "long") else "SHORT" if side_text in ("sell", "short") else ""
                    close_price = _get_position_reference_price(current_position) or _get_position_reference_price(final_position_payload)
                    close_text = f" @ {close_price:.4f}" if close_price is not None else ""
                    detail_entry = _with_meta(
                        f"[{sym}] - closed {direction or 'position'} {abs(initial_position_amount):.4f}{close_text} "
                        f"(lev x{symbol_leverage})"
                    )
                elif action == "manage":
                    orders_desc = (
                        f"orders updated (was {initial_orders_snapshot}; now {final_orders_snapshot})"
                        if orders_activity
                        else f"orders unchanged ({final_orders_snapshot})"
                    )
                    pos_label_before = _format_position_snapshot(initial_position_amount)
                    pos_label_after = _format_position_snapshot(final_position_amount)
                    # Classify manage into higher-level categories for readability.
                    manage_label = "change_orders"
                    if abs(initial_position_amount) <= amount_tolerance and abs(final_position_amount) > amount_tolerance:
                        manage_label = "open"
                    elif abs(initial_position_amount) > amount_tolerance and abs(final_position_amount) <= amount_tolerance:
                        manage_label = "close"
                    elif initial_position_amount * final_position_amount < -amount_tolerance:
                        manage_label = "flip"
                    elif abs(final_position_amount) > abs(initial_position_amount) + amount_tolerance:
                        manage_label = "increase"
                    elif abs(final_position_amount) < abs(initial_position_amount) - amount_tolerance:
                        manage_label = "reduce"
                    if initial_position_amount == 0.0 and final_position_amount == 0.0:
                        detail_entry = _with_meta(f"[{sym}] - {manage_label} with no open position ({orders_desc})")
                    else:
                        parts: list[str] = [f"{manage_label}: {pos_label_before} -> {pos_label_after}", orders_desc]
                        if size_change_label:
                            parts.append(size_change_label)
                        detail_entry = _with_meta(f"[{sym}] - {', '.join(parts)}")
                elif action in ("hold", "none"):
                    change_parts: list[str] = []
                    if size_change_label:
                        change_parts.append(size_change_label)
                    protection_changes = _describe_protection_changes(
                        initial_protection_orders,
                        final_protection_orders,
                    )
                    change_parts.extend(protection_changes)
                    if not change_parts:
                        change_parts.append("no changes")
                    detail_entry = _with_meta(f"[{sym}] - holding position ({', '.join(change_parts)})")
                elif action == "skip":
                    detail_entry = _with_meta(f"[{sym}] - skip" + (f" - {reason}" if reason else ""))
                else:
                    detail_entry = _with_meta(f"[{sym}] - skip" + (f" - {reason}" if reason else ""))
            if detail_entry and sym_confidence_text:
                tag_suffix = f" {sym_confidence_tag}" if sym_confidence_tag else ""
                detail_entry = f"{detail_entry} [conf {sym_confidence_text}{tag_suffix}]"
            if detail_entry:
                decisions_details.append(detail_entry)
                try:
                    _append_user_bybit_log(USER_ID, detail_entry)
                except Exception:
                    pass

            summary_action = action
            summary_outcome = "skip"
            if summary_action == "open":
                summary_outcome = "open" if open_success else "skip"
            elif summary_action == "close":
                summary_outcome = "close"
            elif summary_action == "skip":
                summary_outcome = "skip"
            summary_keys = {"open", "close", "skip"}
            if counted_action in summary_keys and summary_outcome in summary_keys:
                if summary_outcome != counted_action:
                    counts[counted_action] = max(0, counts.get(counted_action, 0) - 1)
                    counts[summary_outcome] = counts.get(summary_outcome, 0) + 1
            elif counted_action in summary_keys and summary_outcome not in summary_keys:
                counts[counted_action] = max(0, counts.get(counted_action, 0) - 1)
            elif summary_outcome in summary_keys and counted_action not in summary_keys:
                counts[summary_outcome] = counts.get(summary_outcome, 0) + 1

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
            if sym_cleanup in symbols_with_position_seen:
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
            send_tg(f"📈 {sym_cleanup}: убраны reduce-only стопы (без позиции): {summary}")
    if cleanup_failures:
        details = "; ".join(f"{sym}:{oid} -> {err}" for sym, oid, err in cleanup_failures)
        log(f"⚠️ Не удалось отменить reduce-only стоп-ордера: {details}", Fore.YELLOW)
        send_tg(f"ℹ️ Ошибка отмены reduce-only стоп-ордеров: {details}")

    if decisions_total>0:
        pct={k:(v/decisions_total)*100 for k,v in counts.items()}
        summary=f"📈 Итоги: открыто {counts['open']} ({pct['open']:.1f}%), " \
                f"закрыто {counts['close']} ({pct['close']:.1f}%), " \
                f"пропуск {counts['skip']} ({pct['skip']:.1f}%) — всего {decisions_total}"
        log(summary, Fore.CYAN)
        send_tg(summary)
        try:
            _append_user_bybit_log(USER_ID, summary)
        except Exception:
            pass
        if decisions_details:
            detail_msg = "\n".join(decisions_details)
            log(detail_msg, Fore.LIGHTBLACK_EX)
            send_tg(detail_msg)
            try:
                _append_user_bybit_log(USER_ID, detail_msg)
            except Exception:
                pass
    if MASTER_DECISIONS_SHARE and is_master_user:
        _persist_master_decisions({"user_id": active_user_id})

    final_positions_map, final_positions_count = fetch_positions_snapshot(ex)
    final_positions_available = final_positions_count is not None
    if not final_positions_available:
        final_positions_map = dict(positions_map)

    log(f"[DEBUG] Building positions_summary from final_positions_map: {list((final_positions_map or {}).keys())}", Fore.LIGHTBLACK_EX)
    positions_summary: list[dict[str, Any]] = []
    for sym_active, payload in (final_positions_map or {}).items():
        if not isinstance(payload, dict):
            continue
        amount_val = safe_float(payload.get("amount") or payload.get("contracts"))
        if amount_val is None or not math.isfinite(amount_val) or abs(amount_val) <= 0:
            continue
        entry_val = safe_float(payload.get("entryPrice") or payload.get("average") or payload.get("avgEntryPrice"))
        unreal_val = safe_float(payload.get("unrealizedPnl") or (payload.get("raw") or {}).get("unrealisedPnl"))
        side_raw = str(payload.get("side") or (payload.get("info") or {}).get("side") or "").lower()
        if side_raw in {"sell", "short"}:
            side_label = "SHORT"
        elif side_raw in {"buy", "long"}:
            side_label = "LONG"
        else:
            side_label = "LONG" if amount_val > 0 else "SHORT"
        positions_summary.append(
            {
                "symbol": sym_active,
                "side": side_label,
                "amount": float(amount_val),
                "entry": float(entry_val) if entry_val is not None else None,
                "unrealized": float(unreal_val) if unreal_val is not None else 0.0,
            }
        )
    LATEST_STATUS["positions"] = positions_summary
    try:
        if positions_summary:
            log(f"[DEBUG] positions_summary: {positions_summary}", Fore.LIGHTBLACK_EX)
            status_lines = ["📈 Открытые позиции:"]
            total_unrealized = 0.0
            for pos in positions_summary[:20]:
                entry_val = pos.get("entry")
                entry_txt = ""
                if entry_val is not None and math.isfinite(entry_val):
                    entry_txt = f" @ {entry_val:.4f}"
                amount_val = pos.get("amount")
                amount_txt = f"{amount_val:.4f}" if amount_val is not None and math.isfinite(amount_val) else "?"
                unreal_val = safe_float(pos.get("unrealized", 0.0)) or 0.0
                total_unrealized += unreal_val
                status_lines.append(
                    f"- {pos.get('symbol')} {pos.get('side')} {amount_txt}{entry_txt} (PnL {unreal_val:+.2f} USDT)"
                )
            if len(positions_summary) > 20:
                status_lines.append(f"… ещё {len(positions_summary) - 20} позиций")
            status_lines.append(f"? PnL: {total_unrealized:+.2f} USDT")
        else:
            status_lines = ["📈 Открытых позиций нет."]
        cash_flows = fetch_unified_cash_flows(ex, limit=20)
        for warn in cash_flows.get("warnings", []):
            log(warn, Fore.YELLOW)
        records = cash_flows.get("records") or []
        totals = cash_flows.get("totals") or {}
        deposits_total = totals.get("deposit") or {}
        withdrawals_total = totals.get("withdraw") or {}
        if deposits_total or withdrawals_total:
            status_lines.append("")
            status_lines.append("💵 Unified Trading (последние операции):")
            if deposits_total:
                deposit_summary = ", ".join(
                    f"{coin}:{amount:.4f}" for coin, amount in sorted(deposits_total.items())
                )
                status_lines.append(f"- Ввод: {deposit_summary}")
            else:
                status_lines.append("- Ввод: нет")
        _send_status_notification("\n".join(status_lines))
    except Exception as exc_status:
        log(f"[STATUS] Не удалось отправить список позиций: {exc_status}", Fore.YELLOW)

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
        mark_price = safe_float(payload.get("markPrice") or (payload.get("raw") or {}).get("markPrice"))
        last_price = safe_float(payload.get("lastPrice") or (payload.get("raw") or {}).get("lastPrice"))
        px_val = mark_price if mark_price is not None and math.isfinite(mark_price) else last_price
        protective_orders = _extract_protection_orders(orders_snapshot)
        has_stop, has_take, categorized = _evaluate_position_protection(
            payload,
            protective_orders,
            price_hint=px_val,
        )
        # Always log what protection levels we currently see for each open position.
        px_text = f"{px_val:.4f}" if px_val is not None and math.isfinite(px_val) else "n/a"
        stop_vals = [p for p, _amt in (categorized.get("stop") or []) if p is not None]
        take_vals = [p for p, _amt in (categorized.get("take_profit") or []) if p is not None]
        stop_text = ",".join(f"{float(p):.2f}" for p in stop_vals[:5]) if stop_vals else "n/a"
        take_text = ",".join(f"{float(p):.2f}" for p in take_vals[:5]) if take_vals else "n/a"
        log(
            f"[PROTECT] {sym_active}: px={px_text} stop={stop_text} take={take_text} has_stop={has_stop} has_take={has_take}",
            Fore.LIGHTBLACK_EX,
        )
        needs_protection = False
        if not has_stop:
            needs_protection = True
        elif REQUIRE_TAKE_PROFIT and not has_take:
            needs_protection = True
        if needs_protection:
            missing_parts: list[str] = []
            if not has_stop:
                missing_parts.append("stop")
            if REQUIRE_TAKE_PROFIT and not has_take:
                missing_parts.append("take")
            reason = ", ".join(missing_parts) if missing_parts else "unknown"
            live_price = None
            try:
                ticker = ex.fetch_ticker(sym_active)
                if isinstance(ticker, dict):
                    live_price = safe_float(
                        ticker.get("last")
                        or ticker.get("close")
                        or (ticker.get("info") or {}).get("lastPrice")
                        or (ticker.get("info") or {}).get("price")
                    )
            except Exception:
                live_price = None
            mark_price = safe_float(
                (payload.get("markPrice") if isinstance(payload, dict) else None)
                or (payload.get("raw") or {}).get("markPrice") if isinstance(payload, dict) else None
            )
            price_note = ""
            if live_price is not None and math.isfinite(live_price):
                price_note = f", px={live_price:.4f}"
            elif mark_price is not None and math.isfinite(mark_price):
                price_note = f", mark={mark_price:.4f}"
            level_descript: list[str] = []
            stop_vals = [p for p, _amt in categorized.get("stop", []) if p is not None]
            take_vals = [p for p, _amt in categorized.get("take_profit", []) if p is not None]
            if stop_vals:
                level_descript.append("stop=" + ",".join(f"{p:.6f}" for p in stop_vals))
            if take_vals:
                level_descript.append("take=" + ",".join(f"{p:.6f}" for p in take_vals))
            level_suffix = f" ({'; '.join(level_descript)})" if level_descript else ""
            warn_msg = f"[WARN] {sym_active}: protection missing ({reason}){price_note}{level_suffix}, attempting restore"
            log(warn_msg, Fore.YELLOW)
            try:
                send_tg(warn_msg)
            except Exception:
                pass
            try:
                db_logger.log_protection_check(sym_active, reason, has_stop, has_take, needs_protection)
            except Exception:
                pass
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
        has_stop_after, has_take_after, categorized_after = _evaluate_position_protection(
            position_payload,
            protective_orders_after,
            price_hint=px_val,
        )
        stop_vals_dbg = [p for p, _amt in (categorized_after.get("stop") or []) if p is not None]
        take_vals_dbg = [p for p, _amt in (categorized_after.get("take_profit") or []) if p is not None]
        has_any_level = bool(stop_vals_dbg or take_vals_dbg)

        if has_stop_after and (not REQUIRE_TAKE_PROFIT or has_take_after):
            # Полная защита восстановлена.
            stop_hint = f"{stop_vals_dbg[-1]:.2f}" if stop_vals_dbg else "n/a"
            take_hint = f"{take_vals_dbg[0]:.2f}" if take_vals_dbg else "n/a"
            parts_text = f"stop={stop_hint},take={take_hint}"
            log(
                f"[INFO] {sym_unprotected}: protection restored ({parts_text}; {len(protective_orders_after)} orders)",
                Fore.CYAN,
            )
            restored = True
            continue

        if has_stop_after and REQUIRE_TAKE_PROFIT and not has_take_after:
            # Есть стоп, но нет тейка – считаем позицию защищённой стопом и не закрываем её.
            stop_hint = f"{stop_vals_dbg[-1]:.2f}" if stop_vals_dbg else "n/a"
            log(
                f"[WARN] {sym_unprotected}: take-profit still missing after restore "
                f"(stop={stop_hint}, takes={','.join(f'{p:.2f}' for p in take_vals_dbg) or 'n/a'}) – keeping position with stop-only",
                Fore.YELLOW,
            )
            restored = True
            continue

        if has_any_level:
            # Есть какие‑то защитные уровни, но классификация считает их невалидными — не закрываем автоматически.
            log(
                f"[WARN] {sym_unprotected}: ambiguous protection after restore "
                f"(has_stop={has_stop_after}, has_take={has_take_after}, "
                f"stops={','.join(f'{p:.2f}' for p in stop_vals_dbg) or 'n/a'}, "
                f"takes={','.join(f'{p:.2f}' for p in take_vals_dbg) or 'n/a'}) – skipping auto-close",
                Fore.YELLOW,
            )
            unresolved_unprotected.append(sym_unprotected)
            continue

        # Действительно нет ни стопа, ни тейка – закрываем позицию как раньше.
        position_side_field = str(position_payload.get("side") or "").lower()
        if position_side_field in {"long", "buy"}:
            close_side = "sell"
        elif position_side_field in {"short", "sell"}:
            close_side = "buy"
        else:
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
    schedule_now_utc = datetime.datetime.now(datetime.timezone.utc)
    prev_volatility_ratio = safe_float((cycle_state or {}).get("last_volatility_ratio"))
    prev_interval_from_start = safe_float((cycle_state or {}).get("last_interval_from_start_minutes"))
    current_cycle_no = safe_int(_CURRENT_CYCLE_NUMBER)
    ai_offline_active = (
        current_cycle_no is not None
        and _AI_OFFLINE_ACTIVE_CYCLE is not None
        and safe_int(_AI_OFFLINE_ACTIVE_CYCLE) == current_cycle_no
    )
    atr_ratio_median: float | None = None
    vol_source = "hints"
    source_tags: list[str] = []
    bar_deltas: list[float] = []
    volatility_sample_note = ""
    try:
        atr_samples: list[float] = []
        hint_samples: list[float] = []
        for sym_hint in (selected_symbols or []):
            hint = SYMBOL_MARKET_MODE_HINTS.get(sym_hint) or {}
            analysis_balance = hint.get("analysis_balance") if isinstance(hint, dict) else None
            ratio = safe_float((analysis_balance or {}).get("atr_ratio")) if isinstance(analysis_balance, dict) else None
            if ratio is not None and math.isfinite(ratio) and ratio > 0:
                hint_samples.append(float(ratio))
        if hint_samples:
            atr_samples.extend(hint_samples)
            source_tags.append("hints")
        def _timeframe_minutes(tf_value: str | None) -> float | None:
            if not tf_value:
                return None
            tf_clean = str(tf_value).strip().lower()
            if not tf_clean:
                return None
            unit = tf_clean[-1]
            try:
                value = float(tf_clean[:-1])
            except Exception:
                return None
            if unit == "m":
                return value
            if unit == "h":
                return value * 60.0
            if unit == "d":
                return value * 1440.0
            return None

        def _tf_candidates_from_interval(interval_minutes: float | None) -> list[str]:
            standards: list[tuple[float, str]] = [
                (5.0, "5m"),
                (15.0, "15m"),
                (30.0, "30m"),
                (60.0, "1h"),
                (120.0, "2h"),
                (240.0, "4h"),
            ]
            if interval_minutes is None or not math.isfinite(interval_minutes) or interval_minutes <= 0:
                interval_minutes = float(DEFAULT_NEXT_RUN_MINUTES)
            interval_minutes = max(5.0, min(interval_minutes, 240.0))
            primary: str | None = None
            secondary: str | None = None
            for minutes, label in standards:
                if minutes <= interval_minutes + 1e-9:
                    if primary is None or minutes >= (_timeframe_minutes(primary) or 0):
                        secondary = primary if primary != label else secondary
                        primary = label
            if primary is None:
                primary = "5m"
            tf_list = [primary]
            if secondary and secondary not in tf_list:
                tf_list.append(secondary)
            if "5m" not in tf_list:
                tf_list.append("5m")
            return tf_list

        symbols_for_vol = set(selected_symbols or [])
        if isinstance(final_positions_map, dict):
            symbols_for_vol.update(final_positions_map.keys())
        if not symbols_for_vol:
            fallback_symbols: list[str] = []
            for raw_pair in PAIR_LIST:
                normalized = normalize_symbol(raw_pair, record_missing=False) or raw_pair
                if normalized and normalized not in fallback_symbols:
                    fallback_symbols.append(normalized)
                if len(fallback_symbols) >= 6:
                    break
            symbols_for_vol.update(fallback_symbols)
        timeframes_for_vol: list[str] = []
        tf_candidates = set(_tf_candidates_from_interval(prev_interval_from_start))
        tf_env = str(TIMEFRAME or "").strip()
        if tf_env:
            tf_env_norm = tf_env.lower()
            minutes_env = _timeframe_minutes(tf_env_norm)
            if minutes_env is None or minutes_env <= (prev_interval_from_start or minutes_env or 0):
                tf_candidates.add(tf_env_norm)
        for tf_candidate in tf_candidates:
            tf_value = (tf_candidate or "").strip()
            if not tf_value:
                continue
            if tf_value not in timeframes_for_vol:
                timeframes_for_vol.append(tf_value)
        bar_samples: list[float] = []
        bar_delta_sources: list[str] = []
        for sym_vol in symbols_for_vol:
            for tf_vol in timeframes_for_vol:
                try:
                    df_vol = fetch_df(ex, sym_vol, tf_vol)
                except Exception:
                    df_vol = None
                if df_vol is None or df_vol.empty:
                    continue
                df_vol_local = df_vol.copy()
                if "atr" not in df_vol_local.columns:
                    try:
                        df_vol_local["atr"] = atr(df_vol_local, 14)
                    except Exception:
                        continue
                last_row = df_vol_local.iloc[-1]
                atr_val = safe_float(last_row.get("atr") or last_row.get("atr14"))
                close_val = safe_float(last_row.get("close") or last_row.get("c"))
                if (
                    atr_val is None
                    or close_val is None
                    or not math.isfinite(atr_val)
                    or not math.isfinite(close_val)
                    or close_val <= 0
                ):
                    continue
                ratio = float(atr_val) / float(close_val)
                if ratio <= 0:
                    continue
                bar_samples.append(ratio)
                if len(df_vol_local) >= 2:
                    prev_row = df_vol_local.iloc[-2]
                    atr_prev = safe_float(prev_row.get("atr") or prev_row.get("atr14"))
                    close_prev = safe_float(prev_row.get("close") or prev_row.get("c"))
                    if (
                        atr_prev is not None
                        and close_prev is not None
                        and math.isfinite(atr_prev)
                        and math.isfinite(close_prev)
                        and close_prev > 0
                    ):
                        prev_ratio = float(atr_prev) / float(close_prev)
                        if math.isfinite(prev_ratio):
                            delta_val = ratio - prev_ratio
                            bar_deltas.append(delta_val)
                            bar_delta_sources.append(f"{sym_vol}@{tf_vol}:{delta_val:+.4f}")
        if bar_samples:
            atr_samples.extend(bar_samples)
            source_tags.append("bars")
        if bar_delta_sources:
            preview = ", ".join(bar_delta_sources[:3])
            if len(bar_delta_sources) > 3:
                preview = f"{preview}, +{len(bar_delta_sources) - 3} more"
            volatility_sample_note = f"deltas[{len(bar_delta_sources)}]: {preview}"
        elif bar_samples:
            volatility_sample_note = f"bars[{len(bar_samples)}]"
        if atr_samples:
            atr_samples.sort()
            atr_ratio_median = atr_samples[len(atr_samples) // 2]
            if source_tags:
                vol_source = "+".join(source_tags)
    except Exception:
        atr_ratio_median = None
    if (atr_ratio_median is None or not math.isfinite(atr_ratio_median)) and prev_volatility_ratio is not None and math.isfinite(prev_volatility_ratio):
        # If we couldn't compute a fresh volatility ratio this cycle, carry forward the last known value
        # so scheduling remains stable and deltas can still be applied once hints resume.
        atr_ratio_median = float(prev_volatility_ratio)
        vol_source = "carry"

    # Prefer explicit next_run_time (absolute timestamp); if missing, use next_run_minutes as an interval
    # anchored to the *start* of this cycle (cycle_start_utc) rather than the end.
    interval_floor = float(MIN_NEXT_RUN_MINUTES)
    interval_cap = float(MAX_NEXT_RUN_FROM_START_MINUTES)
    if ai_offline_active:
        # Offline mode: use OFFLINE_* bounds from .env (defaults 5-35m)
        min_delay_override = float(OFFLINE_MIN_NEXT_RUN_MINUTES or 5.0)
        max_delay_override = float(OFFLINE_MAX_NEXT_RUN_MINUTES or 35.0)
        log(
            f"[SCHED] AI offline bounds applied: {min_delay_override:.1f}-{max_delay_override:.1f}m window while offline mode active",
            Fore.LIGHTBLACK_EX,
        )
        interval_floor = max(interval_floor, min_delay_override)
        interval_cap = min(interval_cap, max_delay_override)
    elif rate_limit_backoff:
        # Rate-limit backoff (provider still available): use BACKOFF_* bounds (defaults 25-55m)
        backoff_min = float(os.getenv("BACKOFF_MIN_NEXT_RUN_MINUTES", "25") if BACKOFF_MIN_NEXT_RUN_MINUTES is None else BACKOFF_MIN_NEXT_RUN_MINUTES)
        backoff_max = float(os.getenv("BACKOFF_MAX_NEXT_RUN_MINUTES", "55") if BACKOFF_MAX_NEXT_RUN_MINUTES is None else BACKOFF_MAX_NEXT_RUN_MINUTES)
        min_delay_override = backoff_min
        max_delay_override = backoff_max
        log(f"[SCHED] rate-limit backoff active: bounds {min_delay_override:.1f}-{max_delay_override:.1f}m", Fore.LIGHTBLACK_EX)
        interval_floor = max(interval_floor, min_delay_override)
        interval_cap = min(interval_cap, max_delay_override)
    else:
        # Online mode: use ONLINE_* bounds from .env (defaults 10-45m)
        online_min = float(ONLINE_MIN_NEXT_RUN_MINUTES or MIN_NEXT_RUN_MINUTES)
        online_max = float(ONLINE_MAX_NEXT_RUN_MINUTES or MAX_NEXT_RUN_FROM_START_MINUTES)
        min_delay_override = online_min
        max_delay_override = online_max
    if interval_cap < interval_floor:
        interval_cap = interval_floor
        # Offline mode: use OFFLINE_* bounds from .env (defaults 5-35m)
        min_delay_override = float(OFFLINE_MIN_NEXT_RUN_MINUTES or 5.0)
        max_delay_override = float(OFFLINE_MAX_NEXT_RUN_MINUTES or 35.0)
        log(
            f"[SCHED] AI offline bounds applied: {min_delay_override:.1f}-{max_delay_override:.1f}m window while offline mode active",
            Fore.LIGHTBLACK_EX,
        )
        interval_floor = max(interval_floor, min_delay_override)
        interval_cap = min(interval_cap, max_delay_override)
    else:
        # Online mode: use ONLINE_* bounds from .env (defaults 10-45m)
        online_min = float(ONLINE_MIN_NEXT_RUN_MINUTES or MIN_NEXT_RUN_MINUTES)
        online_max = float(ONLINE_MAX_NEXT_RUN_MINUTES or MAX_NEXT_RUN_FROM_START_MINUTES)
        min_delay_override = online_min
        max_delay_override = online_max
    if interval_cap < interval_floor:
        interval_cap = interval_floor
    min_delay = max(0.0, min_delay_override)
    max_delay = max(
        0.0,
        (cycle_start_utc + datetime.timedelta(minutes=float(max_delay_override)) - schedule_now_utc).total_seconds() / 60.0,
    )
    timing_debug_parts: list[str] = []
    if next_delay_minutes is None:
        # Fallback: start from previous interval and nudge ±5/±10 minutes based on volatility change.
        prev_interval = prev_interval_from_start
        if prev_interval is None or not math.isfinite(prev_interval) or prev_interval <= 0:
            prev_interval = float(DEFAULT_NEXT_RUN_MINUTES)
        prev_interval = min(max(float(prev_interval), interval_floor), interval_cap)

        delta = 0.0
        volatility_note = ""
        thresholds_note = ""
        diff_value: float | None = None
        diff_cycle: float | None = None
        diff_intrabar: float | None = None
        if atr_ratio_median is not None and math.isfinite(atr_ratio_median):
            base_ref = prev_volatility_ratio if prev_volatility_ratio and math.isfinite(prev_volatility_ratio) else atr_ratio_median
            base_tol = max(0.0001, base_ref * 0.03 if base_ref and math.isfinite(base_ref) else 0.0001)
            strong_threshold = max(base_tol * 2.0, 0.001)
            thresholds_note = f"thr5={base_tol:.4f}, thr10={strong_threshold:.4f}, src={vol_source}"
            diff_candidate = None
            if prev_volatility_ratio is not None and math.isfinite(prev_volatility_ratio):
                diff_cycle = atr_ratio_median - prev_volatility_ratio
            if bar_deltas:
                bar_deltas.sort()
                diff_intrabar = bar_deltas[len(bar_deltas) // 2]
                thresholds_note += "+intrabar"
            if diff_cycle is not None and math.isfinite(diff_cycle) and diff_intrabar is not None and math.isfinite(diff_intrabar):
                diff_candidate = diff_cycle if abs(diff_cycle) >= abs(diff_intrabar) else diff_intrabar
            elif diff_cycle is not None and math.isfinite(diff_cycle):
                diff_candidate = diff_cycle
            elif diff_intrabar is not None and math.isfinite(diff_intrabar):
                diff_candidate = diff_intrabar
            if diff_candidate is not None and math.isfinite(diff_candidate):
                diff = diff_candidate
                diff_value = diff
                if diff > base_tol:
                    strong = diff >= strong_threshold
                    delta = -10.0 if strong else -5.0
                    volatility_note = "vol↑↑" if strong else "vol↑"
                elif diff < -base_tol:
                    strong = diff <= -strong_threshold
                    delta = 10.0 if strong else 5.0
                    volatility_note = "vol↓↓" if strong else "vol↓"
            else:
                volatility_note = "vol=init"

        target_interval = prev_interval + delta
        target_interval = round(target_interval / 5.0) * 5.0
        target_interval = min(max(target_interval, interval_floor), interval_cap)
        fallback_interval_from_start = target_interval
        vol_text = f"{atr_ratio_median:.4f}" if atr_ratio_median is not None and math.isfinite(atr_ratio_median) else "n/a"
        diff_text = f"{diff_value:.4f}" if diff_value is not None and math.isfinite(diff_value) else "n/a"
        cycle_diff_text = f"{diff_cycle:.4f}" if diff_cycle is not None and math.isfinite(diff_cycle) else "n/a"
        intrabar_diff_text = f"{diff_intrabar:.4f}" if diff_intrabar is not None and math.isfinite(diff_intrabar) else "n/a"
        timing_debug_parts.append(
            f"fallback=adaptive prev={prev_interval:.2f}m delta={delta:+.1f}m -> {fallback_interval_from_start:.2f}m {volatility_note or ''} "
            f"(prev_vol={prev_volatility_ratio if prev_volatility_ratio is not None else 'n/a'}, vol={vol_text}, diff={diff_text}; "
            f"diff_cycle={cycle_diff_text}, diff_intrabar={intrabar_diff_text}; {thresholds_note})"
        )
        if volatility_sample_note:
            timing_debug_parts.append(f"[samples] {volatility_sample_note}")

        target_dt = cycle_start_utc + datetime.timedelta(minutes=float(fallback_interval_from_start))
        remaining = (target_dt - schedule_now_utc).total_seconds() / 60.0
        clamped = min(max(remaining, min_delay), max_delay if max_delay > 0 else remaining)
        next_delay_minutes = max(0.0, clamped)
        next_run_dt = schedule_now_utc + datetime.timedelta(minutes=next_delay_minutes)
        fallback_msg = (
            f"Next cycle fallback (adaptive): interval {fallback_interval_from_start:.1f}m from cycle start "
            f"(sleep {next_delay_minutes:.1f} min, bounds {min_delay:.1f}-{max_delay:.1f})."
        )
        log(fallback_msg, Fore.LIGHTBLACK_EX)
        send_tg(fallback_msg)
    elif next_run_dt is None:
        next_run_dt = schedule_now_utc + datetime.timedelta(minutes=next_delay_minutes)

    if next_run_dt is not None:
        aligned_dt, aligned_delay, aligned_method = _align_next_run_to_step(
            target_dt=next_run_dt,
            now_utc=schedule_now_utc,
            min_delay_minutes=min_delay,
            max_delay_minutes=max_delay,
            step_minutes=5,
        )
        if aligned_method and aligned_dt != next_run_dt:
            before_local = next_run_dt.astimezone(_current_local_tz() or datetime.datetime.now().astimezone().tzinfo)
            after_local = aligned_dt.astimezone(_current_local_tz() or datetime.datetime.now().astimezone().tzinfo)
            log(
                "Next run aligned to 5m grid: "
                f"{before_local.strftime('%Y-%m-%d %H:%M:%S %Z')} -> {after_local.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                f"(delay {next_delay_minutes:.2f} -> {aligned_delay:.2f} min)",
                Fore.LIGHTBLACK_EX,
            )
            next_run_dt = aligned_dt
            next_delay_minutes = aligned_delay
        final_local = next_run_dt.astimezone(_current_local_tz() or datetime.datetime.now().astimezone().tzinfo)
        final_delay_minutes = max(0.0, (next_run_dt - schedule_now_utc).total_seconds() / 60.0)
        final_msg = (
            f"ℹ️ Следующая сессия запланирована на {final_local.strftime('%Y-%m-%d %H:%M:%S %Z')} "
            f"(~{final_delay_minutes:.1f} мин)"
        )
        log(final_msg, Fore.CYAN)
        send_tg(final_msg)
        if timing_debug_parts:
            log("[SCHED] timing debug: " + "; ".join(timing_debug_parts), Fore.LIGHTBLACK_EX)
        try:
            if isinstance(cycle_state, dict):
                cycle_state["last_next_delay_minutes"] = round(float(final_delay_minutes), 2)
                cycle_state["last_interval_from_start_minutes"] = round(
                    float((next_run_dt - cycle_start_utc).total_seconds() / 60.0),
                    2,
                )
                if atr_ratio_median is not None and math.isfinite(atr_ratio_median):
                    cycle_state["last_volatility_ratio"] = float(atr_ratio_median)
                cycle_state["rate_limit_backoff"] = bool(rate_limit_backoff)
        except Exception:
            pass
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
    release_thread = TELEGRAM_RELEASE_THREAD_ID if TELEGRAM_RELEASE_THREAD_ID is not None else TG_TOPIC_ID
    send_kwargs.setdefault("thread_id", release_thread)
    if changelog_message_id or link:
        send_tg(f"ℹ️ Версия {version_display}", **send_kwargs)
    else:
        send_tg(f"ℹ️ Версия {BOT_VERSION}. {BOT_CHANGELOG}", thread_id=release_thread)
    balance_snapshot_end: dict[str, Any] | None = None
    closed_order_details: list[dict[str, Any]] = []
    try:
        equity_end, available_end, balance_snapshot_end = fetch_usdt_equity(ex)
    except Exception as exc_equity:
        end_balance_text = f"ℹ️ Не удалось обновить баланс: {exc_equity}"
        log(end_balance_text, Fore.YELLOW)
        send_tg(end_balance_text)
    else:
        end_stable_label = _stable_currency_label(balance_snapshot_end)
        end_balance_text = f"Баланс: {equity_end:.2f} {end_stable_label}, доступно {available_end:.2f} {end_stable_label}"
        log(f"✅ Завершение сессии. {end_balance_text}", Fore.GREEN)
        send_tg(f"ℹ️ Завершение сессии. {end_balance_text}")
        realized_end = None
        if isinstance(balance_snapshot_end, dict):
            realized_end = balance_snapshot_end.get("_realizedPnl")
        try:
            history_entries = _load_equity_history()
            results_state = _load_results_state()
            results_state_dirty = False
            raw_recent_symbols = results_state.get("recent_symbols")
            recent_symbols_list: list[str] = []
            recent_symbols_seen: set[str] = set()
            if isinstance(raw_recent_symbols, (list, tuple)):
                for entry in raw_recent_symbols:
                    if not isinstance(entry, str):
                        continue
                    sym_candidate = entry.strip()
                    if not sym_candidate:
                        continue
                    normalized_sym = normalize_symbol(sym_candidate, record_missing=False) or sym_candidate
                    if normalized_sym in recent_symbols_seen:
                        continue
                    recent_symbols_seen.add(normalized_sym)
                    recent_symbols_list.append(normalized_sym)

            now_utc = datetime.datetime.now(datetime.timezone.utc)
            closed_symbols_set: set[str] = set(available_pairs)
            closed_symbols_set.update(order_symbols)
            closed_symbols_set.update(order_symbols_non_reduce)
            closed_symbols_set.update(position_symbols)
            if recent_symbols_list:
                closed_symbols_set.update(recent_symbols_list)
            if isinstance(global_open_orders, dict):
                closed_symbols_set.update(global_open_orders.keys())
            for raw_pair in PAIR_LIST:
                normalized_pair = normalize_symbol(raw_pair, record_missing=False)
                if normalized_pair:
                    closed_symbols_set.add(normalized_pair)
            window_start = now_utc - datetime.timedelta(hours=PNL_LOOKBACK_HOURS)
            closed_pnl_value, closed_pnl_count, closed_warnings, closed_order_details = _collect_recent_closed_pnl(
                ex,
                sorted(closed_symbols_set),
                window_start,
                now_utc,
            )
            for warning_msg in closed_warnings[:3]:
                log(warning_msg, Fore.LIGHTBLACK_EX)
            if len(closed_warnings) > 3:
                log(f"[PnL] Suppressed {len(closed_warnings) - 3} additional warnings.", Fore.LIGHTBLACK_EX)
            if closed_order_details:
                max_recent_symbols = 24
                for detail in closed_order_details:
                    sym_detail = detail.get("symbol")
                    sym_text = str(sym_detail or "").strip()
                    if not sym_text:
                        continue
                    normalized_sym = normalize_symbol(sym_text, record_missing=False) or sym_text
                    try:
                        recent_symbols_list.remove(normalized_sym)
                    except ValueError:
                        if normalized_sym not in recent_symbols_seen:
                            recent_symbols_seen.add(normalized_sym)
                    recent_symbols_list.append(normalized_sym)
                    recent_symbols_seen.add(normalized_sym)
                if len(recent_symbols_list) > max_recent_symbols:
                    overflow = len(recent_symbols_list) - max_recent_symbols
                    if overflow > 0:
                        dropped = recent_symbols_list[:overflow]
                        recent_symbols_list = recent_symbols_list[overflow:]
                        for dropped_sym in dropped:
                            if dropped_sym not in recent_symbols_list:
                                recent_symbols_seen.discard(dropped_sym)
                results_state["recent_symbols"] = recent_symbols_list
                results_state_dirty = True
            unreal_total, unreal_count = _sum_unrealized_pnl(final_positions_map)

            unreal_reported = False
            pnl_value: float | None = None
            reference_value: float | None = None
            pnl_basis: str | None = None
            if closed_pnl_value is not None:
                risk_msg = _adjust_dynamic_risk_from_pnl(closed_pnl_value, "closed", closed_pnl_count)
                if risk_msg:
                    log(risk_msg, Fore.CYAN if CURRENT_RISK_PCT >= initial_cycle_risk_pct else Fore.YELLOW)
                    try:
                        send_tg(risk_msg)
                    except Exception:
                        pass
                _emit_realized_pnl_message("", closed_pnl_value, closed_pnl_count)
                _emit_unrealized_pnl_message("end", unreal_total, unreal_count)
                unreal_reported = True
            else:
                pnl_value, reference_value, pnl_basis = _compute_recent_pnl(
                    history_entries,
                    now_utc,
                    equity_end,
                    realized_end,
                )
                if pnl_value is not None and reference_value is not None:
                    risk_msg_equity = _adjust_dynamic_risk_from_pnl(pnl_value, pnl_basis or "equity", 0)
                    if risk_msg_equity:
                        log(risk_msg_equity, Fore.CYAN if CURRENT_RISK_PCT >= initial_cycle_risk_pct else Fore.YELLOW)
                        try:
                            send_tg(risk_msg_equity)
                        except Exception:
                            pass
                    basis_label = "realized" if pnl_basis == "realized" else "equity"
                    pnl_message = f"PnL ({_pnl_window_label()} {basis_label}): {pnl_value:+.2f} USDT (ref {reference_value:.2f})"
                    log(pnl_message, Fore.CYAN if pnl_value >= 0 else Fore.YELLOW)
                    send_tg(pnl_message)
                    _emit_unrealized_pnl_message("end", unreal_total, unreal_count)
                    unreal_reported = True
                else:
                    if DYNAMIC_RISK_ENABLED:
                        baseline_msg = _adjust_dynamic_risk_from_pnl(0.0, "baseline", 0)
                        if baseline_msg:
                            log(baseline_msg, Fore.LIGHTBLACK_EX)
                            try:
                                send_tg(baseline_msg)
                            except Exception:
                                pass
            results_lines: list[str] = ["📊 Итоги последних 6 часов"]
            if closed_pnl_value is not None:
                results_lines.append(f"PnL (закрытые ордера): {closed_pnl_value:+.2f} USDT ({closed_pnl_count} ордеров)")
            elif pnl_value is not None:
                basis_label = pnl_basis or "equity"
                extra = f", ref {reference_value:.2f}" if reference_value is not None and math.isfinite(reference_value) else ""
                results_lines.append(f"PnL ({basis_label}): {pnl_value:+.2f} USDT{extra}")
            else:
                results_lines.append("PnL: данные недоступны.")

            latest_equity_point: tuple[datetime.datetime, float, float | None] | None = None
            if isinstance(equity_end, (int, float)) and math.isfinite(equity_end):
                latest_equity_point = (
                    now_utc,
                    float(equity_end),
                    float(realized_end) if isinstance(realized_end, (int, float)) and math.isfinite(realized_end) else None,
                )
            daily_closed_map = results_state.get("daily_closed_pnl")
            if not isinstance(daily_closed_map, dict):
                daily_closed_map = {}
            daily_chart = None
            daily_points: list[tuple[datetime.date, float]] = []
            chart_closed, points_closed = _build_daily_closed_pnl_chart(daily_closed_map, history_entries, days=7)
            if chart_closed:
                daily_chart, daily_points = chart_closed, points_closed
            else:
                daily_chart, daily_points = _build_daily_pnl_percent_chart(
                    history_entries,
                    latest_point=latest_equity_point,
                    days=7,
                )
            if daily_chart:
                legend_tail = ", ".join(
                    f"{day.strftime('%m-%d')}: {pct:+.1f}%"
                    for day, pct in daily_points[-3:]
                )
                if legend_tail:
                    results_lines.append(f"📆 Ежесуточный PnL (%): {daily_chart} ({legend_tail})")
                else:
                    results_lines.append(f"📆 Ежесуточный PnL (%): {daily_chart}")
            else:
                results_lines.append("📆 Ежесуточный PnL (%): недостаточно данных.")

            if closed_order_details:
                sample_ids = [str(detail.get("id") or "") for detail in closed_order_details[:5]]
                log(
                    f"[RESULTS] closed_order_details count={len(closed_order_details)} sample_ids={', '.join(sample_ids)}",
                    Fore.LIGHTBLACK_EX,
                )

            reported_ids = set(results_state.get("closed_order_ids") or [])
            new_orders: list[dict[str, Any]] = []
            new_keys: list[str] = []
            for detail in closed_order_details:
                key = _results_order_key(detail)
                if not key or key in reported_ids:
                    continue
                new_orders.append(detail)
                new_keys.append(key)
            if new_orders:
                sample_new_ids = [str(detail.get("id") or "") for detail in new_orders[:5]]
                log(
                    f"[RESULTS] new_closed_orders count={len(new_orders)} sample_ids={', '.join(sample_new_ids)}",
                    Fore.LIGHTBLACK_EX,
                )
                total_new_pnl = sum(
                    detail.get("pnl", 0.0) for detail in new_orders if isinstance(detail.get("pnl"), (int, float))
                )
                results_lines.append(f"? новых закрытий: {total_new_pnl:+.2f} USDT ({len(new_orders)} ордеров)")
                for detail in new_orders[:RESULTS_CLOSED_ORDER_DISPLAY_LIMIT]:
                    results_lines.append(_format_closed_order_line(detail))
                if len(new_orders) > RESULTS_CLOSED_ORDER_DISPLAY_LIMIT:
                    results_lines.append(f"… ещё {len(new_orders) - RESULTS_CLOSED_ORDER_DISPLAY_LIMIT} ордер(ов)")
                if _update_daily_closed_pnl_map(daily_closed_map, new_orders):
                    results_state["daily_closed_pnl"] = daily_closed_map
                    results_state_dirty = True
            else:
                results_lines.append("🧾 Новых закрытых ордеров за 6ч нет.")
            _send_results_notification("\n".join(results_lines))
            if new_keys:
                updated_ids = list(reported_ids) + new_keys
                max_len = 500
                if len(updated_ids) > max_len:
                    updated_ids = updated_ids[-max_len:]
                results_state["closed_order_ids"] = updated_ids
                results_state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                latest_ts = None
                for detail in closed_order_details:
                    ts_iso = detail.get("timestamp")
                    if not ts_iso:
                        continue
                    if latest_ts is None or ts_iso > latest_ts:
                        latest_ts = ts_iso
                if latest_ts:
                    results_state["last_timestamp"] = latest_ts
                results_state_dirty = True
            if results_state_dirty:
                _save_results_state(results_state)
            if not unreal_reported:
                _emit_unrealized_pnl_message("end", unreal_total, unreal_count)
            LATEST_STATUS.update(
                {
                    "equity_end": equity_end,
                    "available_end": available_end,
                    "closed_pnl": closed_pnl_value if closed_pnl_value is not None else pnl_value,
                    "unrealized": unreal_total,
                    "stable_label": end_stable_label,
                }
            )
            _update_equity_history(history_entries, now_utc, equity_end, realized_end)
        except Exception as exc_pnl:
            log(f"[WARN] Failed to update PnL history: {exc_pnl}", Fore.YELLOW)
    decision_log_path = _resolve_log_path(AI_LOG_FILE)
    if decision_log_path:
        summary_output = decision_log_path.with_name("ai_decision_summary.json")
        _generate_ai_decision_summary(decision_log_path, summary_output, window_hours=24.0)
    end_dt = _current_log_time()
    end_stamp = end_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    end_banner = f"{session_separator} END SESSION {end_stamp} {session_separator}"
    log(end_banner, Fore.MAGENTA)
    send_tg(f"{session_separator}\nEND SESSION {end_stamp}\n{session_separator}")
    _flush_tg_log_buffer(force=True)
    # Persist per-symbol unrealized PnL snapshot for next-cycle pseudo-trailing decisions.
    if isinstance(cycle_state, dict):
        try:
            cycle_state["positions_unrealized"] = {
                str(sym): float(val)
                for sym, val in (
                    (
                        symbol_key,
                        safe_float(payload.get("unrealizedPnl") or (payload.get("raw") or {}).get("unrealisedPnl")),
                    )
                    for symbol_key, payload in (final_positions_map or {}).items()
                )
                if val is not None and math.isfinite(val)
            }
        except Exception:
            pass
        try:
            open_trailing_syms = set(cycle_state.get("positions_unrealized", {}).keys())
            cycle_state["positions_trailing"] = {
                str(sym): {
                    "stop": float(vals.get("stop")) if isinstance(vals, dict) and vals.get("stop") is not None and math.isfinite(vals.get("stop")) else None,
                    "take": float(vals.get("take")) if isinstance(vals, dict) and vals.get("take") is not None and math.isfinite(vals.get("take")) else None,
                    "base_stop": float(vals.get("base_stop")) if isinstance(vals, dict) and vals.get("base_stop") is not None and math.isfinite(vals.get("base_stop")) else None,
                    "base_take": float(vals.get("base_take")) if isinstance(vals, dict) and vals.get("base_take") is not None and math.isfinite(vals.get("base_take")) else None,
                    "base_unreal": float(vals.get("base_unreal")) if isinstance(vals, dict) and vals.get("base_unreal") is not None and math.isfinite(vals.get("base_unreal")) else None,
                    "base_price": float(vals.get("base_price")) if isinstance(vals, dict) and vals.get("base_price") is not None and math.isfinite(vals.get("base_price")) else None,
                    "take_shift_total": float(vals.get("take_shift_total")) if isinstance(vals, dict) and vals.get("take_shift_total") is not None and math.isfinite(vals.get("take_shift_total")) else None,
                    "take_extensions": safe_int(vals.get("take_extensions")) if isinstance(vals, dict) else None,
                    "take_last_extended_cycle": safe_int(vals.get("take_last_extended_cycle")) if isinstance(vals, dict) else None,
                    "tightening_count": safe_int(vals.get("tightening_count")) if isinstance(vals, dict) else None,
                    "position_side": (vals.get("position_side") if isinstance(vals, dict) else None),
                    "position_qty": float(vals.get("position_qty")) if isinstance(vals, dict) and vals.get("position_qty") is not None else None,
                    "activated_cycle": safe_int(vals.get("activated_cycle")) if isinstance(vals, dict) else None,
                }
                for sym, vals in (_TRAIL_PROTECTION or {}).items()
                if str(sym) in open_trailing_syms
            }
        except Exception:
            pass
    cycle_commit_hash = source_hash or head_commit_hash or None
    cycle_commit_timestamp = source_timestamp_env or head_commit_timestamp or None
    _record_cycle_completion(
        cycle_state,
        cycle_kind=cycle_kind or "normal",
        cycle_mode=cycle_mode or "last",
        branch_name=branch_name,
        commit_hash=cycle_commit_hash or source_ref or None,
        commit_timestamp=cycle_commit_timestamp,
    )
    return next_delay_minutes

def main():
    ensure_version_backup()
    refresh_settings()
    base_margin_utilization = ORDER_MARGIN_UTILIZATION
    base_auto_margin_ratio = AUTO_MARGIN_SCALE_RATIO
    configure_telegram_bot()
    start_telegram_webhook_server()
    start_telegram_long_polling()
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
        try:
            maybe_send_graphs()
        except Exception as exc:
            log(f"[GRAPH] {exc}", Fore.YELLOW)
        base_delay = delay_minutes if delay_minutes and delay_minutes > 0 else DEFAULT_NEXT_RUN_MINUTES
        while True:
            delay_minutes, target_dt, override_applied = _consume_schedule_override(base_delay)
            if delay_minutes <= 0:
                break
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if target_dt is None:
                try:
                    delay_minutes = float(delay_minutes)
                except (TypeError, ValueError):
                    base_delay = DEFAULT_NEXT_RUN_MINUTES
                    delay_minutes = base_delay
                    continue
                if not math.isfinite(delay_minutes) or delay_minutes <= 0:
                    break
                target_dt = now_utc + datetime.timedelta(minutes=delay_minutes)
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=datetime.timezone.utc)
            remaining_seconds = max(0.0, (target_dt - now_utc).total_seconds())
            delay_minutes = remaining_seconds / 60.0
            if remaining_seconds <= 0:
                break
            local_tz = _current_local_tz() or datetime.datetime.now().astimezone().tzinfo
            next_local = target_dt.astimezone(local_tz)
            eta_msg = (
                f"ℹ️ Следующая сессия запланирована на {next_local.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                f"(~{delay_minutes:.1f} мин)"
            )
            log(eta_msg, Fore.LIGHTBLACK_EX)
            send_tg(eta_msg)
            _write_runtime_status(delay_minutes, target_dt, "sleeping")

            progress_enabled = remaining_seconds >= 180
            progress_interval = (
                min(300.0, max(90.0, remaining_seconds / 4.0)) if progress_enabled else remaining_seconds
            )
            interrupted = False
            try:
                while True:
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    remaining_seconds = (target_dt - now_utc).total_seconds()
                    if remaining_seconds <= 0:
                        break
                    step = min(progress_interval, remaining_seconds)
                    if _SCHEDULE_EVENT.wait(step):
                        _SCHEDULE_EVENT.clear()
                        interrupted = True
                        break
                    if not progress_enabled:
                        continue
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    remaining_seconds = max(0.0, (target_dt - now_utc).total_seconds())
                    if remaining_seconds <= 0:
                        continue
                    minutes_left = remaining_seconds / 60.0
                    eta_local = target_dt.astimezone(local_tz)
                    progress_msg = (
                        f"ℹ️ Осталось ~{minutes_left:.1f} мин до следующей сессии "
                        f"({eta_local.strftime('%H:%M:%S %Z')})"
                    )
                    log(progress_msg, Fore.LIGHTBLACK_EX)
                    send_tg(progress_msg)
            except KeyboardInterrupt:
                log("Interrupted during sleep.", Fore.YELLOW)
                _write_runtime_status(None, None, "stopped")
                break
            if interrupted:
                continue
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



def _handle_tokens_command(args: list[str]) -> str:
    global AI_TOKEN_BUDGET_CYCLE, AI_HARD_STOP_BUDGET, AI_SECONDARY_BUDGET_START
    global AI_TOKEN_USAGE_TOTAL, AI_TOKEN_USAGE_BY_MODEL
    if not args:
        return _format_token_usage_message()
    action = args[0].lower()
    if action in {"status", "show"}:
        return _format_token_usage_message()
    if action in {"reset"}:
        AI_TOKEN_USAGE_TOTAL = 0
        AI_TOKEN_USAGE_BY_MODEL.clear()
        return "🔄 Счётчики токенов сброшены для текущего цикла."
    if action in {"budget", "soft"}:
        value_token = args[1] if len(args) > 1 else None
        parsed = _parse_minutes_argument(value_token or "") if value_token else None
        if parsed is None:
            return "❌ Укажите числовой лимит токенов. Пример: /tokens budget 150000"
        AI_TOKEN_BUDGET_CYCLE = max(1000, int(parsed))
        os.environ["OPENAI_TOKEN_BUDGET_PER_CYCLE"] = str(AI_TOKEN_BUDGET_CYCLE)
        _persist_env_values({"OPENAI_TOKEN_BUDGET_PER_CYCLE": str(AI_TOKEN_BUDGET_CYCLE)})
        return f"✅ Лимит токенов на цикл обновлён: {AI_TOKEN_BUDGET_CYCLE}"
    if action in {"hard", "stop"}:
        value_token = args[1] if len(args) > 1 else None
        if not value_token or value_token.lower() in {"off", "none", "0"}:
            AI_HARD_STOP_BUDGET = 0
            os.environ.pop("OPENAI_HARD_STOP_BUDGET", None)
            _persist_env_values({"OPENAI_HARD_STOP_BUDGET": ""})
            return "✅ Жёсткий стоп отключён."
        parsed = _parse_minutes_argument(value_token)
        if parsed is None:
            return "❌ Укажите числовое значение. Пример: /tokens hard 200000"
        AI_HARD_STOP_BUDGET = max(0, int(parsed))
        os.environ["OPENAI_HARD_STOP_BUDGET"] = str(AI_HARD_STOP_BUDGET)
        _persist_env_values({"OPENAI_HARD_STOP_BUDGET": str(AI_HARD_STOP_BUDGET)})
        return f"✅ Жёсткий стоп обновлён: {AI_HARD_STOP_BUDGET}"
    if action in {"secondary", "cheap"}:
        value_token = args[1] if len(args) > 1 else None
        if not value_token or value_token.lower() in {"off", "none", "0"}:
            AI_SECONDARY_BUDGET_START = 0
            os.environ.pop("OPENAI_SECONDARY_BUDGET_START", None)
            _persist_env_values({"OPENAI_SECONDARY_BUDGET_START": ""})
            return "✅ Порог переключения на дешёвую модель отключён."
        parsed = _parse_minutes_argument(value_token)
        if parsed is None:
            return "❌ Укажите числовое значение. Пример: /tokens secondary 70000"
        AI_SECONDARY_BUDGET_START = max(0, int(parsed))
        os.environ["OPENAI_SECONDARY_BUDGET_START"] = str(AI_SECONDARY_BUDGET_START)
        _persist_env_values({"OPENAI_SECONDARY_BUDGET_START": str(AI_SECONDARY_BUDGET_START)})
        return f"✅ Порог переключения обновлён: {AI_SECONDARY_BUDGET_START}"
    return (
        "ℹ️ Использование: /tokens, /tokens budget 150000, "
        "/tokens hard 200000, /tokens hard off, "
        "/tokens secondary 70000, /tokens reset"
    )


def _handle_bybit_key_command(args: list[str]) -> str:
    if not args:
        return "Использование: /bybitkey <apiKey> <apiSecret> или /bybitkey clear"
    action = args[0].strip().lower()
    if action in {"clear", "reset"}:
        _store_bybit_credentials(None, None)
        return "ℹ️? Ключи Bybit удалены. Добавьте новые ключи перед следующим запуском."
    if len(args) < 2:
        return "Укажите apiKey и apiSecret: /bybitkey <apiKey> <apiSecret>"
    api_key = args[0].strip()
    api_secret = args[1].strip()
    if not api_key or not api_secret:
        return "Ключ и секрет не должны быть пустыми."
    _store_bybit_credentials(api_key, api_secret)
    masked_key = _mask_api_value(api_key)
    masked_secret = _mask_api_value(api_secret)
    return (
        f"✅ Ключи Bybit обновлены (apiKey {masked_key}, secret {masked_secret}). "
        "Перезапустите цикл или дождитесь следующего запуска, чтобы применить их."
    )


def _persist_env_file(path: Path, updates: dict[str, str | None]) -> bool:
    try:
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        existing_lines = []
    except Exception:
        existing_lines = []
    seen: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _, _ = line.partition("=")
        key_clean = key.strip()
        if key_clean in updates:
            value = updates[key_clean]
            new_lines.append(f"{key_clean}={'' if value is None else value}")
            seen.add(key_clean)
        else:
            new_lines.append(line)
    for key, value in updates.items():
        if key in seen:
            continue
        new_lines.append(f"{key}={'' if value is None else value}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        log(f"[CONFIG] Не удалось обновить {path}: {exc}", Fore.YELLOW)
        return False


def _read_env_value(path: Path, key: str) -> Optional[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in lines:
        if not line or "=" not in line or line.strip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v
    return None


def _get_bot_config_path(bot_id: str, root: Path) -> Path:
    if bot_id == "default":
        return root / ".env"
    return root / "users" / bot_id / USERS_PUBLIC_ENV_FILE


def _resolve_config_scope(scope: str, user_id: Optional[int], bot_id: str) -> tuple[str, Path, Optional[dict], str]:
    scope_lower = scope.lower()
    if scope_lower in {"prod", "production", "main"}:
        return "prod", REPO_ROOT, None, bot_id
    if scope_lower.startswith("sandbox"):
        parts = scope_lower.split(":", 1)
        if len(parts) == 1 or not parts[1]:
            raise ValueError("Укажите идентификатор песочницы: sandbox:<id>")
        sandbox_id = parts[1]
        entry = _find_sandbox_entry(sandbox_id)
        if not entry:
            raise ValueError(f"Песочница {sandbox_id} не найдена.")
        path = Path(entry.get("path") or "")
        if not path.exists():
            raise ValueError(f"Каталог песочницы {sandbox_id} недоступен.")
        entry_bot = entry.get("bot_id") or bot_id
        return "sandbox", path, entry, entry_bot
    raise ValueError("Неизвестная область. Используйте prod или sandbox:<id>.")


def _config_list_scopes(user_id: Optional[int], bot_id: str) -> str:
    lines = ["Доступные области:"]
    if is_bot_owner(user_id, bot_id):
        lines.append("- prod — текущее прод-окружение бота")
    sandboxes = []
    if user_id is not None:
        sandboxes = _get_user_sandboxes(int(user_id))
    if sandboxes:
        lines.append("- sandbox:<id> — одна из ваших песочниц")
        for entry in sandboxes:
            status = "да" if entry.get("promotable") else "нет"
            lines.append(f"  • {entry.get('id')}: бот {entry.get('bot_id')} (promote={status})")
    else:
        lines.append("- У вас нет активных песочниц. Создайте их через Support.")
    return "\n".join(lines)


def _config_set(scope: str, key: str, value: str, *, user_id: Optional[int], bot_id: str) -> str:
    try:
        scope_type, root_path, sandbox_entry, target_bot_id = _resolve_config_scope(scope, user_id, bot_id)
    except ValueError as exc:
        return f"ℹ️ {exc}"
    if scope_type == "prod":
        if not is_bot_owner(user_id, target_bot_id):
            return "🚫 У вас нет прав изменять прод-окружение этого бота."
    else:
        entry_user = safe_int(sandbox_entry.get("user_id")) if sandbox_entry else None
        if user_id is None or (entry_user != user_id and not is_main_owner(user_id)):
            return "🚫 Вы можете менять параметры только в своих песочницах."
        target_bot_id = sandbox_entry.get("bot_id") or target_bot_id
    target_path = _get_bot_config_path(target_bot_id, root_path)
    updates = {key: value if value.lower() != "null" else None}
    if not _persist_env_file(target_path, updates):
        return "ℹ️ Не удалось обновить файл настроек."
    location = "проде" if scope_type == "prod" else f"песочнице {sandbox_entry.get('id')}"
    return f"✅ Параметр {key} обновлён в {location} ({target_path})."


def _config_get(scope: str, key: str, *, user_id: Optional[int], bot_id: str) -> str:
    try:
        scope_type, root_path, sandbox_entry, target_bot_id = _resolve_config_scope(scope, user_id, bot_id)
    except ValueError as exc:
        return f"ℹ️ {exc}"
    if scope_type == "prod":
        if not is_bot_owner(user_id, target_bot_id):
            return "🚫 У вас нет прав читать параметры этого прод-окружения."
    else:
        entry_user = safe_int(sandbox_entry.get("user_id")) if sandbox_entry else None
        if user_id is None or (entry_user != user_id and not is_main_owner(user_id)):
            return "🚫 Эта песочница вам не принадлежит."
        target_bot_id = sandbox_entry.get("bot_id") or target_bot_id
    target_path = _get_bot_config_path(target_bot_id, root_path)
    value = _read_env_value(target_path, key)
    if value is None:
        return f"ℹ️ {key} не задан в {target_path}."
    return f"{key} = {value}"


def _config_help(bot_id: str, user_id: Optional[int]) -> str:
    parts = [
        "Команда /config управляет параметрами бота.",
        "Примеры:",
        "  /config list — показать доступные области",
    ]
    if is_bot_owner(user_id, bot_id):
        parts.append("  /config set prod KEY VALUE — изменить параметр в прод-окружении")
        parts.append("    например: /config set prod AUTO_UPDATE 0 — отключить автообновление")
        parts.append("    или: /config set prod TARGET_VERSION 2025.11.06")
    parts.append("  /config set sandbox:<id> KEY VALUE — изменить параметр в песочнице")
    parts.append("  /config get sandbox:<id> KEY — посмотреть значение в песочнице")
    if is_bot_owner(user_id, bot_id):
        parts.append("Для прод-окружения обязательно указывайте область (prod или sandbox).")
    else:
        parts.append("Невладельцам доступны только песочницы.")
    return "\n".join(parts)


def _handle_config_command(
    args: list[str],
    *,
    user_id: Optional[int],
    bot_id: str,
) -> Optional[str]:
    if user_id is None:
        return "Команда доступна только после начала диалога с ботом. Откройте личный чат и нажмите Start."
    if not args:
        return _config_help(bot_id, user_id)
    action = args[0].lower()
    if action == "list":
        return _config_list_scopes(user_id, bot_id)
    if action == "set":
        if len(args) < 4:
            return _config_help(bot_id, user_id)
        scope = args[1]
        key = args[2]
        value = " ".join(args[3:])
        if scope.lower() in {"prod", "production"} and not is_bot_owner(user_id, bot_id):
            return (
                "Уточните песочницу: /config set sandbox:<id> KEY VALUE. "
                "Невладельцы не могут менять прод-окружение."
            )
        return _config_set(scope, key, value, user_id=user_id, bot_id=bot_id)
    if action == "get":
        if len(args) < 3:
            return "Использование: /config get <scope> <key>"
        scope = args[1]
        key = args[2]
        return _config_get(scope, key, user_id=user_id, bot_id=bot_id)
    return _config_help(bot_id, user_id)


def _delete_sandbox_directory(path: Path) -> tuple[bool, str]:
    try:
        if path.exists():
            shutil.rmtree(path)
        return True, "Песочница удалена."
    except Exception as exc:
        return False, f"Не удалось удалить каталог песочницы: {exc}"


def _promote_sandbox_entry(entry: dict[str, Any], user_id: Optional[int]) -> tuple[bool, str]:
    sandbox_id = entry.get("id")
    bot_id = entry.get("bot_id") or USER_ID
    if not is_bot_owner(user_id, bot_id):
        return False, "У вас нет прав переносить изменения этого бота."
    sandbox_path = Path(entry.get("path") or "")
    if not sandbox_path.exists():
        return False, "Каталог песочницы недоступен."
    targets: list[tuple[Path, Path]] = []
    if bot_id == "default":
        src = sandbox_path / ".env"
        dest = REPO_ROOT / ".env"
        targets.append((src, dest))
    else:
        src_dir = sandbox_path / "users" / bot_id
        dest_dir = USERS_DIR / bot_id
        src_env = src_dir / USERS_PUBLIC_ENV_FILE
        dest_env = dest_dir / USERS_PUBLIC_ENV_FILE
        targets.append((src_env, dest_env))
    copied = 0
    for src, dest in targets:
        if not src.exists():
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        shutil.copy2(src, dest)
        copied += 1
    if copied == 0:
        return False, "В песочнице не найдено нужных файлов для переноса."
    entry["promoted_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = _load_sandbox_state()
    sandboxes = state.get("sandboxes") or []
    for idx, existing in enumerate(sandboxes):
        if isinstance(existing, dict) and existing.get("id") == sandbox_id:
            sandboxes[idx] = entry
            break
    state["sandboxes"] = sandboxes
    _save_sandbox_state(state)
    return True, "Изменения из песочницы перенесены в прод."


def _handle_sandbox_command(args: list[str], *, user_id: Optional[int]) -> Optional[str]:
    if user_id is None:
        return "Команда доступна только после начала диалога с ботом. Откройте личный чат и нажмите Start."
    user_id_int = int(user_id)
    action = args[0].lower() if args else "list"
    if action in {"list", "ls"}:
        entries = _get_user_sandboxes(user_id_int)
        if not entries:
            return "У вас нет активных песочниц."
        lines = ["Ваши песочницы:"]
        for entry in sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True):
            lines.append(
                f"- {entry.get('id')}: бот {entry.get('bot_id')} "
                f"(promote={'да' if entry.get('promotable') else 'нет'}, создана {entry.get('created_at')})"
            )
        lines.append("Удаление: /sandbox delete <id>")
        lines.append("Перенос в прод (если доступно): /sandbox promote <id>")
        return "\n".join(lines)
    if action in {"delete", "rm"}:
        if len(args) < 2:
            return "Использование: /sandbox delete <id>"
        sandbox_id = args[1]
        entry = _find_sandbox_entry(sandbox_id)
        if not entry:
            return f"Песочница {sandbox_id} не найдена."
        entry_user = safe_int(entry.get("user_id"))
        if entry_user != user_id_int and not is_main_owner(user_id):
            return "Вы можете удалять только свои песочницы."
        success, message = _delete_sandbox_directory(Path(entry.get("path") or ""))
        if success:
            _remove_sandbox_entry(sandbox_id)
        return message
    if action == "promote":
        if len(args) < 2:
            return "Использование: /sandbox promote <id>"
        sandbox_id = args[1]
        entry = _find_sandbox_entry(sandbox_id)
        if not entry:
            return f"Песочница {sandbox_id} не найдена."
        if not entry.get("promotable"):
            return "Перенос изменений из этой песочницы запрещён."
        entry_user = safe_int(entry.get("user_id"))
        if entry_user != user_id_int and not is_main_owner(user_id):
            return "Вы можете переносить изменения только из собственных песочниц."
        success, message = _promote_sandbox_entry(entry, user_id)
        return message
    return (
        "Использование: /sandbox list, /sandbox delete <id>, /sandbox promote <id>\n"
        "Песочницы создаются автоматически через Support."
    )



def _send_adduser_prompt(user_id: int, text: str, *, parse_mode: str | None = "Markdown") -> bool:
    msg_id = send_tg(
        text,
        chat_id_override=user_id,
        no_log_forward=True,
        no_prefix=True,
        parse_mode=parse_mode,
    )
    return msg_id is not None


def _bot_id_exists(bot_id: str) -> bool:
    config = _load_users_config()
    for entry in config.get("users") or []:
        if isinstance(entry, dict) and entry.get("id") == bot_id:
            return True
    return bot_id == "default"


def _handle_add_user_command(
    args: list[str],
    *,
    user_id: Optional[int],
    origin_chat: int,
    origin_thread: Optional[int],
) -> Optional[str]:
    if user_id is None:
        return (
            "Для использования команды /adduser откройте личный чат с ботом и отправьте команду там "
            "(это необходимо, чтобы ключи не попали в общий чат)."
        )
    if user_id in PENDING_USERBOT_CREATION:
        return "Вы уже начали добавление юзер-бота. Завершите текущий процесс или отправьте /cancel в личном чате."
    target_id_default = str(user_id)
    label_default = args[0].strip() if args else ""
    flow_state = {
        "initiated_at": time.time(),
        "origin_chat": origin_chat,
        "origin_thread": origin_thread,
        "target_id": target_id_default,
        "label": label_default.strip() or target_id_default,
        "stage": "await_label" if not label_default else "await_api_key",
        "api_key": None,
        "api_secret": None,
        "owner_id": int(user_id),
        "timezone": None,
        "order_margin": None,
        "risk_pct": None,
        "leverage": None,
        "min_notional": None,
        "position_mode": None,
        "max_positions": None,
        "default_next_run": None,
        "state_dir": str(Path("runtime") / target_id_default),
    }
    label_prompt = (
        f"ID бота будет `{target_id_default}`.\n"
        "Введите отображаемое имя бота (или оставьте пустым, чтобы использовать тот же ID)."
    )
    start_prompt = (
        "🧩 Создание юзер-бота.\n"
        "Ответьте на вопросы последовательно; сообщения с чувствительными данными будут удалены.\n"
        "Для отмены отправьте /cancel.\n\n"
        + (
            label_prompt
            if flow_state["stage"] == "await_label"
            else "Введите API Key одной строкой. Сообщение будет удалено."
        )
    )
    dm_ready = _send_adduser_prompt(user_id, start_prompt)
    if not dm_ready:
        return (
            "Не удалось отправить личное сообщение. Убедитесь, что вы начали диалог с ботом (нажмите Start в личном чате), "
            "и повторите /adduser."
        )
    PENDING_USERBOT_CREATION[user_id] = flow_state
    return "📬 Продолжение — в личных сообщениях."
def _finalize_userbot_profile(flow_state: dict[str, Any]) -> tuple[bool, str]:
    target_id = flow_state.get("target_id")
    label = flow_state.get("label") or target_id
    api_key = flow_state.get("api_key")
    api_secret = flow_state.get("api_secret")
    if not target_id or not api_key or not api_secret:
        return False, "Недостаточно данных для создания профиля."
    try:
        _write_user_secrets(target_id, api_key, api_secret)
    except Exception as exc:
        return False, str(exc)
    state_dir = flow_state.get("state_dir") or f"runtime/{target_id}"
    public_env_rel = Path("users") / target_id / USERS_PUBLIC_ENV_FILE
    try:
        _ensure_user_entry(
            target_id,
            label,
            state_dir=state_dir,
            public_env_file=public_env_rel,
            owner_id=safe_int(flow_state.get("owner_id")),
        )
    except Exception as exc:
        return False, f"Не удалось обновить users.json: {exc}"
    env_updates = {
        "LOG_TIMEZONE": flow_state.get("timezone") or USERBOT_DEFAULTS["timezone"],
        "ORDER_MARGIN_UTILIZATION": f"{float(flow_state.get('order_margin') or USERBOT_DEFAULTS['order_margin']):.6f}",
        "RISK_PCT": f"{float(flow_state.get('risk_pct') or USERBOT_DEFAULTS['risk_pct']):.6f}",
        "LEVERAGE": str(int(flow_state.get("leverage") or USERBOT_DEFAULTS["leverage"])),
        "MIN_NOTIONAL_USDT": f"{float(flow_state.get('min_notional') or USERBOT_DEFAULTS['min_notional']):.6f}",
        "BYBIT_POSITION_MODE": (flow_state.get("position_mode") or USERBOT_DEFAULTS["position_mode"]),
        "MAX_OPEN_POSITIONS": str(int(flow_state.get("max_positions") or USERBOT_DEFAULTS["max_positions"])),
        "DEFAULT_NEXT_RUN_MINUTES": f"{float(flow_state.get('default_next_run') or USERBOT_DEFAULTS['default_next_run']):.3f}",
    }
    public_env_abs = USERS_DIR / target_id / USERS_PUBLIC_ENV_FILE
    _persist_env_file(public_env_abs, env_updates)
    return True, "Профиль создан."


def _process_pending_userbot_message(from_user_id: int, chat_id: int, message: dict) -> bool:
    flow_state = PENDING_USERBOT_CREATION.get(from_user_id)
    if not flow_state:
        return False
    if chat_id != from_user_id:
        return False
    text = (message.get("text") or "").strip()
    if not text:
        return True
    if text.startswith("/cancel"):
        PENDING_USERBOT_CREATION.pop(from_user_id, None)
        send_tg("🚫 Создание юзер-бота отменено.", chat_id_override=from_user_id, no_log_forward=True, no_prefix=True)
        return True
    message_id = message.get("message_id")
    if isinstance(message_id, int):
        _delete_tg_message(chat_id, message_id)
    stage = flow_state.get("stage") or "await_label"
    target_id = flow_state.get("target_id")
    if stage == "await_label":
        label = text.strip() or (target_id or "bot")
        flow_state["label"] = label
        flow_state["stage"] = "await_api_key"
        _send_adduser_prompt(
            from_user_id,
            "🔑 Теперь отправьте API Key одной строкой. Сообщение будет удалено.",
        )
        return True
    if stage == "await_api_key":
        flow_state["api_key"] = text
        flow_state["stage"] = "await_api_secret"
        _send_adduser_prompt(
            from_user_id,
            "🔒 API Key сохранён.\nТеперь отправьте *API Secret* (сообщение тоже будет удалено).",
        )
        return True
    if stage == "await_api_secret":
        flow_state["api_secret"] = text
        flow_state["stage"] = "await_timezone"
        _send_adduser_prompt(
            from_user_id,
            "🌐 Укажите часовой пояс бота (например, `UTC+3`). Оставьте пустым для UTC.",
        )
        return True
    if stage == "await_timezone":
        timezone = text.strip() or USERBOT_DEFAULTS["timezone"]
        flow_state["timezone"] = timezone
        flow_state["stage"] = "await_order_margin"
        _send_adduser_prompt(
            from_user_id,
            f"💰 Максимальный процент использования депозита (ORDER_MARGIN_UTILIZATION).\n"
            f"Введите число от 0 до 1 (например, 0.75). По умолчанию {USERBOT_DEFAULTS['order_margin']}.",
        )
        return True
    if stage == "await_order_margin":
        raw_text = text.strip()
        if not raw_text:
            val = USERBOT_DEFAULTS["order_margin"]
        else:
            try:
                val = float(raw_text.replace(",", "."))
            except ValueError:
                _send_adduser_prompt(
                    from_user_id,
                    "❗ Введите число от 0 до 1 (например, 0.75). Попробуйте ещё раз.",
                    parse_mode=None,
                )
                return True
        if val < 0 or val > 1:
            _send_adduser_prompt(
                from_user_id,
                "❗ Введите число от 0 до 1 (например, 0.75). Попробуйте ещё раз.",
                parse_mode=None,
            )
            return True
        flow_state["order_margin"] = val
        flow_state["stage"] = "await_risk_pct"
        _send_adduser_prompt(
            from_user_id,
            f"ℹ️ Максимальный риск на сделку (RISK_PCT).\n"
            f"Введите долю от депозита (например, 0.005 для 0.5%). По умолчанию {USERBOT_DEFAULTS['risk_pct']}.",
        )
        return True
    if stage == "await_risk_pct":
        raw_text = text.strip()
        if not raw_text:
            val = USERBOT_DEFAULTS["risk_pct"]
        else:
            try:
                val = float(raw_text.replace(",", "."))
            except ValueError:
                _send_adduser_prompt(
                    from_user_id,
                    "❗ Введите число (например, 0.005). Попробуйте ещё раз.",
                    parse_mode=None,
                )
                return True
        if val <= 0 or val > 1:
            _send_adduser_prompt(
                from_user_id,
                "❗ Введите число (например, 0.005). Попробуйте ещё раз.",
                parse_mode=None,
            )
            return True
        flow_state["risk_pct"] = val
        flow_state["stage"] = "await_leverage"
        _send_adduser_prompt(
            from_user_id,
            f"📈 Плечо (LEVERAGE). Введите целое число, например 5. По умолчанию {USERBOT_DEFAULTS['leverage']}.",
        )
        return True
    if stage == "await_leverage":
        raw_text = text.strip()
        if not raw_text:
            leverage = int(USERBOT_DEFAULTS["leverage"])
        else:
            try:
                leverage = int(float(raw_text))
            except ValueError:
                _send_adduser_prompt(
                    from_user_id,
                    "❗ Введите целое число (например, 5). Попробуйте ещё раз.",
                    parse_mode=None,
                )
                return True
        if leverage <= 0 or leverage > 100:
            _send_adduser_prompt(
                from_user_id,
                "❗ Введите целое число (например, 5). Попробуйте ещё раз.",
                parse_mode=None,
            )
            return True
        flow_state["leverage"] = leverage
        flow_state["stage"] = "await_min_notional"
        _send_adduser_prompt(
            from_user_id,
            f"🔢 Минимальный размер позиции (MIN_NOTIONAL_USDT). Введите число в USDT (может быть 0). По умолчанию "
            f"{USERBOT_DEFAULTS['min_notional']}.",
        )
        return True
    if stage == "await_min_notional":
        raw_text = text.strip()
        if not raw_text:
            min_notional = USERBOT_DEFAULTS["min_notional"]
        else:
            try:
                min_notional = float(raw_text.replace(",", "."))
            except ValueError:
                _send_adduser_prompt(
                    from_user_id,
                    "❗ Введите число (например, 5). Попробуйте ещё раз.",
                    parse_mode=None,
                )
                return True
        if min_notional < 0:
            _send_adduser_prompt(
                from_user_id,
                "❗ Введите число (например, 5). Попробуйте ещё раз.",
                parse_mode=None,
            )
            return True
        flow_state["min_notional"] = min_notional
        flow_state["stage"] = "await_position_mode"
        _send_adduser_prompt(
            from_user_id,
            f"🔀 Режим позиций (BYBIT_POSITION_MODE).\nВведите `hedged` (хедж) или `oneway` (по умолчанию {USERBOT_DEFAULTS['position_mode']}).",
        )
        return True
    if stage == "await_position_mode":
        mode = text.strip().lower()
        if mode in {"hedge", "hedged", "dual", "dual_side"}:
            normalized_mode = "hedged"
        elif mode in {"oneway", "one_way", "single"}:
            normalized_mode = "oneway"
        elif not mode:
            normalized_mode = USERBOT_DEFAULTS["position_mode"]
        else:
            _send_adduser_prompt(
                from_user_id,
                "❗ Допустимые значения: hedged или oneway. Попробуйте ещё раз.",
                parse_mode=None,
            )
            return True
        flow_state["position_mode"] = normalized_mode
        flow_state["stage"] = "await_max_positions"
        _send_adduser_prompt(
            from_user_id,
            "📊 Максимальное число открытых позиций (MAX_OPEN_POSITIONS). Введите целое число.",
        )
        return True
    if stage == "await_max_positions":
        raw_text = text.strip()
        if not raw_text:
            max_positions = int(USERBOT_DEFAULTS["max_positions"])
        else:
            try:
                max_positions = int(float(raw_text))
            except ValueError:
                _send_adduser_prompt(
                    from_user_id,
                    "❗ Введите целое число (например, 4). Попробуйте ещё раз.",
                    parse_mode=None,
                )
                return True
        if max_positions <= 0 or max_positions > 20:
            _send_adduser_prompt(
                from_user_id,
                "❗ Введите целое число (например, 4). Попробуйте ещё раз.",
                parse_mode=None,
            )
            return True
        flow_state["max_positions"] = max_positions
        flow_state["stage"] = "await_default_next_run"
        _send_adduser_prompt(
            from_user_id,
            f"⏱ Дефолтный интервал между циклами (DEFAULT_NEXT_RUN_MINUTES). Введите минуты (может быть дробным числом). "
            f"По умолчанию {USERBOT_DEFAULTS['default_next_run']}.",
        )
        return True
    if stage == "await_default_next_run":
        raw_text = text.strip()
        if not raw_text:
            default_run = float(USERBOT_DEFAULTS["default_next_run"])
        else:
            try:
                default_run = float(raw_text.replace(",", "."))
            except ValueError:
                _send_adduser_prompt(
                    from_user_id,
                    "❗ Введите число (например, 45). Попробуйте ещё раз.",
                    parse_mode=None,
                )
                return True
        if default_run < 0:
            _send_adduser_prompt(
                from_user_id,
                "❗ Значение не может быть отрицательным.",
                parse_mode=None,
            )
            return True
        flow_state["default_next_run"] = default_run
        flow_state["stage"] = "finalizing"
        success, detail = _finalize_userbot_profile(flow_state)
        target_id = flow_state.get("target_id")
        label = flow_state.get("label") or target_id
        origin_thread = flow_state.get("origin_thread")
        origin_chat = flow_state.get("origin_chat")
        PENDING_USERBOT_CREATION.pop(from_user_id, None)
        _refresh_userbot_owners()
        if success:
            preview_timezone = flow_state.get("timezone") or USERBOT_DEFAULTS["timezone"]
            preview_margin = flow_state.get("order_margin") or USERBOT_DEFAULTS["order_margin"]
            preview_risk = flow_state.get("risk_pct") or USERBOT_DEFAULTS["risk_pct"]
            preview_leverage = flow_state.get("leverage") or USERBOT_DEFAULTS["leverage"]
            preview_min_notional = flow_state.get("min_notional") or USERBOT_DEFAULTS["min_notional"]
            preview_position_mode = flow_state.get("position_mode") or USERBOT_DEFAULTS["position_mode"]
            preview_max_positions = flow_state.get("max_positions") or USERBOT_DEFAULTS["max_positions"]
            preview_default_next = flow_state.get("default_next_run") or USERBOT_DEFAULTS["default_next_run"]
            config_preview = (
                f"🛠 Параметры:\n"
                f"- timezone: {preview_timezone}\n"
                f"- margin: {preview_margin}\n"
                f"- risk_pct: {preview_risk}\n"
                f"- leverage: {preview_leverage}\n"
                f"- min_notional: {preview_min_notional}\n"
                f"- position_mode: {preview_position_mode}\n"
                f"- max_positions: {preview_max_positions}\n"
                f"- default_next_run: {preview_default_next}"
            )
            send_tg(
                f"✅ Юзер-бот `{target_id}` создан.\nКлючи безопасно сохранены.\n{config_preview}",
                chat_id_override=from_user_id,
                no_log_forward=True,
                no_prefix=True,
                parse_mode=None,
            )
            if origin_chat is not None:
                summary = (
                    f"✅ Пользователь `{target_id}` ({label}) добавлен.\n"
                    f"API Key: {_mask_sensitive(flow_state.get('api_key', ''))}\n"
                    f"API Secret: {_mask_sensitive(flow_state.get('api_secret', ''))}"
                )
                send_tg(
                    summary,
                    thread_id=origin_thread,
                    chat_id_override=origin_chat,
                    no_log_forward=True,
                    parse_mode="Markdown",
                )
        else:
            send_tg(
                f"ℹ️ Не удалось создать юзер-бота: {detail}",
                chat_id_override=from_user_id,
                no_log_forward=True,
                no_prefix=True,
            )
        return True
    return False
