# -*- coding: utf-8 -*-
"""
Bybit Intraday AI Trading Bot — 30m, 5 пар USDT Perpetual
Сбалансированный интрадей-бот с поддержкой OpenAI GPT, Telegram и расширенным контекстом.
"""

# Версия бота: обновляйте при каждом релизе/значимых изменениях
BOT_VERSION = "2025.10.16.6"
BOT_CHANGELOG = (
    "Автоучёт открытых ордеров: поддержка cancel_orders/replace_orders и обновление TP/SL без новых лимитов; "
    "версия и изменение выводятся в конце цикла."
)

# --- Безопасные настройки OpenBLAS (исключаем падения из-за многопоточности) ---
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# --- Импорты ---
import math, time, json, traceback, datetime, random, warnings, re
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

# Подавляем FutureWarning от pandas
warnings.filterwarnings("ignore", category=FutureWarning)
init(autoreset=True)

LOG_TZINFO = None
LOG_TIMEZONE = ""
_LOG_TZ_WARNING_EMITTED = False


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
    global PAIR_LIST, TIMEFRAME, LEVERAGE, RISK_PCT, SL_ATR, TP_ATR
    global MIN_NOTIONAL_USDT, AI_AFTER_NEEDS_BIAS, MAX_OPEN_POSITIONS
    global MIN_CONTEXT_30M, MIN_CONTEXT_4H, DEFAULT_CONTEXT_30M, DEFAULT_CONTEXT_4H
    global CONTEXT_STEP_30M, CONTEXT_STEP_4H
    global TG_TOKEN, TG_CHAT, AI_MODEL, AI_KEY
    global NEWS_API_TOKEN, NEWS_API_ENDPOINT, NEWS_API_KINDS, NEWS_API_FILTER, NEWS_ITEMS_LIMIT
    global POSITION_MODE, HEDGE_MODE, ORDER_MARGIN_UTILIZATION
    global LOG_TIMEZONE, LOG_TZINFO, _LOG_TZ_WARNING_EMITTED
    PAIR_LIST = os.getenv("PAIR_LIST", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,DOGE/USDT:USDT").split(",")
    TIMEFRAME = os.getenv("TIMEFRAME", "30m")
    LEVERAGE = int(os.getenv("LEVERAGE", 10))
    RISK_PCT = float(os.getenv("RISK_PCT", os.getenv("RISK_EQUITY_PCT", 0.015)))
    SL_ATR = float(os.getenv("SL_ATR", os.getenv("SL_ATR_MULT", 0.8)))
    TP_ATR = float(os.getenv("TP_ATR", os.getenv("TP_ATR_MULT", 1.6)))
    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", 5.0))
    AI_AFTER_NEEDS_BIAS = int(os.getenv("AI_AFTER_NEEDS_BIAS", 1))
    MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 0)

    TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")
    AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
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

    NEWS_API_TOKEN = os.getenv("CRYPTO_NEWS_TOKEN") or os.getenv("NEWS_API_TOKEN")
    NEWS_API_ENDPOINT = os.getenv("CRYPTO_NEWS_ENDPOINT", "https://cryptopanic.com/api/v1/posts/")
    NEWS_API_KINDS = os.getenv("CRYPTO_NEWS_KIND", "news,media")
    NEWS_API_FILTER = os.getenv("CRYPTO_NEWS_FILTER", "important")
    NEWS_ITEMS_LIMIT = env_int("CRYPTO_NEWS_LIMIT", 5)
    POSITION_MODE = (os.getenv("BYBIT_POSITION_MODE") or "oneway").strip().lower()
    HEDGE_MODE = POSITION_MODE in ("hedge", "hedged", "dual", "dual_side", "dual-side")
    try:
        ORDER_MARGIN_UTILIZATION = float(os.getenv("ORDER_MARGIN_UTILIZATION", 0.95))
    except (TypeError, ValueError):
        ORDER_MARGIN_UTILIZATION = 0.95
    ORDER_MARGIN_UTILIZATION = max(0.1, min(ORDER_MARGIN_UTILIZATION, 1.0))
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

# --- Вспомогательные функции ---
def _current_log_time():
    base = datetime.datetime.now(datetime.timezone.utc)
    if LOG_TZINFO is not None:
        return base.astimezone(LOG_TZINFO)
    return base.astimezone()


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

def send_tg(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      json={"chat_id": TG_CHAT, "text": msg}, timeout=5)
    except Exception as e:
        log(f"Ошибка Telegram: {e}", Fore.YELLOW)

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


def fetch_open_orders_for_symbol(exchange, symbol, limit=10):
    has_attr = getattr(exchange, "has", {})
    if isinstance(has_attr, dict) and not has_attr.get("fetchOpenOrders", False):
        return []
    try:
        raw_orders = exchange.fetch_open_orders(symbol)
    except Exception as e:
        log(f"⚠️ Не удалось получить открытые ордера для {symbol}: {e}", Fore.YELLOW)
        return []
    simplified = []
    for order in raw_orders:
        simplified.append(simplify_order(order))
        if len(simplified) >= limit:
            break
    return simplified


def cancel_order_by_id(exchange, symbol, order_id: str):
    try:
        exchange.cancel_order(order_id, symbol)
        return True, None
    except Exception as e:
        return False, str(e)


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
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
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
            item = {
                "title": title,
                "url": link,
                "source": entry.get("source", {}).get("title") if isinstance(entry.get("source"), dict) else entry.get("source"),
                "published_at": entry.get("published", entry.get("updated"))
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


def get_news(symbol):
    base = symbol.split("/")[0].split(":")[0].upper()
    limit = max(1, NEWS_ITEMS_LIMIT)
    if NEWS_API_TOKEN:
        params = {
            "auth_token": NEWS_API_TOKEN,
            "currencies": base,
            "kind": NEWS_API_KINDS,
            "filter": NEWS_API_FILTER,
            "public": "true"
        }
        try:
            resp = requests.get(NEWS_API_ENDPOINT, params=params, timeout=6)
            resp.raise_for_status()
            payload = resp.json()
            entries = payload.get("results") or payload.get("data") or []
            news_items = []
            for entry in entries:
                if len(news_items) >= limit:
                    break
                title = entry.get("title") or entry.get("headline")
                url = entry.get("url")
                source = (entry.get("source") or {}).get("title") if isinstance(entry.get("source"), dict) else entry.get("source")
                published = entry.get("published_at") or entry.get("created_at") or entry.get("timestamp")
                news_items.append({
                    "title": title,
                    "url": url,
                    "source": source,
                    "kind": entry.get("kind"),
                    "published_at": to_iso_utc(published)
                })
            if news_items:
                latest = news_items[0].get("published_at")
                summary = f"{len(news_items)} новостей CryptoPanic, последняя {latest}"
                return {"summary": summary, "items": news_items, "asset": base, "source": "cryptopanic"}
            log(f"ℹ️ CryptoPanic не вернул новости для {symbol}, используем RSS", Fore.LIGHTBLACK_EX)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else None
            color = Fore.LIGHTBLACK_EX if status and status >= 500 else Fore.YELLOW
            status_text = f"HTTP {status}" if status else "HTTP error"
            log(f"⚠️ CryptoPanic недоступен для {symbol}: {status_text} — {e}", color)
        except Exception as e:
            log(f"⚠️ CryptoPanic недоступен для {symbol}: {e}", Fore.YELLOW)
    # Fallback to RSS
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
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=200)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def ensure_position_mode(exchange):
    desired = "hedged" if HEDGE_MODE else "oneway"
    try:
        if hasattr(exchange, "set_position_mode") and PAIR_LIST:
            exchange.set_position_mode(HEDGE_MODE, PAIR_LIST[0])
            log(f"⚙️ Режим позиций установлен: {desired}", Fore.LIGHTBLACK_EX)
    except Exception as e:
        code = get_bybit_retcode(e)
        if code == 110025:
            log(f"ℹ️ Режим позиций уже установлен ({desired}, код {code})", Fore.LIGHTBLACK_EX)
        else:
            log(f"⚠️ Не удалось установить режим позиций ({desired}): {e}", Fore.YELLOW)


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


def execute_extra_orders(exchange, symbol, orders, current_position=None, open_orders=None):
    executed = []
    open_orders = open_orders or []
    reduce_only_map = {}
    for existing in open_orders:
        try:
            reduce_flag = existing.get("reduceOnly")
        except AttributeError:
            continue
        if reduce_flag in (True, "true", "1", 1):
            side_key = (existing.get("side") or "").lower()
            reduce_only_map.setdefault(side_key, []).append(existing)
    if not isinstance(orders, (list, tuple)):
        log(f"⚠️ Некорректный формат orders для {symbol}: ожидается список", Fore.YELLOW)
        return executed, False
    cancelled_success = []
    cancel_errors = []
    for idx, order in enumerate(orders, 1):
        if not isinstance(order, dict):
            log(f"⚠️ Пропуск order #{idx} для {symbol}: ожидается объект", Fore.YELLOW)
            continue
        order_type_key = (order.get("type") or "limit").lower()
        params = dict(order.get("params") or {})
        note = order.get("note") or order.get("comment") or ""
        reduce_only = order.get("reduceOnly")
        if reduce_only is not None:
            params["reduceOnly"] = bool(reduce_only)
        side = (order.get("side") or "").lower()
        amount = compute_order_amount(order, current_position)
        price = order.get("price")
        try:
            if price is not None:
                price = float(price)
        except (TypeError, ValueError):
            log(f"⚠️ Некорректная цена в order #{idx} для {symbol}", Fore.YELLOW)
            continue

        if order_type_key == "partial_close":
            base_order_type = (order.get("orderType") or order.get("order_type") or order.get("ccxt_type") or "market").lower()
            ccxt_type = ORDER_TYPE_MAP.get(base_order_type, base_order_type)
            params.setdefault("reduceOnly", True)
            if not side and current_position:
                side = "sell" if (current_position.get("amount") or 0) > 0 else "buy"
        else:
            ccxt_type = ORDER_TYPE_MAP.get(order_type_key, order_type_key)

        if not side:
            log(f"⚠️ Не указан side в order #{idx} для {symbol}", Fore.YELLOW)
            continue
        if amount is None:
            log(f"⚠️ Не удалось определить объём ордера #{idx} для {symbol}", Fore.YELLOW)
            continue
        position_idx = order.get("positionIdx")
        if position_idx is None:
            params.setdefault("positionIdx", get_position_idx(side))
        else:
            params["positionIdx"] = position_idx

        if ccxt_type in ("limit", "stopLimit", "takeProfit", "stopLoss") and price is None:
            log(f"⚠️ Нужна цена для ордера #{idx} ({ccxt_type}) {symbol}", Fore.YELLOW)
            continue

        if params.get("reduceOnly"):
            existing_list = reduce_only_map.get(side)
            if existing_list:
                for existing_order in existing_list:
                    oid = existing_order.get("id")
                    if not oid:
                        continue
                    success, err = cancel_order_by_id(exchange, symbol, str(oid))
                    if success:
                        cancelled_success.append(str(oid))
                        log(f"🗑️ Отменён существующий reduce-only ордер {oid} для {symbol} перед заменой", Fore.LIGHTBLUE_EX)
                    else:
                        cancel_errors.append((oid, err))
                        log(f"⚠️ Не удалось отменить reduce-only ордер {oid} для {symbol}: {err}", Fore.YELLOW)
                reduce_only_map[side] = []

        try:
            order_id = exchange.create_order(symbol, ccxt_type, side, amount, price, params)
            desc = f"{ccxt_type.upper()} {side.upper()} {amount}"
            if price:
                desc += f" @ {price}"
            if note:
                desc += f" — {note}"
            executed.append(desc)
            log(f"🛠️ Доп. ордер для {symbol}: {desc}", Fore.LIGHTBLUE_EX)
        except Exception as e:
            log(f"❌ Ошибка доп. ордера #{idx} для {symbol}: {e}", Fore.RED)
    if cancelled_success:
        send_tg(f"🗑️ {symbol}: отменены ордера {', '.join(cancelled_success)} перед заменой")
    if cancel_errors:
        errs = "; ".join(f"{oid}: {err}" for oid, err in cancel_errors)
        send_tg(f"⚠️ {symbol}: ошибки отмены ордеров — {errs}")
    actions_performed = bool(executed or cancelled_success or cancel_errors)
    return executed, actions_performed

# --- Решение модели (2 прохода, русский лог) ---
def ai_decision(symbol, df_30m, equity, available_margin, exchange, current_position=None, open_orders=None):
    if not AI_KEY:
        log("❌ Не указан OPENAI_API_KEY", Fore.RED)
        return None

    client = OpenAI(api_key=AI_KEY, timeout=15)
    df_30m["ema20"] = ema(df_30m["close"],20)
    df_30m["ema50"] = ema(df_30m["close"],50)
    df_30m["rsi"] = rsi(df_30m["close"],14)
    df_30m["atr"] = atr(df_30m,14)
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

    system_msg = (
        "\u0422\u044b \u2014 \u0418\u0418-\u043f\u043e\u043c\u043e\u0449\u043d\u0438\u043a \u043f\u043e \u0442\u0440\u0435\u0439\u0434\u0438\u043d\u0433\u0443 \u0432
\u0441\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e\u043c \u0438\u043d\u0442\u0440\u0430\u0434\u0435\u0439 \u0441\u0442\u0438\u043b\u0435. "
        "\u0420\u0430\u0431\u043e\u0442\u0430\u0435\u0448\u044c \u043f\u043e \u0441\u0446\u0435\u043d\u0430\u0440\u0438\u044e: (1) \u0435\u0441\u043b\u0438 \u0442\u0440\u0435\u043d\u0434\u044b 30m \u0438 4h \u0441\u043e\u0432\u043f\u0430\u0434\u0430\u044e\u0442
\u0438 RSI \u043d\u0435 \u0432 \u044d\u043a\u0441\u0442\u0440\u0435\u043c\u0443\u043c\u0430\u0445 \u2014 \u0432\u0445\u043e\u0434\u0438 \u043f\u043e \u0442\u0440\u0435\u043d\u0434\u0443; "
        "(2) \u0435\u0441\u043b\u0438 30m \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 \u0437\u0430\u0440\u043e\u0436\u0434\u0430\u044e\u0449\u0438\u0439\u0441\u044f \u0440\u0430\u0437\u0432\u043e\u0440\u043e\u0442
\u043f\u0440\u043e\u0442\u0438\u0432 \u0441\u043b\u0430\u0431\u043e\u0433\u043e \u0442\u0440\u0435\u043d\u0434\u0430 \u043d\u0430 4h \u2014 \u0434\u043e\u043f\u0443\u0441\u043a\u0430\u0435\u0442\u0441\u044f
\u043a\u043e\u043d\u0442\u0440\u0442\u0440\u0435\u043d\u0434 \u0441 \u043a\u043e\u0440\u043e\u0442\u043a\u043e\u0439 \u0446\u0435\u043b\u044c\u044e; "
        "(3) skip \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442\u0441\u044f \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u0440\u0438 \u0440\u0435\u0430\u043b\u044c\u043d\u043e\u043c \u043a\u043e\u043d\u0444\u043b\u0438\u043a\u0442\u0435
\u0441\u0438\u0433\u043d\u0430\u043b\u043e\u0432 \u0438\u043b\u0438 \u044f\u0432\u043d\u043e\u0439 \u043d\u0435\u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0451\u043d\u043d\u043e\u0441\u0442\u0438. "
        "\u041e\u0431\u044f\u0437\u0430\u0442\u0435\u043b\u044c\u043d\u043e \u0430\u043d\u0430\u043b\u0438\u0437\u0438\u0440\u0443\u0439 EMA20/EMA50, RSI(14), ATR(14) \u043d\u0430 30m \u0438 4h, \u0444\u043e\u0440\u043c\u0438\u0440\u0443\u0439
\u043f\u043e\u043d\u044f\u0442\u043d\u044b\u0439 \u0440\u0438\u0441\u043a/\u0438\u0434\u0435\u044e. "
        "\u0415\u0441\u043b\u0438 \u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u044c < 70% \u0438\u043b\u0438 \u0441\u0438\u0433\u043d\u0430\u043b\u044b \u0440\u0430\u0441\u0445\u043e\u0434\u044f\u0442\u0441\u044f \u2014
\u0441\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u043f\u0440\u043e\u0441\u0438 \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0447\u0435\u0440\u0435\u0437
\u043f\u043e\u043b\u0435 'needs' "
        "(\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e: higher_tf:<tf>, funding, open_interest, news), \u0438 \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u043e\u0441\u043b\u0435 \u0434\u043e\u043f. \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438
\u0432\u044b\u0431\u0438\u0440\u0430\u0439 \u043a\u043e\u043d\u0435\u0447\u043d\u043e\u0435 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435. "
        "\u0415\u0441\u043b\u0438 \u043f\u043e\u0437\u0438\u0446\u0438\u044f \u0443\u0436\u0435 \u043e\u0442\u043a\u0440\u044b\u0442\u0430, \u043d\u0435 \u043e\u0442\u043a\u0440\u044b\u0432\u0430\u0439 \u0435\u0451 \u0437\u0430\u043d\u043e\u0432\u043e:
\u043e\u0446\u0435\u043d\u0438 \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e\u0441\u0442\u044c \u0447\u0430\u0441\u0442\u0438\u0447\u043d\u043e\u0433\u043e \u0441\u043e\u043a\u0440\u0430\u0449\u0435\u043d\u0438\u044f,
\u0437\u0430\u043a\u0440\u044b\u0442\u0438\u044f \u0438\u043b\u0438 \u0443\u0434\u0435\u0440\u0436\u0430\u043d\u0438\u044f. "
        "\u0415\u0441\u043b\u0438 \u043f\u043e \u0441\u0438\u043c\u0432\u043e\u043b\u0443 \u0435\u0441\u0442\u044c \u0430\u043a\u0442\u0438\u0432\u043d\u044b\u0435
\u043b\u0438\u043c\u0438\u0442\u043d\u044b\u0435/\u0441\u0442\u043e\u043f-\u043e\u0440\u0434\u0435\u0440\u0430 (open_orders), \u043d\u0435 \u0434\u0443\u0431\u043b\u0438\u0440\u0443\u0439 \u0438\u0445 \u0431\u0435\u0437
\u043f\u0435\u0440\u0435\u0441\u043c\u043e\u0442\u0440\u0430. "
        "\u0414\u043b\u044f \u043e\u0442\u043c\u0435\u043d\u044b/ \u0437\u0430\u043c\u0435\u043d\u044b \u043e\u0440\u0434\u0435\u0440\u043e\u0432 \u043f\u0435\u0440\u0435\u0434\u0430\u0432\u0430\u0439 cancel_orders \u0438 replace_orders. "
        "\u0414\u043b\u044f \u0447\u0430\u0441\u0442\u0438\u0447\u043d\u044b\u0445 \u0437\u0430\u043a\u0440\u044b\u0442\u0438\u0439, \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0445
\u043b\u0438\u043c\u0438\u0442\u043e\u0432/\u0441\u0442\u043e\u043f\u043e\u0432, \u0442\u0440\u0435\u0439\u043b\u0438\u043d\u0433\u043e\u0432 \u0438 \u0434\u0440\u0443\u0433\u0438\u0445 \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u0439
\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 \u043c\u0430\u0441\u0441\u0438\u0432 'orders', \u043e\u043f\u0438\u0441\u044b\u0432\u0430\u044f \u043e\u0440\u0434\u0435\u0440\u0430 \u0432 \u0441\u0442\u0438\u043b\u0435 CCXT (type, side, amount/percent,
price, params). "
        "\u0415\u0441\u043b\u0438 \u0432\u044b\u0431\u0438\u0440\u0430\u0435\u0448\u044c action=\\\"skip\\\", \u043e\u0431\u044f\u0437\u0430\u0442\u0435\u043b\u044c\u043d\u043e \u0443\u043a\u0430\u0436\u0438 \u043f\u0440\u0438\u0447\u0438\u043d\u0443,
\u043e\u043f\u0438\u0440\u0430\u044f\u0441\u044c \u043d\u0430 \u043f\u043e\u043a\u0430\u0437\u0430\u043d\u0438\u044f \u044d\u0442\u0438\u0445 \u0438\u043d\u0434\u0438\u043a\u0430\u0442\u043e\u0440\u043e\u0432. "
        "\u041e\u0442\u0432\u0435\u0442 \u0441\u0442\u0440\u043e\u0433\u043e \u0432 \u0444\u043e\u0440\u043c\u0430\u0442\u0435 JSON \u0431\u0435\u0437 \u0442\u0435\u043a\u0441\u0442\u0430."
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
        news_desc = "CryptoPanic API (fallback: RSS крипто-ленты)" if NEWS_API_TOKEN else "RSS новости по ключевому активу"
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

    messages_init, tokens_init, _ = prepare_messages(stage="initial")
    log(f"ℹ️ Токены запроса (initial) для {symbol}: {tokens_init}", Fore.LIGHTBLACK_EX)

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
    msg = res.choices[0].message.content
    decision = json.loads(msg)
    needs = decision.get("needs", [])
    auto_needs_triggered = False
    action_initial = (decision.get("action") or "").lower()
    if not needs and action_initial == "skip":
        reason_text = (decision.get("reason") or "").lower()
        keywords_auto_needs = ("запрос", "needs", "дополнитель", "подтвержден")
        if any(word in reason_text for word in keywords_auto_needs):
            auto_needs = ["funding", "open_interest", "news"]
            decision["needs"] = auto_needs
            needs = auto_needs
            auto_needs_triggered = True
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
            "auto_needs": auto_needs_triggered
        }
    )

    # --- Если запрошен контекст ---
    if needs:
        if auto_needs_triggered:
            log(f"🤖 Автозапрос дополнительного контекста по причине низкой уверенности: {needs}", Fore.CYAN)
            send_tg(f"🤖 Автозапрос данных для {symbol}: {needs}")
        else:
            log(f"🤖 Модель запросила дополнительный контекст: {needs}", Fore.CYAN)
            send_tg(f"🤖 Модель запросила контекст для {symbol}: {needs}")
        extra = {}
        needs_followup = []
        for n in needs:
            if n.startswith("higher_tf"):
                tf = n.split(":",1)[1] if ":" in n else "4h"
                extra.setdefault("higher_tf", {})[tf] = get_higher_tf(exchange, symbol, tf)
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

        bias_flag = bool(AI_AFTER_NEEDS_BIAS)
        messages_extra, tokens_extra, _ = prepare_messages(stage="extra", extra=extra, bias=bias_flag)
        log(f"ℹ️ Токены запроса (extra) для {symbol}: {tokens_extra}", Fore.LIGHTBLACK_EX)
        start_extra = time.perf_counter()
        res2 = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0,
            response_format={"type":"json_object"},
            messages=messages_extra
        )
        duration_extra = time.perf_counter() - start_extra
        log(f"⏱️ OpenAI extra запрос для {symbol}: {duration_extra:.2f} c", Fore.LIGHTBLACK_EX)
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
def main():
    refresh_settings()
    ex = init_exchange()
    ex.load_markets()
    ensure_position_mode(ex)
    positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=PAIR_LIST)
    if MAX_OPEN_POSITIONS > 0 and open_positions is None:
        log("⚠️ Не удалось определить количество открытых позиций — лимит по позициям отключён на этот цикл", Fore.YELLOW)
        open_positions = None
    equity, available_margin, _ = fetch_usdt_equity(ex)
    if equity <= 0:
        equity = 64.0
    if available_margin <= 0:
        available_margin = equity
    last_equity = equity
    last_available_margin = available_margin
    log(f"🚀 Бот v{BOT_VERSION} запущен. Баланс: {equity:.2f} USDT, доступно {available_margin:.2f} USDT", Fore.GREEN)
    send_tg(f"🚀 Бот запущен. Баланс: {equity:.2f} USDT, доступно {available_margin:.2f} USDT")

    decisions_total = 0
    counts = {"open":0,"close":0,"skip":0}

    for i,sym in enumerate(PAIR_LIST,1):
        log(f"[{i}/{len(PAIR_LIST)}] {sym}", Fore.LIGHTBLUE_EX)
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
            df = fetch_df(ex, sym, "30m")
            current_position = positions_map.get(sym)
            open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)
            dec = ai_decision(
                sym,
                df,
                equity,
                available_margin,
                ex,
                current_position=current_position,
                open_orders=open_orders_symbol
            )
            if not dec: continue
            save_json_line(AI_LOG_FILE, {"timestamp":datetime.datetime.now().isoformat(),
                                         "symbol":sym,"decision":dec})
            action = (dec.get("action") or "skip").lower()
            side = dec.get("side") or ""
            reason = dec.get("reason") or ""
            counts[action] = counts.get(action,0)+1
            decisions_total += 1

            extra_orders_raw = dec.get("orders") or dec.get("adjustments") or dec.get("extra_orders") or []
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

            def try_cancel(order_id: str, source: str):
                oid = str(order_id)
                if not oid or oid in cancelled_ids:
                    return
                success, err = cancel_order_by_id(ex, sym, oid)
                if success:
                    cancelled_ids.add(oid)
                    cancelled_success.append((oid, source))
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
                            positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=PAIR_LIST)
                            current_position = positions_map.get(sym)
                        except Exception as e:
                            err_text = str(e)
                            log(f"❌ Ошибка закрытия {sym}: {err_text}", Fore.RED)
                            send_tg(f"❌ Ошибка закрытия для {sym}: {err_text}")
            elif action == "hold":
                log(f"⏳ Удерживаем {sym} ({reason})", Fore.BLUE)
                send_tg(f"⏳ {sym}: удерживаем позицию — {reason or 'причина не указана'}")
            elif action == "open":
                if current_position and abs(float(current_position.get("amount") or 0)) > 0:
                    log(f"⚠️ Позиция по {sym} уже открыта (side={current_position.get('side')}, amount={current_position.get('amount')}), пропускаем повторное открытие", Fore.YELLOW)
                    send_tg(f"⚠️ {sym}: позиция уже открыта, сигнал open пропущен")
                elif MAX_OPEN_POSITIONS > 0 and open_positions is not None and open_positions >= MAX_OPEN_POSITIONS:
                    log(f"⛔ Лимит открытых позиций достигнут ({open_positions}/{MAX_OPEN_POSITIONS}), пропускаем {sym}", Fore.YELLOW)
                    send_tg(f"⛔ Лимит открытых позиций достигнут ({open_positions}/{MAX_OPEN_POSITIONS}), {sym} пропущен")
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
                    risk_distance = abs(price - sl)
                    if risk_distance <= 0 or not math.isfinite(risk_distance):
                        log(f"⚠️ Невалидная дистанция до SL для {sym}, пропуск сигнала", Fore.YELLOW)
                        send_tg(f"⚠️ {sym}: не удалось вычислить расстояние до стопа")
                        continue
                    risk_budget_base = max(0.0, min(equity, available_margin))
                    risk_capital = risk_budget_base * RISK_PCT
                    if risk_capital <= 0:
                        log(f"⛔ Недостаточно доступной маржи для {sym} ({available_margin:.2f} USDT)", Fore.YELLOW)
                        send_tg(f"⛔ {sym}: недостаточно свободной маржи ({available_margin:.2f} USDT)")
                        continue
                    qty = risk_capital / risk_distance
                    if not math.isfinite(qty) or qty <= 0:
                        log(f"⚠️ Расчёт объёма дал некорректное значение для {sym}", Fore.YELLOW)
                        continue
                    notional = qty * price
                    if not math.isfinite(notional) or notional <= 0:
                        log(f"⚠️ Невозможно определить нотионал для {sym}", Fore.YELLOW)
                        continue
                    if notional < MIN_NOTIONAL_USDT:
                        qty = MIN_NOTIONAL_USDT / price
                        notional = qty * price
                    effective_margin = max(0.0, available_margin * ORDER_MARGIN_UTILIZATION)
                    max_notional = effective_margin * LEVERAGE
                    if max_notional <= 0:
                        log(f"⛔ Доступная маржа для {sym} исчерпана", Fore.YELLOW)
                        send_tg(f"⛔ {sym}: доступная маржа исчерпана")
                        continue
                    if max_notional < MIN_NOTIONAL_USDT:
                        log(f"⛔ Недостаточно маржи для минимального ордера {sym} (доступно {available_margin:.2f} USDT)", Fore.YELLOW)
                        send_tg(f"⛔ {sym}: маржа меньше минимального объёма (доступно {available_margin:.2f} USDT)")
                        continue
                    if notional > max_notional:
                        qty = max_notional / price
                        notional = max_notional
                        log(f"ℹ️ Объём {sym} уменьшен до {qty:.4f} (~{notional:.2f} USDT) из-за лимита маржи", Fore.LIGHTBLACK_EX)
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
                        ex.set_leverage(LEVERAGE, sym)
                    except Exception as e:
                        code = get_bybit_retcode(e)
                        if code == 110043:
                            log(f"ℹ️ Плечо {LEVERAGE}x уже установлено для {sym} (код {code})", Fore.LIGHTBLACK_EX)
                        else:
                            log(f"⚠️ Ошибка установки плеча: {e}", Fore.YELLOW)
                    try:
                        position_idx = get_position_idx(side)
                        params = {"takeProfit": tp, "stopLoss": sl, "tpSlMode": "Full", "reduceOnly": False}
                        if position_idx is not None:
                            params["positionIdx"] = position_idx
                        margin_required = notional / LEVERAGE if LEVERAGE else notional
                        ex.create_order(sym, "limit", side, qty, price, params)
                        log(f"✅ Ордер {sym} {side.upper()} {qty:.4f}@{price:.2f} SL:{sl:.2f} TP:{tp:.2f}", Fore.GREEN)
                        send_tg(
                            f"✅ {sym} {side.upper()} @ {price:.2f}\n"
                            f"SL {sl:.2f} TP {tp:.2f}\n"
                            f"Объём {notional:.2f} USDT, маржа {margin_required:.2f} USDT"
                        )
                        positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=PAIR_LIST)
                        current_position = positions_map.get(sym)
                    except Exception as e:
                        err_text = str(e)
                        log(f"❌ Ошибка ордера: {err_text}", Fore.RED)
                        send_tg(f"❌ Ошибка ордера для {sym}: {err_text}")
            else:
                if action not in ("hold", "manage", "none", "", None):
                    log(f"ℹ️ Неизвестное действие \"{action}\" для {sym}, обработка только дополнительных ордеров", Fore.YELLOW)

            if extra_orders:
                executed, actions_performed = execute_extra_orders(
                    ex,
                    sym,
                    extra_orders,
                    current_position=current_position,
                    open_orders=open_orders_symbol
                )
                if executed:
                    send_tg("🛠️ " + sym + " доп. ордера:\n- " + "\n- ".join(executed))
                if actions_performed:
                    positions_map, open_positions = fetch_positions_snapshot(ex, symbols_filter=PAIR_LIST)
                    current_position = positions_map.get(sym)
                    open_orders_symbol = fetch_open_orders_for_symbol(ex, sym)

        except Exception as e:
            log(f"Ошибка {sym}: {e}\n{traceback.format_exc()}", Fore.RED)

    if decisions_total>0:
        pct={k:(v/decisions_total)*100 for k,v in counts.items()}
        summary=f"📈 Статистика: открыто {counts['open']} ({pct['open']:.1f}%), " \
                f"закрыто {counts['close']} ({pct['close']:.1f}%), " \
                f"пропуск {counts['skip']} ({pct['skip']:.1f}%) — всего {decisions_total}"
        log(summary, Fore.CYAN)
        send_tg(summary)
    send_tg("✅ Цикл завершён.")
    send_tg(f"ℹ️ Версия {BOT_VERSION}. {BOT_CHANGELOG}")

if __name__ == "__main__":
    main()