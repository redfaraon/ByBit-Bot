#!/usr/bin/env python3
"""
EngineCore: standalone orchestration of the research → universe → trade plan flow.

The goal is to keep this module independent from bybitbot_impl so the CLI engine
and future services can reuse the same logic without importing the monolith.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
from openai import OpenAI


DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "BNB/USDT:USDT",
    "LTC/USDT:USDT",
]
DEFAULT_INDICATORS = [
    {"indicator": "ema", "length": 20},
    {"indicator": "ema", "length": 50},
    {"indicator": "ema", "length": 200},
    {"indicator": "rsi", "length": 14},
    {"indicator": "macd"},
    {"indicator": "atr", "length": 14},
    "volume",
]
SUPPORTED_ACTIONS = {
    "open_position",
    "close_position",
    "increase_position",
    "decrease_position",
    "reduce_position",
    "exit",
}
BASE_INDICATOR_MAX = 12
UNIVERSE_MAX_PAIRS = int(os.getenv("ENGINE_MAX_PAIRS", "8"))
UNIVERSE_MAX_TIMEFRAMES = int(os.getenv("ENGINE_MAX_TF_PER_PAIR", "3"))
UNIVERSE_DEFAULT_DEPTH = int(os.getenv("ENGINE_DEFAULT_DEPTH", "180"))
OHLCV_HARD_CAP = int(os.getenv("ENGINE_MAX_OHLCV_POINTS", "400"))
RECENT_BAR_SLICE = int(os.getenv("ENGINE_RECENT_BAR_COUNT", "60"))
NEWS_SYMBOL_LIMIT = int(os.getenv("ENGINE_NEWS_SYMBOL_LIMIT", "5"))
NEWS_ITEMS_LIMIT = int(os.getenv("ENGINE_NEWS_ITEMS", "3"))


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _estimate_tokens(payload: Any) -> int:
    """
    Rough token estimator (chars/4) to stay below OpenAI hard limits without pulling tiktoken.
    """
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False)
    return max(1, len(text) // 4)


def _sma(values: Sequence[float], length: int) -> float | None:
    if not values or length <= 0 or len(values) < length:
        return None
    window = values[-length:]
    return sum(window) / float(length)


def _ema(values: Sequence[float], length: int) -> float | None:
    if not values or length <= 0 or len(values) < length:
        return None
    seed = _sma(values[:length], length)
    if seed is None:
        return None
    multiplier = 2 / (length + 1)
    ema_val = seed
    for price in values[length:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def _rsi(values: Sequence[float], length: int = 14) -> float | None:
    if not values or length <= 0 or len(values) <= length:
        return None
    gains = []
    losses = []
    for prev, curr in zip(values[-length - 1:-1], values[-length:]):
        change = curr - prev
        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss if avg_loss else 0.0
    return 100 - (100 / (1 + rs))


def _true_range(high: float, low: float, prev_close: float | None) -> float:
    ranges = [high - low]
    if prev_close is not None:
        ranges.append(abs(high - prev_close))
        ranges.append(abs(low - prev_close))
    return max(ranges)


def _atr(values: Sequence[Sequence[float]], length: int = 14) -> float | None:
    if not values or length <= 0 or len(values) < length + 1:
        return None
    trs: list[float] = []
    prev_close = None
    for candle in values[-(length + 1):]:
        _, _, high, low, close, _ = candle
        tr = _true_range(high, low, prev_close)
        trs.append(tr)
        prev_close = close
    if len(trs) < length:
        return None
    return sum(trs[-length:]) / float(length)


def _macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float] | None:
    if len(values) < slow + signal:
        return None
    fast_ema = _ema(values, fast)
    slow_ema = _ema(values, slow)
    if fast_ema is None or slow_ema is None:
        return None
    macd_line = fast_ema - slow_ema
    signal_line = _ema(values[-(slow + signal):], signal)
    if signal_line is None:
        return None
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def _realized_volatility(values: Sequence[float]) -> float | None:
    if not values or len(values) < 3:
        return None
    closes = list(values[-120:])
    returns: list[float] = []
    for prev, curr in zip(closes, closes[1:]):
        if prev == 0:
            continue
        returns.append((curr - prev) / prev)
    if not returns:
        return None
    stdev = statistics.pstdev(returns)
    return stdev * math.sqrt(max(1, len(returns)))


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    length: int | None = None

    @property
    def alias(self) -> str:
        if self.length:
            return f"{self.name}{self.length}"
        return self.name


@dataclass
class EngineSettings:
    universe_model: str = os.getenv("ENGINE_UNIVERSE_MODEL", os.getenv("OPENAI_UNIVERSE_MODEL", "gpt-4o-mini"))
    trade_model: str = os.getenv("ENGINE_TRADE_MODEL", os.getenv("OPENAI_TRADE_MODEL", "gpt-4o-mini"))
    max_universe_tokens: int = int(os.getenv("ENGINE_MAX_UNIVERSE_TOKENS", "200000"))
    cost_cap_per_million: float = float(os.getenv("ENGINE_COST_CAP_PER_MILLION", "0.3"))
    default_symbols: list[str] = field(default_factory=lambda: _parse_symbol_list(os.getenv("PAIR_LIST")))
    news_provider: str = os.getenv("ENGINE_NEWS_PROVIDER", "cryptocompare")
    universe_cache_file: Path = Path("runtime/engine/universe_cache.json")
    selection_snapshot: Path = Path("runtime/engine/selection.json")
    trade_plan_snapshot: Path = Path("runtime/engine/trade_plan.json")
    bundle_snapshot: Path = Path("runtime/engine/bundle.json")

    def ensure_runtime_dirs(self) -> None:
        self.selection_snapshot.parent.mkdir(parents=True, exist_ok=True)


def _parse_symbol_list(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SYMBOLS)
    items = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            items.append(token)
    return items or list(DEFAULT_SYMBOLS)


class EngineCore:
    def __init__(
        self,
        exchange,
        *,
        settings: EngineSettings | None = None,
        runtime_dir: Path | str | None = None,
    ) -> None:
        self.exchange = exchange
        self.settings = settings or EngineSettings()
        if runtime_dir:
            runtime_dir = Path(runtime_dir)
            self.settings.selection_snapshot = Path(runtime_dir) / "engine" / "selection.json"
            self.settings.trade_plan_snapshot = Path(runtime_dir) / "engine" / "trade_plan.json"
            self.settings.universe_cache_file = Path(runtime_dir) / "engine" / "universe_cache.json"
        self.settings.ensure_runtime_dirs()
        self._ai_key = os.getenv("OPENAI_API_KEY") or os.getenv("BYBIT_OPENAI_KEY")
        self._ai_client: OpenAI | None = None
        if self._ai_key:
            self._ai_client = OpenAI(api_key=self._ai_key, timeout=30)
        self.universe_cache = self._load_universe_cache()

    # --- public API -----------------------------------------------------
    def build_trade_plan(
        self,
        *,
        symbol_candidates: Sequence[str] | None = None,
        positions_map: Mapping[str, Any] | None = None,
        equity: float = 0.0,
        available_margin: float = 0.0,
    ) -> dict[str, Any]:
        """
        Main orchestrator: gather news, ask universe model, build per-symbol context,
        ask per-symbol model and persist the resulting trade plan snapshot.
        """
        if not self._ai_client:
            raise RuntimeError("OPENAI_API_KEY is required for engine operation")
        symbols = self._normalize_symbols(symbol_candidates)
        positions_map = dict(positions_map or {})
        news_digest = self._build_news_digest(symbols)
        universe_payload = self._request_universe(
            symbols=symbols,
            news_digest=news_digest,
            positions_map=positions_map,
            equity=equity,
            available_margin=available_margin,
        )
        if not universe_payload.get("pairs"):
            raise RuntimeError("Universe model returned no pairs to analyze")
        bundle_symbols: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        weights: dict[str, float] = {}
        ideas_by_symbol: dict[str, Any] = {}
        for pair_entry in universe_payload["pairs"]:
            symbol = pair_entry.get("symbol")
            if not symbol:
                continue
            try:
                context = self._build_pair_context(pair_entry, news_digest.get(symbol))
            except Exception as exc:
                self._log(f"[Engine] Context build failed for {symbol}: {exc}")
                continue
            bundle_symbols.append(context)
            idea_response = self._request_trade_ideas(
                pair_entry,
                context,
                positions_map=positions_map,
                equity=equity,
                available_margin=available_margin,
                global_volatility=universe_payload.get("volatility"),
            )
            if not idea_response:
                continue
            ideas_by_symbol[symbol] = idea_response
            weight = _safe_float(idea_response.get("weight"))
            if weight is not None:
                weights[symbol] = max(0.0, min(1.0, weight))
            ideas = idea_response.get("ideas") or []
            normalized = self._ideas_to_decisions(symbol, ideas, default_weight=weight)
            if normalized:
                decisions.extend(normalized)
        snapshot = self._assemble_snapshot(
            universe_payload=universe_payload,
            news_digest=news_digest,
            bundle={"symbols": bundle_symbols, "meta": {"volatility": universe_payload.get("volatility")}},
            trade_plan={
                "generated_at": _iso_now(),
                "volatility": universe_payload.get("volatility"),
                "trade_plan": decisions,
                "weights": weights,
                "ideas": ideas_by_symbol,
            },
        )
        self._save_snapshot(snapshot)
        return snapshot

    # --- universe building ----------------------------------------------
    def _normalize_symbols(self, symbol_candidates: Sequence[str] | None) -> list[str]:
        base = list(symbol_candidates or self.settings.default_symbols or DEFAULT_SYMBOLS)
        result: list[str] = []
        seen: set[str] = set()
        for item in base:
            symbol = (item or "").strip()
            if not symbol or symbol in seen:
                continue
            result.append(symbol)
            seen.add(symbol)
        return result[: UNIVERSE_MAX_PAIRS]

    def _load_universe_cache(self) -> dict[str, Any]:
        path = self.settings.universe_cache_file
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_universe_cache(self, payload: dict[str, Any]) -> None:
        serialized = self._serialize_universe(payload)
        try:
            self.settings.universe_cache_file.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        self.universe_cache = serialized

    def _build_news_digest(self, symbols: Sequence[str]) -> dict[str, Any]:
        ext_symbols = list(symbols[:NEWS_SYMBOL_LIMIT])
        self._log(f"[Engine] Gathering news for {len(ext_symbols)} symbols: {', '.join(ext_symbols)}")
        digest: dict[str, Any] = {}
        for symbol in symbols[:NEWS_SYMBOL_LIMIT]:
            payload = self._fetch_news(symbol)
            if payload:
                digest[symbol] = payload
        return digest

    def _fetch_news(self, symbol: str) -> dict[str, Any] | None:
        base = (symbol or "").split("/")[0].split(":")[0].upper()
        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params = {"lang": "EN", "sortOrder": "latest"}
        try:
            resp = requests.get(url, params=params, timeout=6)
            resp.raise_for_status()
            data = resp.json().get("Data") or []
        except Exception:
            return None
        focused: list[dict[str, Any]] = []
        general: list[dict[str, Any]] = []
        for entry in data:
            title = entry.get("title") or ""
            categories = (entry.get("categories") or "").upper()
            item = {
                "title": title,
                "url": entry.get("url"),
                "source": (entry.get("source_info") or {}).get("name") or entry.get("source"),
                "published_at": entry.get("published_on"),
            }
            if base and (base in title.upper() or base in categories):
                focused.append(item)
            else:
                general.append(item)
            if len(focused) >= NEWS_ITEMS_LIMIT and len(general) >= NEWS_ITEMS_LIMIT:
                break
        selected = focused[:NEWS_ITEMS_LIMIT] if focused else general[:NEWS_ITEMS_LIMIT]
        summary = f"{len(selected)} headlines for {base}" if base else f"{len(selected)} latest crypto stories"
        return {"asset": base, "summary": summary, "items": selected}

    def _request_universe(
        self,
        *,
        symbols: Sequence[str],
        news_digest: Mapping[str, Any],
        positions_map: Mapping[str, Any],
        equity: float,
        available_margin: float,
    ) -> dict[str, Any]:
        payload = {
            "candidate_symbols": list(symbols),
            "news": news_digest,
            "positions": positions_map,
            "equity_usdt": equity,
            "available_margin_usdt": available_margin,
            "universe_cache": self.universe_cache,
            "constraints": {
                "max_pairs": UNIVERSE_MAX_PAIRS,
                "max_timeframes_per_pair": UNIVERSE_MAX_TIMEFRAMES,
                "max_dataset_tokens": self.settings.max_universe_tokens,
                "max_cost_per_million_tokens": self.settings.cost_cap_per_million,
                "supported_indicators": DEFAULT_INDICATORS,
            },
        }
        system_msg = (
            "You are a systematic crypto portfolio architect. "
            "Using the supplied news digest and open exposure, build the trading universe for the next run. "
            "TOTAL context for the universe (including all pairs/timeframes/indicators) must stay below 200,000 tokens "
            "and the expected cost must remain under $0.3 assuming $0.30 per million tokens. "
            "Respond strictly in JSON with keys: "
            '"volatility" (global volatility label), '
            '"next_run_minutes" (float), '
            '"notes" (optional string), '
            '"global_timeframes" (list of 1-3 timeframes), '
            '"global_indicators" (list of 4-8 indicator specs), '
            '"pairs" (list of objects with keys: symbol, market, volatility, weight_hint, '
            'timeframes=[{tf, depth, indicators}], trade_model). '
            "Timeframe depth must be <= 400 bars. Indicators must come from supported list."
        )
        self._log(
            f"[Engine] Sending universe request to {self.settings.universe_model} with {len(payload['candidate_symbols'])} symbols"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        tokens = _estimate_tokens(messages)
        if tokens > self.settings.max_universe_tokens:
            raise RuntimeError(f"Universe payload exceeds {self.settings.max_universe_tokens} token heuristic")
        result = self._call_openai_json(self.settings.universe_model, messages, context_label="universe")
        universe = self._normalize_universe(result)
        self._log(f"[Engine] Universe parsed: pairs={len(universe.get('pairs', []))}, timeframes={len(universe.get('global_timeframes', []))}")
        self._save_universe_cache(universe)
        return universe

    def _normalize_universe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        universe: dict[str, Any] = {
            "volatility": payload.get("volatility") or "balanced",
            "next_run_minutes": _safe_float(payload.get("next_run_minutes")) or 15.0,
            "notes": payload.get("notes"),
        }
        global_timeframes = payload.get("global_timeframes") or []
        if not isinstance(global_timeframes, list):
            global_timeframes = [global_timeframes]
        universe["global_timeframes"] = [str(tf).strip() for tf in global_timeframes if tf]
        raw_indicators = payload.get("global_indicators") or DEFAULT_INDICATORS
        universe["global_indicators"] = self._normalize_indicator_specs(raw_indicators)
        pairs: list[dict[str, Any]] = []
        for entry in payload.get("pairs") or []:
            if not isinstance(entry, Mapping):
                continue
            symbol = str(entry.get("symbol") or "").strip()
            if not symbol:
                continue
            timeframes = []
            for tf_entry in entry.get("timeframes") or []:
                tf_name = str(tf_entry.get("tf") or tf_entry.get("timeframe") or "").strip()
                if not tf_name:
                    continue
                depth = _safe_int(tf_entry.get("depth")) or UNIVERSE_DEFAULT_DEPTH
                depth = max(50, min(OHLCV_HARD_CAP, depth))
                tf_indicators = self._normalize_indicator_specs(
                    tf_entry.get("indicators") or DEFAULT_INDICATORS
                )
                timeframes.append(
                    {
                        "tf": tf_name,
                        "depth": depth,
                        "indicators": tf_indicators[:6],
                    }
                )
                if len(timeframes) >= UNIVERSE_MAX_TIMEFRAMES:
                    break
            if not timeframes:
                continue
            pair_payload = {
                "symbol": symbol,
                "market": entry.get("market") or "crypto",
                "volatility": entry.get("volatility") or universe["volatility"],
                "weight_hint": entry.get("weight_hint"),
                "timeframes": timeframes,
                "trade_model": entry.get("trade_model") or self.settings.trade_model,
            }
            pairs.append(pair_payload)
            if len(pairs) >= UNIVERSE_MAX_PAIRS:
                break
        universe["pairs"] = pairs
        return universe

    # --- per pair ------------------------------------------------------
    def _build_pair_context(self, pair_entry: Mapping[str, Any], news_payload: Mapping[str, Any] | None) -> dict[str, Any]:
        symbol = pair_entry["symbol"]
        context: dict[str, Any] = {
            "symbol": symbol,
            "market": pair_entry.get("market") or "crypto",
            "volatility": pair_entry.get("volatility"),
            "timeframes": [],
            "news": news_payload or {},
        }
        for tf_entry in pair_entry.get("timeframes") or []:
            tf_name = tf_entry.get("tf")
            depth = tf_entry.get("depth") or UNIVERSE_DEFAULT_DEPTH
            try:
                candles = self._fetch_ohlcv(symbol, tf_name, depth)
            except Exception as exc:
                self._log(f"[Engine] fetch_ohlcv failed for {symbol} {tf_name}: {exc}")
                continue
            if not candles:
                continue
            indicators = self._calc_indicators(candles, tf_entry.get("indicators"))
            timeframe_context = {
                "tf": tf_name,
                "depth": len(candles),
                "recent_bars": self._compress_bars(candles),
                "indicators": indicators,
            }
            atr_value = next((item["value"] for item in indicators if item["name"].startswith("atr")), None)
            if atr_value is not None:
                timeframe_context["atr"] = atr_value
            vol_value = _realized_volatility([bar[4] for bar in candles])
            if vol_value is not None:
                timeframe_context["volatility"] = vol_value
            context["timeframes"].append(timeframe_context)
        return context

    def _fetch_ohlcv(self, symbol: str, timeframe: str, depth: int) -> list[list[float]]:
        limit = max(50, min(OHLCV_HARD_CAP, int(depth)))
        candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return candles or []

    def _compress_bars(self, candles: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
        bars = []
        for ts, _, high, low, close, volume in candles[-RECENT_BAR_SLICE:]:
            bars.append(
                {
                    "ts": ts,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        return bars

    def _calc_indicators(self, candles: Sequence[Sequence[float]], indicator_specs: Sequence[Any] | None) -> list[dict[str, Any]]:
        closes = [bar[4] for bar in candles]
        indicators = self._normalize_indicator_specs(indicator_specs or DEFAULT_INDICATORS)
        results: list[dict[str, Any]] = []
        for spec in indicators:
            value = None
            if spec.name == "ema":
                value = _ema(closes, spec.length or 20)
            elif spec.name == "sma":
                value = _sma(closes, spec.length or 20)
            elif spec.name == "rsi":
                value = _rsi(closes, spec.length or 14)
            elif spec.name == "atr":
                value = _atr(candles, spec.length or 14)
            elif spec.name == "macd":
                macd_payload = _macd(closes)
                if macd_payload:
                    value = macd_payload
            elif spec.name == "volume":
                value = candles[-1][5] if candles else None
            if value is None:
                continue
            results.append(
                {
                    "name": spec.alias,
                    "value": value,
                }
            )
        return results

    def _normalize_indicator_specs(self, specs: Sequence[Any]) -> list[IndicatorSpec]:
        normalized: list[IndicatorSpec] = []
        seen: set[tuple[str, int | None]] = set()
        for item in specs:
            if isinstance(item, Mapping):
                name = str(item.get("indicator") or item.get("name") or "").lower()
                if not name:
                    continue
                length = _safe_int(item.get("length") or item.get("period"))
            else:
                token = str(item).strip().lower()
                if token.startswith("ema"):
                    name = "ema"
                    length = _safe_int(token[3:]) or 20
                elif token.startswith("sma"):
                    name = "sma"
                    length = _safe_int(token[3:]) or 20
                elif token.startswith("rsi"):
                    name = "rsi"
                    length = _safe_int(token[3:]) or 14
                elif token.startswith("atr"):
                    name = "atr"
                    length = _safe_int(token[3:]) or 14
                elif token.startswith("macd"):
                    name = "macd"
                    length = None
                elif token.startswith("vol"):
                    name = "volume"
                    length = None
                else:
                    name = token
                    length = None
            key = (name, length)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(IndicatorSpec(name=name, length=length))
            if len(normalized) >= BASE_INDICATOR_MAX:
                break
        if not normalized:
            return [IndicatorSpec(name="ema", length=20), IndicatorSpec(name="rsi", length=14)]
        return normalized

    # --- trade plan -----------------------------------------------------
    def _request_trade_ideas(
        self,
        pair_entry: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        positions_map: Mapping[str, Any],
        equity: float,
        available_margin: float,
        global_volatility: Any,
    ) -> dict[str, Any] | None:
        symbol = pair_entry["symbol"]
        payload = {
            "pair": symbol,
            "market": pair_entry.get("market"),
            "volatility": pair_entry.get("volatility") or global_volatility,
            "timeframes": context.get("timeframes"),
            "news": context.get("news"),
            "position": positions_map.get(symbol),
            "risk": {"equity": equity, "available_margin": available_margin},
            "weight_hint": pair_entry.get("weight_hint"),
        }
        system_msg = (
            "You are a quantitative trading assistant. "
            "Read the OHLCV + indicator context and return actionable ideas. "
            "Allowed actions: open_position, close_position, increase_position, decrease_position, reduce_position. "
            "Each idea must include fields: action, direction (long/short), timeframe, "
            "order_type (market/limit), size_pct (percentage of account equity), "
            "notional_hint (USDT), entry (price), sl_atr, tp_atr, comment. "
            "Use ATR-based stop/target multiples. Do not propose actions if conviction is low."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            return self._call_openai_json(pair_entry.get("trade_model") or self.settings.trade_model, messages, context_label=symbol)
        except Exception as exc:
            self._log(f"[Engine] Trade idea request failed for {symbol}: {exc}")
            return None

    def _ideas_to_decisions(self, symbol: str, ideas: Iterable[Mapping[str, Any]], default_weight: float | None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for idea in ideas:
            action = str(idea.get("action") or "").lower()
            if action not in SUPPORTED_ACTIONS:
                continue
            direction = (idea.get("direction") or "long").lower()
            order_type = (idea.get("order_type") or "market").lower()
            size_pct = _safe_float(idea.get("size_pct") or idea.get("weight") or default_weight)
            notional_pct = None
            if size_pct is not None:
                notional_pct = size_pct if size_pct <= 1 else size_pct / 100.0
            notional_usdt = _safe_float(idea.get("notional_hint"))
            entry = _safe_float(idea.get("entry"))
            params: dict[str, Any] = {}
            exchange_action = "buy" if direction == "long" else "sell"
            if action in {"close_position", "reduce_position", "decrease_position", "exit"}:
                params["reduce_only"] = True
            results.append(
                {
                    "symbol": symbol,
                    "action": exchange_action,
                    "order_type": order_type,
                    "notional_pct": notional_pct,
                    "notional_usdt": notional_usdt,
                    "price": entry,
                    "params": params,
                    "direction": direction,
                    "raw_action": action,
                    "sl_atr": idea.get("sl_atr"),
                    "tp_atr": idea.get("tp_atr"),
                    "comment": idea.get("comment"),
                }
            )
        return results

    def _assemble_snapshot(
        self,
        *,
        universe_payload: Mapping[str, Any],
        news_digest: Mapping[str, Any],
        bundle: Mapping[str, Any],
        trade_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        serialized_universe = self._serialize_universe(universe_payload)
        selection = {
            "pairs": [entry["symbol"] for entry in serialized_universe.get("pairs", [])],
            "global_timeframes": serialized_universe.get("global_timeframes"),
            "global_indicators": self._serialize_indicator_specs(universe_payload.get("global_indicators"), alias_only=True),
            "next_run_minutes": serialized_universe.get("next_run_minutes"),
            "volatility": serialized_universe.get("volatility"),
            "notes": serialized_universe.get("notes"),
        }
        return {
            "selection": selection,
            "universe": serialized_universe,
            "news": news_digest,
            "news_requests": [],
            "bundle": bundle,
            "trade_plan": trade_plan,
        }

    def _save_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        try:
            self.settings.selection_snapshot.write_text(
                json.dumps({"selection": snapshot.get("selection"), "universe": snapshot.get("universe")}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self._log(f"[Engine] Failed to write selection snapshot: {exc}")
        try:
            self.settings.trade_plan_snapshot.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._log(f"[Engine] Failed to write trade plan snapshot: {exc}")

    # --- helpers --------------------------------------------------------
    def _call_openai_json(self, model: str, messages: Sequence[Mapping[str, Any]], *, context_label: str) -> dict[str, Any]:
        if not self._ai_client:
            raise RuntimeError("OpenAI client is not initialized")
        start = time.perf_counter()
        res = self._ai_client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=list(messages),
        )
        took = time.perf_counter() - start
        try:
            content = res.choices[0].message.content
            payload = json.loads(content)
            self._log(f"[Engine] OpenAI {context_label} ({model}) succeeded in {took:.2f}s")
            return payload
        except Exception as exc:
            raise RuntimeError(f"OpenAI {context_label} decode failed: {exc}") from exc

    def _log(self, message: str) -> None:
        print(message)

    def _serialize_indicator_specs(self, specs: Sequence[Any] | None, *, alias_only: bool = False) -> list[Any]:
        result: list[Any] = []
        for spec in specs or []:
            if isinstance(spec, IndicatorSpec):
                if alias_only:
                    result.append(spec.alias)
                else:
                    result.append({"name": spec.name, "length": spec.length})
            else:
                result.append(spec)
        return result

    def _serialize_universe(self, universe: Mapping[str, Any]) -> dict[str, Any]:
        serialized = {
            "volatility": universe.get("volatility"),
            "next_run_minutes": universe.get("next_run_minutes"),
            "notes": universe.get("notes"),
            "global_timeframes": universe.get("global_timeframes"),
            "global_indicators": self._serialize_indicator_specs(universe.get("global_indicators")),
            "pairs": [],
        }
        for entry in universe.get("pairs", []):
            tf_serialized = []
            for tf in entry.get("timeframes", []):
                tf_serialized.append(
                    {
                        "tf": tf.get("tf"),
                        "depth": tf.get("depth"),
                        "indicators": self._serialize_indicator_specs(tf.get("indicators")),
                    }
                )
            serialized["pairs"].append(
                {
                    "symbol": entry.get("symbol"),
                    "market": entry.get("market"),
                    "volatility": entry.get("volatility"),
                    "weight_hint": entry.get("weight_hint"),
                    "trade_model": entry.get("trade_model"),
                    "timeframes": tf_serialized,
                }
            )
        return serialized
