from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


def _default_spec() -> dict[str, Any]:
    return {
        "version": "1.1.1",
        "context": {
            "universe": [
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "XRP/USDT",
                "DOGE/USDT",
                "TON/USDT",
                "ADA/USDT",
                "AVAX/USDT",
            ],
            "timeframes": {
                "primary": "30m",
                "secondary": "4h",
                "open_interest": "1h",
                "funding": "8h",
            },
            "news": {"positive": 0.55, "negative": -0.55, "neutral_band": 0.15},
        },
        "thresholds": {
            "atr_sigma_hot": 2.5,
            "atr_limit_multiplier": 1.7,
            "atr_range_ratio": 0.008,
            "atr_extreme_ratio": 0.025,
            "oi_change_pct": 0.012,
        },
        "sizing": {
            "risk_multiplier": {"trend": 1.15, "counter": 0.55, "flat": 0.4},
            "min_pct": 0.0025,
            "max_pct": 0.05,
        },
        "rules": {
            "trend": {
                "long": {
                    "rsi_max": 65,
                    "funding_min": -0.0002,
                    "news_block": ["negative"],
                    "require_oi_up": True,
                    "confidence": {"market": 0.82, "limit": 0.78},
                },
                "short": {
                    "rsi_min": 35,
                    "funding_max": 0.0002,
                    "news_block": ["positive"],
                    "require_oi_up": True,
                    "confidence": {"market": 0.82, "limit": 0.78},
                },
            },
            "countertrend": {
                "long": {"rsi_max": 30, "news_block": ["negative"], "confidence": 0.72},
                "short": {"rsi_min": 70, "news_block": ["positive"], "confidence": 0.72},
            },
            "flat": {"rsi_band": [45, 55]},
        },
        "events": {
            "limit_gap_pct": 0.002,
            "limit_offsets": {"buy": 0.998, "sell": 1.002},
            "tp": {"atr_multiple": 2.0, "rsi_long": 70, "rsi_short": 30},
            "hedge": {"funding_flip": 0.0001, "size_pct": 0.5},
            "modify_position": {
                "rsi_long": [40, 65],
                "rsi_short": [35, 60],
                "confidence": 0.58,
                "scale": 0.5,
            },
        },
        "events": {
            "entry_ladder": [(0.6, 0.0), (0.4, 0.6)],
            "tp_ladder": [(0.33, 1.2), (0.33, 2.0), (0.34, 3.0)],
        },
    }


def _load_spec() -> dict[str, Any]:
    default_spec = _default_spec()
    spec_path = Path(__file__).with_name("strategy_spec.json")
    last_good_path = spec_path.with_suffix(".last_good.json")
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            try:
                last_good_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            return payload
    except Exception:
        if last_good_path.exists():
            try:
                payload = json.loads(last_good_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
    return default_spec


SPEC = _load_spec()
CONTEXT_SPEC = SPEC.get("context", {})
THRESHOLDS = SPEC.get("thresholds", {})
RULES_SPEC = SPEC.get("rules", {})
SIZE_SPEC = SPEC.get("sizing", {})
EVENTS_SPEC = SPEC.get("events", {})
ENTRY_LADDER = EVENTS_SPEC.get("entry_ladder") or []
TP_LADDER = EVENTS_SPEC.get("tp_ladder") or []

WATCHLIST_BASE = [sym.upper() for sym in CONTEXT_SPEC.get("universe", [])] or [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "TON/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]
WATCHLIST = WATCHLIST_BASE[:]
WATCHLIST_SOURCE = "strategy_spec.json"

INDICATORS = [
    "ema20",
    "ema50",
    "rsi14",
    "atr14",
    "funding_rate",
    "open_interest",
]

NEWS_THRESHOLDS = CONTEXT_SPEC.get("news", {})
NEWS_POSITIVE = float(NEWS_THRESHOLDS.get("positive", 0.55))
NEWS_NEGATIVE = float(NEWS_THRESHOLDS.get("negative", -0.55))
NEWS_NEUTRAL_BAND = float(NEWS_THRESHOLDS.get("neutral_band", 0.15))
SCHEDULE_SPEC = CONTEXT_SPEC.get("schedule", {}) if isinstance(CONTEXT_SPEC.get("schedule", {}), dict) else {}

ATR_SIGMA_HOT = float(THRESHOLDS.get("atr_sigma_hot", 2.5))
ATR_LIMIT_MULT = float(THRESHOLDS.get("atr_limit_multiplier", 1.7))
ATR_RANGE_RATIO = float(THRESHOLDS.get("atr_range_ratio", 0.008))
ATR_EXTREME_RATIO = float(THRESHOLDS.get("atr_extreme_ratio", 0.025))
OI_CHANGE_THRESHOLD = float(THRESHOLDS.get("oi_change_pct", 0.012))

SIZING_MULTIPLIERS = SIZE_SPEC.get(
    "risk_multiplier",
    {"trend": 1.15, "counter": 0.55, "flat": 0.4},
)
SIZE_MIN = float(SIZE_SPEC.get("min_pct", 0.0025))
SIZE_MAX = float(SIZE_SPEC.get("max_pct", 0.05))


def _normalize_side(side: str | None) -> str | None:
    if not side:
        return None
    side_lower = str(side).strip().lower()
    if side_lower in {"buy", "long"}:
        return "long"
    if side_lower in {"sell", "short"}:
        return "short"
    return None


def _is_reduce_only(order: dict[str, Any]) -> bool:
    flag = order.get("reduceOnly")
    if flag is None and isinstance(order.get("info"), dict):
        info_flag = order["info"].get("reduceOnly")
    else:
        info_flag = None
    flag = flag if flag is not None else info_flag
    if flag is None:
        return False
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)):
        return bool(flag)
    return str(flag).strip().lower() in {"1", "true", "yes", "on"}


def _news_bias(score: float | None) -> str:
    if score is None or not math.isfinite(score):
        return "neutral"
    if score >= NEWS_POSITIVE:
        return "positive"
    if score <= NEWS_NEGATIVE:
        return "negative"
    if abs(score) <= NEWS_NEUTRAL_BAND:
        return "neutral"
    return "uncertain"


def schedule_bounds(for_mode: str) -> tuple[float, float]:
    """
    Return (min, max) minutes for a given mode: 'offline', 'online', or 'backoff'.
    Falls back to legacy defaults if not present in SCHEDULE_SPEC.
    """
    mode = for_mode.lower().strip()
    defaults = {
        "offline": (5.0, 35.0),
        "online": (10.0, 45.0),
        "backoff": (25.0, 55.0),
    }
    spec = SCHEDULE_SPEC if isinstance(SCHEDULE_SPEC, dict) else {}
    key_min = f"{mode}_min"
    key_max = f"{mode}_max"
    try:
        min_val = float(spec.get(key_min, defaults.get(mode, (5.0, 35.0))[0]))
    except Exception:
        min_val = defaults.get(mode, (5.0, 35.0))[0]
    try:
        max_val = float(spec.get(key_max, defaults.get(mode, (5.0, 35.0))[1]))
    except Exception:
        max_val = defaults.get(mode, (5.0, 35.0))[1]
    if max_val < min_val:
        max_val = min_val
    return (min_val, max_val)


def schedule_news_bias_factors() -> tuple[float, float]:
    """
    Return (boost, cut) multipliers based on news bias. boost < 1.0 shortens interval,
    cut > 1.0 lengthens interval.
    """
    spec = SCHEDULE_SPEC if isinstance(SCHEDULE_SPEC, dict) else {}
    try:
        boost = float(spec.get("news_bias_boost", 0.9))
    except Exception:
        boost = 0.9
    try:
        cut = float(spec.get("news_bias_cut", 1.15))
    except Exception:
        cut = 1.15
    return boost, cut


def _extract_oi_values(history: Sequence[Any] | None) -> list[float]:
    values: list[float] = []
    if not history:
        return values
    for item in history:
        candidate = None
        if isinstance(item, (int, float)):
            candidate = float(item)
        elif isinstance(item, dict):
            raw = (
                item.get("openInterestValue")
                or item.get("openInterestAmount")
                or item.get("value")
            )
            if raw is not None:
                try:
                    candidate = float(raw)
                except (TypeError, ValueError):
                    candidate = None
        if candidate is None:
            continue
        if math.isfinite(candidate):
            values.append(candidate)
    return values


def _open_interest_trend(values: Sequence[float]) -> str:
    if len(values) < 4:
        return "flat"
    recent = values[-4:]
    baseline = values[-8:-4] or values[:-4]
    if not baseline:
        baseline = values[:-len(recent)]
    if not baseline:
        baseline = values
    head = mean(baseline[-4:]) if len(baseline) >= 4 else mean(baseline)
    tail = mean(recent)
    if head == 0:
        head = 1e-9
    change = (tail - head) / abs(head)
    if change > OI_CHANGE_THRESHOLD:
        return "up"
    if change < -OI_CHANGE_THRESHOLD:
        return "down"
    return "flat"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


@dataclass
class IndicatorBlock:
    close: float
    ema20: float
    ema50: float
    rsi: float
    atr: float
    atr_mean: float | None = None
    atr_std: float | None = None

    @property
    def atr_sigma(self) -> float:
        if not self.atr_std or self.atr_std <= 0:
            return 0.0
        if self.atr is None or not math.isfinite(self.atr):
            return 0.0
        baseline = self.atr_mean or 0.0
        return (self.atr - baseline) / self.atr_std

    @property
    def atr_is_hot(self) -> bool:
        if not self.atr_mean or self.atr_mean <= 0:
            return False
        if not math.isfinite(self.atr):
            return False
        return self.atr > self.atr_mean * ATR_LIMIT_MULT


@dataclass
class StrategyContext:
    symbol: str
    price: float
    tf30: IndicatorBlock
    tf4h: IndicatorBlock
    news_score: float | None = None
    funding_rate: float | None = None
    open_interest_history: Sequence[Any] | None = None
    has_position: bool = False
    position_side: str | None = None
    position_size: float = 0.0
    open_orders: Sequence[dict[str, Any]] | None = None
    pending_entry_price: float | None = None
    risk_pct: float = 0.01
    timestamp: float | None = None

    def __post_init__(self) -> None:
        self.symbol = (self.symbol or "").upper()
        self.position_side = _normalize_side(self.position_side)
        self.position_size = float(self.position_size or 0.0)
        if self.open_orders is None:
            self.open_orders = []

    @property
    def news_bias(self) -> str:
        return _news_bias(self.news_score)

    @property
    def oi_values(self) -> list[float]:
        return _extract_oi_values(self.open_interest_history)

    @property
    def oi_trend(self) -> str:
        return _open_interest_trend(self.oi_values)

    @property
    def trend_bias(self) -> str | None:
        bull30 = self.tf30.ema20 and self.tf30.ema50 and self.tf30.ema20 > self.tf30.ema50
        bull4h = self.tf4h.ema20 and self.tf4h.ema50 and self.tf4h.ema20 > self.tf4h.ema50
        bear30 = self.tf30.ema20 and self.tf30.ema50 and self.tf30.ema20 < self.tf30.ema50
        bear4h = self.tf4h.ema20 and self.tf4h.ema50 and self.tf4h.ema20 < self.tf4h.ema50
        if bull30 and bull4h:
            return "long"
        if bear30 and bear4h:
            return "short"
        return None

    @property
    def is_flat(self) -> bool:
        if not self.price or not math.isfinite(self.price):
            return False
        spread = abs(self.tf30.ema20 - self.tf30.ema50)
        flat_rule = RULES_SPEC.get("flat", {})
        spread_limit = float(flat_rule.get("ema_spread_pct", 0.003))
        if spread / self.price >= spread_limit:
            return False
        lower, upper = (flat_rule.get("rsi_band") or [45, 55])[:2]
        if not (lower <= self.tf30.rsi <= upper):
            return False
        if self.tf30.atr_mean and self.tf30.atr > self.tf30.atr_mean:
            return False
        return True

    @property
    def countertrend_bias(self) -> str | None:
        counter_rules = RULES_SPEC.get("countertrend", {})
        long_rule = counter_rules.get("long", {})
        short_rule = counter_rules.get("short", {})
        if self.tf30.rsi <= float(long_rule.get("rsi_max", 30)):
            return "long"
        if self.tf30.rsi >= float(short_rule.get("rsi_min", 70)):
            return "short"
        return None

    @property
    def atr_sigma(self) -> float:
        return self.tf30.atr_sigma

    @property
    def entry_orders(self) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        for order in self.open_orders or []:
            if not isinstance(order, dict):
                continue
            if _is_reduce_only(order):
                continue
            orders.append(order)
        return orders

    @property
    def protection_orders(self) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        for order in self.open_orders or []:
            if not isinstance(order, dict):
                continue
            if _is_reduce_only(order):
                orders.append(order)
        return orders

    @property
    def entry_price(self) -> float | None:
        if self.pending_entry_price:
            return self.pending_entry_price
        if not self.entry_orders:
            return None
        price = _safe_float(self.entry_orders[0].get("price"))
        return price

    @property
    def price_gap_to_entry(self) -> float | None:
        entry = self.entry_price
        if entry is None or not self.price:
            return None
        return abs(self.price - entry) / self.price


@dataclass
class StrategyEvent:
    name: str
    side: str | None = None
    order_type: str = "market"
    reason: str = ""
    size_pct: float = 0.01
    confidence: float = 0.75
    metadata: dict[str, Any] = field(default_factory=dict)


def _with_trace(metadata: dict[str, Any] | None, trace: Sequence[str] | None) -> dict[str, Any]:
    meta = dict(metadata or {})
    if trace:
        meta["trace"] = list(trace)
    return meta


def _size_for_regime(ctx: StrategyContext, regime: str, *, scale: float = 1.0) -> float:
    base = ctx.risk_pct or 0.01
    multiplier = float(SIZING_MULTIPLIERS.get(regime, 1.0))
    base *= multiplier
    base *= scale
    base = max(SIZE_MIN, min(SIZE_MAX, base))
    return round(base, 6)


def should_open(ctx: StrategyContext) -> StrategyEvent | None:
    if not _symbol_allowed(ctx.symbol):
        return StrategyEvent(
            "skip",
            reason="symbol not monitored",
            confidence=0.0,
            metadata=_with_trace(None, ["skip", "symbol_not_monitored"]),
        )
    if ctx.news_bias == "uncertain":
        return StrategyEvent(
            "skip",
            reason="news uncertain, skip entries",
            confidence=0.0,
            metadata=_with_trace(None, ["skip", "news_uncertain"]),
        )
    if ctx.atr_sigma > ATR_SIGMA_HOT:
        return StrategyEvent(
            "skip",
            reason="atr spike, unsafe to open",
            confidence=0.0,
            metadata=_with_trace(None, ["skip", "atr_sigma_hot"]),
        )

    oi_up = ctx.oi_trend == "up"
    funding = ctx.funding_rate or 0.0
    news = ctx.news_bias
    trend = ctx.trend_bias
    base_reason: list[str] = []
    trend_rules = RULES_SPEC.get("trend", {})
    long_rule = trend_rules.get("long", {})
    short_rule = trend_rules.get("short", {})

    if trend == "long":
        confidence_map = long_rule.get("confidence", {})
        cond = (
            ctx.tf30.rsi <= float(long_rule.get("rsi_max", 65))
            and news not in set(long_rule.get("news_block", []))
            and funding >= float(long_rule.get("funding_min", -0.0002))
            and (not long_rule.get("require_oi_up", True) or oi_up)
        )
        if cond:
            order_type = "limit" if (ctx.tf30.atr_is_hot or ctx.is_flat) else "market"
            if ctx.is_flat:
                order_type = "limit"
            base_reason.append(f"trend long ema20>ema50 on 30m/4h (rsi={ctx.tf30.rsi:.1f})")
            if news == "positive":
                base_reason.append("news positive bias")
            if ctx.is_flat:
                base_reason.append("range regime -> limit only")
            reason = "; ".join(base_reason) or "trend long confluence"
            metadata: dict[str, Any] = {"regime": "trend"}
            if order_type == "limit" and ENTRY_LADDER:
                metadata["ladder_orders"] = ENTRY_LADDER
            metadata = _with_trace(
                metadata,
                [
                    "open",
                    "trend.long",
                    f"order={order_type}",
                    f"news={news}",
                    f"oi={ctx.oi_trend}",
                ],
            )
            return StrategyEvent(
                f"open_{order_type}",
                side="buy",
                order_type=order_type,
                reason=reason,
                size_pct=_size_for_regime(ctx, "trend"),
                confidence=float(confidence_map.get("market" if order_type == "market" else "limit", 0.8)),
                metadata=metadata,
            )
    elif trend == "short":
        confidence_map = short_rule.get("confidence", {})
        cond = (
            ctx.tf30.rsi >= float(short_rule.get("rsi_min", 35))
            and news not in set(short_rule.get("news_block", []))
            and funding <= float(short_rule.get("funding_max", 0.0002))
            and (not short_rule.get("require_oi_up", True) or oi_up)
        )
        if cond:
            order_type = "limit" if (ctx.tf30.atr_is_hot or ctx.is_flat) else "market"
            if ctx.is_flat:
                order_type = "limit"
            base_reason.append(f"trend short ema20<ema50 (rsi={ctx.tf30.rsi:.1f})")
            if news == "negative":
                base_reason.append("news negative bias")
            if ctx.is_flat:
                base_reason.append("range regime -> limit only")
            reason = "; ".join(base_reason) or "trend short confluence"
            metadata: dict[str, Any] = {"regime": "trend"}
            if order_type == "limit" and ENTRY_LADDER:
                metadata["ladder_orders"] = ENTRY_LADDER
            metadata = _with_trace(
                metadata,
                [
                    "open",
                    "trend.short",
                    f"order={order_type}",
                    f"news={news}",
                    f"oi={ctx.oi_trend}",
                ],
            )
            return StrategyEvent(
                f"open_{order_type}",
                side="sell",
                order_type=order_type,
                reason=reason,
                size_pct=_size_for_regime(ctx, "trend"),
                confidence=float(confidence_map.get("market" if order_type == "market" else "limit", 0.8)),
                metadata=metadata,
            )

    # Countertrend entries
    counter = ctx.countertrend_bias
    if not counter:
        return None
    atr_calm = (ctx.tf30.atr_mean and ctx.tf30.atr <= ctx.tf30.atr_mean) or False
    oi_flat = ctx.oi_trend != "up"
    counter_rules = RULES_SPEC.get("countertrend", {})
    long_rule_ct = counter_rules.get("long", {})
    short_rule_ct = counter_rules.get("short", {})
    if (
        counter == "long"
        and news not in set(long_rule_ct.get("news_block", []))
        and atr_calm
        and oi_flat
    ):
        reason = "countertrend long: RSI extreme, ATR cooling, OI not rising"
        meta: dict[str, Any] = {"regime": "counter"}
        if ENTRY_LADDER:
            meta["ladder_orders"] = ENTRY_LADDER
        meta = _with_trace(
            meta,
            [
                "open",
                "countertrend.long",
                f"news={news}",
                f"oi={ctx.oi_trend}",
            ],
        )
        return StrategyEvent(
            "open_limit",
            side="buy",
            order_type="limit",
            reason=reason,
            size_pct=_size_for_regime(ctx, "counter"),
            confidence=float(long_rule_ct.get("confidence", 0.72)),
            metadata=meta,
        )
    if (
        counter == "short"
        and news not in set(short_rule_ct.get("news_block", []))
        and atr_calm
        and oi_flat
    ):
        reason = "countertrend short: RSI extreme, ATR cooling, OI not rising"
        meta: dict[str, Any] = {"regime": "counter"}
        if ENTRY_LADDER:
            meta["ladder_orders"] = ENTRY_LADDER
        meta = _with_trace(
            meta,
            [
                "open",
                "countertrend.short",
                f"news={news}",
                f"oi={ctx.oi_trend}",
            ],
        )
        return StrategyEvent(
            "open_limit",
            side="sell",
            order_type="limit",
            reason=reason,
            size_pct=_size_for_regime(ctx, "counter"),
            confidence=float(short_rule_ct.get("confidence", 0.72)),
            metadata=meta,
        )
    return None


def should_close(ctx: StrategyContext) -> StrategyEvent | None:
    if not ctx.has_position:
        return None
    side = ctx.position_side
    if side not in {"long", "short"}:
        return None
    ema_cross = False
    if side == "long":
        ema_cross = ctx.tf30.ema20 < ctx.tf30.ema50
    else:
        ema_cross = ctx.tf30.ema20 > ctx.tf30.ema50
    tp_spec = EVENTS_SPEC.get("tp", {}) if isinstance(EVENTS_SPEC, dict) else {}
    rsi_long_tp = float(tp_spec.get("rsi_long", 70))
    rsi_short_tp = float(tp_spec.get("rsi_short", 30))
    rsi_extreme = (side == "long" and ctx.tf30.rsi >= rsi_long_tp) or (side == "short" and ctx.tf30.rsi <= rsi_short_tp)
    news_against = (side == "long" and ctx.news_bias == "negative") or (side == "short" and ctx.news_bias == "positive")
    close_spec = EVENTS_SPEC.get("close", {}) if isinstance(EVENTS_SPEC, dict) else {}
    oi_reverse_enabled = bool(close_spec.get("oi_reverse", True))
    oi_flip = oi_reverse_enabled and ctx.oi_trend == ("down" if side == "long" else "up")
    oi_confirm = close_spec.get("oi_reverse_confirm") if isinstance(close_spec, dict) else {}
    require_price_weak = bool(oi_confirm.get("price_weak")) if isinstance(oi_confirm, dict) else False
    require_atr_hot = bool(oi_confirm.get("atr_hot")) if isinstance(oi_confirm, dict) else False
    if oi_flip:
        if require_price_weak:
            if side == "long" and ctx.trend_bias not in {"short", "range"}:
                oi_flip = False
            elif side == "short" and ctx.trend_bias not in {"long", "range"}:
                oi_flip = False
        if oi_flip and require_atr_hot and not ctx.tf30.atr_is_hot:
            oi_flip = False
    if ema_cross or (rsi_extreme and news_against) or oi_flip:
        close_side = "sell" if side == "long" else "buy"
        reasons: list[str] = []
        trace: list[str] = ["close_position"]
        if ema_cross:
            reasons.append("ema20 cross against position")
            trace.append("ema_cross")
        if rsi_extreme and news_against:
            reasons.append("rsi extreme + adverse news")
            trace.append("rsi_extreme_news")
        if oi_flip:
            reasons.append("open interest reversed")
            trace.append("oi_flip")
        reason = "; ".join(reasons) or "exit signal"
        return StrategyEvent(
            "close_position",
            side=close_side,
            order_type="market",
            reason=reason,
            size_pct=1.0,
            confidence=float(close_spec.get("confidence", 0.84)),
            metadata=_with_trace(None, trace),
        )
    return None


def _should_hedge(ctx: StrategyContext) -> StrategyEvent | None:
    if not ctx.has_position:
        return None
    side = ctx.position_side
    if side not in {"long", "short"}:
        return None
    news_against = (side == "long" and ctx.news_bias == "negative") or (side == "short" and ctx.news_bias == "positive")
    funding = ctx.funding_rate or 0.0
    hedge_spec = EVENTS_SPEC.get("hedge", {})
    funding_threshold = float(hedge_spec.get("funding_flip", 0.0001))
    funding_flip = (side == "long" and funding < -funding_threshold) or (side == "short" and funding > funding_threshold)
    oi_drop = ctx.oi_trend == "down"
    if not (news_against or funding_flip or oi_drop):
        return None
    hedge_side = "sell" if side == "long" else "buy"
    reason_bits = []
    trace: list[str] = ["hedge_open"]
    if news_against:
        reason_bits.append("adverse news")
        trace.append("news_against")
    if funding_flip:
        reason_bits.append("funding flipped")
        trace.append("funding_flip")
    if oi_drop:
        reason_bits.append("oi drop")
        trace.append("oi_drop")
    reason = "hedge trigger: " + ", ".join(reason_bits)
    return StrategyEvent(
        "hedge_open",
        side=hedge_side,
        order_type="market",
        reason=reason,
        size_pct=min(float(hedge_spec.get("size_pct", 0.5)), _size_for_regime(ctx, "counter")),
        confidence=0.6,
        metadata=_with_trace(None, trace),
    )


def should_modify(ctx: StrategyContext) -> StrategyEvent | None:
    # Pending limits management when flat
    entries = ctx.entry_orders
    limit_gap_pct = float(EVENTS_SPEC.get("limit_gap_pct", 0.002))
    limit_offsets = EVENTS_SPEC.get("limit_offsets", {"buy": 0.998, "sell": 1.002})
    if not ctx.has_position:
        if not entries:
            return None
        if ctx.news_bias == "uncertain":
            return StrategyEvent(
                "cancel_limit",
                reason="news uncertain",
                confidence=0.55,
                metadata=_with_trace(None, ["cancel_limit", "news_uncertain"]),
            )
        if ctx.trend_bias is None and ctx.countertrend_bias is None:
            return StrategyEvent(
                "cancel_limit",
                reason="trend flipped vs pending limit",
                confidence=0.55,
                metadata=_with_trace(None, ["cancel_limit", "trend_flipped"]),
            )
        gap = ctx.price_gap_to_entry
        if gap and gap >= limit_gap_pct:
            first_side = entries[0].get("side", "").lower()
            offset = float(limit_offsets.get("buy" if first_side == "buy" else "sell", 1.0))
            new_price = ctx.price * offset if offset > 0 else ctx.price
            meta = {
                "new_price": new_price,
                "order_ids": [order.get("id") for order in entries],
            }
            meta = _with_trace(meta, ["modify_limit", "price_gap"])
            return StrategyEvent(
                "modify_limit",
                reason="price drifted >0.2%, refresh limit",
                confidence=0.65,
                metadata=meta,
            )
        return None

    # Manage existing position: place TP or update protection
    side = ctx.position_side
    tp_spec = EVENTS_SPEC.get("tp", {})
    atr_multiple = float(tp_spec.get("atr_multiple", 2.0))
    if side == "long" and ctx.tf30.rsi >= float(tp_spec.get("rsi_long", 70)):
        tp_price = ctx.price + (ctx.tf30.atr * atr_multiple if math.isfinite(ctx.tf30.atr) else 0)
        metadata: dict[str, Any] = {"tp_price": tp_price}
        if TP_LADDER and math.isfinite(ctx.tf30.atr):
            metadata["tp_ladder"] = TP_LADDER
        metadata = _with_trace(metadata, ["place_limit_TP", "rsi_long"])
        return StrategyEvent(
            "place_limit_TP",
            side="sell",
            order_type="limit",
            reason="RSI>70 -> scale out / TP",
            confidence=0.62,
            metadata=metadata,
        )
    if side == "short" and ctx.tf30.rsi <= float(tp_spec.get("rsi_short", 30)):
        tp_price = ctx.price - (ctx.tf30.atr * atr_multiple if math.isfinite(ctx.tf30.atr) else 0)
        metadata = {"tp_price": tp_price}
        if TP_LADDER and math.isfinite(ctx.tf30.atr):
            metadata["tp_ladder"] = TP_LADDER
        metadata = _with_trace(metadata, ["place_limit_TP", "rsi_short"])
        return StrategyEvent(
            "place_limit_TP",
            side="buy",
            order_type="limit",
            reason="RSI<30 -> scale out / TP",
            confidence=0.62,
            metadata=metadata,
        )
    regime = ctx.trend_bias
    modify_spec = EVENTS_SPEC.get("modify_position", {})
    if regime == side:
        rsi = ctx.tf30.rsi
        conf_value = float(modify_spec.get("confidence", 0.58))
        scale = float(modify_spec.get("scale", 0.5))
        if side == "long":
            lower, upper = (modify_spec.get("rsi_long") or [40, 65])[:2]
            if lower <= rsi <= upper and ctx.oi_trend == "up":
                size = _size_for_regime(ctx, "trend", scale=scale)
                return StrategyEvent(
                    "modify_position",
                    side="buy",
                    order_type="market",
                    reason="trend strengthening, scale in long",
                    confidence=conf_value,
                    metadata=_with_trace(
                        {"direction": "increase", "size_pct": size},
                        ["modify_position", "trend_strengthening"],
                    ),
                )
        elif side == "short":
            lower, upper = (modify_spec.get("rsi_short") or [35, 60])[:2]
            if lower <= rsi <= upper and ctx.oi_trend == "up":
                size = _size_for_regime(ctx, "trend", scale=scale)
                return StrategyEvent(
                    "modify_position",
                    side="sell",
                    order_type="market",
                    reason="trend strengthening, scale in short",
                    confidence=conf_value,
                    metadata=_with_trace(
                        {"direction": "increase", "size_pct": size},
                        ["modify_position", "trend_strengthening"],
                    ),
                )
    return None


def get_signal_without_ai(ctx: StrategyContext) -> StrategyEvent:
    if ctx.price is None or not math.isfinite(ctx.price):
        return StrategyEvent(
            "skip",
            reason="no price data",
            confidence=0.0,
            metadata=_with_trace(None, ["skip", "no_price"]),
        )
    if ctx.has_position:
        event = should_close(ctx)
        if event:
            return event
        event = _should_hedge(ctx)
        if event:
            return event
        event = should_modify(ctx)
        if event:
            return event
        return StrategyEvent(
            "skip",
            reason="hold position",
            confidence=0.45,
            metadata=_with_trace(None, ["skip", "hold_position"]),
        )
    if not _symbol_allowed(ctx.symbol):
        return StrategyEvent(
            "skip",
            reason="symbol not monitored",
            confidence=0.0,
            metadata=_with_trace(None, ["skip", "symbol_not_monitored"]),
        )
    event = should_open(ctx)
    if event:
        return event
    event = should_modify(ctx)
    if event:
        return event
    return StrategyEvent(
        "skip",
        reason="no confluence",
        confidence=0.0,
        metadata=_with_trace(None, ["skip", "no_confluence"]),
    )
def _symbol_allowed(symbol: str) -> bool:
    if not symbol:
        return False
    sym_upper = symbol.upper()
    if sym_upper in WATCHLIST:
        return True
    base = sym_upper.split(":")[0]
    return base in WATCHLIST


def is_symbol_monitored(symbol: str) -> bool:
    return _symbol_allowed(symbol)


def set_runtime_watchlist(symbols: Sequence[str] | None, *, source: str = "runtime") -> None:
    """
    Override the watchlist for this process runtime (e.g. per-cycle universe).
    Symbols are normalized to upper-case; for derivatives forms like 'BTC/USDT:USDT'
    we also include the base 'BTC/USDT' so checks work consistently.
    """
    global WATCHLIST, WATCHLIST_SOURCE
    if not symbols:
        WATCHLIST = WATCHLIST_BASE[:]
        WATCHLIST_SOURCE = "strategy_spec.json"
        return
    seen: set[str] = set()
    updated: list[str] = []
    for item in symbols:
        raw = str(item).strip().upper()
        if not raw:
            continue
        candidates = [raw]
        if ":" in raw:
            candidates.append(raw.split(":", 1)[0])
        for sym in candidates:
            if sym and sym not in seen:
                updated.append(sym)
                seen.add(sym)
    WATCHLIST = updated if updated else WATCHLIST_BASE[:]
    WATCHLIST_SOURCE = source or "runtime"
