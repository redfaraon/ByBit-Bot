from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable, Sequence


WATCHLIST = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "TON/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]

INDICATORS = [
    "ema20",
    "ema50",
    "rsi14",
    "atr14",
    "funding_rate",
    "open_interest",
]


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
    if score >= 0.55:
        return "positive"
    if score <= -0.55:
        return "negative"
    if abs(score) <= 0.15:
        return "neutral"
    return "uncertain"


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
    if change > 0.012:
        return "up"
    if change < -0.012:
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
        return self.atr > self.atr_mean * 1.7


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
        if spread / self.price >= 0.003:
            return False
        if not (45 <= self.tf30.rsi <= 55):
            return False
        if self.tf30.atr_mean and self.tf30.atr > self.tf30.atr_mean:
            return False
        return True

    @property
    def countertrend_bias(self) -> str | None:
        if self.tf30.rsi <= 30:
            return "long"
        if self.tf30.rsi >= 70:
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


def _size_for_regime(ctx: StrategyContext, regime: str, *, scale: float = 1.0) -> float:
    base = ctx.risk_pct or 0.01
    if regime == "trend":
        base *= 1.15
    elif regime == "counter":
        base *= 0.55
    elif regime == "flat":
        base *= 0.4
    base *= scale
    base = max(0.0025, min(0.05, base))
    return round(base, 6)


def should_open(ctx: StrategyContext) -> StrategyEvent | None:
    if ctx.symbol not in WATCHLIST:
        return StrategyEvent("skip", reason="symbol outside manual watchlist", confidence=0.0)
    if ctx.news_bias == "uncertain":
        return StrategyEvent("skip", reason="news uncertain, skip entries", confidence=0.0)
    if ctx.atr_sigma > 2.5:
        return StrategyEvent("skip", reason="atr spike, unsafe to open", confidence=0.0)

    oi_up = ctx.oi_trend == "up"
    funding = ctx.funding_rate or 0.0
    news = ctx.news_bias
    trend = ctx.trend_bias
    base_reason: list[str] = []

    if trend == "long":
        cond = (
            ctx.tf30.rsi < 65
            and news != "negative"
            and funding >= -0.0002
            and oi_up
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
            return StrategyEvent(
                f"open_{order_type}",
                side="buy",
                order_type=order_type,
                reason=reason,
                size_pct=_size_for_regime(ctx, "trend"),
                confidence=0.82 if order_type == "market" else 0.78,
                metadata={"regime": "trend"},
            )
    elif trend == "short":
        cond = (
            ctx.tf30.rsi > 35
            and news != "positive"
            and funding <= 0.0002
            and oi_up
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
            return StrategyEvent(
                f"open_{order_type}",
                side="sell",
                order_type=order_type,
                reason=reason,
                size_pct=_size_for_regime(ctx, "trend"),
                confidence=0.82 if order_type == "market" else 0.78,
                metadata={"regime": "trend"},
            )

    # Countertrend entries
    counter = ctx.countertrend_bias
    if not counter:
        return None
    atr_calm = (ctx.tf30.atr_mean and ctx.tf30.atr <= ctx.tf30.atr_mean) or False
    oi_flat = ctx.oi_trend != "up"
    if counter == "long" and news != "negative" and atr_calm and oi_flat:
        reason = "countertrend long: RSI<30, ATR cooling, OI not rising"
        return StrategyEvent(
            "open_limit",
            side="buy",
            order_type="limit",
            reason=reason,
            size_pct=_size_for_regime(ctx, "counter"),
            confidence=0.72,
            metadata={"regime": "counter"},
        )
    if counter == "short" and news != "positive" and atr_calm and oi_flat:
        reason = "countertrend short: RSI>70, ATR cooling, OI not rising"
        return StrategyEvent(
            "open_limit",
            side="sell",
            order_type="limit",
            reason=reason,
            size_pct=_size_for_regime(ctx, "counter"),
            confidence=0.72,
            metadata={"regime": "counter"},
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
    rsi_extreme = (side == "long" and ctx.tf30.rsi >= 70) or (side == "short" and ctx.tf30.rsi <= 30)
    news_against = (side == "long" and ctx.news_bias == "negative") or (side == "short" and ctx.news_bias == "positive")
    oi_flip = ctx.oi_trend == ("down" if side == "long" else "up")
    if ema_cross or (rsi_extreme and news_against) or oi_flip:
        close_side = "sell" if side == "long" else "buy"
        reasons: list[str] = []
        if ema_cross:
            reasons.append("ema20 cross against position")
        if rsi_extreme and news_against:
            reasons.append("rsi extreme + adverse news")
        if oi_flip:
            reasons.append("open interest reversed")
        reason = "; ".join(reasons) or "exit signal"
        return StrategyEvent(
            "close_position",
            side=close_side,
            order_type="market",
            reason=reason,
            size_pct=1.0,
            confidence=0.84,
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
    funding_flip = (side == "long" and funding < -0.0001) or (side == "short" and funding > 0.0001)
    oi_drop = ctx.oi_trend == "down"
    if not (news_against or funding_flip or oi_drop):
        return None
    hedge_side = "sell" if side == "long" else "buy"
    reason_bits = []
    if news_against:
        reason_bits.append("adverse news")
    if funding_flip:
        reason_bits.append("funding flipped")
    if oi_drop:
        reason_bits.append("oi drop")
    reason = "hedge trigger: " + ", ".join(reason_bits)
    return StrategyEvent(
        "hedge_open",
        side=hedge_side,
        order_type="market",
        reason=reason,
        size_pct=min(0.5, _size_for_regime(ctx, "counter")),
        confidence=0.6,
    )


def should_modify(ctx: StrategyContext) -> StrategyEvent | None:
    # Pending limits management when flat
    entries = ctx.entry_orders
    if not ctx.has_position:
        if not entries:
            return None
        if ctx.news_bias == "uncertain":
            return StrategyEvent("cancel_limit", reason="news uncertain", confidence=0.55)
        if ctx.trend_bias is None and ctx.countertrend_bias is None:
            return StrategyEvent("cancel_limit", reason="trend flipped vs pending limit", confidence=0.55)
        gap = ctx.price_gap_to_entry
        if gap and gap >= 0.002:
            new_price = ctx.price * (0.998 if entries[0].get("side", "").lower() == "buy" else 1.002)
            meta = {
                "new_price": new_price,
                "order_ids": [order.get("id") for order in entries],
            }
            return StrategyEvent("modify_limit", reason="price drifted >0.2%, refresh limit", confidence=0.65, metadata=meta)
        return None

    # Manage existing position: place TP or update protection
    side = ctx.position_side
    if side == "long" and ctx.tf30.rsi >= 70:
        tp_price = ctx.price + (ctx.tf30.atr * 2 if math.isfinite(ctx.tf30.atr) else 0)
        return StrategyEvent(
            "place_limit_TP",
            side="sell",
            order_type="limit",
            reason="RSI>70 -> scale out / TP",
            confidence=0.62,
            metadata={"tp_price": tp_price},
        )
    if side == "short" and ctx.tf30.rsi <= 30:
        tp_price = ctx.price - (ctx.tf30.atr * 2 if math.isfinite(ctx.tf30.atr) else 0)
        return StrategyEvent(
            "place_limit_TP",
            side="buy",
            order_type="limit",
            reason="RSI<30 -> scale out / TP",
            confidence=0.62,
            metadata={"tp_price": tp_price},
        )
    regime = ctx.trend_bias
    if regime == side:
        rsi = ctx.tf30.rsi
        if side == "long" and 40 <= rsi <= 65 and ctx.oi_trend == "up":
            size = _size_for_regime(ctx, "trend", scale=0.5)
            return StrategyEvent(
                "modify_position",
                side="buy",
                order_type="market",
                reason="trend strengthening, scale in long",
                confidence=0.58,
                metadata={"direction": "increase", "size_pct": size},
            )
        if side == "short" and 35 <= rsi <= 60 and ctx.oi_trend == "up":
            size = _size_for_regime(ctx, "trend", scale=0.5)
            return StrategyEvent(
                "modify_position",
                side="sell",
                order_type="market",
                reason="trend strengthening, scale in short",
                confidence=0.58,
                metadata={"direction": "increase", "size_pct": size},
            )
    return None


def get_signal_without_ai(ctx: StrategyContext) -> StrategyEvent:
    if ctx.symbol not in WATCHLIST:
        return StrategyEvent("skip", reason="symbol not monitored", confidence=0.0)
    if ctx.price is None or not math.isfinite(ctx.price):
        return StrategyEvent("skip", reason="no price data", confidence=0.0)
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
        return StrategyEvent("skip", reason="hold position", confidence=0.45)
    event = should_open(ctx)
    if event:
        return event
    event = should_modify(ctx)
    if event:
        return event
    return StrategyEvent("skip", reason="no confluence", confidence=0.0)


def _limit_price(event: StrategyEvent, ctx: StrategyContext) -> float:
    price = ctx.price or 0.0
    if event.side == "buy":
        return round(price * 0.998, 6)
    if event.side == "sell":
        return round(price * 1.002, 6)
    return price


def apply_event(event: StrategyEvent, ctx: StrategyContext) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "symbol": ctx.symbol,
        "reason": event.reason,
        "strategy_event": event.name,
        "confidence": event.confidence,
    }
    if event.name in {"open_market", "open_limit"}:
        decision["action"] = "open"
        decision["side"] = event.side
        decision["notional_pct"] = event.size_pct
        if event.name == "open_market":
            decision["order_type"] = "market"
        else:
            decision["order_type"] = "limit"
            limit_price = event.metadata.get("limit_price") or _limit_price(event, ctx)
            decision["orders"] = [
                {
                    "type": "limit",
                    "side": event.side,
                    "price": limit_price,
                    "params": {"reduceOnly": False},
                }
            ]
    elif event.name == "hedge_open":
        decision["action"] = "open"
        decision["side"] = event.side
        decision["order_type"] = "market"
        decision["notional_pct"] = min(0.5, event.size_pct)
        decision["reason"] = event.reason
    elif event.name == "close_position":
        decision["action"] = "close"
        decision["side"] = event.side
        decision["order_type"] = "market"
    elif event.name == "place_limit_TP":
        decision["action"] = "manage"
        tp_price = event.metadata.get("tp_price") or _limit_price(event, ctx)
        decision["orders"] = [
            {
                "type": "limit",
                "side": event.side,
                "price": tp_price,
                "params": {"reduceOnly": True},
            }
        ]
    elif event.name == "cancel_limit":
        decision["action"] = "manage"
        order_ids = [order.get("id") for order in ctx.entry_orders if order.get("id")]
        if order_ids:
            decision["cancel_orders"] = order_ids
        else:
            decision["action"] = "skip"
    elif event.name == "modify_limit":
        decision["action"] = "manage"
        new_price = event.metadata.get("new_price") or _limit_price(event, ctx)
        replacements = []
        for order in ctx.entry_orders:
            oid = order.get("id")
            if not oid:
                continue
            replacements.append({"cancel": oid, "order": {"type": "limit", "side": order.get("side"), "price": new_price}})
        if replacements:
            decision["replace_orders"] = replacements
        else:
            decision["action"] = "skip"
    elif event.name == "modify_position":
        direction = event.metadata.get("direction", "increase")
        if direction == "increase":
            decision["action"] = "open"
            decision["side"] = event.side
            decision["order_type"] = "market"
            decision["notional_pct"] = event.metadata.get("size_pct", event.size_pct)
        else:
            decision["action"] = "close"
            decision["side"] = event.side
            decision["order_type"] = "market"
            decision["notional_pct"] = min(0.5, event.metadata.get("size_pct", event.size_pct))
    else:
        decision["action"] = "skip"
    return decision
