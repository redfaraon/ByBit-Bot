from __future__ import annotations
MODULE_VERSION = "1.3.10"


from typing import Any

from strategy import StrategyEvent, StrategyContext, EVENTS_SPEC


def _limit_price(event: StrategyEvent, ctx: StrategyContext) -> float:
    price = ctx.price or 0.0
    offsets = EVENTS_SPEC.get("limit_offsets", {"buy": 0.998, "sell": 1.002})
    if event.side == "buy":
        return round(price * float(offsets.get("buy", 0.998)), 6)
    if event.side == "sell":
        return round(price * float(offsets.get("sell", 1.002)), 6)
    return price


def apply_event(event: StrategyEvent, ctx: StrategyContext) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "symbol": ctx.symbol,
        "reason": event.reason,
        "strategy_event": event.name,
        "confidence": event.confidence,
    }
    ladder_orders = event.metadata.get("ladder_orders") if isinstance(event.metadata, dict) else None
    if event.name in {"open_market", "open_limit"}:
        decision["action"] = "open"
        decision["side"] = event.side
        decision["notional_pct"] = event.size_pct
        if event.name == "open_market":
            decision["order_type"] = "market"
        else:
            decision["order_type"] = "limit"
            if ladder_orders:
                orders: list[dict[str, Any]] = []
                for share, offset in ladder_orders:
                    try:
                        share_f = float(share)
                        offset_f = float(offset)
                    except (TypeError, ValueError):
                        continue
                    price = ctx.price
                    if price and ctx.tf30 and ctx.tf30.atr and ctx.tf30.atr > 0:
                        price = ctx.price - offset_f * ctx.tf30.atr if event.side == "buy" else ctx.price + offset_f * ctx.tf30.atr
                    if price is None:
                        continue
                    orders.append({"type": "limit", "side": event.side, "price": round(price, 6), "params": {"reduceOnly": False}, "share": share_f})
                decision["orders"] = orders or None
            else:
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
        decision["notional_pct"] = event.size_pct
        decision["reason"] = event.reason
    elif event.name == "close_position":
        decision["action"] = "close"
        decision["side"] = event.side
        decision["order_type"] = "market"
    elif event.name == "place_limit_TP":
        decision["action"] = "manage"
        tp_price = event.metadata.get("tp_price") or _limit_price(event, ctx)
        tp_ladder = event.metadata.get("tp_ladder") if isinstance(event.metadata, dict) else None
        if tp_ladder and ctx.tf30 and ctx.tf30.atr:
            orders = []
            for share, mult in tp_ladder:
                try:
                    share_f = float(share)
                    mult_f = float(mult)
                except (TypeError, ValueError):
                    continue
                price = tp_price
                if ctx.price and ctx.tf30.atr:
                    price = ctx.price + mult_f * ctx.tf30.atr if event.side == "sell" else ctx.price - mult_f * ctx.tf30.atr
                orders.append({"type": "limit", "side": event.side, "price": price, "params": {"reduceOnly": True}, "share": share_f})
            decision["orders"] = orders or None
        else:
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
        new_price = event.metadata.get("new_price") or ctx.price
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
            decision["notional_pct"] = event.metadata.get("size_pct", event.size_pct)
    else:
        decision["action"] = "skip"
    return decision
