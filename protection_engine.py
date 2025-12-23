from __future__ import annotations
MODULE_VERSION = "1.3.10"


import json
import math
import numbers
from typing import Any, Optional

import pandas as pd
from colorama import Fore

import order_cleanup
import trailing_utils
from order_utils import get_position_idx, get_trigger_direction_for_side, _summarize_order_spec

_PREV_UNREALIZED_PNL: dict[str, float] = {}
_TRAIL_PROTECTION: dict[str, dict[str, float]] = {}
_CURRENT_CYCLE_NUMBER: int | None = None

SL_ATR = 0.8
TP_ATR = 1.6
TRAILING_ATR_MULT = 1.0
TRAILING_DYNAMIC_TRIGGER_ATR = 1.4
TRAILING_DYNAMIC_FACTOR = 0.65
TRAILING_DYNAMIC_MIN_ATR = 0.35
PSEUDOTRAIL_MIN_IMPROVE_ATR = 0.35
PSEUDOTRAIL_STOP_LOCK_FACTOR = 0.35
PSEUDOTRAIL_TP_EXTEND_FACTOR = 0.25
PSEUDOTRAIL_MAX_TAKE_EXTENDS = 2
PSEUDOTRAIL_MAX_TAKE_SHIFT_ATR_MULT = 0.5
PSEUDOTRAIL_POSITION_STALE_PCT = 0.03
PSEUDOTRAIL_POSITION_SIZE_STALE_RATIO = 0.6
PROTECTION_MAX_PRICE_RATIO = 10.0
PROTECTION_MIN_PRICE_RATIO = 0.05
BREAKEVEN_ENABLED = True
BREAKEVEN_ATR_MULT = 0.6
BREAKEVEN_BUFFER_ATR = 0.15
IMMEDIATE_CLOSE_ON_BREACH = False
PARTIAL_TP_SCHEME = [(0.33, 1.2), (0.33, 2.0), (0.34, 3.0)]
MIN_NOTIONAL_USDT = 5.0
TIMEFRAME = "30m"

def ensure_protection(
    exchange: Any,
    symbol: str,
    position: dict | None,
    df_primary: Any,
    open_orders: list[dict] | None,
    *,
    config: dict | None,
    handler,
    log_fn=None,
):
    """Thin wrapper to call protection logic with consistent logging."""
    if log_fn:
        log_fn(
            f"[MODULE][protection] sym={symbol} pos={'yes' if position else 'no'} open_orders={len(open_orders or [])} config_keys={list((config or {}).keys()) if isinstance(config, dict) else []}"
        )
    result = handler(exchange, symbol, position, df_primary, open_orders, config)
    if log_fn:
        result_len = len(result or []) if isinstance(result, (list, tuple)) else "n/a"
        log_fn(f"[MODULE][protection] sym={symbol} result_orders={result_len}")
    return result

REQUIRED_GLOBALS = {
    'safe_float',
    'safe_int',
    '_is_truthy_flag',
    'log',
    'send_tg',
    '_resolve_symbol_alias',
    '_infer_market_category',
    'fetch_open_orders_for_symbol',
    'atr',
    'TIMEFRAME',
    'SL_ATR',
    'TP_ATR',
    'TRAILING_ATR_MULT',
    'TRAILING_DYNAMIC_TRIGGER_ATR',
    'TRAILING_DYNAMIC_FACTOR',
    'TRAILING_DYNAMIC_MIN_ATR',
    'PSEUDOTRAIL_MIN_IMPROVE_ATR',
    'PSEUDOTRAIL_STOP_LOCK_FACTOR',
    'PSEUDOTRAIL_TP_EXTEND_FACTOR',
    'PSEUDOTRAIL_MAX_TAKE_EXTENDS',
    'PSEUDOTRAIL_MAX_TAKE_SHIFT_ATR_MULT',
    'PSEUDOTRAIL_POSITION_STALE_PCT',
    'PSEUDOTRAIL_POSITION_SIZE_STALE_RATIO',
    'PROTECTION_MAX_PRICE_RATIO',
    'PROTECTION_MIN_PRICE_RATIO',
    'BREAKEVEN_ENABLED',
    'BREAKEVEN_ATR_MULT',
    'BREAKEVEN_BUFFER_ATR',
    'IMMEDIATE_CLOSE_ON_BREACH',
    'PARTIAL_TP_SCHEME',
    'MIN_NOTIONAL_USDT',
}

def configure(bindings: dict[str, Any]) -> None:
    for name in REQUIRED_GLOBALS:
        if name in bindings:
            globals()[name] = bindings[name]


class ProtectionMissingError(RuntimeError):
    """Raised when open positions remain without mandatory protective orders."""

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

def _summarize_open_orders_for_log(orders, limit: int = 6) -> str:
    if not orders:
        return "[]"
    summary: list[dict[str, Any]] = []
    for order in orders:
        if not isinstance(order, dict):
            continue
        price = safe_float(order.get("price"))
        stop = safe_float(order.get("stopPrice") or order.get("triggerPrice"))
        qty = safe_float(order.get("amount") or order.get("qty") or order.get("size"))
        summary.append(
            {
                "id": order.get("id"),
                "side": order.get("side"),
                "type": order.get("type"),
                "price": price,
                "stop": stop,
                "qty": qty,
                "reduceOnly": order.get("reduceOnly"),
            }
        )
        if len(summary) >= limit:
            break
    suffix = ""
    try:
        if len(orders) > limit:
            suffix = f" (+{len(orders) - limit} more)"
    except Exception:
        suffix = ""
    return json.dumps(summary, ensure_ascii=True) + suffix

def _select_best_protection_levels(
    position_payload: dict[str, Any] | None,
    orders,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """
    Returns (entry_price, ref_price, qty, best_stop, best_take).

    best_stop: closest protective stop-loss on the loss side of the entry.
    best_take: closest take-profit on the profit side of the entry.
    """
    if not isinstance(position_payload, dict):
        return None, None, None, None, None
    qty = safe_float(position_payload.get("amount") or position_payload.get("contracts"))
    side_raw = str(position_payload.get("side") or "").lower()
    is_long = side_raw in {"buy", "long"} or (qty is not None and qty > 0)

    entry_price = safe_float(
        position_payload.get("entryPrice")
        or position_payload.get("avgEntryPrice")
        or position_payload.get("average")
        or position_payload.get("avgEntryPrice")
    )
    mark_price = safe_float(
        position_payload.get("markPrice")
        or position_payload.get("mark_price")
        or position_payload.get("lastPrice")
    )
    ref_price = mark_price if mark_price is not None and math.isfinite(mark_price) else entry_price

    categorized = _categorize_protection_orders(orders)
    stop_candidates = [price for price, _amt in categorized.get("stop", []) if price is not None and math.isfinite(price)]
    take_candidates = [price for price, _amt in categorized.get("take_profit", []) if price is not None and math.isfinite(price)]

    if entry_price is None or not math.isfinite(entry_price):
        return None, ref_price, qty, None, None

    best_stop = None
    if is_long:
        below = [p for p in stop_candidates if p < entry_price - 1e-9]
        if below:
            best_stop = max(below)  # closest below entry
    else:
        above = [p for p in stop_candidates if p > entry_price + 1e-9]
        if above:
            best_stop = min(above)  # closest above entry

    best_take = None
    if is_long:
        above = [p for p in take_candidates if p > entry_price + 1e-9]
        if above:
            best_take = min(above)  # closest above entry
    else:
        below = [p for p in take_candidates if p < entry_price - 1e-9]
        if below:
            best_take = max(below)  # closest below entry

    return entry_price, ref_price, qty, best_stop, best_take

def _format_progress_to_levels(
    position_payload: dict[str, Any] | None,
    orders,
) -> tuple[str | None, dict[str, float | None]]:
    """
    Returns (human_text, metrics) where metrics contain:
    - progress_target_pct: 0..inf (profit progress vs closest TP)
    - risk_used_pct: 0..inf (drawdown vs closest SL)
    - pnl_usdt: unrealized pnl if available
    """
    entry, ref_px, qty, stop_px, take_px = _select_best_protection_levels(position_payload, orders)
    if entry is None or ref_px is None or not math.isfinite(entry) or not math.isfinite(ref_px) or entry <= 0:
        return None, {"progress_target_pct": None, "risk_used_pct": None, "pnl_usdt": None}
    qty_val = qty if qty is not None and math.isfinite(qty) else None
    side_raw = str((position_payload or {}).get("side") or "").lower()
    is_long = side_raw in {"buy", "long"} or (qty_val is not None and qty_val > 0)

    pnl_unreal = safe_float((position_payload or {}).get("unrealizedPnl") or ((position_payload or {}).get("raw") or {}).get("unrealisedPnl"))

    # Directional move from entry
    move = (ref_px - entry) if is_long else (entry - ref_px)

    progress_pct = None
    if take_px is not None and math.isfinite(take_px) and take_px > 0:
        target_dist = (take_px - entry) if is_long else (entry - take_px)
        if target_dist and math.isfinite(target_dist) and target_dist > 0:
            progress_pct = max(0.0, (move / target_dist) * 100.0)

    risk_used_pct = None
    if stop_px is not None and math.isfinite(stop_px) and stop_px > 0:
        risk_dist = (entry - stop_px) if is_long else (stop_px - entry)
        if risk_dist and math.isfinite(risk_dist) and risk_dist > 0:
            drawdown = max(0.0, (-move))
            risk_used_pct = max(0.0, (drawdown / risk_dist) * 100.0)

    parts: list[str] = []
    if progress_pct is not None and math.isfinite(progress_pct):
        parts.append(f"+{progress_pct:.0f}% target")
    if risk_used_pct is not None and math.isfinite(risk_used_pct):
        parts.append(f"{risk_used_pct:.0f}% risk")
    if pnl_unreal is not None and math.isfinite(pnl_unreal):
        parts.append(f"PnL {pnl_unreal:+.2f} USDT")

    text = "; ".join(parts) if parts else None
    return text, {"progress_target_pct": progress_pct, "risk_used_pct": risk_used_pct, "pnl_usdt": pnl_unreal}

def _format_close_reason(
    initial_position_payload: dict[str, Any] | None,
    close_price: float | None,
    orders_before_close,
) -> str | None:
    entry, _ref_px, qty, stop_px, take_px = _select_best_protection_levels(initial_position_payload, orders_before_close)
    if entry is None or close_price is None or qty is None:
        return None
    if not (math.isfinite(entry) and math.isfinite(close_price) and math.isfinite(qty)):
        return None
    if abs(qty) <= 1e-12:
        return None
    side_raw = str((initial_position_payload or {}).get("side") or "").lower()
    is_long = side_raw in {"buy", "long"} or qty > 0
    pnl = (close_price - entry) * abs(qty) if is_long else (entry - close_price) * abs(qty)

    risk_usdt = None
    if stop_px is not None and math.isfinite(stop_px):
        risk_dist = (entry - stop_px) if is_long else (stop_px - entry)
        if risk_dist and math.isfinite(risk_dist) and risk_dist > 0:
            risk_usdt = risk_dist * abs(qty)
    target_usdt = None
    if take_px is not None and math.isfinite(take_px):
        target_dist = (take_px - entry) if is_long else (entry - take_px)
        if target_dist and math.isfinite(target_dist) and target_dist > 0:
            target_usdt = target_dist * abs(qty)

    if pnl >= 0:
        pct = None
        if target_usdt is not None and math.isfinite(target_usdt) and target_usdt > 0:
            pct = (pnl / target_usdt) * 100.0
        pct_text = f", {pct:.0f}% цели" if pct is not None and math.isfinite(pct) else ""
        return f"фиксация прибыли {pnl:+.2f} USDT{pct_text}"
    pct = None
    if risk_usdt is not None and math.isfinite(risk_usdt) and risk_usdt > 0:
        pct = (abs(pnl) / risk_usdt) * 100.0
    pct_text = f", {pct:.0f}% риска" if pct is not None and math.isfinite(pct) else ""
    return f"фиксация убытка {pnl:+.2f} USDT{pct_text}"

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
        cancelled_stop_entries, cancel_stop_errors = order_cleanup.cleanup_redundant_stops(
            exchange,
            symbol,
            refreshed_reduce,
            protection_side,
            position_qty,
            is_long,
            keep_ids_preferred=stop_ids_preferred,
            handler=_cleanup_redundant_stop_orders,
            log_fn=lambda msg: log(msg, Fore.LIGHTBLACK_EX),
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
    if symbol and symbol.upper().startswith("DOGE"):
        log(
            f"[MODULE][protection] {symbol}: input_orders={_summarize_open_orders_for_log(open_orders)}",
            Fore.LIGHTBLACK_EX,
        )
    updated_orders = trailing_utils.apply_trailing(
        exchange,
        symbol,
        position,
        protection_df,
        open_orders,
        config=symbol_meta,
        handler=ensure_position_protection,
        log_fn=lambda msg: log(msg, Fore.LIGHTBLACK_EX),
    )
    if symbol and symbol.upper().startswith("DOGE"):
        log(
            f"[MODULE][trailing] {symbol}: output_orders={_summarize_open_orders_for_log(updated_orders)}",
            Fore.LIGHTBLACK_EX,
        )
    has_stop = False
    has_take = False
    if updated_orders:
        has_stop, has_take, _ = _evaluate_position_protection(position, updated_orders or [])
    if not has_take:
        log(f"[WARN] {symbol}: protection refresh left position without take-profit, retrying once", Fore.YELLOW)
        updated_orders = ensure_protection(
            exchange,
            symbol,
            position,
            protection_df,
            updated_orders,
            config=symbol_meta,
            handler=ensure_position_protection,
            log_fn=lambda msg: log(msg, Fore.LIGHTBLACK_EX),
        )
        if symbol and symbol.upper().startswith("DOGE"):
            log(
                f"[MODULE][protection] {symbol}: retry_orders={_summarize_open_orders_for_log(updated_orders)}",
                Fore.LIGHTBLACK_EX,
            )
        _, has_take, _ = _evaluate_position_protection(position, updated_orders or [])
        if not has_take:
            log(f"[WARN] {symbol}: still no take-profit after retry; monitor manually", Fore.YELLOW)
    return updated_orders, True
