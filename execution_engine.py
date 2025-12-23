from __future__ import annotations

import math
from typing import Any
from colorama import Fore

from order_utils import (
    ORDER_TYPE_MAP,
    VALID_ORDER_TYPES,
    _normalize_order_side,
    _order_allows_increase,
    _sanitize_order_params_for_category,
    _spot_funds_sufficient,
    _summarize_order_spec,
    compute_order_amount,
    get_position_idx,
    normalize_order_type_key,
)

ORDER_MARGIN_UTILIZATION = 0.95
NON_REDUCE_PRICE_DECIMALS = 4

REQUIRED_GLOBALS = {
    'safe_float',
    'log',
    'send_tg',
    'cancel_order_by_id',
    '_resolve_symbol_alias',
    '_infer_market_category',
    '_is_truthy_flag',
    'ORDER_MARGIN_UTILIZATION',
    'NON_REDUCE_PRICE_DECIMALS',
}

def configure(bindings: dict[str, Any]) -> None:
    for name in REQUIRED_GLOBALS:
        if name in bindings:
            globals()[name] = bindings[name]


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
