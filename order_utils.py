from __future__ import annotations
MODULE_VERSION = "1.3.10"


import math
import numbers
import re
from typing import Any

ACTIVE_HEDGE_MODE = False

REQUIRED_GLOBALS = {
    'safe_float',
    '_is_truthy_flag',
    'ACTIVE_HEDGE_MODE',
}

def configure(bindings: dict[str, Any]) -> None:
    for name in REQUIRED_GLOBALS:
        if name in bindings:
            globals()[name] = bindings[name]


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
    # PATCH: если символ деривативный (содержит ':'), не применять спотовые проверки
    try:
        if ':' in str(symbol):
            return True, None
    except Exception:
        pass

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
