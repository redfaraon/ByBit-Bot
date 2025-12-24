from __future__ import annotations
MODULE_VERSION = "1.3.10"


from typing import Any, Callable, Mapping, Sequence

TRAILING_CONFIG: dict[str, float] = {
    "sl_atr": 1.6,
    "tp_atr": 2.0,
    "trailing_atr_mult": 1.0,
    "trailing_dynamic_trigger_atr": 1.4,
    "trailing_dynamic_factor": 0.65,
    "trailing_dynamic_min_atr": 0.35,
}

TRAILING_PER_SYMBOL: dict[str, dict[str, float]] = {}
TRAILING_PER_REGIME: dict[str, dict[str, float]] = {}


def _normalize_symbol_key(symbol: str) -> str:
    text = (symbol or "").strip().upper()
    if not text:
        return text
    if ":" in text:
        text = text.split(":", 1)[0]
    return text


def resolve_trailing_config(
    symbol: str | None,
    *,
    regime: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    effective = dict(TRAILING_CONFIG)
    if regime:
        regime_key = (regime or "").strip().lower()
        overrides = TRAILING_PER_REGIME.get(regime_key)
        if overrides:
            effective.update(overrides)
    if symbol:
        sym_key = _normalize_symbol_key(symbol)
        overrides = TRAILING_PER_SYMBOL.get(sym_key) or TRAILING_PER_SYMBOL.get(str(symbol).strip().upper())
        if overrides:
            effective.update(overrides)
    if config:
        for key, value in config.items():
            if key not in effective:
                continue
            try:
                effective[key] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
    return effective


def _coerce_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not (val is None):
            result[key] = max(0.0, val)
    return result


def _coerce_nested_float_map(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, float]] = {}
    for raw_key, raw_cfg in value.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip()
        if not key:
            continue
        cfg = _coerce_float_map(raw_cfg)
        if cfg:
            result[key] = cfg
    return result


def configure_from_spec(spec: Mapping[str, Any] | None) -> None:
    if not spec:
        return

    nested = spec.get("trailing") if isinstance(spec.get("trailing"), Mapping) else None
    if nested:
        for key in list(TRAILING_CONFIG):
            value = nested.get(key)
            if value is None:
                continue
            try:
                TRAILING_CONFIG[key] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue

    for key in list(TRAILING_CONFIG):
        value = spec.get(key)
        if value is None:
            continue
        try:
            TRAILING_CONFIG[key] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue

    raw_symbol = spec.get("trailing_per_symbol") or spec.get("trailing_atr_mult_per_symbol")
    raw_regime = spec.get("trailing_per_regime") or spec.get("trailing_atr_mult_per_regime")
    try:
        TRAILING_PER_SYMBOL.clear()
        for sym, cfg in _coerce_nested_float_map(raw_symbol).items():
            TRAILING_PER_SYMBOL[_normalize_symbol_key(sym)] = cfg
    except Exception:
        pass
    try:
        TRAILING_PER_REGIME.clear()
        for name, cfg in _coerce_nested_float_map(raw_regime).items():
            TRAILING_PER_REGIME[str(name).strip().lower()] = cfg
    except Exception:
        pass


def apply_trailing(
    exchange: Any,
    symbol: str,
    position: dict | None,
    df_primary: Any,
    open_orders: Sequence[dict] | None,
    *,
    config: Mapping[str, Any] | None = None,
    handler: Callable[[Any, str, dict | None, Any, Sequence[dict] | None, dict | None], Sequence[dict]] | None,
    log_fn: Callable[[str], None] | None = None,
) -> Sequence[dict]:
    effective_config = resolve_trailing_config(symbol, config=config)
    if log_fn:
        log_fn(
            f"[MODULE][trailing] sym={symbol} pos={'yes' if position else 'no'} open_orders={len(open_orders or [])} config_keys={list(effective_config.keys())}"
        )
    if handler is None:
        return list(open_orders or [])
    result = handler(exchange, symbol, position, df_primary, open_orders, effective_config)
    if log_fn:
        log_fn(
            f"[MODULE][trailing] sym={symbol} result_orders={len(result or []) if isinstance(result, (list, tuple)) else 'n/a'}"
        )
    return result
