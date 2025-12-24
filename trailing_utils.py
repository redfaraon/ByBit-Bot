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


def configure_from_spec(spec: Mapping[str, Any] | None) -> None:
    if not spec:
        return
    for key in list(TRAILING_CONFIG):
        value = spec.get(key)
        if value is None:
            continue
        try:
            TRAILING_CONFIG[key] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue


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
    effective_config = dict(TRAILING_CONFIG)
    if config:
        effective_config.update(config)
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
