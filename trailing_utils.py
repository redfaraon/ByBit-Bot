from __future__ import annotations
MODULE_VERSION = "1.3.10"


from typing import Any, Callable, Sequence


def apply_trailing(
    exchange: Any,
    symbol: str,
    position: dict | None,
    df_primary: Any,
    open_orders: Sequence[dict] | None,
    *,
    config: dict | None,
    handler: Callable[[Any, str, dict | None, Any, Sequence[dict] | None, dict | None], Sequence[dict]] | None,
    log_fn: Callable[[str], None] | None = None,
) -> Sequence[dict]:
    """
    Thin wrapper that delegates trailing/protection to the injected handler.
    This keeps the trailing step modular without duplicating exchange logic here.
    """
    if log_fn:
        log_fn(
            f"[MODULE][trailing] sym={symbol} pos={'yes' if position else 'no'} open_orders={len(open_orders or [])} config_keys={list((config or {}).keys()) if isinstance(config, dict) else []}"
        )
    if handler is None:
        return list(open_orders or [])
    result = handler(exchange, symbol, position, df_primary, open_orders, config)
    if log_fn:
        log_fn(
            f"[MODULE][trailing] sym={symbol} result_orders={len(result or []) if isinstance(result, (list, tuple)) else 'n/a'}"
        )
    return result
