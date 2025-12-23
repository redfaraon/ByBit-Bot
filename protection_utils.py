from __future__ import annotations
MODULE_VERSION = "1.3.10"


from typing import Any, Callable, Sequence


def ensure_protection(
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
    Route protection/trailing application to the provided handler.
    The heavy logic stays in bybitbot_impl; this wrapper keeps the interface modular.
    """
    if log_fn:
        log_fn(
            f"[MODULE][protection] sym={symbol} pos={'yes' if position else 'no'} open_orders={len(open_orders or [])} config_keys={list((config or {}).keys()) if isinstance(config, dict) else []}"
        )
    if handler is None:
        return list(open_orders or [])
    result = handler(exchange, symbol, position, df_primary, open_orders, config)
    if log_fn:
        log_fn(
            f"[MODULE][protection] sym={symbol} result_orders={len(result or []) if isinstance(result, (list, tuple)) else 'n/a'}"
        )
    return result
