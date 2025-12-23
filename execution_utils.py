from __future__ import annotations
MODULE_VERSION = "1.3.10"


from typing import Any, Callable, Sequence


def execute_orders(
    exchange: Any,
    symbol: str,
    orders: Sequence[dict] | None,
    *,
    equity: float | None = None,
    current_position: dict | None = None,
    open_orders: Sequence[dict] | None = None,
    available_margin: float | None = None,
    symbol_leverage: float | None = None,
    max_limits_per_side: int = 1,
    handler: Callable[..., tuple] | None,
    log_fn: Callable[[str], None] | None = None,
    include_errors: bool = True,
) -> tuple:
    """
    Thin wrapper around order execution to keep logging/modularity in one place.
    """
    if log_fn:
        log_fn(
            f"[MODULE][execution] sym={symbol} orders={len(orders or [])} open_orders={len(open_orders or [])}"
        )
    if handler is None:
        return ([], False, []) if include_errors else ([], False)
    result = handler(
        exchange,
        symbol,
        orders or [],
        equity=equity,
        current_position=current_position,
        open_orders=open_orders,
        available_margin=available_margin,
        symbol_leverage=symbol_leverage,
        max_limits_per_side=max_limits_per_side,
    )
    if log_fn:
        if isinstance(result, tuple) and len(result) >= 2:
            executed = result[0]
            actions = result[1]
            exec_count = len(executed or []) if isinstance(executed, (list, tuple)) else "n/a"
            log_fn(f"[MODULE][execution] sym={symbol} executed={exec_count} actions={bool(actions)}")
        else:
            log_fn(f"[MODULE][execution] sym={symbol} result=n/a")
    if include_errors:
        return result
    if isinstance(result, tuple):
        return result[:2]
    return result
