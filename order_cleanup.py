from __future__ import annotations
MODULE_VERSION = "1.3.10"


from typing import Any, Callable, Iterable, Sequence


def cleanup_excess_non_reduce_limits(
    exchange: Any,
    symbol: str,
    open_orders: Sequence[dict] | None,
    position_side: str | None,
    max_per_side: int,
    *,
    handler: Callable[[Any, str, Sequence[dict] | None, str | None, int], Iterable[dict]] | None,
    log_fn: Callable[[str], None] | None = None,
) -> Sequence[dict]:
    """
    Delegate limit cleanup to the provided handler (kept in bybitbot_impl for full logic).
    Returns the updated open orders snapshot from the handler.
    """
    if log_fn:
        log_fn(
            f"[MODULE][cleanup_excess_non_reduce_limits] sym={symbol} open_orders={len(open_orders or [])} pos_side={position_side} max_per_side={max_per_side}"
        )
    if handler is None:
        result = list(open_orders or [])
    else:
        result = handler(exchange, symbol, open_orders, position_side, max_per_side)
    if log_fn:
        log_fn(f"[MODULE][cleanup_excess_non_reduce_limits] result_count={len(result or [])}")
    return result


def cleanup_redundant_stops(
    exchange: Any,
    symbol: str,
    reduce_orders: Sequence[dict] | None,
    protection_side: str,
    position_qty: float,
    is_long: bool,
    *,
    keep_ids_preferred: set[str] | None,
    handler: Callable[[Any, str, Sequence[dict] | None, str, float, bool, set[str] | None], tuple[list[str], list[tuple[str, str]]]] | None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Delegate redundant stop cleanup to the provided handler.
    """
    if log_fn:
        log_fn(
            f"[MODULE][cleanup_redundant_stops] sym={symbol} reduce_orders={len(reduce_orders or [])} side={protection_side} qty={position_qty} is_long={is_long}"
        )
    if handler is None:
        return [], []
    cancelled, errors = handler(exchange, symbol, reduce_orders, protection_side, position_qty, is_long, keep_ids_preferred)
    if log_fn:
        log_fn(f"[MODULE][cleanup_redundant_stops] cancelled={cancelled} errors={errors}")
    return cancelled, errors
