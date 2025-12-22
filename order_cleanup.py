from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence


def cleanup_excess_non_reduce_limits(
    exchange: Any,
    symbol: str,
    open_orders: Sequence[dict] | None,
    position_side: str | None,
    max_per_side: int,
    *,
    handler: Callable[[Any, str, Sequence[dict] | None, str | None, int], Iterable[dict]] | None,
) -> Sequence[dict]:
    """
    Delegate limit cleanup to the provided handler (kept in bybitbot_impl for full logic).
    Returns the updated open orders snapshot from the handler.
    """
    if handler is None:
        return list(open_orders or [])
    return handler(exchange, symbol, open_orders, position_side, max_per_side)


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
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Delegate redundant stop cleanup to the provided handler.
    """
    if handler is None:
        return [], []
    return handler(exchange, symbol, reduce_orders, protection_side, position_qty, is_long, keep_ids_preferred)
