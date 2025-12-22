from __future__ import annotations

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
) -> Sequence[dict]:
    """
    Route protection/trailing application to the provided handler.
    The heavy logic stays in bybitbot_impl; this wrapper keeps the interface modular.
    """
    if handler is None:
        return list(open_orders or [])
    return handler(exchange, symbol, position, df_primary, open_orders, config)
