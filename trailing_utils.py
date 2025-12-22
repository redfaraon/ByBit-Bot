from __future__ import annotations

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
) -> Sequence[dict]:
    """
    Thin wrapper that delegates trailing/protection to the injected handler.
    This keeps the trailing step modular without duplicating exchange logic here.
    """
    if handler is None:
        return list(open_orders or [])
    return handler(exchange, symbol, position, df_primary, open_orders, config)
