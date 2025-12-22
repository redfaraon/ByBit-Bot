from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from colorama import Fore


@dataclass
class AccountSnapshot:
    equity: float
    available_margin: float
    positions: dict[str, dict]
    open_orders: dict[str, list]
    balance_snapshot: dict[str, Any] | None = None


def build_snapshot(
    *,
    equity: float,
    available_margin: float,
    positions: dict[str, dict],
    open_orders: dict[str, list],
    balance_snapshot: dict[str, Any] | None = None,
) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        available_margin=available_margin,
        positions=dict(positions or {}),
        open_orders=dict(open_orders or {}),
        balance_snapshot=dict(balance_snapshot or {}) if isinstance(balance_snapshot, dict) else None,
    )


def log_snapshot(
    snapshot: AccountSnapshot,
    *,
    log_fn: Callable[[str, str], None],
    user_log_fn: Callable[[str, Any], None],
    tag: str = "[ACCOUNT]",
) -> None:
    equity = snapshot.equity
    available = snapshot.available_margin
    positions_count = len(snapshot.positions)
    orders_count = sum(len(v or []) for v in snapshot.open_orders.values())
    log_fn(f"{tag} equity={equity:.2f} available={available:.2f} positions={positions_count} open_orders={orders_count}", Fore.LIGHTBLACK_EX)
    user_log_fn(
        f"{tag} equity={equity:.2f} available={available:.2f} positions={positions_count} open_orders={orders_count}",
        color=Fore.LIGHTBLACK_EX,
    )
