from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableSequence, Sequence


@dataclass
class VirtualOrder:
    symbol: str
    action: str
    price: float | None = None
    qty: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class VirtualPosition:
    symbol: str
    side: str
    qty: float
    entry_price: float


@dataclass
class ExecutionEmulator:
    initial_balance: float
    balance: float | None = None
    equity: float | None = None
    positions: dict[str, VirtualPosition] = field(default_factory=dict)
    active_orders: MutableSequence[VirtualOrder] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.balance = self.initial_balance
        self.equity = self.initial_balance

    def apply_decision(self, decision: Mapping[str, Any], point: Any) -> None:
        order = VirtualOrder(
            symbol=decision.get("symbol", "UNKNOWN"),
            action=decision.get("action", "skip"),
            price=decision.get("price"),
            qty=decision.get("qty"),
            metadata=decision.get("metadata") or {},
        )
        if order.action != "skip":
            self.active_orders.append(order)

    def settle_order(self, order: VirtualOrder, price: float) -> None:
        pos = self.positions.get(order.symbol)
        qty = order.qty or 0.0
        if not pos:
            pos = VirtualPosition(
                symbol=order.symbol, side=order.action.lower(), qty=qty, entry_price=price
            )
            self.positions[order.symbol] = pos
        else:
            pos.qty += qty
        self.active_orders.remove(order)

    def update_account(self, delta: float) -> None:
        if self.balance is None:
            self.balance = self.initial_balance
        self.balance += delta
        self.equity = (self.equity or self.initial_balance) + delta
