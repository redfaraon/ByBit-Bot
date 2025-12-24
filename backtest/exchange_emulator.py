from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .execution_emulator import ExecutionEmulator, VirtualOrder


@dataclass
class ExchangeEmulator:
    time_step_minutes: int = 1
    history: list[VirtualOrder] = field(default_factory=list)

    def advance(self, executor: ExecutionEmulator, point: Any) -> None:
        for order in list(executor.active_orders):
            self.history.append(order)
            price = point.prices.get(order.symbol) if hasattr(point, "prices") else None
            if price:
                executor.settle_order(order, price)
