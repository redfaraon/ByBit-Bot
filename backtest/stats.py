from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableSequence


@dataclass
class BacktestStats:
    points: MutableSequence[Mapping[str, Any]] = field(default_factory=list)

    def record_point(self, point: Mapping[str, Any]) -> None:
        self.points.append(point)

    def summary(self) -> Mapping[str, Any]:
        if not self.points:
            return {"points": 0}
        return {
            "points": len(self.points),
            "final_balance": self.points[-1].get("balance"),
            "final_equity": self.points[-1].get("equity"),
        }
