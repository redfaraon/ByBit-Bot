from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableSequence, Sequence

import pandas as pd

import strategy
import strategy_context
import signal_intent_mapper
from .exchange_emulator import ExchangeEmulator
from .execution_emulator import ExecutionEmulator, VirtualOrder, VirtualPosition
from .stats import BacktestStats


def _load_strategy_spec(path: str | None) -> None:
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as fp:
            spec = json.load(fp)
    except Exception:
        return
    strategy.SPEC.clear()
    strategy.SPEC.update(spec)
    strategy.CONTEXT_SPEC.clear()
    strategy.CONTEXT_SPEC.update(spec.get("context", {}))
    strategy.RULES_SPEC.clear()
    strategy.RULES_SPEC.update(spec.get("rules", {}))
    strategy.SIZE_SPEC.clear()
    strategy.SIZE_SPEC.update(spec.get("sizing", {}))
    strategy.EVENTS_SPEC.clear()
    strategy.EVENTS_SPEC.update(spec.get("events", {}))


def _to_dataframe(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return pd.DataFrame(value)
    if isinstance(value, Mapping):
        return pd.DataFrame(value)
    return None


@dataclass
class HistoricPoint:
    timestamp: datetime.datetime
    prices: Mapping[str, float]
    bars: Mapping[str, Mapping[str, Any]]
    indicators: Mapping[str, Any]
    news: Mapping[str, float]


@dataclass
class BacktestParams:
    strategy_path: str
    start: datetime.datetime
    end: datetime.datetime
    init_balance: float
    universe_config: Mapping[str, Any] | None = None
    ticker_provider: Iterable[HistoricPoint] | None = None


@dataclass
class BacktestResult:
    stats: Mapping[str, Any]
    timeline: list[Mapping[str, Any]]


class BacktestSession:
    def __init__(self, params: BacktestParams) -> None:
        self.params = params
        self.executor = ExecutionEmulator(initial_balance=params.init_balance)
        self.exchange = ExchangeEmulator()
        self.stats = BacktestStats()
        self.state: dict[str, Any] = {"current_time": params.start}

    def run(self) -> BacktestResult:
        timeline: list[Mapping[str, Any]] = []
        points = list(self.params.ticker_provider or [])
        if not points:
            return BacktestResult(stats={}, timeline=timeline)
        for point in points:
            self.state["current_time"] = point.timestamp
            self._build_universe(point)
            self._collect_context(point)
            decisions = self._apply_strategy(point)
            for decision in decisions:
                self.executor.apply_decision(decision, point)
            self._run_cleaning(point)
            self._run_trailing(point)
            self._run_protection(point)
            self.exchange.advance(self.executor, point)
            snapshot = self._capture_snapshot(point)
            timeline.append(snapshot)
            self.stats.record_point(snapshot)
        summary = self.stats.summary()
        return BacktestResult(stats=summary, timeline=timeline)

    def _build_universe(self, point: HistoricPoint) -> None:
        self.state["universe"] = list(point.prices.keys())

    def _collect_context(self, point: HistoricPoint) -> None:
        self.state["last_context"] = {
            "bars": point.bars,
            "indicators": point.indicators,
            "news": point.news,
            "prices": point.prices,
        }

    def _apply_strategy(self, point: HistoricPoint) -> Sequence[Mapping[str, Any]]:
        decisions: list[Mapping[str, Any]] = []
        for symbol in sorted(point.prices):
            ctx = self._build_strategy_context(symbol, point)
            if ctx is None:
                continue
            event = strategy.get_signal_without_ai(ctx)
            decision = signal_intent_mapper.apply_event(event, ctx)
            decisions.append(decision)
        return decisions

    def _build_strategy_context(self, symbol: str, point: HistoricPoint):
        tf30_df = _to_dataframe(point.bars.get("30m", {}).get(symbol))
        tf4h_df = _to_dataframe(point.bars.get("4h", {}).get(symbol))
        primary_df = tf30_df
        current_position = self._position_for_symbol(symbol)
        open_orders = self._open_orders_for_symbol(symbol)
        news_score = point.news.get(symbol)
        return strategy_context.build_manual_strategy_context(
            symbol,
            tf30_df,
            tf4h_df,
            primary_df,
            current_position=current_position,
            open_orders=open_orders,
            pending_info=None,
            news_score=news_score,
            funding_snapshot=None,
            open_interest_history=None,
            risk_pct=self.params.init_balance,
        )

    def _position_for_symbol(self, symbol: str) -> dict[str, Any] | None:
        pos = self.executor.positions.get(symbol)
        if not pos:
            return None
        return {"symbol": pos.symbol, "side": pos.side, "amount": pos.qty}

    def _open_orders_for_symbol(self, symbol: str) -> list[Mapping[str, Any]]:
        return [asdict(order) for order in self.executor.active_orders if order.symbol == symbol]

    def _run_cleaning(self, point: HistoricPoint) -> None:
        pass

    def _run_trailing(self, point: HistoricPoint) -> None:
        pass

    def _run_protection(self, point: HistoricPoint) -> None:
        pass

    def _capture_snapshot(self, point: HistoricPoint) -> Mapping[str, Any]:
        return {
            "timestamp": point.timestamp.isoformat(),
            "balance": self.executor.balance,
            "equity": self.executor.equity,
            "positions": {k: asdict(v) for k, v in self.executor.positions.items()},
            "orders": [asdict(o) for o in self.executor.active_orders],
        }


def _parse_iso(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value)


def _load_points(path: str) -> list[HistoricPoint]:
    if not path:
        return []
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    try:
        data: list[Mapping[str, Any]] = json.loads(raw)
    except Exception:
        return []
    points: list[HistoricPoint] = []
    for entry in data:
        timestamp = _parse_iso(entry.get("timestamp", "1970-01-01T00:00:00"))
        points.append(
            HistoricPoint(
                timestamp=timestamp,
                prices=entry.get("prices", {}),
                bars=entry.get("bars", {}),
                indicators=entry.get("indicators", {}),
                news=entry.get("news", {}),
            )
        )
    return sorted(points, key=lambda p: p.timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the manual strategy on historical data.")
    parser.add_argument("--strategy", required=True, help="Path to the strategy JSON spec.")
    parser.add_argument("--from", dest="start", required=True, help="ISO timestamp for start.")
    parser.add_argument("--to", dest="end", help="ISO timestamp for end.")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance.")
    parser.add_argument("--data", required=True, help="Path to historical dataset (JSON).")
    args = parser.parse_args()

    _load_strategy_spec(args.strategy)
    start = _parse_iso(args.start)
    end = _parse_iso(args.end) if args.end else start
    points = _load_points(args.data)
    params = BacktestParams(
        strategy_path=args.strategy,
        start=start,
        end=end,
        init_balance=args.balance,
        ticker_provider=points,
    )
    session = BacktestSession(params)
    result = session.run()
    print("Backtest result:", result.stats)
    for idx, snapshot in enumerate(result.timeline, 1):
        print(f"[{idx:03}] {snapshot['timestamp']} balance={snapshot['balance']:.2f} equity={snapshot['equity']:.2f}")


if __name__ == "__main__":
    main()
