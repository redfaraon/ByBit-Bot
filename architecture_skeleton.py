"""
High-level skeleton for future modular architecture.

This file defines minimal interfaces / base classes for the
planned services so they can be referenced and gradually
implemented without breaking the existing monolith.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Service(Protocol):
    """Generic service lifecycle interface."""

    def start(self) -> None:  # pragma: no cover - skeleton
        ...

    def stop(self) -> None:  # pragma: no cover - skeleton
        ...


@dataclass
class HealthHelper(Service):
    """
    Helper for keeping services healthy:
    - health checks
    - metrics
    - restart/backoff policies
    """

    name: str = "health_helper"

    def start(self) -> None:  # pragma: no cover - skeleton
        pass

    def stop(self) -> None:  # pragma: no cover - skeleton
        pass


@dataclass
class MarketController(Service):
    """
    Market controllers:
    - encapsulate logic per asset class (crypto, stocks, FX, metals, commodities)
    - know how to fetch and normalize market data for a given universe
    """

    market_kind: str  # e.g. "crypto", "stocks", "forex"

    def start(self) -> None:  # pragma: no cover - skeleton
        pass

    def stop(self) -> None:  # pragma: no cover - skeleton
        pass


@dataclass
class BrokerController(Service):
    """
    Broker controllers:
    - wrap concrete exchanges / brokers (Bybit, etc.)
    - provide a unified interface for orders, balances, positions
    """

    broker_name: str  # e.g. "bybit"

    def start(self) -> None:  # pragma: no cover - skeleton
        pass

    def stop(self) -> None:  # pragma: no cover - skeleton
        pass


@dataclass
class ContextAggregator(Service):
    """
    Context collector:
    - aggregates news, indicators, price movements, external signals
    - produces a normalized context payload per universe/market
    """

    def start(self) -> None:  # pragma: no cover - skeleton
        pass

    def stop(self) -> None:  # pragma: no cover - skeleton
        pass

    def build_context(self, universe: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - skeleton
        return {}


@dataclass
class AIProviderController(Service):
    """
    AI vendor controller:
    - abstracts interaction with providers (OpenAI, DeepSeek, etc.)
    - manages models, token budgets, retries and observability
    """

    provider_name: str  # e.g. "openai"

    def start(self) -> None:  # pragma: no cover - skeleton
        pass

    def stop(self) -> None:  # pragma: no cover - skeleton
        pass

    def run_universe_model(self, context: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - skeleton
        return {}

    def run_trade_plan_model(self, context: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - skeleton
        return {}


@dataclass
class UserBotWorker(Service):
    """
    Per-user worker:
    - applies trade plans to a specific account
    - owns user-specific env, logs, Telegram and risk parameters
    """

    user_id: str

    def start(self) -> None:  # pragma: no cover - skeleton
        pass

    def stop(self) -> None:  # pragma: no cover - skeleton
        pass

    def apply_trade_plan(self, plan: dict[str, Any]) -> None:  # pragma: no cover - skeleton
        pass


@dataclass
class EngineCore(Service):
    """
    Core engine:
    - orchestrates markets, brokers, context aggregation and AI providers
    - produces universe + trade plans for userbot workers
    """

    def start(self) -> None:  # pragma: no cover - skeleton
        pass

    def stop(self) -> None:  # pragma: no cover - skeleton
        pass

    def build_trade_plan(self, universe_hint: dict[str, Any] | None = None) -> dict[str, Any]:  # pragma: no cover - skeleton
        return {}

