from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from strategy import IndicatorBlock, StrategyContext


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num


def _indicator_block_from_df(tf_df: pd.DataFrame | None) -> IndicatorBlock | None:
    if tf_df is None or tf_df.empty:
        return None
    last_row = tf_df.iloc[-1]
    close_val = _safe_float(last_row.get("close"))
    ema20_val = _safe_float(last_row.get("ema20"))
    ema50_val = _safe_float(last_row.get("ema50"))
    rsi_val = _safe_float(last_row.get("rsi14") or last_row.get("rsi"))
    atr_val = _safe_float(last_row.get("atr14") or last_row.get("atr"))
    if close_val is None or ema20_val is None or ema50_val is None or rsi_val is None or atr_val is None:
        return None
    atr_mean = None
    atr_std = None
    for column in ("atr14", "atr"):
        if column in tf_df.columns:
            series = tf_df[column].dropna().tail(120)
            if not series.empty:
                try:
                    atr_mean = float(series.mean())
                except Exception:
                    atr_mean = None
                try:
                    atr_std = float(series.std(ddof=0))
                except Exception:
                    atr_std = None
            break
    return IndicatorBlock(
        close=close_val,
        ema20=ema20_val,
        ema50=ema50_val,
        rsi=rsi_val,
        atr=atr_val,
        atr_mean=atr_mean,
        atr_std=atr_std,
    )


def _pending_limit_price(pending_info: dict[str, Any] | None) -> float | None:
    if not pending_info:
        return None
    for key in ("price", "px", "limit", "target"):
        val = _safe_float(pending_info.get(key))
        if val is not None and val > 0:
            return val
    return None


def build_manual_strategy_context(
    symbol: str,
    tf30_df: pd.DataFrame | None,
    tf4h_df: pd.DataFrame | None,
    primary_df: pd.DataFrame | None,
    *,
    current_position: dict[str, Any] | None,
    open_orders: Sequence[dict[str, Any]] | None,
    pending_info: dict[str, Any] | None,
    news_score: float | None,
    funding_snapshot: dict[str, Any] | None,
    open_interest_history: Sequence[Any] | None,
    risk_pct: float,
) -> StrategyContext | None:
    block_30m = _indicator_block_from_df(tf30_df)
    block_4h = _indicator_block_from_df(tf4h_df)
    if block_30m is None or block_4h is None:
        return None
    price_val = None
    if primary_df is not None and not primary_df.empty:
        price_val = _safe_float(primary_df.iloc[-1].get("close"))
    if price_val is None or price_val <= 0:
        price_val = block_30m.close
    if price_val is None or price_val <= 0:
        return None
    amount_val = _safe_float((current_position or {}).get("amount") or (current_position or {}).get("contracts")) or 0.0
    side_raw = (current_position or {}).get("side")
    if not side_raw and amount_val:
        side_raw = "buy" if amount_val > 0 else "sell"
    funding_rate = None
    if isinstance(funding_snapshot, dict):
        funding_rate = _safe_float(
            funding_snapshot.get("fundingRate")
            or funding_snapshot.get("funding_rate")
            or funding_snapshot.get("rate")
        )
    pending_price = _pending_limit_price(pending_info)
    return StrategyContext(
        symbol=symbol,
        price=price_val,
        tf30=block_30m,
        tf4h=block_4h,
        news_score=news_score,
        funding_rate=funding_rate,
        open_interest_history=open_interest_history or [],
        has_position=abs(amount_val) > 0,
        position_side=side_raw,
        position_size=abs(amount_val),
        open_orders=list(open_orders or []),
        pending_entry_price=pending_price,
        risk_pct=risk_pct,
    )
