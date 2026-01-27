from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from strategy import IndicatorBlock, StrategyContext
import trading_context


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
    if len(tf_df) < 2:
        return None
    last_row = tf_df.iloc[-1]
    prev_row = tf_df.iloc[-2]
    close_val = _safe_float(last_row.get("close"))
    ema20_val = _safe_float(last_row.get("ema20"))
    ema50_val = _safe_float(last_row.get("ema50"))
    ema20_prev = _safe_float(prev_row.get("ema20"))
    ema50_prev = _safe_float(prev_row.get("ema50"))
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
        ema20_prev=ema20_prev,
        ema50_prev=ema50_prev,
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
    primary_tf: str = "30m",
    secondary_tf: str = "4h",
) -> StrategyContext | None:
    return trading_context.build_symbol_context(
        symbol,
        tf30_df,
        tf4h_df,
        primary_df,
        current_position=current_position,
        open_orders=open_orders,
        pending_info=pending_info,
        news_score=news_score,
        funding_snapshot=funding_snapshot,
        open_interest_history=open_interest_history,
        risk_pct=risk_pct,
        primary_tf=primary_tf,
        secondary_tf=secondary_tf,
    )
