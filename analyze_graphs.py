#!/usr/bin/env python3

"""Generate diagnostic graphs for equity, signals, timers, and commits."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]
    import matplotlib.dates as mdates  # type: ignore[import]
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    mdates = None  # type: ignore[assignment]
    print(f"[WARN] matplotlib unavailable: {exc}", file=sys.stderr)

_PNG_1PX_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/gn9zqUAAAAASUVORK5CYII="
)


def _write_fallback_png(output_dir: Path, filename: str) -> None:
    try:
        data = base64.b64decode(_PNG_1PX_B64)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / filename
        with out.open("wb") as fh:
            fh.write(data)
        print(f"[INFO] Saved minimal placeholder to {out}")
    except Exception as exc:
        print(f"[WARN] Failed to write fallback PNG {filename}: {exc}", file=sys.stderr)


def _save_placeholder(output_dir: Path, filename: str, title: str, subtitle: str = "No data") -> None:
    if plt is None:
        _write_fallback_png(output_dir, filename)
        return
    try:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.65, title, ha="center", va="center", fontsize=14, fontweight="bold")
        plt.text(0.5, 0.35, subtitle, ha="center", va="center", fontsize=11)
        output_path = output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] Saved placeholder graph to {output_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save placeholder {filename}: {exc}", file=sys.stderr)


def _save_plot_with_formats(output_dir: Path, base_name: str) -> None:
    png_path = output_dir / f"{base_name}.png"
    jpg_path = output_dir / f"{base_name}.jpg"
    plt.savefig(png_path)
    jpg_ok = False
    try:
        plt.savefig(jpg_path, format="jpeg")
        jpg_ok = True
    except Exception:
        jpg_path = None
    if jpg_ok:
        print(f"[INFO] Saved {base_name} plot to {png_path} and {jpg_path}")
    else:
        print(f"[INFO] Saved {base_name} plot to {png_path}")


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            try:
                return datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except Exception:
                return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return None


def _read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return []
    if limit:
        lines = lines[-limit:]
    out: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _load_commit_timestamps(repo_root: Path, limit: int = 500) -> List[datetime]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", f"--max-count={limit}", "--format=%ct"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return []
    timestamps: List[datetime] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ts = datetime.fromtimestamp(float(line), tz=timezone.utc)
        except Exception:
            continue
        timestamps.append(ts)
    return sorted(set(timestamps))


def _add_commit_lines(ax, commits: List[datetime], start: datetime, end: datetime) -> None:
    if not commits:
        return
    for ts in commits:
        if ts < start or ts > end:
            continue
        ax.axvline(ts, color="gray", linestyle=":", alpha=0.35, linewidth=0.8)


def _filter_entries(entries: List[Dict[str, Any]], window_hours: float | None) -> List[Dict[str, Any]]:
    if window_hours is None:
        return entries
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    filtered = []
    for entry in entries:
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        if ts and ts >= cutoff:
            filtered.append(entry)
    return filtered


def _extract_equity_series(entries: List[Dict[str, Any]]) -> Tuple[List[datetime], List[float], List[float], List[float]]:
    timestamps: List[datetime] = []
    equities: List[float] = []
    balances: List[float] = []
    drawdowns: List[float] = []
    peak = None
    for entry in entries:
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        eq_val = entry.get("equity") or entry.get("equity_total")
        if ts is None or eq_val is None:
            continue
        try:
            equity = float(eq_val)
        except Exception:
            continue
        balance_val = entry.get("balance")
        if balance_val is None:
            unreal = entry.get("unrealized") or entry.get("unrealized_pnl")
            try:
                unreal_val = float(unreal) if unreal is not None else None
            except Exception:
                unreal_val = None
            if unreal_val is not None and math.isfinite(unreal_val):
                balance_val = equity - unreal_val
        try:
            balance = float(balance_val) if balance_val is not None else math.nan
        except Exception:
            balance = math.nan
        peak = equity if peak is None or equity > peak else peak
        drawdown = (equity - peak) / peak * 100.0 if peak and peak > 0 else 0.0
        timestamps.append(ts)
        equities.append(equity)
        balances.append(balance)
        drawdowns.append(drawdown)
    return timestamps, equities, balances, drawdowns


def plot_equity_window(
    entries: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
    commits: List[datetime],
) -> None:
    base_name = f"equity_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    if not entries:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    timestamps, equities, balances, drawdowns = _extract_equity_series(entries)
    if not timestamps:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    try:
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(timestamps, equities, label="Equity", color="tab:blue", linewidth=1.8)
        if any(math.isfinite(x) for x in balances):
            ax1.plot(timestamps, balances, label="Balance", color="tab:green", linestyle="--", linewidth=1.4)
        ax1.set_title(title)
        ax1.set_ylabel("USDT")
        ax1.grid(axis="x", linestyle=":", alpha=0.4)
        if mdates is not None:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(formatter)
        ax2 = ax1.twinx()
        ax2.plot(timestamps, drawdowns, label="Drawdown %", color="tab:red", linestyle=":", linewidth=1.2)
        ax2.set_ylabel("Drawdown %")
        start, end = min(timestamps), max(timestamps)
        _add_commit_lines(ax1, commits, start, end)
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def _extract_timer_series(entries: List[Dict[str, Any]]) -> Tuple[List[datetime], List[float], List[float]]:
    timestamps: List[datetime] = []
    delays: List[float] = []
    intervals: List[float] = []
    for entry in entries:
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        if ts is None:
            continue
        delay_val = (
            entry.get("next_delay_minutes")
            or entry.get("cycle_next_delay_minutes")
            or entry.get("last_next_delay_minutes")
        )
        interval_val = (
            entry.get("cycle_interval_minutes")
            or entry.get("interval_minutes")
            or entry.get("last_interval_from_start_minutes")
        )
        try:
            delay = float(delay_val) if delay_val is not None else math.nan
        except Exception:
            delay = math.nan
        try:
            interval = float(interval_val) if interval_val is not None else math.nan
        except Exception:
            interval = math.nan
        if not (math.isfinite(delay) or math.isfinite(interval)):
            continue
        timestamps.append(ts)
        delays.append(delay)
        intervals.append(interval)
    return timestamps, delays, intervals


def plot_timer_window(
    entries: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
    commits: List[datetime],
) -> None:
    base_name = f"timer_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    timestamps, delays, intervals = _extract_timer_series(entries)
    if not timestamps:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timestamps, delays, label="Next delay (min)", color="tab:purple", linewidth=1.5)
        ax.plot(timestamps, intervals, label="Cycle interval (min)", color="tab:orange", linestyle="--", linewidth=1.3)
        ax.set_title(title)
        ax.set_ylabel("Minutes")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        if mdates is not None:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        start, end = min(timestamps), max(timestamps)
        _add_commit_lines(ax, commits, start, end)
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def _normalize_action(event_name: str | None) -> str:
    if not event_name:
        return "other"
    text_raw = str(event_name).strip()
    if not text_raw:
        return "other"
    text_lower = text_raw.lower()
    if "skip" in text_lower:
        return "skip"
    if "close" in text_lower:
        return "close"
    if "hedge" in text_lower:
        return "hedge"
    if "open" in text_lower:
        return "open"
    return text_raw


def _skip_reason_category(reason: str | None) -> str:
    if not reason:
        return "other"
    text = str(reason).strip()
    if not text:
        return "other"
    lower = text.lower()
    if "limit" in lower:
        return "limit"
    if "confluence" in lower or "converge" in lower:
        return "confluence"
    if "hold" in lower or "wait" in lower:
        return "hold"
    if "risk" in lower or "volatility" in lower:
        return "risk"
    if "conflict" in lower or "opposed" in lower or "duplicate" in lower:
        return "conflict"
    return text


def _normalize_reason(reason: str | None) -> str:
    if not reason:
        return "other"
    text = str(reason).lower()
    if "range" in text or "mean-reversion" in text:
        return "range"
    if "flat" in text or "sideways" in text:
        return "flat"
    if "counter" in text:
        return "countertrend"
    if "trend" in text:
        return "trend"
    if "news" in text:
        return "news"
    if "breakout" in text:
        return "breakout"
    return "other"


def _bucketize(entries: Iterable[Tuple[datetime, str]], bucket_minutes: int) -> Dict[datetime, Dict[str, int]]:
    buckets: Dict[datetime, Dict[str, int]] = {}
    for ts, key in entries:
        if bucket_minutes <= 60:
            bucket = ts.replace(minute=(ts.minute // bucket_minutes) * bucket_minutes, second=0, microsecond=0)
        else:
            bucket = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        bucket_map = buckets.setdefault(bucket, {})
        bucket_map[key] = bucket_map.get(key, 0) + 1
    return buckets


def plot_skip_reason_timeseries(
    events: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
    commits: List[datetime],
    bucket_minutes: int,
) -> None:
    base_name = f"skip_reasons_timeseries_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    points: List[Tuple[datetime, str]] = []
    for entry in events:
        action = _normalize_action(entry.get("event") or entry.get("action"))
        if action != "skip":
            continue
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        if ts is None:
            continue
        reason = entry.get("reason")
        category = _skip_reason_category(reason)
        points.append((ts, category))
    if not points:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    buckets = _bucketize(points, bucket_minutes)
    series_keys = sorted({key for bucket in buckets.values() for key in bucket})
    bucket_times = sorted(buckets.keys())
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        for key in series_keys:
            values = [buckets[ts].get(key, 0) for ts in bucket_times]
            ax.plot(bucket_times, values, label=key)
        ax.set_title(title)
        ax.set_ylabel("Skip count")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        if mdates is not None:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        start, end = min(bucket_times), max(bucket_times)
        _add_commit_lines(ax, commits, start, end)
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def plot_signal_timeseries(
    events: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
    commits: List[datetime],
    bucket_minutes: int,
) -> None:
    base_name = f"signals_timeseries_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    points: List[Tuple[datetime, str]] = []
    for entry in events:
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        action = entry.get("event") or entry.get("action")
        if ts is None:
            continue
        points.append((ts, _normalize_action(action)))
    if not points:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    buckets = _bucketize(points, bucket_minutes)
    series_keys = sorted({key for bucket in buckets.values() for key in bucket})
    bucket_times = sorted(buckets.keys())
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        for key in series_keys:
            values = [buckets[ts].get(key, 0) for ts in bucket_times]
            ax.plot(bucket_times, values, label=key)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        if mdates is not None:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        start, end = min(bucket_times), max(bucket_times)
        _add_commit_lines(ax, commits, start, end)
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def plot_signal_pie(
    events: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
) -> None:
    base_name = f"signals_pie_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    counts: Dict[str, int] = {}
    for entry in events:
        action = entry.get("event") or entry.get("action")
        key = _normalize_action(action)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    try:
        plt.figure(figsize=(6, 6))
        labels = list(counts.keys())
        sizes = [counts[label] for label in labels]
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(title)
        plt.axis("equal")
        plt.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def plot_reason_timeseries(
    events: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
    commits: List[datetime],
    bucket_minutes: int,
) -> None:
    base_name = f"reasons_timeseries_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    points: List[Tuple[datetime, str]] = []
    for entry in events:
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        event_name = entry.get("event") or entry.get("action")
        if ts is None or _normalize_action(event_name) != "open":
            continue
        points.append((ts, _normalize_reason(entry.get("reason"))))
    if not points:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    buckets = _bucketize(points, bucket_minutes)
    series_keys = sorted({key for bucket in buckets.values() for key in bucket})
    bucket_times = sorted(buckets.keys())
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        for key in series_keys:
            values = [buckets[ts].get(key, 0) for ts in bucket_times]
            ax.plot(bucket_times, values, label=key)
        ax.set_title(title)
        ax.set_ylabel("Open signals")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        if mdates is not None:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        start, end = min(bucket_times), max(bucket_times)
        _add_commit_lines(ax, commits, start, end)
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def plot_equity_daily_bars(
    entries: List[Dict[str, Any]],
    output_dir: Path,
    commits: List[datetime],
) -> None:
    base_name = "equity_daily_bars_all"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    daily: Dict[datetime, Dict[str, float]] = {}
    for entry in entries:
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        if ts is None:
            continue
        day = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        equity_val = entry.get("equity") or entry.get("equity_total")
        balance_val = entry.get("balance")
        if balance_val is None:
            unreal = entry.get("unrealized") or entry.get("unrealized_pnl")
            try:
                unreal_val = float(unreal) if unreal is not None else None
            except Exception:
                unreal_val = None
            if equity_val is not None and unreal_val is not None and math.isfinite(unreal_val):
                try:
                    balance_val = float(equity_val) - unreal_val
                except Exception:
                    balance_val = None
        try:
            equity_float = float(equity_val) if equity_val is not None else None
        except Exception:
            equity_float = None
        try:
            balance_float = float(balance_val) if balance_val is not None else None
        except Exception:
            balance_float = None
        if equity_float is None or not math.isfinite(equity_float):
            continue
        daily[day] = {"equity": equity_float, "balance": balance_float}
    if not daily:
        _save_placeholder(output_dir, f"{base_name}.png", "Daily equity/balance")
        return
    days = sorted(daily.keys())
    equities = [daily[day]["equity"] for day in days]
    balances = [daily[day]["balance"] if daily[day]["balance"] is not None else math.nan for day in days]
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        width = 0.4
        x_vals = mdates.date2num(days) if mdates is not None else list(range(len(days)))
        ax.bar([x - width / 2 for x in x_vals], equities, width=width, label="Equity")
        if any(math.isfinite(x) for x in balances):
            ax.bar([x + width / 2 for x in x_vals], balances, width=width, label="Balance")
        ax.set_title("Daily equity/balance (all time)")
        ax.set_ylabel("USDT")
        if mdates is not None:
            ax.xaxis_date()
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        start, end = min(days), max(days)
        _add_commit_lines(ax, commits, start, end)
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def plot_commit_deltas(
    entries: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
    commits: List[datetime],
) -> None:
    base_name = f"equity_commit_deltas_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    if not commits or not entries:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    entries_sorted = sorted(entries, key=lambda e: _parse_timestamp(e.get("timestamp") or e.get("time") or e.get("ts")) or datetime.min)
    series: List[Tuple[datetime, float, float | None]] = []
    for commit_ts in commits:
        last_entry = None
        for entry in entries_sorted:
            ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
            if ts is None or ts > commit_ts:
                break
            last_entry = entry
        if not last_entry:
            continue
        equity_val = last_entry.get("equity") or last_entry.get("equity_total")
        balance_val = last_entry.get("balance")
        if balance_val is None:
            unreal = last_entry.get("unrealized") or last_entry.get("unrealized_pnl")
            try:
                unreal_val = float(unreal) if unreal is not None else None
            except Exception:
                unreal_val = None
            if equity_val is not None and unreal_val is not None and math.isfinite(unreal_val):
                try:
                    balance_val = float(equity_val) - unreal_val
                except Exception:
                    balance_val = None
        try:
            equity_float = float(equity_val) if equity_val is not None else None
        except Exception:
            equity_float = None
        try:
            balance_float = float(balance_val) if balance_val is not None else None
        except Exception:
            balance_float = None
        if equity_float is None or not math.isfinite(equity_float):
            continue
        series.append((commit_ts, equity_float, balance_float))
    if not series:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    latest_entry = entries_sorted[-1]
    latest_ts = _parse_timestamp(latest_entry.get("timestamp") or latest_entry.get("time") or latest_entry.get("ts"))
    if latest_ts:
        equity_val = latest_entry.get("equity") or latest_entry.get("equity_total")
        balance_val = latest_entry.get("balance")
        if balance_val is None:
            unreal = latest_entry.get("unrealized") or latest_entry.get("unrealized_pnl")
            try:
                unreal_val = float(unreal) if unreal is not None else None
            except Exception:
                unreal_val = None
            if equity_val is not None and unreal_val is not None and math.isfinite(unreal_val):
                try:
                    balance_val = float(equity_val) - unreal_val
                except Exception:
                    balance_val = None
        try:
            equity_float = float(equity_val) if equity_val is not None else None
        except Exception:
            equity_float = None
        try:
            balance_float = float(balance_val) if balance_val is not None else None
        except Exception:
            balance_float = None
        if equity_float is not None and math.isfinite(equity_float):
            series.append((latest_ts, equity_float, balance_float))
    if len(series) < 2:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    deltas_equity: List[float] = []
    deltas_balance: List[float] = []
    delta_times: List[datetime] = []
    for idx in range(1, len(series)):
        prev = series[idx - 1]
        cur = series[idx]
        delta_times.append(cur[0])
        deltas_equity.append(cur[1] - prev[1])
        if cur[2] is not None and prev[2] is not None:
            deltas_balance.append(cur[2] - prev[2])
        else:
            deltas_balance.append(math.nan)
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        x_vals = mdates.date2num(delta_times) if mdates is not None else list(range(len(delta_times)))
        width = 0.4
        ax.bar([x - width / 2 for x in x_vals], deltas_equity, width=width, label="Equity delta")
        if any(math.isfinite(x) for x in deltas_balance):
            ax.bar([x + width / 2 for x in x_vals], deltas_balance, width=width, label="Balance delta")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Delta (USDT)")
        if mdates is not None:
            ax.xaxis_date()
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        start, end = min(delta_times), max(delta_times)
        _add_commit_lines(ax, commits, start, end)
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def _extract_delta_series(entries: List[Dict[str, Any]]) -> List[Tuple[datetime, float, float | None]]:
    series: List[Tuple[datetime, float, float | None]] = []
    for entry in sorted(
        entries,
        key=lambda e: _parse_timestamp(e.get("timestamp") or e.get("time") or e.get("ts")) or datetime.min,
    ):
        ts = _parse_timestamp(entry.get("timestamp") or entry.get("time") or entry.get("ts"))
        if ts is None:
            continue
        equity_val = entry.get("equity") or entry.get("equity_total")
        if equity_val is None:
            continue
        try:
            equity_float = float(equity_val)
        except Exception:
            continue
        balance_val = entry.get("balance")
        if balance_val is None:
            unreal = entry.get("unrealized") or entry.get("unrealized_pnl")
            try:
                unreal_float = float(unreal) if unreal is not None else None
            except Exception:
                unreal_float = None
            if unreal_float is not None and math.isfinite(unreal_float):
                try:
                    balance_val = equity_float - unreal_float
                except Exception:
                    balance_val = None
        balance_float: float | None
        try:
            balance_float = float(balance_val) if balance_val is not None else None
        except Exception:
            balance_float = None
        series.append((ts, equity_float, balance_float))
    deltas: List[Tuple[datetime, float, float | None]] = []
    for prev, cur in zip(series, series[1:]):
        delta_time = cur[0]
        delta_equity = cur[1] - prev[1]
        prev_balance = prev[2]
        cur_balance = cur[2]
        delta_balance: float | None
        if prev_balance is not None and cur_balance is not None and math.isfinite(prev_balance) and math.isfinite(cur_balance):
            delta_balance = cur_balance - prev_balance
        else:
            delta_balance = None
        deltas.append((delta_time, delta_equity, delta_balance))
    return deltas


def plot_delta_window(
    entries: List[Dict[str, Any]],
    output_dir: Path,
    suffix: str,
    title: str,
    commits: List[datetime],
) -> None:
    base_name = f"equity_delta_{suffix}"
    if plt is None:
        _write_fallback_png(output_dir, f"{base_name}.png")
        return
    deltas = _extract_delta_series(entries)
    if not deltas:
        _save_placeholder(output_dir, f"{base_name}.png", title)
        return
    times = [item[0] for item in deltas]
    equity_deltas = [item[1] for item in deltas]
    balance_deltas = [item[2] if item[2] is not None else math.nan for item in deltas]
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        x_vals = mdates.date2num(times) if mdates is not None else list(range(len(times)))
        width = 0.4
        ax.bar([x - width / 2 for x in x_vals], equity_deltas, width=width, label="Equity delta")
        if any(math.isfinite(x) for x in balance_deltas):
            ax.bar([x + width / 2 for x in x_vals], balance_deltas, width=width, label="Balance delta")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Delta (USDT)")
        if mdates is not None:
            ax.xaxis_date()
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        start, end = min(times), max(times)
        _add_commit_lines(ax, commits, start, end)
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, base_name)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot {base_name}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate diagnostic graphs for Bybit bot")
    parser.add_argument("--output", type=Path, default=Path("graphs"), help="Directory for generated plots")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("assets"),
        help="Directory that stores bot state files (equity_history/results_state).",
    )
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        script_dir = Path(__file__).resolve().parent
        state_dir = args.state_dir.expanduser()
        env_state_dir = os.getenv("BYBITBOT_STATE_DIR")
        raw_candidates = [
            state_dir,
            script_dir,
            script_dir / "assets",
            script_dir.parent,
            Path.cwd(),
            Path.cwd() / "assets",
        ]
        if env_state_dir:
            raw_candidates.append(Path(env_state_dir).expanduser())
        candidate_dirs: List[Path] = []
        for candidate in raw_candidates:
            if not candidate:
                continue
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved not in candidate_dirs:
                candidate_dirs.append(resolved)

        def _find_state_path(name: str) -> Path:
            return next((d / name for d in candidate_dirs if (d / name).exists()), state_dir / name)

        equity_path = _find_state_path("equity_history.json")
        print(f"[INFO] Equity source: {equity_path}")
        equity_data = _read_json(equity_path)
        equity_history: List[Dict[str, Any]]
        if isinstance(equity_data, dict):
            equity_history = equity_data.get("history") or equity_data.get("entries") or []
        else:
            equity_history = equity_data or []

        diagnostics_path = next(
            (
                p
                for d in candidate_dirs
                for p in (
                    d / "diagnostics" / "strategy_events.jsonl",
                    d / "strategy_events.jsonl",
                )
                if p.exists()
            ),
            None,
        )
        strategy_events = _read_jsonl(diagnostics_path) if diagnostics_path else []

        repo_root = next((d for d in [script_dir, script_dir.parent] if (d / ".git").exists()), script_dir)
        commit_ts = _load_commit_timestamps(repo_root)

        windows = {
            "all": None,
            "week": 7 * 24,
            "day": 24,
        }
        for suffix, hours in windows.items():
            entries = _filter_entries(equity_history, hours)
            events = _filter_entries(strategy_events, hours)
        label = "All-time" if suffix == "all" else "Last 7 days" if suffix == "week" else "Last 24h"
        plot_equity_window(entries, args.output, suffix, f"Equity / Balance ({label})", commit_ts)
        plot_delta_window(entries, args.output, suffix, f"Equity delta ({label})", commit_ts)
        plot_timer_window(entries, args.output, suffix, f"Cycle timers ({label})", commit_ts)
        bucket_minutes = 60 if suffix == "day" else 24 * 60
        plot_signal_timeseries(events, args.output, suffix, f"Signals ({label})", commit_ts, bucket_minutes)
        plot_signal_pie(events, args.output, suffix, f"Signal distribution ({label})")
        plot_reason_timeseries(events, args.output, suffix, f"Open reasons ({label})", commit_ts, bucket_minutes)
        plot_skip_reason_timeseries(events, args.output, suffix, f"Skip reasons ({label})", commit_ts, bucket_minutes)
        plot_commit_deltas(entries, args.output, suffix, f"Commit deltas ({label})", commit_ts)

        plot_equity_daily_bars(equity_history, args.output, commit_ts)
        return 0
    except Exception as exc:  # pragma: no cover
        import traceback

        print(f"[ERROR] Failed to generate graphs: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
