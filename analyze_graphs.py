#!/usr/bin/env python3

"""Generate equity, PnL, and signal distribution plots from Bybit bot logs."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]
    import matplotlib.dates as mdates  # type: ignore[import]
except Exception as exc:  # pragma: no cover - protects from missing dependencies
    plt = None  # type: ignore[assignment]
    mdates = None  # type: ignore[assignment]
    print(f"[WARN] matplotlib unavailable: {exc}", file=sys.stderr)

# 1x1 PNG placeholder (white) for environments without matplotlib
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
        plt.axis('off')
        plt.text(0.5, 0.65, title, ha='center', va='center', fontsize=14, fontweight='bold')
        plt.text(0.5, 0.35, subtitle, ha='center', va='center', fontsize=11)
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


def _load_signal_history_from_ai_log(path: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return []
    if limit > 0:
        lines = lines[-limit:]
    history: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        decision = payload.get("decision") or {}
        if not isinstance(decision, dict):
            continue
        ts_val = payload.get("timestamp") or payload.get("time") or payload.get("ts")
        if ts_val and "_source_timestamp" not in decision:
            decision["_source_timestamp"] = ts_val
        history.append(decision)
    return history


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return None


def plot_equity(history: List[Dict[str, Any]], output_dir: Path) -> None:
    if plt is None:
        print("[WARN] Matplotlib not available; using minimal placeholder for equity.")
        _write_fallback_png(output_dir, "equity.png")
        return
    if not history:
        _save_placeholder(output_dir, "equity.png", "Equity / Available margin")
        return
    equities: List[float] = []
    availables: List[float] = []
    timestamps: List[datetime] = []
    for entry in history:
        equity = entry.get("equity") or entry.get("equity_total")
        ts = entry.get("timestamp") or entry.get("time") or entry.get("ts")
        parsed_ts = _parse_timestamp(ts)
        if equity is None or parsed_ts is None:
            continue
        try:
            equities.append(float(equity))
        except Exception:
            continue
        timestamps.append(parsed_ts)
        available = entry.get("available") or entry.get("available_margin")
        try:
            availables.append(float(available) if available is not None else math.nan)
        except Exception:
            availables.append(math.nan)
    if not equities:
        _save_placeholder(output_dir, "equity.png", "Equity / Available margin")
        return
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        x_values = timestamps if timestamps else list(range(len(equities)))
        ax.plot(x_values, equities, label="Equity")
        if any(math.isfinite(x) for x in availables):
            ax.plot(x_values, availables, label="Available", linestyle="--")
        ax.set_title("Equity / Available Margin")
        ax.set_ylabel("USDT")
        if timestamps and mdates is not None:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_xlabel("Time")
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            fig.autofmt_xdate()
        else:
            ax.set_xlabel("Samples")
        ax.legend()
        fig.tight_layout()
        _save_plot_with_formats(output_dir, "equity")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot equity graph: {exc}", file=sys.stderr)
        return


def plot_pnl(
    history: List[Dict[str, Any]],
    output_dir: Path,
    fallback_history: List[Dict[str, Any]] | None = None,
) -> None:
    if plt is None:
        print("[WARN] Matplotlib not available; using minimal placeholder for PnL.")
        _write_fallback_png(output_dir, "pnl.png")
        return
    used_fallback = False
    if not history and fallback_history:
        history = fallback_history
        used_fallback = True
    if not history:
        _save_placeholder(output_dir, "pnl.png", "Closed / Unrealized PnL")
        return
    closed_pnls: List[float] = []
    unrealized: List[float] = []
    timestamps: List[datetime | None] = []
    for entry in history:
        closed = (
            entry.get("cycle_closed_pnl")
            or entry.get("closed_pnl")
            or entry.get("closedPnL")
            or (entry.get("pnl") or {}).get("cycle")
            or (entry.get("pnl") or {}).get("closed")
            or entry.get("realized")
            or entry.get("realized_pnl")
            or entry.get("realizedPnl")
        )
        unreal = (
            entry.get("unrealized")
            or entry.get("unrealized_pnl")
            or entry.get("unrealizedPnl")
            or (entry.get("pnl") or {}).get("unrealized")
        )
        ts = entry.get("timestamp") or entry.get("time") or entry.get("ts")
        parsed_ts = _parse_timestamp(ts)
        timestamps.append(parsed_ts)
        try:
            closed_pnls.append(float(closed) if closed is not None else math.nan)
        except Exception:
            closed_pnls.append(math.nan)
        try:
            unrealized.append(float(unreal) if unreal is not None else math.nan)
        except Exception:
            unrealized.append(math.nan)
    if not closed_pnls:
        _save_placeholder(output_dir, "pnl.png", "Closed / Unrealized PnL")
        return
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        valid_time = all(ts is not None for ts in timestamps) and mdates is not None
        if valid_time:
            x_values = [ts for ts in timestamps if ts is not None]
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.set_xlabel("Time")
            ax.grid(axis="x", linestyle=":", alpha=0.4)
        else:
            x_values = list(range(len(closed_pnls)))
            ax.set_xlabel("Samples")
        ax.plot(x_values, closed_pnls, label="Closed PnL")
        if any(math.isfinite(x) for x in unrealized):
            ax.plot(x_values, unrealized, label="Unrealized", linestyle="--")
        ax.set_title("PnL over time")
        ax.set_ylabel("USDT")
        ax.legend()
        fig.tight_layout()
        source_label = "results_state" if not used_fallback else "equity_history"
        if x_values:
            if valid_time:
                start = x_values[0].isoformat()
                end = x_values[-1].isoformat()
                print(f"[INFO] Closed PnL: {len(closed_pnls)} points ({source_label}, {start} -> {end})")
            else:
                print(f"[INFO] Closed PnL: {len(closed_pnls)} points ({source_label})")
        _save_plot_with_formats(output_dir, "pnl")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to plot PnL graph: {exc}", file=sys.stderr)
        return


def plot_signal_distribution(
    history: List[Dict[str, Any]],
    output_dir: Path,
    *,
    fallback_log: Path | None = None,
    source_label: str = "results_state",
) -> None:
    if plt is None:
        print("[WARN] Matplotlib not available; using minimal placeholder for signal distribution.")
        _write_fallback_png(output_dir, "signals.png")
        return
    source_used = source_label
    if not history and fallback_log:
        history = _load_signal_history_from_ai_log(fallback_log)
        source_used = "ai_decisions.log"
    if not history:
        _save_placeholder(output_dir, "signals.png", "Signal distribution")
        return
    actions = {}
    timestamps: List[datetime | None] = []
    for entry in history:
        action = (entry.get("action") or entry.get("summary") or "").lower()
        if not action:
            continue
        key = "open" if "open" in action else "close" if "close" in action else "skip"
        actions[key] = actions.get(key, 0) + 1
        ts_val = entry.get("timestamp") or entry.get("time") or entry.get("_source_timestamp")
        timestamps.append(_parse_timestamp(ts_val))
    if not actions:
        _save_placeholder(output_dir, "signals.png", "Signal distribution")
        return
    labels = list(actions.keys())
    sizes = [actions[label] for label in labels]
    try:
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Signal/Action Distribution")
        plt.axis("equal")
        plt.tight_layout()
        if timestamps:
            valid_ts = [ts for ts in timestamps if ts is not None]
            if valid_ts:
                start = min(valid_ts).isoformat()
                end = max(valid_ts).isoformat()
                print(f"[INFO] Signal distribution: {len(history)} entries ({source_used}, {start} -> {end})")
            else:
                print(f"[INFO] Signal distribution: {len(history)} entries ({source_used}, no timestamps)")
        else:
            print(f"[INFO] Signal distribution: {len(history)} entries ({source_used})")
        _save_plot_with_formats(output_dir, "signals")
        plt.close()
    except Exception as exc:
        print(f"[WARN] Failed to plot signal distribution: {exc}", file=sys.stderr)
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate diagnostic graphs for Bybit bot")
    parser.add_argument("--output", type=Path, default=Path("graphs"), help="Directory for generated plots")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("assets"),
        help="Directory that stores bot state files (equity_history/results_state). Defaults to ./assets",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Results state JSON path (defaults to <state-dir>/results_state.json)",
    )
    parser.add_argument(
        "--equity",
        type=Path,
        default=None,
        help="Equity history JSON path (defaults to <state-dir>/equity_history.json)",
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

        equity_path = args.equity or _find_state_path("equity_history.json")
        results_path = args.results or _find_state_path("results_state.json")
        print(f"[INFO] Equity source: {equity_path}")
        if results_path.exists():
            print(f"[INFO] Results source: {results_path}")
        else:
            print(f"[WARN] Results state file not found; falling back to equity history where needed.")

        equity_data = read_json(equity_path)
        if isinstance(equity_data, dict):
            equity_history = equity_data.get("history") or equity_data.get("entries") or []
        else:
            equity_history = equity_data or []
        plot_equity(equity_history, args.output)

        results_data = read_json(results_path)
        if isinstance(results_data, dict):
            results_history = results_data.get("history") or results_data.get("entries") or []
        else:
            results_history = results_data or []
        plot_pnl(results_history, args.output, fallback_history=equity_history)

        ai_log_path = next((d / "ai_decisions.log" for d in candidate_dirs if (d / "ai_decisions.log").exists()), state_dir / "ai_decisions.log")
        plot_signal_distribution(
            results_history,
            args.output,
            fallback_log=ai_log_path,
            source_label="results_state",
        )

        return 0
    except Exception as exc:  # pragma: no cover - ensures CLI is resilient
        import traceback

        print(f"[ERROR] Failed to generate graphs: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

