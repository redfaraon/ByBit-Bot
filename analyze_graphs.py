#!/usr/bin/env python3

"""Generate equity, PnL, and signal distribution plots from Bybit bot logs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    if not history:
        print("[INFO] No equity history available; skipping equity plot.")
        return
    equities = []
    availables = []
    for entry in history:
        equity = entry.get("equity") or entry.get("equity_total")
        available = entry.get("available") or entry.get("available_margin")
        if equity is None:
            continue
        try:
            equities.append(float(equity))
        except Exception:
            continue
        try:
            availables.append(float(available) if available is not None else math.nan)
        except Exception:
            availables.append(math.nan)
    if not equities:
        print("[INFO] Equity history empty after filtering.")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(equities, label="Equity")
    if any(math.isfinite(x) for x in availables):
        plt.plot(availables, label="Available", linestyle="--")
    plt.title("Equity / Available Margin")
    plt.xlabel("Samples")
    plt.ylabel("USDT")
    plt.legend()
    output_path = output_dir / "equity.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved equity graph to {output_path}")


def plot_pnl(history: List[Dict[str, Any]], output_dir: Path) -> None:
    if not history:
        print("[INFO] No results history available; skipping PnL plot.")
        return
    closed_pnls = []
    unrealized = []
    for entry in history:
        closed = entry.get("closed_pnl") or entry.get("closedPnL")
        unreal = entry.get("unrealized")
        try:
            closed_pnls.append(float(closed) if closed is not None else math.nan)
        except Exception:
            closed_pnls.append(math.nan)
        try:
            unrealized.append(float(unreal) if unreal is not None else math.nan)
        except Exception:
            unrealized.append(math.nan)
    if not closed_pnls:
        print("[INFO] PnL history empty.")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(closed_pnls, label="Closed PnL")
    if any(math.isfinite(x) for x in unrealized):
        plt.plot(unrealized, label="Unrealized", linestyle="--")
    plt.title("PnL over time")
    plt.xlabel("Samples")
    plt.ylabel("USDT")
    plt.legend()
    output_path = output_dir / "pnl.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved PnL graph to {output_path}")


def plot_signal_distribution(history: List[Dict[str, Any]], output_dir: Path) -> None:
    if not history:
        print("[INFO] No signal history available; skipping distribution plot.")
        return
    actions = {}
    for entry in history:
        action = (entry.get("action") or entry.get("summary") or "").lower()
        if not action:
            continue
        key = "open" if "open" in action else "close" if "close" in action else "skip"
        actions[key] = actions.get(key, 0) + 1
    if not actions:
        print("[INFO] No actions recorded; skipping distribution plot.")
        return
    labels = list(actions.keys())
    sizes = [actions[label] for label in labels]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Signal/Action Distribution")
    plt.axis("equal")
    output_path = output_dir / "signals.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved signal distribution graph to {output_path}")


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

    args.output.mkdir(parents=True, exist_ok=True)
    state_dir = args.state_dir.expanduser()
    equity_path = args.equity or (state_dir / "equity_history.json")
    results_path = args.results or (state_dir / "results_state.json")

    equity_data = read_json(equity_path)
    if isinstance(equity_data, dict):
        history = equity_data.get("history") or equity_data.get("entries") or []
    else:
        history = equity_data or []
    plot_equity(history, args.output)

    results_data = read_json(results_path)
    if isinstance(results_data, dict):
        history = results_data.get("history") or results_data.get("entries") or []
    else:
        history = results_data or []
    plot_pnl(history, args.output)
    plot_signal_distribution(history, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
