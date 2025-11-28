#!/usr/bin/env python3

"""Generate equity, PnL, and signal distribution plots from Bybit bot logs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List
import base64

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import]
except Exception as exc:  # pragma: no cover - protects from missing dependencies
    plt = None  # type: ignore[assignment]
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
        plt.savefig(output_path);\n        try:\n            plt.savefig(str(output_path).replace('.png','.jpg'), format='jpeg')\n        except Exception:\n            pass
        plt.close()
        print(f"[INFO] Saved placeholder graph to {output_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save placeholder {filename}: {exc}", file=sys.stderr)


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
        _save_placeholder(output_dir, "equity.png", "Equity / Available margin")
        return
    try:
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
        plt.savefig(output_path);\n        try:\n            plt.savefig(str(output_path).replace('.png','.jpg'), format='jpeg')\n        except Exception:\n            pass
        plt.close()
        print(f"[INFO] Saved equity graph to {output_path}")
    except Exception as exc:
        print(f"[WARN] Failed to plot equity graph: {exc}", file=sys.stderr)
        return


def plot_pnl(history: List[Dict[str, Any]], output_dir: Path) -> None:
    if plt is None:
        print("[WARN] Matplotlib not available; using minimal placeholder for PnL.")
        _write_fallback_png(output_dir, "pnl.png")
        return
    if not history:
        _save_placeholder(output_dir, "pnl.png", "Closed / Unrealized PnL")
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
        _save_placeholder(output_dir, "pnl.png", "Closed / Unrealized PnL")
        return
    try:
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
        plt.savefig(output_path);\n        try:\n            plt.savefig(str(output_path).replace('.png','.jpg'), format='jpeg')\n        except Exception:\n            pass
        plt.close()
        print(f"[INFO] Saved PnL graph to {output_path}")
    except Exception as exc:
        print(f"[WARN] Failed to plot PnL graph: {exc}", file=sys.stderr)
        return


def plot_signal_distribution(history: List[Dict[str, Any]], output_dir: Path) -> None:
    if plt is None:
        print("[WARN] Matplotlib not available; using minimal placeholder for signal distribution.")
        _write_fallback_png(output_dir, "signals.png")
        return
    if not history:
        _save_placeholder(output_dir, "signals.png", "Signal distribution")
        return
    actions = {}
    for entry in history:
        action = (entry.get("action") or entry.get("summary") or "").lower()
        if not action:
            continue
        key = "open" if "open" in action else "close" if "close" in action else "skip"
        actions[key] = actions.get(key, 0) + 1
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
        output_path = output_dir / "signals.png"
        plt.tight_layout()
        plt.savefig(output_path);\n        try:\n            plt.savefig(str(output_path).replace('.png','.jpg'), format='jpeg')\n        except Exception:\n            pass
        plt.close()
        print(f"[INFO] Saved signal distribution graph to {output_path}")
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
    # Resolve state dir: prefer provided; otherwise auto-detect nearby files
    state_dir = args.state_dir.expanduser()
    candidate_dirs = [state_dir, script_dir, script_dir / "assets"]
    def _find_state_path(name: str) -> Path:
        # If explicit path is given via args, honor it
        return next((d / name for d in candidate_dirs if (d / name).exists()), state_dir / name)
    equity_path = args.equity or _find_state_path("equity_history.json")
    results_path = args.results or _find_state_path("results_state.json")

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
    except Exception as exc:  # pragma: no cover - ensures CLI is resilient
        import traceback

        print(f"[ERROR] Failed to generate graphs: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

