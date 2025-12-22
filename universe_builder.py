from __future__ import annotations

from typing import Iterable, Sequence


def build_universe(context_spec: dict, open_position_symbols: Iterable[str] | None = None) -> list[str]:
    """
    Build the trading universe based on JSON context settings and open positions.
    - source: 'fixed' | 'news' (news currently falls back to fixed ordering)
    - max_symbols: cap applied after adding open position symbols
    - include_positions: always include open-position symbols
    """
    ctx = context_spec or {}
    mode = (ctx.get("universe_mode") or {}) if isinstance(ctx.get("universe_mode"), dict) else {}
    source = str(mode.get("source") or "fixed").strip().lower()
    max_symbols = int(mode.get("max_symbols") or 8)
    include_positions = bool(mode.get("include_positions", True))

    fixed_list = [str(sym).strip().upper() for sym in (ctx.get("universe") or []) if sym]
    base_list = fixed_list[:]
    if source == "news":
        # placeholder: news-driven selection can reorder later; keep fixed list as seed
        base_list = fixed_list[:]

    seen: set[str] = set()
    result: list[str] = []
    for sym in base_list:
        if sym and sym not in seen:
            result.append(sym)
            seen.add(sym)
            if len(result) >= max_symbols:
                break

    if include_positions and open_position_symbols:
        for sym in open_position_symbols:
            norm = str(sym).strip().upper()
            if norm not in seen:
                result.append(norm)
                seen.add(norm)
            if len(result) >= max_symbols:
                break
    return result
