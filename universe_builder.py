from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def build_universe(
    context_spec: dict,
    open_position_symbols: Iterable[str] | None = None,
    *,
    news_digest: Mapping[str, Any] | None = None,
) -> list[str]:
    """
    Build the trading universe based on JSON context settings and open positions.
    - source: 'fixed' | 'news' (news currently falls back to fixed ordering)
    - max_symbols: cap applied after adding open position symbols
    - include_positions: always include open-position symbols
    """
    ctx = context_spec or {}
    mode = (ctx.get("universe_mode") or {}) if isinstance(ctx.get("universe_mode"), dict) else {}
    source = str(mode.get("source") or "fixed").strip().lower()
    max_symbols = max(1, int(mode.get("max_symbols") or 8))
    include_positions = bool(mode.get("include_positions", True))

    fixed_list = [str(sym).strip().upper() for sym in (ctx.get("universe") or []) if sym]
    news_priority = mode.get("news_priority") if isinstance(mode.get("news_priority"), dict) else {}
    min_news_items = max(0, int(news_priority.get("min_items") or 1))

    seen: set[str] = set()
    result: list[str] = []
    if source == "news" and news_digest:
        entries: list[tuple[str, int]] = []
        for sym, payload in news_digest.items():
            normalized = str(sym).strip().upper()
            if not normalized:
                continue
            items = payload.get("items") if isinstance(payload, dict) else None
            item_count = len(items) if isinstance(items, Sequence) else 0
            entries.append((normalized, item_count))
        entries.sort(key=lambda entry: (-entry[1], entry[0]))
        for normalized, count in entries:
            if count < min_news_items:
                continue
            if normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
            if len(result) >= max_symbols:
                break

    for sym in fixed_list:
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
