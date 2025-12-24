from __future__ import annotations
MODULE_VERSION = "1.3.10"


from typing import Any, Iterable, Mapping, Sequence


def build_universe(
    context_spec: dict,
    open_position_symbols: Iterable[str] | None = None,
    *,
    news_digest: Mapping[str, Any] | None = None,
) -> list[str]:
    universe, _meta = build_universe_with_metadata(
        context_spec,
        open_position_symbols,
        news_digest=news_digest,
    )
    return universe


def build_universe_with_metadata(
    context_spec: dict,
    open_position_symbols: Iterable[str] | None = None,
    *,
    news_digest: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """
    Build the trading universe based on JSON context settings and open positions.
    - source: 'fixed' | 'news'
    - max_symbols: cap applied after adding open position symbols
    - include_positions: always include open-position symbols (even above cap)
    Returns: (universe_list, metadata)
    """
    ctx = context_spec or {}
    mode = (ctx.get("universe_mode") or {}) if isinstance(ctx.get("universe_mode"), dict) else {}
    source = str(mode.get("source") or "fixed").strip().lower()
    max_symbols = max(1, int(mode.get("max_symbols") or 8))
    include_positions = bool(mode.get("include_positions", True))

    fixed_list = [str(sym).strip().upper() for sym in (ctx.get("universe") or []) if sym]

    def _normalize(symbol: str | None) -> str | None:
        if not symbol:
            return None
        normalized = str(symbol).strip().upper()
        return normalized or None

    def _try_add(target: list[str], normalized: str | None, seen: set[str]) -> bool:
        if not normalized or normalized in seen:
            return False
        target.append(normalized)
        seen.add(normalized)
        return True

    news_priority = mode.get("news_priority") if isinstance(mode.get("news_priority"), dict) else {}
    min_news_items = max(0, int(news_priority.get("min_items") or 1))

    meta: dict[str, Any] = {
        "requested_source": source,
        "max_symbols": max_symbols,
        "include_positions": include_positions,
        "min_news_items": min_news_items,
        "fixed_count": len(fixed_list),
        "news_digest_present": bool(news_digest),
        "news_symbols_total": 0,
        "news_symbols_used": 0,
        "fixed_symbols_used": 0,
        "position_symbols_used": 0,
    }

    seen: set[str] = set()
    result: list[str] = []

    # Always include open positions first; cap is applied after that.
    if include_positions and open_position_symbols:
        for sym in open_position_symbols:
            if _try_add(result, _normalize(sym), seen):
                meta["position_symbols_used"] += 1

    cap = max(max_symbols, len(result))

    if source == "news" and news_digest:
        entries: list[tuple[str, int]] = []
        for sym, payload in news_digest.items():
            normalized = _normalize(sym)
            if not normalized:
                continue
            items = payload.get("items") if isinstance(payload, dict) else None
            item_count = len(items) if isinstance(items, Sequence) else 0
            entries.append((normalized, item_count))
        meta["news_symbols_total"] = len(entries)
        entries.sort(key=lambda entry: (-entry[1], entry[0]))
        for normalized, count in entries:
            if len(result) >= cap:
                break
            if count < min_news_items:
                continue
            if _try_add(result, normalized, seen):
                meta["news_symbols_used"] += 1

    if len(result) < cap:
        for sym in fixed_list:
            if len(result) >= cap:
                break
            if _try_add(result, _normalize(sym), seen):
                meta["fixed_symbols_used"] += 1

    meta["final_size"] = len(result)
    meta["used_news"] = meta["news_symbols_used"] > 0
    meta["used_fixed"] = meta["fixed_symbols_used"] > 0
    meta["used_positions"] = meta["position_symbols_used"] > 0
    meta["result"] = result[:]

    return result, meta
