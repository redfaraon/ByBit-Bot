## JSON Strategy Specification

This document describes the contract for `strategy_spec.json`, the file that drives the rule‑based/manual trading flow.

### 1. Context Block

```jsonc
"context": {
  "universe": ["BTC/USDT", ...],        // up to 8 symbols
  "timeframes": {
    "primary": "30m",
    "secondary": "4h",
    "open_interest": "1h",
    "funding": "8h"
  },
  "news": {
    "positive": 0.55,                   // >= threshold → positive bias
    "negative": -0.55,                  // <= threshold → negative bias
    "neutral_band": 0.15                // |score| ≤ band → neutral
  },
  "schedule_minutes": 10,               // recommended cycle interval
  "env_vars": ["RISK_PCT", ...]         // allowed env inputs (no API keys)
}
```

### 2. Thresholds

```jsonc
"thresholds": {
  "atr_sigma_hot": 2.5,                 // skip opens above this z-score
  "atr_limit_multiplier": 1.7,          // ATR hotness for switching to limit entries
  "atr_range_ratio": 0.008,             // EMA spread vs price for flat regime
  "atr_extreme_ratio": 0.025,           // reduces size when ATR/price is high
  "oi_change_pct": 0.012                // OI % change to label up/down trend
}
```

### 3. Sizing

```jsonc
"sizing": {
  "risk_multiplier": {
    "trend": 1.15,
    "counter": 0.55,
    "flat": 0.4
  },
  "min_pct": 0.0025,
  "max_pct": 0.05
}
```

Manual decisions multiply the current risk pct by the regime multiplier, clamp it to `[min_pct, max_pct]`, then optionally scale (`events.modify_position.scale`, etc.).

### 4. Rules

```jsonc
"rules": {
  "trend": {
    "long": {
      "rsi_max": 65,
      "funding_min": -0.0002,
      "news_block": ["negative"],
      "require_oi_up": true,
      "confidence": { "market": 0.82, "limit": 0.78 }
    },
    "short": { ... }
  },
  "countertrend": {
    "long": { "rsi_max": 30, "news_block": ["negative"], "confidence": 0.72 },
    "short": { "rsi_min": 70, "news_block": ["positive"], "confidence": 0.72 }
  },
  "flat": {
    "ema_spread_pct": 0.003,
    "rsi_band": [45, 55]
  }
}
```

Trend rules gate long/short entries. Countertrend rules describe RSI extremes and news vetoes. Flat rules decide whether the regime is “range only”.

### 5. Events

```jsonc
"events": {
  "limit_gap_pct": 0.002,
  "limit_offsets": { "buy": 0.998, "sell": 1.002 },
  "tp": { "atr_multiple": 2.0, "rsi_long": 70, "rsi_short": 30 },
  "hedge": { "funding_flip": 0.0001, "size_pct": 0.5 },
  "modify_position": {
    "rsi_long": [40, 65],
    "rsi_short": [35, 60],
    "confidence": 0.58,
    "scale": 0.5
  }
}
```

- `limit_gap_pct`: percentage move away from a pending limit order that triggers `modify_limit`.
- `limit_offsets`: default limit price offsets for new entries.
- `tp`: ATR multiplier + RSI triggers for placing reduce‑only take profits.
- `hedge`: funding flip threshold and hedge size as a fraction of notional.
- `modify_position`: RSI ranges and sizing for scale‑ins when trend entries stay valid.

### 6. Execution Flow

1. **Context builder** (see `strategy_context.py`) collects bars/indicators/news/OI/funding according to the spec.
2. **Interpreter** (`strategy.py`) loads `strategy_spec.json`, evaluates the rules for each symbol, and emits `StrategyEvent` objects.
3. **Adapter** (`apply_event`) converts events into decisions compatible with `run_cycle`.

To extend behaviour, adjust `strategy_spec.json` fields; no code changes are required unless new event types are introduced.***
