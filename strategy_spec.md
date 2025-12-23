## Strategy Spec (JSON, v1.1.1)

`strategy_spec.json` is the single source of truth for the manual (non-AI) strategy. JSON overrides `.env` for every trading knob; `.env` is reserved for secrets and telemetry only. If an env variable should be read, its name must appear in `context.env_vars`.

### Context
```jsonc
"context": {
  "universe": ["BTC/USDT", ...],            // up to 8 symbols; drives MANUAL_STRATEGY_SYMBOLS
  "timeframes": {                           // public context to fetch
    "primary": "30m",
    "secondary": "4h",
    "open_interest": "1h",
    "funding": "8h"
  },
  "news": { "positive": 0.55, "negative": -0.55, "neutral_band": 0.15 },
  "schedule_minutes": 10,
  "env_vars": ["RISK_PCT", "MIN_DYNAMIC_RISK_PCT", "MAX_DYNAMIC_RISK_PCT"], // optional overrides
  "universe_mode": {
    "source": "fixed",              // fixed | news
    "max_symbols": 8,               // cap, includes symbols with open positions
    "include_positions": true       // always add open-position symbols
  }
}
```

### Risk & Sizing
```jsonc
"risk": { "base_pct": 0.02, "min_pct": 0.01, "max_pct": 0.0375 },
"sizing": {
  "risk_multiplier": { "trend": 1.15, "counter": 0.55, "flat": 0.4 },
  "min_pct": 0.0025, "max_pct": 0.05
}
```
`risk` feeds bot-level CURRENT/MIN/MAX risk. `sizing` is used by the interpreter when turning regimes into notional percentages.

### Thresholds & Rules
```jsonc
"thresholds": {
  "atr_sigma_hot": 2.5, "atr_limit_multiplier": 1.7,
  "atr_range_ratio": 0.008, "atr_extreme_ratio": 0.025, "oi_change_pct": 0.012
},
"rules": {
  "trend": { "long": {...}, "short": {...} },
  "countertrend": { "long": {...}, "short": {...} },
  "flat": { "ema_spread_pct": 0.003, "rsi_band": [45, 55] }
}
```
These classify market regimes and gate entries for each side.

### Events
```jsonc
"events": {
  "limit_gap_pct": 0.002,
  "limit_offsets": { "buy": 0.998, "sell": 1.002 },
  "entry_ladder": [[0.6, 0.0], [0.4, 0.6]],   // share, atr_offset
  "tp_ladder": [[0.33, 1.2], [0.33, 2.0], [0.34, 3.0]], // share, atr_multiple
  "tp": { "atr_multiple": 2.0, "rsi_long": 70, "rsi_short": 30 },
  "hedge": { "funding_flip": 0.0001, "size_pct": 0.5 },
  "modify_position": { "rsi_long": [40, 65], "rsi_short": [35, 60], "confidence": 0.58, "scale": 0.5 }
}
```
Interpreter emits one of: `open_market`, `open_limit`, `hedge_open`, `place_limit_TP`, `modify_limit`, `cancel_limit`, `modify_position`, `close_position`, `skip`. `signal_intent_mapper.py` turns these into concrete orders/decisions.

### Account & Providers
```jsonc
"account": { "fields": ["equity", "available_margin", "positions", "open_orders"], "log_tag": "[ACCOUNT]" },
"providers": { "news": ["cryptopanic", "rss", "coindesk"] }   // open sources only; tokens live in env
```
`account` is not user data storage; it only tells `account_context.py` which private fields to fetch/log per cycle. `providers.news` configures the news source and which env var holds the token.

### Execution
```jsonc
"execution": {
  "manual_only": true,
  "leverage": 10,
  "order_margin_utilization": 0.95,
  "sl_atr": 1.6,
  "tp_atr": 2.8,
  "trailing_atr_mult": 1.0,
  "max_open_positions": 4,
  "max_positions_per_base": 1,
  "default_next_run_minutes": 10,
  "max_retries": 1,
  "retry_delay_sec": 2,
  "limit_to_market_seconds": 10,
  "fallbacks": { "market_on_timeout": true, "cancel_on_conflict": true }
}
```
These values configure order sizing, leverage, protection parameters, retry/fallback behaviour, and disable the legacy AI/offline planner (`manual_only=true`).

### Flow & Priority
1) `strategy_context.py` builds public context for `context.universe`, logging `[MANUAL][CONTEXT] ...`.  
2) `account_context.py` logs private fields listed in `account.fields` with `[ACCOUNT] ...`.  
3) `strategy.py` evaluates rules and emits `StrategyEvent`s for each symbol.  
4) `signal_intent_mapper.py` applies `execution` settings to produce actionable decisions.  
5) `bybitbot_impl.py` runs the decisions; all per-symbol logs are tagged `[MANUAL]`.

JSON beats `.env` for every trading parameter. Only secrets (API keys, tokens) and the env vars explicitly listed under `context.env_vars` are read from `.env`.
