## JSON Strategy Specification

`strategy_spec.json` is the single source of truth for every manual decision. JSON overrides `.env` for all trading parameters (env keeps only secrets). If an env variable must remain configurable it has to be listed under `context.env_vars`; otherwise it is ignored.

### 1. Context

```jsonc
"context": {
  "universe": ["BTC/USDT", ...],      // up to 8 symbols, drives MANUAL_STRATEGY_SYMBOLS
  "timeframes": {                     // which slices the public context loader must fetch
    "primary": "30m",
    "secondary": "4h",
    "open_interest": "1h",
    "funding": "8h"
  },
  "news": {                          // CryptoPanic/RSS sentiment thresholds
    "positive": 0.55,
    "negative": -0.55,
    "neutral_band": 0.15
  },
  "schedule_minutes": 10,            // recommended cycle interval
  "env_vars": ["RISK_PCT", ...]      // env overrides that stay allowed (secrets excluded)
}
```

### 2. Risk & Sizing

```jsonc
"risk": {
  "base_pct": 0.02,
  "min_pct": 0.01,
  "max_pct": 0.0375
},
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

`risk` feeds the bot-level risk window (CURRENT/MIN/MAX). `sizing` is used by the interpreter when it converts a regime into notional percentages.

### 3. Thresholds & Rules

```jsonc
"thresholds": {
  "atr_sigma_hot": 2.5,
  "atr_limit_multiplier": 1.7,
  "atr_range_ratio": 0.008,
  "atr_extreme_ratio": 0.025,
  "oi_change_pct": 0.012
},
"rules": {
  "trend": {
    "long": {
      "rsi_max": 65,
      "funding_min": -0.0002,
      "news_block": ["negative"],
      "require_oi_up": true,
      "confidence": { "market": 0.82, "limit": 0.78 }
    },
    "short": { "...": "..." }
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

The indicator thresholds classify the market into trend/countertrend/flat regimes and gate entries.

### 4. Events

```jsonc
"events": {
  "limit_gap_pct": 0.002,                   // refresh pending limits when drift >0.2%
  "limit_offsets": { "buy": 0.998, "sell": 1.002 },
  "tp": {                                   // TP/scale-out rules
    "atr_multiple": 2.0,
    "rsi_long": 70,
    "rsi_short": 30
  },
  "hedge": {                                // hedge_open trigger and sizing
    "funding_flip": 0.0001,
    "size_pct": 0.5
  },
  "modify_position": {                      // scale in/out rules
    "rsi_long": [40, 65],
    "rsi_short": [35, 60],
    "confidence": 0.58,
    "scale": 0.5
  }
}
```

The interpreter emits one of the documented events (open_market, open_limit, hedge_open, place_limit_TP, modify_limit, cancel_limit, modify_position, close_position, skip). `strategy_executor.py` converts an event into explicit actions/orders while respecting `execution`.

### 5. Account & Context Logging

```jsonc
"account": {
  "fields": ["equity", "available_margin", "positions", "open_orders"],
  "log_tag": "[ACCOUNT]"
}
```

`account_context.py` must collect the listed private fields from Bybit, build a snapshot, and log one `[ACCOUNT] ...` line per cycle into `assets/bybit.log`. Public-market context (candles, indicators, news, funding, OI) is logged via `[MANUAL][CONTEXT]`.

### 6. Execution Parameters

```jsonc
"execution": {
  "manual_only": true,
  "max_retries": 1,
  "retry_delay_sec": 2,
  "limit_to_market_seconds": 10,
  "fallbacks": {
    "market_on_timeout": true,
    "cancel_on_conflict": true
  }
}
```

`strategy_executor.py` reads this section while preparing orders (e.g., deciding if a pending limit should flip to market). `manual_only=true` disables the legacy AI/offline planner so every symbol is handled by the JSON interpreter.

### 7. Flow Summary

1. **Public context** — `strategy_context.py` builds the indicator snapshot for every symbol in `context.universe` and logs `[MANUAL][CONTEXT]`.
2. **Account context** — `account_context.py` logs balances/positions/orders using `account.fields`.
3. **Interpreter** — `strategy.py` parses the JSON, evaluates rules, and emits `StrategyEvent` results along with confidence/size metadata.
4. **Executor** — `strategy_executor.py` applies `execution` settings and returns the final `decision` dict. `bybitbot_impl.py` executes it and logs `[MANUAL][EXEC]` entries.

Whenever JSON and `.env` disagree, JSON wins. Only the env variables listed under `context.env_vars` are even read; every other trading parameter (universe, sizing, risk, events, etc.) must come from `strategy_spec.json`. Credentials (API keys, Telegram tokens, etc.) stay in `.env`.
