## Project Overview (ByBit Bot)

### Modules & Roles
- `bybitbot.py` — launcher: pulls latest HEAD, runs `bybitbot_impl`, and on failures falls back to stable/tag/commit via full-repo `git worktree` snapshots (so all modules stay consistent).
- `bybitbot_impl.py` — orchestrator: loads settings, fetches public/private context, routes manual strategy decisions to execution, delegates execution/protection/trailing to module engines, Telegram handlers, persistence (state/log files).
- `universe_builder.py` — builds the manual universe from JSON (`context.universe_mode`) and injects symbols with open positions.
- `strategy.py` — JSON-driven manual strategy (no AI): builds `StrategyContext`, detects regimes, emits `StrategyEvent`s, holds sizing/thresholds from `strategy_spec.json`.
- `strategy_context.py` — builds indicator/news/funding/OI context for the manual strategy.
- `trading_context.py` — per-symbol context builder (bars/indicators/news/open interest/funding) used by the interpreter.
- `strategy_executor.py` — maps `StrategyEvent` → executable decision dict (orders, side, type, notional_pct).
- `account_context.py` — snapshots private account state (equity, margin, positions, open orders) and logs `[ACCOUNT] ...`.
- `execution_engine.py` — order execution core (extra orders, reduce-only handling, logging) used by the orchestrator.
- `protection_engine.py` — protection/trailing engine (SL/TP/trailing cleanup + logging) used by wrappers and the orchestrator.
- `order_utils.py` — shared order normalization helpers (order types, trigger directions, sizing).
- `order_cleanup.py` — wrappers for limit/stop cleanup (delegates to protection engine handlers).
- `protection_utils.py` — wrapper for protection/trailing application (delegates to protection engine).
- `trailing_utils.py` — wrapper for trailing/protection step (delegates to protection engine).
- `strategy_spec.json` / `strategy_spec.md` — canonical strategy/config contract (universe, risk, execution, providers, events).
- `assets/bybit.log` — main combined log (stdout/stderr mirrored), includes `[MANUAL]`, `[ACCOUNT]`, protection/trade/order messages.

### Key Constants (set in impl via JSON where provided)
- `BOT_VERSION` — release marker.
- `LEVERAGE`, `ORDER_MARGIN_UTILIZATION`, `SL_ATR`, `TP_ATR`, `TRAILING_ATR_MULT` — execution knobs (from `execution` block).
- `RISK_PCT`, `MIN_DYNAMIC_RISK_PCT`, `MAX_DYNAMIC_RISK_PCT` — risk window (from `risk` block, env only if whitelisted in `context.env_vars`).
- `MAX_OPEN_POSITIONS`, `MAX_POSITIONS_PER_BASE`, `DEFAULT_NEXT_RUN_MINUTES` — portfolio pacing/limits (from `execution`).
- `NEWS_PROVIDER`, `NEWS_API_TOKEN` — news source configuration (from `providers.news` + env token).
- `ENTRY_LADDER_SCHEME`, `PARTIAL_TP_SCHEME` — laddering ratios for entries/take-profits (env fallback if not overridden elsewhere).

### Core Functions (selected)
- `run_cycle()` (impl) — orchestrates a full trading cycle: context fetch → manual decision → execution → protection/cleanup → scheduling.
- `strategy.get_signal_without_ai(ctx)` — regime detection + event selection (manual-only).
- `strategy_executor.apply_event(event, ctx)` — produce executable decision dict.
- `strategy_context.build_manual_strategy_context(...)` — assemble per-symbol manual context.
- `execution_engine.execute_extra_orders(...)` — applies extra orders from decisions.
- `protection_engine.ensure_position_protection(...)` — core SL/TP/trailing enforcement.
- `protection_utils.ensure_protection(...)` / `trailing_utils.apply_trailing(...)` — delegate to protection engine logic.
- `order_cleanup.cleanup_excess_non_reduce_limits(...)` / `cleanup_redundant_stops(...)` — delegate to protection engine cleanup logic.
- Telegram handlers: `handle_telegram_command`, `_handle_schedule_command`, `_handle_logs_command`, `_handle_tokens_command`, `_handle_bybit_key_command`, `_handle_add_user_command`, `_handle_config_command`, `_handle_sandbox_command`.

### Logging Conventions
- `[MANUAL][CONTEXT]` — per-symbol public snapshot (price/trend/news/OI/risk).
- `[MANUAL][SIGNAL]` — interpreter decision summary before execution.
- `[MANUAL][EXEC]` — final action/side/type/reason.
- `[ACCOUNT]` — equity/margin/positions/open_orders snapshot per cycle.
- Other notable tags: protection/trailing warnings, cleanup summaries, margin/risk messages; AI/offline tags are suppressed in manual-only mode.
- Telegram mirrors key events (protection changes, order placements/cancellations, warnings) and command replies; `send_tg_decision` mirrors decision confidence.

### Telegram Commands (handled in impl)
- `/start`, `/help` — help text.
- `/status`, `/positions`, `/risk`, `/version` — runtime summaries.
- `/logs` (`logtail`, `log`) — fetch log snippets.
- `/logmode` — adjust forwarding verbosity.
- `/schedule` — set/cancel next run (minutes, absolute time, now).
- `/tokens` — token budget info.
- `/bybitkey` — supply/clear API keys (owner/DM).
- `/adduser`, `/config`, `/sandbox` — multi-user and sandbox controls.
- `/ai payload` — inspect last AI payload (legacy; manual flow suppresses AI).

### Data & State Files
- `assets/bybit.log` — main rotating log (stdout/stderr tee).
- `assets/error.log` — error mirror (if enabled).
- `cycle_state.json`, `results_state.json`, `equity_history.json`, `fallback_history.json`, `release_state.json` — runtime/state snapshots.
- `bybit_credentials.json` — stored API keys per user (secrets).
- `users/*.json` — multi-user profiles (user IDs, preferences, secrets paths).
- `strategy_spec.json` — active strategy config (universe, risk, execution, providers, events).

### Laddering / Orders
- Opens: ladder driven by `events.entry_ladder` in `strategy_spec.json` (share, ATR offset). Defaults `(0.6@0 ATR, 0.4@0.6 ATR)`. Limit drift refresh uses `events.limit_gap_pct`.
- Takes/scale-outs: `events.tp_ladder` defines reduce-only TP ladder (share, ATR multiple). Protection refresh enforces SL/TP/trailing per `execution` ATR multipliers.
- Order cleanup: non-reduce entry limits pruned to `MAX_NON_REDUCE_LIMITS_PER_SIDE`; redundant reduce-only stops trimmed after fresh protection is placed.

### Flow (impl orchestrator, manual-only)
1. `bybitbot.py` updates HEAD and manages full-repo fallback snapshots (stable/tag/commit) via `git worktree`; on new HEAD it exits backup mode and resumes the latest code.
2. Collect user account context (equity/margin/positions/orders) via `account_context`; log `[ACCOUNT] ...`.
3. Build universe via `universe_builder` (fixed/news per JSON, always includes open positions).
4. For each symbol:
   - Build trading context (bars/indicators/news/funding/oi) via `trading_context`/`strategy_context`.
   - Run strategy → signals/events; log `[MANUAL][SIGNAL]`.
   - Execute via `strategy_executor` (order/place/cancel/modify) and log `[MANUAL][EXEC]`.
   - Cleanup stale limits, apply trailing, enforce protection; if protection fails to place, close the position and surface an error.
5. Summarize cycle, compute next start time, and wait until the scheduled run.
