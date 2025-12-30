# Quick Start

1. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip -y
   python3 -m pip --version
   python3 -m pip install --upgrade pip
   python3 -m pip install --user -U ccxt pandas requests colorama "openai>=1.0.0" python-dotenv pyyaml matplotlib
   ```

2. Git auth (server/systemd)

   - For a **public** repository, `git fetch/pull` over HTTPS should work without any credentials.
   - Avoid `git config --global credential.helper store` on servers: a stale/invalid credential can make `git fetch` prompt for username/password, and `systemd` has no TTY (auto-pull will fail after reboot).
   - If you still need non-interactive auth (e.g., private fork): use an SSH key or set `BYBITBOT_GIT_TOKEN` / `GITHUB_TOKEN` in `.env`.

3. Export environment variables (example):
   ```bash
   export BYBIT_API_KEY=your_api_key
   export BYBIT_API_SECRET=your_api_secret
   export TELEGRAM_BOT_TOKEN=your_tg_token
   export TELEGRAM_CHAT_ID=your_chat_id
   export OPENAI_API_KEY=your_openai_key
   ```

   Optional Telegram controls:
   ```bash
   export TELEGRAM_FORWARD_LOGS=1              # enable buffered log forwarding
   export TELEGRAM_LOG_BATCH_SIZE=12          # max log lines grouped into one TG message
   export TELEGRAM_LOG_FLUSH_INTERVAL=5       # seconds before flushing a smaller batch
   export TELEGRAM_LOG_RATE_LIMIT=15         # max log messages per rate window (default 18)
   export TELEGRAM_LOG_RATE_WINDOW=60        # seconds tracked by the log rate limiter
   export TELEGRAM_LOG_THREAD_ID=12345        # topic/thread for log batches
   export TELEGRAM_WEBHOOK_URL=https://...    # set webhook endpoint (leave empty to disable)
   export TELEGRAM_WEBHOOK_HOST=0.0.0.0       # local webhook bind host
   export TELEGRAM_WEBHOOK_PORT=8082          # local webhook port
   export TELEGRAM_WEBHOOK_PATH=/telegram     # webhook path prefix
   export TELEGRAM_WEBHOOK_SECRET=secret123   # optional Telegram secret token
   export TELEGRAM_ALLOWED_CHAT_IDS=-1001234567890,-1005678901234
   export TELEGRAM_COMMANDS="status:������� ������;help:�������"  # slash-command overrides
   export NEWS_PROVIDER=hybrid                   # news sources: hybrid (default), cryptocompare, rss
   ```
## AI model configuration

- Set `OPENAI_API_KEY` to your API key (shared across all traders unless overridden per user).
- `OPENAI_MODEL_PRIMARY` � default `gpt-4.1-mini` (used for heavy planning).
- `OPENAI_MODEL_CHEAP` � default `gpt-4o-mini` (used once `/tokens` shows usage above `OPENAI_MODEL_CHEAP_THRESHOLD`, default 5 symbols).
- `AI_SUPPORT_MODEL` � optional override for `/support` replies (falls back to the primary trading model).
- `/tokens` displays current budget, model switches, and can be used to verify limits.

### Telegram diagnostics

- `/ai payload [context]` � dumps the last OpenAI request/response snapshot (pass `universe` or `trade` to focus on a stage; without arguments the latest exchange is shown).
- `/logs [count|SYMBOL window]` - show logs; e.g. `/logs 30` or `/logs BTC 60` for the last hour.

### Telegram log mirroring

Set `TELEGRAM_FORWARD_LOGS=1` to mirror console logs into Telegram. The bot batches entries (`TELEGRAM_LOG_BATCH_SIZE` / `TELEGRAM_LOG_FLUSH_INTERVAL`) and respects a rate window (`TELEGRAM_LOG_RATE_LIMIT` / `TELEGRAM_LOG_RATE_WINDOW`) so Telegram never returns 429 during bursts.


   Dynamic trailing-stop tuning:
   ```bash
   export TRAILING_DYNAMIC_TRIGGER_ATR=1.4    # ATR distance before tightening trailing stop
   export TRAILING_DYNAMIC_FACTOR=0.65        # ATR multiplier for tightened trailing stop
   export TRAILING_DYNAMIC_MIN_ATR=0.35       # floor ATR multiplier when tightening
   ```

4. Run the bot:
   ```bash
   python bybitbot.py
   ```

## Running as a systemd service (recommended)

If you're deploying on a Linux server (Ubuntu/Debian), run the bot under `systemd` so it autostarts on reboot and restarts on failures.

1. Create a log directory:
   ```bash
   sudo mkdir -p /var/log/bybitbot
   ```

2. Create a unit file (example: `/etc/systemd/system/bybitbot.service`):
   ```ini
   [Unit]
   Description=ByBit Bot
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   WorkingDirectory=/root/bot/ByBit-Bot
   Environment=PY_COLORS=1
   Environment=FORCE_COLOR=1
   ExecStart=/usr/bin/python3 /root/bot/ByBit-Bot/bybitbot.py
   Restart=always
   RestartSec=5
   StandardOutput=append:/var/log/bybitbot/console.log
   StandardError=append:/var/log/bybitbot/console.log

   [Install]
   WantedBy=multi-user.target
   ```

   If the repository is private, ensure `git fetch` can run non-interactively (otherwise auto-pull will fail after reboot). Recommended: use an SSH deploy key and set `origin` to `git@github.com:redfaraon/ByBit-Bot.git`. Alternative: set `BYBITBOT_GIT_TOKEN` (or `GITHUB_TOKEN`) in `.env` so the bot can authenticate during `git fetch`.

3. Enable and start it:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now bybitbot
   ```

4. Useful commands:
    ```bash
    sudo systemctl status bybitbot --no-pager
    sudo systemctl restart bybitbot
    sudo journalctl -u bybitbot -f
    tail -f /var/log/bybitbot/console.log
    ```

## Demo trading (Bybit testnet) — recommended workflow

Fast and safe option: run **a second bot instance** on Bybit testnet (demo) as a separate user + separate `systemd` unit. This way you can test strategy changes without risking the real account, and you can stop/start demo independently.

### 1) Create a demo user

- Start a separate process: `python bybitbot.py --user demo`
- Put demo keys into `users/demo/secrets.env`:
  ```env
  BYBIT_API_KEY=...        # testnet keys
  BYBIT_API_SECRET=...
  BYBIT_SANDBOX=1          # enables ccxt sandbox/testnet mode
  ```

### 2) Create a demo systemd unit

Example: `/etc/systemd/system/bybitbot-demo.service`:

```ini
[Unit]
Description=ByBit Bot (demo/testnet)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/root/bot/ByBit-Bot
Environment=PY_COLORS=1
Environment=FORCE_COLOR=1
Environment=BYBITBOT_LOCK_FILE=/var/run/bybitbot-demo.lock
ExecStart=/usr/bin/python3 /root/bot/ByBit-Bot/bybitbot.py --user demo
Restart=always
RestartSec=5
StandardOutput=append:/var/log/bybitbot/demo.console.log
StandardError=append:/var/log/bybitbot/demo.console.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bybitbot-demo
```

### 3) Telegram control

- Use `/demo` in Telegram to see hints and control the demo unit (`/demo status|start|stop|restart`).
- The bot manages `bybitbot-demo.service` by default (override via `BYBITBOT_DEMO_SERVICE`).

# Scheduled restart & update workflow

Use `manage_update.py` to stop the bot, run your update commands, and restart it at the planned trading time.
If `runtime_status.json` contains `next_run_utc`, the manager aligns the restart to that timestamp; otherwise it falls back to the nearest HH:01 / HH:31 slot.

### Cron-based autostart

The previous production setup ran via cron. A helper script is provided in `scripts/manage_update_cron.sh`; it:

1. Moves to the repository root.
2. Attempts to load `load_env.sh` (or a simple `.env`).
3. Calls `manage_update.py schedule --update-cmd "git pull --ff-only"`.

Install it into crontab (every 5 minutes, adjust as desired):

```cron
*/5 * * * * /usr/bin/env bash /path/to/ByBit\ Bot/scripts/manage_update_cron.sh >> /path/to/ByBit\ Bot/logs/manage_update.log 2>&1
```

`manage_update.py` reads `runtime_status.json` to honour model-provided `next_run_minutes` / `next_run_time`. The cron job merely wakes the supervisor; the actual trading cadence remains controlled by the model.

### Script options
```
python manage_update.py schedule  [--update-cmd cmd] [--resume-minutes 1 31] [--prep-seconds 20]
python manage_update.py immediate [--update-cmd cmd]
python manage_update.py status
```

- `--update-cmd` can be repeated to run several commands (default: none).
- `--resume-minutes` controls the fallback slots when no next_run is available.
- `--log-file` / `--no-log` configure log capture.
- `--no-restart` stops and updates without starting the bot.
- `--foreground` keeps the bot attached to the current terminal.

# Auto-update across branches

Before each cycle the bot:

1. Fetches the configured remote (`git fetch --prune`).
2. Reads `refs/remotes/<remote>/HEAD` and checks out whichever branch it currently points at (`git checkout -B <branch> <remote>/<branch>`).
3. Runs `git pull --ff-only` to fast-forward that branch and restart from the new commit.

In practice this keeps every instance aligned with the latest remote `HEAD` (any branch), so when a new commit lands on another branch the bot will switch to that branch automatically after the next fetch/pull cycle.

## Bybit timing/nonce errors

If you see InvalidNonce / retCode 10002 from Bybit complaining about timestamp/recv_window, set a larger receive window and let CCXT adjust for server time:

```bash
export BYBIT_RECV_WINDOW_MS=15000   # 1k..60k ms (15s is safe default)
```

Also ensure your server clock is synchronized (e.g., systemd-timesyncd, chrony, or ntpdate).

## Spot + Futures on Bybit (Unified)

- Mark spot pairs explicitly in `PAIR_LIST` with the `:SPOT` suffix, e.g. `BTC/USDT:SPOT, ETH/USDT:SPOT`.
- Futures (USDT-perp) use `:USDT` settle suffix, e.g. `BTC/USDT:USDT`. If no suffix is provided, the bot defaults to derivatives.
- Mixed mode is supported on a unified account:
  - Funding/Open Interest are queried only for derivatives.
  - Spot orders do not use reduceOnly/positionIdx/conditional fields.
  - The bot skips �open� short on spot (`SELL` to open). Closing spot exposure is done by explicit `SELL` of held assets.
- Ensure free balances exist for spot orders:
  - `BUY`: free `USDT` must cover notional + fees.
  - `SELL`: free base asset must cover the sell amount.

## Adding a new trader without sharing Bybit keys

1. **Collect the Telegram user id** of the new trader (they can forward any of their messages to `@userinfobot`). Decide on a unique bot id, e.g. `alice`.
2. **Create a profile**: either run `/adduser alice <telegram_id>` in the Commands topic or append to `users/users.json`. Keep the entry minimal�`id`, optional `label`, `owner_id`, and per-user overrides such as:
   ```json
   {
     "id": "alice",
     "label": "Alice",
     "owner_id": 123456789,
     "env": {
       "PAIR_LIST": "BTC/USDT:USDT,ETH/USDT:USDT",
       "TELEGRAM_MESSAGE_PREFIX": "Alice"
     }
   }
   ```
3. **Spin up the trader's process**: run `python bybitbot.py --user alice` (or schedule it via `manage_update.py --user alice`). The process reads only `users/alice/*.env` plus the global `.env`.
4. **Have the trader set their own keys**:
   - They can DM the bot `/bybitkey <apiKey> <apiSecret>` while the `--user alice` process is online. The bot writes the credentials into `users/alice/secrets.env` (git-ignored), so you never see the raw keys.
   - Alternatively they can edit `users/alice/secrets.env` directly via SSH:
     ```env
     BYBIT_API_KEY=xxx
     BYBIT_API_SECRET=yyy
     ```
5. **Share optional overrides**: `users/alice/public.env` can hold non-secret tweaks (pair list, leverage, Telegram topics). The trader edits only their directory.
6. **Operate the session**: the trader interacts with the shared Telegram group (their process has its own prefix) and can rotate keys any time by rerunning `/bybitkey`.

Each trader runs on their own Bybit account and equity: the per-user API keys determine balances, so deposits are isolated.

This flow keeps Bybit credentials in the trader's hands while letting you manage the shared infrastructure.


