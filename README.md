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
    export TELEGRAM_COMMANDS="status:Status command;help:Help command"  # slash-command overrides
    export NEWS_PROVIDER=hybrid                   # news sources: hybrid (default), cryptocompare, rss
    ```

## AI model configuration

- Set `OPENAI_API_KEY` to your OpenAI API key (shared across traders unless overridden per user profile).
- `OPENAI_MODEL_PRIMARY` defaults to `gpt-4.1-mini` (used for planning-heavy tasks).
- `OPENAI_MODEL_CHEAP` defaults to `gpt-4o-mini` and is used when `/tokens` shows the cheap threshold was hit.
- `AI_SUPPORT_MODEL` can override `/support` replies (falls back to the primary trading model if empty).
- `/tokens` displays budgets, model switches, and remaining tokens.

### Telegram diagnostics

- `/ai payload [context]` shows the last OpenAI request/response snapshot (pass `universe` or `trade` to focus).
- `/logs [count|SYMBOL window]` prints console logs; e.g. `/logs 30` or `/logs BTC 60`.

### Telegram log mirroring

Enable `TELEGRAM_FORWARD_LOGS=1` to mirror console logs into Telegram. The bot batches entries (`TELEGRAM_LOG_BATCH_SIZE` / `TELEGRAM_LOG_FLUSH_INTERVAL`) and respects the rate window (`TELEGRAM_LOG_RATE_LIMIT` / `TELEGRAM_LOG_RATE_WINDOW`).

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

## Environments: prod / testnet / demo

- **Prod (real)** uses user default and users/default/secrets.env; no additional flags are needed.
- **Testnet (sandbox)** runs separately (for example user demo), points at users/demo/secrets.env, and sets BYBIT_SANDBOX=1 or BYBIT_TESTNET=1.
- **Demo Trading (api-demo.bybit.com)** reuses the same code but connects to the Demo Trading API (real-like market data). Set BYBIT_DEMO=1 or BYBIT_ENV=demo in users/demo/secrets.env; BYBIT_API_BASE=https://api-demo.bybit.com is injected automatically.

Keep separate processes/units for prod/testnet/demo to avoid mixing keys and logs.

### 1) Configure users and keys

users/users.json:
`json
{
  "users": [
    { "id": "default", "label": "real", "enabled": true, "env_files": ["users/default/secrets.env"] },
    { "id": "demo", "label": "demo", "enabled": true, "env_files": ["users/demo/secrets.env"] }
  ]
}
`

users/default/secrets.env (real) example:
`env
BYBIT_API_KEY=REAL_KEY
BYBIT_API_SECRET=REAL_SECRET
# no BYBIT_SANDBOX/BYBIT_DEMO
`

users/demo/secrets.env gets one of the following:
`env
# Option 1: Testnet (sandbox)
BYBIT_API_KEY=TESTNET_KEY
BYBIT_API_SECRET=TESTNET_SECRET
BYBIT_SANDBOX=1

# Option 2: Demo Trading (api-demo.bybit.com)
# BYBIT_API_KEY=DEMO_TRADING_KEY
# BYBIT_API_SECRET=DEMO_TRADING_SECRET
# BYBIT_DEMO=1
# (optional) BYBIT_API_BASE=https://api-demo.bybit.com
`

### 2) Run manually

```bash
cd /root/bot/ByBit-Bot
git pull --ff-only --autostash
# real account
python3 bybitbot.py --user default
# demo/testnet (depending on flags in users/demo/secrets.env)
python3 bybitbot.py --user demo
```

Log output is available under `assets/bybit.log` or `runtime/<user>/bybit.log` when per-user logging is active.

### 3) Separate demo/testnet/api-demo systemd unit

`/etc/systemd/system/bybitbot-demo.service`:
```ini
[Unit]
Description=ByBit Bot (demo/testnet/api-demo)
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

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bybitbot-demo
sudo journalctl -u bybitbot-demo -f
```

### 4) Telegram controls

- `/demo` controls the `bybitbot-demo.service` (override with `BYBITBOT_DEMO_SERVICE` if needed).
- `/bybitkey <apiKey> <apiSecret>` updates `users/<user>/secrets.env` for the active user.


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
2. Finds the **newest commit across all remote branches** (by committer timestamp).
3. If the current checkout is older, checks out the branch that contains that newest commit and then runs `git pull --ff-only`.

Practical behavior:

- If you manually switch the server to the branch that currently has the newest commit, the bot will **not** jump away.
- If you manually switch to an older branch/commit, the bot will automatically jump back to the newest commit.
- If backup mode is active, the bot stays on stable/backup until a **newer** commit appears, then exits backup mode automatically.

Manual switch after stable fallback (pick a newer non-stable commit):

```bash
git fetch --prune
git for-each-ref refs/remotes/origin --sort=-committerdate --format="%(refname:short) %(committerdate:iso8601) %(objectname:short)" \
  | grep -v "origin/stable" | head -n 1
git checkout -B <branch> origin/<branch>
git pull --ff-only
# Restart the bot (systemd/pm2/docker/etc or: python bybitbot.py --user <id>)
```

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
  - The bot avoids opening short entries on spot (`SELL` to open); close spot exposure with explicit `SELL`.
- Ensure free balances exist for spot orders:
  - `BUY`: free `USDT` must cover notional + fees.
  - `SELL`: free base asset must cover the sell amount.

## Adding a new trader without sharing Bybit keys

1. **Collect the Telegram user id** of the new trader (they can forward any of their messages to `@userinfobot`). Decide on a unique bot id, e.g. `alice`.
2. **Create a profile**: either run `/adduser alice <telegram_id>` in the Commands topic or append to `users/users.json`. Keep the entry minimal (fields: `id`, optional `label`, `owner_id`, plus per-user overrides) and include overrides such as:
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


