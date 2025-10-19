# Quick Start

1. Install dependencies:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install --user -U ccxt pandas requests colorama "openai>=1.0.0" python-dotenv pyyaml
   ```

2. Export environment variables (example):
   ```bash
   export BYBIT_API_KEY=your_api_key
   export BYBIT_API_SECRET=your_api_secret
   export TELEGRAM_BOT_TOKEN=your_tg_token
   export TELEGRAM_CHAT_ID=your_chat_id
   export OPENAI_API_KEY=your_openai_key
   ```

3. Run the bot:
   ```bash
   python bybitbot.py
   ```

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
