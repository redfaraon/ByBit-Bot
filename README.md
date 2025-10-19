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

### Cron example (every 30 minutes)
```cron
*/30 * * * * cd /home/user/bybitbot && /usr/bin/python3 manage_update.py schedule --update-cmd "/bin/bash /home/user/bybitbot/update.sh"
```

`update.sh` is your custom update script (download new code, copy files, etc.). If you do not need any update step, omit `--update-cmd`. Bot output is appended to `bybit.log` unless you pass `--no-log`.

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
