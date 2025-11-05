# Multi-user setup

Each trader lives under `users/<user-id>/`. The bot reads three sources when a profile is
activated (in order):

1. Global `.env` / `.env.local` — shared settings, OpenAI and Telegram tokens.
2. Profile overrides from `users/users.json` (`env` block).
3. Private secrets from any files listed in `env_files`, plus the default
   `users/<user-id>/secrets.env`.

The secrets file is `.gitignored`; it should include the Bybit API key/secret and any
per-user credentials you do **not** want in the repository, for example:

```env
BYBIT_API_KEY=xxx
BYBIT_API_SECRET=yyy
```

To create a new trader:

1. Copy `users/users.example.json` to `users/users.json` (or append to an existing file).
2. Add a profile entry with a unique `id`, optional `label`, per-user `state_dir`, and any
   environment overrides (risk, pair list, Telegram thread IDs, etc).
3. Create `users/<id>/secrets.env` and place the trader's Bybit credentials there.
4. Start the bot with `python bybitbot.py --user <id>` (one process per trader). The
   process keeps running just like the single-user version.

Shared Telegram/OpenAI tokens remain in the global `.env`. Each user can still override
topics or prefixes by adding values like `TELEGRAM_MESSAGE_PREFIX` in their profile.
