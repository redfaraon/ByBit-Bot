# TODO / Roadmap

## Текущие задачи (v1.4.1)

- [ ] 1) Добавить трассировку защиты для всех тикеров (вход/выход + открытые ордера + параметры SL/TP/Trailing).
- [ ] 2) Убедиться, что `git push` работает локально (commit + push).
- [ ] 3) Расширить конфиг `ENV` + `strategy_spec.json` (параметры одинаковые), добавить:
  - per-symbol/per-regime `trailing_*`
  - фильтры входа/выхода (волатильность/«температура» рынка)
  - `news` thresholds/weights (providers/hybrid)
  - новые ENV: `MAX_TRAILING_LOSS_PCT`, `MIN_POSITION_SIZE`, `NEWS_WEIGHT` (+ совместимость при фолбэке).
- [ ] 4) Backtest:
  - [ ] CLI запуск (`python -m backtest.session --strategy ... --from ... --to ...`)
  - [ ] Команда в Telegram (`/backtest <from> <to>`) + генерация графика equity/balance/available
  - [ ] Отправка изображения в Telegram.

## Правила фикса

- Версии модулей менять только если модуль реально изменён.
- `ENV` и `strategy_spec.json` должны быть синхронизированы, чтобы при фолбэке стратегия не менялась.

