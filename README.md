# TimesOil

Прогноз дебита нефти и жидкости добывающих скважин на 6 месяцев вперёд:
месторождение с экранирующими разломами (6 блоков), поздняя стадия заводнения,
месячные данные гидродинамического симулятора (49 скважин, 2007-05…2015-11).

Текущий контур объединяет **Chronos-2, TiRex-2, TiDE, LightGBM** и физические
CRM/двухфазную CRM/CRMP-модели; SPDM/ManiMamba сохранён как исследовательская
линия. Последний зафиксированный ансамбль: **нефть WAPE 3,74 %, жидкость
3,21 %** на трёх канонических срезах по 6 месяцев.
Подробности: [`docs/отчёт_прогноз_дебита_6мес.md`](./docs/отчёт_прогноз_дебита_6мес.md).

Правила работы с инструментами (fff / codegraph / serena), памятью Honcho и
workflow сервера a100 — в [`CLAUDE.md`](./CLAUDE.md); «Project Purpose» там же.
`AGENTS.md` = `CLAUDE.md` побайтно (для Codex).

## Быстрый старт

```bash
uv venv --python 3.13 .venv    # окружение (Python 3.13, только uv)
uv sync                        # зависимости из uv.lock
```

- **Правки** — локально в WSL; **прогоны** — на сервере a100
  (`/root/projects/TimesOil/`, `ssh a100` / `ssh a100-remote`), синхронизация через git.
- Рабочий журнал — [`docs/roadmap.md`](./docs/roadmap.md) (Plan → Act → Verify → Report).

## Запуск агента

```bash
cd /home/ruslan_safaev/TimesOil
codex
```

Codex наследует глобальные MCP, permissions и Honcho из WSL-профиля; локальный
`.mcp.json` не нужен. В начале работы агент проверяет `codegraph status`,
активирует Serena и читает `mem:core`. Полный регламент находится в
[`AGENTS.md`](./AGENTS.md), идентичном `CLAUDE.md`.
