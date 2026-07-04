# TimesOil

Инициализированный репозиторий проекта — каркас для работы агентов **Claude Code**
и **Codex**. Домен ориентировочно: анализ и прогноз временных рядов в нефтегазе.

Назначение, стек и структура уточняются при первой задаче — см. раздел
«Project Purpose» в [`CLAUDE.md`](./CLAUDE.md). Там же — правила работы с
инструментами (fff / codegraph / serena), памятью Honcho и workflow сервера a100.
`AGENTS.md` = `CLAUDE.md` побайтно (для Codex).

## Быстрый старт

```bash
uv venv --python 3.13 .venv    # окружение (Python 3.13, только uv)
uv sync                        # зависимости из uv.lock
```

- **Правки** — локально в WSL; **прогоны** — на сервере a100
  (`/root/projects/TimesOil/`, `ssh a100` / `ssh a100-remote`), синхронизация через git.
- Рабочий журнал — [`docs/roadmap.md`](./docs/roadmap.md) (Plan → Act → Verify → Report).
