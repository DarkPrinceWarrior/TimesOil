# Task Completion Gate

## Любое изменение Python

```bash
uv sync --locked
uv run python -m compileall -q src scripts
uv run python -c "import timesoil"
git diff --check
```

- Проверить `git status --short`; не включать чужие изменения, данные, результаты, веса и секреты.
- Если менялись `AGENTS.md`/`CLAUDE.md`, требовать `cmp -s AGENTS.md CLAUDE.md`.
- После внешних bulk-правок выполнить `codegraph sync`; индекс `.codegraph/` не коммитить.

## Модель/расчёт/метрика

- На a100 выполнить релевантный воспроизводимый сценарий на явно свободной/выделенной GPU; длительный запуск — tmux.
- Сохранить параметры, срезы, метрики и отрицательные результаты в `docs/roadmap.md`; не подменять серверный численный прогон локальным import-smoke.
- Commit/push локально, затем `git pull --ff-only` на `/root/projects/TimesOil`; подтвердить одинаковый SHA и чистое серверное дерево.

## Инструкции/инфраструктура агента

- Проверить fff base path = TimesOil, `codegraph status` up to date, Serena project = TimesOil и наличие `mem:core`, `mem:tech_stack`, `mem:suggested_commands`, `mem:conventions`, `mem:task_completion`.
- Не создавать project-level MCP-конфиг и не копировать глобальные credentials.