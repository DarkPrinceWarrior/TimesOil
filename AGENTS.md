# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and Codex when working
with code in this repository.

> `AGENTS.md` is kept **byte-identical** to this file (same instructions, same rules).
> Edit one, mirror the change to the other.

## Tooling

Code navigation uses three MCP servers, each with one job — do not duplicate them:

- **fff** — locate files and literal text (strings, comments, log messages).
  Use fff instead of shell `find`/`grep`/`rg` — быстрее и точнее на больших репо.
  One bare identifier per query; after two greps, read the code.
  Tools: `find_files` (какие файлы/модули существуют), `grep` (содержимое по
  идентификатору), `multi_grep` (OR по нескольким паттернам / вариантам регистра).
- **codegraph** — structural questions over a tree-sitter symbol graph.
  `codegraph_explore "<task>"` — **основной** инструмент (точки входа + связанные
  символы + исходники в одном вызове). Также `codegraph_search` (символ по имени —
  предпочесть вместо `fff grep`), `codegraph_callers`/`codegraph_callees`,
  `codegraph_impact` (радиус изменений перед рефактором), `codegraph_node`,
  `codegraph_files`, `codegraph_status`. Доверять результатам — полный AST-парс,
  не перепроверять grep'ом. `codegraph status` в начале сессии, `codegraph sync`
  после bulk-изменений.
- **serena** — LSP-точная навигация по символам и **единственный** инструмент,
  который *редактирует* на уровне символов (`find_symbol`, `get_symbols_overview`,
  `find_referencing_symbols`, `replace_symbol_body`, `insert_before_symbol`,
  `insert_after_symbol`, `rename_symbol`, `safe_delete_symbol`). Предпочесть чтению
  целых файлов. Язык проекта: **python** (см. `.serena/project.yml`). Serena
  активирует проект автоматически (ищет `.serena/project.yml` или `.git` вверх от
  cwd); в начале задачи полезно вызвать `initial_instructions`.

Цикл: locate (fff / `codegraph_search`) → understand (`codegraph_explore`) →
assess risk (`codegraph_impact`) → read & edit (serena) → verify.

Прочие MCP / плагины: **context7** — версионно-зависимые доки библиотек
(предпочесть веб-поиску); **tavily** — общий веб-поиск (`tavily_search`,
`tavily_extract`); **playwright** — браузерные smoke-проверки любого веб-UI.

## Memory (Honcho)

Память — через установленный host-плагин: `honcho@honcho` из
`plastic-labs/claude-honcho` в Claude Code и `codex-honcho` в Codex. Контекст о
пользователе, предпочтениях и прошлой работе загружается хуками плагина в начале
сессии; доверять ему, но перед ответами о предпочтениях проекта, рабочих правилах,
прошлых решениях и запомненном контексте — свериться с Honcho ещё раз. Для активной
работы: `search`/`chat` (recall), `get_context`/`get_representation` (текущая модель
пользователя/проекта), `create_conclusions`/`create_conclusion` (сохранить
устойчивые предпочтения, решения, паттерны, грабли). Разделять подтверждённое
(файлы, вывод команд) и предположения (память, архитектурные догадки) до проверки.

## Project Purpose

**TimesOil** — прогноз дебита нефти и жидкости добывающих скважин нефтяного
месторождения на 6 месяцев вперёд. Данные: `raw_data/` (в git не входят —
передаются `scp`) — два Excel (один датасет: длинный `Dataset.xlsx::MODEL_Y` и
широкий «Dataset Шутову АА+.xlsx», численно идентичны) + карта разломов
`image (5).png`. 49 скважин (33 добывающих + 16 нагнетательных), месячная
история 2007-05..2015-11 из гидродинамического симулятора; поле разбито
разломами на 6 блоков (оцифровано в `src/timesoil/wells.py`).

**Стек**: Python 3.13 + uv; `pandas/numpy/scipy/matplotlib`; модели —
TiRex-2 (NX-AI, zero-shot, extra `tirex`), SPDM/ManiMamba (обучение на a100,
отдельное окружение `external/spdm/.venv`, python 3.12 + mamba-ssm cu12),
физика — CRM (`pywaterflood`) и фракционная модель Джентила.

**Структура и точки входа**:
- `src/timesoil/` — данные (`data.py` — все причуды исходников задокументированы
  в докстринге), фонд/блоки (`wells.py`), метрики, бейслайны, бэктест,
  раннер TiRex-2; этап 2: `crm.py` (ёмкостно-резистивная модель),
  `allocation.py` (адресная закачка), `fractional.py` (обводнённость);
- `scripts/run_baselines.py`, `run_tirex.py`, `run_crm.py`,
  `run_fractional.py` — бэктест (3 среза x 6 мес);
- `scripts/calibrate_intervals.py` — конформная калибровка квантилей;
- `scripts/prepare_spdm_data.py` -> `spdm_run.sh` (на a100, tmux) ->
  `eval_spdm.py` — контур SPDM;
- `scripts/forecast_forward.py` — итоговый прогноз 2015-12..2016-05
  (стек: CRM-жидкость, Джентил-нефть, интервалы TiRex-2 с множителями);
- `scripts/collect_results.py`, `make_figs.py` — сводка и графики;
- `results/` (вне git), отчёт — `docs/`.

Ключевые «грабли» данных: колонка THP в MODEL_Y — на самом деле **пластовое**
давление; DobG — жидкость, не газ; последний месяц (2015-12) — мусор;
нули до старта скважины — «скважины ещё нет» (маркер WEFF=0); метрики — WAPE
(скв. 1 полностью обводнена, MAPE взрывается); закачка в м3, добыча в тоннах.

## Setup

Окружение управляется через **`uv`**, Python **3.13**:

```bash
uv venv --python 3.13 .venv        # создать окружение
uv add <package>                   # добавить зависимость (пишет в pyproject + uv.lock)
uv sync                            # установить из uv.lock
uv run python <script>.py          # запуск в окружении проекта
```

Не использовать «голый» `pip` — только `uv`.

## Server workflow (a100)

**Рабочая модель:** правки вносятся **локально в WSL**, прогоны и вычисления —
**на сервере a100** (тот же физический хост, где лежит `rag_app`). Локальная копия —
источник изменений; сервер — рабочее место для запусков и GPU.

- **SSH:** `ssh a100` (LAN `192.168.101.12`, из офиса/VPN) или `ssh a100-remote`
  (из любой сети, через jump host). Один физический хост `zeta` (Proxmox),
  окружение — контейнер **LXC 135**.
- **Проектная директория на сервере:** `/root/projects/TimesOil/`; `uv` —
  `/root/.local/bin/uv`; окружение `uv venv --python 3.13 .venv`.
- **Синхронизация:** правка в WSL → `git commit && git push` → на a100 `git pull`.
  `rsync` на сервере нет — для файлов вне git используйте `scp -p`.
- **GPU:** 6× A100-SXM4-40GB. **NVLink физически отсутствует** → межкарточно только
  PCIe, P2P выключен (`NCCL_P2P_DISABLE=1` — норма для этого бокса). GPU0 занят
  сервисом `whisperx` → использовать `CUDA_VISIBLE_DEVICES=1..5`. Для нескольких
  карт: предпочтительно независимые задачи 1-на-GPU, либо DDP **внутри одного
  NUMA-острова** (`{1,2}` ↔ node0 / `{3,4,5}` ↔ node1).
- **Long-running** запускать в `tmux new -d -s <name>`.

## Conventions

- `from __future__ import annotations` в начале модулей; type hints; `X | Y` (3.10+).
- Зависимости — только через `uv` (не `pip install` напрямую).
- Экспертные отчёты/документы — на русском, без англоязычного жаргона;
  формулы — в LaTeX.
- Рабочий журнал проекта — `docs/roadmap.md` (схема Plan → Act → Verify → Report).

## Local tool versions

Verified **2026-07-04**: `fff-mcp 0.9.6`, `Serena 1.5.4.dev0` (git-main
`oraios/serena`, новее релиза v1.5.3), `codegraph 1.2.0`, `codex-cli 0.142.5`,
`node 22.20.0`, `npm 11.17.0`. MCP-серверы настроены **глобально**
(`~/.claude.json`): `serena`, `tavily`, `fff`, `codegraph`; плагины Claude Code:
`context7`, `playwright`, `honcho`. Новый проект наследует их автоматически —
отдельная установка не нужна (нужен лишь `.serena/project.yml`, он уже создан).
