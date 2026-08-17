# Global instructions

## Host and permissions

This Codex installation runs natively in WSL. Use Linux tools and paths for WSL
repositories. Windows repositories mounted under `/mnt/c` may be inspected and
edited from WSL, but platform SDK commands must use the repository's documented
native host toolchain.

The user explicitly selected full access for this WSL Codex installation:
`sandbox_mode = "danger-full-access"` and `approval_policy = "never"`. Do not
weaken or override those defaults, and do not add MCP approval prompts, unless
the user explicitly requests a safer profile. Platform-enforced approval gates
may still apply outside Codex configuration.

## File search and grep

For any file lookup or literal search in the current git repository, use fff
first. Do not use shell `find`, `grep`, or `rg` when fff can express the query.
Use short bare identifiers and `multi_grep` for OR queries. After two searches,
inspect the selected code instead of repeating variants.

## Structural navigation

Use CodeGraph for structural questions over symbols, calls, dependencies, and
impact. Prefer `codegraph_explore` as the primary entry point for unfamiliar
features and bugs. Run `codegraph status` at the start of a work session and
`codegraph sync` after bulk external changes or reported staleness. If a
repository is not initialized, state that and continue with fff and Serena; do
not initialize it without the user's request.

Do not run Windows and WSL CodeGraph processes concurrently against one
`.codegraph` SQLite index on `/mnt/c`. For slow mounted filesystems,
`codegraph serve --mcp --no-watch` is available; when using it, sync explicitly
after meaningful external edits.

## Symbolic navigation and editing

Use Serena after fff or CodeGraph identifies the relevant area. Call
`initial_instructions` before coding tasks. Prefer `get_symbols_overview`,
`find_symbol`, `find_referencing_symbols`, `replace_symbol_body`, `insert_*`,
`rename_symbol`, and `safe_delete_symbol` when symbolic operations are
sufficient. Serena runs with `--project-from-cwd`.

## Skills, docs, and web research

Use a skill when the task matches its description. Use Context7 before relying
on memory for version-sensitive library or framework behavior. Use Tavily for
fresh general web research. For OpenAI and Codex behavior, use the current
official OpenAI documentation workflow.

## Memory

Use Honcho through the installed `codex-honcho` host plugin. The plugin owns
hooks, its skill, and MCP registration; do not hand-maintain a separate Honcho
MCP block. Consult Honcho before answering about remembered preferences,
project rules, or prior decisions. Save only durable preferences, decisions,
patterns, and gotchas. Treat memory as inference until verified in files or
command output.

## Verified local tools

Verified on 2026-08-17: Codex CLI `0.147.0`, Claude Code `2.1.231`,
uv `0.11.25`, Node `22.20.0`, npm `11.17.0`, fff-mcp `0.10.5`,
CodeGraph `1.5.0`, Serena `1.7.0`, Context7 `4.0.2`, Playwright MCP `0.0.79`,
Tavily MCP `0.2.22`, DBHub `1.2.0`, Ponytail `4.9.0`, Caveman MCP
(`@caveman-ai/cli` `1.2.0`), zoxide `0.10.0`, fzf `0.74.1`, and
Starship `1.26.0`. Global MCP includes fff, CodeGraph, Serena, Context7,
Tavily, Playwright, Figma, Honcho, DBHub, Ponytail, and Caveman.

## Working style

Before changing code, briefly state what was found, what will change, and why.
Prefer minimal, localized edits. Preserve unrelated work in dirty trees. Do not
rename or move files, change public interfaces, install project dependencies,
or edit secrets unless the task requires it. Separate confirmed facts from
inference and verify completion against the actual current state.

---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and Codex when working
with code in this repository.

> `AGENTS.md` is kept **byte-identical** to this file (same instructions, same rules).
> Edit one, mirror the change to the other.

## Tooling

Code navigation uses three MCP servers, each with one job — do not duplicate them:

- **fff 0.10.5** — locate files and literal text (strings, comments, log
  messages). Tools: `find_files`, `grep`, `multi_grep`. Use fff instead of
  shell `find`/`grep`/`rg`. One bare identifier per query; after two greps,
  read the code. Grep keeps FilePath scope in regex/literal fallback;
  `multi_grep` accepts standalone constraints. Do not index `$HOME` or `/`
  (`--enable-home-scan` / `--enable-root-scan` stay off).
- **codegraph 1.5.0** — structural questions over a tree-sitter symbol graph.
  Native Rust parse engine; a save reaches the graph in well under a second
  (watcher quiet window 300ms). MCP tool is only `codegraph_explore` (entry
  points + related symbols + source). Callers/impact/status/sync are CLI —
  there is no `codegraph_context` / `codegraph_search` / `codegraph_impact`
  MCP tool. Trust the AST result and do not re-check it with grep. Run
  `codegraph status` at session start and `codegraph sync` after bulk
  external edits.
- **serena 1.7.0** — LSP-precise symbol navigation and symbol-level editing
  (`find_symbol`, `get_symbols_overview`, `find_referencing_symbols`,
  `find_declaration`, `find_implementations`, `replace_symbol_body`,
  `insert_*`, `rename_symbol`, `safe_delete_symbol`, `replace_content`,
  `replace_in_files`). Prefer it over reading whole code files. Project
  language: **python** (`.serena/project.yml`); call `initial_instructions`
  before coding. Project key `languages` → `language_servers` (auto-migrated).
  Explicit file tools (`read_file`, `replace_content`, `create_text_file`)
  are no longer blocked by ignore lists; subtree tools take
  `skip_ignored_files`. `get_current_config` includes language-server status.
  Activation errors appear in the system prompt, not only logs. Dashboard is
  disabled globally — do not re-enable it.

Цикл: locate (fff) → understand (`codegraph_explore`) → assess risk
(`codegraph impact` CLI) → read & edit (serena) → verify.

The global MCP configuration is inherited from the WSL user account. Do not
create a project `.mcp.json` or copy credentials into this repository.
Context7 `4.0.2` (`resolve-library-id`, `query-docs`; one concept per query;
HTTP stateless after MCP SDK v2) handles version-sensitive library docs;
Tavily `0.2.22` handles fresh general research. Playwright `0.0.79`:
`browser_take_screenshot` accepts `type` png/jpeg/webp; `--output-mode` is
gone. TimesOil currently has no web UI, so Playwright is relevant only if a
web surface is added. DBHub `1.2.0` returns one result set per SQL statement.

**Ponytail 4.9.0** shrinks *code*: always-on 7-rung ladder (YAGNI → reuse →
stdlib → native → installed dep → one line → minimum). MCP:
`ponytail_instructions`. Skills: `/ponytail`, `/ponytail-review`,
`/ponytail-audit`. **Caveman** shrinks *prose*: terse talk, code stays exact.
MCP: `caveman_compress` / `caveman_retrieve` / `caveman_stats` /
`caveman_toon_encode` / `caveman_toon_decode`. Use both together.

## Memory (Honcho)

Use the installed host plugin: `codex-honcho` in Codex and `honcho@honcho` from
`plastic-labs/claude-honcho` in Claude Code. The plugin owns hooks, its skill,
and MCP registration; do not add a second hand-maintained Honcho block. Use
`search`/`chat` for recall and `create_conclusions` for durable decisions and
gotchas. Treat memory as inference until files or command output confirm it.
Serena project memories complement Honcho: start with `mem:core`.

## Project Purpose

**TimesOil** — прогноз дебита нефти и жидкости добывающих скважин нефтяного
месторождения на 6 месяцев вперёд. Данные: `raw_data/` (в git не входят —
передаются `scp`) — два Excel (один датасет: длинный `Dataset.xlsx::MODEL_Y` и
широкий «Dataset Шутову АА+.xlsx», численно идентичны) + карта разломов
`image (5).png`. 49 скважин (33 добывающих + 16 нагнетательных), месячная
история 2007-05..2015-11 из гидродинамического симулятора; поле разбито
разломами на 6 блоков (оцифровано в `src/timesoil/wells.py`).

**Стек**: Python 3.13 + uv; `pandas/numpy/scipy/matplotlib`, LightGBM и
MLForecast. Модельный контур включает Chronos-2, TiRex-2, TiDE, LightGBM и
физические CRM/двухфазную CRM/CRMP с давлением и модель Джентила; SPDM/ManiMamba
остаётся исследовательской линией на a100 в отдельном Python 3.12-окружении.
Последний зафиксированный ансамбль: WAPE **3,74 % по нефти** и **3,21 % по
жидкости** на трёх канонических срезах.

**Структура и точки входа**:
- `src/timesoil/` — данные (`data.py` — все причуды исходников задокументированы
  в докстринге), фонд/блоки (`wells.py`), метрики, бейслайны, бэктест,
  раннер TiRex-2; этап 2: `crm.py` (ёмкостно-резистивная модель),
  `allocation.py` (адресная закачка), `fractional.py` (обводнённость);
- `scripts/run_baselines.py`, `run_tirex.py`, `run_crm.py`,
  `run_fractional.py`, `run_chronos.py`, `run_lgbm.py`, `run_ensemble.py`,
  `run_stacking.py` — бэктест и ансамбли;
- `scripts/calibrate_intervals.py` — конформная калибровка квантилей;
- `scripts/prepare_spdm_data.py` -> `spdm_run.sh` (на a100, tmux) ->
  `eval_spdm.py` — контур SPDM;
- `scripts/forecast_forward.py` — итоговый прогноз 2015-12..2016-05
  (стек: CRM-жидкость, Джентил-нефть, интервалы TiRex-2 с множителями);
- `scripts/optimize_injection.py` — перераспределение закачки (SLSQP
  поверх стека CRM x Джентил, сумма закачки фиксирована);
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
uv sync --extra tirex              # TiRex/Chronos; обычный sync может снять extra
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
- **GPU:** 6× A100-SXM4-40GB. **NVLink физически отсутствует** → межкарточно
  только PCIe, P2P выключен (`NCCL_P2P_DISABLE=1` — норма для этого бокса).
  Сервер разделяется с production-сервисами других проектов: ни одна карта не
  считается свободной по умолчанию. Перед каждым запуском проверить
  `nvidia-smi`; не останавливать и не вытеснять чужие сервисы. Использовать
  только явно свободную или выделенную карту. Для нескольких карт
  предпочтительны независимые задачи 1-на-GPU либо DDP внутри одного
  NUMA-острова (`{1,2}` ↔ node0 / `{3,4,5}` ↔ node1).
- **Long-running** запускать в `tmux new -d -s <name>`.

## Conventions

- Владелец разрешил автономно менять код, запускать проверки, скачивать открытые
  модели, выполнять согласованные A100-прогоны и синхронизацию. Спрашивать
  только при платформенном подтверждении либо необратимом и неоднозначном
  действии. Это не разрешает останавливать сервисы других проектов.
- `from __future__ import annotations` в начале модулей; type hints; `X | Y` (3.10+).
- Зависимости — только через `uv` (не `pip install` напрямую).
- `raw_data/`, `data/`, `results/`, модельные веса и крупные артефакты не
  коммитить; передавать через `scp -p`.
- Экспертные отчёты/документы — на русском, без англоязычного жаргона;
  формулы — в LaTeX.
- Рабочий журнал проекта — `docs/roadmap.md` (схема Plan → Act → Verify → Report).

## Task completion

Минимальный локальный шлюз для изменений Python:

```bash
uv sync --locked
uv run python -m compileall -q src scripts
uv run python -c "import timesoil"
git diff --check
```

После изменения модели или расчёта дополнительно выполнить релевантный
воспроизводимый скрипт на a100, сохранить метрики в `docs/roadmap.md`, затем
commit/push → `git pull --ff-only` на сервере. Не считать локальный smoke
заменой серверного численного прогона.

## Agent bootstrap

Запускать Codex из корня проекта, иначе fff/Serena/CodeGraph получат другой cwd:

```bash
cd /home/ruslan_safaev/TimesOil
codex
```

В начале сессии: `codegraph status`, Serena `initial_instructions`, затем
`mem:core`. CodeGraph уже инициализирован локально; его SQLite-индекс
не коммитится. MCP-серверы и Honcho подключаются глобально, дополнительных
project-level credential/config файлов не требуется.
