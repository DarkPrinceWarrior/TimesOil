# Рабочий журнал — TimesOil

Схема записи каждой задачи: **Plan → Act → Verify → Report**.

## Журнал

| Дата | Действие | Результат | Статус |
|------|----------|-----------|--------|
| 2026-07-04 | Инициализация репозитория: каркас тулинга агентов (`CLAUDE.md`/`AGENTS.md` побайтно, `.serena/project.yml` python, `.gitignore`, `README.md`, `docs/roadmap.md`). MCP-инструменты глобальные (serena/tavily/fff/codegraph + плагины context7/playwright/honcho) — наследуются. codegraph обновлён 1.1.3→1.2.0. | Публичный GitHub-репозиторий создан, продублирован на a100 `/root/projects/TimesOil`. Готово к работе нового агента. | ✅ |
| 2026-07-04 | **Задача 1: прогноз дебита на 6 мес (TiRex-2 + SPDM).** План: изучить `raw_data` (2 Excel + карта разломов) → исследовать модели → постановка (3 среза × 6 мес, WAPE) → пайплайн (`src/timesoil`, `scripts/`) → прогоны (TiRex-2 локально на CPU, SPDM на a100 GPU 5, tmux) → отчёт. Act: оцифрованы 5 разломов с карты → 6 блоков; выявлены ловушки данных (THP=пластовое, DobG=жидкость, мусорный 2015-12, нули до пуска, перевод 15 скв. под нагнетание в 2008). SPDM: 3 дефекта репо исправлены (конфиги, freq `ME`, путь чекпоинта), 8 обучений на a100. Verify: сверка двух Excel (тождественны), бэктест 10 моделей/конфигураций. | **TiRex-2 blocks_cov лучшая: нефть WAPE 8.3 %, жидкость 7.7 %** (Арпс 9.4 %/10.3 %, SPDM 10.0 %/12.5 %). Закачка по блокам как известный план — главный фактор (жидкость 13.4→7.7 %). Прогноз вперёд 2015-12..2016-05 в `results/forward_*.csv`. Отчёт: `docs/отчёт_прогноз_дебита_6мес.md`. | ✅ |

## Онбординг

- **Назначение проекта:** прогноз дебита нефти/жидкости скважин на 6 месяцев
  (месторождение с разломами, поздняя стадия заводнения, данные симулятора).
- **Технологический стек:** Python 3.13 + uv; pandas/numpy/scipy/matplotlib;
  TiRex-2 (zero-shot, extra `tirex`, работает на CPU); SPDM/ManiMamba
  (обучение на a100, отдельное окружение `external/spdm/.venv`, py3.12+mamba-ssm).
- **Структура и точки входа:** `src/timesoil/` (data/wells/metrics/baselines/
  backtest/tirex_runner) + `scripts/` (`run_baselines.py`, `run_tirex.py`,
  `prepare_spdm_data.py` → `spdm_run.sh` → `eval_spdm.py`,
  `forecast_forward.py`, `collect_results.py`, `make_figs.py`).
  Данные `raw_data/` вне git (передавать `scp`), результаты `results/` вне git.
