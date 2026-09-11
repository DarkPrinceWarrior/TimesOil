# Воспроизведение прогона трека 2 на чистой машине

Цель: из этого репозитория получить для кейса Model Z расписание управления
(`wells_schedule.inc` за период управления) и официальный ЧДД за тот же период.
Цепочка: приём архива → базовый прогон OPM Flow → связность и блоки → поиск
графика (CMA-ES + агенты на суррогате TimesFM 3.0, ноль вызовов OPM) → печать
выбора → ровно один финальный прогон OPM → официальный калькулятор → отчёт.

## 1. Требования

| Компонент | Что нужно |
|---|---|
| ОС / CPU | Linux x86-64, ≥ 32 ядер (OPM Flow идёт на 16 MPI-процессах) |
| GPU | NVIDIA, ≥ 24 ГБ, драйвер с CUDA 12.8 (суррогат TimesFM 3.0, FP64 variate-softmax) |
| Python | 3.13, `uv` |
| Docker | образ OPM Flow `openporousmedia/opmreleases:2026.04_amd64@sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9` (тянется автоматически) |
| Сеть | доступ к OpenAI-совместимому эндпоинту Qwen (Татнефть LiteLLM или Cerebras) и к Hugging Face (`google/timesfm-3.0-pytorch`) |
| Архив кейса | `Model_Z_*.zip` от организаторов: ровно один `.DATA`, `SCHEDULE` через `INCLUDE`, METRIC |

## 2. Окружения

```bash
git clone <репозиторий> TimesOil && cd TimesOil
uv sync --locked                                   # основное окружение (OPM, калькулятор, API)
uv run python -m compileall -q src scripts && PYTHONPATH=src:scripts uv run pytest tests -q

uv venv --python 3.13 .venv-gpu                    # окружение суррогата, точные версии
uv pip install --python .venv-gpu/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  -r requirements-gpu.txt
```

## 3. Веса суррогата (обязательны для плана A)

Обученная голова TimesFM 3.0 с девятью экономическими целями и файл геологии, на
котором она обучалась. Файлы велики для git и передаются отдельным артефактом
(пакет «timesfm-economic-regimes-precise-z-20260910» в реестре пакетов
репозитория либо по запросу у команды). Хеши SHA-256 обязательны к проверке:

| Файл | SHA-256 | Размер |
|---|---|---|
| `training/full-model.pt` | `5be40e0cd1bae70e9b30ea7def1454f8033e66b2d100323f70827052e24a3712` | 1 324 373 140 |
| `training/report.json` | `216c198fa64a7b868995d86e548a517c62bd81bd1272a7d4d913f25f8abbced7` | — |
| `model-z/connectivity.json` | `cff65939ad943dd1df28460306fc433663a22707bd66a087732077135f7992c0` | 368 109 |

Разложить, например, так:

```text
$R/timesfm-economic-regimes-precise-z-20260910/training/full-model.pt
$R/timesfm-economic-regimes-precise-z-20260910/training/report.json
$R/static-head-geology-20260909/model-z/connectivity.json
```

где `R` — корень результатов (любой каталог). Без весов доступен план B: полный
путь с банком допустимых режимов и дообучением от базового чекпойнта
`google/timesfm-3.0-pytorch` (часы GPU; `scripts/run_case_z.sh` без `--plan-a`).

## 4. Переменные окружения

```bash
set -a; . config/llm.env; set +a          # маршрут и ключ LLM (шаблон: config/llm.env.example)
export R=/data/timesoil-results           # корень результатов; каталог прогона создаётся внутри
export PY_PROJECT=$PWD/.venv/bin/python PY_TORCH=$PWD/.venv-gpu/bin/python
export WEIGHTS=$R/timesfm-economic-regimes-precise-z-20260910/training/full-model.pt
export WEIGHTS_REPORT=$R/timesfm-economic-regimes-precise-z-20260910/training/report.json
export HEAD_CONNECTIVITY=$R/static-head-geology-20260909/model-z/connectivity.json
export CUDA_VISIBLE_DEVICES=0
export CPUS=0-15 OPM_CPU_AFFINITY=16-31   # ядра поиска и ядра OPM (16 MPI × 1 поток) внутри cpuset машины
```

Профиль ограничений кейса — `config/case_z_test.json` (лимиты 600/600 м³/сут,
забойные давления 50/300 атм, коридор компенсации 0,85–1,15 по трёхмесячному
окну, 16 обязательных ремонтов). Другой профиль — `PROFILE=<файл>`.

## 5. Запуск

```bash
PYTHONPATH=src:scripts $PY_PROJECT scripts/intake_case_z.py inspect /path/Model_Z_case.zip   # раскладка, ~3 с
bash scripts/run_case_z.sh /path/Model_Z_case.zip --plan-a            # ≈ 35–45 мин
# расписание кейса заканчивается на отсечении 31.12.2006 -> добавить --extend-schedule
```

Этапы и их журналы — в `$R/case-z-<метка>/`: `run.log`, `protocol.json`,
`<этап>.log`, `<этап>.exit`. Первый ненулевой код останавливает цепочку;
каталог прогона не переиспользуется.

## 6. Результат

| Что | Где |
|---|---|
| Печать выбора (до OPM) | `search/selection-before-opm.json`, SHA-256 в `run.log` |
| Финальный прогон OPM | `final/<run>/` — `manifest.json`, `canonical/chdd.csv`, `summary-report.txt` |
| Официальный ЧДД за период управления | `final/<run>/economics-2007/manifest.json` → `management_period.total_chdd_m` (млн руб.); сводка — `final/audit.json` |
| Расписание управления за период управления | блок после маркера `-- TIMESOIL AIOS GENERATED` в `final/<run>/input/<deck>/<schedule>.inc`: `sed -n '/TIMESOIL AIOS GENERATED/,$p' <schedule>.inc > wells_schedule.inc` |
| Отчёт интерпретируемости | `explain/interpretability/report.json`, `summary.md` |
| Веб-интерфейс результатов | `docker compose --env-file .env.example up -d --build` → `http://127.0.0.1:8000/app/` (`TIMESOIL_RESULTS_DIR=$R`) |

Заявленный ЧДД — только из `economics-2007` финального прогона; прогнозный ЧДД
поиска официальным не является.
