# Инструкция оператора · Трек 2, Model Z

Рабочая цепочка выполняется строго в этом порядке: банк сценариев OPM →
связность → обучение Google TimesFM → оценка на закреплённом наборе → поиск
графика по прогнозу и фиксация выбора → **одна** финальная проверка OPM →
парный аудит. Прогнозный ЧДД никогда не является официальным: официальным
считается только результат калькулятора организаторов на выходе OPM при
сопоставимой парной базе.

Разделы 2–7 — вычислительные, они выполняются на A100. Разделы 1, 8, 9
выполняются и в WSL.

## 1. Окружение

### WSL: код, документация, доставка

```bash
uv sync --locked
uv run python -m compileall -q src scripts
uv run pytest tests -q
uv run timesoil-aios doctor
```

`pythonpath = ["src", "scripts"]` задан в `pyproject.toml`, отдельный
`PYTHONPATH` для локального прогона тестов не нужен.

Код доставляется только через Git: коммит в WSL, push, затем на A100
`git pull --ff-only` либо `git worktree add --detach <SHA>`. Копирование
исходников и правка на сервере не используются.

### A100: все расчёты

```text
репозиторий        /root/projects/TimesOil
результаты         R=/root/projects/TimesOil/results/audit-20260909
venv проекта       /root/projects/TimesOil/.venv/bin/python
venv Torch/TimesFM /tmp/timesoil-kt3-20260908/venv/bin/python   (tmpfs, теряется при перезагрузке)
GPU                физический GPU 5 → CUDA_VISIBLE_DEVICES=5 (cuda:0 внутри процесса)
закреплённые деки  /tmp/timesoil-kt2/model_z/Model_Z_final_OPM.zip
```

Доступ: `ssh -o BatchMode=yes -o ConnectTimeout=10 a100-remote`. Карта и хост
общие с другими проектами: до запуска проверить `nvidia-smi`, чужие процессы и
tmux-сессии не останавливать. Длительные задания запускать в `tmux` с
уникальным именем, зафиксировав PID, каталог и время.

Закреплённый симулятор:

```text
openporousmedia/opmreleases:2026.04_amd64@sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9
```

Крупные входы хранятся вне Git, для каждого заранее фиксируется SHA-256.
Исходная модель организаторов:
`Model_Z_final_OPM.zip`, SHA-256
`4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa`;
deck `Model_Z/Model_Z.data`, include расписания `Model_Z/Model_Z_sch.inc`.

### Размещение OPM по ядрам

Общий `OpmFlowRunner` читает переменные окружения:

- `OPM_MPI_PROCESSES` — число MPI-процессов, 1–64, по умолчанию 1;
- `OPM_THREADS_PER_PROCESS` — потоки OpenMP на процесс;
- `OPM_CPU_AFFINITY` — список CPU для `taskset` внутри контейнера;
- `OPM_WORKER_CPU_AFFINITIES` — наборы через `;` для `run_track2_scenarios.py`,
  по одному набору на каждого `--workers`.

Доступные контейнеру наборы физических ядер — **14-29** и **32-47**, по одному
NUMA-узлу на сценарий. Ядра 48-63 контейнеру полностью не доступны; перед
использованием новых наборов проверять фактический `cpuset.cpus.effective`.
Измерения и точность MPI: [A100: производительность](A100_PERFORMANCE_20260909.md).

Каждый эксперимент пишет в **новый** каталог. Драйверы используют
`mkdir(exist_ok=False)` и `open('x')` намеренно: повторный запуск поверх
существующего результата запрещён, а не «чинится».

## 2. Банк сценариев OPM

Базовый прогон, канонический экспорт и десять полных 224-месячных сценариев
готовит один драйвер; он сам вызывает `scripts/generate_track2_scenarios.py`
и `scripts/run_track2_scenarios.py`.

```bash
export OPM_MPI_PROCESSES=16 OPM_THREADS_PER_PROCESS=1
export OPM_CPU_AFFINITY=32-47
export OPM_WORKER_CPU_AFFINITIES='14-29;32-47'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python \
  scripts/benchmark_bhp_surrogate.py \
  --source /tmp/timesoil-kt2/model_z/Model_Z_final_OPM.zip \
  --output "$R/<новый-каталог>" --workers 2
```

Драйвер проверяет SHA-256 архива, выполняет базовый OPM (либо переиспользует
готовый через `--reference-run <каталог>`), экспортирует канонический baseline,
требует ровно 224 месяца × 103 скважины, затем генерирует bundle сценариев
(`--scenario-count 10 --seed 20260909 --perturbation-fraction 0.15
--bhp-perturbation-fraction 0.15`) и считает их с `--include-bhp`.
Раскладка результата: `scenario-bundle/index.json`, `scenario-runs/dataset/*.csv`,
`scenario-runs/manifests/*.json`, общий `scenario-runs/manifest.json`, `plan.json`.
Несовпадение любого хеша завершает команду ошибкой. `--self-check` выполняет
только встроенные проверки выравнивания периода без расчёта.

Отдельные генерация и прогон, если bundle нужен вне драйвера:

```bash
uv run python scripts/generate_track2_scenarios.py \
  <baseline-controls.csv> <Model_Z_sch.inc> <bundle> \
  --scenario-count 10 --seed 20260909 \
  --perturbation-fraction 0.15 --bhp-perturbation-fraction 0.15

uv run python scripts/run_track2_scenarios.py \
  <Model_Z_final_OPM.zip> <bundle> <scenario-runs> \
  --source-sha256 <sha> --scenario-index-sha256 <sha> \
  --baseline-chdd-sha256 <sha> \
  --schedule-relative-path Model_Z/Model_Z_sch.inc \
  --deck Model_Z/Model_Z.data --parsing-strictness low \
  --timeout-seconds 7200 --include-bhp --workers 2
```

Два дополнительных набора расширяют покрытие режимов.

Расходные режимы `physical-sweep-NN` — восемь пар множителей добычи/закачки
относительно переданного incumbent-запроса, каждый прогоняется полным циклом:

```bash
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python \
  scripts/run_full_period_sweep.py <incumbent-request.json> <baseline-run> <output>
```

Режимы давления `bhp-only-NN` — фиксированные расходы, роли и статусы,
изменяется только граница BHP:

```bash
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python \
  scripts/run_bhp_validation.py \
  --request <request.json> --reference <канонический-экспорт> \
  --output <output> --transition-coverage
```

Взаимоисключающие наборы плана: `--transition-coverage` (режимы разработки),
`--uncertainty-validation` (закреплённые калибровочные/тестовые случаи, никогда
не входящие в разработку), `--local-reference-evaluation`; без флага
используется набор по умолчанию. `--self-check` считает только проверки плана.
Разбиение случаев зафиксировано: calibration `[0, 1, 3, 4, 6]`,
test `[2, 5, 7]`; отбор модели на тестовых случаях запрещён.

## 3. Связность

```bash
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python \
  scripts/export_opm_connectivity.py <проверенный-OPM-run> <output>
```

Скрипт сжимает экспортированные проводимости INIT/EGRID, включая параллельные
и несоседние соединения (3865 NNC), в `connectivity.json`. Используемый файл Z:
`R/static-head-geology-20260909/model-z/connectivity.json`, SHA-256
`cff65939ad943dd1df28460306fc433663a22707bd66a087732077135f7992c0`.
Тот же файл передаётся в обучение, оценку и поиск.

## 4. Обучение TimesFM

Обучаются девять экономических выходов (`WOMR`, `WLPR`, `WWIR`, `THP`=WBP9,
`BHP`, `WEFF`, `WOMT_Diff`, `WLPT_Diff`, `WWIT_Diff`). Обучение идёт по
**всей** сети: `--unfreeze-backbone`; слово `head` в именах CLI и файлов не
означает обучение только последнего слоя.

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:scripts taskset -c 14-29 \
  /tmp/timesoil-kt3-20260908/venv/bin/python scripts/finetune_timesfm_head.py \
  --batch "$R/bhp-training-v3-20260909/scenario-runs" \
  --batch-sha256 4dbab179f94ca1800b052e1346591a685eb9fa2d8dc900d917fc6ad66d149893 \
  --connectivity "$R/static-head-geology-20260909/model-z/connectivity.json" \
  --initial-head "$R/timesfm-economic-targets-z-20260910/training/full-model.pt" \
  --initial-head-sha256 9b954acbe64f0c5ff8dc213315a46b9324a948294ac19bc3b087d55fb50844ae \
  --regime-calibration "$R/physical-z-forecast-validation-20260909" \
  --regime-calibration-sha256 69a92d83bc7824b68af1eaddbddd884b589e4009b7d12de4d23be2e0a227277f \
  --bhp-calibration "$R/transition-coverage-z-20260910/scenarios" \
  --bhp-calibration-sha256 9bffec89541467afaf904819ad77baa4469bd4d726ff55d3e08f2ba92c2fae8d \
  --output "$OUT/training" --epochs 60 --learning-rate 1e-5 \
  --unfreeze-backbone --condition-last-layer --condition-first-layer \
  --cold-start-normalization --retain-initial-scale \
  --economic-targets --precise-variate-softmax
```

Выход: `training/full-model.pt` и `training/report.json`. В отчёте проверяются
`complete`, `checkpoint_sha256` и девять `economic_targets`. Первый прогон без
`--initial-head`/`--regime-calibration`/`--bhp-calibration` даёт стартовый
40-эпоховый checkpoint; приведённая команда продолжает обучение с него.

`--precise-variate-softmax` считает variate softmax в FP64 и возвращает FP32:
это устраняет неповторяемость CUDA, метаданные восстанавливают режим при
загрузке. Допуск проверки decoder и FP64 не ослаблять ради прохождения теста
или обхода OOM.

Измерено 10 сентября: 60 эпох, лучшая 60, около **5486,7 с**; validation loss
0,1740743965 → 0,06936201453.

Разбиение зафиксировано и не меняется: train — `baseline`,
`perturbation-001/002/003/005/006`, `physical-sweep-00/01/02/07`,
`bhp-only-00/01/03/06`; validation — `perturbation-009`, `physical-sweep-04`,
`bhp-only-04`; development test — `perturbation-004/007/008`. Обучение или
отбор модели на тестовых сценариях запрещены.

## 5. Оценка на закреплённом наборе

```bash
OUT="$R/economic-uncertainty-z-<метка>"
test ! -e "$OUT" && mkdir "$OUT"
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:scripts taskset -c 14-29 \
  /tmp/timesoil-kt3-20260908/venv/bin/python scripts/evaluate_timesfm_scenarios.py \
  --batch "$R/fresh-uncertainty-z-20260910/scenarios" \
  --batch-sha256 4a9a8e53f8a44813fae8211d3a902f43f3c251b747d623c5dc47e78697a8eeef \
  --head-report "$MODEL/report.json" --head "$MODEL/full-model.pt" \
  --connectivity "$R/static-head-geology-20260909/model-z/connectivity.json" \
  --fixed-origin-only --output "$OUT/evaluation" >"$OUT/evaluation.log" 2>&1
printf '%s\n' "$?" >"$OUT/evaluation.exit"
```

Режим: фиксированная точка прогноза, 103 скважины, 224 месяца, девять целей;
неизвестные будущие наблюдения маскируются `NaN`. Физическая reference-траектория
из будущего в прогноз не подаётся. После прогона проверить `complete`, SHA
модели и отчёта, все восемь исходных сценариев, девять `radius_by_target`,
исключение обучающих случаев и фактические ошибки/покрытие.

**Открытая проблема.** Последний запуск завершился CUDA OOM на
`torch.softmax(..., dtype=float64)`: запрошено 3,13 GiB при лимите процесса
13,82 GiB (доля памяти GPU `0.35`, жёстко задана в
`scripts/evaluate_timesfm_scenarios.py`); обучение в том же режиме помещалось
с долей 0,60. Это ресурсная ошибка запуска, метрик она не дала. Минимальный
следующий шаг — согласовать лимит evaluator с доступной памятью после проверки
загрузки GPU 5. Не уменьшать фонд и горизонт, не отключать FP64 и не менять
разбиение ради прохождения. Подробности — §11 передачи
[10 сентября](HANDOFF_CLAUDE_CODE_20260910.md).

## 6. Поиск графика и фиксация выбора

Внутри поиска **нет ни одного вызова OPM**: варианты ранжируются официальным
калькулятором по прогнозным экономическим рядам. Недопустимые по собственным
заданиям скважин варианты отбрасываются **до** ранжирования.

Окружение поиска (основной маршрут — Cerebras `qwen-3.8-27b` с высоким
рассуждением, нулевой температурой и фиксированным seed; с A100 только через
локальный прокси, прямой доступ к `api.cerebras.ai` закрыт гео-блоком):

```bash
export LLM_BASE_URL=https://api.cerebras.ai/v1
export LLM_MODEL=qwen-3.8-27b
export LLM_REASONING_EFFORT=high LLM_SEED=20260909
export LLM_PROXY_URL=http://127.0.0.1:10809
export LLM_TIMEOUT_SECONDS=600
export LLM_MAX_OUTPUT_TOKENS=8192
test -s /root/.config/timesoil/cerebras-key
export LLM_API_KEY="$(</root/.config/timesoil/cerebras-key)"
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPM_MPI_PROCESSES=16 OPM_THREADS_PER_PROCESS=1 OPM_CPU_AFFINITY=30-45
export TIMESOIL_CASE_SOURCE_SHA256=<sha256 архива кейса>
```

Резервный маршрут — Татнефть (`LLM_BASE_URL=https://litellm.tatneft.guru/v1`,
`LLM_MODEL=qwen3.8-27b`, ключ в `/dev/shm/timesoil-tatneft-20260909-key`).

Ключ читается из runtime-файла в переменную окружения процесса. Проверка
наличия — `test -s`, не `cat`. Ключ не печатать, не коммитить, не включать в
логи и бандлы.

```bash
PYTHONPATH=src:scripts \
  /tmp/timesoil-kt3-20260908/venv/bin/python scripts/propose_track2_policies.py \
  <baseline-view> <request.json> "$OUT/search" \
  --rounds 3 --economic-selection \
  --case-profile config/case_z_test.json \
  --search cma --search-seconds 900 --blocks "$OUT/blocks.json" \
  --head "$MODEL/full-model.pt" --head-sha256 <sha> \
  --head-report "$MODEL/report.json" \
  --connectivity "$OUT/connectivity.json"
```

`--case-profile` включает профиль кейса: ворота G1 (задания), G2 (прогноз и
производные векторы), ремонт масштабированием вместо штрафов, 16 обязательных
ремонтов. `--search cma` заменяет перебор сетки поиском CMA-ES по 18-мерному
пространству политик (`policy_space.py`, `cma_search.py`) с затравками Sobol и
предложениями агентов блоков (`--blocks`, `planning.py`); `--search grid`
оставляет прежний перебор. Все ворота одинаково жёсткие: 600/600 в каждый
момент, ВКЗ 0,85–1,15 по трёхмесячному окну с обеих сторон, забойные давления,
ремонты. След поиска — `search/search_trace.json`, элита — `search/elite.json`;
оба входят в печать выбора.

`<request.json>` — полный запрос управления на 224 месяца (контракт — раздел 7),
`<baseline-view>` — аутентифицированный канонический экспорт исходного графика.
Протокол агентных раундов: до 3 ограниченных раундов, одна попытка исправления
невалидного предложения; после двух невалидных ответов раунд отклоняется;
для успешного поиска нужен минимум один согласованный агентный вариант.

Фиксация выполняется внутри поиска через
`track2_final_selection.seal_forecast_selection`: результат —
`search/selection-before-opm.json`. Его SHA-256 и есть печать выбора:

```bash
sha256sum "$OUT/search/selection-before-opm.json"
```

После вычисления печати выбор изменить нельзя.

## 7. Единственная финальная проверка

```bash
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python \
  scripts/track2_final_selection.py \
  "$OUT/search" "$R/timesfm-bhp-policy-20260909/cycles/baseline" "$OUT/final" \
  --seal-sha256 <sha печати>
```

Аргументы позиционные: каталог поиска, каталог парного baseline-прогона, новый
каталог результата; `--seal-sha256` обязателен. Скрипт сверяет печать,
записывает `search/final-verification-attempt.json`, запускает **ровно один**
`timesoil-aios full-cycle` для зафиксированного графика и формирует
`final/final-audit.json` и `final/selected/full-cycle-receipt.json`.

Выбор другого графика после получения физического результата запрещён.
Sealed-запуск неизменяем: не перезапускать, не переназначать, не использовать
его физику для перевыбора. Попытку не стирать даже при неудаче.

### Контракт запроса full-cycle

`request.json` — JSON-объект только со следующими корневыми ключами:

- обязательные `context`, `controls`, `source`, `deck`,
  `schedule_relative_path`, `scenario_id`, `source_model`, `start_year`;
- необязательные `parsing_strictness`, `density_map`, `charge_initial_pump`,
  `horizon_months` (по умолчанию 6; для полного периода Z — 224);
- `context` — объект без ключей секретов, для этого контура `track` равен `2`;
- `source` — ZIP, каталог или DATA-файл Model Z относительно каталога request;
  `deck` и `schedule_relative_path` — безопасные относительные POSIX-пути;
- `source_model` имеет точное значение `model_z_opm`; `start_year` равен году
  первого месяца управления;
- `controls` содержит ровно один объект на каждую управляющую скважину и каждый
  из `horizon_months` последовательных месяцев. Для полного периода Z это
  **23 072 объекта = 224 × 103 скважины**. Набор имён скважин извлекается из
  WCON-записей фактического snapshot расписания, а не доверяется request;
- каждый control имеет поля `month`, `well`, `role`, `status`, `target`,
  `value` и необязательный `bhp_limit`: месяц `YYYY-MM-01`; роли
  `producer`/`injector`; статусы `OPEN`/`SHUT`; цель производителя `ORAT` или
  `LRAT`, нагнетателя — `WRAT`; значение — конечное неотрицательное число, для
  `SHUT` только `0`; `bhp_limit` — конечное положительное число.

Минимальная форма, где `controls` заполняется по указанному контракту:

```json
{
  "context": {"track": 2, "facts": {}, "constraints": {}},
  "controls": [],
  "source": "Model_Z_final_OPM.zip",
  "deck": "Model_Z/Model_Z.data",
  "schedule_relative_path": "Model_Z/Model_Z_sch.inc",
  "scenario_id": "model-z-final",
  "source_model": "model_z_opm",
  "start_year": 2007,
  "parsing_strictness": "low",
  "charge_initial_pump": false,
  "horizon_months": 224
}
```

Отдельный запуск полного цикла вне финальной проверки:

```bash
uv run timesoil-aios full-cycle <request.json> \
  --runs-dir <runs> --run-id <идентификатор> --timeout 7200
```

При успехе CLI печатает JSON с путём `<runs>/<run-id>/full-cycle-receipt.json`
и его SHA-256. Квитанция имеет схему `timesoil.aios.track2-full-cycle/v1`,
режим `0444`, содержит привязку Git/source map, хеш полного набора controls и
source inventory, закреплённый OPM, аутентифицированные SUMMARY/export,
официальный ЧДД и честное решение критика; `organizer_certified=false`.

Коды завершения: `0` — терминальные ворота пройдены и критик одобрил; `2` —
терминальные артефакты и квитанция получены, но критик отклонил; `1` —
fail-closed ошибка без подтверждённой успешной квитанции. Команда отказывает
до вызова Qwen и OPM при неполном наборе controls, пропущенной или неизвестной
скважине, секрете в `context`, грязных исполняемых файлах либо существующем
`run-id`. Ошибка Qwen, OPM, SUMMARY/export, ЧДД, изменение source/commit или
хеша артефакта также не создаёт `complete=true` receipt. Автоматического retry
и локальной LLM нет. При отклонении плана до OPM сохраняется отдельный
`<run-id>.planning-rejected.json` с решениями ролей; подготовленный каталог
удаляется. Одобрение планирования разрешает расчёт, финальное решение остаётся
за критиком после получения терминальных доказательств.

### Парный аудит результата

Сравнение двух завершённых прогонов официальным ЧДД полного периода:

```bash
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python \
  scripts/compare_track2_cycles.py \
  <baseline-run> <candidate-run> <output.json> --expected-months 224
```

Скрипт проверяет уже существующие результаты, не запускает OPM и отказывается
перезаписывать итог. `--expected-months` отбраковывает укороченное окно.
Воспроизведение самой симуляции требует оригинальных архивов и закреплённого
Docker-образа.

### Экономические соглашения

OPM хранит конец отчётного интервала: продукция январского управления берётся
из отчёта 1 февраля. Для калькулятора даты сдвигаются на месяц назад; отчёты
после конца управления исключаются. История сохраняет состояние насосов и
официальное распределение годового налога; её денежные потоки в итог не входят.
`result.json` и Excel остаются исходными результатами официального
калькулятора; сдаваемый итог находится в
`manifest.json.management_period.total_chdd_m` и терминальной квитанции.
Суммарный исторический `summary.totalChddM` подменять этим итогом нельзя.

Файлы `deliverables/track2_model_z/*.json` — публичные сводки, не
самостоятельные receipts; для проверки запуска нужны `full-cycle-receipt.json`
и перечисленные в нём хешированные манифесты.

### Измеренная длительность, 10 сентября

| Этап | Время |
|---|---:|
| Поиск целиком | 18 мин 33 с |
| — 8 вариантов сетки | 1 мин 49 с |
| — 3 раунда Qwen | ≈5,5 мин на раунд |
| Фиксация выбора | 3 с |
| Финальный full-cycle | 7 мин 36 с |
| — OPM внутри него | 3 мин 37 с |
| **Итого** | **26 мин 13 с** |

## 7а. Прогон тестового кейса одной командой

Разделы 2–7 описывают цепочку по шагам — так она собиралась и проверялась. В день
кейса та же цепочка запускается одной командой: `scripts/run_case_z.sh ARCHIVE.zip`.
Скрипт делает приём архива, базовый цикл инкумбента, связность и блоки **из этого же
прогона**, банк и его прогоны OPM, сборку батча, дообучение с 60-эпохной головы, поиск,
**единственную** финальную проверку и отчёт интерпретируемости. На каждый этап — свой
каталог, `<этап>.log`, `<этап>.exit`, время и запись в `protocol.json` с хешами выходов;
первый ненулевой `exit` останавливает цепочку.

```bash
scripts/run_case_z.sh /root/projects/case_z_20260911/case_z.zip --dry-run   # печать всех команд
scripts/run_case_z.sh /root/projects/case_z_20260911/case_z.zip             # полный путь, ≈2,5–3 ч
scripts/run_case_z.sh /root/projects/case_z_20260911/case_z.zip --plan-a    # быстрый путь, ≈50 мин
```

По умолчанию: банк включён (`--bank-runs 16`), дообучение включено (`--epochs 20`),
оценка на закреплённом наборе выключена (`--evaluate` включает; драйверу нужен сплит 5/3,
которого у банка кейса нет). Перед каждым необязательным этапом печатается остаток
бюджета (`--budget-seconds`, по умолчанию 4 часа).

Регламент дежурного на 17:00 — доставка архива и кода, сухой прогон, что смотреть в логе,
тайминги по этапам и запасные пути (план A, маршрут LLM, отличия раскладки архива):
[`docs/CASE_INTAKE_20260911.md`](CASE_INTAKE_20260911.md).

## 8. Веб-интерфейс и API

```bash
cp config/aios.example.env config/aios.env
mkdir -p secrets
chmod 700 secrets
printf '%s' "$QWEN_API_KEY" > secrets/qwen_api_key
chmod 600 secrets/qwen_api_key
docker compose --env-file .env.example up -d --build --wait --wait-timeout 180
curl --noproxy '*' --fail http://127.0.0.1:8000/health
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/capabilities
```

Сервис — `uvicorn timesoil.aios.api:app`. Эндпоинты: `GET /` (операторская
страница), `GET /health`, `GET /v1/capabilities`, `POST /v1/experiments/agents`,
`POST /v1/economics/chdd`.

Compose использует минимальный bootstrap с правами только на чтение секрета и
смену UID/GID: перед запуском API он сбрасывает capabilities и переходит на
UID/GID `10001`; сам API от root не работает. Docker socket в API-контейнер не
передаётся, поэтому OPM из контейнера API не запускается. Ключ не передавать
аргументом командной строки и не добавлять в Git.
`docker compose --env-file .env.example down` останавливает сервис и сохраняет
том расчётов; удалять том можно только после отдельного копирования результатов.

Профиль A100 — отдельное имя проекта и непересекающийся порт:

```bash
AIOS_PORT=18082 docker compose --env-file .env.example \
  -p scorp-timesoil-kt3 -f compose.yaml -f compose.a100.yaml \
  up -d --build --wait --wait-timeout 180
curl --noproxy '*' --fail http://127.0.0.1:18082/health
curl --noproxy '*' --fail http://127.0.0.1:18082/v1/capabilities
```

`compose.a100.yaml` подключает контейнер к сетевому пространству хоста,
сохраняя bind API на `127.0.0.1`. Профиль предназначен для Linux.

`connectivity_verified=false` в `/v1/capabilities` означает отсутствие сетевой
пробы внутри этого GET; доступность провайдера проверяется отдельным
`/v1/experiments/agents` или CLI.

### LLM-маршрут

Рабочий маршрут — прямой HTTPS к Татнефти: `LLM_BASE_URL`
`https://litellm.tatneft.guru/v1`, `LLM_MODEL` `qwen3.8-27b`. Клиент проверяет
соответствие модели и endpoint.

`LLM_PROXY_URL` — необязательный явный CONNECT-прокси на случай, когда прямой
выход недоступен (например `http://127.0.0.1:10809`). Пустое значение означает
прямое соединение. Системные `HTTP_PROXY`/`HTTPS_PROXY` клиент не читает. TLS
проверяется клиентом, ключ не передаётся прокси открытым текстом. Внутри
API-контейнера без host network `127.0.0.1` обозначает сам контейнер и не
ведёт к прокси хоста.

### Проверка четырёх ролей

```bash
set -a
. ./config/aios.env
set +a
export LLM_API_KEY="$(<secrets/qwen_api_key)"
uv run timesoil-aios agent-experiment examples/agent_context.json
```

Контекст намеренно незавершён: корректный критик должен потребовать численные
доказательства, а не объявить готовность по текстовой рекомендации.

Подкоманды CLI: `doctor` (готовность компонентов без секретов),
`agent-experiment`, `full-cycle`.

## 9. Проверки

Локально, в WSL:

```bash
uv run python -m compileall -q src scripts
uv run pytest tests -q
git diff --check
```

После численного или модельного изменения дополнительно выполнить профильные
проверки на A100 с venv проекта, например:

```bash
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python -m pytest -q \
  tests/test_timesfm_scenario_economics.py \
  tests/test_aios_operating_constraints.py \
  tests/test_policy_control_repair.py \
  tests/test_track2_final_selection.py
```

Метрики и хеши квитанций записывать в соответствующую запись `docs/` или
`deliverables/`. Локальная проверка не заменяет серверный прогон: если
выполнена только она, так и указывать.

**Связанные материалы:**
[передача 10 сентября](HANDOFF_CLAUDE_CODE_20260910.md) ·
[приёмочная матрица](BOTH_TRACKS_ACCEPTANCE_20260909.md) ·
[алгоритм обоих треков](ALGORITHM_TRACKS_1_2.md) ·
[управления, расходы и пробелы](CONTROL_COST_GAPS_20260909.md) ·
[A100: производительность](A100_PERFORMANCE_20260909.md).
