# Инструкция оператора

## 1. Окружение

```bash
uv sync --locked
uv run pytest -q
```

Крупные входы хранить вне Git. Для каждого входа заранее зафиксировать SHA-256.

Закреплённый симулятор:

```text
openporousmedia/opmreleases:2026.04_amd64@sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9
```

## 2. Сценарии OPM

Эти команды выполняются на операторском хосте с Docker. API-контейнер не
получает доступ к Docker socket.

Подготовить вне Git:

- `inputs/Model_Z_final_OPM.zip` — исходная модель организаторов;
- `inputs/Model_Z/Model_Z_sch.inc` — исходный файл расписания из архива;

Канонические базовые управления уже входят в поставку:
`examples/model_z_baseline_controls_v4.csv`. Файл содержит 38 213 записей для
103 скважин за каждый месяц 1994-11-01..2025-09-01 с полями
`date,well,control_value,control_target,status`.

```bash
RUN_ROOT=artifacts/track2-model-z-v4
BUNDLE="$RUN_ROOT/scenario-bundle"
BATCH="$RUN_ROOT/scenario-runs"
MODEL_Z_SOURCE=inputs/Model_Z_final_OPM.zip
BASELINE_SCHEDULE=inputs/Model_Z/Model_Z_sch.inc
BASELINE_CONTROLS=examples/model_z_baseline_controls_v4.csv

SOURCE_SHA256="$(sha256sum "$MODEL_Z_SOURCE" | awk '{print $1}')"
test "$SOURCE_SHA256" = \
  4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa
BASELINE_CONTROLS_SHA256="$(sha256sum "$BASELINE_CONTROLS" | awk '{print $1}')"
test "$BASELINE_CONTROLS_SHA256" = \
  1a92c1e031ab7dca843f3f8824070f7fe85a2955fa270d45eb66b0638e88752f

uv run python scripts/generate_track2_scenarios.py \
  "$BASELINE_CONTROLS" "$BASELINE_SCHEDULE" "$BUNDLE" \
  --scenario-count 10 --seed 20260831 \
  --perturbation-fraction 0.15 --liquid-rate-scale 1.0

SCENARIO_INDEX_SHA256="$(sha256sum "$BUNDLE/index.json" | awk '{print $1}')"
test "$SCENARIO_INDEX_SHA256" = \
  69697fede3bafe9fd50f7ba568a7aaec3d2f98a9726fde94595feea82f10e317
BASELINE_CHDD_SHA256=446c24eaa063710422835a745be157abdce66d602c75f33de50a8e75881d3884

uv run python scripts/run_track2_scenarios.py \
  "$MODEL_Z_SOURCE" "$BUNDLE" "$BATCH" \
  --source-sha256 "$SOURCE_SHA256" \
  --scenario-index-sha256 "$SCENARIO_INDEX_SHA256" \
  --baseline-chdd-sha256 "$BASELINE_CHDD_SHA256" \
  --schedule-relative-path Model_Z/Model_Z_sch.inc \
  --deck Model_Z/Model_Z.data \
  --timeout-seconds 7200 --parsing-strictness low
```

Прогоны выполняются строго последовательно. Ожидаемая раскладка:
`$BATCH/dataset/{baseline,perturbation-001..009}.csv`,
`$BATCH/manifests/{baseline,perturbation-001..009}.json` и общий
`$BATCH/manifest.json`. Несовпадение любого хеша завершает команду ошибкой.

## 3. Суррогат

```bash
uv run python scripts/train_track2_surrogate.py \
  --dataset "$BATCH/dataset" --manifest "$BATCH/manifests" \
  --batch-manifest "$BATCH/manifest.json" \
  --scenario-index-sha256 "$SCENARIO_INDEX_SHA256" \
  --output "$RUN_ROOT/training" --test-fraction 0.25 \
  --ensemble-size 5 --n-estimators 160 --horizon 6 \
  --seed 20260831 --conformal-level 0.90
```

## 4. Поиск и финальный повторный расчёт

```bash
uv run python scripts/search_track2_schedule.py search \
  "$RUN_ROOT/training/model" "$BATCH/dataset/baseline.csv" \
  "$BATCH/manifests/baseline.json" "$RUN_ROOT/training/metrics.json" \
  "$MODEL_Z_SOURCE" "$BASELINE_SCHEDULE" "$RUN_ROOT/search" \
  --scenario-id baseline --start-date 2007-01-01 \
  --candidate-count 500 --seed 20260831 --perturbation-fraction 0.05 \
  --uncertainty-weight 1 --injection-cost-equivalent 0.01 \
  --deck Model_Z/Model_Z.data \
  --schedule-relative-path Model_Z/Model_Z_sch.inc \
  --timeout-seconds 3600 --parsing-strictness low

uv run python scripts/search_track2_schedule.py replay \
  "$MODEL_Z_SOURCE" "$RUN_ROOT/search" "$RUN_ROOT/search-final-opm" \
  --deck Model_Z/Model_Z.data \
  --schedule-relative-path Model_Z/Model_Z_sch.inc \
  --timeout-seconds 3600 --parsing-strictness low

sha256sum \
  "$RUN_ROOT/training/metrics.json" \
  "$RUN_ROOT/training/model/manifest.json" \
  "$RUN_ROOT/search/manifest.json" \
  "$RUN_ROOT/search/lineage.json" \
  "$RUN_ROOT/search-final-opm/final-replay-receipt.json"
```

Итогом являются `wells_schedule.inc`, изменённый include Model Z, квитанция
повторного расчёта и результат ЧДД. Поиск изменяет только управления добывающих
скважин; управления каждой нагнетательной скважины остаются базовыми. Поиск
выполняет только отбор; улучшение можно заявлять лишь по терминальной квитанции
повторного OPM. Прогнозное значение суррогата не является итоговым ЧДД.

## 5. Единый внешний Qwen → OPM → export → ЧДД

Команду выполнять на операторском хосте из чистого зафиксированного Git checkout:
она проверяет HEAD и хеши исполняемых файлов до и после расчёта. Нужны Docker с
закреплённым выше образом OPM Flow, внешний HTTPS endpoint `/v1` и секретный файл
ключа; Docker socket в API-контейнер не передаётся.

Обязательное окружение и один запуск без перезаписи:

```bash
export LLM_BASE_URL=https://qwen-api.example/v1
export LLM_MODEL=qwen3.6-35b-a3b
export LLM_TIMEOUT_SECONDS=120
export LLM_MAX_OUTPUT_TOKENS=4096
test -s secrets/qwen_api_key
export LLM_API_KEY="$(<secrets/qwen_api_key)"

uv run timesoil-aios full-cycle inputs/full-cycle-request.json \
  --runs-dir artifacts/full-cycle \
  --run-id model-z-full-cycle-v1 \
  --timeout 7200
```

`inputs/full-cycle-request.json` — JSON-объект только со следующими корневыми
ключами:

- обязательные `context`, `controls`, `source`, `deck`,
  `schedule_relative_path`, `scenario_id`, `source_model`, `start_year`;
- необязательные `parsing_strictness`, `density_map`, `charge_initial_pump`;
- `context` — объект без ключей секретов, для этого контура `track` равен `2`;
- `source` — ZIP, каталог или DATA-файл Model Z относительно каталога request;
  `deck` и `schedule_relative_path` — безопасные относительные POSIX-пути;
- `source_model` имеет точное значение `model_z_opm`; `start_year` — полный год;
- `controls` содержит ровно один объект на каждую управляющую скважину и каждый
  из шести последовательных месяцев. Для текущего подготовленного schedule это
  **618 объектов = 6 × 103 скважины**. Набор 103 имён извлекается из WCON-записей
  фактического snapshot schedule, а не доверяется request;
- каждый control имеет ровно поля `month`, `well`, `role`, `status`, `target`,
  `value`: месяц `YYYY-MM-01`; роли `producer`/`injector`; статусы `OPEN`/`SHUT`;
  цель производителя `ORAT` или `LRAT`, нагнетателя — `WRAT`; значение — конечное
  неотрицательное число, для `SHUT` только `0`.

Минимальная форма request, где `controls` необходимо заполнить всеми 618
объектами по указанному контракту:

```json
{
  "context": {"track": 2, "facts": {}, "constraints": {}},
  "controls": [],
  "source": "Model_Z_final_OPM.zip",
  "deck": "Model_Z/Model_Z.data",
  "schedule_relative_path": "Model_Z/Model_Z_sch.inc",
  "scenario_id": "model-z-full-cycle-v1",
  "source_model": "model_z_opm",
  "start_year": 2007,
  "parsing_strictness": "strict",
  "charge_initial_pump": false
}
```

При успехе CLI печатает JSON с путём
`artifacts/full-cycle/model-z-full-cycle-v1/full-cycle-receipt.json` и его
SHA-256. Квитанция имеет схему `timesoil.aios.track2-full-cycle/v1`, режим `0444`,
содержит привязку Git/source map, хеш полного набора controls и source inventory,
закреплённый OPM, аутентифицированные SUMMARY/export, официальный ЧДД и честное
решение критика; `organizer_certified=false`.

Коды завершения: `0` — терминальные ворота пройдены и критик одобрил; `2` —
терминальные артефакты и квитанция получены, но критик отклонил; `1` — fail-closed
ошибка без подтверждённой успешной квитанции. Команда отказывает до Qwen/OPM
execution при неполных 6×103 controls, пропущенной или неизвестной скважине,
секрете в context, грязных исполняемых файлах либо существующем run-id. Ошибка
Qwen, OPM, SUMMARY/export, ЧДД, изменение source/commit или хеша артефакта также
не создаёт `complete=true` receipt. Автоматического retry и локальной LLM нет.

Файлы `deliverables/track2_model_z/*.json` являются публичными сводками, не
самостоятельными receipts; для проверки запуска нужны `full-cycle-receipt.json`
и перечисленные в нём хешированные манифесты.

## 6. AIOS API

```bash
cp config/aios.example.env config/aios.env
# заменить LLM_BASE_URL в config/aios.env на выданный внешний HTTPS endpoint
mkdir -p secrets
chmod 700 secrets
printf '%s' "$QWEN_API_KEY" > secrets/qwen_api_key
chmod 600 secrets/qwen_api_key
docker compose --env-file .env.example up -d --build --wait --wait-timeout 180
curl --noproxy '*' --fail http://127.0.0.1:8000/health
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/capabilities
```

Ключ не передавать аргументом командной строки и не добавлять в Git.
`docker compose --env-file .env.example down` останавливает сервис и сохраняет
том расчётов. Удалять том можно только после отдельного копирования результатов.

Проверка четырёх ролей на операторском хосте:

```bash
set -a
. ./config/aios.env
set +a
export LLM_API_KEY="$(<secrets/qwen_api_key)"
uv run timesoil-aios agent-experiment examples/agent_context.json
```

Контекст намеренно незавершён: корректный критик должен потребовать численные
доказательства, а не объявить готовность по текстовой рекомендации.
