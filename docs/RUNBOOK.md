# Инструкция оператора

## 1. Окружение

```bash
uv sync --locked
uv run pytest tests -q
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
RUN_ROOT=artifacts/track2-model-z-kt3
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
  --seed 20260831 --conformal-level 0.90 \
  --interwell-source "$MODEL_Z_SOURCE"
```

## 4. Поиск и финальный повторный расчёт

```bash
uv run python scripts/search_track2_schedule.py search \
  "$RUN_ROOT/training/model" "$BATCH/dataset/baseline.csv" \
  "$BATCH/manifests/baseline.json" "$RUN_ROOT/training/metrics.json" \
  "$MODEL_Z_SOURCE" "$BASELINE_SCHEDULE" "$RUN_ROOT/search" \
  --scenario-id baseline --start-date 2007-01-01 \
  --candidate-count 500 --seed 20260831 --perturbation-fraction 0.05 \
  --perturb-injection --candidate-rank 0 \
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
повторного расчёта и результат ЧДД. `--perturb-injection` включает перераспределение
при неизменной месячной сумме заданных WRAT, ролях и статусах, с покважинной границей
`--perturbation-fraction`. Флаг требует модели, обученной с `--interwell-source`.
Без него закачка остаётся базовой. Для сравнения следующих кандидатов повторить
поиск с теми же параметрами, `--candidate-rank 1`, затем `2`, в новых каталогах;
каждый replay запускается по `final_replay_argv` его манифеста.
`--injection-only` сохраняет добывающие управления и требует `--perturb-injection`
при `--liquid-rate-scale 1`. Для проверки зависимости от широких интервалов
можно отдельно отобрать кандидата с `--uncertainty-weight 0`; OPM остаётся воротами. Прогоны OPM
выполнять последовательно. Улучшение подтверждается сравнением официального ЧДД
кандидата и baseline за один период с одинаковыми нормативами.

Экономика replay и full-cycle использует только шесть месяцев управления.
OPM хранит конец отчётного интервала: продукция январского управления берётся
из отчёта 1 февраля. Для калькулятора даты сдвигаются на месяц назад; отчёты
после конца управления исключаются. История сохраняет состояние насосов и
официальное распределение годового налога; её денежные потоки в итог не входят.
`result.json` и Excel остаются исходными результатами официального калькулятора;
сдаваемый итог находится в `manifest.json.management_period.total_chdd_m`
и терминальной квитанции. Суммарный исторический `summary.totalChddM` подменять
этим итогом нельзя.

## 5. Единый внешний Qwen → OPM → export → ЧДД

Команду выполнять на операторском хосте из чистого зафиксированного Git checkout:
она проверяет HEAD и хеши исполняемых файлов до и после расчёта. Нужны Docker с
закреплённым выше образом OPM Flow, внешний HTTPS endpoint `/v1` и секретный файл
ключа; Docker socket в API-контейнер не передаётся.

Обязательное окружение и один запуск без перезаписи:

```bash
export LLM_BASE_URL=https://api.cerebras.ai/v1
export LLM_MODEL=qwen-3.8-27b
export LLM_TIMEOUT_SECONDS=120
export LLM_MAX_OUTPUT_TOKENS=4096
# При запущенном обратном SSH-туннеле на A100:
# export LLM_PROXY_URL=http://127.0.0.1:18889
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
- `source_model` имеет точное значение `model_z_opm`; `start_year` равен году
  первого месяца управления;
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
При отклонении плана до OPM сохраняется отдельный
`<run-id>.planning-rejected.json` с решениями ролей; подготовленный каталог
удаляется. Одобрение планирования разрешает расчёт, финальное решение остаётся
за критиком после получения терминальных доказательств.

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

### Выход A100 через существующий прокси

Проверено 9 сентября: на A100 работает `nci-egress-proxy.service` (Xray,
VLESS/REALITY). Для CLI на хосте и Compose с host network использовать:

```bash
export LLM_PROXY_URL=http://127.0.0.1:10809
```

Авторизованный запрос к Cerebras `qwen-3.8-27b` прошёл за 0,51 с.
Этот маршрут не зависит от WSL. `agent_rag` в своей Docker bridge-сети
обращается к тому же прокси через `172.23.0.1:10809`; этот адрес относится
к его сети, для TimesOil с host network используется loopback.
Проверять нужно авторизованный запрос модели: запрос `/models` без ключа
возвращает 403 даже при исправном маршруте.

### Резервный выход A100 через WSL

Проверенный маршрут: A100 `127.0.0.1:18889` → обратный SSH-туннель →
WSL `127.0.0.1:10809` → Cerebras. В WSL должен быть доступен HTTP CONNECT-прокси
на порту 10809. Из WSL держать отдельную сессию:

```bash
ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18889:127.0.0.1:10809 a100-remote
```

На хосте A100 добавить `LLM_PROXY_URL=http://127.0.0.1:18889` в окружение
CLI. `LLM_BASE_URL` остаётся `https://api.cerebras.ai/v1`; TLS проверяется
клиентом, ключ не передаётся прокси открытым текстом. Системные `HTTP_PROXY`
и `HTTPS_PROXY` клиент не читает. Остановка SSH-сессии закрывает маршрут.
Эта команда предназначена для CLI на хосте: `127.0.0.1` внутри API-контейнера
обозначает сам контейнер и не ведёт к туннелю хоста.

## 6. Docker на A100 через существующий прокси

В серверном `config/aios.env` указать
`LLM_PROXY_URL=http://127.0.0.1:10809` для `nci-egress-proxy` из раздела 5.
Остальные значения взять из текущего `config/aios.example.env`, включая каталог
модели v5. Ключ остаётся в отдельном secret-файле.

```bash
# Linux/A100: отдельное имя проекта; порт не конфликтует с другими API.
AIOS_PORT=18082 docker compose --env-file .env.example \
  -p scorp-timesoil-kt3 -f compose.yaml -f compose.a100.yaml \
  up -d --build --wait --wait-timeout 180
curl --noproxy '*' --fail http://127.0.0.1:18082/health
curl --noproxy '*' --fail http://127.0.0.1:18082/v1/capabilities
```

`compose.a100.yaml` подключает контейнер к сетевому пространству хоста для
доступа к loopback-прокси, сохраняя bind API на `127.0.0.1`. Этот профиль
предназначен для Linux. Доступ к Cerebras зависит от работающего SSH-туннеля
и прокси WSL; автоматического переключения маршрута нет.

Проверенная репетиция использовала API `127.0.0.1:18082`; модель v5 и ЧДД готовы,
живой агентный эксперимент завершён. `connectivity_verified=false` в
`/v1/capabilities` означает отсутствие сетевой пробы внутри этого GET;
доступность провайдера проверяется отдельным `/v1/experiments/agents`.


Снимки реально исполненных исходников репетиции находятся в
`results/kt3-model-z-v5/source-snapshots/`: обучение — 6 файлов, поиск/replay —
10, полный цикл — 16. `index.json` хранит исходные абсолютные пути и SHA-256;
содержимое каждого файла проверено перед копированием. Ссылки старых квитанций
остаются привязаны к исходным путям: при восстановлении архива нужны эти пути
либо новый воспроизводимый запуск с новыми квитанциями.


Итог репетиции 8 сентября: из трёх проверенных кандидатов выбран №290,
ЧДД 800,2466164462826 млн руб. за январь–июнь 2007 года, прирост
4,325833983137045 млн руб. к сопоставимому baseline. Сдаваемый файл репетиции:
`deliverables/track2_model_z/kt3/wells_schedule.inc`; подробная сводка —
`deliverables/track2_model_z/kt3_completion_summary.json`.
Значение не переносится на будущую историю или другой период управления.
