# Контрольная точка 2 TimesOil AIOS — 31 августа 2026

## Техническое резюме

КТ2 подтверждена по трём независимым слоям: агентный контур Qwen3.6 с
request-scoped инструментами, полный базовый расчёт Model Z в закреплённом OPM
Flow 2026.04 и проверяемая цепочка `Flow → SUMMARY replay → 14-польный ЧДД`.
Track 1 завершил source-bound одномесячный baseline/candidate proof: no-op
выиграл у alternative среди двух проверенных кандидатов. Track 2 завершил
четыре сценарных Flow-прогона, обучение v3 и proxy-search v3; выбран baseline.
Выбранный график прошёл терминальный OPM replay и официальный калькулятор ЧДД.
Цепочка self-replay аутентифицирована, но `organizer_certified=false`;
улучшение baseline не заявляется.

На текущем локальном дереве пройдены `173 passed, 2 skipped, 9 subtests passed`.
Оба пропуска относятся к необязательной внешней межъязыковой сверке ЧДД.
`doctor` намеренно возвращает `track1.certified=false`,
`track2.certified=false`, `track2.model_z_trained=false`: component readiness
не подменяет терминальный расчёт и сертификацию. Последний флаг описывает
локальный/API runtime без подключённого внешнего artifact; A100 training receipt
отдельно подтверждает существование обученной Model Z.

Подтверждённые результаты:

- terminal v3 receipt подтверждает один host-side live workflow внешнего
  Tatneft `qwen3.6-35b-a3b`: 8/8 provider responses, четыре роли и четыре
  actual allow-listed read-tool calls. Approval ролей — `[true,true,true,false]`,
  `critic_approved=false`; секреты, reasoning и raw response content не
  печатались и не сохранялись. Это не доказывает Compose-connectivity;
- Model Z v4 завершён в закреплённом образе OPM с `returncode=0`; исходный ZIP,
  raw SUMMARY, команда extraction и побайтовый replay связаны SHA-256;
- экспорт содержит 38 213 строк и 103 скважины; метод массы использует
  фактические PVT-плотности без масштабирования и клампинга;
- организаторский full-history профиль (`start_year=1991`,
  `chargeInitialPump=true`) дал ЧДД **5181,184136 млн руб.** против reference
  OPM **5157,016330 млн руб.**; результат лежит внутри опубликованного
  межсимуляторного диапазона `[5157,016330; 5218,944908]` млн руб.;
- Track 1 baseline Model Y дал **5334,357811 млн руб.** против reference
  **5326,453466 млн руб.**; независимый срез 2014 дал **1082,233753351** против
  **1082,233695354 млн руб.**; source-bound v2.1 дал **110,782383361 млн руб.**
  для no-op и **110,776225880 млн руб.** для alternative 140. No-op выбран и по
  objective, и оператором; доказан только один месяц, не полный горизонт;
- для оптимизационного горизонта с 01.01.2007 применяется отдельный профиль
  `operational_sunk_assets`: `chargeInitialPump=false`, потому что уже
  установленные на эту дату ЭЦН повторно не оплачиваются;
- Track 2 batch завершил baseline + три perturbation без artifact mismatch;
  dataset hash — `1253a351c8ea58dbd7e618cd50b9817f6d6b646962d2052744d4efb21a778cc0`;
- training v3 дал на test WAPE **7,7940 %** по нефти и **5,1943 %** по жидкости,
  pressure RMSE **8,808 bar**, OOD-rate **0,5464 %**; nominal 90%-coverage
  **40,50 %**, uncertainty не откалиброван конформно;
- proxy-search принял 32/32 кандидата и выбрал `baseline` со score
  **127494,1351**. Это proxy, не ЧДД;
- выбранный baseline прошёл OPM с `returncode=0`; 38 213 строк canonical CHDD
  совпали с аутентифицированным baseline, а официальный калькулятор с
  `operational_sunk_assets`, `start_year=2007`, `chargeInitialPump=false` дал
  **11918,789227262983 млн руб.** Receipt
  [`model_z_track2_final_replay_a100_v3.json`](./evidence/model_z_track2_final_replay_a100_v3.json)
  и audit
  [`model_z_track2_final_replay_audit_a100_v3.json`](./evidence/model_z_track2_final_replay_audit_a100_v3.json)
  имеют SHA-256 `359bd379d77adedb8d4bfb39f267335af40f64fcd67314e4cb8111d45fed483c`
  и `3c6d50e5cf355b08f0c67d5feaadee9e319050499d91162c6061e87a650dda32`;
- Docker v5 terminal receipt подтверждает fresh-context build, UID 10001,
  `CapEff=0`, no-new-privileges, read-only rootfs, HTTP/doctor PASS и cleanup.
  Qwen workflow и connectivity в этой проверке не запускались.

## Матрица требований КТ2

| Требование | Реализовано и доказано | Ограничение / следующий gate | Статус |
|---|---|---|---|
| Комплект, конфигурация и запуск | `uv.lock`, Dockerfile, Compose, FastAPI, CLI; Docker v5 прошёл runtime/security gates; отдельный host-side Qwen v3 workflow завершён | Docker v5 не вызывал Qwen workflow, поэтому Compose-connectivity не доказана | PASS раздельных gates; не единый Compose→Qwen chain |
| Controls → simulator → ЧДД, baseline | Model Z v4: закреплённый Flow, неизменяемый source, authenticated SUMMARY replay, source-correct mass export, официальный ЧДД | Baseline не является оптимизированным графиком | PASS |
| Track 1 | Source-bound Model Y baseline и два full-replay OPM-кандидата; no-op выбран с ЧДД `110,782383361` млн руб. | Доказан один месяц, `binary_restart=false`, не шестимесячный MPC | PASS для заявленного one-month proof |
| AIOS agents и Qwen3.6 | Terminal v3: 8/8 provider responses, четыре роли, 4 actual allow-listed tools, exact model | `critic_approved=false`; tool проверил 412 действий Jan–Apr, не полный план 618; recommendation-only | PASS live workflow с ограничениями; не certification |
| Track 2: датасет и сценарии | Verified bundle и четыре последовательных OPM-сценария; exact dataset/manifests, identity baseline и secondary WCON/BHP сохранены | Четыре траектории — демонстрационный минимум | PASS |
| Track 2: модель и метрики | Source-bound training v3, grouped split, artifact `82dfc80d…`, train/test metrics и OOD diagnostics | 90%-coverage только 40,50 %, не conformal; production-обобщение не заявляется | PASS training; не certification |
| Track 2: поиск и проверка | Search v3 выбрал baseline; OPM replay, SUMMARY extraction, canonical export и официальный ЧДД аутентифицированы | Proxy `127494,1351` не ЧДД; выбран baseline, улучшение не заявляется; `organizer_certified=false` | PASS self-replay; не organizer certification |
| Единицы ЧДД | `WLPR` — м³/сут; `WLPT` и `WLPT_Diff` — т; `WOMR` — т/сут; `WOMT` — т; Track 2 `liquid_tpd` — т/сут | Не требуется для baseline v4 | PASS |

## Доказанный baseline Model Z

### Закреплённый расчёт и provenance

Расчёт выполнен образом:

```text
openporousmedia/opmreleases:2026.04_amd64@sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9
```

Исходный ZIP имеет SHA-256
`4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa`.
Flow завершился за 1076,877 с с кодом 0. Raw-манифест имеет SHA-256
`dec2e59d09e8b29ed38064b4af37d60632f9171f6c77fee3fe2cdbe32b99a960`.
Извлечение SUMMARY закрепляет SMSPEC, UNSMRY, полный ordered vector selection,
точную команду `summary -r` и SHA-256 отчёта. Перед экспортом та же команда
обязательно повторяется в том же образе; stdout сравнивается побайтно.

Цепочка закрыта по умолчанию: несовпадение образа, команды, raw-хешей, report,
CSV или любого sidecar прерывает экспорт и запрещает `model_z_ready`.

### Масса и единицы без подгонки

Экспорт применяет source-correct hybrid:

- для 80 скважин с единственным PVT-регионом — авторитетные well surface
  oil/water volumes и однозначные surface densities;
- для 23 multi-PVT скважин — signed production completion increments
  `COPT`/`CWPT`, взвешенные по PVT-плотности соответствующей ячейки;
- для скважин без connection/density mapping отсутствие плотности разрешено
  только при строго нулевых фазовых объёмах;
- никакие значения не масштабируются и не клампятся к reference.

Pure-completion метод для всех скважин отклонён: он создавал 74 отрицательных
well-month mass diff. Нормализованные по well total варианты также отклонены,
поскольку выводили эффективную плотность за диапазон deck. Организаторский
workbook построен по другой численной траектории OPM, поэтому годовые строки
служат диагностикой, а не требованием побайтового равенства.

### Два экономических профиля имеют разный смысл

| Профиль | Горизонт | `chargeInitialPump` | Смысл | Доказанный результат |
|---|---:|---:|---|---:|
| `organizer_reference` | 1991 — полная история | `true` | Начальные ЭЦН входят в CAPEX, как в организаторском reference | 5181,184136 млн руб. |
| `operational_sunk_assets` | с 01.01.2007 | `false` | Существующие на начало оптимизации ЭЦН — уже понесённые затраты | 11918,789227262983 млн руб. в терминальном replay Track 2 |

Профили нельзя взаимозаменять. Значение 5181,184136 сравнивается с reference
5157,016330 только в full-history профиле. Для финальной оптимизации Track 2
скрипт replay явно передал `charge_initial_pump=False`; имя и семантика профиля
записаны в терминальном receipt.

## Сценарий 30-минутной демонстрации

Долгие Flow и обучение не запускаются в аудитории. Показываются сохранённые
артефакты и команды их полного воспроизведения.

Перед воспроизведением скопировать несекретный шаблон, заменить только
`/CHANGE_ME/` и загрузить переменные. `LLM_API_KEY` в этот файл не писать.

```bash
cp config/kt2.operator.example.env /tmp/kt2.operator.env
${EDITOR:-vi} /tmp/kt2.operator.env
set -a
source /tmp/kt2.operator.env
set +a
cd "$A100_PROJECT_ROOT"
```

### 0–4 мин — локальный gate и честная readiness

```bash
uv sync --locked
uv run pytest -q
uv run python -m timesoil.aios.cli doctor
uv run python -m compileall -q src scripts
uv run python -c "import timesoil"
git diff --check
```

Ожидаемый тестовый итог текущего дерева: `173 passed, 2 skipped, 9 subtests
passed`. Без runtime-secret и подключённых внешних terminal artifacts ожидаются
`qwen.configured=false`, `qwen.connectivity_verified=false`,
`track1.certified=false`, `track2.certified=false`,
`track2.model_z_trained=false`.

### 4–8 мин — контейнер и API

```bash
export QWEN_API_KEY_FILE=/dev/shm/timesoil-qwen-api-key
sudo /bin/sh -eu -c '
LLM_API_KEY=
  . /etc/nci-analytics/nci-analytics.env
  : "${LLM_API_KEY:?LLM_API_KEY is absent or empty}"
  umask 077
  printf "%s" "$LLM_API_KEY" > "$1"
  chown 10001:10001 "$1"
  chmod 0400 "$1"
  unset LLM_API_KEY
' sh "$QWEN_API_KEY_FILE"
docker compose config --quiet
docker compose build --pull
docker compose up -d --wait
curl --fail --silent http://127.0.0.1:8000/health
curl --fail --silent http://127.0.0.1:8000/v1/capabilities
docker compose run --rm --no-deps api python -m timesoil.aios.cli doctor
docker compose down -v
sudo rm -f -- "$QWEN_API_KEY_FILE"
```

В shell экспортируется только путь `QWEN_API_KEY_FILE`; значение ключа не
экспортируется и не печатается. `/etc/nci-analytics/nci-analytics.env` —
host-specific защищённый источник A100; на другом host его заменяют собственным
root-only secret source. Compose монтирует RAM-файл как read-only secret в
`/run/secrets/qwen_api_key`. Пустой или недоступный secret обязан остановить
приложение до старта.

На удалённой машине порт не публикуется наружу. Доступ с рабочего места:

```bash
ssh -L 8000:127.0.0.1:8000 a100-remote
```

FastAPI предоставляет `GET /health`, `GET /v1/capabilities`,
`POST /v1/experiments/agents` и `POST /v1/economics/chdd`. Uvicorn внутри
контейнера слушает `0.0.0.0:8000`, Compose публикует только loopback host.

### 8–13 мин — четыре grounded обращения Qwen

```bash
sha256sum \
  docs/hackathon/evidence/qwen36_agent_terminal_failure_a100_v2.json \
  docs/hackathon/evidence/qwen36_agent_tool_registry_a100_v3.json
jq '{schema,outcome,complete,terminal_exit_code,provider:{requested_model,provider_request_attempts,provider_responses,local_model_used},workflow:{role_order,actual_tool_count,approved,critic_approved,elapsed_seconds},control_scope,claim_limits}' \
  docs/hackathon/evidence/qwen36_agent_tool_registry_a100_v3.json
```

Ожидаемые SHA-256: failure v2
`e3a8fe69cc2764972e3679edb953e6f599c87875ef39718923f5a8bdb6547ecb`
и terminal v3
`6098179b0f21362a8cde72c58a5156616f5986729c1fffea978272b53ed8b1c5`.
V2 fail-closed завершился timeout до response headers: 0 ролей и 0 tool calls,
без retry. V3 выполнил один workflow без retry: 8/8 provider responses, exact
model, четыре роли, 4 actual read-tool calls, `elapsed_seconds=666,356`.
Роли вернули `[true,true,true,false]`; отказ критика обязателен к показу.

Agent-tool context содержит точный bounded slice из 412 действий Jan–Apr.
Полные 618 действий не прошли agent-tool validation, но отдельно прошли полный
OPM replay и ЧДД `11918,789227262983` млн руб. Qwen не запускает OPM/ЧДД,
остаётся recommendation-only; uncertainty не conformal, результат не
organizer-certified и не доказывает Compose-connectivity.

### 13–19 мин — baseline Flow, authenticated extraction и ЧДД

На A100:

```bash
RUN=/tmp/timesoil-kt2/evidence/model-z-baseline-20260831-a100-v4-production-vectors
RUN_MANIFEST="$RUN/manifest.json"
SUMMARY_REPORT="$RUN/summary-report.txt"
EXTRACTION_MANIFEST="$RUN/summary-extraction.json"
RESOLVED_DECK_DIR="$RUN/input/Model_Z"
EXPORT_REPLAY="$RUN/canonical-replay-demo"
CHDD_CSV="$EXPORT_REPLAY/model-z-chdd.csv"
TRACK2_CSV="$EXPORT_REPLAY/model-z-track2.csv"
EXPORT_MANIFEST="$EXPORT_REPLAY/model-z-export-manifest.json"
test ! -e "$EXPORT_REPLAY"
sha256sum "$RUN/manifest.json" "$RUN/summary-extraction.json" \
  "$RUN/summary-report.txt" "$RUN/canonical-final/model-z-export-manifest.json" \
  "$RUN/canonical-final/model-z-chdd.csv" \
  "$RUN/canonical-final/model-z-economic-evidence.json"
jq '{acceptance,canonical_chdd,export_manifest,organizer_reference_profile}' \
  "$RUN/canonical-final/model-z-economic-evidence.json"
```

Команда полного повторного Flow, не для live-demo:

```bash
uv run python -m timesoil.aios.cli opm-baseline "$MODEL_Z_SOURCE" \
  --deck "$MODEL_Z_DECK" --low \
  --runs-dir "$OPM_RUNS" --run-id model-z-baseline-20260831-replay \
  --timeout 3600
```

Низкая строгость включается только явно. Unsupported keywords остаются в
журнале; source не изменяется, расчёт идёт в отдельной run-copy.

Команда проверяемого экспорта:

```bash
uv run python scripts/export_opm_chdd.py "$SUMMARY_REPORT" \
  --scenario-id model-z-baseline --source-model model_z_opm \
  --opm-run-manifest "$RUN_MANIFEST" \
  --summary-extraction-manifest "$EXTRACTION_MANIFEST" \
  --deck-dir "$RESOLVED_DECK_DIR" --unit-system METRIC \
  --chdd-output "$CHDD_CSV" --trajectory-output "$TRACK2_CSV" \
  --manifest "$EXPORT_MANIFEST"
```

### 19–24 мин — Track 2 сценарии, training v3 и границы доказательства

```bash
BUNDLE=/tmp/timesoil-kt2/track2-v2/scenario-bundle
sha256sum "$BUNDLE/index.json" \
  /tmp/timesoil-kt2/track2-v2/scenario-bundle-verification.json

uv run python scripts/generate_track2_scenarios.py \
  "$TRACK2_CSV" "$SCHEDULE_INCLUDE" "$OUTPUT_BUNDLE" \
  --scenario-count 4 --seed 20260831 --perturbation-fraction 0.15
```

Preflight PASS означает: 13 regular read-only files, без symlink/path escape;
identity-baseline побайтно равен исходному schedule include; три perturbation
меняют только основную уставку и сохраняют effective вторичные WCON-поля,
включая BHP; по 38 213 действий, 371 месяцу и 103 скважинам месячная суммарная
закачка сохранена с нулевым residual, role, target и status не изменены.
Generator сам детерминированно проецирует canonical 10-column trajectory в
пять полей управления. Regenerated bundle побайтно совпал; canonical-repro
receipt имеет SHA-256 `e0354df254d22b74dad3157c7453819a1f6f59af0529776119be4039bad9ef2e`.
Сам preflight ещё не является OPM run. Отдельный terminal batch receipt затем
подтвердил четыре последовательных OPM-прогона, четыре exact dataset/manifests и
dataset hash `1253a351c8ea58dbd7e618cd50b9817f6d6b646962d2052744d4efb21a778cc0`.
SHA-256 receipt —
`d50386ca122c5e3608b14157661ff0974c9329f89072e7b46cbfc3b3d0f797b6`,
batch manifest —
`fd4a8d11ca85886f56e143ef0a48686da2677b80d770f95b5fd7e938ac33e946`.

Команды терминального контура после preflight:

```bash
uv run python scripts/run_track2_scenarios.py \
  "$MODEL_Z_SOURCE" "$SCENARIO_BUNDLE" "$SCENARIO_RUNS" \
  --source-sha256 4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa \
  --scenario-index-sha256 "$TRACK2_SCENARIO_INDEX_SHA256" \
  --baseline-chdd-sha256 "$MODEL_Z_BASELINE_CHDD_SHA256" \
  --schedule-relative-path "$SCHEDULE_RELATIVE_PATH" --deck "$MODEL_Z_DECK" \
  --timeout-seconds 3600 --parsing-strictness low

uv run python scripts/train_track2_surrogate.py \
  --dataset "$MODEL_Z_DATASET_DIR" --manifest "$MODEL_Z_MANIFEST_DIR" \
  --output "$TRACK2_TRAIN_OUTPUT" --test-fraction 0.25 \
  --ensemble-size 5 --n-estimators 160 --horizon 6 --seed 20260831
```

Runner не доверяет хешам внутри bundle как единственному основанию: SHA-256
`index.json` должен совпасть с отдельно закреплённым
`71edcb70cf4e04871f81e6d6ed4842f8cc91d542731024269060a1c8f5cfaf54`.
Первым запускается identity-baseline; его 14-польный CHDD обязан совпасть с
эталонным `446c24eaa063710422835a745be157abdce66d602c75f33de50a8e75881d3884`,
иначе perturbations и итоговый batch manifest не допускаются.

Training v3 receipt имеет SHA-256
`511e1950d43244ab9cd7ca034a26cd811e7532df155fe259555728a53c10036a`.
Model artifact — `82dfc80d535345fddcf3ec3540c8ea66df89bce7ff50f1f262256fdf07cce4d3`,
manifest — `6f9532414dc8ea7291fca11159ab751c5370f9e59f234041ab017eda76698772`,
metrics — `95e3773ac6dc5036304c6f5fd697b9e603c9220ab7747d36469426a57ab8c4c4`.
На train: oil/liquid WAPE `7,1949 %`/`4,5042 %`, pressure RMSE `8,728 bar`,
OOD-rate `0,7286 %`, coverage `43,71 %`, 183 rollout. На test:
`7,7940 %`/`5,1943 %`, `8,808 bar`, `0,5464 %`, coverage `40,50 %`, 61
rollout. Ансамблевое стандартное отклонение не называется калиброванным
интервалом.

### 24–28 мин — Track 2 поиск и доказанный Track 1

```bash
uv run python scripts/search_track2_schedule.py search \
  "$TRACK2_MODEL" "$MODEL_Z_DATASET" "$MODEL_Z_EXPORT_MANIFEST" \
  "$TRACK2_METRICS" "$MODEL_Z_SOURCE" "$SCHEDULE_INCLUDE" "$SEARCH_DIR" \
  --scenario-id "$TRACK2_SEARCH_SCENARIO_ID" --start-date 2007-01-01 \
  --candidate-count 32 --seed 20260831 --perturbation-fraction 0.05 \
  --uncertainty-weight 1.0 --injection-cost-equivalent 0.01 \
  --deck "$MODEL_Z_DECK" \
  --schedule-relative-path "$SCHEDULE_RELATIVE_PATH" \
  --timeout-seconds 3600 --parsing-strictness low

uv run python scripts/search_track2_schedule.py replay \
  "$MODEL_Z_SOURCE" "$SEARCH_DIR" "$REPLAY_DIR" \
  --deck "$MODEL_Z_DECK" --schedule-relative-path "$SCHEDULE_RELATIVE_PATH" \
  --timeout-seconds 3600 --parsing-strictness low

test ! -e "$TRACK1_PROOF_OUTPUT_ROOT"
uv run python scripts/run_model_y_track1_proof.py --resume-baseline \
  --source "$MODEL_Y_SOURCE" \
  --baseline-dir "$TRACK1_BASELINE_DIR" \
  --reference-workbook "$TRACK1_REFERENCE_WORKBOOK" \
  --deck "$TRACK1_DECK_RELATIVE_PATH" \
  --schedule-relative-path "$TRACK1_SCHEDULE_RELATIVE_PATH" \
  --output-root "$TRACK1_PROOF_OUTPUT_ROOT"
```

Track 1 proof сам выводит конфигурацию MPC из authenticated baseline state и
затем вызывает `run_track1_mpc.py`; статический JSON с неподтверждённым
`restart_ref` намеренно не поставляется. Source-bound v2.1 выбрал no-op:
`110,782383361` против `110,776225880` млн руб. у alternative; это по-прежнему
только один месяц без binary restart.

Search v3 завершён отдельно: 32/32 кандидата приняты, выбран `baseline`, proxy
score `127494,1351`, controls SHA
`74580379bf3b1551eac0b85fd9684dd6873a69149924491871dad92b7b31e659`,
`wells_schedule.inc` SHA
`2cf99d0e70901d3881c8ce14b9901b82fa21e0ee2945ff6f1ee82a35429af372`,
full overlay SHA
`4fa3d5efb189bb365b426c6c6acca98cd41894ad88a53749b4d7742a83d0af35`,
manifest SHA
`b4a6721adf38da833d41e21dd64a2496ee833e4b84e10dc2d583b5511052e5e8`
и lineage SHA
`1cfddbd255d55a048ffe2da93af9b1db878111cd70a50b08efa0e959bc49de24`.
Receipt `ccc9705475209baaa306a2e8a4bbff034cfa6996564f725fd4f58fcd9333c006`
имеет `selection_only=true`: 0 кандидатов отклонены по OOD. Выбранный baseline
затем прошёл replay-команду выше с `returncode=0`; OPM manifest
`e4eba3f347d5e46d9319a8c52592c39635c3f2192974166e866fdc42b5bcb617`,
SUMMARY extraction
`2587105e3b4d14f26855d6a575408f6db921321436611f0ee0b915f727611b15`
и официальный ЧДД `11918,789227262983` млн руб. аутентифицированы. Локальный
комплект: [`deliverables/track2_model_z_v3/`](./deliverables/track2_model_z_v3/README.md).

Первая replay-попытка сохранена как fail-closed receipt
[`model_z_track2_final_replay_failure_a100_v3.json`](./evidence/model_z_track2_final_replay_failure_a100_v3.json),
SHA-256 `a4593ea36d39529a44784313a875ad286e78b268f0a83fbe5716237cb7c02f09`.
OPM и canonical export успели завершиться, но официальный ЧДД не запускался:
88 598 498-байтовый SUMMARY превысил общий evidence-cap 64 MiB. Единственный
разрешённый scoped cap-fix снял этот размерный барьер только для уже
аутентифицированных связанных replay-артефактов; источник до/после запуска и
все хеши повторно проверены audit receipt.

### 28–30 мин — итог без расширения claims

Показать матрицу выше и SHA ledger ниже. Итог демо: Qwen v3 host-side live,
Model Z baseline/scenarios/training/search и Track 1
source-bound one-month proof доказаны в заявленных границах. Track 2 self-replay
и Docker v5 terminal runtime доказаны; выбран baseline, поэтому улучшение не
заявляется, а `organizer_certified=false`. Qwen critic отклонил bounded context;
Compose-connectivity отдельно не доказана.

## Архитектура и fail-closed границы

```text
operator
   ├── CLI / FastAPI ── grounded Qwen workflow ── advisory decision
   │                         └── typed request-only tools
   ├── ScheduleCompiler ── OPM Flow (pinned digest) ── raw SMSPEC/UNSMRY
   │                                                    │
   │                         deterministic summary -r replay
   │                                                    │
   │                         canonical trajectory + 14-field CHDD
   │                                                    │
   │                                  official CHDD subprocess
   ├── Track 1 MPC ── monthly real-OPM candidates ──────┘
   └── Track 2 ── scenario OPM runs ── surrogate ── search ── one real replay
```

Ключевые границы:

- Qwen — советующий слой. Его `approved` не является сертификатом и не может
  заменить Flow, provenance или ЧДД;
- grounded tools читают только текущий request, валидируют controls через
  типизированные контракты и не запускают ГДМ/ЧДД;
- subprocess запускаются без shell, с timeout; container cleanup ограничен
  cidfile конкретного запуска;
- ZIP/deck проверяются от path traversal и zip-slip; source immutable;
- `model_z_identity=true` у canonical trajectory означает доказанное
  происхождение датасета, но не означает `model_z_trained=true`;
- Track 2 имеет полный terminal lineage и официальный ЧДД, но это
  self-authentication; сертификация организаторов отдельно не заявляется.

## Ограничения на момент КТ2

1. Docker terminal v5 доказал fresh-context build, non-root UID 10001,
   `CapEff=0`, read-only rootfs, no-new-privileges, loopback HTTP, Compose-run
   doctor и cleanup. Пустой и недоступный secret fail-closed. Receipt не
   запускал Qwen workflow и не проверял connectivity.
2. Четыре сценария, training v3 и search v3 закрывают демонстрационный pipeline,
   но не production-обобщение. Coverage `40,50 %` далеко от nominal 90 % и не
   откалиброван конформно; выбран baseline только по proxy.
3. Source-bound Track 1 proof ограничен одним месяцем. Выбор no-op относится
   только к двум проверенным кандидатам и не доказывает глобальный optimum.
4. Организаторский workbook и v4 получены на близких, но не идентичных
   численных траекториях. Межсимуляторный диапазон — проверка разумности
   экономики, не утверждение точной parity.
5. Два пропущенных теста требуют внешнего Node/corrected `core.js`; основной
   Python-контур и официальный Python ЧДД ими не блокируются.
6. В API нет Docker socket, локальной LLM, OPM, training/search scripts и model
   weights. Образ — только API/control plane + Python ЧДД; численный контур и
   большие artifacts остаются внешними host-side ресурсами.
7. История Track 1 сохраняет post-hoc v1 и fail-closed v2. Успешный v2.1 имеет
   `cryptographic_execution_binding=true`, но `binary_restart=false`, один месяц
   и `organizer_certified=false`.
8. Qwen v3 — отдельный host-side live receipt, не Compose-path. Critic отказал:
   agent-tool проверил 412 действий Jan–Apr, не полный план 618. Полный plan
   доказан OPM/ЧДД независимо от Qwen.
9. Первый финальный replay остановился после успешного OPM/export, но до ЧДД на
   общем 64 MiB evidence-cap. Failure receipt сохранён; scoped cap-fix применён
   только к аутентифицированным связанным артефактам, успешный повтор и source
   bytes подтверждены audit v3.
10. Терминальный Track 2 выбрал baseline. ЧДД `11918,789227262983` млн руб.
    доказывает воспроизводимость выбранного управления, но не улучшение и не
    сертификацию организаторов.

## Реестр доказательств

| Доказательство | Расположение | SHA-256 / ключевой факт | Вердикт |
|---|---|---|---|
| Grounded Qwen v1 | `docs/hackathon/evidence/qwen36_agent_tool_registry_a100.json` | `42b05683992588fafe3a9eaac2545b9df6e86df5bcfe0975da339c3ac16bc031` | История; superseded terminal v3 |
| Qwen terminal timeout v2 | `docs/hackathon/evidence/qwen36_agent_terminal_failure_a100_v2.json` | `e3a8fe69cc2764972e3679edb953e6f599c87875ef39718923f5a8bdb6547ecb`; timeout до headers, 0 responses/roles/tools | FAIL сохранён; no retry |
| Qwen terminal live v3 | `docs/hackathon/evidence/qwen36_agent_tool_registry_a100_v3.json` | `6098179b0f21362a8cde72c58a5156616f5986729c1fffea978272b53ed8b1c5`; 8/8 responses, 4 роли, 4 actual tools, 666,356 с | PASS workflow; `critic_approved=false`, 412/618 bounded scope, не Compose/certification |
| Docker v1 receipt | `docs/hackathon/evidence/docker_a100_v1.json` | `f82e2ee311bfa6ce587e57923bad37da1ef420bd83b1c3b7e4fbd553d324dc53` | История; superseded v5 |
| Docker terminal v5 | `docs/hackathon/evidence/docker_a100_v5.json` | `c7a48d202cd337f3d0915aac47688834d8a5ff6647b1aa6911535f633f00cf12`; image `sha256:41116e94f97801d7f5b234a1e597134149e22ddccf60d9b58445dc6459b56802` | PASS runtime/security; Qwen workflow не вызван |
| Model Z baseline receipt | `docs/hackathon/evidence/model_z_baseline_a100_v1.json` | `78993a8e0f5b3c1cd53c4bd41c50478280d94bbafef6730df193f76393fb0927`; 22/22 Flow + 18/18 economics artifacts | PASS |
| Model Z v4 raw manifest | A100: `.../model-z-baseline-20260831-a100-v4-production-vectors/manifest.json` | `dec2e59d09e8b29ed38064b4af37d60632f9171f6c77fee3fe2cdbe32b99a960` | PASS |
| Authenticated SUMMARY extraction | A100: тот же run, `summary-extraction.json` | `f751816169b9141aaaab4ca5f43062c71f226e7884df6b96ec43a71667986adb` | PASS |
| Canonical SUMMARY report | A100: тот же run, `summary-report.txt` | `844d7afe9fdeeda5a907badb32e70c85486e9d7b45fbd64cbf00fd405a234e67`, 88 598 498 байт | PASS |
| Model Z export manifest | A100: `canonical-final/model-z-export-manifest.json` | `c2a3e5e82e0fcd250384bec0d3bbb12b207babccc6154a1cf899045e4255c1a5` | PASS |
| 14-field Model Z CHDD CSV | A100: `canonical-final/model-z-chdd.csv` | `446c24eaa063710422835a745be157abdce66d602c75f33de50a8e75881d3884`, 38 213 строк | PASS |
| Model Z Track 2 trajectory | A100: `canonical-final/model-z-track2.csv` | `b49d2ff4b5cd7b6ef08f30f2b2f79365209b40d77349986a584fba34b6f2190d` | Provenance PASS; использована в scenario bundle |
| Terminal economics receipt v2 | `docs/hackathon/evidence/model_z_economics_terminal_a100_v2.json` | `6a814805cce34aaf91f815b567248dde88966e522cd5a11526b4d77611858d8c`; organizer-1991 = `5181.184136469775`, operational-2007 = `11918.789227262983` | PASS baseline economics; supersedes pre-terminal `3e28…` |
| Track 1 historical proof v1 | `docs/hackathon/evidence/model_y_track1_a100_v1.json` | `1b59a7c60289fac36c05851c23af69955749b0987bf496cab63a65fe2d9b02a9` | Superseded; один месяц |
| Track 1 post-hoc source ledger | `docs/hackathon/evidence/model_y_track1_source_provenance_posthoc_v1.json` | `fe3aae8e462bd331647ebc03d3316027ca77133a950b5997ca45a2a2dca1f84e` | История; execution binding не заявляется |
| Track 1 fail-closed source-bound v2 | `docs/hackathon/evidence/model_y_track1_source_bound_failure_a100_v2.json` | `f2d838ea5e9e4dc3fd5febe32e6918c79030c8d62abcfc67d9ca483013c7da01`; остановка до candidate Flow | FAIL сохранён; output отсутствует |
| Track 1 source-bound v2.1 | `docs/hackathon/evidence/model_y_track1_source_bound_a100_v2_1.json` | `b3aebefce30a0a948a84702d3b6fd7598aa183c735f296d9cb7ae5a0003cf050`; no-op `110.782383361`, alternative `110.776225880` | PASS, one month; не organizer-certified |
| Локальный комплект Track 1 v2.1 | `docs/hackathon/deliverables/track1_source_bound_v2_1/` | `wells_schedule.inc` `5a21c8a293e0b48502d67b7240125733fba9091cd2c4a82024975402f8881cb4` | Source-bound КТ2 proof; не финальная заявка |
| Track 2 bundle index | A100: `/tmp/timesoil-kt2/track2-v2/scenario-bundle/index.json` | `71edcb70cf4e04871f81e6d6ed4842f8cc91d542731024269060a1c8f5cfaf54` | PREFLIGHT PASS |
| Track 2 bundle verification | A100: `/tmp/timesoil-kt2/track2-v2/scenario-bundle-verification.json` | `f8284da948312446318e14acaf768bd2f06bacb61dec7f1fc667b7400b4273d9` | PREFLIGHT PASS |
| Canonical→controls reproducibility | `docs/hackathon/evidence/track2_scenario_bundle_repro_a100_v1.json` | `e0354df254d22b74dad3157c7453819a1f6f59af0529776119be4039bad9ef2e`; 13/13 файлов побайтно совпали, index `71ed…` | PASS |
| Live source/process observation Track 2 | `docs/hackathon/evidence/track2_scenario_runner_live_source_observation_v1.json` | `59c37301bd5cc0cf323f4e5d7664aa24b0bb18fbfd74ddb06a9a2360fa6025ba`; snapshot map `d7ad…` | Наблюдение; pre-execution binding не заявляется |
| Terminal scenario batch Track 2 | `docs/hackathon/evidence/track2_scenario_batch_completion_a100_v1.json` | `d50386ca122c5e3608b14157661ff0974c9329f89072e7b46cbfc3b3d0f797b6`; batch `fd4a…`, 4 сценария × 38 213 строк, mismatches 0 | PASS; dataset `1253a351…` |
| Track 2 training v1 | `docs/hackathon/evidence/model_z_track2_training_a100_v1.json` | `dc8dcab4de429ec9b7c96c1a368cf69dc204c332cc1741deb8c06de4438593b6` | История; OOD contract superseded v3 |
| Track 2 search failure v1 | `docs/hackathon/evidence/model_z_track2_search_failure_a100_v1.json` | `0387cc735f15c97448b541d3bb8bd8b7394ca03daa79fed8b853ad614df0a033`; baseline outside domain | FAIL сохранён; no OPM/ЧДД |
| Track 2 training terminal v3 | `docs/hackathon/evidence/model_z_track2_training_terminal_a100_v3.json` | `511e1950d43244ab9cd7ca034a26cd811e7532df155fe259555728a53c10036a`; model `82dfc80d…`, metrics `95e3773a…` | PASS training; uncertainty не conformal |
| Track 2 search terminal v3 | `docs/hackathon/evidence/model_z_track2_search_terminal_a100_v3.json` | `ccc9705475209baaa306a2e8a4bbff034cfa6996564f725fd4f58fcd9333c006`; selected baseline, schedule `2cf99d0e…` | PASS selection; replay закрыт последующими receipts |
| Track 2 replay failure v3 | `docs/hackathon/evidence/model_z_track2_final_replay_failure_a100_v3.json` | `a4593ea36d39529a44784313a875ad286e78b268f0a83fbe5716237cb7c02f09`; SUMMARY 88 598 498 байт > 64 MiB cap | FAIL сохранён; OPM/export PASS, ЧДД не запускался |
| Track 2 final replay v3 | `docs/hackathon/evidence/model_z_track2_final_replay_a100_v3.json` | `359bd379d77adedb8d4bfb39f267335af40f64fcd67314e4cb8111d45fed483c`; OPM `e4eba3f3…`, ЧДД `11918.789227262983` | PASS self-replay; baseline, `organizer_certified=false` |
| Track 2 replay audit v3 | `docs/hackathon/evidence/model_z_track2_final_replay_audit_a100_v3.json` | `3c6d50e5cf355b08f0c67d5feaadee9e319050499d91162c6061e87a650dda32`; 20/20 linked, 9/9 execution sources, 22/22 OPM artifacts | PASS scoped cap-fix audit |
| Локальный комплект Track 2 v3 | `docs/hackathon/deliverables/track2_model_z_v3/` | submission `695fcb09f4448c95c502d72b764468188024e74fef98b4f1142ee0fd55add849`; README `a19e19062b4f15017e6fb2f8e626eb72045975416685e54dd94c20054c074ff5` | PASS self-replay; не organizer-certified |
| Текущий source OPM adapter | `src/timesoil/aios/opm.py` | `7113e7575b1e06a4a59a85c6463fc5d3c95bb1a4609059a06364e08d4fddbd57` | Terminal source |
| Текущий source exporter | `src/timesoil/aios/opm_chdd.py` | `9b6c3c47b5c345f908f322043b91b08b2f0e6ef39ee95861e5ab3c00b4ae5572` | Terminal source |
| Текущий source economics adapter | `src/timesoil/aios/economics.py` | `99ef89bd2f089256ee877bb70d3ec799bb1384b1b4a3b8f5801285d06c7a1718` | Terminal source |

## Терминальные значения и evidence

Заменять маркер разрешено только числом, SHA-256 и путём к завершённому receipt.

| Поле / маркер | Подтверждённое значение или требуемое доказательство |
|---|---|
| Scenario runs | Receipt `d50386ca…`; четыре успешных OPM manifests |
| Dataset | `1253a351c8ea58dbd7e618cd50b9817f6d6b646962d2052744d4efb21a778cc0` |
| Model | Artifact `82dfc80d535345fddcf3ec3540c8ea66df89bce7ff50f1f262256fdf07cce4d3`; manifest `6f9532414dc8ea7291fca11159ab751c5370f9e59f234041ab017eda76698772` |
| Train metrics | oil/liquid WAPE `7,1949 %`/`4,5042 %`, pressure RMSE `8,728 bar`, OOD `0,7286 %`, 183 rollout |
| Test metrics | oil/liquid WAPE `7,7940 %`/`5,1943 %`, pressure RMSE `8,808 bar`, OOD `0,5464 %`, coverage `40,50 %`, 61 rollout |
| Search | Receipt `ccc97054…`; 32/32 accepted, baseline selected; proxy не ЧДД |
| Final Track 2 replay | Receipt `359bd379d77adedb8d4bfb39f267335af40f64fcd67314e4cb8111d45fed483c`, audit `3c6d50e5cf355b08f0c67d5feaadee9e319050499d91162c6061e87a650dda32`; baseline, OPM returncode 0, ЧДД `11918.789227262983` млн руб. |
| Track 2 deliverable | [`track2_model_z_v3`](./deliverables/track2_model_z_v3/README.md); self-replay authenticated, `organizer_certified=false` |
| Qwen terminal live | Receipt `6098179b0f21362a8cde72c58a5156616f5986729c1fffea978272b53ed8b1c5`; 8/8 responses, 4 роли/4 tools, approvals `[true,true,true,false]`, 412/618 bounded scope |
| Docker terminal | Receipt `c7a48d202cd337f3d0915aac47688834d8a5ff6647b1aa6911535f633f00cf12`; runtime/security PASS, Qwen workflow не вызван |

Track 2 self-replay закрыт, но API `track2.certified=false` остаётся корректным:
организаторская сертификация отсутствует. Обученный artifact существует на
A100, но локальный/API `track2.model_z_trained` остаётся `false`, пока artifact
не подключён к runtime. Qwen live workflow закрыт host-side receipt, но
Compose-connectivity и organizer certification не заявляются.
