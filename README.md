# TimesOil

Прогноз дебита нефти и жидкости добывающих скважин на 6 месяцев вперёд:
месторождение с экранирующими разломами (6 блоков), поздняя стадия заводнения,
месячные данные гидродинамического симулятора (49 скважин, 2007-05…2015-11).

Текущий контур объединяет **Chronos-2, TiRex-2, TiDE, LightGBM** и физические
CRM/двухфазную CRM/CRMP-модели; SPDM/ManiMamba сохранён как исследовательская
линия. Последний зафиксированный ансамбль: **нефть WAPE 3,74 %, жидкость
3,21 %** на трёх канонических срезах по 6 месяцев.
Подробности: [`docs/отчёт_прогноз_дебита_6мес.md`](./docs/отчёт_прогноз_дебита_6мес.md).

Правила работы с инструментами (fff / codegraph / serena), памятью Honcho и
workflow сервера a100 — в [`CLAUDE.md`](./CLAUDE.md); «Project Purpose» там же.
`AGENTS.md` = `CLAUDE.md` побайтно (для Codex).

## Быстрый старт

```bash
uv venv --python 3.13 .venv    # окружение (Python 3.13, только uv)
uv sync --locked               # зависимости строго из uv.lock
```

- **Правки** — локально в WSL; **прогоны** — на сервере a100
  (`/root/projects/TimesOil/`, `ssh a100` / `ssh a100-remote`), синхронизация через git.
- Рабочий журнал — [`docs/roadmap.md`](./docs/roadmap.md) (Plan → Act → Verify → Report).

## Контрольная точка 2: AIOS

Текущий KT2-контур и честные границы готовности описаны в
[`docs/hackathon/checkpoint_2_2026-08-31.md`](./docs/hackathon/checkpoint_2_2026-08-31.md).

Фактический срез КТ2: source-bound proof Track 1 v2.1 выбрал no-op с ЧДД
`110,782383361` млн руб. на одном месяце; Track 2 завершил четыре OPM-сценария,
обучение Model Z и proxy-search 32 кандидатов. Proxy выбрал baseline, но это
не означало улучшения. Выбранный baseline затем прошёл аутентифицированный
шестимесячный OPM replay с `returncode=0`; официальный калькулятор с профилем
`operational_sunk_assets` дал ЧДД `11918,789227263` млн руб. Это внутреннее
доказательство воспроизводимости, не сертификация организаторов:
`organizer_certified=false`. Реестр хешей и отрицательных запусков:
[`docs/hackathon/evidence/README.md`](./docs/hackathon/evidence/README.md).

Структура:

- `src/timesoil/aios/` — контракты, четыре агента, клиент только для Qwen, Track 1,
  OPM, Track 2, официальный ЧДД, FastAPI и CLI;
- `scripts/generate_track2_scenarios.py`, `run_track2_scenarios.py` —
  детерминированный Model Z bundle, identity-baseline, lossless WCON-overlay и
  последовательные OPM-прогоны с внешними index/CHDD hash-gates;
- `scripts/train_track2_surrogate.py` — обучение суррогата; по умолчанию Model Y
  используется только как проверка контура;
- `scripts/search_track2_schedule.py` — surrogate search и единственный
  финальный OPM/ЧДД replay выбранного расписания;
- `scripts/export_opm_chdd.py` — экспорт OPM SUMMARY в контракты ЧДД и Track 2;
- `docs/hackathon/deliverables/track1_source_bound_v2_1/` — локальный
  одномесячный source-bound комплект Track 1 с выбранным расписанием и честной
  границей применимости;
- `docs/hackathon/deliverables/track2_model_z_v3/` — воспроизведённый комплект
  Track 2: baseline schedule, полный overlay, lineage, manifest и submission;
- `Dockerfile`, `compose.yaml`, `config/aios.env`, `.env.example` — поставка API
  без локальной LLM и без Docker socket; `config/aios.env` содержит только
  воспроизводимые несекретные параметры.

Параметры обоих треков — официальные SHA-256 исходников, аутентифицированные
Track 2 index/baseline CHDD hashes, закреплённый образ OPM, пути deck/schedule,
горизонты, профили ЧДД и лимиты — собраны в
`config/kt2.operator.example.env`. Перед запуском скопируйте шаблон во внешний
рабочий каталог и замените только `/CHANGE_ME/`; ключ Qwen в файл не
записывать, он передаётся Compose через runtime secret `LLM_API_KEY`. Контракт
поставки проверяется командой:

```bash
cp config/kt2.operator.example.env /tmp/kt2.operator.env
${EDITOR:-vi} /tmp/kt2.operator.env
set -a
source /tmp/kt2.operator.env
set +a
uv run pytest -q tests/test_delivery_contract.py
```

Локальный запуск:

```bash
uv sync --locked
uv run python -m timesoil.aios.cli doctor
uv run python -m timesoil.aios.cli serve
uv run pytest -q
```

Текущий локальный итог: **173 passed, 2 skipped, 9 subtests passed**.

Контейнерный запуск:

Проверенная среда сборки: Linux `amd64`, Docker Engine `29.2.0`, Docker
Compose `5.0.2`. Compose использует file secret.

```bash
cp .env.example .env          # только порт и путь к RAM-secret; значения ключа нет
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
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/capabilities
```

В shell экспортируется только путь `QWEN_API_KEY_FILE`, не значение ключа.
`/etc/nci-analytics/nci-analytics.env` — host-specific защищённый источник A100;
на другом host его нужно заменить собственным root-only secret source.
Compose монтирует RAM-файл как read-only `/run/secrets/qwen_api_key`; entrypoint
fail-closed при пустом или недоступном secret. Ключ не попадает в image config,
history или командную строку. Контейнер работает от UID `10001`, с read-only
rootfs, `CapEff=0` и `no-new-privileges`. Запись разрешена только в volume
`/app/runs` и tmpfs `/tmp`.

Сборка детерминирована `uv.lock` и digest-pinned базовыми образами. Для передачи
готового образа организаторам:

```bash
docker image inspect timesoil-kt2-api:local --format '{{.Id}}'
docker save timesoil-kt2-api:local | gzip -n > timesoil-kt2-api.tar.gz
sha256sum timesoil-kt2-api.tar.gz
```

Запуск сохранённого образа через тот же безопасный Compose-профиль:

```bash
docker load < timesoil-kt2-api.tar.gz
test -r "$QWEN_API_KEY_FILE"
docker compose up -d --no-build --wait
curl --fail http://127.0.0.1:8000/health
```

После показа удалить изолированные runtime-ресурсы и RAM-secret:

```bash
docker compose down -v
sudo rm -f -- "$QWEN_API_KEY_FILE"
```

Прямой `docker run -e LLM_API_KEY=...` намеренно не документируется: значение
попало бы в inspect-конфигурацию контейнера.

Terminal receipt
[`docker_a100_v5.json`](./docs/hackathon/evidence/docker_a100_v5.json), SHA-256
`c7a48d202cd337f3d0915aac47688834d8a5ff6647b1aa6911535f633f00cf12`,
подтверждает fresh-context build, HTTP/Compose-run doctor, negative secret
gates, runtime security и cleanup. Qwen workflow/connectivity в нём не
проверялись. Все 11 зафиксированных source hashes совпадают с текущим деревом.

Образ намеренно не содержит локальную LLM, `torch`, model weights, vLLM или
llama.cpp: агентный слой использует только `qwen3.6-35b-a3b` через утверждённый
Tatneft API. Без runtime-секрета агентный endpoint fail-closed; `/health`
не поднимается, потому что entrypoint останавливает весь API до старта.

Терминальная Qwen v3-квитанция
[`qwen36_agent_tool_registry_a100_v3.json`](./docs/hackathon/evidence/qwen36_agent_tool_registry_a100_v3.json),
SHA-256 `6098179b0f21362a8cde72c58a5156616f5986729c1fffea978272b53ed8b1c5`,
подтверждает один host-side live workflow внешнего Tatneft
`qwen3.6-35b-a3b`: 8/8 provider responses, четыре роли и 4 actual allow-listed
read-tool calls. Critic вернул отказ (`critic_approved=false`), потому что
agent-tool проверил только 412 действий января–апреля, не все 618. Полный план
отдельно подтверждён OPM/ЧДД replay; Qwen остаётся recommendation-only и receipt
не доказывает Compose-connectivity или сертификацию организаторов.

OPM Flow запускается отдельно операторским CLI на хосте; API-контейнер его не
содержит. В образ также не входят Track 2 training/search scripts, сохранённый
суррогат и большие evidence-артефакты: это API/control-plane поставка, а не
автономный численный runner. Команды 30-минутной демонстрации, подтверждённые
метрики и поля для финальных A100/ЧДД run-id находятся в документе KT2.

## Запуск агента

```bash
cd /home/ruslan_safaev/TimesOil
codex
```

Codex наследует глобальные MCP, permissions и Honcho из WSL-профиля; локальный
`.mcp.json` не нужен. В начале работы агент проверяет `codegraph status`,
активирует Serena и читает `mem:core`. Полный регламент находится в
[`AGENTS.md`](./AGENTS.md), идентичном `CLAUDE.md`.
