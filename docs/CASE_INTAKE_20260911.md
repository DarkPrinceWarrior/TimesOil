# Приём тестового кейса трека 2 — регламент дежурного на 17:00, 11 сентября 2026

Одна страница. Кейс приходит в **17:00 МСК**, окно расчёта **17:00–21:00** (4 часа).
Код заморожен в 17:00: после заморозки правок нет, запускается только то, что закоммичено.

Весь прогон — **одна команда**: `scripts/run_case_z.sh`. Она сама делает приём архива,
базовый цикл, связность, блоки, банк, дообучение, поиск, единственный финальный OPM и
отчёт интерпретируемости; на каждый этап — свой каталог, файл `.log`, файл `.exit`,
время и запись в `protocol.json` с хешами ключевых выходов.

**Правило дежурного.** Ни одна проверка не «обходится». Отказ проверки — это результат,
который надо записать и доложить, а не препятствие, которое надо снять. Прогнозный ЧДД
никогда не является официальным: официальный — только калькулятор организаторов на выходе
OPM при сопоставимой парной базе.

---

## 0. До 17:00 — два блокирующих вопроса к `config/case_z_test.json`

Измерено локально 11 сентября, `.venv`, OPM не запускался:

| Что | Что происходит |
|---|---|
| `water_balance.deficit_m3: null` | `load_case_profile` отвергает: `water_balance.deficit_m3 must be a finite number`. **Этап `intake` падает на первой секунде.** Профиль ждёт число (в примере `config/case_constraints.test.example.json` стоит `0`). |
| `pressure.field_min_bar: 109.431` | `CaseProfile.operating_rules()` отказывает fail-closed: `numeric reservoir pressure thresholds need FPR/RPR in the canonical export; they are not checked here`. Этот вызов стоит в `scripts/propose_track2_policies.py:605`, то есть **этап `search` упадёт через ~2 часа после старта.** |

Оба решения — за ведущим, до заморозки. Ослабление порога (перевод `field_min_bar` в `null`)
означает, что ограничение FPR ≥ 109,431 бар не проверяется и это надо записать в протокол
как непроверенное, а не как выполненное. Альтернатива — поддержка FPR/RPR в каноническом
экспорте до 17:00.

Проверка после правки (секунда, OPM не нужен):

```bash
cd /home/ruslan_safaev/TimesOil && PYTHONPATH=src:scripts .venv/bin/python -c "
from datetime import date
from timesoil.aios.case_profile import load_case_profile
p = load_case_profile('config/case_z_test.json')
print(len(p.operating_rules(wells=('1','2'), start=date(2007,1,1), end=date(2025,9,1))), 'rules')"
```

---

## 1. 17:00 — доставка архива и кода (≈3 минуты)

Архив кладётся в подготовленный каталог на A100, код приезжает **только через git**,
отдельным worktree на замороженном коммите.

```bash
# WSL: архив на сервер
scp <полученный архив> a100-remote:/root/projects/case_z_20260911/case_z.zip

# WSL: заморозить и опубликовать код
git log -1 --format=%H                       # это и есть замороженный коммит FREEZE
git push origin track-2-model-z

# A100: новый worktree ровно на этом коммите, ничего не правим на сервере
ssh a100-remote
cd /root/projects/TimesOil && git fetch --all
git worktree add --detach /root/projects/TimesOil-case-20260911 <FREEZE>
cd /root/projects/TimesOil-case-20260911
sha256sum /root/projects/case_z_20260911/case_z.zip    # записать в журнал дежурного
```

Тот же SHA-256 скрипт сам экспортирует в `TIMESOIL_CASE_SOURCE_SHA256` **до** первого
запуска python, и после приёма переэкспортирует его из `intake/manifest.json`
(`request.source_sha256`) — на случай `--extend-schedule`, который собирает свой архив с
другим хешем. Все ворота сверяются с ним строгим равенством.

Перед запуском: `nvidia-smi` (GPU 5 общий, чужое не трогать), `tmux ls` (чужие сессии не
останавливать). Наши ядра — 14–29, OPM пинится на 30–45.

## 2. Сухой прогон (10 секунд, ничего не создаёт)

```bash
scripts/run_case_z.sh /root/projects/case_z_20260911/case_z.zip --dry-run
```

Печатает все команды в порядке исполнения, с подставленными путями и флагами. Сверить
глазами: профиль — `config/case_z_test.json`, голова — `timesfm-economic-regimes-precise-z-20260910`
(60 эпох), связность — из **нового** прогона кейса, не из учебной геологии.

## 3. Одна команда

```bash
tmux new -s timesoil-case-20260911 \
  "scripts/run_case_z.sh /root/projects/case_z_20260911/case_z.zip 2>&1 | tee /root/case-z-launch.log"
```

Быстрый путь (план A, ≈50 минут, без банка и дообучения, поиск на замороженной голове):

```bash
tmux new -s timesoil-case-20260911 \
  "scripts/run_case_z.sh /root/projects/case_z_20260911/case_z.zip --plan-a 2>&1 | tee /root/case-z-launch.log"
```

Флаги: `--plan-a`, `--skip-bank`, `--skip-finetune`, `--bank-runs N` (по умолчанию 16),
`--epochs N` (20), `--evaluate` (по умолчанию **выключено**), `--extend-schedule`,
`--search-seconds N` (900), `--budget-seconds N` (14400), `--dry-run`.
Переопределяются переменными окружения: `OUT`, `R`, `PY_PROJECT`, `PY_TORCH`, `PROFILE`,
`WEIGHTS`, `CONNECTIVITY`, `CPUS`, `OPM_*`.

Результат — новый каталог `$R/case-z-<UTC-метка>` (`R=/root/projects/TimesOil/results/audit-20260909`).

## 4. Этапы, что смотреть, сколько ждать

| # | Этап | Каталог / выход | Время | Чем проверяется |
|---|---|---|---|---|
| 1 | `intake` | `intake/{request.json,manifest.json,inspect.json}` | ≈3 с | девять именованных проверок в `manifest.checks`, `CycleRequest.from_mapping` внутри команды |
| 2 | `baseline` | `baseline/incumbent/` | ≈8 мин | это парная база аудита; `canonical/{chdd.csv,manifest.json}` |
| 3 | `connectivity` | `connectivity/connectivity.json` | 3–6 мин | INIT/EGRID **этого** прогона, `convertECL` в закреплённом образе |
| 4 | `blocks` | `blocks.json` | ≈1 мин | 6 блоков геометрии для агентов, семейства F5 и отчёта |
| 5 | `bank` | `bank/manifest.json` | ≈1 мин | 34 сценария, кэпы 600/600, `--injection-basis cap` |
| 5а | `bank-subset` | `bank-subset/` | секунды | из 34 берутся `--bank-runs` штук по кругу семейств F1…F5 |
| 5б | `bank-run` | `bank-runs/{cycles,exit,logs,failed.txt}` | ≈4 мин × N ÷ 2 воркера | ненулевой файл в `exit/` — это находка, а не повод перезапустить |
| 5в | `batch-baseline` | `bank-runs/cycles/baseline/` | ≈8 мин | ассемблер не переносит прогоны, инкумбент обязан быть исполнен **внутри** каталога батча |
| 5г | `assemble` | `bank-runs/cycles/{dataset,manifests,manifest.json}` | ≈1 мин | `scenario_id == "baseline"` обязателен, поэтому приём идёт с `--scenario-id baseline` |
| 6 | `finetune` | `training/{full-model.pt,report.json}` | 20–40 мин | старт с 60-эпохной головы, без калибровок режимов и BHP, `--precise-variate-softmax` |
| 6б | `evaluate` | `evaluation/` | ≈5 мин | **выключено**: драйверу нужен сплит 5/3, которого у банка кейса нет |
| 7 | `search` | `search/{candidates.json,proposal-receipt.json,selection-before-opm.json}` | 20–35 мин | **ноль вызовов OPM**; печать выбора = SHA-256 `selection-before-opm.json` |
| 8 | `final` | `final/`, `search/final-verification-attempt.json` | ≈8 мин | **ровно один** OPM, после него график не выбирается заново |
| 9 | `explain` | `explain/interpretability/` | <1 мин | отчёт, не трогает опломбированные артефакты |

Итог: план A ≈ 45–55 минут, полный путь ≈ 2,5–3 часа при 16 прогонах банка.
Измерены 10 сентября только поиск (18 мин 33 с) и финальный полный цикл (7 мин 36 с,
OPM внутри 3 мин 37 с); остальные строки — оценка, а не измерение.

Скрипт печатает остаток бюджета перед каждым необязательным этапом
(`# budget before bank … 240 min remaining`). Если остаток уходит в минус, он это пишет,
но сам ничего не прерывает — решение о переходе на план A принимает дежурный.

За чем следить в `tee`-логе: строки `>>> имя` / `<<< имя exit=0 Nс`. Первый ненулевой
`exit` останавливает цепочку; подробности — в `$OUT/<имя>.log`, сводка — в
`$OUT/protocol.json` (`stages[]` с хешами выходов).

## 5. Запасные пути

**План A (быстрый путь).** Если к 19:00 не пройден этап 6 или упал банк — остановить,
запустить заново с `--plan-a` в **новый** каталог (`OUT=` или просто новая UTC-метка).
Поиск пойдёт на замороженной 60-эпохной голове, обученной на учебной деке: это
законный результат, в протоколе он так и записан (`skip_bank`, `skip_finetune`).
Уже посчитанный каталог не переиспользовать и не «дописывать».

**LLM: Cerebras → Татнефть.** Маршрут выбирается автоматически по наличию файла ключа:
сначала `/root/.config/timesoil/cerebras-key` (`https://api.cerebras.ai/v1`, `qwen-3.8-27b`,
`LLM_REASONING_EFFORT=high`, `LLM_SEED=20260909`), иначе
`/dev/shm/timesoil-tatneft-20260909-key` (`https://litellm.tatneft.guru/v1`, `qwen3.8-27b`).
Оба файла на 11 сентября на месте (проверено `test -s`, содержимое не читалось).
Принудительный переход на Татнефть — задать окружение до запуска:

```bash
export LLM_BASE_URL=https://litellm.tatneft.guru/v1 LLM_MODEL=qwen3.8-27b
export LLM_API_KEY="$(</dev/shm/timesoil-tatneft-20260909-key)"
```

Ключ не печатать, не коммитить, не включать в логи и бандлы. Журнал вызовов —
`$OUT/llm_calls.jsonl`.

**Флаги поиска.** `--search cma --search-seconds --llm-round0 --blocks` скрипт определяет
пробой `propose_track2_policies.py --help` и пишет решение в лог и в
`protocol.json.search_flags`. Если там оказалось `baseline grid + agent rounds only` —
значит CMA-часть в замороженный коммит не попала; это надо доложить, а не дописывать флаги руками.

**Раскладка архива отличается от учебной.** Приём отказывает одной строкой и кодом 2.
Диагностика — отдельной командой, она ничего не пишет:

```bash
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python scripts/intake_case_z.py \
    inspect /root/projects/case_z_20260911/case_z.zip --cut 2006-12-31
```

| Симптом в `inspect` | Что делать |
|---|---|
| не один `.DATA` или не один `INCLUDE` в `SCHEDULE` | распаковать архив в каталог, оставить один дек / одно включение (нужное определяется наличием `WCONPROD`/`WCONINJE`) и подать каталог вместо zip: `CaseArchive` принимает и каталог. SHA-256 пересчитывается и попадает в манифест |
| `deck.unit_system` не `METRIC` | **остановиться и доложить**. Пересчёт `FIELD`→`METRIC` меняет физику и экономику; вслепую не делать |
| расписание обрывается на дате отсечения (ожидаемый случай боевого кейса) | перезапустить с `--extend-schedule`: собирается `intake/case-extended.zip`, все члены байт в байт, дописаны **только** недостающие ежемесячные `DATES`, ни одной записи управления. Внутренние дыры не чинятся — команда откажет. Скрипт сам перепинит `TIMESOIL_CASE_SOURCE_SHA256` на хеш расширенного архива |
| `the case profile schedules a repair on unknown well '<имя>'` | сверить `wells.names` из `inspect` и исправить `repairs` в профиле |
| `cut.unreadable_regimes` не пуст | режим не `LRAT` (добыча) или не `WATER`/`RATE` (закачка). Разобрать эти скважины отдельно; продолжать их режим наугад нельзя |
| `cut.roles` даёт не 57/23 | **доложить измеренное число, не подгонять.** Учебная дека даёт 58 добывающих / 23 нагнетательных при обещанных документом 57 |
| `keywords_after_cut` содержит `WCONPROD`/`WCONINJE` | нам отдали и будущее управление организаторов — записать в протокол |

**OPM упал на одном сценарии банка.** `bank-runs/failed.txt` перечисляет ненулевые
`exit`. Это находка: записать и идти дальше на оставшихся прогонах либо перейти на план A.
Перезапуск того же каталога запрещён (`mkdir` намеренно падает на существующем).

## 6. Что сдаём

`$OUT/protocol.json` (схема `timesoil.case-z-run/v1`) — коммит кода, хеш архива, хеш
профиля, флаги, и по этапам: `exit`, секунды, SHA-256 выходов. Рядом — печать выбора
(`search/selection-before-opm.json`), квитанция единственного финального OPM
(`final/`, `search/final-verification-attempt.json`) и отчёт `explain/interpretability/`.

Контракт пломбы, который скрипт не даёт нарушить и записывает в протокол явно:
`search_opm_calls = 0`, `final_opm_calls_allowed = 1`,
`reselection_after_opm_allowed = false`.
