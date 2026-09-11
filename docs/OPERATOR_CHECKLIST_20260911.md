# Чек-лист дежурного: тестовый кейс трека 2, 11.09.2026, 17:00–21:00

Код заморожен на коммите ветки `track-2-model-z` (см. `git log -1`); на A100 он
развёрнут как `/root/projects/TimesOil-freeze-<sha>`. Ниже — только действия
флагами, окружением и размещением файлов. Порядок — по вероятности отказа.

## 0. Запуск (одна команда)

```bash
ssh a100-remote
cp ~/case_z.zip /root/projects/case_z_20260911/case_z.zip          # архив от организаторов
sha256sum /root/projects/case_z_20260911/case_z.zip                 # в журнал
cd /root/projects/TimesOil-freeze-<sha>
PYTHONPATH=src:scripts /root/projects/TimesOil/.venv/bin/python scripts/intake_case_z.py \
  inspect /root/projects/case_z_20260911/case_z.zip                 # ~3 с: раскладка, скважины, отсечение
tmux new -s case-z-run-$(date +%H%M) \
  "bash scripts/run_case_z.sh /root/projects/case_z_20260911/case_z.zip --plan-a \
     > /root/projects/case_z_20260911/driver.log 2>&1; echo \$? > /root/projects/case_z_20260911/driver.exit"
tail -f /root/projects/case_z_20260911/driver.log
```

План A (`--plan-a`) — повторное использование 60-эпоховых весов: intake →
baseline (OPM ≈4,5 мин) → connectivity → blocks → search (CMA-ES, 15 мин по
умолчанию, `--search-seconds`) → один финальный OPM (≈5 мин) → explain.
Итого ≈ 35–40 мин. Если план A отказан (п. 3), запускать без `--plan-a` —
полный путь с банком и дообучением (≈2,5–3 ч), решение принимать сразу.

Пломба выбора — `search/selection-before-opm.json` (sha256 в `run.log`);
финальный ЧДД — `final/audit.json`; сдаваемое расписание — оверлей победителя
в `final/` (см. §7 RUNBOOK). Трекеру: `wells_schedule.inc` и ЧДД **только за
период управления с 01.01.2007**.

## 1. Расписание кейса обрывается на отсечении 31.12.2006 (ожидаемый случай)

Симптом на `intake`: «schedule does not cover the management period». Запускать с
`--extend-schedule`: первый intake пишет `intake/case-extended.zip`, второй
(инкамбент) читает тот же архив — хеши источника совпадают (проверка за 10 с:
`request.source_sha256` в `intake/manifest.json` и `intake-incumbent/manifest.json`
равны). Не совпали — остановить, не ждать `search`.

## 2. Torch-venv в tmpfs пропал после перезагрузки

До запуска: `test -x /tmp/timesoil-kt3-20260908/venv/bin/python && test -s
$R/timesfm-economic-regimes-precise-z-20260910/training/full-model.pt`. Нет —
`PY_TORCH=` на пересобранное окружение (torch 2.9.1+cu128, timesfm 3.0.1, cma).

## 3. План A и геология кейса

Драйвер сверяет `well_ids`, `static`, `weights` экспорта кейса с файлом геологии
головы; при равенстве поиск получает файл головы с поручительством по хешу
(`TIMESOIL_HEAD_GEOLOGY_VERIFIED_SHA256`, записывается в квитанцию как
`vouched_by_driver`). Отказ «case geology differs from the head's geology» —
другая сетка/фонд: план A невозможен, запускать полный путь без `--plan-a`.

## 4. Имена скважин ремонтов не совпадают с кейсом

`intake` падает за ~3 с: «the case profile schedules a repair on unknown well».
16 ремонтов в `config/case_z_test.json` записаны как `"90"`, `"27"` … Сверить с
`wells.names` из `inspect`, положить исправленный профиль рядом (это данные, не
код) и запустить с `PROFILE=/root/projects/case_z_20260911/case_profile_fixed.json`.
Хеш профиля попадает в квитанцию и пломбу — записать в журнал.

## 5. Архив

Подавать только zip с расширением `.zip`, один `.DATA`, `SCHEDULE` через
`INCLUDE`, UTF-8, METRIC. Каталог или вложенный zip — распаковать, вычистить
(`__MACOSX`, лишние `.DATA`), перепаковать: `cd dir && zip -r ../case_z_fixed.zip .`.
Кодировка cp1251 — `iconv -f cp1251 -t utf-8`, перепаковать, новый sha в журнал.
`unit_system ≠ METRIC` — остановиться и доложить.

## 6. `DATES` и режимы управления

Отказы `intake` без обхода флагами: несколько дат в одном блоке `DATES`, дыра
внутри расписания, нет отчётной даты `2006-12-01`. История на `WCONHIST`
не читается: смотреть в `inspect` `wells.controlled` против `wells.welspecs`,
`cut.roles`, `cut.unreadable_regimes`; режим добычи не `LRAT` или закачка не
`WATER/RATE` — отказ с именем скважины. Скважина, чей `WELSPECS` появляется
после первого месяца управления, роняет OPM на `baseline` (~8 мин).

## 7. Ресурсы

`taskset -c 14-29` (поиск), `OPM_CPU_AFFINITY=30-45` (OPM 16 MPI), GPU 5.
Проверить `nproc`, `nvidia-smi`, `tmux ls`; занято — переопределить `CPUS=`,
`OPM_CPU_AFFINITY=`, `CUDA_VISIBLE_DEVICES=`. Чужие процессы не трогать. Имя
tmux-сессии уникальное. LLM: Cerebras через прокси `127.0.0.1:10809`; прокси
лежит — `LLM_BASE_URL=https://litellm.tatneft.guru/v1 LLM_MODEL=qwen3.8-27b
LLM_API_KEY="$(</dev/shm/timesoil-tatneft-20260909-key)"` до запуска. Ключи
не печатать. Каждый запуск — новый каталог (`mkdir` намеренно падает).

## 8. Что воротами не проверяется (записать как непроверенное)

`pressure.field_min_bar: null` — среднее пластовое давление ≥ 108 атм
(109,431 бар) проверяется руками по `FPR` в summary финального OPM;
`water_balance.deficit_m3: null` — внешняя вода в пределах 600 м³/сут. BHP
проверяется на отчётных шагах. Отказ ворот — результат, который докладывается.
