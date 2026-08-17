# Модели хакатона AIOS

Краткая инвентаризация официальных архивов гидродинамических моделей и рекомендации по их использованию.

## Треки и актуальные архивы

| Трек | Модель | Актуальный архив | Путь в репозитории |
|------|--------|------------------|--------------------|
| **Трек 1** | **Model Y** | `Model_Y (3).zip` | `docs/hackathon/models/model_y/` |
| **Трек 2** | **Model Z** | **`Model_Z_final_OPM.zip`** | `docs/hackathon/models/model_z_final_opm/` |

**Model_Z_final_OPM заменяет старый Model_Z.zip.** Старый архив (`models/model_z_old/Model_Z.zip`, дата 2026-08-07) сохранён только для сравнения; для прогонов и сдачи используйте **final OPM**.

Сводка по трекам и ЧДД — [tracks.md](tracks.md). Симуляторы и открытые полигоны — [simulators.md](simulators.md).

---

## Model Y (трек 1)

**Архив:** `Model_Y (3).zip` (~7,2 МБ сжатый, ~46 МБ распакованный, 11 файлов, дата 2026-07-28).

| Файл | Размер | Назначение |
|------|--------|------------|
| `MODEL_Y/MODEL_Y.DATA` | 1 КБ | Главный DATA-deck |
| `MODEL_Y/INCLUDE/DemoSpe_002_2_grid.inc` | **45,8 МБ** | Сетка (не извлекать целиком в git) |
| `MODEL_Y/INCLUDE/DemoSpe_002_2_sch.inc` | 52 КБ | Расписание |
| `MODEL_Y/INCLUDE/DemoSpe_002_2_regs.inc` | 97 КБ | Регионы |
| `MODEL_Y/INCLUDE/DemoSpe_002_2_pvt.inc` | 4 КБ | PVT |
| `MODEL_Y/INCLUDE/DemoSpe_002_2_rp.inc` | 2 КБ | ОФП (STONE2 в DATA) |
| `MODEL_Y/INCLUDE/DemoSpe_002_2_init.inc` | 305 Б | Начальные условия |
| `MODEL_Y/SUMMARY/SUMMARY.INC` | 1 КБ | Сводка (полевые + скважинные F*/W*/G*) |
| `MODEL_Y/USER/MODEL_Y/script_1.py` | 3,4 КБ | Экспорт отчёта tNavigator → CSV |

**Ключевые параметры DATA:**

- **START:** 01 MAY 2007
- **DIMENS:** 49 × 47 × 141
- **WELLDIMS:** 54 118 3 54 …
- **FAULTDIM:** 479
- **EQLOPTS THPRES**, **MESSAGES**, **IMPLICIT** — типичные ключи tNavigator
- Генератор: tNavigator (RFDynamics)

**Расписание (`DemoSpe_002_2_sch.inc`):**

- **103 шага DATES:** 01 JUN 2007 … 01 DEC 2015
- **49 скважин** в WELSPECS (33 добывающие + 16 нагнетательных по методике трека)
- **WCONPROD:** 58 блоков, **WCONINJE:** 27 блоков
- **WCONHIST** на старте истории, далее WCONPROD / WELSPECS + COMPDAT

**Старт учёта ЧДД по регламенту:** 01.01.2014 (см. [tracks.md](tracks.md)); история модели с 2007-05/06.

---

## Model Z (трек 2)

**Актуальный архив:** `Model_Z_final_OPM.zip` (~18,4 МБ сжатый, ~80 МБ распакованный, 14 файлов, дата 2026-08-14).

**Устаревший:** `Model_Z.zip` (~17,7 МБ, 8 файлов, 2026-08-07) — без папки USER и без правок под OPM Flow.

| Параметр | Model Z (old) | Model Z final OPM |
|----------|---------------|-------------------|
| **START** | 01 JUN 1991 | 01 JUN 1991 |
| **DIMENS** | 91 × 102 × 59 | 91 × 102 × 59 |
| **WELLDIMS** | 137 109 2 137 | 137 109 2 137 |
| **Шагов DATES** | 371 | 371 |
| **Диапазон DATES** | 01 NOV 1994 … 01 SEP 2025 | то же |
| **WCONINJE** | 339 | 338 |
| **WCONPROD** | 371 | 370 |
| **Уник. нагнетатели (WCONINJE)** | 41 | 41 |
| **Уник. добывающие (WCONPROD)** | 92 | 92 |
| **WELSPECS (добывающие)** | 103 | 103 |

Подробное побайтовое сравнение — [model_z_opm_diff.md](model_z_opm_diff.md).

**Старт учёта ЧДД:** 01.01.2007.

---

## Как получить «гидрокартинки» и обучающие прогоны

Организаторы **не выкладывают дополнительные дампы результатов симуляции** (суррогатные обучающие выборки для Model Z). Их нужно **сгенерировать самим**, прогоняя модель на сценариях, которые выдаёт ваша МАС.

| Инструмент | Когда использовать |
|------------|-------------------|
| **tNavigator** | Если есть лицензия — нативный формат исходников (Model Y; старый Model Z с WELLTRACK) |
| **[OPM Flow](https://opm-project.org/?page_id=19)** | **Рекомендуется для Model Z final OPM** — открытый симулятор, Eclipse/INCLUDE |
| **[MRST](https://www.sintef.no/projectweb/mrst/)** | Прототипы, исследования, постобработка полей |

Типовой цикл для трека 2:

1. МАС генерирует `wells_schedule.inc` (или эквивалент).
2. Подставить в SCHEDULE / прогнать **OPM Flow** (или tNavigator для сверки).
3. Считать SUMMARY / RFT / restart → признаки для суррогата и финальная верификация кандидата.

`script_1.py` в USER — вспомогательный экспорт tNavigator в CSV (`E:\reports\…`); для OPM Flow используйте стандартный post-processing (Python + `.SMSPEC`/`.UNSMRY` или конвертеры).

---

## Открытые датасеты (опционально)

Для отладки суррогата, пайплайна ЧДД и генератора расписания — **не замена** официальных Model Y / Model Z:

- [OPM Data](https://github.com/OPM/opm-data) — SPE, Norne и др.
- [Equinor Volve](https://data.equinor.com/dataset/volve)
- [ORSD](https://developer.ibm.com/exchanges/data/all/oil-reservoir-simulations/) — массовые прогоны OPM на SPE9
- [TPU Reservoir Models](https://hw.tpu.ru/en/dataset/) — 1500 секторных моделей (запрос доступа)

---

## Локальная распаковка для анализа

Без полных `*grid.inc` (>40–70 МБ) маленькие файлы лежат в:

- `models/model_y/unpacked_small/` — DATA, SUMMARY, `script_1.py`, HEAD/TAIL `*_sch.inc`
- `models/model_z_old/unpacked_small/` — DATA, summary, init, props, HEAD/TAIL sch
- `models/model_z_final_opm/unpacked_small/` — то же + `script_1.py`

Полные сетки остаются только внутри zip; для прогона копируйте архив целиком на машину с симулятором.
