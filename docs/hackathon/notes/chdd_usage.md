# Запуск калькулятора ЧДД — команды и тесты

Корень калькулятора: `/home/ruslan_safaev/TimesOil/docs/hackathon/chdd/CHDD_PYTHON/`

Зависимость `openpyxl>=3.1` закреплена в `pyproject.toml` и `uv.lock`. Окружение
устанавливается и запускается только через `uv`.

## Базовые расчёты организаторов от 31.08.2026

Четыре исходных файла организаторов сохранены отдельно от наших запусков:
[chdd/reference_baselines](../chdd/reference_baselines/README.md).

- Model Y: 5326,453 млн руб. в OPM Flow и 5261,159 млн руб. в tNavigator;
- Model Z: 5157,016 млн руб. в OPM Flow и 5218,945 млн руб. в tNavigator;
- расхождение между симуляторами: 1,241% и 1,187% соответственно.

В файлах накопление начинается с первого указанного года, а не с конкурсных
дат 2014/2007. Model Y OPM охватывает 2007–2015; точное последнее значение —
`5326.453465501771` млн руб. Сопоставимый reference от 2014 рассчитывается из
опубликованных FCF:

$$
610.387348812715 + \frac{519.0309811955153}{1.1}
= 1082.233695354093\ \text{млн руб.}
$$

P0 единиц закрыт: canonical-экспорт формирует `WLPT` и `WLPT_Diff` как массу
жидкости в тоннах. Новый расчёт от 2014 даёт `1082.233753351284` млн руб.;
отклонение от reference равно `0.000057997191` млн руб. и меньше допуска,
обусловленного точностью исходных `float32`-векторов. Полная история Model Y с
учётом первоначального парка насосов даёт `5334.357811` млн руб. против
организаторского OPM `5326.453466`.

Для Model Z полная базовая история с профилем организаторов даёт
`5181.184136469775` млн руб. против OPM `5157.016329811206`; отклонение
`24.167806659` меньше опубликованного разрыва OPM↔tNavigator
`61.928578273`. Насосы совпадают с reference: `145` замен и `261.0` млн руб.
эксплуатационных затрат.

Профиль задаётся явно и попадает в provenance:

- `charge_initial_pump=true` — сверка полной истории от начала модели;
- `charge_initial_pump=false` — оптимизационный горизонт, где установленные к
  его началу насосы являются уже понесёнными затратами.

---

## Быстрый прогон на примере

```bash
cd /home/ruslan_safaev/TimesOil/docs/hackathon/chdd/CHDD_PYTHON
printf '%s\n' "Пример_исходных_данных.xlsx" > ВХОДНОЙ_ФАЙЛ.txt

cd /home/ruslan_safaev/TimesOil
uv sync --locked
uv run python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py --start-year 2014
```

Ожидаемый вывод (проверено локально):

```
Готово
Файл: .../output/Расчет_ЧДД_Пример_исходных_данных.xlsx
Период: 2014–2015
ЧДД: 54.106 млн руб.
ИДДз: 1.193
```

---

## Свой сценарий (CSV/XLSX из симулятора)

1. Скопировать файл в `input/`.
2. Указать имя в `ВХОДНОЙ_ФАЙЛ.txt` (одна строка, без пути).
3. Запустить с нужным годом старта:

**Трек 1:**

```bash
cd /home/ruslan_safaev/TimesOil
uv run python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --start-year 2014 \
  --output docs/hackathon/chdd/CHDD_PYTHON/output/Расчет_ЧДД_мой_сценарий.xlsx
```

**Трек 2:**

```bash
uv run python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py --start-year 2007
```

**JSON-дамп** (для diff с организаторами):

```bash
uv run python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --start-year 2014 \
  --json docs/hackathon/chdd/CHDD_PYTHON/output/result.json
```

**Прямой путь к входу** (минуя `ВХОДНОЙ_ФАЙЛ.txt`):

```bash
uv run python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --input docs/hackathon/chdd/CHDD_PYTHON/input/мой_файл.csv \
  --start-year 2014
```

---

## Через проектное окружение TimesOil

```bash
cd /home/ruslan_safaev/TimesOil
uv sync --locked
uv run python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --start-year 2014
```

---

## Windows

```bat
cd docs\hackathon\chdd\CHDD_PYTHON
echo Пример_исходных_данных.xlsx> ВХОДНОЙ_ФАЙЛ.txt
ЗАПУСК_WINDOWS.bat
```

Или вручную: `py РАСЧЕТ_ЧДД.py --start-year 2014`.

---

## Unit-тесты

```bash
cd /home/ruslan_safaev/TimesOil
uv run python -m unittest discover \
  -s docs/hackathon/chdd/CHDD_PYTHON/tests \
  -p 'test_*.py' -v
```

### Результат прогона калькулятора (2026-08-31)

| Статус | Тест |
|--------|------|
| **ok** | `test_input_selection` (5 тестов) |
| **ok** | `test_methodology` (9 тестов) |
| **skipped** | `test_full_csv_parity` — нужны `CHDD_NODE` и `CHDD_WEB_CORE` + полный CSV в `input/` |
| **skipped** | `test_web_core_parity` — нужны Node.js и `core.js` |

**Итого: 14 passed, 2 skipped, 0 failed** — `OK`. Полный проектный шлюз:
**142 passed, 2 skipped, 2 subtests passed**.

Межъязыковая сверка Python ↔ JavaScript (опционально):

```bash
export CHDD_NODE=$(which node)
export CHDD_WEB_CORE=/path/to/core.js
uv run python -m unittest \
  docs/hackathon/chdd/CHDD_PYTHON/tests/test_web_core_parity.py -v
```

---

## Чеклист перед сдачей

- [ ] Исходник — `.xlsx` или `.csv` с обязательными 14 столбцами
- [ ] `WLPT` и `WLPT_Diff` — масса жидкости в т, не поверхностный объём в м³
- [ ] Нет `WLPR > 500` м³/сут
- [ ] Нет отрицательных `WLPT_Diff` / `WOMT_Diff` / `WWIT_Diff` (или осознанно приняты исключения)
- [ ] `--start-year` соответствует треку (2014 или 2007)
- [ ] `totalChddM` совпадает с пересчётом организаторов (допуск ~2 % только на стороне симулятора, **не** на ЧДД)
