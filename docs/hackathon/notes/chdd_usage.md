# Запуск калькулятора ЧДД — команды и тесты

Корень калькулятора: `/home/ruslan_safaev/TimesOil/docs/hackathon/chdd/CHDD_PYTHON/`

Зависимость: `openpyxl>=3.1,<4` (не входит в `pyproject.toml` TimesOil — подключается через `uv run --with`).

---

## Быстрый прогон на примере

```bash
cd /home/ruslan_safaev/TimesOil/docs/hackathon/chdd/CHDD_PYTHON
printf '%s\n' "Пример_исходных_данных.xlsx" > ВХОДНОЙ_ФАЙЛ.txt

cd /home/ruslan_safaev/TimesOil
uv run --with openpyxl python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py --start-year 2014
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
uv run --with openpyxl python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --start-year 2014 \
  --output docs/hackathon/chdd/CHDD_PYTHON/output/Расчет_ЧДД_мой_сценарий.xlsx
```

**Трек 2:**

```bash
uv run --with openpyxl python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py --start-year 2007
```

**JSON-дамп** (для diff с организаторами):

```bash
uv run --with openpyxl python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --start-year 2014 \
  --json docs/hackathon/chdd/CHDD_PYTHON/output/result.json
```

**Прямой путь к входу** (минуя `ВХОДНОЙ_ФАЙЛ.txt`):

```bash
uv run --with openpyxl python docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --input docs/hackathon/chdd/CHDD_PYTHON/input/мой_файл.csv \
  --start-year 2014
```

---

## Через `.venv` TimesOil (без uv ephemeral)

Если в `.venv` уже есть `openpyxl`:

```bash
/home/ruslan_safaev/TimesOil/.venv/bin/pip install 'openpyxl>=3.1,<4'   # один раз
/home/ruslan_safaev/TimesOil/.venv/bin/python \
  /home/ruslan_safaev/TimesOil/docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
  --start-year 2014
```

> Предпочтительный способ без изменения `pyproject.toml` — `uv run --with openpyxl`.

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
uv run --with openpyxl python -m unittest discover \
  -s docs/hackathon/chdd/CHDD_PYTHON/tests \
  -p 'test_*.py' -v
```

### Результат прогона (2026-08-17)

| Статус | Тест |
|--------|------|
| **ok** | `test_input_selection` (5 тестов) |
| **ok** | `test_methodology` (9 тестов) |
| **skipped** | `test_full_csv_parity` — нужны `CHDD_NODE` и `CHDD_WEB_CORE` + полный CSV в `input/` |
| **skipped** | `test_web_core_parity` — нужны Node.js и `core.js` |

**Итого: 14 passed, 2 skipped, 0 failed** — `OK`.

Межъязыковая сверка Python ↔ JavaScript (опционально):

```bash
export CHDD_NODE=$(which node)
export CHDD_WEB_CORE=/path/to/core.js
uv run --with openpyxl python -m unittest \
  docs/hackathon/chdd/CHDD_PYTHON/tests/test_web_core_parity.py -v
```

---

## Чеклист перед сдачей

- [ ] Исходник — `.xlsx` или `.csv` с обязательными 14 столбцами
- [ ] Нет `WLPR > 500` м³/сут
- [ ] Нет отрицательных `WLPT_Diff` / `WOMT_Diff` / `WWIT_Diff` (или осознанно приняты исключения)
- [ ] `--start-year` соответствует треку (2014 или 2007)
- [ ] `totalChddM` совпадает с пересчётом организаторов (допуск ~2 % только на стороне симулятора, **не** на ЧДД)
