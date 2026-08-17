#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from chdd_model import compute_calculation
from excel_io import load_norms, load_source_records, save_result_json, write_annual_report


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
NORMS_FILE = INPUT_DIR / "Нормативы_ЧДД.xlsx"
INPUT_SELECTION_FILE = BASE_DIR / "ВХОДНОЙ_ФАЙЛ.txt"


def read_configured_input_file(
    selection_file: Path = INPUT_SELECTION_FILE,
    input_dir: Path = INPUT_DIR,
    norms_file: Path = NORMS_FILE,
) -> Path:
    if not selection_file.is_file():
        raise FileNotFoundError(
            f"Не найден файл выбора исходных данных: {selection_file}. "
            "В нем должно быть указано имя файла .xlsx или .csv из папки input."
        )

    lines = [
        line.strip()
        for line in selection_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(
            f"В файле {selection_file.name} должна быть ровно одна непустая строка "
            "с именем исходного файла."
        )

    file_name = lines[0].strip("\"'").strip()
    if not file_name or "/" in file_name or "\\" in file_name or Path(file_name).is_absolute():
        raise ValueError(
            f"В файле {selection_file.name} укажите только имя файла без пути, "
            "например: Пример_исходных_данных.xlsx"
        )
    if Path(file_name).suffix.casefold() not in {".xlsx", ".csv"}:
        raise ValueError(
            f"Файл {file_name!r} имеет неподдерживаемое расширение. "
            "Допустимы только .xlsx и .csv."
        )
    if file_name.casefold() == norms_file.name.casefold():
        raise ValueError(
            f"В файле {selection_file.name} указан файл нормативов. "
            "Укажите имя файла с исходными данными."
        )
    if file_name.startswith("~$"):
        raise ValueError("Нельзя использовать временный файл, созданный Excel.")

    input_file = input_dir / file_name
    if not input_file.is_file():
        raise FileNotFoundError(
            f"В папке input не найден файл, указанный в {selection_file.name}: {file_name}"
        )
    return input_file


def calculate(
    input_file: Path,
    norms_file: Path,
    output_file: Path,
    start_year: int | None = None,
    json_file: Path | None = None,
) -> dict:
    headers, records = load_source_records(input_file)
    assumptions, pumps = load_norms(norms_file)
    result = compute_calculation(
        records,
        headers=headers,
        assumptions=assumptions,
        pumps=pumps,
        start_date=f"{start_year}-01-01" if start_year else None,
        source_file=str(input_file.resolve()),
    )
    write_annual_report(result, output_file)
    if json_file:
        save_result_json(result, json_file)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Расчет ЧДД из файла, указанного в {INPUT_SELECTION_FILE.name}."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Необязательно: путь к исходному файлу вместо выбора из файла настройки.",
    )
    parser.add_argument("--norms", type=Path, default=NORMS_FILE)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_file = args.input or read_configured_input_file()
        output_file = args.output or OUTPUT_DIR / f"Расчет_ЧДД_{input_file.stem}.xlsx"
        result = calculate(
            input_file=input_file,
            norms_file=args.norms,
            output_file=output_file,
            start_year=args.start_year,
            json_file=args.json,
        )
    except Exception as error:
        print(f"Ошибка: {error}")
        return 1

    print("Готово")
    print(f"Файл: {output_file.resolve()}")
    print(f"Период: {result['startDate'][:4]}–{result['maxDate'][:4]}")
    print(f"ЧДД: {result['summary']['totalChddM']:.3f} млн руб.")
    print(f"ИДДз: {result['summary']['profitabilityIndex']:.3f}")
    for warning in result["diagnostics"].get("warnings", []):
        print(f"Внимание: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
