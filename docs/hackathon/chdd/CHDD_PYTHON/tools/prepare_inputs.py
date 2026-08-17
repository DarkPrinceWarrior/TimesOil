#!/usr/bin/env python3

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from excel_io import create_input_template, create_norms_workbook


def main() -> int:
    input_dir = BASE_DIR / "input"
    create_norms_workbook(input_dir / "Нормативы_ЧДД.xlsx")
    create_input_template(input_dir / "Пример_исходных_данных.xlsx", with_example=True)
    create_input_template(input_dir / "Макет_исходных_данных.xlsx", with_example=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

