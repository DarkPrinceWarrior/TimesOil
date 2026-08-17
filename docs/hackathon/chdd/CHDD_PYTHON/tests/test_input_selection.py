from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_FILE = PROJECT_DIR / "РАСЧЕТ_ЧДД.py"
SPEC = importlib.util.spec_from_file_location("chdd_cli", SCRIPT_FILE)
assert SPEC and SPEC.loader
CHDD_CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHDD_CLI)


class InputSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temporary_directory.name)
        self.input_dir = self.base_dir / "input"
        self.input_dir.mkdir()
        self.selection_file = self.base_dir / "ВХОДНОЙ_ФАЙЛ.txt"
        self.norms_file = self.input_dir / "Нормативы_ЧДД.xlsx"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def select(self) -> Path:
        return CHDD_CLI.read_configured_input_file(
            selection_file=self.selection_file,
            input_dir=self.input_dir,
            norms_file=self.norms_file,
        )

    def test_selects_named_xlsx_when_other_sources_are_present(self):
        selected = self.input_dir / "данные.xlsx"
        selected.touch()
        (self.input_dir / "другие.csv").touch()
        self.selection_file.write_text("данные.xlsx\n", encoding="utf-8")

        self.assertEqual(self.select(), selected)

    def test_selects_csv(self):
        selected = self.input_dir / "данные.csv"
        selected.touch()
        self.selection_file.write_text("данные.csv", encoding="utf-8")

        self.assertEqual(self.select(), selected)

    def test_rejects_path_instead_of_file_name(self):
        self.selection_file.write_text("input/данные.xlsx", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "только имя файла"):
            self.select()

    def test_rejects_unsupported_extension(self):
        self.selection_file.write_text("данные.xls", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Допустимы только"):
            self.select()

    def test_reports_missing_selected_file(self):
        self.selection_file.write_text("нет_файла.csv", encoding="utf-8")

        with self.assertRaisesRegex(FileNotFoundError, "не найден файл"):
            self.select()


if __name__ == "__main__":
    unittest.main()
