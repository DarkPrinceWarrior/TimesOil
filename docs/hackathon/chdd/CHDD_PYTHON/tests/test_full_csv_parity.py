from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from chdd_model import compute_calculation
from excel_io import load_norms, load_source_records


class FullCsvParityTests(unittest.TestCase):
    def test_python_and_javascript_match_on_packaged_full_csv(self):
        node = os.environ.get("CHDD_NODE") or shutil.which("node")
        core = Path(os.environ.get("CHDD_WEB_CORE", ""))
        if not node or not core.is_file():
            self.skipTest("Set CHDD_NODE and CHDD_WEB_CORE to run full cross-language parity.")

        csv_path = PROJECT_DIR / "input" / "Модель2трек_Оптимальный_002_2_OPTIMIZED_AlenkinDA.csv"
        norms_path = PROJECT_DIR / "input" / "Нормативы_ЧДД.xlsx"
        headers, records = load_source_records(csv_path)
        assumptions, pumps = load_norms(norms_path)
        self.assertGreater(len(records), 30_000, "Packaged CSV is truncated.")

        python_result = compute_calculation(
            records,
            headers=headers,
            assumptions=assumptions,
            pumps=pumps,
        )
        payload = {
            "headers": headers,
            "records": records,
            "assumptions": assumptions,
            "pumps": pumps,
        }
        script = """
const fs = require('node:fs');
const Core = require(process.argv[1]);
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = Core.computeCalculation(payload);
process.stdout.write(JSON.stringify({annual: result.annual, summary: result.summary, diagnostics: result.diagnostics}));
"""
        completed = subprocess.run(
            [node, "-e", script, str(core)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=True,
        )
        javascript_result = json.loads(completed.stdout)

        summary_keys = [
            "totalChddM", "totalFcfM", "totalOilKt", "totalLiquidKt",
            "totalInjectionK", "totalRevenueM", "totalOpexM", "totalCapexM",
            "profitabilityIndex", "pumpChanges", "startStopCount", "conversionCount",
        ]
        for key in summary_keys:
            self.assertAlmostEqual(
                python_result["summary"][key],
                javascript_result["summary"][key],
                places=8,
                msg=f"Summary mismatch: {key}",
            )

        annual_keys = [
            "oilKt", "liquidKt", "injectionKm3", "revenueM", "opexM", "capexM",
            "profitTaxM", "fcfM", "chddM", "cumulativeChddM", "profitabilityIndex",
            "pumpChanges", "startStopCount", "conversionCount",
        ]
        self.assertEqual(len(python_result["annual"]), len(javascript_result["annual"]))
        for python_year, javascript_year in zip(python_result["annual"], javascript_result["annual"]):
            self.assertEqual(python_year["year"], javascript_year["year"])
            for key in annual_keys:
                self.assertAlmostEqual(
                    python_year[key],
                    javascript_year[key],
                    places=8,
                    msg=f"Annual mismatch: {python_year['year']} {key}",
                )

        self.assertEqual(python_result["diagnostics"]["negativeOilRows"], 90)
        self.assertEqual(javascript_result["diagnostics"]["negativeOilRows"], 90)
        self.assertEqual(python_result["diagnostics"]["excludedNegativeRows"], 102)
        self.assertEqual(javascript_result["diagnostics"]["excludedNegativeRows"], 102)


if __name__ == "__main__":
    unittest.main()
