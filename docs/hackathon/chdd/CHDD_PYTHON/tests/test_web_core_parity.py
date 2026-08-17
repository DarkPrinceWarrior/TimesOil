from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from datetime import date, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from chdd_model import compute_calculation
from excel_io import load_norms, load_source_records


class WebCoreParityTests(unittest.TestCase):
    def test_annual_results_match_corrected_javascript_core(self):
        node = os.environ.get("CHDD_NODE") or shutil.which("node")
        core = Path(
            os.environ.get(
                "CHDD_WEB_CORE",
                str(PROJECT_DIR.parent / "AIOS_ECONOMIC_reconciled" / "core.js"),
            )
        )
        if not node or not core.exists():
            self.skipTest("Для межъязыковой сверки нужны Node.js и corrected core.js.")

        headers, records = load_source_records(PROJECT_DIR / "input" / "Пример_исходных_данных.xlsx")
        assumptions, pumps = load_norms(PROJECT_DIR / "input" / "Нормативы_ЧДД.xlsx")
        normalized_for_json = []
        for record in records:
            normalized_record = {}
            for key, value in record.items():
                normalized_record[key] = value.isoformat() if isinstance(value, (date, datetime)) else value
            normalized_for_json.append(normalized_record)

        python_result = compute_calculation(
            records,
            headers=headers,
            assumptions=assumptions,
            pumps=pumps,
            start_date="2014-01-01",
        )
        payload = {
            "headers": headers,
            "records": normalized_for_json,
            "assumptions": assumptions,
            "pumps": pumps,
            "startDate": "2014-01-01",
        }
        script = """
const fs = require('node:fs');
const Core = require(process.argv[1]);
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = Core.computeCalculation(payload);
process.stdout.write(JSON.stringify({annual: result.annual, summary: result.summary}));
"""
        completed = subprocess.run(
            [node, "-e", script, str(core)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=True,
        )
        javascript_result = json.loads(completed.stdout)

        keys = [
            "oilKt",
            "liquidKt",
            "injectionKm3",
            "averageActiveWells",
            "revenueM",
            "pumpCapexM",
            "capexM",
            "oilOpexM",
            "liquidOpexM",
            "injectionOpexM",
            "fundOpexM",
            "pumpOperationM",
            "startStopCount",
            "startStopCostM",
            "conversionCount",
            "conversionOpexM",
            "propertyTaxM",
            "opexM",
            "depreciationM",
            "deductionsM",
            "taxableProfitM",
            "profitTaxM",
            "ebitdaM",
            "fcfM",
            "discountFactor",
            "chddM",
            "discountedInflowM",
            "discountedOutflowM",
            "profitabilityIndex",
            "cumulativeChddM",
            "pumpChanges",
        ]
        self.assertEqual(len(python_result["annual"]), len(javascript_result["annual"]))
        for python_year, javascript_year in zip(python_result["annual"], javascript_result["annual"]):
            self.assertEqual(python_year["year"], javascript_year["year"])
            for key in keys:
                self.assertAlmostEqual(
                    python_year[key],
                    javascript_year[key],
                    places=9,
                    msg=f"Расхождение Python/JavaScript: {python_year['year']} {key}",
                )
            self.assertEqual(python_year["pumpInstallCounts"], javascript_year["pumpInstallCounts"])


if __name__ == "__main__":
    unittest.main()
