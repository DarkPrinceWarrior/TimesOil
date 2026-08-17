from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from chdd_model import DEFAULT_ASSUMPTIONS, DEFAULT_PUMPS, REQUIRED_COLUMNS, compute_calculation


def row(date_value: str, well: str, **values):
    result = {column: 0 for column in REQUIRED_COLUMNS}
    result.update({"DATA": date_value, "well": well, "WEFF": 1})
    result.update(values)
    return result


def calculate(records, **assumptions):
    merged = dict(DEFAULT_ASSUMPTIONS)
    merged.update(assumptions)
    return compute_calculation(
        records,
        headers=REQUIRED_COLUMNS,
        assumptions=merged,
        pumps=DEFAULT_PUMPS,
        start_date="2014-01-01",
    )


class MethodologyTests(unittest.TestCase):
    def test_stop_start_and_conversion_are_paid_correctly(self):
        result = calculate(
            [
                row("2014-01-01", "A", WLPR=100, WOMR=10, WLPT_Diff=1000, WOMT_Diff=400),
                row("2014-02-01", "A"),
                row("2014-03-01", "A", WLPR=100, WOMR=10, WLPT_Diff=1000, WOMT_Diff=400),
                row("2014-01-01", "B", WLPR=100, WOMR=10, WLPT_Diff=1000, WOMT_Diff=400),
                row("2014-02-01", "B", WWIR=80, WWIT_Diff=2000),
            ]
        )
        annual = result["annual"][0]
        self.assertEqual(annual["stopCount"], 1)
        self.assertEqual(annual["startCount"], 1)
        self.assertEqual(annual["startStopCount"], 2)
        self.assertAlmostEqual(annual["startStopCostM"], 2.0)
        self.assertEqual(annual["conversionCount"], 1)
        self.assertAlmostEqual(annual["conversionOpexM"], 5.0)
        self.assertEqual(annual["pumpChanges"], 0)

    def test_annual_profit_tax_and_fcf(self):
        result = calculate(
            [row("2014-01-01", "A", WLPR=100, WOMR=10, WLPT_Diff=1000, WOMT_Diff=400)]
        )
        annual = result["annual"][0]
        self.assertAlmostEqual(annual["profitTaxM"], max(0, annual["taxableProfitM"]) * 0.25)
        self.assertAlmostEqual(
            annual["fcfM"], annual["ebitdaM"] - annual["profitTaxM"] - annual["capexM"]
        )

    def test_no_profit_tax_for_annual_loss(self):
        result = calculate([row("2014-01-01", "I", WWIR=50, WWIT_Diff=1000)])
        annual = result["annual"][0]
        self.assertLess(annual["taxableProfitM"], 0)
        self.assertEqual(annual["profitTaxM"], 0)

    def test_rows_with_any_negative_monthly_value_are_excluded(self):
        result = calculate(
            [
                row("2014-01-01", "A", WLPR=100, WOMR=10, WLPT_Diff=200, WOMT_Diff=100),
                row("2014-02-01", "A", WLPR=100, WOMR=10, WLPT_Diff=200, WOMT_Diff=-40),
                row("2014-03-01", "A", WLPR=100, WOMR=10, WLPT_Diff=-10, WOMT_Diff=20),
                row("2014-04-01", "A", WWIR=50, WWIT_Diff=-10),
            ]
        )
        self.assertAlmostEqual(result["annual"][0]["oilKt"], 0.1)
        self.assertAlmostEqual(result["summary"]["totalOilKt"], 0.1)
        self.assertEqual(len(result["fieldMonthly"]), 1)
        self.assertEqual(result["maxDate"], "2014-01-01")
        self.assertEqual(result["diagnostics"]["negativeOilRows"], 1)
        self.assertEqual(result["diagnostics"]["excludedNegativeRows"], 3)
        self.assertIn("полностью исключено", "\n".join(result["diagnostics"]["warnings"]))

    def test_discount_factor(self):
        result = calculate(
            [
                row("2014-01-01", "D", WLPR=20, WOMR=2, WLPT_Diff=100, WOMT_Diff=50),
                row("2015-01-01", "D", WLPR=20, WOMR=2, WLPT_Diff=100, WOMT_Diff=50),
            ]
        )
        self.assertAlmostEqual(result["annual"][0]["discountFactor"], 1.0)
        self.assertAlmostEqual(result["annual"][1]["discountFactor"], 1 / 1.1)

    def test_pump_up_and_down_rules(self):
        result = calculate(
            [
                row("2014-01-01", "P", WLPR=100, WOMR=10),
                row("2014-02-01", "P", WLPR=170, WOMR=15),
                row("2014-03-01", "P", WLPR=40, WOMR=4),
            ]
        )
        annual = result["annual"][0]
        self.assertEqual(annual["pumpUps"], 1)
        self.assertEqual(annual["pumpDowns"], 0)
        self.assertEqual(annual["pumpChanges"], 1)

        down_result = calculate(
            [
                row("2014-01-01", "P", WLPR=100, WOMR=10),
                row("2014-02-01", "P", WLPR=170, WOMR=15),
                row("2014-03-01", "P", WLPR=20, WOMR=2),
            ]
        )
        self.assertEqual(down_result["annual"][0]["pumpDowns"], 1)

    def test_initial_pump_is_free_by_default(self):
        result = calculate([row("2014-01-01", "A", WLPR=100, WOMR=10)])
        self.assertEqual(result["annual"][0]["pumpChanges"], 0)
        self.assertEqual(result["annual"][0]["pumpCapexM"], 0)

    def test_wlpr_above_500_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "WLPR свыше 500"):
            calculate([row("2014-01-01", "X", WLPR=501, WOMR=10)])

    def test_fixed_methodology_values_cannot_be_overridden(self):
        result = calculate(
            [row("2014-01-01", "A", WLPR=20, WOMR=2)],
            stopStartCostM=999,
            conversionBaseCostM=999,
            downsizeThreshold=1,
        )
        self.assertEqual(result["assumptions"]["stopStartCostM"], 1)
        self.assertEqual(result["assumptions"]["conversionBaseCostM"], 5)
        self.assertEqual(result["assumptions"]["downsizeThreshold"], 100)


if __name__ == "__main__":
    unittest.main()
