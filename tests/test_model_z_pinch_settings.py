from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pinch_study", Path(__file__).parents[1] / "scripts/check_model_z_pinch.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_changed_financial_results_preserve_strict_settings_comparison():
    original = dict(calculator_sha256={"calculator": "a"}, norms_sha256="b",
                    norms_source_sha256="c", start_year=2007, assumption_overrides={"chargeInitialPump": False},
                    management_period=dict(start_inclusive="2007-01-01", end_exclusive="2025-09-01",
                                           months=["2007-01"], historical_cash_flows_included=False,
                                           history_use="state", total_chdd_m=11873, profitability_index=1.25))
    changed = deepcopy(original)
    changed["management_period"].update(total_chdd_m=7750, profitability_index=1.23)
    study.compare_economics_settings(changed, original)
    for key in original:
        if key == "management_period":
            continue
        bad = deepcopy(changed)
        bad[key] = None
        with pytest.raises(AssertionError, match="economics drift"):
            study.compare_economics_settings(bad, original)
    for key in ("start_inclusive", "end_exclusive", "months", "historical_cash_flows_included", "history_use", "unexpected"):
        bad = deepcopy(changed)
        bad["management_period"][key] = None
        with pytest.raises(AssertionError, match="management_period"):
            study.compare_economics_settings(bad, original)
