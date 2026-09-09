import json
import shutil

from openpyxl import load_workbook
import pytest

from timesoil.aios.economics import CHDD_FIELDS, CHDDEconomicsAdapter, EconomicsError


def test_profile_matches_official_result_and_rejects_unaccounted_norms(tmp_path):
    adapter = CHDDEconomicsAdapter()
    profile = adapter.normative_profile()
    assert len(profile["assumptions"]) == 25
    assert len(profile["pumps"]) == 12
    assert len(profile["source_sha256"]) == 4
    assert profile["assumptions"]["injectionOpexRubM3"] == 30
    assert profile["assumptions"]["conversionBaseCostM"] == 5
    assert adapter.normative_profile(charge_initial_pump=True)["assumptions"]["chargeInitialPump"] is True
    source = tmp_path / "calculator"
    shutil.copytree(adapter.chdd_dir, source)
    norms = source / "input/Нормативы_ЧДД.xlsx"
    book = load_workbook(norms)
    overrides = {"annualDepreciationM": 6, "residualStartM": 24, "residualEndM": 12,
                 "conversionBaseCostM": 99}
    for row in book["Нормативы"].iter_rows(min_row=2):
        if row[0].value in overrides:
            row[2].value = overrides[row[0].value]
    book.save(norms)
    modified = CHDDEconomicsAdapter(source)
    assert modified.normative_profile()["assumptions"]["annualDepreciationM"] == 6
    assert modified.normative_profile()["assumptions"]["conversionBaseCostM"] == 5
    row = {field: 0 for field in CHDD_FIELDS}
    row.update(DATA="2014-01-01", well="P1", WLPR=100, WOMR=30, WLPT=3100, WOMT=1000,
               WLPT_Diff=3100, WOMT_Diff=1000)
    calculated = modified.calculate([row], start_year=2014, output_dir=tmp_path / "costs")
    actual = json.loads((calculated.output_dir / "result.json").read_text())["fieldMonthly"][0]
    assert actual["depreciationM"] == .5
    assert actual["propertyTaxM"] == pytest.approx(.033)
    book["Нормативы"].append(["unaccountedRepairCost", "repair", 7])
    book.save(norms)
    book.close()
    with pytest.raises(EconomicsError, match="unknown or duplicate.*unaccountedRepairCost"):
        CHDDEconomicsAdapter(source).normative_profile()
