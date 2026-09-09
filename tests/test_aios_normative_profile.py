from pathlib import Path
import shutil

from openpyxl import load_workbook
import pytest

from timesoil.aios.economics import CHDDEconomicsAdapter, EconomicsError


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
    book["Нормативы"].append(["unaccountedRepairCost", "repair", 7])
    book.save(norms)
    book.close()
    with pytest.raises(EconomicsError, match="unknown or duplicate.*unaccountedRepairCost"):
        CHDDEconomicsAdapter(source).normative_profile()
