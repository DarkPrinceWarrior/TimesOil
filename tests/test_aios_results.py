from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from timesoil.aios import results


def _build_run(root: Path) -> Path:
    run = root / "case-z-test"
    canonical = run / "final" / "cycle-full" / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "chdd.csv").write_text(
        "DATA,well,WLPT,WLPR,WOMT,WOMR,WWIR,WWIT,THP,BHP,WEFF,"
        "WLPT_Diff,WOMT_Diff,WWIT_Diff\n"
        "2026-01-31,P1,3100,100,850,27,0,0,10,120,1,3100,850,0\n"
        "2026-02-28,P1,5900,100,1700,27,0,0,10,118,1,2800,850,0\n"
        "2026-01-31,I1,0,0,0,0,120,3720,10,220,1,0,0,3720\n"
        "2026-02-28,I1,0,0,0,0,120,7080,10,220,0,0,0,3360\n",
        encoding="utf-8",
    )
    (run / "final" / "audit.json").write_text(
        json.dumps({"accepted": True, "commit": "deadbeef"}), encoding="utf-8"
    )
    economics = run / "final" / "economics-final"
    economics.mkdir()
    (economics / "result.json").write_text(
        json.dumps({"final_m": 1150.0, "incumbent_m": 1000.0, "oil_total_t": 1700.0}),
        encoding="utf-8",
    )
    (run / "blocks.json").write_text(
        json.dumps({"P1": {"block": 3, "x": 10.0, "y": 20.0}, "I1": {"block": 4, "centroid": [30, 40]}}),
        encoding="utf-8",
    )
    (run / "case-profile.json").write_text(
        json.dumps(
            {
                "liquid_cap_m3d": 600,
                "injection_cap_m3d": 600,
                "vrr": {"min": 0.8, "max": 1.3, "window_months": 6},
                "bhp_bounds": [50, 300],
            }
        ),
        encoding="utf-8",
    )
    explain = run / "explain" / "interpretability"
    explain.mkdir(parents=True)
    (explain / "report.json").write_text(json.dumps({"drivers": ["injection"]}), encoding="utf-8")
    (run / "explain" / "summary.md").write_text("# Итог\n", encoding="utf-8")
    (run / "final" / "controls-diff.json").write_text(
        json.dumps([{"month": "2026-01-01", "well": "P1", "field": "value", "incumbent": 90, "final": 100}]),
        encoding="utf-8",
    )
    return run


@pytest.fixture()
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    _build_run(root)
    monkeypatch.setenv(results.RUNS_ROOT_ENV, str(root))
    return root


def test_list_runs_reports_final_runs(runs_root: Path) -> None:
    rows = results.list_runs(runs_root)
    assert [row["run_id"] for row in rows] == ["case-z-test"]
    assert rows[0]["final"] is True
    assert rows[0]["updated_utc"].endswith("Z")


def test_load_run_builds_full_response(runs_root: Path) -> None:
    payload = results.load_run(runs_root, "case-z-test")
    assert payload is not None
    assert payload["official"] is True
    assert payload["commit"] == "deadbeef"
    assert payload["months"] == ["2026-01-01", "2026-02-01"]
    assert payload["profile"]["bhp_bounds"] == [50.0, 300.0]

    field = payload["series"]["field"]
    assert field["oil_t"] == [850.0, 850.0]
    assert field["liquid_m3"] == [3100.0, 2800.0]
    assert field["injection_m3"] == [3720.0, 3360.0]
    assert field["liquid_m3d"] == pytest.approx([100.0, 100.0])
    assert field["injection_m3d"] == pytest.approx([120.0, 120.0])
    assert field["vrr"] == pytest.approx([1.2, 1.2])
    for key in ("oil_t", "liquid_m3", "water_m3", "injection_m3"):
        assert len(field[key]) == len(payload["months"])

    by_well = {well["well"]: well for well in payload["wells"]}
    assert by_well["P1"]["role"] == "producer"
    assert by_well["P1"]["status_last"] == "OPEN"
    assert (by_well["P1"]["x"], by_well["P1"]["y"]) == (10.0, 20.0)
    assert by_well["I1"]["role"] == "injector"
    assert by_well["I1"]["status_last"] == "SHUT"
    assert (by_well["I1"]["x"], by_well["I1"]["y"]) == (30.0, 40.0)
    assert by_well["P1"]["totals"]["oil_t"] == 1700.0

    assert payload["npv"]["final_m"] == 1150.0
    assert payload["npv"]["gain_pct"] == pytest.approx(15.0)
    assert payload["constraints"]["vrr_max"] == 1.3
    assert payload["constraints"]["violations"] == []
    assert payload["controls_diff"][0]["well"] == "P1"
    assert payload["interpretability"]["report"] == {"drivers": ["injection"]}
    assert payload["interpretability"]["summary_md"].startswith("# Итог")
    assert set(payload["sources"]) == {
        "final/audit.json",
        "final/cycle-full/canonical/chdd.csv",
    }


def test_vrr_violation_is_reported(runs_root: Path) -> None:
    profile = runs_root / "case-z-test" / "case-profile.json"
    profile.write_text(
        json.dumps({"liquid_cap_m3d": 50, "vrr": {"min": 0.8, "max": 1.0}}), encoding="utf-8"
    )
    payload = results.load_run(runs_root, "case-z-test")
    assert payload is not None
    rules = {item["rule"] for item in payload["constraints"]["violations"]}
    assert rules == {"liquid_cap", "vrr_max"}


def test_missing_artifacts_yield_nulls(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    (root / "case-z-empty").mkdir(parents=True)
    payload = results.load_run(root, "case-z-empty")
    assert payload is not None
    assert payload["official"] is False
    assert payload["commit"] is None
    assert payload["months"] == []
    assert payload["wells"] == []
    assert payload["series"]["wells"] == {}
    assert payload["npv"]["final_m"] is None
    assert payload["controls_diff"] == []
    assert payload["interpretability"] == {"summary_md": None, "report": None}
    assert payload["sources"] == {}


@pytest.mark.parametrize("run_id", ["..", "a/b", "", "."])
def test_invalid_run_id_is_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(results.ResultsError):
        results.load_run(tmp_path, run_id)


def test_endpoints(runs_root: Path) -> None:
    from timesoil.aios.api import app

    client = TestClient(app)
    listing = client.get("/v1/results")
    assert listing.status_code == 200
    assert [row["run_id"] for row in listing.json()["runs"]] == ["case-z-test"]

    detail = client.get("/v1/results/case-z-test")
    assert detail.status_code == 200
    assert detail.json()["months"] == ["2026-01-01", "2026-02-01"]

    assert client.get("/v1/results/case-z-missing").status_code == 404
    assert client.get("/v1/results/..").status_code in {404, 422}
