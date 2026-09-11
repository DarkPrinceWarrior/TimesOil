"""Interpretability layer: decomposition against the real calculator, margins, evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from timesoil.aios.economics import CHDDEconomicsAdapter
from timesoil.aios.explain import (
    NPV_TERMS, candidate_lineage, check_evidence, constraint_margins, forecast_vs_physics,
    npv_decomposition, search_trace, selection_summary, sensitivity, write_report,
)

WELLS = ("11", "12", "13")
MONTHS = ("2020-01-01", "2020-02-01", "2020-03-01")


def _row(date: str, well: str, *, liquid: float, oil: float, injection: float,
         bhp: float, cumulative: dict[str, float]) -> dict[str, object]:
    cumulative["WLPT"] += liquid
    cumulative["WOMT"] += oil
    cumulative["WWIT"] += injection
    producing = liquid > 0
    return {"DATA": date, "well": well,
            "WLPT": cumulative["WLPT"], "WLPR": liquid / 30 if producing else 0.0,
            "WOMT": cumulative["WOMT"], "WOMR": oil / 30 if producing else 0.0,
            "WWIR": injection / 30 if injection > 0 else 0.0, "WWIT": cumulative["WWIT"],
            "THP": 120.0 if producing or injection > 0 else 0.0,
            "BHP": bhp, "WEFF": 1.0 if producing or injection > 0 else 0.0,
            "WLPT_Diff": liquid, "WOMT_Diff": oil, "WWIT_Diff": injection}


def _trajectory(scale: float, *, shut_third_month: bool = False) -> list[dict[str, object]]:
    """Three wells: two producers and one injector, three months."""
    rows: list[dict[str, object]] = []
    for well in WELLS:
        cumulative = {"WLPT": 0.0, "WOMT": 0.0, "WWIT": 0.0}
        for index, date in enumerate(MONTHS):
            injector = well == "13"
            idle = shut_third_month and well == "12" and index == 2
            liquid = 0.0 if injector or idle else 3000.0 * scale * (1 + 0.1 * index)
            oil = 0.0 if injector or idle else liquid * 0.4
            injection = 0.0 if not injector else 4000.0 * scale
            bhp = 0.0 if idle else (250.0 if injector else 90.0)
            rows.append(_row(date, well, liquid=liquid, oil=oil, injection=injection,
                             bhp=bhp, cumulative=cumulative))
    return rows


@pytest.fixture(scope="module")
def official_norms() -> dict[str, object]:
    return CHDDEconomicsAdapter().normative_profile()


def test_decomposition_reproduces_the_official_total(tmp_path: Path, official_norms) -> None:
    """Per well-month terms must sum to the calculator's own totalChddM."""
    base, candidate = _trajectory(1.0), _trajectory(1.2)
    adapter = CHDDEconomicsAdapter()
    reference = {name: adapter.calculate(rows, start_year=2020, output_dir=tmp_path / name).total_chdd_m
                 for name, rows in (("base", base), ("candidate", candidate))}

    result = npv_decomposition(base, candidate, official_norms["assumptions"],
                               start_year=2020, pumps=official_norms["pumps"],
                               block_of_well={"11": "A", "12": "A", "13": "B"})

    assert result["base_chdd_m"] == pytest.approx(reference["base"], rel=1e-9)
    assert result["candidate_chdd_m"] == pytest.approx(reference["candidate"], rel=1e-9)
    assert result["delta_chdd_m"] == pytest.approx(reference["candidate"] - reference["base"], rel=1e-9)
    assert sum(result["by_term"].values()) == pytest.approx(result["delta_chdd_m"], abs=1e-9)
    assert sum(row["delta_chdd_m"] for row in result["by_well"]) == pytest.approx(
        result["delta_chdd_m"], abs=1e-9)
    assert sum(row["delta_chdd_m"] for row in result["by_year"]) == pytest.approx(
        result["delta_chdd_m"], abs=1e-9)
    assert sum(row["delta_chdd_m"] for row in result["by_block"]) == pytest.approx(
        result["delta_chdd_m"], abs=1e-9)
    assert {row["block"] for row in result["by_block"]} == {"A", "B"}
    assert set(result["by_term"]) == set(NPV_TERMS)


def test_decomposition_terms_match_hand_computed_prices(official_norms) -> None:
    """Revenue and the three volume OPEX items are exact prices times exact volumes."""
    base, candidate = _trajectory(1.0), _trajectory(1.2)
    econ = official_norms["assumptions"]
    result = npv_decomposition(base, candidate, econ, start_year=2020, pumps=official_norms["pumps"])
    oil = sum(row["WOMT_Diff"] for row in candidate) - sum(row["WOMT_Diff"] for row in base)
    liquid = sum(row["WLPT_Diff"] for row in candidate) - sum(row["WLPT_Diff"] for row in base)
    injection = sum(row["WWIT_Diff"] for row in candidate) - sum(row["WWIT_Diff"] for row in base)

    assert result["by_term"]["revenue"] == pytest.approx(oil * econ["oilPriceRubT"] / 1e6)
    assert result["by_term"]["deductions"] == pytest.approx(-oil * econ["deductionsRubT"] / 1e6)
    assert result["by_term"]["oil_opex"] == pytest.approx(-oil * econ["oilOpexRubT"] / 1e6)
    assert result["by_term"]["liquid_opex"] == pytest.approx(-liquid * econ["liquidOpexRubT"] / 1e6)
    assert result["by_term"]["injection_opex"] == pytest.approx(
        -injection * econ["injectionOpexRubM3"] / 1e6)
    # One year only: the discount factor is 1, so discounting moves nothing.
    assert result["by_term"]["discount"] == pytest.approx(0.0, abs=1e-12)
    # The injector earns no revenue but carries every cubic metre of injection OPEX.
    injector = next(row for row in result["by_well"] if row["well"] == "13")
    assert injector["terms"]["revenue"] == pytest.approx(0.0)
    assert injector["terms"]["injection_opex"] == pytest.approx(
        -injection * econ["injectionOpexRubM3"] / 1e6)


def test_stopping_a_well_shows_up_as_a_start_stop_event(official_norms) -> None:
    result = npv_decomposition(_trajectory(1.0), _trajectory(1.0, shut_third_month=True),
                               official_norms["assumptions"], start_year=2020,
                               pumps=official_norms["pumps"])
    assert result["by_term"]["start_stop"] == pytest.approx(
        -official_norms["assumptions"]["stopStartCostM"])
    assert result["by_term"]["fund_opex"] == pytest.approx(
        official_norms["assumptions"]["fundAnnualRubWell"] / 12 / 1e6)
    stopped = next(row for row in result["by_well"] if row["well"] == "12")
    assert stopped["terms"]["start_stop"] == pytest.approx(
        -official_norms["assumptions"]["stopStartCostM"])


PROFILE = {
    "liquid_cap_m3d": 600.0, "injection_cap_m3d": 600.0, "bhp_bounds": [50.66, 303.98],
    "vrr": {"min": 0.85, "max": 1.15, "window_months": 3},
    "pressure": {"field_min_bar": 109.43},
    "repairs": [{"well": "12", "start": "2020-03-01", "end": "2020-03-01"}],
}


def test_constraint_margins_reports_caps_vrr_bhp_pressure_and_repairs() -> None:
    rows = _trajectory(1.0, shut_third_month=True)
    summary = [{"DATE": month, "FPR": pressure}
               for month, pressure in zip(MONTHS, (118.0, 112.0, 105.0))]
    result = constraint_margins(rows, PROFILE, summary_rows=summary,
                                densities={"oil": 0.85, "water": 1.0})

    assert [row["month"] for row in result["months"]] == ["2020-01", "2020-02", "2020-03"]
    january = result["months"][0]
    # 2 producers x 3000 t: 1200 t oil / 0.85 + 4800 m3 water, over 31 days.
    assert january["liquid_m3d"] == pytest.approx((2400 / 0.85 + 3600) / 31)
    assert january["injection_m3d"] == pytest.approx(4000 / 31)
    assert january["liquid_margin_m3d"] == pytest.approx(600 - january["liquid_m3d"])
    assert january["vrr3"] is None and result["months"][2]["vrr3"] is not None
    assert result["worst"]["injection"]["month"] in {"2020-01", "2020-02", "2020-03"}

    # Producers are measured against the floor, the injector against the ceiling.
    margins = {row["well"]: row for row in result["bhp"]["per_well"]}
    assert margins["11"]["margin_bar"] == pytest.approx(90.0 - 50.66)
    assert margins["13"]["margin_bar"] == pytest.approx(303.98 - 250.0)
    # The repair month is excluded from well 12's BHP margin, and the repair is idle.
    assert margins["12"]["month"] != "2020-03"
    assert result["repairs"][0] == {"well": "12", "start": "2020-03-01", "end": "2020-03-01",
                                    "months_observed": 1, "max_observed_rate": 0.0, "idle": True}
    assert result["pressure"]["min_fpr_bar"] == 105.0
    assert result["pressure"]["margin_bar"] == pytest.approx(105.0 - 109.43)


def test_constraint_margins_without_densities_leaves_liquid_and_vrr_null() -> None:
    result = constraint_margins(_trajectory(1.0), PROFILE)
    assert all(row["liquid_m3d"] is None and row["vrr3"] is None for row in result["months"])
    assert all(row["injection_m3d"] is not None for row in result["months"])
    assert any("densities" in note for note in result["notes"])


def test_forecast_vs_physics_ranks_the_wells_that_ate_the_shortfall(official_norms) -> None:
    import numpy as np

    physical = _trajectory(1.0)
    targets = ("WOMT_Diff", "WLPT_Diff", "WWIT_Diff")
    truth = np.array([[[float(next(r for r in physical if r["DATA"] == month and r["well"] == well)[field])
                        for field in targets] for well in WELLS] for month in MONTHS])
    prediction = truth.copy()
    prediction[:, 0, 0] *= 1.5  # well 11 oil overstated by 50%
    forecast = {"prediction": prediction, "timestamps": np.array(MONTHS),
                "well_ids": np.array(WELLS), "targets": np.array(targets)}

    result = forecast_vs_physics(forecast, physical, norms=official_norms["assumptions"])

    assert result["wape_by_target"]["WLPT_Diff"] == pytest.approx(0.0)
    assert result["wape_by_target"]["WOMT_Diff"] > 0
    assert result["money_shortfall_by_well"][0]["well"] == "11"
    assert result["money_shortfall_by_well"][0]["overstatement_m"] > 0
    assert result["covered_cells"] == truth.size


TRACE = [
    {"id": 0, "generation": 0, "x": [0.1, 0.5], "npv": 100.0, "feasible": True, "sigma": 0.25},
    {"id": 1, "generation": 0, "x": [0.5, 0.5], "npv": 140.0, "feasible": True, "sigma": 0.25},
    {"id": 2, "generation": 0, "x": [0.5, 0.9], "npv": 142.0, "feasible": False,
     "rejection": "injection cap"},
    {"id": 3, "generation": 1, "x": [0.9, 0.5], "npv": 180.0, "feasible": True, "sigma": 0.2,
     "injected": True, "parent_id": 1},
]


def test_search_trace_and_sensitivity_use_only_evaluated_points() -> None:
    trace = search_trace(TRACE)
    assert [row["generation"] for row in trace["generations"]] == [0, 1]
    assert trace["generations"][0]["feasible_fraction"] == pytest.approx(2 / 3)
    assert trace["generations"][0]["best_feasible_npv"] == 140.0
    assert trace["best"]["id"] == 3

    result = sensitivity(TRACE, param_names=["producer_scale", "phi"])
    axis = {row["parameter"]: row for row in result["parameters"]}
    # npv rises 100 -> 140 -> 180 as x0 goes 0.1 -> 0.5 -> 0.9 at fixed x1: slope 100/unit.
    assert axis["producer_scale"]["median_slope"] == pytest.approx(100.0)
    assert axis["producer_scale"]["pairs"] == 3
    assert axis["phi"]["median_slope"] is None  # no feasible pair varies x1
    assert axis["producer_scale"]["at_bounds"] is False
    assert result["winner_on_box_boundary"] == []


def test_candidate_lineage_keeps_parents_and_rejections() -> None:
    table = candidate_lineage(TRACE)
    assert [row["id"] for row in table["candidates"]] == [0, 1, 2, 3]
    assert table["candidates"][3]["parent_id"] == 1
    assert table["rejection_counts"] == {"injection cap": 1}


def test_selection_summary_reports_runners_up_and_control_diff() -> None:
    seal = {"selected_id": 3, "selection_metric": "forecast_chdd_m", "forecast_chdd_m": 180.0,
            "baseline_forecast_chdd_m": 100.0, "forecast_uplift_percent": 80.0,
            "search_opm_calls": 0, "maximum_final_opm_calls": 1,
            "reselection_after_opm_allowed": False}
    candidates = [{"id": 0, "forecast_chdd_m": 100.0, "forecast_eligible": True},
                  {"id": 1, "forecast_chdd_m": 140.0, "forecast_eligible": True},
                  {"id": 2, "forecast_chdd_m": 142.0, "forecast_eligible": False},
                  {"id": 3, "forecast_chdd_m": 180.0, "forecast_eligible": True}]
    baseline_controls = [{"month": "2020-01-01", "well": "11", "status": "OPEN",
                          "target": "LRAT", "value": 100.0}]
    selected_controls = [{"month": "2020-01-01", "well": "11", "status": "OPEN",
                          "target": "LRAT", "value": 120.0}]

    result = selection_summary(seal, candidates, selected_controls=selected_controls,
                               baseline_controls=baseline_controls)
    assert result["delta_vs_baseline_m"] == pytest.approx(80.0)
    assert [row["id"] for row in result["runners_up"]] == [1, 0]
    assert result["runners_up"][0]["delta_vs_selected_m"] == pytest.approx(-40.0)
    assert result["eligible_count"] == 3
    assert result["control_diff"] == [{"month": "2020-01-01", "well": "11",
                                       "changed": {"value": [100.0, 120.0]}}]


def test_evidence_check_confirms_differs_and_missing_citations() -> None:
    context = {"candidates": [{"forecast_chdd_m": 12505.33}, {"forecast_chdd_m": 11280.37}],
               "constraints": {"count": 0}}
    decisions = [{"role": "planner", "approved": True, "summary": "план", "evidence": [
        "candidates[0].forecast_chdd_m: 12505.33",
        "candidates[1].forecast_chdd_m: 11280.37 (baseline)",
        "candidates[0].forecast_chdd_m: 13000.0",
        "candidates[9].forecast_chdd_m: 1.0",
        "constraints.operating_constraints: [] (нет запретов)",
    ]}]
    result = check_evidence(decisions, context)
    statuses = [check["status"] for check in result["decisions"][0]["checks"]]
    assert statuses == ["confirmed", "confirmed", "differs", "not_found", "not_numeric"]
    assert result["totals"] == {"confirmed": 2, "differs": 1, "not_found": 1, "not_numeric": 1}


def test_write_report_records_a_failing_section_instead_of_raising(tmp_path: Path) -> None:
    def broken() -> dict[str, object]:
        raise ValueError("no candidate ledger in this directory")

    source = tmp_path / "input.json"
    source.write_text("{}\n", encoding="utf-8")
    report = write_report(tmp_path, {"selection": broken, "search_trace": lambda: search_trace(TRACE)},
                          inputs={"ledger": source})

    assert "selection" in report["errors"] and report["written"] == ["search_trace"]
    root = tmp_path / "interpretability"
    header = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert header["official"] is False
    assert header["errors"]["selection"].endswith("no candidate ledger in this directory")
    assert header["inputs"]["ledger"]["sha256"]
    assert not (root / "selection.json").exists()
    body = (root / "summary.md").read_text(encoding="utf-8")
    assert "Not computable: ValueError: no candidate ledger" in body
    # Deterministic: the same input rewrites byte-identical files.
    before = (root / "search_trace.json").read_bytes()
    write_report(tmp_path, {"selection": broken, "search_trace": lambda: search_trace(TRACE)},
                 inputs={"ledger": source})
    assert (root / "search_trace.json").read_bytes() == before


def test_management_period_maps_opm_endpoints_to_elapsed_months() -> None:
    """The canonical CSV is a report-endpoint grid; the period must come from the run."""
    from datetime import date

    from explain_run import management_period

    audit = {"start_inclusive": "2007-01-01", "end_exclusive": "2025-09-01"}
    official = {"startDate": "2007-01-01", "maxDate": "2025-08-01"}
    assert management_period(official, audit) == (date(2007, 1, 1), date(2025, 9, 1))
    assert management_period(official, None) == (date(2007, 1, 1), date(2025, 9, 1))
    assert management_period({"startDate": "2007-01-01", "maxDate": "2024-12-01"}, None) == (
        date(2007, 1, 1), date(2025, 1, 1))
    assert management_period(None, None) is None
