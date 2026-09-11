"""Case intake checks: the training archive and a synthetic case with a truncated schedule.

No OPM runs here. The point is that the layout assumptions, the incumbent derivation and
the request contract all fail loudly on a differing archive instead of silently drifting.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import zipfile

import pytest

from timesoil.aios.case_profile import load_case_profile
from timesoil.aios.workflow import CycleRequest, _source_control_inventory, _validate_source_well_scope

from intake_case_z import (
    CaseArchive,
    IntakeError,
    Schedule,
    build_request,
    inspect_case,
    month_range,
    read_layout,
)

TRAINING_ARCHIVE = (Path(__file__).resolve().parents[1]
                    / "docs/hackathon/models/model_z_final_opm/Model_Z_final_OPM.zip")
CUT = date(2006, 12, 31)
START = date(2007, 1, 1)
END = date(2025, 9, 1)

# The official case (ORGANIZER_CHAT_CLARIFICATIONS_20260910, last section): the eight
# production and eight injection repairs, named by the plain well numbers of the deck.
CASE_REPAIRS = [
    ("90", "2007-04-01", "2007-06-01"), ("54", "2008-03-01", "2008-08-01"),
    ("75", "2009-05-01", "2009-08-01"), ("3", "2020-04-01", "2020-08-01"),
    ("30", "2021-05-01", "2021-09-01"), ("47", "2022-06-01", "2022-10-01"),
    ("54", "2023-09-01", "2024-01-01"), ("55", "2024-11-01", "2025-03-01"),
    ("27", "2007-12-01", "2008-04-01"), ("59", "2008-10-01", "2009-03-01"),
    ("6", "2009-12-01", "2010-04-01"), ("26", "2020-11-01", "2021-03-01"),
    ("24", "2021-12-01", "2022-04-01"), ("13", "2023-02-01", "2023-06-01"),
    ("42", "2024-04-01", "2024-08-01"), ("7", "2025-05-01", "2025-09-01"),
]


def write_profile(path: Path, repairs) -> Path:
    path.write_text(json.dumps({
        "liquid_cap_m3d": 600, "injection_cap_m3d": 600, "bhp_bounds": [50.66, 303.98],
        "vrr": {"min": 0.85, "max": 1.15, "window_months": 3,
                "denominator": "liquid_reservoir", "lower_bound_status": "diagnostic"},
        "water_balance": {"deficit_m3": 0, "carryover": False},
        "pressure": {"field_min_bar": None, "block_min_bar": None, "regions": "FIP_C1"},
        "repairs": [{"well": well, "start": start, "end": end} for well, start, end in repairs],
        "selection_margins": {"eps_liquid": 0.03, "eps_injection": 0.03, "phi": 0.95},
    }, indent=2))
    return path


pytestmark = pytest.mark.skipif(
    not TRAINING_ARCHIVE.is_file(), reason="the training archive is not checked in")


# ------------------------------------------------------------------ the training archive


@pytest.fixture(scope="module")
def training_inspection() -> dict:
    return inspect_case(CaseArchive(TRAINING_ARCHIVE), CUT)


@pytest.fixture(scope="module")
def training_intake(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("intake")
    profile = load_case_profile(write_profile(root / "case_constraints.json", CASE_REPAIRS))
    return build_request(
        CaseArchive(TRAINING_ARCHIVE), profile, cut=CUT, start=START, end=END,
        scenario_id="case-z-incumbent", parsing_strictness="low",
        output=root / "out", extend_schedule=False)


def test_inspect_reports_the_deck_layout(training_inspection):
    report = training_inspection
    assert report["deck"]["member"] == "Model_Z/Model_Z.data"
    assert report["deck"]["unit_system"] == "METRIC"
    assert report["schedule"]["member"] == "Model_Z/Model_Z_sch.inc"
    assert report["schedule"]["monthly_grid"] is True
    assert report["schedule"]["first_date"] == "1994-11-01"
    assert report["schedule"]["last_date"] == "2025-09-01"
    # The training archive carries the organizers' future schedule, so the cut does not
    # close it. A real case is expected to stop at the cut; both are reported, not assumed.
    assert report["cut"]["last_date_not_after_cut"] is False
    assert report["keywords_after_cut"] == ["COMPDAT", "DATES", "WCONINJE", "WCONPROD", "WPIMULT"]
    assert report["regions"] == {"FIP_C1": 3, "FIP_ZONE": 6}
    assert report["archive"]["sha256"] == (
        "4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa")


def test_inspect_reports_the_cut_state(training_inspection):
    report = training_inspection
    assert report["wells"]["welspecs"] == 103
    assert report["wells"]["controlled"] == 103
    assert report["wells"]["welspecs_without_control"] == []
    # Measured from the archive, not taken from the case document: it claims 57 producers,
    # the deck's last WCON records at 2006-12 name 58. The measurement is what is reported.
    assert report["cut"]["roles"] == {"producer": 58, "injector": 23}
    assert report["cut"]["active_roles"] == {"producer": 58, "injector": 23}
    assert report["cut"]["wells_with_control"] == 81
    assert len(report["cut"]["wells_without_control"]) == 22
    assert report["cut"]["unreadable_regimes"] == []
    assert report["cut"]["field_liquid_target_m3d"] == pytest.approx(1505.0)
    assert report["cut"]["field_injection_target_m3d"] == pytest.approx(1495.0)
    regime = report["cut"]["regimes"]["1"]
    assert regime["role"] == "producer" and regime["target"] == "LRAT"
    assert regime["status"] == "OPEN" and regime["bhp_limit"] == pytest.approx(50.0)


def test_build_request_covers_the_management_period(training_intake):
    incumbent = training_intake["incumbent"]
    assert incumbent["months"] == 224
    assert incumbent["first_month"] == "2007-01-01"
    assert incumbent["last_month"] == "2025-08-01"
    assert incumbent["wells"] == 103
    request = json.loads(Path(training_intake["request"]["path"]).read_text())
    assert len(request["controls"]) == 224 * 103
    assert request["horizon_months"] == 224
    assert request["deck"] == "Model_Z/Model_Z.data"
    assert request["schedule_relative_path"] == "Model_Z/Model_Z_sch.inc"
    assert request["context"]["facts"]["is_baseline"] is True


def test_build_request_scales_field_targets_under_the_caps(training_intake):
    incumbent = training_intake["incumbent"]
    assert incumbent["field_targets_before_scaling"]["max_liquid_m3d"] > 600
    assert incumbent["field_targets_before_scaling"]["max_injection_m3d"] > 600
    assert incumbent["field_targets_after_scaling"]["max_liquid_m3d"] <= 600 + 1e-9
    assert incumbent["field_targets_after_scaling"]["max_injection_m3d"] <= 600 + 1e-9
    assert 0 < incumbent["scale_factors"]["producer"] < 1
    assert 0 < incumbent["scale_factors"]["injector"] < 1

    request = json.loads(Path(training_intake["request"]["path"]).read_text())
    liquid: dict[str, float] = {}
    injection: dict[str, float] = {}
    for action in request["controls"]:
        if action["status"] != "OPEN":
            continue
        bucket = injection if action["role"] == "injector" else liquid
        bucket[action["month"]] = bucket.get(action["month"], 0.0) + action["value"]
        if action["target"] == "LRAT":
            assert action["value"] <= 500.0
    assert max(liquid.values()) <= 600 + 1e-9
    assert max(injection.values()) <= 600 + 1e-9


def test_build_request_applies_the_repair_calendar(training_intake):
    request = json.loads(Path(training_intake["request"]["path"]).read_text())
    by_key = {(action["month"], action["well"]): action for action in request["controls"]}
    # Producer 90 stops 2007-04-01..2007-06-01: the half-open interval shuts two months.
    assert by_key[("2007-04-01", "90")]["status"] == "SHUT"
    assert by_key[("2007-05-01", "90")]["status"] == "SHUT"
    assert by_key[("2007-04-01", "90")]["value"] == 0.0
    assert by_key[("2007-03-01", "90")]["status"] == "OPEN"
    assert by_key[("2007-06-01", "90")]["status"] == "OPEN"
    # Injector 27 stops 2007-12-01..2008-04-01.
    assert [by_key[(f"2008-{month:02d}-01", "27")]["status"] for month in (1, 2, 3, 4)] == [
        "SHUT", "SHUT", "SHUT", "OPEN"]
    assert by_key[("2007-11-01", "27")]["status"] == "OPEN"
    assert set(training_intake["incumbent"]["repair_shut_wells"]) <= {
        well for well, _, _ in CASE_REPAIRS}


def test_build_request_keeps_uncompleted_wells_shut(training_intake):
    request = json.loads(Path(training_intake["request"]["path"]).read_text())
    late = training_intake["incumbent"]["wells_shut_before_first_source_control"]
    assert "105" in late  # first source control only in 2013
    by_key = {(action["month"], action["well"]): action for action in request["controls"]}
    assert by_key[("2007-01-01", "105")]["status"] == "SHUT"
    assert by_key[("2007-01-01", "105")]["value"] == 0.0
    assert by_key[("2013-05-01", "105")]["status"] == "OPEN"


def test_built_request_passes_the_production_contract(training_intake):
    path = Path(training_intake["request"]["path"])
    request = json.loads(path.read_text())
    checked = CycleRequest.from_mapping(request)
    assert checked.controls_sha256 == training_intake["request"]["controls_sha256"]
    schedule_text = CaseArchive(TRAINING_ARCHIVE).text(request["schedule_relative_path"])
    months = sorted({action.month for action in checked.controls})
    inventory = _source_control_inventory(schedule_text, months)
    _validate_source_well_scope(checked.controls, inventory, allow_conversion_to_injection=True)


def test_manifest_records_every_check(training_intake):
    names = [item["check"] for item in training_intake["checks"]]
    assert names == [
        "single_data_deck_and_schedule_include", "unit_system_is_metric", "management_period",
        "cut_month_present", "schedule_covers_management_period", "repair_calendar_applied",
        "field_targets_within_case_caps", "no_production_before_first_source_control",
        "request_validates"]
    assert len(training_intake["archive"]["sha256"]) == 64
    assert len(training_intake["case_profile"]["sha256"]) == 64
    assert training_intake["case_profile"]["repairs"] == 16


# ------------------------------------------------- a synthetic case that stops at the cut


def make_case(root: Path) -> Path:
    deck = """RUNSPEC
METRIC
DIMENS
 1 1 1 /
REGIONS
INCLUDE
 'regs.inc' /
SCHEDULE
INCLUDE
 'sch.inc' /
END
"""
    regs = "FIPNUM\n 2*1 /\n"
    schedule = """WELSPECS
 'P1' 'G' 1 1 1* 'OIL' /
 'P2' 'G' 1 1 1* 'OIL' /
 'I1' 'G' 1 1 1* 'WATER' /
/

DATES
 01 OCT 2006 /
/

WCONPROD
 'P1' 'OPEN' 'LRAT' 1* 1* 1* 100.0 1* 60 1* 1* /
 'P2' 'OPEN' 'LRAT' 1* 1* 1* 200.0 1* 40 1* 1* /
/

WCONINJE
 'I1' 'WATER' 'OPEN' 'RATE' 400.0 1* 280 1* 1* /
/

DATES
 01 NOV 2006 /
/

DATES
 01 DEC 2006 /
/
"""
    path = root / "case.zip"
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("case/case.DATA", deck)
        archive.writestr("case/regs.inc", regs)
        archive.writestr("case/sch.inc", schedule)
    return path


def test_synthetic_layout_is_read(tmp_path):
    archive = CaseArchive(make_case(tmp_path))
    layout = read_layout(archive)
    assert layout.deck_member == "case/case.DATA"
    assert layout.schedule_member == "case/sch.inc"
    report = inspect_case(archive, CUT)
    assert report["cut"]["roles"] == {"producer": 2, "injector": 1}
    assert report["cut"]["last_date_not_after_cut"] is True
    assert report["regions"] == {"FIPNUM": 1}


def test_schedule_stopping_at_the_cut_is_refused_then_extended(tmp_path):
    source = make_case(tmp_path)
    profile = load_case_profile(write_profile(
        tmp_path / "profile.json", [("P1", "2007-03-01", "2007-05-01")]))
    start, end = date(2007, 1, 1), date(2007, 7, 1)

    with pytest.raises(IntakeError, match="--extend-schedule"):
        build_request(CaseArchive(source), profile, cut=CUT, start=start, end=end,
                      scenario_id="case-z-incumbent", parsing_strictness="low",
                      output=tmp_path / "refused", extend_schedule=False)

    manifest = build_request(
        CaseArchive(source), profile, cut=CUT, start=start, end=end,
        scenario_id="case-z-incumbent", parsing_strictness="low",
        output=tmp_path / "out", extend_schedule=True)
    extended = Path(manifest["request"]["source"])
    assert extended.name == "case-extended.zip"
    assert manifest["request"]["source_sha256"] != manifest["archive"]["sha256"]
    schedule = Schedule(CaseArchive(extended).text("case/sch.inc"))
    assert schedule.months[-1] == end
    assert set(month_range(start, end)) <= set(schedule.months)
    # Only report dates were appended; every control record survives untouched.
    assert len(schedule.templates) == 3

    request = json.loads(Path(manifest["request"]["path"]).read_text())
    checked = CycleRequest.from_mapping(request)
    inventory = _source_control_inventory(schedule.text, sorted(month_range(start, end)))
    _validate_source_well_scope(checked.controls, inventory, allow_conversion_to_injection=True)
    by_key = {(action["month"], action["well"]): action for action in request["controls"]}
    # Field targets are under the caps already, so no scaling; BHP is tightened both ways.
    assert manifest["incumbent"]["scale_factors"] == {"producer": 1.0, "injector": 1.0}
    assert by_key[("2007-01-01", "P1")]["value"] == pytest.approx(100.0)
    assert by_key[("2007-01-01", "P1")]["bhp_limit"] == pytest.approx(60.0)
    assert by_key[("2007-01-01", "P2")]["bhp_limit"] == pytest.approx(50.66)
    assert by_key[("2007-01-01", "I1")]["bhp_limit"] == pytest.approx(280.0)
    assert by_key[("2007-03-01", "P1")]["status"] == "SHUT"
    assert by_key[("2007-05-01", "P1")]["status"] == "OPEN"


def test_unknown_repair_well_is_refused(tmp_path):
    profile = load_case_profile(write_profile(
        tmp_path / "profile.json", [("NOPE", "2007-03-01", "2007-05-01")]))
    with pytest.raises(IntakeError, match="unknown well"):
        build_request(CaseArchive(make_case(tmp_path)), profile, cut=CUT,
                      start=date(2007, 1, 1), end=date(2007, 7, 1),
                      scenario_id="case-z-incumbent", parsing_strictness="low",
                      output=tmp_path / "out", extend_schedule=True)


def test_month_range_is_half_open():
    assert month_range(date(2007, 1, 1), date(2007, 4, 1)) == (
        date(2007, 1, 1), date(2007, 2, 1), date(2007, 3, 1))
    assert len(month_range(START, END)) == 224
    with pytest.raises(IntakeError):
        month_range(date(2007, 1, 1), date(2007, 1, 1))
    with pytest.raises(IntakeError):
        month_range(date(2007, 1, 15), date(2008, 1, 1))
