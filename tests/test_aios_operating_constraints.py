from dataclasses import replace
from datetime import date

import pytest

from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
from timesoil.aios.operating_constraints import (
    check_controls, check_observed, check_summary, evaluate_summary, failures, parse_constraints,
)


def test_monthly_group_limits_outages_and_watercut():
    month = date(2014, 1, 1)
    raw = [{"start": "2014-01-01", "end": "2014-01-01", "wells": ["P", "I"],
            "limits": {"max_oil_m3d": 10, "max_liquid_m3d": 100, "min_injection_m3d": 20,
                       "max_injection_m3d": 40, "max_watercut": .9, "min_bhp_bar": 50, "max_bhp_bar": 300}}]
    rules = parse_constraints(raw, wells=["P", "I"], start=month, end=month)
    p = ControlAction(month, "P", WellRole.PRODUCER, WellStatus.OPEN, ControlTarget.LIQUID_RATE, 100)
    i = ControlAction(month, "I", WellRole.INJECTOR, WellStatus.OPEN, ControlTarget.WATER_INJECTION_RATE, 30)
    check_controls(rules, (p, i))
    values = {"P": dict(WOPR=10, WLPR=100, WWIR=0, WBHP=70),
              "I": dict(WOPR=0, WLPR=0, WWIR=30, WBHP=280)}
    check_observed(rules, month, values)
    for key, cap in (("max_oil_m3d", 9), ("max_liquid_m3d", 99), ("max_injection_m3d", 29),
                     ("min_injection_m3d", 31), ("max_watercut", .89), ("min_bhp_bar", 71), ("max_bhp_bar", 279)):
        with pytest.raises(ValueError, match="observed"):
            check_observed((replace(rules[0], limits=((key, cap),)),), month, values)
    unavailable = (replace(rules[0], wells=("P",), limits=(), unavailable=True),)
    with pytest.raises(ValueError, match="unavailable"):
        check_controls(unavailable, (p, i))
    with pytest.raises(ValueError, match="unavailable"):
        check_observed(unavailable, month, values)
    check_controls(unavailable, (replace(p, status=WellStatus.SHUT, value=0), i))
    check_observed(unavailable, month, {**values, "P": dict(WOPR=0, WLPR=0, WWIR=0, WBHP=0)})
    for change in ({"limits": {"drilling_cost": 5}}, {"wells": ["unknown"]}, {"limits": {"max_watercut": 90}}):
        with pytest.raises(ValueError):
            parse_constraints([{**raw[0], **change}], wells=["P", "I"], start=month, end=month)


def test_monthly_water_balance_uses_reservoir_deltas_and_explicit_groups(tmp_path):
    import csv
    from timesoil.aios.opm_chdd import REQUIRED_VECTORS, OpmChddError, _read_summary

    month = date(2014, 1, 1)
    wells = ["P", "I", "P2", "I2"]
    optional = ("WVPT", "WVIT")
    vectors = (*REQUIRED_VECTORS, *optional)
    report = tmp_path / "summary.csv"
    # Reservoir production/injection: first pair 100/150, second 100/50.
    # Whole field ratio is 1.0; first pair violates its 1.15 ceiling.
    increments = {"P": dict(WVPT=100, WWPT=40), "I": dict(WVIT=150, WWIT=50),
                  "P2": dict(WVPT=100, WWPT=40), "I2": dict(WVIT=50, WWIT=50)}
    def write_report(*, keep=optional, dates=("2014-01-01", "2014-02-01"), reverse=False, omit=None):
        columns = [(w, v) for w in wells for v in (*REQUIRED_VECTORS, *keep) if (w, v) != omit]
        with report.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["DATE", *(f"{v}:{w}" for w, v in columns)])
            for stamp in dates:
                end = stamp == "2014-02-01"
                # Historical cumulative totals hide monthly overinjection unless differenced.
                writer.writerow([stamp, *(10000 + (-1 if reverse else 1) * increments[w].get("WWPT" if v == "WLPT" else v, 0) * end
                                          if v in ("WVPT", "WVIT", "WWPT", "WWIT", "WLPT") else 0
                                          for w, v in columns)])
    def rules(group=wells, **limits):
        return parse_constraints([dict(start=str(month), end=str(month), wells=group, limits=limits)],
                                 wells=wells, start=month, end=month)
    def check(rule):
        check_summary(rule, report, deck_dir=tmp_path, months=(month,), unit_system="METRIC")
    write_report()
    check(rules(min_monthly_voidage_replacement=.85, max_monthly_voidage_replacement=1.15))
    with pytest.raises(ValueError, match="observed max_monthly_voidage"):
        check(rules(["P", "I"], max_monthly_voidage_replacement=1.15))
    check(rules(max_monthly_water_deficit_m3=20))
    with pytest.raises(ValueError, match="observed max_monthly_water_deficit"):
        check(rules(max_monthly_water_deficit_m3=19))
    with pytest.raises(ValueError, match="minimum operating limit"):
        rules(min_monthly_voidage_replacement=1.2, max_monthly_voidage_replacement=1.1)
    write_report(keep=())
    check(rules(max_monthly_water_deficit_m3=20))
    with pytest.raises(ValueError, match="misses WV"):
        check(rules(max_monthly_voidage_replacement=1.15))
    write_report(dates=("2014-01-02", "2014-02-01"))
    with pytest.raises(ValueError, match="start-of-month"):
        check(rules(max_monthly_water_deficit_m3=20))
    write_report(reverse=True)
    with pytest.raises(ValueError, match="cumulative volume decreased"):
        check(rules(max_monthly_voidage_replacement=1.15))
    write_report(omit=("P", "WVPT"))
    with pytest.raises(OpmChddError, match="complete well coverage"):
        _read_summary(report)
    # No case constraint means no invented water restriction or requirement for new vectors.
    check_summary((), tmp_path / "absent", deck_dir=tmp_path, months=(month,), unit_system="METRIC")


def test_voidage_zero_production_and_invalid_observations():
    month = date(2014, 1, 1)
    rules = parse_constraints([dict(start=str(month), end=str(month), wells=["I"],
                                   limits=dict(min_monthly_voidage_replacement=.85, max_monthly_voidage_replacement=1.15))],
                              wells=["I"], start=month, end=month)
    idle = dict(WOPR=0, WLPR=0, WWIR=0, WBHP=0, WVPT_DELTA=0, WVIT_DELTA=0)
    check_observed(rules, month, {"I": idle})
    for delta in (1, -1, float("nan")):
        with pytest.raises(ValueError):
            check_observed(rules, month, {"I": dict(idle, WVIT_DELTA=delta)})


MONTHS = (date(2014, 1, 1), date(2014, 2, 1), date(2014, 3, 1))


def _field_summary(path, *, producer_bhp=(60., 60., 60., 40.)):
    """Four monthly records; monthly averages and window sums disagree with the endpoints."""
    import csv
    from timesoil.aios.opm_chdd import REQUIRED_VECTORS

    wells = ("P", "I")
    optional = ("WVPT", "WVIT")
    stamps = ("2014-01-01", "2014-02-01", "2014-03-01", "2014-04-01")
    # Liquid: 3100/2800/6200 m3 over 31/28/31 days -> 100, 100, 200 m3/day while WLPR stays 100.
    liquid = (0., 3100., 5900., 12100.)
    oil = (0., 10., 20., 30.)
    injected = (0., 3100., 5900., 9000.)
    voidage_out = (0., 100., 200., 300.)
    voidage_in = (0., 130., 210., 330.)
    columns = [(well, vector) for well in wells for vector in (*REQUIRED_VECTORS, *optional)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["DATE", *(f"{v}:{w}" for w, v in columns)])
        for index, stamp in enumerate(stamps):
            row = {("P", "WLPR"): 100., ("P", "WOPR"): 10., ("P", "WWPR"): 90.,
                   ("P", "WLPT"): liquid[index], ("P", "WOPT"): oil[index],
                   ("P", "WWPT"): liquid[index] - oil[index],
                   ("P", "WBHP"): producer_bhp[index], ("P", "WBP9"): 120., ("P", "WEFF"): 1.,
                   ("P", "WVPT"): voidage_out[index],
                   ("I", "WWIR"): 50., ("I", "WWIT"): injected[index],
                   ("I", "WBHP"): 280., ("I", "WBP9"): 130., ("I", "WEFF"): 1.,
                   ("I", "WVIT"): voidage_in[index]}
            writer.writerow([stamp, *(row.get((w, v), 0.) for w, v in columns)])
    return path


def _rules(limits, *, window=1, status="hard"):
    return parse_constraints([dict(start="2014-01-01", end="2014-03-01", wells=["P", "I"],
                                   limits=limits, window_months=window, status=status)],
                             wells=["P", "I"], start=MONTHS[0], end=MONTHS[-1])


def test_field_caps_are_monthly_averages_not_endpoint_rates(tmp_path):
    report = _field_summary(tmp_path / "summary.csv")

    def verdicts(rules, months=MONTHS):
        return evaluate_summary(rules, report, deck_dir=tmp_path, months=months, unit_system="METRIC")

    # The endpoint rate never exceeds its cap; the March monthly average does.
    assert not failures(verdicts(_rules({"max_liquid_m3d": 150})))
    violated = failures(verdicts(_rules({"max_monthly_liquid_m3d": 150})))
    assert [v.month for v in violated] == [MONTHS[2]]
    assert violated[0].worst_value == pytest.approx(200.) and violated[0].margin == pytest.approx(-50.)
    assert not failures(verdicts(_rules({"max_monthly_liquid_m3d": 250})))
    # Injection: 3100/2800/3100 m3 is 100 m3/day every month, while WWIR reports 50.
    assert not failures(verdicts(_rules({"max_injection_m3d": 90})))
    assert len(failures(verdicts(_rules({"max_monthly_injection_m3d": 90})))) == 3
    assert not failures(verdicts(_rules({"max_monthly_injection_m3d": 150})))
    with pytest.raises(ValueError, match="observed max_monthly_liquid_m3d"):
        check_summary(_rules({"max_monthly_liquid_m3d": 150}), report,
                      deck_dir=tmp_path, months=MONTHS, unit_system="METRIC")


def test_voidage_window_uses_sums_over_the_available_window(tmp_path):
    report = _field_summary(tmp_path / "summary.csv")

    def verdicts(rules):
        return evaluate_summary(rules, report, deck_dir=tmp_path, months=MONTHS, unit_system="METRIC")

    # Monthly ratios 1.30 / 0.80 / 1.20; three-month windows 1.30 / 1.05 / 1.10.
    monthly = failures(verdicts(_rules({"max_monthly_voidage_replacement": 1.15})))
    assert [v.month for v in monthly] == [MONTHS[0], MONTHS[2]]
    window = failures(verdicts(_rules({"max_window_voidage_replacement": 1.15}, window=3)))
    assert [v.month for v in window] == [MONTHS[0]]
    assert window[0].worst_value == pytest.approx(1.30)
    ratios = [v.worst_value for v in verdicts(_rules({"max_window_voidage_replacement": 9}, window=3))
              if v.rule == "max_window_voidage_replacement"]
    assert ratios == pytest.approx([1.30, 1.05, 1.10])


def test_diagnostic_rule_is_reported_but_never_makes_a_candidate_infeasible(tmp_path):
    report = _field_summary(tmp_path / "summary.csv")
    limits = {"min_window_voidage_replacement": 1.1}
    diagnostic = _rules(limits, window=3, status="diagnostic")
    reported = evaluate_summary(diagnostic, report, deck_dir=tmp_path, months=MONTHS, unit_system="METRIC")
    breached = [v for v in reported if not v.ok]
    assert [v.month for v in breached] == [MONTHS[1]] and breached[0].status == "diagnostic"
    assert not failures(reported)
    check_summary(diagnostic, report, deck_dir=tmp_path, months=MONTHS, unit_system="METRIC")
    assert breached[0].to_dict()["month"] == "2014-02-01"
    with pytest.raises(ValueError, match="min_window_voidage_replacement"):
        check_summary(_rules(limits, window=3), report, deck_dir=tmp_path,
                      months=MONTHS, unit_system="METRIC")
    with pytest.raises(ValueError, match="one window length"):
        evaluate_summary((*diagnostic, *_rules({"max_window_voidage_replacement": 2}, window=2)),
                         report, deck_dir=tmp_path, months=MONTHS, unit_system="METRIC")


def test_bhp_is_checked_on_records_that_are_not_management_report_months(tmp_path):
    report = _field_summary(tmp_path / "summary.csv")
    rules = _rules({"min_bhp_bar": 50, "max_bhp_bar": 300})
    # The 40 bar record closes March, which no longer is a checked management month.
    reported = evaluate_summary(rules, report, deck_dir=tmp_path, months=MONTHS[:2], unit_system="METRIC")
    worst = [v for v in reported if v.scope == "all_summary_steps"]
    assert {v.rule: (v.worst_value, v.wells, v.month) for v in worst} == {
        "min_bhp_bar": (40., ("P",), MONTHS[2]), "max_bhp_bar": (280., ("I",), MONTHS[0])}
    assert [v.margin for v in worst if v.rule == "min_bhp_bar"] == [-10.]
    assert [v.margin for v in worst if v.rule == "max_bhp_bar"] == [20.]
    assert [v.month for v in failures(reported)] == [MONTHS[2]]
    with pytest.raises(ValueError, match="2014-04-01"):
        check_summary(rules, report, deck_dir=tmp_path, months=MONTHS[:2], unit_system="METRIC")
    healthy = _field_summary(tmp_path / "healthy.csv", producer_bhp=(60., 60., 60., 61.))
    margins = {v.rule: v.margin for v in evaluate_summary(
        rules, healthy, deck_dir=tmp_path, months=MONTHS[:2], unit_system="METRIC")
        if v.scope == "all_summary_steps"}
    assert margins == {"min_bhp_bar": pytest.approx(10.), "max_bhp_bar": pytest.approx(20.)}


def test_summary_overlay_requests_the_field_and_region_vectors_of_gate_g3():
    from timesoil.aios.opm import OPM_EXPORT_VECTORS, OPM_FIELD_VECTORS, OpmError, build_summary_overlay

    overlay = build_summary_overlay(("P", "I"))
    lines = overlay.splitlines()
    # Existing well and connection requests keep their order, byte for byte.
    assert lines[:3] == ["-- TIMESOIL AIOS SUMMARY OVERLAY; GENERATED IN RUN SNAPSHOT", "DATE", "WLPR"]
    assert OPM_EXPORT_VECTORS[0] == "WLPR" and "FPR" not in OPM_EXPORT_VECTORS
    assert lines[-len(OPM_FIELD_VECTORS) - 1:-1] == list(OPM_FIELD_VECTORS)
    assert lines[-1] == "-- RPR OMITTED: DECK FIPNUM REGION COUNT UNKNOWN"
    regional = build_summary_overlay(("P", "I"), fipnum_regions=2).splitlines()
    assert regional[-2:] == ["RPR", " 1 2 /"]
    assert regional[:-2] == lines[:-1]
    for bad in (0, -1, 1.0, True):
        with pytest.raises(OpmError, match="fipnum_regions"):
            build_summary_overlay(("P",), fipnum_regions=bad)
