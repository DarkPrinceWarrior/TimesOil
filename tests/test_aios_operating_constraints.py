from dataclasses import replace
from datetime import date

import pytest

from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
from timesoil.aios.operating_constraints import check_controls, check_observed, check_summary, parse_constraints


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
