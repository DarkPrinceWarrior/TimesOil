from dataclasses import replace
from datetime import date

import pytest

from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
from timesoil.aios.operating_constraints import check_controls, check_observed, parse_constraints


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
