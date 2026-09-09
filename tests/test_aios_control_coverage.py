from dataclasses import replace
from datetime import date

import pytest

from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
from timesoil.aios.scenario_generation import _actions_sha256, load_control_records
from timesoil.aios.track2 import _action_cube
from timesoil.aios.workflow import CycleError, _SourceControl, _action, _controls, _validate_source_well_scope


def test_optional_pressure_roundtrip_and_one_way_conversion():
    month, next_month = date(2014, 1, 1), date(2014, 2, 1)
    old = ControlAction(month, "P1", WellRole.PRODUCER, WellStatus.OPEN, ControlTarget.LIQUID_RATE, 100)
    bounded = replace(old, bhp_limit=70)
    assert "bhp_limit" not in _action(old)
    assert _controls([_action(bounded)]) == (bounded,)
    assert _actions_sha256((old,)) != _actions_sha256((bounded,))
    assert load_control_records([{"date": month.isoformat(), "well": "P1", "control_value": 100,
                                  "control_target": "LRAT", "status": "OPEN", "bhp_limit": 70}]) == (bounded,)
    with pytest.raises(ValueError, match="BHP action feature"):
        _action_cube((bounded,), (month,), ("P1",))
    converted = replace(old, role=WellRole.INJECTOR, target=ControlTarget.WATER_INJECTION_RATE, bhp_limit=280)
    inventory = {m: {"P1": _SourceControl(WellRole.PRODUCER, False, month)} for m in (month, next_month)}
    controls = (converted, replace(converted, month=next_month))
    with pytest.raises(CycleError):
        _validate_source_well_scope(controls, inventory)
    _validate_source_well_scope(controls, inventory, allow_conversion_to_injection=True)
    with pytest.raises(CycleError, match="reverse conversion"):
        _validate_source_well_scope((converted, replace(old, month=next_month)), inventory,
                                    allow_conversion_to_injection=True)
    with pytest.raises(CycleError):
        _validate_source_well_scope(tuple(replace(a, bhp_limit=None) for a in controls), inventory,
                                    allow_conversion_to_injection=True)
