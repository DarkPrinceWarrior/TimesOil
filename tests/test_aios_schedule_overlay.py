from __future__ import annotations

from dataclasses import replace
from datetime import date
from hashlib import sha256

import pytest

from timesoil.aios.contracts import (
    ControlAction,
    ControlTarget,
    WellRole,
    WellStatus,
)
from timesoil.aios.schedule_overlay import (
    ScheduleOverlayError,
    apply_schedule_overlay,
)


def _action(
    month: date,
    well: str = "P1",
    role: WellRole = WellRole.PRODUCER,
    value: float = 42.0,
) -> ControlAction:
    target = (
        ControlTarget.LIQUID_RATE
        if role is WellRole.PRODUCER
        else ControlTarget.WATER_INJECTION_RATE
    )
    return ControlAction(month, well, role, WellStatus.OPEN, target, value)


def _source() -> str:
    return """-- real-style quoted/unquoted dates and include
DATES -- January report date
 01 'JAN' 2025 / -- accepted by the simulator
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 3* 10 1* 50 1* 1* /
/
INCLUDE
 'controls/january.inc' /
-- overlay must follow INCLUDE and baseline controls
DATES
 01 FEB 2025 /
/
WCONINJE
 'I1' 'WATER' 'OPEN' 'RATE' 20 1* 300 1* 1* /
/
DATES
 01 MAR 2025 /
/
END
"""


def test_pressure_and_conversion_preserve_original_bounds() -> None:
    january = replace(_action(date(2025, 1, 1)), bhp_limit=70)
    february = replace(_action(date(2025, 2, 1), "P1", WellRole.INJECTOR, 25), bhp_limit=280)
    assert february.to_dict()["bhp_limit"] == 280
    assert "bhp_limit" not in _action(date(2025, 1, 1)).to_dict()
    overlay = apply_schedule_overlay(_source(), (january, february), known_wells=("P1", "I1"))
    assert "70.000000" in overlay.text and "280.000000" in overlay.text
    with pytest.raises(ScheduleOverlayError, match="relaxes"):
        apply_schedule_overlay(_source(), (replace(january, bhp_limit=40),), known_wells=("P1", "I1"))
    with pytest.raises(ScheduleOverlayError, match="relaxes"):
        apply_schedule_overlay(_source(), (replace(february, well="I1", bhp_limit=310),), known_wells=("P1", "I1"))


def test_control_order_is_canonical_and_override_follows_baseline_controls() -> None:
    source = _source()
    before = source[:]
    actions = (
        _action(date(2025, 1, 1), "P1"),
        _action(date(2025, 1, 1), "I1", WellRole.INJECTOR, 17.0),
    )

    sorted_order = apply_schedule_overlay(source, actions, known_wells=("P1", "I1"))
    reversed_order = apply_schedule_overlay(
        source, reversed(actions), known_wells=("I1", "P1")
    )

    assert source == before
    assert sorted_order == reversed_order
    assert sorted_order.sha256 == sha256(sorted_order.text.encode()).hexdigest()
    assert len(sorted_order.controls_sha256) == 64
    assert sorted_order.provenance["output_sha256"] == sorted_order.sha256
    january = sorted_order.text.split("DATES\n 01 FEB", 1)[0]
    assert january.rfind("-- TIMESOIL AIOS OVERRIDE") > january.rfind("INCLUDE")
    assert january.rstrip().endswith("/")
    assert "'P1' 'OPEN' 'LRAT'" in january
    assert "'I1' 'WATER' 'OPEN' 'RATE'" in january


def test_role_switch_comes_from_each_control_action() -> None:
    actions = (
        _action(date(2025, 1, 1), "P1", WellRole.PRODUCER),
        _action(date(2025, 2, 1), "P1", WellRole.INJECTOR),
    )

    artifact = apply_schedule_overlay(_source(), actions, known_wells=("P1",))

    january, remainder = artifact.text.split("DATES\n 01 FEB", 1)
    february = remainder.split("DATES\n 01 MAR", 1)[0]
    assert "WCONPROD" in january
    assert "'P1' 'OPEN' 'LRAT'" in january
    assert "-- TIMESOIL AIOS OVERRIDE" in february
    assert "'P1' 'WATER' 'OPEN' 'RATE'" in february


def test_overlay_preserves_effective_bhp_constraints() -> None:
    artifact = apply_schedule_overlay(
        _source(),
        (
            _action(date(2025, 3, 1), "P1", value=11.0),
            _action(date(2025, 3, 1), "I1", WellRole.INJECTOR, 21.0),
        ),
        known_wells=("P1", "I1"),
    )

    assert "'P1' 'OPEN' 'LRAT' 1* 1* 1* 11.000000 1* 50 1* 1* /" in artifact.text
    assert "'I1' 'WATER' 'OPEN' 'RATE' 21.000000 1* 300 1* 1* /" in artifact.text


def test_one_month_replay_stops_after_following_date() -> None:
    artifact = apply_schedule_overlay(
        _source(),
        (_action(date(2025, 1, 1)),),
        known_wells=("P1",),
        replay_month=date(2025, 1, 1),
    )

    assert artifact.mode == "one_month"
    assert artifact.truncated_after == date(2025, 2, 1)
    assert artifact.text.count("DATES") == 2
    assert "01 FEB 2025 /" in artifact.text
    assert "'I1' 'WATER'" not in artifact.text
    assert "01 MAR 2025" not in artifact.text
    assert "END" not in artifact.text


def test_horizon_stop_preserves_history_future_completions_and_managed_months() -> None:
    source = _source() + "-- unused future\n"
    actions = (_action(date(2025, 1, 1)), _action(date(2025, 2, 1)))
    full = apply_schedule_overlay(source, actions, known_wells=("P1",))
    bounded = apply_schedule_overlay(
        source, actions, known_wells=("P1",), end_exclusive=date(2025, 3, 1)
    )
    assert bounded.text.endswith(full.text)
    assert bounded.text.startswith("ACTIONX\n 'TSSTOP' 1 /\n")
    assert "MNTH = MAR AND /\n YEAR = 2025 /" in bounded.text
    assert bounded.text.count("-- TIMESOIL AIOS OVERRIDE") == 2
    assert bounded.controls_sha256 == full.controls_sha256
    assert bounded.truncated_after is None
    assert bounded.stopped_after == date(2025, 3, 1)
    for invalid in (date(2025, 2, 1), date(2025, 3, 2), date(2026, 1, 1)):
        with pytest.raises(ScheduleOverlayError):
            apply_schedule_overlay(source, actions, known_wells=("P1",), end_exclusive=invalid)


@pytest.mark.parametrize(
    ("source", "controls", "wells", "replay"),
    [
        (
            _source(),
            (_action(date(2025, 1, 1)), _action(date(2025, 1, 1))),
            ("P1",),
            None,
        ),
        (_source(), (_action(date(2025, 1, 1), "X"),), ("P1",), None),
        (_source(), (_action(date(2026, 1, 1)),), ("P1",), None),
        (_source() + "\x00", (_action(date(2025, 1, 1)),), ("P1",), None),
        (
            _source().replace("01 FEB 2025", "01 JAN 2025"),
            (_action(date(2025, 1, 1)),),
            ("P1",),
            None,
        ),
        (
            _source().replace(" 01 'JAN' 2025 /", " 01 JAN 2025 /\n 02 JAN 2025 /"),
            (_action(date(2025, 1, 1)),),
            ("P1",),
            None,
        ),
        (_source(), (_action(date(2025, 3, 1)),), ("P1",), date(2025, 3, 1)),
    ],
)
def test_fail_closed(
    source: str,
    controls: tuple[ControlAction, ...],
    wells: tuple[str, ...],
    replay: date | None,
) -> None:
    with pytest.raises(ScheduleOverlayError):
        apply_schedule_overlay(
            source, controls, known_wells=wells, replay_month=replay
        )
