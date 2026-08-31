from __future__ import annotations

from dataclasses import replace
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest

from timesoil.aios.contracts import (
    Case,
    ControlAction,
    ControlTarget,
    WellRole,
    WellStatus,
)
from timesoil.aios.schedule import ScheduleCompiler
from timesoil.aios.schedule_overlay import (
    ScheduleOverlayError,
    apply_schedule_overlay,
)


ROOT = Path(__file__).parents[1]
MODEL_Y_SCHEDULE = ROOT / (
    "results/hackathon/certified_opm_runs/model-y-baseline-20260831-local/"
    "input/MODEL_Y/INCLUDE/DemoSpe_002_2_sch.inc"
)
MODEL_Z_SCHEDULE = ROOT / (
    "results/hackathon/certified_opm_runs/model-z-baseline-20260831-local/"
    "input/Model_Z/Model_Z_sch.inc"
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
 01 'JAN' 2025 / -- accepted by Model Y
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


def test_artifact_roundtrip_determinism_and_override_position() -> None:
    source = _source()
    before = source[:]
    case = Case(
        case_id="test",
        start=date(2025, 1, 1),
        end=date(2025, 3, 1),
        economics_start=date(2025, 1, 1),
        producers=("P1",),
        injectors=("I1",),
    )
    actions = (
        _action(date(2025, 1, 1), "P1"),
        _action(date(2025, 1, 1), "I1", WellRole.INJECTOR, 17.0),
    )
    compiled = ScheduleCompiler().compile(case, reversed(actions))

    from_artifact = apply_schedule_overlay(
        source, compiled, known_wells=("P1", "I1")
    )
    from_actions = apply_schedule_overlay(
        source, reversed(actions), known_wells=("I1", "P1")
    )

    assert source == before
    assert from_artifact == from_actions
    assert from_artifact.sha256 == sha256(from_artifact.text.encode()).hexdigest()
    assert from_artifact.controls_sha256 == compiled.sha256
    assert from_artifact.provenance["output_sha256"] == from_artifact.sha256
    january = from_artifact.text.split("DATES\n 01 FEB", 1)[0]
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


def test_tampered_schedule_artifact_fails_closed() -> None:
    case = Case(
        "test",
        date(2025, 1, 1),
        date(2025, 1, 1),
        date(2025, 1, 1),
        ("P1",),
        (),
    )
    artifact = ScheduleCompiler().compile(case, (_action(date(2025, 1, 1)),))
    malicious = artifact.text + "END\n"
    tampered = replace(
        artifact, text=malicious, sha256=sha256(malicious.encode()).hexdigest()
    )

    with pytest.raises(ScheduleOverlayError, match="canonical hash"):
        apply_schedule_overlay(_source(), tampered, known_wells=("P1",))


@pytest.mark.parametrize(
    ("path", "month", "role", "next_date"),
    [
        (MODEL_Y_SCHEDULE, date(2015, 11, 1), WellRole.PRODUCER, "01 'DEC' 2015"),
        (MODEL_Z_SCHEDULE, date(2025, 8, 1), WellRole.INJECTOR, "01 SEP 2025"),
    ],
)
def test_real_prepared_model_schedules(
    path: Path, month: date, role: WellRole, next_date: str
) -> None:
    if not path.is_file():
        pytest.skip("prepared organizer schedule is an external test artifact")
    source = path.read_text()
    artifact = apply_schedule_overlay(
        source,
        (_action(month, "1", role),),
        known_wells=("1",),
        replay_month=month,
    )

    assert artifact.source_sha256 == sha256(source.encode()).hexdigest()
    assert next_date in artifact.text
    assert artifact.text.rstrip().endswith("/")
    expected_keyword = "WCONPROD" if role is WellRole.PRODUCER else "WCONINJE"
    marker = artifact.text.rfind("-- TIMESOIL AIOS OVERRIDE")
    assert marker < artifact.text.rfind(expected_keyword) < artifact.text.rfind("DATES")
