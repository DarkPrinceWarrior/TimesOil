from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from datetime import date
from hashlib import sha256
from math import fsum, isclose
from pathlib import Path

import pytest

from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
from timesoil.aios.scenario_generation import (
    ScenarioGenerationError,
    ScenarioGeneratorConfig,
    generate_control_scenarios,
    load_control_csv,
    load_control_records,
)


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month, p1_target in (("2026-01-01", "LRAT"), ("2026-02-01", "WRAT")):
        rows.extend(
            [
                {
                    "date": month,
                    "well": "P1",
                    "control_value": 200 if p1_target == "LRAT" else 80,
                    "control_target": p1_target,
                    "status": "OPEN",
                },
                {
                    "date": month,
                    "well": "P2",
                    "control_value": 120,
                    "control_target": "ORAT",
                    "status": "OPEN",
                },
                {
                    "date": month,
                    "well": "I1",
                    "control_value": 150,
                    "control_target": "WRAT",
                    "status": "OPEN",
                },
                {
                    "date": month,
                    "well": "I2",
                    "control_value": 50,
                    "control_target": "WRAT",
                    "status": "OPEN",
                },
                {
                    "date": month,
                    "well": "S1",
                    "control_value": 0,
                    "control_target": "LRAT",
                    "status": "SHUT",
                },
            ]
        )
    return rows


def test_bundled_model_z_baseline_controls_are_exact() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "model_z_baseline_controls_v4.csv"
    )
    assert sha256(source.read_bytes()).hexdigest() == (
        "1a92c1e031ab7dca843f3f8824070f7fe85a2955fa270d45eb66b0638e88752f"
    )
    controls = load_control_csv(source)
    assert len(controls) == 38_213
    assert len({item.month for item in controls}) == 371
    assert len({item.well for item in controls}) == 103
    assert (controls[0].month, controls[-1].month) == (
        date(1994, 11, 1),
        date(2025, 9, 1),
    )


def test_generation_is_deterministic_diverse_and_immutable() -> None:
    baseline = load_control_records(_records())
    config = ScenarioGeneratorConfig(
        seed=17,
        perturbation_fraction=0.2,
        liquid_rate_scale=1.1,
        monthly_liquid_rate_cap=450,
    )

    first = generate_control_scenarios(baseline, config)
    second = generate_control_scenarios(reversed(baseline), config)

    assert first == second
    assert len(first) == 4
    assert len({item.sha256 for item in first}) == 4
    assert first[0].scenario_id == "baseline"
    assert first[0].actions == baseline
    assert dict(first[3].generator_parameters)["seed"] == 17
    with pytest.raises(FrozenInstanceError):
        first[0].scenario_id = "changed"  # type: ignore[misc]


def test_invariants_injection_preservation_cap_and_role_switch() -> None:
    baseline = load_control_records(_records())
    scenarios = generate_control_scenarios(
        baseline,
        ScenarioGeneratorConfig(
            seed=23,
            perturbation_fraction=0.3,
            monthly_liquid_rate_cap=350,
        ),
    )

    assert {
        action.role
        for action in baseline
        if action.well == "P1"
    } == {WellRole.PRODUCER, WellRole.INJECTOR}
    baseline_injection = {
        month: fsum(
            action.value
            for action in baseline
            if action.month == month
            and action.target is ControlTarget.WATER_INJECTION_RATE
        )
        for month in {action.month for action in baseline}
    }
    for scenario in scenarios[1:]:
        assert len({(action.month, action.well) for action in scenario.actions}) == len(
            scenario.actions
        )
        assert all(action.value >= 0 for action in scenario.actions)
        assert all(
            action.value <= 500
            for action in scenario.actions
            if action.target is ControlTarget.LIQUID_RATE
        )
        assert all(
            action.value == 0
            for action in scenario.actions
            if action.status is WellStatus.SHUT
        )
        for month, expected in baseline_injection.items():
            original_by_well = {
                action.well: action.value
                for action in baseline
                if action.month == month
                and action.target is ControlTarget.WATER_INJECTION_RATE
                and action.value > 0
            }
            perturbed_by_well = {
                action.well: action.value
                for action in scenario.actions
                if action.month == month
                and action.target is ControlTarget.WATER_INJECTION_RATE
                and action.value > 0
            }
            actual = fsum(
                action.value
                for action in scenario.actions
                if action.month == month
                and action.target is ControlTarget.WATER_INJECTION_RATE
            )
            assert isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
            assert all(
                0.7 * original_by_well[well]
                <= value
                <= 1.3 * original_by_well[well]
                for well, value in perturbed_by_well.items()
            )
            liquid = fsum(
                action.value
                for action in scenario.actions
                if action.month == month
                and action.target is ControlTarget.LIQUID_RATE
            )
            assert liquid <= 350 + 1e-12


def test_producer_only_search_scenarios_freeze_injection_controls() -> None:
    baseline = load_control_records(_records())
    scenarios = generate_control_scenarios(
        baseline,
        ScenarioGeneratorConfig(
            seed=29,
            perturbation_fraction=0.3,
            perturb_injection=False,
        ),
    )
    expected = {
        (action.month, action.well): action
        for action in baseline
        if action.target is ControlTarget.WATER_INJECTION_RATE
    }
    for scenario in scenarios[1:]:
        assert {
            (action.month, action.well): action
            for action in scenario.actions
            if action.target is ControlTarget.WATER_INJECTION_RATE
        } == expected
    assert any(
        candidate.value != original.value
        for candidate, original in zip(scenarios[1].actions, baseline, strict=True)
        if original.target is not ControlTarget.WATER_INJECTION_RATE
    )


def test_csv_roundtrip(tmp_path) -> None:
    path = tmp_path / "controls.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_records()[0]))
        writer.writeheader()
        writer.writerows(_records())

    assert load_control_csv(path) == load_control_records(_records())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows[:-1], "missing monthly well"),
        (lambda rows: rows + [rows[0].copy()], "duplicate control"),
        (
            lambda rows: [
                {**row, "control_target": "BHP"} if index == 0 else row
                for index, row in enumerate(rows)
            ],
            "invalid control row",
        ),
        (
            lambda rows: [
                {**row, "control_value": 1} if row["status"] == "SHUT" else row
                for row in rows
            ],
            "shut well",
        ),
        (
            lambda rows: [
                {**row, "control_value": float("nan")} if index == 0 else row
                for index, row in enumerate(rows)
            ],
            "finite",
        ),
        (
            lambda rows: [{key: value for key, value in row.items() if key != "status"} for row in rows],
            "exactly",
        ),
    ],
)
def test_records_fail_closed(mutate, message: str) -> None:
    with pytest.raises(ScenarioGenerationError, match=message):
        load_control_records(mutate(_records()))


def test_missing_month_lrat_limit_and_invalid_configuration_fail_closed() -> None:
    records = _records()
    march = [{**row, "date": "2026-03-01"} for row in records[:5]]
    with pytest.raises(ScenarioGenerationError, match="missing month"):
        load_control_records(records[:5] + march)

    records[0]["control_value"] = 501
    with pytest.raises(ScenarioGenerationError, match="LRAT control exceeds 500"):
        load_control_records(records)

    with pytest.raises(ScenarioGenerationError, match="scenario_count"):
        ScenarioGeneratorConfig(scenario_count=3)


def test_degenerate_baseline_cannot_claim_distinct_scenarios() -> None:
    baseline = (
        ControlAction(
            date(2026, 1, 1),
            "S1",
            WellRole.PRODUCER,
            WellStatus.SHUT,
            ControlTarget.LIQUID_RATE,
            0,
        ),
    )
    with pytest.raises(ScenarioGenerationError, match="insufficient controllable"):
        generate_control_scenarios(baseline)
