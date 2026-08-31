"""Deterministic control-scenario generation for Track 2 OPM runs.

This module creates controls only. Simulator outputs and surrogate metrics must
come from subsequent, provenance-checked OPM runs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .contracts import ControlAction, ControlTarget, WellRole, WellStatus


_FIELDS = frozenset({"date", "well", "control_value", "control_target", "status"})
_MAX_LRAT = 500.0
GeneratorValue: TypeAlias = str | int | float | None


class ScenarioGenerationError(ValueError):
    """Baseline controls or generator parameters are unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class ScenarioGeneratorConfig:
    """Bounds shared by all reproducible scenarios in one generation batch."""

    scenario_count: int = 4
    seed: int = 20260831
    perturbation_fraction: float = 0.15
    liquid_rate_scale: float = 1.0
    monthly_liquid_rate_cap: float | None = None
    perturb_injection: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.scenario_count, bool) or self.scenario_count < 4:
            raise ScenarioGenerationError("scenario_count must be an integer >= 4")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ScenarioGenerationError("seed must be an integer")
        if (
            not isfinite(self.perturbation_fraction)
            or not 0 < self.perturbation_fraction <= 1
        ):
            raise ScenarioGenerationError(
                "perturbation_fraction must be finite and in (0, 1]"
            )
        if not isfinite(self.liquid_rate_scale) or self.liquid_rate_scale <= 0:
            raise ScenarioGenerationError("liquid_rate_scale must be finite and positive")
        if not isinstance(self.perturb_injection, bool):
            raise ScenarioGenerationError("perturb_injection must be a boolean")
        cap = self.monthly_liquid_rate_cap
        if cap is not None and (not isfinite(cap) or cap <= 0):
            raise ScenarioGenerationError(
                "monthly_liquid_rate_cap must be finite and positive"
            )


@dataclass(frozen=True, slots=True)
class ScenarioArtifact:
    """Immutable simulator input with a content hash and exact generator settings."""

    scenario_id: str
    actions: tuple[ControlAction, ...]
    sha256: str
    generator_parameters: tuple[tuple[str, GeneratorValue], ...]

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.actions:
            raise ScenarioGenerationError("scenario_id and actions are required")
        if self.sha256 != _actions_sha256(self.actions):
            raise ScenarioGenerationError("scenario action hash mismatch")
        names = [name for name, _ in self.generator_parameters]
        if len(names) != len(set(names)):
            raise ScenarioGenerationError("duplicate generator parameter")


def load_control_records(
    records: Iterable[Mapping[str, object]],
) -> tuple[ControlAction, ...]:
    """Parse canonical long records into a complete monthly control trajectory."""

    try:
        rows = tuple(records)
    except TypeError as exc:
        raise ScenarioGenerationError("control records must be iterable") from exc
    if not rows:
        raise ScenarioGenerationError("control trajectory is empty")

    actions: list[ControlAction] = []
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or set(row) != _FIELDS:
            raise ScenarioGenerationError(
                f"row {number} must contain exactly {sorted(_FIELDS)}"
            )
        try:
            month = _month(row["date"])
            well = row["well"]
            if not isinstance(well, str):
                raise ValueError("well must be a string")
            target = ControlTarget(row["control_target"])
            status = WellStatus(row["status"])
            value = _value(row["control_value"])
            role = (
                WellRole.INJECTOR
                if target is ControlTarget.WATER_INJECTION_RATE
                else WellRole.PRODUCER
            )
            actions.append(ControlAction(month, well, role, status, target, value))
        except (TypeError, ValueError) as exc:
            raise ScenarioGenerationError(f"invalid control row {number}: {exc}") from exc
    return _validate_trajectory(actions)


def load_control_csv(path: str | Path) -> tuple[ControlAction, ...]:
    """Load the exact canonical control CSV schema."""

    source = Path(path)
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            if header is None or len(header) != len(set(header)) or set(header) != _FIELDS:
                raise ScenarioGenerationError(
                    f"CSV header must contain exactly {sorted(_FIELDS)}"
                )
            return load_control_records(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ScenarioGenerationError(f"cannot read control CSV {source}: {exc}") from exc


def generate_control_scenarios(
    baseline: Iterable[ControlAction],
    config: ScenarioGeneratorConfig | None = None,
) -> tuple[ScenarioArtifact, ...]:
    """Return baseline plus bounded, seeded controls for real OPM simulations."""

    settings = config or ScenarioGeneratorConfig()
    actions = _validate_trajectory(baseline)
    _validate_generation_bounds(actions, settings)

    artifacts = [_artifact("baseline", actions, settings, 0)]
    content_hashes = {artifacts[0].sha256}
    for index in range(1, settings.scenario_count):
        rng = np.random.default_rng(np.random.SeedSequence([settings.seed, index]))
        for _ in range(64):
            candidate = _perturb(actions, settings, rng)
            digest = _actions_sha256(candidate)
            if digest not in content_hashes:
                break
        else:
            raise ScenarioGenerationError(
                "baseline has insufficient controllable degrees of freedom "
                "for distinct bounded scenarios"
            )
        content_hashes.add(digest)
        artifacts.append(_artifact(f"perturbation-{index:03d}", candidate, settings, index))
    return tuple(artifacts)


def _month(value: object) -> date:
    if type(value) is date:
        month = value
    elif isinstance(value, str):
        month = date.fromisoformat(value)
    else:
        raise ValueError("date must be an ISO date or datetime.date")
    if month.day != 1:
        raise ValueError("date must be the first day of a month")
    return month


def _value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("control_value must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("control_value must be numeric") from exc
    if not isfinite(result) or result < 0:
        raise ValueError("control_value must be finite and non-negative")
    return result


def _next_month(month: date) -> date:
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def _validate_trajectory(actions: Iterable[ControlAction]) -> tuple[ControlAction, ...]:
    try:
        ordered = tuple(sorted(actions, key=lambda item: (item.month, item.well)))
    except (AttributeError, TypeError) as exc:
        raise ScenarioGenerationError(
            "trajectory must contain only ControlAction values"
        ) from exc
    if not ordered or any(not isinstance(item, ControlAction) for item in ordered):
        raise ScenarioGenerationError("trajectory must contain ControlAction values")
    keys = [(item.month, item.well) for item in ordered]
    if len(keys) != len(set(keys)):
        raise ScenarioGenerationError("duplicate control for a monthly well")
    if any(
        item.target is ControlTarget.LIQUID_RATE and item.value > _MAX_LRAT
        for item in ordered
    ):
        raise ScenarioGenerationError("LRAT control exceeds 500")

    by_month: dict[date, set[str]] = defaultdict(set)
    for item in ordered:
        by_month[item.month].add(item.well)
    months = sorted(by_month)
    if any(right != _next_month(left) for left, right in zip(months, months[1:])):
        raise ScenarioGenerationError("control trajectory has a missing month")
    expected_wells = by_month[months[0]]
    if any(wells != expected_wells for wells in by_month.values()):
        raise ScenarioGenerationError("control trajectory has a missing monthly well")
    return ordered


def _validate_generation_bounds(
    actions: tuple[ControlAction, ...], config: ScenarioGeneratorConfig
) -> None:
    cap = config.monthly_liquid_rate_cap
    if cap is None:
        return
    totals: dict[date, float] = defaultdict(float)
    for action in actions:
        if action.target is ControlTarget.LIQUID_RATE:
            totals[action.month] += action.value
    if any(total > cap for total in totals.values()):
        raise ScenarioGenerationError(
            "baseline monthly LRAT total exceeds monthly_liquid_rate_cap"
        )


def _perturb(
    baseline: tuple[ControlAction, ...],
    config: ScenarioGeneratorConfig,
    rng: np.random.Generator,
) -> tuple[ControlAction, ...]:
    values = [action.value for action in baseline]
    by_month: dict[date, list[int]] = defaultdict(list)
    for index, action in enumerate(baseline):
        by_month[action.month].append(index)

    low, high = 1 - config.perturbation_fraction, 1 + config.perturbation_fraction
    for indices in by_month.values():
        injection = [
            index
            for index in indices
            if baseline[index].target is ControlTarget.WATER_INJECTION_RATE
            and baseline[index].status is WellStatus.OPEN
            and baseline[index].value > 0
        ]
        if injection and config.perturb_injection:
            original = [baseline[index].value for index in injection]
            original_total = fsum(original)
            weighted = [
                baseline[index].value * float(rng.uniform(low, high))
                for index in injection
            ]
            denominator = fsum(weighted)
            normalized = [original_total * value / denominator for value in weighted]
            deltas = [value - base for value, base in zip(normalized, original)]
            alpha = min(
                [
                    original[position] * config.perturbation_fraction / abs(delta)
                    for position, delta in enumerate(deltas)
                    if delta
                ]
                + [1.0]
            )
            redistributed = [
                base + min(1.0, alpha) * delta
                for base, delta in zip(original, deltas)
            ]
            redistributed[-1] = original_total - fsum(redistributed[:-1])
            for index, value in zip(injection, redistributed):
                values[index] = value

        lrat: list[int] = []
        for index in indices:
            action = baseline[index]
            if action.status is WellStatus.SHUT or action.value == 0:
                values[index] = 0.0
            elif action.target is ControlTarget.LIQUID_RATE:
                values[index] = min(
                    _MAX_LRAT,
                    action.value
                    * config.liquid_rate_scale
                    * float(rng.uniform(low, high)),
                )
                lrat.append(index)
            elif action.target is ControlTarget.OIL_RATE:
                values[index] = action.value * float(rng.uniform(low, high))

        cap = config.monthly_liquid_rate_cap
        total_lrat = fsum(values[index] for index in lrat)
        if cap is not None and total_lrat > cap:
            factor = cap / total_lrat
            for index in lrat:
                values[index] *= factor

    return tuple(replace(action, value=values[index]) for index, action in enumerate(baseline))


def _canonical_actions(actions: tuple[ControlAction, ...]) -> bytes:
    rows = [
        {
            "date": action.month.isoformat(),
            "well": action.well,
            "role": action.role.value,
            "status": action.status.value,
            "control_target": action.target.value,
            "control_value": action.value,
        }
        for action in actions
    ]
    return json.dumps(
        rows, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _actions_sha256(actions: tuple[ControlAction, ...]) -> str:
    return sha256(_canonical_actions(actions)).hexdigest()


def _artifact(
    scenario_id: str,
    actions: tuple[ControlAction, ...],
    config: ScenarioGeneratorConfig,
    scenario_index: int,
) -> ScenarioArtifact:
    parameters: tuple[tuple[str, GeneratorValue], ...] = (
        ("generator", "timesoil.aios.scenario_generation.v1"),
        ("scenario_index", scenario_index),
        ("scenario_count", config.scenario_count),
        ("seed", config.seed),
        ("perturbation_fraction", config.perturbation_fraction),
        ("liquid_rate_scale", config.liquid_rate_scale),
        ("monthly_liquid_rate_cap", config.monthly_liquid_rate_cap),
    ) + (() if config.perturb_injection else (("perturb_injection", False),))
    return ScenarioArtifact(scenario_id, actions, _actions_sha256(actions), parameters)
