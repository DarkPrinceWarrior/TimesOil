"""Typed, dependency-free contracts shared by AIOS control loops."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from math import isfinite
import re


_WELL_NAME = re.compile(r"[A-Za-z0-9_.-]+")


class ContractError(ValueError):
    """A typed AIOS contract is internally inconsistent."""


class WellRole(StrEnum):
    PRODUCER = "producer"
    INJECTOR = "injector"


class WellStatus(StrEnum):
    OPEN = "OPEN"
    SHUT = "SHUT"


class ControlTarget(StrEnum):
    OIL_RATE = "ORAT"
    LIQUID_RATE = "LRAT"
    WATER_INJECTION_RATE = "WRAT"


def _validate_month(value: date, field: str) -> None:
    if value.day != 1:
        raise ContractError(f"{field} must be the first day of a month")


def _validate_well_name(value: str) -> None:
    if not _WELL_NAME.fullmatch(value):
        raise ContractError(f"unsafe or empty well name: {value!r}")


@dataclass(frozen=True, slots=True)
class Case:
    """Static simulator case contract for an inclusive monthly horizon."""

    case_id: str
    start: date
    end: date
    economics_start: date
    producers: tuple[str, ...]
    injectors: tuple[str, ...]
    max_liquid_rate: float = 500.0

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ContractError("case_id is required")
        _validate_month(self.start, "start")
        _validate_month(self.end, "end")
        _validate_month(self.economics_start, "economics_start")
        if self.start > self.end:
            raise ContractError("case start is after case end")
        for well in (*self.producers, *self.injectors):
            _validate_well_name(well)
        if len(set(self.producers)) != len(self.producers):
            raise ContractError("producer names must be unique")
        if len(set(self.injectors)) != len(self.injectors):
            raise ContractError("injector names must be unique")
        overlap = set(self.producers) & set(self.injectors)
        if overlap:
            raise ContractError(f"wells have two roles: {sorted(overlap)}")
        if not isfinite(self.max_liquid_rate) or self.max_liquid_rate <= 0:
            raise ContractError("max_liquid_rate must be finite and positive")

    def role_of(self, well: str) -> WellRole:
        if well in self.producers:
            return WellRole.PRODUCER
        if well in self.injectors:
            return WellRole.INJECTOR
        raise ContractError(f"well {well!r} is absent from case {self.case_id!r}")


@dataclass(frozen=True, slots=True)
class WellState:
    well: str
    role: WellRole
    active: bool
    oil_rate: float = 0.0
    liquid_rate: float = 0.0
    injection_rate: float = 0.0
    bhp: float | None = None

    def __post_init__(self) -> None:
        _validate_well_name(self.well)
        values = (self.oil_rate, self.liquid_rate, self.injection_rate)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ContractError("well rates must be finite and non-negative")
        if self.bhp is not None and not isfinite(self.bhp):
            raise ContractError("bhp must be finite when present")
        if self.oil_rate > self.liquid_rate:
            raise ContractError("oil rate cannot exceed liquid rate")


@dataclass(frozen=True, slots=True)
class State:
    """State extracted from a certified full-physics restart."""

    case_id: str
    month: date
    restart_ref: str
    wells: tuple[WellState, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.restart_ref.strip():
            raise ContractError("case_id and restart_ref are required")
        _validate_month(self.month, "state month")
        names = [well.well for well in self.wells]
        if len(set(names)) != len(names):
            raise ContractError("state contains duplicate wells")


@dataclass(frozen=True, slots=True)
class ControlAction:
    """One simulator-facing well control for one monthly step."""

    month: date
    well: str
    role: WellRole
    status: WellStatus
    target: ControlTarget
    value: float

    def __post_init__(self) -> None:
        _validate_month(self.month, "action month")
        _validate_well_name(self.well)
        if not isfinite(self.value) or self.value < 0:
            raise ContractError("control value must be finite and non-negative")
        if self.status is WellStatus.SHUT and self.value != 0:
            raise ContractError("a shut well must have a zero target")
        producer_targets = {ControlTarget.OIL_RATE, ControlTarget.LIQUID_RATE}
        if self.role is WellRole.PRODUCER and self.target not in producer_targets:
            raise ContractError("producer control must target ORAT or LRAT")
        if (
            self.role is WellRole.INJECTOR
            and self.target is not ControlTarget.WATER_INJECTION_RATE
        ):
            raise ContractError("injector control must target water injection rate")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One full-GDM monthly result; uncertified results are never accepted."""

    run_id: str
    case_id: str
    month: date
    actions: tuple[ControlAction, ...]
    next_state: State
    simulator: str
    certified: bool
    chdd_complete: bool
    invariant_violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.simulator.strip():
            raise ContractError("run_id and simulator are required")
        _validate_month(self.month, "trajectory month")
        if self.next_state.case_id != self.case_id:
            raise ContractError("trajectory changed case_id")
        if any(action.month != self.month for action in self.actions):
            raise ContractError("trajectory contains an action from another month")
        if self.certified and (not self.chdd_complete or self.invariant_violations):
            raise ContractError("certified trajectory failed a mandatory gate")


@dataclass(frozen=True, slots=True)
class Economics:
    """Official CHDD result for one certified candidate."""

    run_id: str
    start_date: date
    npv_million_rub: float
    complete: bool

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ContractError("economics run_id is required")
        _validate_month(self.start_date, "economics start_date")
        if not isfinite(self.npv_million_rub):
            raise ContractError("NPV must be finite")


@dataclass(frozen=True, slots=True)
class Evidence:
    """Immutable proof chain for all actions accepted by the MPC loop."""

    case_id: str
    trajectories: tuple[Trajectory, ...]
    step_economics: tuple[Economics, ...]
    schedule_sha256: str
    simulator_provenance: str

    def __post_init__(self) -> None:
        if len(self.trajectories) != len(self.step_economics):
            raise ContractError("every trajectory needs an economics result")
        if not self.trajectories:
            raise ContractError("evidence cannot be empty")
        for trajectory, economics in zip(
            self.trajectories, self.step_economics, strict=True
        ):
            if trajectory.case_id != self.case_id or not trajectory.certified:
                raise ContractError("evidence contains an uncertified trajectory")
            if economics.run_id != trajectory.run_id or not economics.complete:
                raise ContractError("evidence contains incomplete or unrelated economics")
        if not re.fullmatch(r"[0-9a-f]{64}", self.schedule_sha256):
            raise ContractError("schedule_sha256 must be a lowercase SHA-256")
        if not self.simulator_provenance.strip():
            raise ContractError("simulator provenance is required")

    @property
    def final_economics(self) -> Economics:
        return self.step_economics[-1]
