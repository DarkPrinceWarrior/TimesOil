"""Full-GDM-certified monthly MPC core for Track 1."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from hashlib import sha256
from typing import Protocol, runtime_checkable

from .contracts import (
    Case,
    ControlAction,
    ControlTarget,
    Economics,
    Evidence,
    State,
    Trajectory,
    WellRole,
    WellState,
    WellStatus,
)
from .schedule import ScheduleArtifact, ScheduleCompiler


Candidate = tuple[ControlAction, ...]
CandidateProvider = Callable[[State], Sequence[Sequence[ControlAction]]]
Prescreener = Callable[[State, tuple[Candidate, ...]], Sequence[Sequence[ControlAction]]]


class CertificationError(RuntimeError):
    """No candidate passed every mandatory full-physics gate."""


@dataclass(frozen=True, slots=True)
class GdmResult:
    trajectory: Trajectory
    economics: Economics


@runtime_checkable
class GdmBackend(Protocol):
    """Adapter boundary for OPM Flow or the official tNavigator backend."""

    def validate_case(self, case: Case) -> None: ...

    def run_from_restart(
        self, case: Case, state: State, actions: Candidate
    ) -> GdmResult: ...

    def get_provenance(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Track1Result:
    schedule: ScheduleArtifact
    evidence: Evidence


def _next_month(month: date) -> date:
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def _candidate_key(actions: Candidate) -> tuple[tuple[str, str, str, str, float], ...]:
    return tuple(
        sorted(
            (
                action.role.value,
                action.well,
                action.status.value,
                action.target.value,
                action.value,
            )
            for action in actions
        )
    )


class MonthlyMPC:
    """Deterministic, fail-closed monthly controller over a certified GDM backend."""

    def __init__(
        self,
        backend: GdmBackend,
        *,
        compiler: ScheduleCompiler | None = None,
        prescreen: Prescreener | None = None,
    ) -> None:
        self.backend = backend
        self.compiler = compiler or ScheduleCompiler()
        self.prescreen = prescreen

    def run_step(
        self,
        case: Case,
        state: State,
        proposed: Sequence[Sequence[ControlAction]],
    ) -> GdmResult:
        """Certify one MPC month and return the deterministic best candidate."""

        self.backend.validate_case(case)
        self._validate_state(case, state)
        return self._select(case, state, proposed)

    def run(
        self, case: Case, initial_state: State, candidates: CandidateProvider
    ) -> Track1Result:
        self.backend.validate_case(case)
        if initial_state.case_id != case.case_id or initial_state.month != case.start:
            raise CertificationError("initial state does not match case start")
        self._validate_state(case, initial_state)

        state = initial_state
        accepted_actions: list[ControlAction] = []
        trajectories: list[Trajectory] = []
        economics: list[Economics] = []
        run_ids: set[str] = set()

        month = case.start
        while month <= case.end:
            if state.month != month:
                raise CertificationError("backend state skipped an MPC month")
            try:
                proposed = candidates(state)
            except Exception as exc:
                raise CertificationError(
                    f"candidate provider failed for {month.isoformat()}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            best = self._select(case, state, proposed)
            if best.trajectory.run_id in run_ids:
                raise CertificationError("backend reused run_id across MPC months")
            run_ids.add(best.trajectory.run_id)
            accepted_actions.extend(best.trajectory.actions)
            trajectories.append(best.trajectory)
            economics.append(best.economics)
            state = best.trajectory.next_state
            month = _next_month(month)

        schedule = self.compiler.compile(case, accepted_actions)
        evidence = Evidence(
            case_id=case.case_id,
            trajectories=tuple(trajectories),
            step_economics=tuple(economics),
            schedule_sha256=schedule.sha256,
            backend_provenance=self.backend.get_provenance(),
        )
        return Track1Result(schedule, evidence)

    def _select(
        self,
        case: Case,
        state: State,
        proposed: Sequence[Sequence[ControlAction]],
    ) -> GdmResult:
        candidates, failures = self._validated_candidates(case, state, proposed)
        if self.prescreen is not None and candidates:
            try:
                prescreened = self.prescreen(state, candidates)
            except Exception as exc:
                raise CertificationError(
                    f"prescreen failed for {state.month.isoformat()}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            candidates, rejected = self._validated_candidates(case, state, prescreened)
            failures.extend(rejected)

        certified: list[GdmResult] = []
        for candidate in candidates:
            try:
                result = self.backend.run_from_restart(case, state, candidate)
                self._check_result(case, state, candidate, result)
                certified.append(result)
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")

        run_ids = [result.trajectory.run_id for result in certified]
        if len(set(run_ids)) != len(run_ids):
            raise CertificationError("backend reused run_id for different candidates")
        if not certified:
            detail = "; ".join(failures) if failures else "no candidates"
            raise CertificationError(
                f"month {state.month.isoformat()} is not certified: {detail}"
            )
        return max(
            certified,
            key=lambda result: (
                result.economics.npv_million_rub,
                _candidate_key(result.trajectory.actions),
            ),
        )

    def _validated_candidates(
        self,
        case: Case,
        state: State,
        proposed: Sequence[Sequence[ControlAction]],
    ) -> tuple[tuple[Candidate, ...], list[str]]:
        try:
            values = tuple(proposed)
        except TypeError as exc:
            return (), [f"{type(exc).__name__}: candidates are not iterable"]

        unique: dict[tuple[tuple[str, str, str, str, float], ...], Candidate] = {}
        failures: list[str] = []
        for candidate in values:
            try:
                normalized = self.compiler.validate(case, candidate)
                if not normalized:
                    raise CertificationError("candidate is empty")
                if any(action.month != state.month for action in normalized):
                    raise CertificationError("candidate contains another month")
                unique[_candidate_key(normalized)] = normalized
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        return tuple(unique[key] for key in sorted(unique)), failures

    @staticmethod
    def _validate_state(case: Case, state: State) -> None:
        if state.case_id != case.case_id:
            raise CertificationError("state belongs to another case")
        if not case.start <= state.month <= case.end:
            raise CertificationError("state month is outside case horizon")
        for well in state.wells:
            try:
                expected = case.role_of(well.well)
            except Exception as exc:
                raise CertificationError(
                    f"state contains unknown well {well.well}"
                ) from exc
            if expected is not well.role:
                raise CertificationError(f"state contains wrong role for {well.well}")

    @staticmethod
    def _check_result(
        case: Case, state: State, actions: Candidate, result: GdmResult
    ) -> None:
        trajectory = result.trajectory
        economics = result.economics
        if not trajectory.certified:
            raise CertificationError("GDM backend did not certify trajectory")
        if trajectory.case_id != case.case_id or trajectory.month != state.month:
            raise CertificationError("GDM result belongs to another case or month")
        if trajectory.actions != actions:
            raise CertificationError("GDM certified different actions")
        if trajectory.next_state.month != _next_month(state.month):
            raise CertificationError("GDM restart is not the next month")
        if trajectory.next_state.restart_ref == state.restart_ref:
            raise CertificationError("GDM backend reused the input restart")
        if not trajectory.chdd_complete or trajectory.invariant_violations:
            raise CertificationError("trajectory failed CHDD or invariant gates")
        if economics.run_id != trajectory.run_id or not economics.complete:
            raise CertificationError("economics is incomplete or unrelated")
        if economics.start_date != case.economics_start:
            raise CertificationError("economics start date differs from case contract")


class DeterministicGdmBackend:
    """Deterministic test backend; never a substitute for full reservoir physics."""

    name = "deterministic-test-gdm-v1"

    def validate_case(self, case: Case) -> None:
        if not case.producers or not case.injectors:
            raise CertificationError("test backend requires both well roles")

    def get_provenance(self) -> str:
        return self.name

    def run_from_restart(
        self, case: Case, state: State, actions: Candidate
    ) -> GdmResult:
        payload = "|".join(
            [
                case.case_id,
                state.restart_ref,
                *(
                    f"{a.month}:{a.well}:{a.role}:{a.status}:{a.target}:{a.value:.12g}"
                    for a in actions
                ),
            ]
        )
        run_id = sha256(payload.encode()).hexdigest()[:24]
        by_well = {well.well: well for well in state.wells}
        for action in actions:
            old = by_well.get(
                action.well,
                WellState(action.well, action.role, action.status is WellStatus.OPEN),
            )
            active = action.status is WellStatus.OPEN
            if action.role is WellRole.INJECTOR:
                by_well[action.well] = replace(
                    old, active=active, injection_rate=action.value if active else 0.0
                )
            elif action.target is ControlTarget.OIL_RATE:
                oil = action.value if active else 0.0
                by_well[action.well] = replace(
                    old, active=active, oil_rate=oil, liquid_rate=max(old.liquid_rate, oil)
                )
            else:
                liquid = action.value if active else 0.0
                by_well[action.well] = replace(
                    old,
                    active=active,
                    liquid_rate=liquid,
                    oil_rate=min(old.oil_rate, liquid),
                )
        next_state = State(
            case.case_id,
            _next_month(state.month),
            f"{run_id}:restart",
            tuple(sorted(by_well.values(), key=lambda well: well.well)),
        )
        trajectory = Trajectory(
            run_id=run_id,
            case_id=case.case_id,
            month=state.month,
            actions=actions,
            next_state=next_state,
            simulator=self.name,
            certified=True,
            chdd_complete=True,
        )
        production = sum(
            action.value
            for action in actions
            if action.role is WellRole.PRODUCER and action.status is WellStatus.OPEN
        )
        injection = sum(
            action.value
            for action in actions
            if action.role is WellRole.INJECTOR and action.status is WellStatus.OPEN
        )
        result_economics = Economics(
            run_id=run_id,
            start_date=case.economics_start,
            npv_million_rub=production - 0.1 * injection,
            complete=True,
        )
        return GdmResult(trajectory, result_economics)
