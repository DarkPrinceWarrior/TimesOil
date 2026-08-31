from __future__ import annotations

from dataclasses import replace
from datetime import date
import unittest

from timesoil.aios.contracts import (
    Case,
    ControlAction,
    ControlTarget,
    State,
    WellRole,
    WellStatus,
)
from timesoil.aios.schedule import ScheduleCompiler, ScheduleError
from timesoil.aios.track1 import (
    CertificationError,
    DeterministicGdmBackend,
    GdmResult,
    MonthlyMPC,
)


def _case(end: date = date(2014, 2, 1)) -> Case:
    return Case(
        case_id="model-y-test",
        start=date(2014, 1, 1),
        end=end,
        economics_start=date(2014, 1, 1),
        producers=("P1",),
        injectors=("I1",),
    )


def _actions(month: date, production: float) -> tuple[ControlAction, ...]:
    return (
        ControlAction(
            month,
            "I1",
            WellRole.INJECTOR,
            WellStatus.OPEN,
            ControlTarget.WATER_INJECTION_RATE,
            100.0,
        ),
        ControlAction(
            month,
            "P1",
            WellRole.PRODUCER,
            WellStatus.OPEN,
            ControlTarget.LIQUID_RATE,
            production,
        ),
    )


class Track1Test(unittest.TestCase):
    def test_schedule_is_deterministic_and_round_trips(self) -> None:
        case = _case(date(2014, 1, 1))
        compiler = ScheduleCompiler()
        actions = _actions(case.start, 120.0)

        first = compiler.compile(case, actions)
        second = compiler.compile(case, reversed(actions))

        self.assertEqual(first.text, second.text)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(compiler.parse(case, first.text), first.actions)
        self.assertIn("DATES", first.text)
        self.assertIn("WCONPROD", first.text)
        self.assertIn("WCONINJE", first.text)

    def test_schedule_rejects_wrong_role_duplicate_and_tampering(self) -> None:
        case = _case(date(2014, 1, 1))
        compiler = ScheduleCompiler()
        actions = _actions(case.start, 120.0)

        with self.assertRaisesRegex(ScheduleError, "duplicate"):
            compiler.compile(case, (*actions, actions[0]))
        with self.assertRaisesRegex(ScheduleError, "wrong role"):
            compiler.compile(
                case,
                (
                    replace(
                        actions[0],
                        role=WellRole.PRODUCER,
                        target=ControlTarget.LIQUID_RATE,
                    ),
                ),
            )
        artifact = compiler.compile(case, actions)
        with self.assertRaises(ScheduleError):
            compiler.parse(case, artifact.text.replace("WCONPROD", "BADKEYWORD"))

    def test_monthly_mpc_accepts_only_backend_certified_steps(self) -> None:
        case = _case(date(2014, 6, 1))
        state = State(case.case_id, case.start, "restart-0", ())
        calls: list[date] = []

        def candidates(current: State) -> tuple[tuple[ControlAction, ...], ...]:
            calls.append(current.month)
            return (_actions(current.month, 100.0), _actions(current.month, 140.0))

        result = MonthlyMPC(DeterministicGdmBackend()).run(case, state, candidates)

        self.assertEqual(calls, [date(2014, month, 1) for month in range(1, 7)])
        self.assertEqual(len(result.evidence.trajectories), 6)
        self.assertTrue(all(step.certified for step in result.evidence.trajectories))
        self.assertTrue(
            all(
                next(action.value for action in step.actions if action.well == "P1")
                == 140.0
                for step in result.evidence.trajectories
            )
        )
        self.assertEqual(result.evidence.schedule_sha256, result.schedule.sha256)
        self.assertEqual(result.evidence.final_economics.npv_million_rub, 130.0)
        self.assertEqual(
            ScheduleCompiler().parse(case, result.schedule.text), result.schedule.actions
        )

    def test_run_step_is_order_independent_and_deduplicates_before_gdm(self) -> None:
        case = _case(date(2014, 1, 1))
        state = State(case.case_id, case.start, "restart-0", ())

        class RecordingBackend(DeterministicGdmBackend):
            def __init__(self) -> None:
                self.production_targets: list[float] = []

            def run_from_restart(self, case, state, actions):  # type: ignore[no-untyped-def]
                self.production_targets.append(
                    next(action.value for action in actions if action.well == "P1")
                )
                return super().run_from_restart(case, state, actions)

        backend = RecordingBackend()
        result = MonthlyMPC(backend).run_step(
            case,
            state,
            (_actions(case.start, 140.0), _actions(case.start, 100.0), _actions(case.start, 140.0)),
        )

        self.assertEqual(backend.production_targets, [100.0, 140.0])
        self.assertEqual(
            next(action.value for action in result.trajectory.actions if action.well == "P1"),
            140.0,
        )

    def test_run_step_rejects_stale_restart_lineage(self) -> None:
        case = _case(date(2014, 1, 1))
        state = State(case.case_id, case.start, "restart-0", ())

        class StaleRestartBackend(DeterministicGdmBackend):
            def run_from_restart(self, case, state, actions):  # type: ignore[no-untyped-def]
                result = super().run_from_restart(case, state, actions)
                next_state = replace(result.trajectory.next_state, restart_ref=state.restart_ref)
                return GdmResult(
                    replace(result.trajectory, next_state=next_state), result.economics
                )

        with self.assertRaisesRegex(CertificationError, "reused the input restart"):
            MonthlyMPC(StaleRestartBackend()).run_step(
                case, state, (_actions(case.start, 100.0),)
            )

    def test_monthly_mpc_fails_closed_when_backend_does_not_certify(self) -> None:
        case = _case(date(2014, 1, 1))
        state = State(case.case_id, case.start, "restart-0", ())

        class RejectingBackend(DeterministicGdmBackend):
            def run_from_restart(self, case, state, actions):  # type: ignore[no-untyped-def]
                result = super().run_from_restart(case, state, actions)
                trajectory = replace(result.trajectory, certified=False)
                return GdmResult(trajectory, result.economics)

        with self.assertRaisesRegex(CertificationError, "did not certify"):
            MonthlyMPC(RejectingBackend()).run(
                case, state, lambda current: (_actions(current.month, 100.0),)
            )


if __name__ == "__main__":
    unittest.main()
