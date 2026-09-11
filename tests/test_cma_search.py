"""Feasibility-first CMA-ES: convergence, reproducibility and LLM injections."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from timesoil.aios.cma_search import Elite, Evaluation, run_cma_search

TARGET = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
# Unconstrained optimum (0.8, 0.8, ...) breaks x0 + x1 <= 1; the projection is (0.5, 0.5, 0.8, ...).
CONSTRAINED = np.array([0.5, 0.5, 0.8, 0.8, 0.8])


def quadratic(points: list[np.ndarray]) -> list[Evaluation]:
    out = []
    for index, point in enumerate(points):
        breach = float(point[0] + point[1] - 1.0)
        npv = -float(np.sum((point - TARGET) ** 2))
        out.append(Evaluation(x=point, feasible=breach <= 0.0, npv=npv,
                              violation=max(breach, 0.0), candidate_id=index))
    return out


def ticking_clock(step: float = 1.0) -> Callable[[], float]:
    state = {'now': 0.0}

    def clock() -> float:
        state['now'] += step
        return state['now']

    return clock


def search(**overrides):
    kwargs = dict(evaluate=quadratic, dim=5, x0=None, seed=20260909, popsize=8,
                  wall_clock_seconds=40.0, sobol_seeds=16, clock=ticking_clock())
    kwargs.update(overrides)
    return run_cma_search(**kwargs)


def test_converges_to_the_constrained_optimum_within_the_budget() -> None:
    result = search()
    assert result.evaluations <= 400
    assert result.stop_reason in ('wall_clock', 'cma_stop')
    best = result.elite[0]
    assert best.x[0] + best.x[1] <= 1.0 + 1e-9
    assert best.x == pytest.approx(CONSTRAINED, abs=0.05)
    assert best.npv == pytest.approx(-0.18, abs=0.01)
    assert len(result.elite) == 8
    assert [item.npv for item in result.elite] == sorted((item.npv for item in result.elite),
                                                         reverse=True)


def test_same_seed_and_fake_clock_reproduce_the_trace() -> None:
    first, second = search(), search()
    assert first.trace == second.trace
    assert first.evaluations == second.evaluations
    assert [item.x.tolist() for item in first.elite] == [item.x.tolist() for item in second.elite]
    assert search(seed=7).trace != first.trace


def test_trace_reports_generation_statistics() -> None:
    result = search(wall_clock_seconds=4.0)
    assert [row['generation'] for row in result.trace] == [0, 1, 2, 3]
    seeded = result.trace[0]
    assert seeded['evaluations'] == 16 and 0.0 < seeded['feasible_share'] <= 1.0
    assert seeded['median_npv'] <= seeded['best_npv']
    assert all(row['sigma'] > 0 for row in result.trace)


def test_infeasible_points_rank_below_every_feasible_point() -> None:
    def all_infeasible(points: list[np.ndarray]) -> list[Evaluation]:
        return [Evaluation(x=point, feasible=False, violation=1.0 + float(point[0]))
                for point in points]

    result = search(evaluate=all_infeasible, wall_clock_seconds=3.0)
    assert result.elite == []
    assert all(row['best_npv'] is None and row['feasible_share'] == 0.0 for row in result.trace)


def test_injections_run_on_schedule_and_are_evaluated() -> None:
    calls: list[Elite] = []

    def inject(elite: Elite) -> list[np.ndarray]:
        calls.append(elite)
        return [np.full(5, 0.4), np.full(5, 0.45)]

    result = search(wall_clock_seconds=10.0, inject=inject, inject_every=2, max_injections=3)
    injected = [row for row in result.trace if row.get('injected')]
    assert len(calls) == 3 and len(injected) == 3
    assert [row['generation'] for row in injected] == [2, 4, 6]
    assert all(isinstance(item, Elite) and item.npv is not None for item in calls)
    # 16 seeds + 9 generations of 8 (the clock ticks once per generation) + 3 injections of 2.
    assert result.evaluations == 16 + 9 * 8 + 3 * 2
    assert search(wall_clock_seconds=10.0).evaluations == 16 + 9 * 8


def test_budget_and_contract_guards() -> None:
    with pytest.raises(ValueError, match='invalid budget'):
        search(popsize=1)
    with pytest.raises(ValueError, match='5 coordinates'):
        search(x0=np.full(3, 0.5))
    with pytest.raises(ValueError, match='finite npv'):
        search(evaluate=lambda points: [Evaluation(x=p, feasible=True) for p in points])
    with pytest.raises(ValueError, match='records for'):
        search(evaluate=lambda points: quadratic(points)[:-1])
    with pytest.raises(ValueError, match='nonnegative'):
        search(evaluate=lambda points: [Evaluation(x=p, feasible=False, violation=-1.0)
                                        for p in points])


def test_x0_is_evaluated_before_the_loop() -> None:
    seen: list[int] = []

    def counting(points: list[np.ndarray]) -> list[Evaluation]:
        seen.append(len(points))
        return quadratic(points)

    result = search(evaluate=counting, x0=np.full(5, 0.5), sobol_seeds=8, wall_clock_seconds=2.0)
    assert seen[0] == 9 and result.trace[0]['evaluations'] == 9
