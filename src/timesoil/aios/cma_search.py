"""CMA-ES over the unit box with feasibility-first ranking (design 20260911, section 3.3).

The searcher never touches physics: it calls an injected ``evaluate`` on whole
generations and ranks the returned :class:`Evaluation` records. Feasible candidates
are always ranked above infeasible ones, so a constraint is never traded for NPV.

Determinism: with a fixed ``seed`` and a deterministic ``evaluate`` and ``clock``
two runs produce identical traces. ``clock`` is called exactly once per generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
import time
from typing import Any, Callable, Sequence

import cma
import numpy as np
from scipy.stats import qmc

_VIOLATION_FLOOR = 1e-12
_SIGMA0 = 0.25


@dataclass(frozen=True)
class Evaluation:
    """One scored candidate. ``violation`` is the sum of normalised breaches, 0 if feasible."""

    x: np.ndarray
    feasible: bool
    npv: float | None = None
    violation: float = 0.0
    candidate_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Elite:
    """A feasible candidate kept for warm starts, LLM injections and the seal."""

    x: np.ndarray
    npv: float
    candidate_id: int | None
    generation: int


@dataclass(frozen=True)
class SearchResult:
    elite: list[Elite]
    trace: list[dict[str, Any]]
    evaluations: int
    stop_reason: str


def _check(evaluations: Sequence[Evaluation], expected: int) -> list[Evaluation]:
    if len(evaluations) != expected:
        raise ValueError(f'evaluate returned {len(evaluations)} records for {expected} points')
    for record in evaluations:
        if record.feasible:
            if record.npv is None or not np.isfinite(float(record.npv)):
                raise ValueError('a feasible evaluation must carry a finite npv')
        else:
            value = float(record.violation)
            if not np.isfinite(value) or value < 0:
                raise ValueError('violation must be finite and nonnegative')
    return list(evaluations)


def _fitness(evaluations: Sequence[Evaluation], ceiling: float | None) -> tuple[list[float], float | None]:
    """Monotone scalar for ``cma.tell``: every feasible point sorts below every infeasible one."""
    feasible = [-float(record.npv) for record in evaluations if record.feasible]
    if feasible:
        best = max(feasible)
        ceiling = best if ceiling is None else max(ceiling, best)
    base = 0.0 if ceiling is None else ceiling
    values = [-float(record.npv) if record.feasible
              else base + max(float(record.violation), _VIOLATION_FLOOR)
              for record in evaluations]
    return values, ceiling


def _keep_elite(elite: list[Elite], evaluations: Sequence[Evaluation], generation: int,
                limit: int = 8) -> list[Elite]:
    merged = list(elite)
    merged.extend(Elite(np.asarray(record.x, dtype=float), float(record.npv),
                        record.candidate_id, generation)
                  for record in evaluations if record.feasible)
    merged.sort(key=lambda item: (-item.npv, item.generation, item.candidate_id is None,
                                  item.candidate_id or 0))
    return merged[:limit]


def _generation_row(index: int, evaluations: Sequence[Evaluation], sigma: float,
                    injections: int, total: int) -> dict[str, Any]:
    npvs = sorted(float(record.npv) for record in evaluations if record.feasible)
    return {'generation': index,
            'best_npv': npvs[-1] if npvs else None,
            'median_npv': median(npvs) if npvs else None,
            'feasible_share': len(npvs) / len(evaluations) if evaluations else 0.0,
            'sigma': float(sigma),
            'injections': injections,
            'evaluations': total}


def run_cma_search(evaluate: Callable[[list[np.ndarray]], list[Evaluation]], dim: int,
                   x0: np.ndarray | None, seed: int, popsize: int, wall_clock_seconds: float,
                   sobol_seeds: int = 32,
                   inject: Callable[[Elite], list[np.ndarray]] | None = None,
                   inject_every: int = 8, max_injections: int = 3,
                   max_generations: int | None = None,
                   clock: Callable[[], float] = time.monotonic) -> SearchResult:
    """Search the unit box under a wall clock, returning the elite and a per-generation trace."""
    if dim < 1 or popsize < 2 or sobol_seeds < 0 or inject_every < 1 or max_injections < 0:
        raise ValueError('run_cma_search got an invalid budget')

    start = clock()
    points: list[np.ndarray] = []
    if x0 is not None:
        seed_point = np.clip(np.asarray(x0, dtype=float).reshape(-1), 0.0, 1.0)
        if seed_point.size != dim:
            raise ValueError(f'x0 must have {dim} coordinates, got {seed_point.size}')
        points.append(seed_point)
    if sobol_seeds:
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        points.extend(np.asarray(row, dtype=float) for row in sampler.random(sobol_seeds))
    if not points:
        points.append(np.full(dim, 0.5))

    ceiling: float | None = None
    elite: list[Elite] = []
    trace: list[dict[str, Any]] = []
    total = 0

    seeded = _check(evaluate(points), len(points))
    total += len(seeded)
    _, ceiling = _fitness(seeded, ceiling)
    elite = _keep_elite(elite, seeded, 0)
    trace.append(_generation_row(0, seeded, _SIGMA0, 0, total))

    origin = elite[0].x if elite else points[0]
    strategy = cma.CMAEvolutionStrategy(list(np.asarray(origin, dtype=float)), _SIGMA0,
                                        {'seed': int(seed), 'bounds': [0.0, 1.0],
                                         'popsize': int(popsize), 'verbose': -9,
                                         'verb_log': 0, 'verb_disp': 0, 'verb_filenameprefix': ''})

    injections = 0
    generation = 0
    stop_reason = 'wall_clock'
    while clock() - start < wall_clock_seconds:
        if max_generations is not None and generation >= max_generations:
            stop_reason = 'max_generations'
            break
        if strategy.stop():
            stop_reason = 'cma_stop'
            break
        generation += 1
        asked = [np.asarray(point, dtype=float) for point in strategy.ask()]
        scored = _check(evaluate(asked), len(asked))
        total += len(scored)
        values, ceiling = _fitness(scored, ceiling)
        strategy.tell(asked, values)
        elite = _keep_elite(elite, scored, generation)
        trace.append(_generation_row(generation, scored, strategy.sigma, injections, total))

        if inject is None or not elite or injections >= max_injections or generation % inject_every:
            continue
        proposed = [np.clip(np.asarray(point, dtype=float).reshape(-1), 0.0, 1.0)
                    for point in inject(elite[0])]
        if not proposed:
            continue
        if any(point.size != dim for point in proposed):
            raise ValueError('an injected point has the wrong dimension')
        injected = _check(evaluate(proposed), len(proposed))
        total += len(injected)
        _, ceiling = _fitness(injected, ceiling)
        elite = _keep_elite(elite, injected, generation)
        # Default (non-forced) injection: cma's documented path, the proposal enters as a
        # direction from the mean. The raw proposals are scored above, so the elite keeps them
        # verbatim even when cma trims the direction.
        strategy.inject([list(point) for point in proposed])
        injections += 1
        trace.append(_generation_row(generation, injected, strategy.sigma, injections, total)
                     | {'injected': True})

    return SearchResult(elite=elite, trace=trace, evaluations=total, stop_reason=stop_reason)
