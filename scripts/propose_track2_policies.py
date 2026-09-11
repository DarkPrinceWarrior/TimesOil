"""Qwen field policies ranked by nine-target TimesFM forecast CHDD, sealed before one final OPM.

Forecast CHDD is a screening estimate, never submitted CHDD. Every emitted request
retains source completions and is verified by the production full-cycle command
over the complete management period.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from math import log
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd

from timesoil.aios import cma_search, planning
from timesoil.aios.agents import AgentRole, AgentWorkflow, ToolDefinition, ToolRegistry
from timesoil.aios.llm import ExternalQwenClient, LLMConfig
from timesoil.aios.policy_space import PolicySpace, block_map
from timesoil.aios.track2 import load_trajectory_dataset
from timesoil.aios.workflow import (CycleError, CycleRequest, _controls,
    _source_control_inventory, _validate_source_well_scope)
from timesoil.aios.opm import OpmFlowRunner
from timesoil.aios.schedule_overlay import apply_schedule_overlay
from timesoil.aios.operating_constraints import check_controls, failures, own_control_constraints, parse_constraints
from timesoil.aios.case_profile import load_case_profile
from timesoil.aios.economics import CHDDEconomicsAdapter, opm_management_rows

SEARCH_SEED = 20260909
GRID = ((1, 1), (1.25, 1), (1, .8), (1, 1.2), (.8, 1), (1.5, 1), (2, 1), (3, 1))
BASE_POLICY = {'producer_scale': 1.0, 'injector_scale': 1.0, 'shut_wells': [], 'well_scales': []}


def agent_candidate_context(candidates):
    """Keep all policy scores and one full-well forecast without repeating model metadata."""
    fields = ('id', 'policy', 'estimated_oil_t', 'estimated_liquid_t', 'planned_injection_m3',
              'predicted_injection_m3', 'forecast_chdd_m', 'forecast_eligible', 'controls_sha256')
    summaries = [{key: row[key] for key in fields if key in row} for row in candidates]
    for summary, row in zip(summaries, candidates, strict=True):
        if 'forecast_constraint_violations' in row:
            summary.update(forecast_constraint_violations=row['forecast_constraint_violations'][:3],
                           forecast_constraint_violation_count=len(row['forecast_constraint_violations']))
    eligible = [row for row in candidates if row.get('forecast_eligible') is True]
    detailed = max(eligible, key=lambda row: row['forecast_chdd_m']) if eligible else candidates[-1]
    return {'candidates': summaries, 'forecast_detail': {'candidate_id': detailed['id'],
        'by_well': detailed['forecast_by_well'],
        'scope': 'Forecast detail for analysis; every complete candidate forecast remains in its persisted artifact.'}}


async def plan_with_control_repair(workflow, context, candidates, attempted, output, index):
    """Return no plan after two invalid proposals; never accept a partial round."""
    before = len(candidates)
    for attempt in range(2):
        try:
            return await workflow.run_plan(context)
        except CycleError as error:
            rejected = {"error": str(error), "attempt": attempt,
                        "policy": attempted[-1] if attempted else None,
                        "accepted_candidates_added": len(candidates) - before}
            with (output / f"rejected-plan-{index:02d}-{attempt}.json").open('x') as stream:
                json.dump(rejected, stream, indent=2)
            if len(candidates) != before:
                raise
            if attempt:
                return None
            context = {**context, "previous_invalid_proposal": rejected,
                       "repair_instruction": "Correct the rejected controls and call propose_policy exactly once. Respect each well's first_source_control_month, original BHP bounds and conversion permission. No invalid controls were accepted. Do not change or bypass these constraints."}


def policy_controls(controls, policy):
    wells = {a["well"] for a in controls}
    shut = set(policy["shut_wells"])
    scales = {}
    for item in policy["well_scales"]:
        if item["well"] in scales:
            raise ValueError("duplicate well scale")
        scales[item["well"]] = item["scale"]
    if not shut <= wells or not scales.keys() <= wells:
        raise ValueError("unknown well in policy")
    values = [policy["producer_scale"], policy["injector_scale"], *scales.values()]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("policy scales must be finite and nonnegative")
    producer_bhp = policy.get('producer_bhp_add', 0)
    injector_bhp = policy.get('injector_bhp_factor', 1)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in (producer_bhp, injector_bhp)) or producer_bhp < 0 or not 0 < injector_bhp <= 1):
        raise ValueError('BHP changes must be finite and cannot relax the reference limits')
    output = []
    for action in controls:
        item = dict(action)
        if item["well"] in shut:
            item.update(status="SHUT", value=0.0)
        elif item["status"] == "OPEN":
            factor = scales.get(item["well"], policy["injector_scale" if item["role"] == "injector" else "producer_scale"])
            item["value"] *= factor
            if item["target"] == "LRAT":
                item["value"] = min(item["value"], 500.0)
            if producer_bhp or injector_bhp != 1:
                if not item.get('bhp_limit', 0) > 0:
                    raise ValueError('global BHP changes require explicit known reference limits')
                item['bhp_limit'] = (item['bhp_limit'] * injector_bhp if item['role'] == 'injector'
                                     else item['bhp_limit'] + producer_bhp)
        output.append(item)
    by_key = {(a['month'], a['well']): a for a in output}
    months = sorted({a['month'] for a in output})
    edited = set()
    for update in policy.get('well_updates', []):
        required = {'well', 'start', 'end'}
        fields = {'role', 'status', 'target', 'value', 'bhp_limit'}
        if (not isinstance(update, dict) or not required <= update.keys()
                or update.keys() - required - fields or not fields.intersection(update)):
            raise ValueError('well update fields are invalid')
        well, start, end = update['well'], update['start'], update['end']
        if well not in wells or start not in months or end not in months or start > end:
            raise ValueError('well update is outside the known well/month grid')
        changes = {key: value for key, value in update.items() if key in fields}
        for month in months:
            if not start <= month <= end:
                continue
            key = month, well
            if key in edited:
                raise ValueError('overlapping well updates')
            edited.add(key)
            action = by_key[key]
            if changes.get('role', action['role']) != action['role']:
                if not {'target', 'value', 'bhp_limit'} <= changes.keys():
                    raise ValueError('conversion requires explicit target, rate and BHP limit')
            action.update(changes)
            if action['status'] == 'SHUT':
                action['value'] = 0.0
    repair = policy.get('monthly_repair') or {}
    if repair:
        if not isinstance(repair, dict) or repair.keys() - {'liquid', 'injection'}:
            raise ValueError('monthly repair accepts only liquid and injection factors')
        factors = {}
        for kind in ('liquid', 'injection'):
            table = repair.get(kind, {})
            if not isinstance(table, dict):
                raise ValueError('monthly repair factors must map a control month to a factor')
            for month, factor in table.items():
                if month not in months:
                    raise ValueError('monthly repair factor is outside the control grid')
                if (isinstance(factor, bool) or not isinstance(factor, (int, float))
                        or not np.isfinite(factor) or not 0 < factor <= 1):
                    raise ValueError('monthly repair may only scale a rate down, by a factor in (0, 1]')
            factors[kind] = table
        for action in output:
            if action['status'] != 'OPEN':
                continue
            if action['target'] == 'ORAT' and factors['liquid'].get(action['month'], 1) < 1:
                raise ValueError('liquid repair cannot scale an ORAT-controlled producer')
            kind = {'LRAT': 'liquid', 'WRAT': 'injection'}.get(action['target'])
            if kind:
                action['value'] *= factors[kind].get(action['month'], 1)
    _controls(output)
    roles = {}
    for action in sorted(output, key=lambda a: (a['month'], a['well'])):
        if action['target'] == 'LRAT' and action['value'] > 500:
            raise ValueError('liquid target exceeds 500 m3/day')
        if roles.get(action['well']) == 'injector' and action['role'] == 'producer':
            raise ValueError('reverse conversion is not permitted')
        roles[action['well']] = action['role']
    return output


def repair_policy(policy, forecast_aggregates, profile):
    """Scale each month's LRAT/WRAT down until the pass-1 forecast respects K1, K2, K4, K5.

    ``forecast_aggregates`` is ``derived_field_vectors(...)['field']`` of the first
    forecast pass; ``profile`` is a validated ``CaseProfile``. Design §1.3:

        s_L(m) = min{1, (1-eps_L) * liquid_cap / L(m)}
        s_I(m) = min{1, (1-eps_I) * injection_cap / I(m), phi * W(m) / I(m),
                     vrr_max * Vp3(m) / Vi3(m)}

    Returns ``(policy with monthly_repair, factors)``; both factors are <= 1, so a
    repaired schedule can only withdraw or inject less than the proposed one.
    """
    margins = profile.selection_margins
    months = list(forecast_aggregates['months'])
    series = [list(forecast_aggregates[key]) for key in
              ('liquid_m3d', 'injection_m3d', 'water_m3d', 'reservoir_production_m3', 'reservoir_injection_m3')]
    if any(len(values) != len(months) for values in series):
        raise ValueError('forecast aggregates must cover exactly the forecast months')
    liquid, injection, water, produced, injected = series
    window = int(profile.vrr['window_months'])
    liquid_cap = (1 - margins['eps_liquid']) * profile.liquid_cap_m3d
    injection_cap = (1 - margins['eps_injection']) * profile.injection_cap_m3d
    # The produced-water term exists only when the case restricts injection to produced
    # water (a finite deficit rule); an external supply is bounded by the injection cap alone.
    produced_water_rule = profile.water_balance.get('deficit_m3') is not None
    scales = {'liquid': {}, 'injection': {}}
    binding = {}
    injected_repaired = list(injected)  # earlier months carry the factors already decided
    for index, month in enumerate(months):
        if liquid[index] > 0 and liquid_cap < liquid[index]:
            scales['liquid'][month] = liquid_cap / liquid[index]
        rate = injection[index]
        if rate <= 0:
            continue
        terms = {'injection_cap': injection_cap / rate}
        if produced_water_rule:
            terms['produced_water'] = margins['phi'] * water[index] / rate
        # VRR upper bound solved for the current month: the earlier months of the window are
        # already repaired, so only this month's injection is the unknown.
        low = max(0, index - window + 1)
        earlier = sum(injected_repaired[low:index])
        if injected[index] > 0:
            allowed = profile.vrr['max'] * sum(produced[low:index + 1]) - earlier
            terms['vrr_upper'] = max(0.0, allowed / injected[index])
        limiting = min(terms, key=lambda key: terms[key])
        if terms[limiting] < 1:
            scales['injection'][month] = terms[limiting]
            binding[month] = limiting
            injected_repaired[index] = injected[index] * terms[limiting]
    factors = {'liquid': scales['liquid'], 'injection': scales['injection'], 'binding': binding,
               'applied': bool(scales['liquid'] or scales['injection']),
               'eps_liquid': margins['eps_liquid'], 'eps_injection': margins['eps_injection'],
               'phi': margins['phi'], 'vrr_max': profile.vrr['max'], 'window_months': window,
               'case_profile_sha256': profile.sha256}
    return {**policy, 'monthly_repair': {'liquid': scales['liquid'], 'injection': scales['injection']}}, factors


def rejected_candidate(policy, attempt, error, *, case_profile_sha256):
    """A refusing guard is a result: recorded with its reason, never joining the sealed ledger.

    The record deliberately has no forecast, no CHDD and no id, so
    ``track2_final_selection.best_forecast`` rejects it if it ever reaches
    ``candidates.json``; rejections live in ``rejected-candidates.json`` instead.
    """
    return {'id': None, 'attempt': attempt, 'policy': policy, 'forecast_eligible': False,
            'rejection': str(error), 'case_profile_sha256': case_profile_sha256}


def baseline_bhp_controls(controls, trajectory):
    """Fill missing bounds from same-month planned actions, never observed pressures."""
    if trajectory.actions.shape[-1] != 4:
        raise ValueError('authenticated baseline BHP action channel required')
    output = [dict(action) for action in controls]
    for action in output:
        if action['status'] != 'OPEN' or action.get('bhp_limit') is not None:
            continue
        row = trajectory.actions[trajectory.dates.get_loc(pd.Timestamp(action['month'])),
                                 trajectory.well_ids.index(action['well'])]
        if (row[1] == 2) != (action['role'] == 'injector'):
            raise ValueError('role conversion requires an explicit BHP bound')
        bound = float(row[3])
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError('open well has no positive planned baseline BHP bound')
        action['bhp_limit'] = bound
    return output


def reject_duplicate_controls(controls_sha256, candidates):
    if any(row['controls_sha256'] == controls_sha256 for row in candidates):
        raise CycleError('policy repeats an already evaluated control schedule; propose different controls')


def violation_score(record):
    """Sum of normalised hard-limit deficits behind one candidate; 0 only when it is eligible.

    ``cma_search`` ranks every infeasible point by this scalar, so a rejected
    candidate that never reached the gates still has to score strictly above zero.
    """
    if record.get('forecast_eligible') is True:
        return 0.0
    total = 0.0
    for verdict in record.get('forecast_constraint_verdicts', ()):
        if verdict.get('ok') is not False or verdict.get('status') != 'hard':
            continue
        margin, worst = verdict.get('margin'), verdict.get('worst_value')
        if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not np.isfinite(margin) or margin >= 0:
            total += 1.0
            continue
        scale = abs(float(worst)) if isinstance(worst, (int, float)) and not isinstance(worst, bool) \
            and np.isfinite(worst) and worst else 1.0
        total += -float(margin) / max(scale, 1e-9)
    return total or 1.0


def representative_rates(controls):
    """First OPEN LRAT/WRAT value per well: the absolute-rate reference ``PolicySpace`` needs."""
    rates = {}
    for action in sorted(controls, key=lambda item: (item['well'], item['month'])):
        if (action['well'] not in rates and action['status'] == 'OPEN'
                and action['target'] in ('LRAT', 'WRAT')):
            rates[action['well']] = float(action['value'])
    return rates


def origin_well_state(history, month, densities, pressures):
    """Per-well surface rates and water cut of the last observed month, tonnes converted with the export densities.

    Only the ``month`` row of the observed CHDD history is read, so no post-origin
    observation can reach the search.
    """
    days = pd.Timestamp(month).days_in_month
    state = {}
    for row in history:
        well = str(row['well'])
        if str(row['DATA']) != month or well not in densities:
            continue
        oil_tonnes, liquid_tonnes = float(row['WOMT_Diff']), float(row['WLPT_Diff'])
        oil = oil_tonnes / (densities[well]['oil_kg_m3'] / 1000.0)
        water = max(liquid_tonnes - oil_tonnes, 0.0) / (densities[well]['water_kg_m3'] / 1000.0)
        liquid = oil + water
        state[well] = {'oil_tpd': oil_tonnes / days, 'liquid_m3d': liquid / days,
                       'water_m3d': water / days, 'injection_m3d': float(row['WWIT_Diff']) / days,
                       'water_cut': water / liquid if liquid > 0 else 0.0,
                       'wbp9': float(pressures.get(well, 0.0))}
    return state


def run_search(args, context):
    """Run the configured search and return the ``proposal-receipt.json`` fragment it produced.

    Everything physical is injected through ``context``: ``evaluate(policy)`` for one
    candidate, ``evaluate_generation(policies)`` for a whole generation,
    ``reject(policy, error)``, the two ledgers, the ``PolicySpace`` and the two LLM
    hooks. Nothing here calls OPM, TimesFM or the GPU, so the loop is testable on its own.
    """
    evaluate, candidates, rejections = context['evaluate'], context['candidates'], context['rejections']
    blocks_sha256 = context.get('blocks_sha256')
    if args.search == 'grid':
        for production, injection in ([(1, 1)] if args.skip_grid else GRID):
            print(json.dumps(evaluate({**BASE_POLICY, 'producer_scale': production,
                                       'injector_scale': injection})), flush=True)
        proposed_ids, skipped = context['agent_rounds']()
        return {'agent_proposal_ids': proposed_ids, 'skipped_invalid_rounds': skipped,
                'search': {'mode': 'grid', 'seconds': None, 'blocks_sha256': blocks_sha256,
                           'evaluations': len(candidates) + len(rejections),
                           'generations': 0, 'injections': 0, 'llm_candidate_ids': []}}
    space, output = context['space'], context['output']
    # The unchanged incumbent is candidate 0 in both modes: the seal reads it as the baseline.
    print(json.dumps(evaluate(dict(BASE_POLICY))), flush=True)
    neutral = space.encode_seed({'field_producer': 1.0, 'block_producer': 1.0,
                                 'watercut_shut': 0.995, 'producer_bhp_add': 0.0})
    genes_by_point, llm_ids = {}, []

    def register(genes, hint, fallback):
        """Keep the genes that belong to a point; the searcher only carries the vector."""
        values = [float(value) for value in hint]
        point = np.clip(np.asarray(values if len(values) == space.dim else fallback, dtype=float), 0.0, 1.0)
        genes_by_point[point.tobytes()] = genes
        return point

    def score(points):
        policies, refused = [], {}
        for index, point in enumerate(points):
            genes = genes_by_point.get(np.asarray(point, dtype=float).tobytes())
            try:
                policies.append(space.decode(point, genes))
            except (ValueError, CycleError) as error:
                refused[index] = context['reject'](
                    {'x': [float(value) for value in np.asarray(point, dtype=float).reshape(-1)],
                     'genes': genes}, error)
                policies.append(None)
        scored = iter(context['evaluate_generation']([item for item in policies if item is not None]))
        records = [refused[index] if policy is None else next(scored)
                   for index, policy in enumerate(policies)]
        evaluations = []
        for point, record in zip(points, records, strict=True):
            if record.get('id') is not None and np.asarray(point, dtype=float).tobytes() in genes_by_point:
                llm_ids.append(record['id'])
            evaluations.append(cma_search.Evaluation(
                x=np.asarray(point, dtype=float), feasible=record.get('forecast_eligible') is True,
                npv=record.get('forecast_chdd_m'), violation=violation_score(record),
                candidate_id=record.get('id')))
        return evaluations

    if args.llm_round0:
        seeds = []
        try:
            seeds = context['llm_round0']()
        except Exception as error:  # The LLM may propose, never stop, the numeric search.
            context['reject']({'llm_round': 'round0'}, error)
        points = []
        for genes, hint in seeds:
            if len(hint) != space.dim and not any(genes.values()):
                continue  # No usable box hint and no genes: nothing to seed.
            point = register(genes, hint, neutral)
            if not any(np.array_equal(point, other) for other in points):
                points.append(point)
        if points:
            score(points)
    seeded = len(candidates) + len(rejections)

    def inject(elite):
        digest = [{'id': elite.candidate_id, 'npv_m': elite.npv,
                   'x': [round(float(value), 6) for value in elite.x]}]
        try:
            proposals = context['llm_injection'](digest)
        except Exception as error:  # As in round 0: a rejection entry, never a dead search.
            context['reject']({'llm_round': 'injection'}, error)
            return []
        return [register(genes, hint, elite.x) for genes, hint in proposals]

    result = cma_search.run_cma_search(
        score, dim=space.dim, x0=neutral, seed=SEARCH_SEED,
        popsize=context.get('popsize', 4 + int(3 * log(space.dim))),
        wall_clock_seconds=args.search_seconds, sobol_seeds=context.get('sobol_seeds', 32),
        inject=inject if args.llm_round0 else None, inject_every=context.get('inject_every', 8),
        max_injections=args.injections, clock=context.get('clock', time.monotonic))
    (output / 'search_trace.json').write_text(json.dumps(
        {'seed': SEARCH_SEED, 'dim': space.dim, 'popsize': context.get('popsize'),
         'stop_reason': result.stop_reason, 'seed_evaluations': seeded, 'generations': result.trace},
        ensure_ascii=False, indent=2, sort_keys=True))
    (output / 'elite.json').write_text(json.dumps(
        [{'candidate_id': item.candidate_id, 'forecast_chdd_m': item.npv, 'generation': item.generation,
          'x': [round(float(value), 12) for value in item.x], 'parameters': space.describe(item.x)}
         for item in result.elite], ensure_ascii=False, indent=2, sort_keys=True))
    return {'agent_proposal_ids': [], 'skipped_invalid_rounds': [],
            'search': {'mode': 'cma', 'seconds': args.search_seconds, 'blocks_sha256': blocks_sha256,
                       'evaluations': seeded + result.evaluations,
                       'generations': max((row['generation'] for row in result.trace), default=0),
                       'injections': sum(1 for row in result.trace if row.get('injected')),
                       'stop_reason': result.stop_reason,
                       'llm_candidate_ids': sorted(set(llm_ids))}}


def self_check():
    reject_duplicate_controls('new', [{'controls_sha256': 'old'}])
    try:
        reject_duplicate_controls('same', [{'controls_sha256': 'same'}])
    except CycleError:
        pass
    else:
        raise AssertionError('duplicate physical controls accepted as a new hypothesis')
    controls = [dict(month="2007-01-01", well="P", role="producer", status="OPEN", target="LRAT", value=300),
                dict(month="2007-01-01", well="I", role="injector", status="OPEN", target="WRAT", value=100),
                dict(month="2007-01-01", well="F", role="producer", status="SHUT", target="LRAT", value=0)]
    policy = dict(producer_scale=2, injector_scale=.8, shut_wells=[], well_scales=[])
    result = policy_controls(controls, policy)
    assert [r["value"] for r in result] == [500, 80, 0] and result[-1]["status"] == "SHUT"
    assert controls[0]["value"] == 300
    pressured = [dict(a, bhp_limit=300 if a['role'] == 'injector' else 50) for a in controls]
    changed = policy_controls(pressured, {**policy, 'producer_bhp_add': 5, 'injector_bhp_factor': .9})
    assert [a['bhp_limit'] for a in changed] == [55, 270, 50]
    assert pressured[0]['bhp_limit'] == 50
    assert policy_controls(controls, {**policy, "shut_wells": ["I"]})[1]["status"] == "SHUT"
    try:
        policy_controls(controls, {**policy, "well_scales": [{"well": "unknown", "scale": 1}]})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown well accepted")
    two_months = controls + [{**a, 'month': '2007-02-01'} for a in controls]
    from types import SimpleNamespace
    planned = SimpleNamespace(dates=pd.to_datetime(['2007-01-01', '2007-02-01']),
        well_ids=('P', 'I', 'F'), actions=np.array([
            [[300, 1, 1, 50], [100, 2, 1, 300], [0, 1, 0, 0]],
            [[300, 1, 1, 60], [100, 2, 1, 280], [0, 1, 0, 0]]], dtype=float))
    filled = baseline_bhp_controls(two_months, planned)
    assert [a.get('bhp_limit') for a in filled] == [50, 300, None, 60, 280, None]
    assert all('bhp_limit' not in a for a in two_months)
    assert baseline_bhp_controls([{**controls[0], 'bhp_limit': 70}], planned)[0]['bhp_limit'] == 70
    adjusted = policy_controls(filled, {**policy, 'producer_bhp_add': 5, 'injector_bhp_factor': .9})
    assert [a.get('bhp_limit') for a in adjusted] == [55, 270, None, 65, 252, None]
    for bad in ([{**controls[0], 'role': 'injector'}], [{**controls[2], 'status': 'OPEN'}]):
        try:
            baseline_bhp_controls(bad, planned)
        except ValueError:
            pass
        else:
            raise AssertionError('unproven BHP bound accepted')
    update = dict(well='P', start='2007-01-01', end='2007-02-01',
                  role='injector', target='WRAT', value=80, bhp_limit=280)
    converted = policy_controls(two_months, {**policy, 'well_updates': [update]})
    assert all(a['role'] == 'injector' and a['value'] == 80 and a['bhp_limit'] == 280
               for a in converted if a['well'] == 'P')
    stopped = policy_controls(two_months, {**policy, 'well_updates': [
        dict(well='P', start='2007-02-01', end='2007-02-01', status='SHUT')]})
    assert stopped[0]['status'] == 'OPEN' and stopped[3]['value'] == 0
    assert two_months[3]['status'] == 'OPEN'
    for updates in ([update, update], [{**update, 'end': '2007-01-01'}],
                    [{**update, 'start': '2006-12-01'}],
                    [{k: v for k, v in update.items() if k != 'bhp_limit'}],
                    [{**update, 'value': True}], [{**update, 'bhp_limit': -1}]):
        try:
            policy_controls(two_months, {**policy, 'well_updates': updates})
        except (ValueError, CycleError):
            pass
        else:
            raise AssertionError('invalid timed control update accepted')
    repaired = policy_controls(controls, {**policy, 'monthly_repair': {
        'liquid': {'2007-01-01': .5}, 'injection': {'2007-01-01': .25}}})
    assert [r['value'] for r in repaired] == [250, 20, 0]
    for bad in ({'liquid': {'2007-02-01': .5}}, {'liquid': {'2007-01-01': 0}},
                {'liquid': {'2007-01-01': 1.5}}, {'injection': {'2007-01-01': True}}, {'unknown': {}}):
        try:
            policy_controls(controls, {**policy, 'monthly_repair': bad})
        except ValueError:
            pass
        else:
            raise AssertionError('invalid monthly repair factor accepted')
    oil_rate = [dict(controls[0], target='ORAT', value=10)]
    try:
        policy_controls(oil_rate, {**policy, 'monthly_repair': {'liquid': {'2007-01-01': .5}}})
    except ValueError:
        pass
    else:
        raise AssertionError('liquid repair silently ignored an ORAT producer')
    print('policy rates, dated status/BHP, conversion persistence and rejection checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument('--connectivity', type=Path)
    parser.add_argument('--head', type=Path)
    parser.add_argument('--head-sha256')
    parser.add_argument('--economic-selection', action='store_true',
        help='Rank nine-target forecasts with official economics and seal one choice before OPM')
    parser.add_argument('--head-report', type=Path)
    parser.add_argument('--skip-grid', action='store_true')
    parser.add_argument('--case-profile', type=Path, required=True,
        help='Validated case_constraints.json; its SHA-256 is sealed with the proposal')
    parser.add_argument('--oil-fvf', type=float, default=1.0,
        help='Bo used to convert forecast surface volumes to reservoir volumes for the VRR gate')
    parser.add_argument('--water-fvf', type=float, default=1.0, help='Bw, as for --oil-fvf')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.5,
        help='Share of the GPU reserved for this process; the search refuses to start unless it is free')
    parser.add_argument('--search', choices=('cma', 'grid'), default='cma',
        help='cma: CMA-ES over the block policy space; grid: the 8-point grid plus Qwen tool rounds')
    parser.add_argument('--search-seconds', type=float, default=600.0,
        help='Wall clock budget of the CMA-ES loop, excluding the seed batch')
    parser.add_argument('--blocks', type=Path,
        help='blocks.json from scripts/export_blocks.py; without it the field is one block')
    parser.add_argument('--llm-round0', action=argparse.BooleanOptionalAction, default=True,
        help='Run the block-agent round 0 and the periodic injections (default on)')
    parser.add_argument('--injections', type=int, default=3,
        help='Maximum LLM injections, one every eight generations')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    self_check()
    case_profile = load_case_profile(args.case_profile)
    case_profile_sha256 = case_profile.sha256
    if not 1 <= args.rounds <= 12:
        parser.error("rounds must be in [1, 12]")
    if args.search_seconds <= 0 or not 0 <= args.injections <= 16:
        parser.error('search budget must be positive and injections must be in [0, 16]')
    if not args.economic_selection:
        parser.error('only sealed nine-target economic selection is supported; pass --economic-selection')
    if not (args.head and args.head_sha256 and args.head_report and args.connectivity):
        parser.error('economic selection requires trained weights, their SHA-256, the training report and connectivity')
    from timesfm_geology import MODEL_REVISION, load_frozen_model, reserve_gpu_memory
    request = json.loads(args.request.read_text())
    checked_request = CycleRequest.from_mapping(request)
    with TemporaryDirectory(prefix='timesoil-policy-source-') as temporary:
        prepared = OpmFlowRunner().prepare(checked_request.source, Path(temporary) / 'case',
                                           deck=checked_request.deck)
        source_schedule = (prepared.input_dir / checked_request.schedule_relative_path).read_text()
        source_inventory = _source_control_inventory(
            source_schedule, sorted({a.month for a in checked_request.controls}))
    allow_conversion = checked_request.context.get('constraints', {}).get('allow_conversion_to_injection', False)
    _validate_source_well_scope(checked_request.controls, source_inventory,
                               allow_conversion_to_injection=allow_conversion)
    calculator = CHDDEconomicsAdapter.from_env()
    normative_profile = calculator.normative_profile(
        charge_initial_pump=checked_request.charge_initial_pump
    )
    raw = (args.baseline_run / "canonical/trajectory.csv").read_bytes()
    manifest = json.loads((args.baseline_run / "canonical/manifest.json").read_text())
    if sha256(raw).hexdigest() != manifest["outputs"]["track2_csv"]["sha256"]:
        raise ValueError("baseline trajectory hash mismatch")
    dataset = load_trajectory_dataset(args.baseline_run / 'canonical/trajectory.csv',
        manifest=args.baseline_run / 'canonical/manifest.json')
    if len(dataset) != 1 or not dataset.model_z_identity:
        raise ValueError('one authenticated Model Z baseline is required')
    trajectory = dataset[0]
    pressure_semantics = {
        'forecast_vector': 'WBP9', 'forecast_definition': manifest['conversion']['THP'],
        'bottom_hole_vector': 'WBHP', 'bottom_hole_definition': manifest['conversion']['BHP'],
        'control_bhp_limit': 'Planned lower producer / upper injector BHP bound; not the forecast target.',
        'inactive_zero': 'A zero WBP9 report for an inactive well is not evidence of zero physical reservoir pressure.',
    }
    start = pd.Timestamp(min(a["month"] for a in request["controls"]))
    origin = int(trajectory.dates.get_loc(start))
    horizon, context = checked_request.horizon_months, 128
    if origin < context or origin + horizon >= len(trajectory.dates):
        raise ValueError('verified history and complete forecast horizon required')
    has_bhp = trajectory.actions.shape[-1] == 4
    if has_bhp:
        request['controls'] = baseline_bhp_controls(request['controls'], trajectory)
        checked_request = CycleRequest.from_mapping(request)
    if not has_bhp and any(a.bhp_limit is not None for a in checked_request.controls):
        raise ValueError('BHP screening requires an authenticated baseline with the BHP action channel')
    from timesoil.aios.interwell import WellConnectivity
    connectivity = WellConnectivity.from_dict(json.loads(args.connectivity.read_text()))
    if (connectivity.well_ids != trajectory.well_ids or connectivity.provenance['source_sha256']
            != manifest['provenance']['opm_source_sha256']):
        raise ValueError('geology does not match the verified reservoir and well order')
    well_index = {w: i for i, w in enumerate(trajectory.well_ids)}
    initial_controls = {a["well"]: a for a in request["controls"] if a["month"] == start.date().isoformat()}
    date_index = {d.date().isoformat(): i for i, d in enumerate(trajectory.dates)}
    first_month = min(a.month for a in checked_request.controls)
    last_month = max(a.month for a in checked_request.controls)
    operating_rules = (
        *parse_constraints(checked_request.context.get('operating_constraints', []),
                           wells=trajectory.well_ids, start=first_month, end=last_month),
        *case_profile.operating_rules(wells=trajectory.well_ids, start=first_month, end=last_month))
    from timesfm_economics import (derived_field_vectors, economic_constraint_verdicts,
        export_densities, validate_economic_constraints)
    validate_economic_constraints(operating_rules, derived=True)
    densities = export_densities(manifest, trajectory.well_ids)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("A100 required")
    torch.set_num_threads(4)
    gpu_reservation = reserve_gpu_memory(args.gpu_memory_fraction)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(20260909)
    forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
        revision=MODEL_REVISION, per_core_batch_size=1, device="cuda"))
    if sha256(args.head.read_bytes()).hexdigest() != args.head_sha256:
        raise ValueError('trained head hash mismatch')
    selected = torch.load(args.head, map_location='cuda', weights_only=True)
    if not selected.get('economic_targets'):
        raise ValueError('a nine-target economic checkpoint is required')
    forecaster.model = load_frozen_model(forecaster.model, connectivity, selected)
    from timesfm_economics import ECONOMIC_TARGETS, forecast_chdd_rows, forecast_economic, observed_economic_history
    training = json.loads(args.head_report.read_text())
    if (training.get('complete') is not True or training.get('checkpoint_sha256') != args.head_sha256
            or training.get('model_revision') != MODEL_REVISION
            or training.get('economic_targets') != list(ECONOMIC_TARGETS)
            or training.get('connectivity_sha256') != sha256(args.connectivity.read_bytes()).hexdigest()):
        raise ValueError('completed training report, economic checkpoint and geology must match')
    economic_history, economic_trajectory = observed_economic_history(
        args.baseline_run / 'canonical', manifest, trajectory, origin)
    candidates = []
    rejections = []
    days = np.array([d.days_in_month for d in trajectory.dates[origin:origin + horizon]])[:, None]
    timestamps = trajectory.dates[origin + 1:origin + horizon + 1].strftime('%Y-%m-%d')

    def decode(policy, attempt):
        """Decode a policy into a validated request; gate G1 runs here, before any forecast."""
        policy = {"producer_scale": 1.0, "injector_scale": 1.0,
                  "shut_wells": [], "well_scales": [], **policy}
        controls = policy_controls(request["controls"], policy)
        proposed = {**request, "scenario_id": f"timesfm-policy-{attempt:02d}", "controls": controls, "context": {
            **request.get("context", {}),
            "objective": "Verify the graph selected and sealed by TimesFM forecast CHDD. Run OPM once; report physical CHDD even if it is worse than predicted. Do not select another graph using the result.",
            "facts": {"schedule_kind": "timesfm_policy_candidate", "is_baseline": False,
                      "surrogate_used_for_candidate_selection": True,
                      "surrogate_used_only_to_propose_hypotheses": False,
                      "simulated_reference_future_used": False,
                      "independent_surrogate_uq_calibrated": False,
                      "optimization_improvement_claimed": False,
                      "selected_before_final_opm": True},
            "policy": policy,
        }}
        checked = CycleRequest.from_mapping(proposed)
        _validate_source_well_scope(checked.controls, source_inventory,
                                   allow_conversion_to_injection=allow_conversion)
        original_roles = {(a['month'], a['well']): a['role'] for a in request['controls']}
        if (not checked.context.get('constraints', {}).get('allow_conversion_to_injection', False)
                and any(a['role'] != original_roles[a['month'], a['well']] for a in controls)):
            raise ValueError('new role changes are not permitted by this case')
        check_controls(operating_rules, checked.controls)
        return policy, controls, proposed, checked

    def plan_actions(controls):
        actions = trajectory.actions.copy()
        for a in controls:
            position = date_index[a['month']], well_index[a['well']]
            value = [a['value'], {'ORAT': 0, 'LRAT': 1, 'WRAT': 2}[a['target']], a['status'] == 'OPEN']
            if has_bhp:
                value.append(a.get('bhp_limit', actions[position][3]))
            elif 'bhp_limit' in a:
                raise ValueError('BHP policy requires the BHP action channel')
            actions[position] = value
        return actions

    def forecast_pass(controls):
        """One TimesFM pass for already validated controls; no OPM, no selection."""
        actions = plan_actions(controls)
        begin = time.monotonic()
        economic_trajectory.actions = actions
        prediction = forecast_economic(forecaster, economic_trajectory, origin, horizon, context, connectivity)
        if not np.isfinite(prediction).all():
            raise ValueError("non-finite TimesFM forecast")
        derived = derived_field_vectors(prediction, timestamps, trajectory.well_ids, densities,
            oil_fvf=args.oil_fvf, water_fvf=args.water_fvf,
            vrr_window=int(case_profile.vrr['window_months']))
        return actions, prediction, derived, time.monotonic() - begin

    def reject(policy, attempt, error):
        """A refusing guard is a result: recorded with its reason, never fatal to the search."""
        record = rejected_candidate(policy, attempt, error, case_profile_sha256=case_profile_sha256)
        rejections.append(record)
        (args.output / 'rejected-candidates.json').write_text(
            json.dumps(rejections, ensure_ascii=False, indent=2))
        return record

    def evaluate(policy):
        """Two passes: forecast, repair the monthly rates, forecast again, then gate G2."""
        attempt = len(candidates) + len(rejections)
        try:
            item = screen_forecast(policy, attempt)
            return screen_finish(item, screen_score(item), len(candidates))
        except (ValueError, CycleError) as error:
            return reject(policy, attempt, error)

    def evaluate_generation(policies):
        """Forecasts stay sequential on the GPU; the official calculator scores the batch in threads.

        Candidate ids are assigned before submission and records are appended in id
        order, so ``candidates.json`` stays the ordered ledger the seal requires.
        """
        slots, pending, reserved = [], [], []
        for policy in policies:
            attempt = len(candidates) + len(rejections) + len(pending)
            try:
                item = screen_forecast(policy, attempt, reserved)
            except (ValueError, CycleError) as error:
                slots.append(reject(policy, attempt, error))
                continue
            reserved.append(item['controls_sha256'])
            item['candidate_id'] = len(candidates) + len(pending)
            slots.append(item)
            pending.append(item)
        if pending:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(screen_score, pending))
            for item, result in zip(pending, results, strict=True):
                item['record'] = screen_finish(item, result, item['candidate_id'])
        return [item['record'] if 'record' in item else item for item in slots]

    def screen_forecast(policy, attempt, reserved=()):
        """Everything before the official calculator: gate G1, two forecast passes, gate G2."""
        policy, controls, proposed, checked = decode(policy, attempt)
        first = forecast_pass(controls)
        repaired, repair_factors = repair_policy(policy, first[2]['field'], case_profile)  # first[2] is derived
        if repair_factors['applied']:
            policy, controls, proposed, checked = decode(repaired, attempt)
        reject_duplicate_controls(checked.controls_sha256, candidates)
        if checked.controls_sha256 in reserved:
            raise CycleError('policy repeats an already evaluated control schedule; propose different controls')
        candidate_rules = (*operating_rules, *own_control_constraints(checked.controls))
        try:
            validate_economic_constraints(candidate_rules, derived=True)
        except ValueError as error:
            raise CycleError(str(error)) from error
        overlay = apply_schedule_overlay(source_schedule, checked.controls,
            known_wells=trajectory.well_ids, end_exclusive=trajectory.dates[origin + horizon].date())
        actions, prediction, derived, inference_seconds = (
            forecast_pass(controls) if repair_factors['applied'] else first)
        verdicts = economic_constraint_verdicts(prediction, timestamps, trajectory.well_ids,
                                                candidate_rules, derived=derived)
        violations = [verdict.message for verdict in failures(verdicts)]
        if not (prediction[..., 1] <= 500 + 1e-6).all():
            violations.append('forecast well liquid rate exceeds 500 m3/day')
        return {'attempt': attempt, 'policy': policy, 'controls': controls, 'proposed': proposed,
                'checked': checked, 'overlay': overlay, 'prediction': prediction, 'derived': derived,
                'future': actions[origin:origin + horizon], 'verdicts': verdicts,
                'violations': violations, 'repair_factors': repair_factors,
                'controls_sha256': checked.controls_sha256, 'inference_seconds': inference_seconds,
                'rows': forecast_chdd_rows(economic_history, timestamps, trajectory.well_ids, prediction)}

    def screen_score(item):
        """The official calculator on one forecast; a separate subprocess, safe to run in a thread."""
        period = (start.date(), trajectory.dates[origin + horizon].date())
        return calculator.calculate(opm_management_rows(item['rows'], period),
            start_year=item['checked'].start_year,
            output_dir=args.output / f"economics-{item['attempt']:02d}",
            charge_initial_pump=item['checked'].charge_initial_pump, management_period=period)

    def screen_finish(item, result, candidate_id):
        """Persist one scored candidate under its pre-assigned id."""
        policy, checked, prediction = item['policy'], item['checked'], item['prediction']
        derived, verdicts, violations = item['derived'], item['verdicts'], item['violations']
        repair_factors, future = item['repair_factors'], item['future']
        forecast_path = args.output / f'forecast-{candidate_id:02d}.npz'
        np.savez_compressed(forecast_path, prediction=prediction, timestamps=np.asarray(timestamps, dtype=str),
                            well_ids=np.asarray(trajectory.well_ids), targets=np.asarray(ECONOMIC_TARGETS))
        injection = float((np.where(future[..., 1] == 2, future[..., 0] * future[..., 2], 0) * days).sum())
        record = {"id": candidate_id, "policy": policy, "lookahead_months": horizon,
            "estimated_oil_t": float(prediction[..., 6].sum()),
            "estimated_liquid_t": float(prediction[..., 7].sum()),
            "planned_injection_m3": injection,
            "full_period_months": checked.horizon_months, "full_period_actions": len(item['controls']),
            "bhp_channel": has_bhp,
            "pressure_semantics": pressure_semantics,
            "trained_head_sha256": args.head_sha256,
            "forecast_by_well": [{'well': well,
                'oil_tonnes': float(prediction[:, i, 6].sum()),
                'liquid_tonnes': float(prediction[:, i, 7].sum()),
                'terminal_reservoir_pressure_bar': float(prediction[-1, i, 3])}
                for i, well in enumerate(trajectory.well_ids)],
            "controls_sha256": checked.controls_sha256, "schedule_overlay_sha256": item['overlay'].sha256,
            "source_schedule_constraints_checked_before_forecast": True,
            "inference_seconds": item['inference_seconds'],
            "is_official_chdd": False,
            'forecast_chdd_m': result.total_chdd_m,
            'predicted_injection_m3': float(prediction[..., 8].sum()),
            'economic_targets': list(ECONOMIC_TARGETS),
            'forecast_economics_manifest_sha256': sha256(result.manifest_path.read_bytes()).hexdigest(),
            'forecast_economics_directory': str(result.output_dir.relative_to(args.output.resolve())),
            'forecast_eligible': not violations,
            'forecast_constraint_violations': violations,
            'forecast_constraint_verdicts': [verdict.to_dict() for verdict in verdicts],
            'case_profile_sha256': case_profile_sha256,
            'repair_factors': repair_factors,
            'repair_passes': 2 if repair_factors['applied'] else 1,
            'formation_volume_factors': derived['formation_volume_factors'],
            'forecast_field_aggregates': derived['field'],
            'eligibility_scope': 'Forecast well liquid limit 500 m3/day, own LRAT/WRAT/BHP/role/status bounds, the case profile field caps, VRR window, water balance and repair calendar, and exact schedule constraints; diagnostic rules are reported but never fatal; physical feasibility and uncertainty require final verification.',
            'training_report_sha256': sha256(args.head_report.read_bytes()).hexdigest(),
            'forecast_sha256': sha256(forecast_path.read_bytes()).hexdigest()}
        candidates.append(record)
        item['proposed']["context"]["screening"] = record
        (args.output / f"request-{candidate_id:02d}.json").write_text(
            json.dumps(item['proposed'], ensure_ascii=False, indent=2))
        (args.output / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
        return record

    schema = {"type": "object", "properties": {
        "producer_scale": {"type": "number"}, "injector_scale": {"type": "number"},
        "shut_wells": {"type": "array", "items": {"type": "string"}},
        "well_scales": {"type": "array", "items": {"type": "object", "properties": {
            "well": {"type": "string"}, "scale": {"type": "number"}},
            "required": ["well", "scale"], "additionalProperties": False}}},
        "required": [], "additionalProperties": False}
    schema['properties']['well_updates'] = {'type': 'array', 'items': {
        'type': 'object', 'properties': {
            'well': {'type': 'string'}, 'start': {'type': 'string'}, 'end': {'type': 'string'},
            'role': {'type': 'string', 'enum': ['producer', 'injector']},
            'status': {'type': 'string', 'enum': ['OPEN', 'SHUT']},
            'target': {'type': 'string', 'enum': ['ORAT', 'LRAT', 'WRAT']},
            'value': {'type': 'number', 'minimum': 0},
            **({'bhp_limit': {'type': 'number', 'exclusiveMinimum': 0}} if has_bhp else {})},
        'required': ['well', 'start', 'end'], 'additionalProperties': False}}
    if has_bhp:
        schema['properties'].update(producer_bhp_add={'type': 'number', 'minimum': 0},
                                    injector_bhp_factor={'type': 'number', 'exclusiveMinimum': 0, 'maximum': 1})
    proposed_ids, skipped_rounds = [], []

    async def propose_round(index):
        before = len(candidates)
        attempted = []
        def propose(policy, _context):
            attempted.append(policy)
            (args.output / f"policy-attempts-{index:02d}.json").write_text(json.dumps(attempted, indent=2))
            record = evaluate(policy)
            if 'rejection' in record:
                # A refusing guard is a result: recorded, returned to the agent, never fatal.
                raise CycleError(record['rejection'])
            return record
        tool = ToolDefinition("propose_policy", "Propose a full-field policy. Optional producer_bhp_add (bar) and injector_bhp_factor tighten open-well BHP limits uniformly before individual updates. well_updates changes a well over inclusive monthly start/end dates after rate scaling: rate, status, target, role, BHP limit. Conversion requires explicit WRAT, value and BHP, must be permitted by the case, and cannot be reversed; extend its role to the end. Omitted scales default to 1, arrays to empty. Official calculator ranks nine-target forecasts; only the sealed winning graph gets final OPM verification.", schema,
            propose)
        context_value = {"track": 2, "round": index,
            "search_focus": checked_request.context.get('search_focus'),
            "conversion_search": {
                "permitted": allow_conversion,
                "instruction": "Assess producer-to-injector conversions across the full eligible producer inventory using only pre-origin state, geology and official conversion costs. For a producer_to_injector search, propose at least one new permanent conversion as a hypothesis for full OPM; include explicit WRAT, rate, start date and injection BHP ceiling. Keep other controls unchanged to isolate its effect. Do not claim gain or surrogate accuracy for new roles before physical validation. Reverse conversion is forbidden.",
            } if allow_conversion else {"permitted": False},
            "pressure_semantics": pressure_semantics,
            "surrogate_evidence": {"model": "Google TimesFM 3.0", "revision": MODEL_REVISION,
                "adapted_checkpoint_loaded": True, "verified_checkpoint_sha256": args.head_sha256,
                "full_period_forecast_already_completed": True, "accuracy_certified": False,
                "interpretation": "Checkpoint loading and forecast execution are verified. Accuracy certification is a separate, unresolved property; it does not mean the model is untrained. The unchanged baseline is already evaluated."},
            "objective": f"Improve forecast CHDD over all {horizon} months using permitted rates, BHP, statuses and producer-to-injector conversions. Call propose_policy exactly once with an unexplored policy. The official calculator scores forecasts, including pumps, conversions, water costs, taxes and discounting. All search evaluations use TimesFM; only the graph with maximum eligible forecast CHDD is sealed for one final OPM check. Never ask for OPM to compare search alternatives.",
            **agent_candidate_context(candidates),
            "verified_well_count": len(well_index),
            "geology": {
                "static_feature_names": connectivity.provenance.get('static_feature_names', []),
                "static_by_well": [[well, values] for well, values in zip(connectivity.well_ids,
                    connectivity.provenance.get('static_features', connectivity.static.tolist()), strict=True)],
                "five_strongest_neighbors": [[well, [[connectivity.well_ids[j], float(connectivity.weights[i, j])]
                    for j in np.argsort(-connectivity.weights[i])[:5] if connectivity.weights[i, j] > 0]]
                    for i, well in enumerate(connectivity.well_ids)],
                "all_links_used_in_forecast": True, "limitations": connectivity.provenance['limitations']},
            "economics": normative_profile,
            "field_state": [{"well": w, "oil_tpd": float(trajectory.states[origin, i, 0]),
                "liquid_tpd": float(trajectory.states[origin, i, 1]), "pressure_bar": float(trajectory.states[origin, i, 2]),
                "initial_control": initial_controls[w]}
                for i, w in enumerate(trajectory.well_ids)],
            "management_period": {'first_control_month': start.date().isoformat(),
                'last_control_month': max(a['month'] for a in request['controls'])},
            "constraints": {"producer_liquid_max_m3d": 500, "source_bhp_limits_cannot_be_relaxed": True,
                "source_completions_preserved": True, 'bhp_channel': has_bhp,
                'case_constraints': checked_request.context.get('constraints', {}),
                'operating_constraints': checked_request.context.get('operating_constraints', []),
                'first_source_control_month': {well: None if item.first_control_month is None
                    else item.first_control_month.isoformat()
                    for well, item in next(iter(source_inventory.values())).items()},
                'before_first_source_control': 'Must remain SHUT with zero rate; well_updates must not open wells earlier.',
                "water_limits_source": "Read operating_constraints. An absent bound is unspecified; it is not proof of unlimited water availability.",
                "water_balance_semantics": {
                    "max_monthly_water_deficit_m3": "Cap max(0, injected minus produced surface water volume) per month and explicit well group; excludes storage/processing losses.",
                    "min_monthly_voidage_replacement": "Lower bound on monthly injected/produced reservoir volumes for the explicit well group.",
                    "max_monthly_voidage_replacement": "Upper bound on monthly injected/produced reservoir volumes; positive injection with zero withdrawal violates this bound.",
                    "verification": "Requires actual OPM cumulative volume differences; predicted oil/liquid/pressure alone cannot certify these limits."}},
            "claim_limits": 'Forecast CHDD uses nine predicted economic outputs and observed history only. It includes all official economic terms, but it is not physically verified CHDD. No independently certified uncertainty, deployment approval or guaranteed uplift. Supplied schedule restrictions remain mandatory; unsupported forecast operating limits stop this mode.'}
        (args.output / f'planning-context-{index:02d}.json').write_text(json.dumps(context_value, ensure_ascii=False, indent=2))
        async with ExternalQwenClient(LLMConfig.from_env()) as client:
            workflow = AgentWorkflow(client, ToolRegistry((tool,)),
                role_tools={AgentRole.PLANNER: (tool.name,)}, required_tools={AgentRole.PLANNER: (tool.name,)})
            plan = await plan_with_control_repair(workflow, context_value, candidates, attempted, args.output, index)
        if plan is None:
            skipped_rounds.append(index)
            return
        (args.output / f"agent-{len(proposed_ids):02d}.json").write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
        if not all(d.approved for d in plan.decisions) or len(candidates) != before + 1:
            raise RuntimeError("Qwen must approve exactly one proposed policy")
        proposed_ids.append(candidates[-1]["id"])
        print(json.dumps(candidates[-1]), flush=True)

    def agent_rounds():
        """The grid fallback keeps the Qwen tool-calling rounds exactly as before."""
        for index in range(args.rounds):
            asyncio.run(propose_round(index))
        if not proposed_ids:
            raise RuntimeError('search requires at least one approved agent proposal')
        return proposed_ids, skipped_rounds

    well_roles = {}
    for action in sorted(request['controls'], key=lambda item: item['month']):
        well_roles.setdefault(action['well'], action['role'])
    origin_month = trajectory.dates[origin].date().isoformat()
    well_state = origin_well_state(economic_history, origin_month, densities,
        {well: float(trajectory.states[origin, i, 2]) for i, well in enumerate(trajectory.well_ids)})
    blocks_payload = json.loads(args.blocks.read_text()) if args.blocks else {
        'blocks': [{'id': 'field', 'wells': sorted(well_roles)}],
        'well_to_block': {well: 'field' for well in sorted(well_roles)}}
    blocks_sha256 = sha256(args.blocks.read_bytes()).hexdigest() if args.blocks else None
    space = PolicySpace(sorted({a['month'] for a in request['controls']}), well_roles,
        blocks=block_map(well_roles, blocks_payload.get('well_to_block')),
        caps=case_profile.to_dict(), baseline_rates=representative_rates(request['controls']),
        water_cut={well: row['water_cut'] for well, row in well_state.items() if well in well_roles} or None)

    def planning_briefs():
        """Briefs describe the field at the origin; nothing after the origin is read."""
        state = {'month': origin_month,
            'wells': [{'well': well, 'role': action['role'], 'status': action['status'],
                       'target': action['target'], 'value': action['value'],
                       'bhp_limit': action.get('bhp_limit')}
                      for well, action in sorted(initial_controls.items())],
            'totals': {key: sum(row[key] for row in well_state.values())
                       for key in ('liquid_m3d', 'oil_tpd', 'injection_m3d')},
            'neighbours': {well: [[connectivity.well_ids[j], float(connectivity.weights[i, j])]
                                  for j in np.argsort(-connectivity.weights[i])[:5]
                                  if connectivity.weights[i, j] > 0]
                           for i, well in enumerate(connectivity.well_ids)}}
        return (planning.build_block_briefs(state, well_state, blocks_payload, case_profile,
                                            normative_profile, candidates, {}),
                planning.build_field_brief(state, well_state, blocks_payload, case_profile,
                                           normative_profile, candidates, {}))

    def llm_round0():
        block_briefs, field_brief = planning_briefs()
        (args.output / 'planning-context-00.json').write_text(json.dumps(
            {'blocks': [brief.to_dict() for brief in block_briefs], 'field': field_brief.to_dict()},
            ensure_ascii=False, indent=2, sort_keys=True))

        async def call():
            async with ExternalQwenClient(LLMConfig.from_env()) as client:
                return await planning.run_round0(client, block_briefs, field_brief, SEARCH_SEED)

        intents, plan, journal = asyncio.run(call())
        (args.output / 'planning-round0.json').write_text(json.dumps(
            {'intents': [intent.to_dict() for intent in intents], 'next_focus': list(plan.next_focus),
             'journal': [entry.to_dict() for entry in journal]}, ensure_ascii=False, indent=2))
        return planning.merge_intents(intents, plan, well_roles)

    injections_done = []

    def llm_injection(digest):
        _, field_brief = planning_briefs()
        injections_done.append(len(injections_done) + 1)
        index = injections_done[-1]
        (args.output / f'planning-context-{index:02d}.json').write_text(json.dumps(
            {'field': field_brief.to_dict(), 'elite': list(digest)},
            ensure_ascii=False, indent=2, sort_keys=True))

        async def call():
            async with ExternalQwenClient(LLMConfig.from_env()) as client:
                return await planning.run_injection(client, digest, field_brief, SEARCH_SEED + index)

        proposals, journal = asyncio.run(call())
        (args.output / f'planning-injection-{index:02d}.json').write_text(json.dumps(
            {'journal': [entry.to_dict() for entry in journal],
             'proposals': [[genes, list(hint)] for genes, hint in proposals]},
            ensure_ascii=False, indent=2))
        return list(proposals)

    fragment = run_search(args, {
        'evaluate': evaluate, 'evaluate_generation': evaluate_generation,
        'reject': lambda policy, error: reject(policy, len(candidates) + len(rejections), error),
        'candidates': candidates, 'rejections': rejections, 'space': space,
        'output': args.output, 'blocks_sha256': blocks_sha256, 'agent_rounds': agent_rounds,
        'llm_round0': llm_round0, 'llm_injection': llm_injection})
    (args.output / "proposal-receipt.json").write_text(json.dumps({
        "source_trajectory_sha256": sha256(raw).hexdigest(), "request_sha256": sha256(args.request.read_bytes()).hexdigest(),
        "timesfm_revision": MODEL_REVISION, "agent_proposal_ids": fragment['agent_proposal_ids'],
        "attempted_agent_rounds": args.rounds if args.search == 'grid' else 0,
        "skipped_invalid_rounds": fragment['skipped_invalid_rounds'],
        "search": fragment['search'],
        "horizon_months": horizon, "head_sha256": args.head_sha256,
        "reference_manifest_sha256": None,
        "reference_correction_sha256": None,
        "connectivity_sha256": sha256(args.connectivity.read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "requires_full_period_opm": True,
        "economic_selection": True,
        "head_report_sha256": sha256(args.head_report.read_bytes()).hexdigest(),
        "normative_profile": normative_profile,
        "case_profile_sha256": case_profile_sha256,
        "case_profile": case_profile.to_dict(),
        "formation_volume_factors": {"oil": args.oil_fvf, "water": args.water_fvf},
        "operating_rules": [rule.to_dict() for rule in operating_rules],
        "rejected_candidates": len(rejections),
        "search_opm_calls": 0,
        **gpu_reservation,
        "final_chdd_computed": False}, indent=2))
    from track2_final_selection import seal_forecast_selection
    print(json.dumps(seal_forecast_selection(args.output)), flush=True)


if __name__ == "__main__":
    main()
