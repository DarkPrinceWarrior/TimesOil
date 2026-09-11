import asyncio
import json

import pytest

from propose_track2_policies import CycleError, agent_candidate_context, plan_with_control_repair, reject_duplicate_controls


def test_all_candidate_scores_and_full_well_detail_fit_without_repetition():
    candidates = [dict(id=i, policy={'injector_scale':1+i/10}, forecast_chdd_m=100+i,
        forecast_eligible=True, controls_sha256=str(i),
        forecast_constraint_violations=['example violation'] * 224,
        forecast_by_well=[dict(well=str(w), oil_tonnes=w, liquid_tonnes=2*w) for w in range(103)],
        pressure_semantics={'repeated':'x'*20000}) for i in range(20)]
    context = agent_candidate_context(candidates)
    assert len(json.dumps(context)) < 20000
    assert [row['forecast_chdd_m'] for row in context['candidates']] == list(range(100,120))
    assert [row['policy'] for row in context['candidates']] == [row['policy'] for row in candidates]
    assert context['forecast_detail']['candidate_id'] == 19
    assert context['candidates'][0]['forecast_constraint_violation_count'] == 224
    assert len(context['candidates'][0]['forecast_constraint_violations']) == 3
    assert len(context['forecast_detail']['by_well']) == 103
    assert len(candidates[0]['forecast_by_well']) == 103


def test_invalid_control_gets_one_repair_without_relaxing_constraints(tmp_path):
    context = {'constraints': {'first_source_control_month': {'P': '2008-01-01'}}}
    attempted = [{'well_updates': [{'well': 'P', 'start': '2007-01-01'}]}]

    class Workflow:
        calls = 0
        async def run_plan(self, current):
            self.calls += 1
            if self.calls == 1:
                raise CycleError('well=P, month=2007-01-01, first_source=2008-01-01')
            assert current['constraints'] == context['constraints']
            assert current['previous_invalid_proposal']['policy'] == attempted[0]
            assert '2008-01-01' in current['previous_invalid_proposal']['error']
            return 'corrected'

    workflow = Workflow()
    assert asyncio.run(plan_with_control_repair(workflow, context, [], attempted, tmp_path, 0)) == 'corrected'
    assert workflow.calls == 2 and 'previous_invalid_proposal' not in context
    assert json.loads((tmp_path / 'rejected-plan-00-0.json').read_text())['accepted_candidates_added'] == 0

    class Broken:
        calls = 0
        async def run_plan(self, current):
            self.calls += 1
            raise CycleError('still invalid')

    broken = Broken()
    assert asyncio.run(plan_with_control_repair(broken, context, [], [], tmp_path, 1)) is None
    assert broken.calls == 2
    assert json.loads((tmp_path / 'rejected-plan-01-1.json').read_text())['accepted_candidates_added'] == 0

    candidates = []
    class Partial:
        calls = 0
        async def run_plan(self, current):
            self.calls += 1
            candidates.append('already accepted')
            raise CycleError('second call invalid')

    partial = Partial()
    with pytest.raises(CycleError):
        asyncio.run(plan_with_control_repair(partial, context, candidates, [], tmp_path, 2))
    assert partial.calls == 1


def test_duplicate_policy_is_returned_to_agent_for_one_correction(tmp_path):
    candidates = [{'controls_sha256': 'existing'}]
    class Workflow:
        calls = 0
        async def run_plan(self, context):
            self.calls += 1
            if self.calls == 1:
                reject_duplicate_controls('existing', candidates)
            assert 'already evaluated' in context['previous_invalid_proposal']['error']
            reject_duplicate_controls('new', candidates)
            return 'corrected'
    workflow = Workflow()
    assert asyncio.run(plan_with_control_repair(workflow, {}, candidates,
        [{'producer_scale':1.5}], tmp_path, 0)) == 'corrected'
    assert workflow.calls == 2 and len(candidates) == 1


from pathlib import Path

from propose_track2_policies import policy_controls, rejected_candidate, repair_policy
from timesoil.aios.case_profile import load_case_profile

PROFILE = load_case_profile(Path(__file__).resolve().parents[1] / 'config/case_constraints.example.json')
MONTHS = ['2007-01-01', '2007-02-01', '2007-03-01']
AGGREGATES = {'months': MONTHS,
              'liquid_m3d': [2000., 1000., 1000.],
              'injection_m3d': [2000., 1000., 1000.],
              'water_m3d': [5000., 500., 5000.],
              'reservoir_production_m3': [1000., 1000., 1000.],
              'reservoir_injection_m3': [1000., 1000., 3000.]}


def test_repair_scales_each_month_to_the_binding_cap_water_or_vrr_bound():
    policy = {'producer_scale': 1.0}
    repaired, factors = repair_policy(policy, AGGREGATES, PROFILE)
    assert factors['applied'] and repaired['producer_scale'] == 1.0
    assert repaired['monthly_repair'] == {'liquid': factors['liquid'], 'injection': factors['injection']}
    # Liquid hits exactly (1 - eps_liquid) * 1500 in the only month that exceeds it.
    assert list(factors['liquid']) == [MONTHS[0]]
    assert AGGREGATES['liquid_m3d'][0] * factors['liquid'][MONTHS[0]] == pytest.approx(.97 * 1500)
    assert factors['binding'] == {MONTHS[0]: 'injection_cap', MONTHS[1]: 'produced_water',
                                  MONTHS[2]: 'vrr_upper'}
    assert AGGREGATES['injection_m3d'][0] * factors['injection'][MONTHS[0]] == pytest.approx(.97 * 1500)
    assert AGGREGATES['injection_m3d'][1] * factors['injection'][MONTHS[1]] == pytest.approx(.95 * 500)
    # Three-month window solved for March: the January/February volumes are already repaired.
    earlier = 1000 * factors['injection'][MONTHS[0]] + 1000 * factors['injection'][MONTHS[1]]
    assert factors['injection'][MONTHS[2]] == pytest.approx((1.15 * 3000 - earlier) / 3000)
    assert factors['window_months'] == 3 and factors['case_profile_sha256'] == PROFILE.sha256
    assert all(0 < value <= 1 for table in ('liquid', 'injection') for value in factors[table].values())

    relaxed = {**AGGREGATES, 'liquid_m3d': [10.] * 3, 'injection_m3d': [10.] * 3,
               'water_m3d': [1000.] * 3, 'reservoir_injection_m3': [100.] * 3,
               'reservoir_production_m3': [1000.] * 3}
    untouched, idle = repair_policy(policy, relaxed, PROFILE)
    assert idle['applied'] is False and untouched['monthly_repair'] == {'liquid': {}, 'injection': {}}
    stopped, stopped_factors = repair_policy(policy, {**relaxed, 'injection_m3d': [0.] * 3}, PROFILE)
    assert stopped_factors['injection'] == {} and stopped['monthly_repair']['injection'] == {}
    with pytest.raises(ValueError, match='exactly the forecast months'):
        repair_policy(policy, {**AGGREGATES, 'water_m3d': [1.]}, PROFILE)


def test_repaired_policy_scales_only_the_month_it_names():
    controls = [dict(month=month, well=well, role=role, status='OPEN', target=target, value=100.)
                for month in MONTHS
                for well, role, target in (('P', 'producer', 'LRAT'), ('I', 'injector', 'WRAT'))]
    repaired, factors = repair_policy({}, AGGREGATES, PROFILE)
    scaled = policy_controls(controls, {'producer_scale': 1.0, 'injector_scale': 1.0,
                                        'shut_wells': [], 'well_scales': [], **repaired})
    values = {(a['month'], a['well']): a['value'] for a in scaled}
    assert values[MONTHS[0], 'P'] == pytest.approx(100 * factors['liquid'][MONTHS[0]])
    assert values[MONTHS[0], 'I'] == pytest.approx(100 * factors['injection'][MONTHS[0]])
    assert values[MONTHS[1], 'P'] == 100.               # no liquid factor for February
    assert values[MONTHS[1], 'I'] == pytest.approx(100 * factors['injection'][MONTHS[1]])
    assert values[MONTHS[2], 'P'] == 100.
    assert values[MONTHS[2], 'I'] == pytest.approx(100 * factors['injection'][MONTHS[2]])
    assert all(action['value'] == 100. for action in controls)


def test_a_refusing_guard_is_recorded_and_never_reaches_the_sealed_ledger():
    record = rejected_candidate({'producer_scale': 3.0}, 7, ValueError('planned max_liquid_m3d exceeds'),
                                case_profile_sha256='a' * 64)
    assert record['forecast_eligible'] is False and record['id'] is None and record['attempt'] == 7
    assert 'max_liquid_m3d' in record['rejection'] and record['case_profile_sha256'] == 'a' * 64
    assert 'forecast_chdd_m' not in record
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'track2_final_selection', Path(__file__).resolve().parents[1] / 'scripts/track2_final_selection.py')
    seal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seal)
    with pytest.raises(ValueError):
        seal.best_forecast([{**record, 'id': 0}])


def test_repair_respects_the_case_water_semantics_and_solves_vrr_per_month():
    """No produced-water term without a deficit rule; the VRR window is solved month by month."""
    import json, tempfile, pathlib
    from timesoil.aios.case_profile import load_case_profile
    from propose_track2_policies import repair_policy
    base = json.loads(pathlib.Path('config/case_z_test.json').read_text())
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / 'case.json').write_text(json.dumps(base))
    profile = load_case_profile(tmp / 'case.json')
    agg = {'months': ['2007-01-01', '2007-02-01', '2007-03-01'],
           'liquid_m3d': [500.0, 500.0, 500.0], 'injection_m3d': [500.0, 500.0, 500.0],
           'water_m3d': [40.0, 40.0, 40.0],  # tiny produced water must NOT throttle injection
           'reservoir_production_m3': [1000.0, 1000.0, 1000.0],
           'reservoir_injection_m3': [1000.0, 1000.0, 3000.0]}
    _, factors = repair_policy({}, agg, profile)
    assert 'produced_water' not in set(factors['binding'].values())
    # window over 2007-01..03: allowed = 1.15 * 3000 - (1000 + 1000) = 1450 -> factor 1450/3000
    assert factors['binding']['2007-03-01'] == 'vrr_upper'
    assert abs(factors['injection']['2007-03-01'] - 1450.0 / 3000.0) < 1e-9
    strict = dict(base); strict['water_balance'] = {'deficit_m3': 0, 'carryover': False}
    (tmp / 'strict.json').write_text(json.dumps(strict))
    _, strict_factors = repair_policy({}, agg, load_case_profile(tmp / 'strict.json'))
    assert set(strict_factors['binding'].values()) == {'produced_water'}
