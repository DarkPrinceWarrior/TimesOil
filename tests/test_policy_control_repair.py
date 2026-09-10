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
