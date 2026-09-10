import asyncio
import json

import pytest

from propose_track2_policies import CycleError, plan_with_control_repair


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
    with pytest.raises(CycleError, match='still invalid'):
        asyncio.run(plan_with_control_repair(broken, context, [], [], tmp_path, 1))
    assert broken.calls == 2

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
