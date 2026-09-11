"""End-to-end drive of the Track 2 search loop with the physics replaced by arithmetic.

``main()`` cannot run here: it needs torch, an A100 and the official calculator. The
loop it delegates to, ``run_search``, takes every physical step as an injected
callable, so this module substitutes a deterministic ledger that still uses the real
``policy_controls``, the real ``PolicySpace`` and the real CMA-ES searcher, and still
produces the artifact set ``track2_final_selection.seal_forecast_selection`` demands.
"""

import json
from types import SimpleNamespace

import pytest

import track2_final_selection as selection
from propose_track2_policies import (BASE_POLICY, CycleError, policy_controls,
                                     representative_rates, run_search, violation_score)
from timesfm_economics import ECONOMIC_TARGETS
from timesoil.aios.policy_space import PolicySpace, block_map
from timesoil.aios.workflow import CycleRequest

MONTHS = [f'2007-{month:02d}-01' for month in range(1, 7)]
PRODUCERS = ('P1', 'P2', 'P3', 'P4')
INJECTORS = ('I1', 'I2')
CAPS = {'liquid_cap_m3d': 600.0, 'injection_cap_m3d': 600.0}
PROFILE_SHA256 = 'c' * 64
NORMS = {'source_sha256': {'Нормативы_ЧДД.xlsx': 'norms', 'calculator.py': 'calculator'}}


def build_request(tmp_path):
    source = tmp_path / 'model.zip'
    source.write_bytes(b'source')
    controls = []
    for month in MONTHS:
        controls.extend(dict(month=month, well=well, role='producer', status='OPEN',
                             target='LRAT', value=100.0, bhp_limit=60.0) for well in PRODUCERS)
        controls.extend(dict(month=month, well=well, role='injector', status='OPEN',
                             target='WRAT', value=120.0, bhp_limit=300.0) for well in INJECTORS)
    return dict(context={'track': 2}, source=str(source), deck='CASE.DATA',
                schedule_relative_path='schedule.inc', scenario_id='baseline',
                source_model='model_z_opm', start_year=2007, charge_initial_pump=False,
                controls=controls)


def build_space(request):
    roles = {}
    for action in sorted(request['controls'], key=lambda item: item['month']):
        roles.setdefault(action['well'], action['role'])
    return PolicySpace(sorted({action['month'] for action in request['controls']}), roles,
                       blocks=block_map(roles), caps=CAPS,
                       baseline_rates=representative_rates(request['controls']),
                       water_cut={well: .5 for well, role in roles.items() if role == 'producer'})


def make_args(**overrides):
    return SimpleNamespace(**{'search': 'cma', 'search_seconds': 4.0, 'skip_grid': False,
                              'llm_round0': True, 'injections': 2, 'rounds': 3, **overrides})


class Ledger:
    """The physics of ``main()`` replaced by arithmetic, keeping every seal-visible artifact.

    Controls come from the production ``policy_controls``, so a policy the search
    decodes but the schema refuses is a rejection here exactly as it is on the A100.
    """

    def __init__(self, root, request):
        self.root, self.request = root, request
        self.candidates, self.rejections = [], []

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False))
        return selection.digest(path)

    def reject(self, policy, error):
        record = {'id': None, 'attempt': len(self.candidates) + len(self.rejections),
                  'policy': policy, 'forecast_eligible': False, 'rejection': str(error),
                  'case_profile_sha256': PROFILE_SHA256}
        self.rejections.append(record)
        return record

    def evaluate(self, policy):
        try:
            return self.screen(policy)
        except (ValueError, CycleError) as error:
            return self.reject(policy, error)

    def evaluate_generation(self, policies):
        return [self.evaluate(policy) for policy in policies]

    def screen(self, policy):
        attempt = len(self.candidates) + len(self.rejections)
        candidate_id = len(self.candidates)
        controls = policy_controls(self.request['controls'], {**BASE_POLICY, **policy})
        proposed = {**self.request, 'scenario_id': f'timesfm-policy-{attempt:02d}', 'controls': controls}
        checked = CycleRequest.from_mapping(proposed, base_dir=self.root)
        if any(row['controls_sha256'] == checked.controls_sha256 for row in self.candidates):
            raise CycleError('policy repeats an already evaluated control schedule')
        liquid, injection = {}, {}
        for action in controls:
            if action['status'] != 'OPEN':
                continue
            totals = injection if action['role'] == 'injector' else liquid
            totals[action['month']] = totals.get(action['month'], 0.0) + action['value']
        verdicts = []
        for rule, totals, cap in (('max_monthly_liquid_m3d', liquid, CAPS['liquid_cap_m3d']),
                                  ('max_monthly_injection_m3d', injection, CAPS['injection_cap_m3d'])):
            worst = max(totals.values(), default=0.0)
            verdicts.append({'rule': rule, 'status': 'hard', 'ok': worst <= cap,
                             'worst_value': worst, 'margin': cap - worst})
        violations = [verdict['rule'] for verdict in verdicts if not verdict['ok']]
        chdd = sum(liquid.values()) * .05 - sum(injection.values()) * .02
        forecast = self.write(f'forecast-{candidate_id:02d}.npz', ['synthetic forecast artifact'])
        directory = f'economics-{attempt:02d}'
        input_sha256 = self.write(f'{directory}/input.csv', ['synthetic calculator input'])
        result_sha256 = self.write(f'{directory}/result.json', {'summary': {'totalChddM': chdd}})
        manifest = self.write(f'{directory}/manifest.json', dict(
            management_period={'total_chdd_m': chdd}, norms_source_sha256='norms',
            calculator_sha256={'calculator.py': 'calculator'},
            assumption_overrides={'chargeInitialPump': False},
            artifacts={'input': 'input.csv', 'result': 'result.json'},
            input_sha256=input_sha256, result_sha256=result_sha256))
        record = {'id': candidate_id, 'policy': policy, 'forecast_chdd_m': chdd,
                  'economic_targets': list(ECONOMIC_TARGETS), 'is_official_chdd': False,
                  'forecast_eligible': not violations, 'forecast_constraint_violations': violations,
                  'forecast_constraint_verdicts': verdicts, 'case_profile_sha256': PROFILE_SHA256,
                  'trained_head_sha256': 'head', 'training_report_sha256': 'training',
                  'controls_sha256': checked.controls_sha256,
                  'schedule_overlay_sha256': f'overlay-{candidate_id}', 'forecast_sha256': forecast,
                  'forecast_economics_directory': directory,
                  'forecast_economics_manifest_sha256': manifest}
        self.candidates.append(record)
        self.write(f'request-{candidate_id:02d}.json', proposed)
        self.write('candidates.json', self.candidates)
        return record


def stepping_clock(step=1.0):
    """One tick per call: run_cma_search calls it once per generation, so the budget is exact."""
    ticks = [-step]

    def clock():
        ticks[0] += step
        return ticks[0]

    return clock


def test_cma_search_fills_an_ordered_ledger_the_seal_accepts(tmp_path):
    root = tmp_path / 'search'
    root.mkdir()
    request = build_request(tmp_path)
    ledger, space = Ledger(root, request), build_space(request)
    seeds = [({'shut_wells': ['P4'], 'well_updates': [], 'overrides': {}},
              tuple(space.encode_seed({'field_producer': 1.1}))),
             ({'shut_wells': [], 'well_updates': [], 'overrides': {}}, ())]
    digests = []

    def llm_injection(digest):
        digests.append(digest)
        return [({'shut_wells': ['P3'], 'well_updates': [], 'overrides': {}},
                 tuple(space.encode_seed({'field_producer': .85})))]

    fragment = run_search(make_args(search_seconds=4.0, injections=2), {
        'evaluate': ledger.evaluate, 'evaluate_generation': ledger.evaluate_generation,
        'reject': ledger.reject, 'candidates': ledger.candidates, 'rejections': ledger.rejections,
        'space': space, 'output': root, 'blocks_sha256': None, 'agent_rounds': None,
        'llm_round0': lambda: seeds, 'llm_injection': llm_injection,
        'popsize': 4, 'sobol_seeds': 4, 'inject_every': 2, 'clock': stepping_clock()})

    # The unchanged incumbent is candidate 0: the seal reads it as the uplift baseline.
    assert ledger.candidates[0]['policy'] == BASE_POLICY
    assert [row['id'] for row in ledger.candidates] == list(range(len(ledger.candidates)))
    assert len(ledger.candidates) > 10 and any(row['forecast_eligible'] for row in ledger.candidates)
    assert all({'forecast_eligible', 'forecast_chdd_m', 'case_profile_sha256'} <= set(row)
               for row in ledger.candidates)
    assert fragment['search']['mode'] == 'cma' and fragment['search']['generations'] == 3
    assert fragment['search']['injections'] >= 1 and digests and digests[0][0]['npv_m'] is not None
    assert fragment['search']['evaluations'] == len(ledger.candidates) + len(ledger.rejections)
    assert fragment['agent_proposal_ids'] == [] and fragment['search']['llm_candidate_ids']
    trace = json.loads((root / 'search_trace.json').read_text())
    assert trace['seed'] == 20260909 and [row['generation'] for row in trace['generations'][:2]] == [0, 1]
    elite = json.loads((root / 'elite.json').read_text())
    assert elite and elite[0]['forecast_chdd_m'] >= elite[-1]['forecast_chdd_m']
    assert any(row['group'] == 'derived' for row in elite[0]['parameters'])

    ledger.write('proposal-receipt.json', dict(
        economic_selection=True, search_opm_calls=0, reference_manifest_sha256=None,
        reference_correction_sha256=None, final_chdd_computed=False, head_sha256='head',
        head_report_sha256='training', normative_profile=NORMS,
        horizon_months=CycleRequest.from_mapping(request).horizon_months,
        agent_proposal_ids=fragment['agent_proposal_ids'], search=fragment['search']))
    seal = selection.seal_forecast_selection(root)
    best = max((row for row in ledger.candidates if row['forecast_eligible']),
               key=lambda row: (row['forecast_chdd_m'], -row['id']))
    assert seal['selected_id'] == best['id'] and seal['forecast_chdd_m'] == best['forecast_chdd_m']


def test_grid_mode_still_evaluates_the_eight_point_grid(tmp_path):
    root = tmp_path / 'grid'
    root.mkdir()
    request = build_request(tmp_path)
    ledger = Ledger(root, request)
    fragment = run_search(make_args(search='grid'), {
        'evaluate': ledger.evaluate, 'evaluate_generation': ledger.evaluate_generation,
        'reject': ledger.reject, 'candidates': ledger.candidates, 'rejections': ledger.rejections,
        'space': None, 'output': root, 'blocks_sha256': 'b' * 64,
        'agent_rounds': lambda: ([len(ledger.candidates) - 1], [2])})
    assert len(ledger.candidates) + len(ledger.rejections) == 8
    assert [row['policy']['producer_scale'] for row in ledger.candidates] == [
        1, 1.25, 1, 1, .8, 1.5, 2, 3]
    assert [row['policy']['injector_scale'] for row in ledger.candidates] == [
        1, 1, .8, 1.2, 1, 1, 1, 1]
    assert fragment['search'] == {'mode': 'grid', 'seconds': None, 'blocks_sha256': 'b' * 64,
                                  'evaluations': 8, 'generations': 0, 'injections': 0,
                                  'llm_candidate_ids': []}
    assert fragment['agent_proposal_ids'] == [7] and fragment['skipped_invalid_rounds'] == [2]
    assert not (root / 'search_trace.json').exists()


def test_an_infeasible_candidate_always_outranks_nothing(tmp_path):
    assert violation_score({'forecast_eligible': True}) == 0.0
    assert violation_score({'forecast_eligible': False}) == 1.0  # No verdicts is still a breach.
    over = {'forecast_eligible': False, 'forecast_constraint_verdicts': [
        {'rule': 'cap', 'status': 'hard', 'ok': False, 'worst_value': 1200.0, 'margin': -600.0},
        {'rule': 'diagnostic-only', 'status': 'diagnostic', 'ok': False,
         'worst_value': 1.0, 'margin': -99.0}]}
    assert violation_score(over) == pytest.approx(.5)  # Diagnostic verdicts never count.
