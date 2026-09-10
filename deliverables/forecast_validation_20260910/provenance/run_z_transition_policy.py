"""Run one frozen-transition TimesFM/Qwen hypothesis through OPM and official CHDD."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path('/root/projects/TimesOil/results/audit-20260909')
OUT = R / 'timesfm-transition-z-policy-20260910'
SESSION = 'timesoil-transition-evaluation-z-20260910'
PYTHON = '/tmp/timesoil-kt3-20260908/venv/bin/python'


def proposal_id(receipt, checkpoint):
    assert receipt['head_sha256'] == checkpoint
    assert receipt['horizon_months'] == 224 and receipt['requires_full_period_opm'] is True
    assert receipt['reference_manifest_sha256'] is None
    assert receipt['reference_correction_sha256'] is None
    ids = receipt['agent_proposal_ids']
    assert len(ids) == 1 and type(ids[0]) is int and ids[0] > 0
    return ids[0]


def self_check():
    receipt = dict(head_sha256='frozen', horizon_months=224, requires_full_period_opm=True,
                   reference_manifest_sha256=None, reference_correction_sha256=None,
                   agent_proposal_ids=[1])
    assert proposal_id(receipt, 'frozen') == 1
    for change in ({'head_sha256': 'other'}, {'horizon_months': 12},
                   {'reference_manifest_sha256': 'future'}, {'agent_proposal_ids': [0]},
                   {'agent_proposal_ids': [True]}, {'agent_proposal_ids': [1, 2]}):
        try:
            proposal_id({**deepcopy(receipt), **change}, 'frozen')
        except AssertionError:
            pass
        else:
            raise AssertionError(f'invalid proposal accepted: {change}')
    print('Frozen weights, full horizon, no future reference and one new proposal verified.')


def run():
    training = R / 'timesfm-transition-coverage-z-20260910'
    evaluation = R / 'timesfm-transition-evaluation-z-20260910'
    protocol = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    waiting_for_session=SESSION, checkpoint_selection='development validation only',
                    experiment='One new Qwen proposal; full 103-well, 224-month OPM and official CHDD',
                    selected_schedule_metric='official full-period CHDD, retaining incumbent when better',
                    forecast_accuracy_certified=False, test_observations_passed_to_agents=False)
    (OUT / 'wait-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    while not (evaluation / 'exit').exists():
        live = subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True).returncode == 0
        if not live and not (evaluation / 'exit').exists():
            raise RuntimeError('evaluation stopped without a completion receipt')
        time.sleep(30)
    assert all((p / 'exit').read_text().strip() == '0' for p in (training, evaluation))
    model = training / 'training'
    report = json.loads((model / 'report.json').read_text())
    checkpoint = sha256((model / 'full-model.pt').read_bytes()).hexdigest()
    assert report['complete'] is True and checkpoint == report['checkpoint_sha256']
    frozen = json.loads((evaluation / 'frozen-models-protocol.json').read_text())
    assert frozen['models']['selected_transition']['head_sha256'] == checkpoint
    assert frozen['models']['selected_transition']['report_sha256'] == sha256((model / 'report.json').read_bytes()).hexdigest()
    assert frozen['model_selection_allowed_on_test'] is False
    for name in ('selected_transition', 'previous_early_identity'):
        tested = json.loads((evaluation / name / 'report.json').read_text())
        assert tested['complete'] and tested['test_only'] and tested['intervals'] == {}
        assert set(tested['source_scenarios']) == {'2', '5', '7'}
    request = R / 'physical-sweep-z-20260909/request-03.json'
    assert sha256(request.read_bytes()).hexdigest() == '2ecf8acc3a5f1cf7ea91b09f45ef7795c584c28a9ea41620a9b420ecf827d5c7'
    baseline = R / 'timesfm-bhp-policy-20260909/cycles/baseline'
    incumbent = R / 'physical-sweep-z-20260909/cycles/candidate-03'
    previous = R / 'timesfm-early-identity-z-policy-20260910/cycles/candidate'
    env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'CUDA_VISIBLE_DEVICES': '5',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'OPM_MPI_PROCESSES': '16', 'OPM_THREADS_PER_PROCESS': '1', 'OPM_CPU_AFFINITY': '30-45',
           'LLM_BASE_URL': 'https://litellm.tatneft.guru/v1', 'LLM_MODEL': 'qwen3.8-27b',
           'LLM_TIMEOUT_SECONDS': '600', 'LLM_MAX_OUTPUT_TOKENS': '8192',
           'LLM_API_KEY': Path('/dev/shm/timesoil-tatneft-20260909-key').read_text().strip()}
    protocol.update(checkpoint_sha256=checkpoint, started_utc=datetime.now(timezone.utc).isoformat(),
                    evaluation_protocol_sha256=sha256((evaluation / 'frozen-models-protocol.json').read_bytes()).hexdigest())
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    def execute(name, command):
        with (OUT / f'{name}.log').open('x') as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (OUT / f'{name}.exit').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'{name} exited with {code}; see its retained log')

    execute('proposal', [PYTHON, 'scripts/propose_track2_policies.py', '--request', str(request),
        '--baseline-run', str(baseline), '--output', str(OUT / 'proposals'), '--rounds', '1', '--skip-grid',
        '--head', str(model / 'full-model.pt'), '--head-sha256', checkpoint,
        '--connectivity', str(R / 'static-head-geology-20260909/model-z/connectivity.json')])
    receipt = json.loads((OUT / 'proposals/proposal-receipt.json').read_text())
    index = proposal_id(receipt, checkpoint)
    execute('candidate', [PYTHON, '-m', 'timesoil.aios.cli', 'full-cycle',
        str(OUT / f'proposals/request-{index:02d}.json'), '--runs-dir', str(OUT / 'cycles'),
        '--run-id', 'candidate', '--timeout', '7200'])
    execute('selection', [PYTHON, 'scripts/compare_track2_cycles.py', str(baseline),
        str(OUT / 'cycles/candidate'), str(OUT / 'selection.json'), '--expected-months', '224',
        '--select-from', str(incumbent), str(previous), '--agent-review'])
    selection = json.loads((OUT / 'selection.json').read_text())
    assert selection['agent_review_approved'] is True
    assert len(selection['agent_review']['decisions']) == 4
    assert all(d['approved'] for d in selection['agent_review']['decisions'])
    (OUT / 'exit').write_text('0\n')


if __name__ == '__main__':
    self_check()
    if '--self-check' not in sys.argv:
        OUT.mkdir(exist_ok=False)
        try:
            run()
        except Exception:
            (OUT / 'exit').write_text('1\n')
            raise
