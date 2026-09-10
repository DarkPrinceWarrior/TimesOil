"""Evaluate one development-selected nine-target checkpoint after its training finishes."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

R = Path('/root/projects/TimesOil/results/audit-20260909')
TRAINING = R / 'timesfm-economic-regimes-precise-z-20260910'
OUT = R / 'economic-uncertainty-z-20260910'
BATCH = R / 'fresh-uncertainty-z-20260910/scenarios'
BATCH_HASH = '4a9a8e53f8a44813fae8211d3a902f43f3c251b747d623c5dc47e78697a8eeef'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run():
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=digest(Path(__file__)), batch_sha256=BATCH_HASH,
        waiting_for_session='timesoil-economic-regimes-precise-z-20260910',
        training_protocol_sha256=digest(TRAINING / 'launch-protocol.json'),
        model_selection='One final checkpoint chosen by the existing 60-epoch development validation rule.',
        calibration_cases=[0, 1, 3, 4, 6], test_cases=[2, 5, 7],
        training_or_model_selection_on_evaluation_cases_allowed=False,
        previously_evaluated_with_three_target_model=True,
        unseen_new_reservoir_claimed=False, coverage_guaranteed=False,
        final_operational_result_used_for_training_or_selection=False,
        forecast_mode='fixed origin, 103 wells, 224 months, nine economic targets; future observations masked',
        new_opm_calls=0)
    if digest(BATCH / 'manifest.json') != BATCH_HASH:
        raise ValueError('frozen evaluation batch changed')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    while not (TRAINING / 'exit').exists():
        live = subprocess.run(['tmux', 'list-panes', '-t', protocol['waiting_for_session'],
                               '-F', '#{pane_dead}'], capture_output=True, text=True)
        if live.returncode or live.stdout.strip() != '0':
            raise RuntimeError('training ended without a completion receipt')
        time.sleep(30)
    if (TRAINING / 'exit').read_text().strip() != '0':
        raise RuntimeError('training failed; no evaluation started')
    if digest(TRAINING / 'launch-protocol.json') != protocol['training_protocol_sha256']:
        raise ValueError('training protocol changed while waiting')
    model = TRAINING / 'training'
    report = json.loads((model / 'report.json').read_text())
    checkpoint = digest(model / 'full-model.pt')
    if not report['complete'] or checkpoint != report['checkpoint_sha256'] or len(report['economic_targets']) != 9:
        raise ValueError('completed nine-target checkpoint required')
    protocol.update(checkpoint_sha256=checkpoint, training_report_sha256=digest(model / 'report.json'),
                    evaluation_started_utc=datetime.now(timezone.utc).isoformat())
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    command = ['taskset', '-c', '14-29', '/tmp/timesoil-kt3-20260908/venv/bin/python',
        'scripts/evaluate_timesfm_scenarios.py', '--batch', str(BATCH), '--batch-sha256', BATCH_HASH,
        '--head-report', str(model / 'report.json'), '--head', str(model / 'full-model.pt'),
        '--connectivity', str(R / 'static-head-geology-20260909/model-z/connectivity.json'),
        '--fixed-origin-only', '--output', str(OUT / 'evaluation')]
    with (OUT / 'evaluation.log').open('x') as log:
        code = subprocess.run(command, env={**os.environ, 'PYTHONPATH': 'src:scripts',
            'CUDA_VISIBLE_DEVICES': '5', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            stdout=log, stderr=subprocess.STDOUT).returncode
    (OUT / 'evaluation.exit').write_text(str(code) + '\n')
    if code:
        raise RuntimeError('economic evaluation failed; retained log and fixed protocol')
    evaluated = json.loads((OUT / 'evaluation/report.json').read_text())
    if (not evaluated['complete'] or evaluated['head_sha256'] != checkpoint
            or set(evaluated['source_scenarios']) != set(map(str, range(8)))
            or len(evaluated['intervals']['trained_economic_fixed_origin_224']['radius_by_target']) != 9
            or digest(model / 'full-model.pt') != checkpoint
            or digest(model / 'report.json') != protocol['training_report_sha256']):
        raise ValueError('evaluation coverage or frozen model identity changed')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    try:
        run()
    except BaseException:
        (OUT / 'exit').write_text('1\n')
        raise
    (OUT / 'exit').write_text('0\n')
