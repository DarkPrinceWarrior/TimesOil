"""Fresh group calibration/test trajectories for one already frozen TimesFM model."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from run_bhp_validation import UNCERTAINTY_DESIGNS, self_check
from run_full_period_sweep import scaled_request

R = Path('/root/projects/TimesOil/results/audit-20260909')
OUT = R / 'fresh-uncertainty-z-20260910'
ROOT_PYTHON = '/root/projects/TimesOil/.venv/bin/python'
GPU_PYTHON = '/tmp/timesoil-kt3-20260908/venv/bin/python'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run():
    OUT.mkdir(exist_ok=False)
    model = R / 'timesfm-transition-coverage-z-20260910/training'
    checkpoint = '7a548a77f2acdb50c1dc6e48e2685b96e4305badb45dfb51dbc1eb705568ada3'
    training = json.loads((model / 'report.json').read_text())
    assert training['complete'] and digest(model / 'full-model.pt') == checkpoint == training['checkpoint_sha256']
    original = R / 'physical-sweep-z-20260909/request-03.json'
    assert digest(original) == '2ecf8acc3a5f1cf7ea91b09f45ef7795c584c28a9ea41620a9b420ecf827d5c7'
    request = scaled_request(json.loads(original.read_text()), 1.09, .93)
    request_path = OUT / 'request.json'
    request_path.write_text(json.dumps(request, indent=2) + '\n')
    development = json.loads((R / 'transition-coverage-z-20260910/scenarios/manifest.json').read_text())
    development_controls = {s['controls_sha256'] for s in development['scenarios']}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint_sha256=checkpoint, training_report_sha256=digest(model / 'report.json'),
        script_sha256=digest(Path(__file__)), request_sha256=digest(request_path),
        development_controls_sha256=sorted(development_controls),
        rate_multipliers=dict(producer=1.09, injector=.93), bhp_designs=UNCERTAINTY_DESIGNS,
        calibration_cases=[0, 1, 3, 4, 6], test_cases=[2, 5, 7],
        training_or_model_selection_on_these_cases_allowed=False,
        forecast_mode='fixed origin, 224 months, no physical reference future',
        calibration_unit='whole 103-well trajectory, not independent well-month rows',
        coverage_guaranteed=False,
        limitation='Eight fixed controls on one training reservoir; exchangeability and final-case transfer unproven.',
        method_source='https://arxiv.org/html/2402.09623v3')
    (OUT / 'frozen-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'CUDA_VISIBLE_DEVICES': '5',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    def execute(name, command):
        with (OUT / f'{name}.log').open('x') as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (OUT / f'{name}.exit').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'{name} failed: {code}')
    try:
        batch = OUT / 'scenarios'
        execute('physics', [ROOT_PYTHON, 'scripts/run_bhp_validation.py', '--request', str(request_path),
            '--reference', str(R / 'physical-z-forecast-validation-20260909/candidate-03'),
            '--output', str(batch), '--uncertainty-validation'])
        manifest = json.loads((batch / 'manifest.json').read_text())
        assert manifest['complete'] and not development_controls & {s['controls_sha256'] for s in manifest['scenarios']}
        assert digest(model / 'full-model.pt') == checkpoint
        assert digest(model / 'report.json') == protocol['training_report_sha256']
        execute('evaluation', [GPU_PYTHON, 'scripts/evaluate_timesfm_scenarios.py',
            '--batch', str(batch), '--batch-sha256', digest(batch / 'manifest.json'),
            '--head-report', str(model / 'report.json'), '--head', str(model / 'full-model.pt'),
            '--connectivity', str(R / 'static-head-geology-20260909/model-z/connectivity.json'),
            '--fixed-origin-only', '--output', str(OUT / 'evaluation')])
        report = json.loads((OUT / 'evaluation/report.json').read_text())
        assert report['complete'] and report['head_sha256'] == checkpoint and report['intervals']
        (OUT / 'exit').write_text('0\n')
    except Exception:
        (OUT / 'exit').write_text('1\n')
        raise


if __name__ == '__main__':
    import sys
    self_check()
    if '--self-check' not in sys.argv:
        run()
