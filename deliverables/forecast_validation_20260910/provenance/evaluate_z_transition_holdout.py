"""Compare two frozen checkpoints on new tests; development groups cannot calibrate intervals."""
import json, os, subprocess, time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

r = Path('/root/projects/TimesOil/results/audit-20260909')
physical = r / 'transition-coverage-z-20260910'
training = r / 'timesfm-transition-coverage-z-20260910'
out_root = r / 'timesfm-transition-evaluation-z-20260910'
out_root.mkdir(exist_ok=False)
(out_root / 'wait-protocol.json').write_text(json.dumps({
    'waiting_for_session': 'timesoil-transition-training-z-20260910',
    'selected_model': 'selected_transition',
    'comparison_model': 'previous_early_identity',
    'evaluated_cases': [2, 5, 7],
    'selection_basis': 'development validation only, before these test forecasts',
    'uncertainty_calibration_claimed': False,
}, indent=2) + '\n')
while not (training / 'exit').exists():
    live = subprocess.run(['tmux', 'has-session', '-t', 'timesoil-transition-training-z-20260910'], capture_output=True).returncode == 0
    if not live and not (training / 'exit').exists():
        raise RuntimeError('training ended without a completion receipt; evaluation is not authorized by a stale file')
    time.sleep(30)
assert all((p / 'exit').read_text().strip() == '0' for p in (physical, training))
batch = physical / 'scenarios'
batch_hash = sha256((batch / 'manifest.json').read_bytes()).hexdigest()
models = [('selected_transition', training / 'training'),
          ('previous_early_identity', r / 'timesfm-early-identity-z-20260910/training')]
protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'batch_sha256': batch_hash, 'amendment': 'Expanded development BHP coverage; only new held-out cases 2/5/7 are evaluated. No intervals from development cases.', 'model_selection_allowed_on_test': False,
    'selected_model': 'selected_transition', 'selection_basis': 'existing train/validation only',
    'forecast_mode': 'direct 224-month Google forecast, no physical reference future or candidate future observations',
    'models': {name: {'report_sha256': sha256((path / 'report.json').read_bytes()).hexdigest(),
                     'head_sha256': sha256((path / 'full-model.pt').read_bytes()).hexdigest()}
               for name, path in models}}
(out_root / 'frozen-models-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
results = []
for name, model in models:
    out = out_root / name
    cmd = ['/tmp/timesoil-kt3-20260908/venv/bin/python', 'scripts/evaluate_timesfm_scenarios.py',
        '--batch', str(batch), '--batch-sha256', batch_hash,
        '--head-report', str(model / 'report.json'), '--head', str(model / 'full-model.pt'),
        '--connectivity', str(r / 'static-head-geology-20260909/model-z/connectivity.json'),
        '--fixed-origin-only', '--test-only', '--output', str(out)]
    with (out_root / f'{name}.log').open('x') as log:
        code = subprocess.run(cmd, cwd='/root/projects/TimesOil-audit-transition-evaluation-z-20260910',
            env={**os.environ, 'PYTHONPATH': 'src:scripts', 'CUDA_VISIBLE_DEVICES': '5',
                 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            stdout=log, stderr=subprocess.STDOUT).returncode
    results.append({'name': name, 'returncode': code})
    (out_root / 'evaluation-completion.json').write_text(json.dumps(results, indent=2) + '\n')
    if code:
        raise RuntimeError(f'{name} evaluation failed with {code}')
    report = json.loads((out / 'report.json').read_text())
    assert report['complete'] and report['test_only'] and report['intervals'] == {}
    assert set(report['source_scenarios']) == {'2', '5', '7'}
(out_root / 'exit').write_text('0\n')
