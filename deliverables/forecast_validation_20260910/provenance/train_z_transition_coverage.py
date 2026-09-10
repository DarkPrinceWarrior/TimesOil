"""Continue frozen Z weights after the new development physics batch succeeds."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

r = Path('/root/projects/TimesOil/results/audit-20260909')
batch_root = r / 'transition-coverage-z-20260910'
out = r / 'timesfm-transition-coverage-z-20260910'
out.mkdir(exist_ok=False)
initial = r / 'timesfm-early-identity-z-20260910/training'
initial_report = json.loads((initial / 'report.json').read_text())
assert initial_report['complete']
assert sha256((initial / 'full-model.pt').read_bytes()).hexdigest() == initial_report['checkpoint_sha256']
protocol = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    initial_checkpoint_sha256=initial_report['checkpoint_sha256'],
    development_train_cases=[0, 1, 3, 6], development_validation_cases=[4],
    excluded_new_test_cases=[2, 5, 7], epochs=40,
    selection='Minimum loss on two existing validation scenarios and new BHP validation case 4; no new test predictions.',
    normalization='Retain authenticated initial training scale; unchanged full-horizon quantile loss.',
    independent_accuracy_claimed=False, waiting_for_session='timesoil-transition-coverage-z-20260910')
(out / 'wait-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
while not (batch_root / 'exit').exists():
    live = subprocess.run(['tmux', 'has-session', '-t', protocol['waiting_for_session']], capture_output=True).returncode == 0
    if not live and not (batch_root / 'exit').exists():
        raise RuntimeError('physics process ended without a completion receipt; do not start training')
    time.sleep(30)
assert (batch_root / 'exit').read_text().strip() == '0'
batch = batch_root / 'scenarios'
manifest_path = batch / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['complete'] and manifest['transition_coverage']
assert [x['index'] for x in manifest['scenarios']] == list(range(8))
assert manifest['calibration_cases'] == [0, 1, 3, 4, 6] and manifest['test_cases'] == [2, 5, 7]
cmd = json.loads((initial.parent / 'launch-protocol.json').read_text())['command']
for flag, value in [('--output', str(out / 'training')), ('--epochs', '40'),
                    ('--initial-head', str(initial / 'full-model.pt')),
                    ('--initial-head-sha256', initial_report['checkpoint_sha256']),
                    ('--bhp-calibration', str(batch)),
                    ('--bhp-calibration-sha256', sha256(manifest_path.read_bytes()).hexdigest())]:
    cmd[cmd.index(flag) + 1] = value
cmd.append('--retain-initial-scale')
protocol['command'] = cmd
protocol['batch_manifest_sha256'] = sha256(manifest_path.read_bytes()).hexdigest()
(out / 'launch-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
started = time.monotonic()
with (out / 'training.log').open('x') as log:
    code = subprocess.run(['taskset', '-c', '30-47', *cmd],
        env={**os.environ, 'PYTHONPATH': 'src:scripts', 'CUDA_VISIBLE_DEVICES': '5',
             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        stdout=log, stderr=subprocess.STDOUT).returncode
(out / 'exit').write_text(str(code) + '\n')
(out / 'completion.json').write_text(json.dumps({'returncode': code, 'seconds': time.monotonic() - started}) + '\n')
