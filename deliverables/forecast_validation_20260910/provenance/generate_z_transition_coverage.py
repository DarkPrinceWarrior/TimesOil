"""Generate fresh development/holdout BHP coverage using the verified full OPM runner."""
import json, os, subprocess, time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from run_full_period_sweep import scaled_request, self_check

self_check()
r = Path('/root/projects/TimesOil/results/audit-20260909')
out = r / 'transition-coverage-z-20260910'
out.mkdir(exist_ok=False)
source = r / 'physical-sweep-z-20260909/request-03.json'
request = scaled_request(json.loads(source.read_text()), 1.09, .93)
request_path = out / 'request.json'
request_path.write_text(json.dumps(request, indent=2) + '\n')
cmd = ['/root/projects/TimesOil/.venv/bin/python', 'scripts/run_bhp_validation.py',
    '--request', str(request_path), '--reference', str(r / 'physical-z-forecast-validation-20260909/candidate-03'),
    '--output', str(out / 'scenarios'), '--transition-coverage']
(out / 'launch-protocol.json').write_text(json.dumps({
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'command': cmd,
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'incumbent_request_sha256': sha256(source.read_bytes()).hexdigest(),
    'new_request_sha256': sha256(request_path.read_bytes()).hexdigest(),
    'rate_multipliers': {'producer': 1.09, 'injector': .93},
    'bhp_designs': [[0, .92], [30, 1], [22.5, .96], [10, .97], [20, .94], [12.5, .89], [30, .88], [27.5, .93]],
    'calibration_cases': [0, 1, 3, 4, 6], 'test_cases': [2, 5, 7],
    'development_train_cases': [0, 1, 3, 6], 'development_validation_cases': [4],
    'diagnostic_sha256': sha256((r / 'z-development-transition-20260910/report.json').read_bytes()).hexdigest(),
    'model_selection_allowed_on_test': False,
    'scope': 'Fresh 103-well 224-month physical trajectories with jointly changed rate limits and BHP, original planned roles/statuses retained. Train only on designated development cases; freeze weights before forecasting new tests.',
}, indent=2) + '\n')
started = time.monotonic()
with (out / 'physics.log').open('x') as log:
    code = subprocess.run(cmd, cwd='/root/projects/TimesOil-audit-delivery-physical-y-20260910',
        env={**os.environ, 'PYTHONPATH': 'src:scripts', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        stdout=log, stderr=subprocess.STDOUT).returncode
(out / 'exit').write_text(str(code) + '\n')
(out / 'completion.json').write_text(json.dumps({'returncode': code, 'seconds': time.monotonic() - started}) + '\n')
