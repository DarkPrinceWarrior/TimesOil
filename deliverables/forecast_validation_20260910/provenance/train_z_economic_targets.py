"""Train economic outputs on the original development scenarios; never use fresh UQ data."""
import json
import os
from pathlib import Path
import subprocess
import time

r = Path('/root/projects/TimesOil/results/audit-20260909')
out = r / 'timesfm-economic-targets-z-20260910'
command = ['/tmp/timesoil-kt3-20260908/venv/bin/python', 'scripts/finetune_timesfm_head.py',
    '--batch', str(r / 'bhp-training-v3-20260909/scenario-runs'),
    '--batch-sha256', '4dbab179f94ca1800b052e1346591a685eb9fa2d8dc900d917fc6ad66d149893',
    '--connectivity', str(r / 'static-head-geology-20260909/model-z/connectivity.json'),
    '--output', str(out / 'training'), '--epochs', '40', '--learning-rate', '1e-5',
    '--unfreeze-backbone', '--condition-last-layer', '--condition-first-layer',
    '--cold-start-normalization', '--economic-targets']
out.mkdir(exist_ok=False)
protocol = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'command': command, 'gpu': 5, 'cpu_affinity': '14-29',
    'selection': 'Minimum normalized quantile loss on development validation scenario 009.',
    'training_scenarios': ['baseline', 'perturbation-001', 'perturbation-002', 'perturbation-003',
                           'perturbation-005', 'perturbation-006'],
    'fresh_uncertainty_cases_used': False, 'independent_accuracy_claimed': False,
    'physical_future_reference_used_at_inference': False,
    'operational_final_only_selection_complete': False}
(out / 'launch-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
started = time.monotonic()
with (out / 'training.log').open('x') as log:
    code = subprocess.run(['taskset', '-c', '14-29', *command],
        env={**os.environ, 'PYTHONPATH': 'src:scripts', 'CUDA_VISIBLE_DEVICES': '5',
             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        stdout=log, stderr=subprocess.STDOUT).returncode
(out / 'exit').write_text(str(code) + '\n')
(out / 'completion.json').write_text(json.dumps({'returncode': code,
    'seconds': time.monotonic() - started}) + '\n')
raise SystemExit(code)
