"""Extend economic training with existing development regimes after search releases GPU 5."""
import json
import os
from pathlib import Path
import subprocess
import time

r = Path('/root/projects/TimesOil/results/audit-20260909')
out = r / 'timesfm-economic-regimes-z-20260910'
search = r / 'timesfm-final-only-z-compact-20260910'
command = ['/tmp/timesoil-kt3-20260908/venv/bin/python', 'scripts/finetune_timesfm_head.py',
    '--batch', str(r / 'bhp-training-v3-20260909/scenario-runs'),
    '--batch-sha256', '4dbab179f94ca1800b052e1346591a685eb9fa2d8dc900d917fc6ad66d149893',
    '--connectivity', str(r / 'static-head-geology-20260909/model-z/connectivity.json'),
    '--initial-head', str(r / 'timesfm-economic-targets-z-20260910/training/full-model.pt'),
    '--initial-head-sha256', '9b954acbe64f0c5ff8dc213315a46b9324a948294ac19bc3b087d55fb50844ae',
    '--regime-calibration', str(r / 'physical-z-forecast-validation-20260909'),
    '--regime-calibration-sha256', '69a92d83bc7824b68af1eaddbddd884b589e4009b7d12de4d23be2e0a227277f',
    '--bhp-calibration', str(r / 'transition-coverage-z-20260910/scenarios'),
    '--bhp-calibration-sha256', '9bffec89541467afaf904819ad77baa4469bd4d726ff55d3e08f2ba92c2fae8d',
    '--output', str(out / 'training'), '--epochs', '60', '--learning-rate', '1e-5',
    '--unfreeze-backbone', '--condition-last-layer', '--condition-first-layer',
    '--cold-start-normalization', '--retain-initial-scale', '--economic-targets']
out.mkdir(exist_ok=False)
protocol = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'command': command, 'gpu': 5, 'cpu_affinity': '14-29', 'fresh_uncertainty_cases_used': False,
    'independent_accuracy_claimed': False, 'selection': 'Minimum mean loss on three development validation scenarios.',
    'original_training': ['baseline','perturbation-001','perturbation-002','perturbation-003','perturbation-005','perturbation-006'],
    'additional_training': ['physical-sweep-00','physical-sweep-01','physical-sweep-02','physical-sweep-07',
                           'bhp-only-00','bhp-only-01','bhp-only-03','bhp-only-06'],
    'validation': ['perturbation-009','physical-sweep-04','bhp-only-04'],
    'current_final_only_search_uses_unchanged_initial_checkpoint': True,
    'waiting_for_session': 'timesoil-final-only-z-compact-20260910'}
(out / 'launch-protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
try:
    while not (search / 'search.exit').exists():
        live = subprocess.run(['tmux','list-panes','-t',protocol['waiting_for_session'],'-F','#{pane_dead}'],
                              capture_output=True, text=True)
        if live.returncode or live.stdout.strip() != '0':
            raise RuntimeError('search ended without its stage completion receipt')
        time.sleep(30)
    with (out / 'training.log').open('x') as log:
        code = subprocess.run(['taskset','-c','14-29',*command], env={**os.environ,
            'PYTHONPATH':'src:scripts','CUDA_VISIBLE_DEVICES':'5','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
            stdout=log, stderr=subprocess.STDOUT).returncode
    (out / 'exit').write_text(str(code) + '\n')
except BaseException:
    (out / 'exit').write_text('1\n')
    raise
