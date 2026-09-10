"""Evaluate the predeclared calibration-only linear baseline without Google."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from benchmark_timesfm3 import metrics
from evaluate_timesfm_scenarios import interval_check
from fit_timesfm_reference import bhp_features
from timesoil.aios.surrogate import _project_physics
from timesoil.aios.track2 import load_trajectory_dataset

r = Path('/root/projects/TimesOil/results/audit-20260909')
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
fit = read(r / 'timesfm-local-correction-z-20260910/report.json')
pilot = r / 'timesfm-reference-pilot-z-v3-20260910/evaluation'
old = r / 'bhp-only-validation-z-20260909'
independent = r / 'bhp-local-independent-z-20260910'
evaluated = r / 'timesfm-local-independent-z-20260910/corrected_reference'
evaluation = read(evaluated / 'report.json')
assert digest(pilot / 'report.json') == fit['pilot_report_sha256']
assert digest(old / 'manifest.json') == fit['calibration_batch_sha256']
assert evaluation['complete'] and digest(independent / 'manifest.json') == evaluation['batch_sha256']
degree = min(fit['cv'], key=lambda row: (row['without_google_delta_score'], row['degree']))['degree']
assert degree == 1  # Fixed by the old calibration report before independent results.
reference_dir = r / 'physical-z-forecast-validation-20260909/candidate-03'
assert digest(reference_dir / 'manifest.json') == fit['reference_manifest_sha256']
reference = load_trajectory_dataset(reference_dir / 'trajectory.csv', manifest=reference_dir / 'manifest.json')[0]
origin = reference.dates.get_loc('2007-01-01')
controls = reference.actions[origin:origin + 224]
truth_reference = reference.states[origin + 1:origin + 225]
features, residuals = [], []
for index in fit['calibration_cases']:
    root = old / f'candidate-{index:02d}'
    t = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')[0]
    assert t.content_hash == fit['source_scenarios'][str(index)]
    assert digest(pilot / f'candidate-{index:02d}.npz') == fit['pilot_array_sha256'][str(index)]
    arrays = np.load(pilot / f'candidate-{index:02d}.npz', allow_pickle=False)
    np.testing.assert_array_equal(arrays['truth'], t.states[origin + 1:origin + 225])
    features.append(bhp_features(t.actions[origin:origin + 224], controls, degree))
    residuals.append((arrays['truth'] - arrays['reference_only_224']).reshape(-1))
coefficients, _, rank, _ = np.linalg.lstsq(features, residuals, rcond=None)
assert rank == 2
np.savez_compressed(out / 'coefficients.npz', coefficients=coefficients)
report = {'schema': 'timesoil.local-physical-reference-ablation/v1',
          'calibration_report_sha256': digest(r / 'timesfm-local-correction-z-20260910/report.json'),
          'independent_report_sha256': digest(evaluated / 'report.json'),
          'coefficient_sha256': digest(out / 'coefficients.npz'),
          'degree': degree, 'training_used_independent_scenarios': False,
          'degree_selection': 'old leave-one-calibration-scenario-out score; before independent results',
          'evaluation_scope': 'Ablation of Google contribution; same local BHP domain, physical reference and old calibration scenarios.',
          'metrics': []}
manifest = read(independent / 'manifest.json')
errors = {}
for row in manifest['scenarios']:
    index = row['index']
    root = independent / f'candidate-{index:02d}'
    assert digest(root / 'trajectory.csv') == row['trajectory_sha256']
    assert digest(root / 'manifest.json') == row['export_manifest_sha256']
    t = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')[0]
    assert t.content_hash == evaluation['source_scenarios'][str(index)]
    assert t.content_hash not in fit['source_scenarios'].values()
    actions = t.actions[origin:origin + 224]
    predicted = truth_reference + (bhp_features(actions, controls, degree) @ coefficients).reshape(truth_reference.shape)
    predicted = _project_physics(predicted, actions, zero_injectors=True)[0]
    truth = t.states[origin + 1:origin + 225]
    errors[index] = np.abs(predicted - truth)
    report['metrics'].append({'index': index, 'split': 'test' if index in manifest['test_cases'] else 'calibration',
        **metrics(truth, predicted), 'first_month': metrics(truth[:1], predicted[:1])})
report['intervals'] = interval_check(np.stack([errors[i] for i in manifest['calibration_cases']]),
    np.stack([errors[i] for i in manifest['test_cases']]))
report.update(complete=True, script_sha256=digest(Path(__file__)))
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
