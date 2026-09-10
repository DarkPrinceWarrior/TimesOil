"""Partition existing calibration pressure error without changing the published metric."""
from hashlib import sha256
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

r = Path('/root/projects/TimesOil/results/audit-20260909')
base = r / 'cold-start-independent-z-20260910'
batch = base / 'scenarios'
manifest = json.loads((batch / 'manifest.json').read_text())
published_path = base / 'selected_early_identity/report.json'
published = json.loads(published_path.read_text())
geology = r / 'static-head-geology-20260909/model-z/connectivity.json'
wells = json.loads(geology.read_text())['well_ids']
digest = lambda p: sha256(p.read_bytes()).hexdigest()
assert digest(batch / 'manifest.json') == published['batch_sha256']
assert digest(geology) == published['connectivity_sha256']
dates = pd.date_range('2007-01-01', periods=225, freq='MS').strftime('%Y-%m-%d').tolist()
report = dict(schema='timesoil.pressure-status-diagnostic/v1', published_report_sha256=digest(published_path),
    script_sha256=digest(Path(__file__)), cases=manifest['calibration_cases'], independent_test=False,
    headline_metric_changed=False, rows=[])
for index in manifest['calibration_cases']:
    record = manifest['scenarios'][index]
    path = Path(record['directory']) / 'trajectory.csv'
    assert digest(path) == record['trajectory_sha256']
    with path.open() as handle:
        rows = {(row['date'], row['well']): row for row in csv.DictReader(handle)}
    active = np.array([[int(rows[date, well]['status']) for well in wells] for date in dates[:-1]], dtype=bool)
    truth = np.array([[float(rows[date, well]['pressure_bar']) for well in wells] for date in dates[1:]])
    prediction_path = base / 'selected_early_identity' / f'candidate-{index:02d}.npz'
    with np.load(prediction_path) as data:
        np.testing.assert_allclose(truth, data['truth'][..., 2], rtol=0, atol=1e-8)
        prediction = data['trained_fixed_origin_224'][..., 2]
    assert active.shape == truth.shape == prediction.shape == (224, 103)
    error = (prediction - truth) ** 2
    expected = next(x for x in published['metrics'] if x['index'] == index and x['mode'] == 'trained_fixed_origin_224')
    np.testing.assert_allclose(np.sqrt(error.mean()), expected['pressure_rmse_bar'], rtol=1e-12)
    zero = truth == 0
    report['rows'].append(dict(index=index, trajectory_sha256=digest(path), predictions_sha256=digest(prediction_path),
        closed_count=int((~active).sum()), closed_nonzero_truth_count=int(((~active) & (~zero)).sum()),
        zero_truth_count=int(zero.sum()), zero_truth_sse_fraction=float(error[zero].sum() / error.sum()),
        closed_sse_fraction=float(error[~active].sum() / error.sum()),
        active_rmse_bar=float(np.sqrt(error[active].mean())), all_rmse_bar=float(np.sqrt(error.mean()))))
assert len(report['rows']) == 5
report['complete'] = True
with Path(sys.argv[1]).open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps(report, indent=2))
