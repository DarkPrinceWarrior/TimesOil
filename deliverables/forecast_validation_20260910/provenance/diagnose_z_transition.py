"""Diagnose the frozen Z model on development data; no independent-test claims."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

from benchmark_timesfm3 import MODEL_REVISION, metrics
from benchmark_timesfm_controls import verified_batch
from benchmark_timesfm_layouts import forecast_layout
from finetune_timesfm_head import verified_regime_calibration
from timesfm_geology import load_frozen_model
from timesoil.aios.interwell import WellConnectivity

r = Path('/root/projects/TimesOil/results/audit-20260909')
training_dir = r / 'timesfm-early-identity-z-20260910/training'
training = json.loads((training_dir / 'report.json').read_text())
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
digest = lambda p: sha256(p.read_bytes()).hexdigest()
assert training['complete'] and training['model_revision'] == MODEL_REVISION
assert digest(training_dir / 'full-model.pt') == training['checkpoint_sha256']
trajectories, origin = verified_batch(r / 'bhp-training-v3-20260909/scenario-runs', training['batch_manifest_sha256'])
baseline = next(t for t in trajectories if t.scenario_id == 'baseline')
trajectories = list(trajectories)
for directory, key in [('physical-z-forecast-validation-20260909', 'regime_calibration_manifest_sha256'),
                       ('bhp-only-validation-z-20260909', 'bhp_calibration_manifest_sha256')]:
    trajectories += verified_regime_calibration(r / directory, training[key], baseline, origin)
selected_ids = training['train_scenarios'] + training['validation_scenarios']
assert set(selected_ids).isdisjoint(training['test_scenarios'])
by_id = {t.scenario_id: t for t in trajectories}
assert all(by_id[n].content_hash == training['source_scenario_hashes'][n] for n in selected_ids)
geology_path = r / 'static-head-geology-20260909/model-z/connectivity.json'
assert digest(geology_path) == training['connectivity_sha256']
connectivity = WellConnectivity.from_dict(json.loads(geology_path.read_text()))
report = dict(schema='timesoil.development-transition-diagnostic/v1', independent_test=False,
              checkpoint_sha256=training['checkpoint_sha256'], training_report_sha256=digest(training_dir / 'report.json'),
              script_sha256=digest(Path(__file__)), scenarios=selected_ids,
              future_observations_used=False, physical_reference_future_used=False, metrics=[])
(out / 'protocol.json').write_text(json.dumps(report, indent=2) + '\n')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from timesfm3 import ModelConfig, TimesFM3Forecaster
assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.35)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(20260909)
forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
    revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
forecaster.model = load_frozen_model(forecaster.model, connectivity,
    torch.load(training_dir / 'full-model.pt', map_location='cuda', weights_only=True))
for name in selected_ids:
    t = by_id[name]
    prediction = forecast_layout(forecaster, t, origin, 224, 128, 'joint', connectivity=connectivity)
    truth = t.states[origin + 1:origin + 225]
    assert prediction.shape == truth.shape == (224, 103, 3)
    assert np.isfinite(prediction).all()
    error = np.abs(prediction - truth)
    worst = np.argsort(error[0, :, 0])[-10:][::-1]
    row = dict(scenario=name, split='train' if name in training['train_scenarios'] else 'validation',
        source_sha256=t.content_hash, full=metrics(truth, prediction),
        first_month=metrics(truth[:1], prediction[:1]), first_six=metrics(truth[:6], prediction[:6]),
        first_month_absolute_error_share=(error[0].sum(axis=0) / np.maximum(error.sum(axis=(0, 1)), 1e-12)).tolist(),
        worst_first_month_oil=[dict(well=t.well_ids[i], truth=truth[0, i].tolist(), prediction=prediction[0, i].tolist(),
            previous_state=t.states[origin, i].tolist(), control=t.actions[origin, i].tolist(),
            baseline_control=baseline.actions[origin, i].tolist()) for i in worst])
    report['metrics'].append(row)
    print(json.dumps({k: v for k, v in row.items() if k != 'worst_first_month_oil'}), flush=True)
    (out / 'report.partial.json').write_text(json.dumps(report, indent=2) + '\n')
assert len(report['metrics']) == len(selected_ids) == 17
report['complete'] = True
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
