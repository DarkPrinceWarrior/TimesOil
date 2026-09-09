"""Compare TimesFM target/control layouts on a fixed-origin, full-field forecast."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmark_timesfm3 import MODEL_REVISION, forecast_inputs, metrics
from timesoil.aios.surrogate import _project_physics
from timesoil.aios.track2 import load_trajectory_dataset


def forecast_layout(forecaster, trajectory, origin, horizon, context, layout,
                    *, target_mode='rates', connectivity=None):
    target, cov = forecast_inputs(trajectory.states, trajectory.actions, origin, context, horizon)
    if target_mode == 'oil_fraction':
        target[:, 0] = np.divide(target[:, 0], target[:, 1], out=np.zeros_like(target[:, 0]),
                                 where=target[:, 1] > 1e-6).clip(0, 1)
    elif target_mode != 'rates':
        raise ValueError('unknown target representation')
    count, _, length = target.shape
    allocated = None
    if connectivity is not None:
        if tuple(connectivity.well_ids) != tuple(trajectory.well_ids):
            raise ValueError('connectivity well order differs from trajectory')
        controls = trajectory.actions[origin - length:origin + horizon]
        allocated = connectivity.features(np.zeros((len(controls) * count, 3)),
            controls.reshape(-1, controls.shape[-1]))[:, 0].reshape(len(controls), count).T
    groups = [np.arange(count)] if layout == 'joint' else [np.array([i]) for i in range(count)]
    histories = [target[group].reshape(-1, length) for group in groups]
    covariates = [cov[group, :-1].reshape(-1, length + horizon) for group in groups]
    if allocated is not None:
        covariates = [np.concatenate([c, allocated[group]], axis=0)
                      for group, c in zip(groups, covariates, strict=True)]
    if layout == 'well_with_field_injection':
        covariates = [np.concatenate([c, cov[0, -1:]], axis=0) for c in covariates]
    elif layout not in {'joint', 'well'}:
        raise ValueError('unknown layout')
    predicted = list(forecaster.predict_batch(histories, horizon=horizon,
        past_future_covariates=covariates, use_symmetric_averaging=False,
        make_positive=True, return_quantiles=False))
    output = np.empty((horizon, count, 3))
    for group, prediction in zip(groups, predicted, strict=True):
        output[:, group] = prediction.forecast.reshape(len(group), 3, horizon).transpose(2, 0, 1)
    if not np.isfinite(output).all():
        raise ValueError('non-finite TimesFM forecast')
    if target_mode == 'oil_fraction':
        output[..., 0] = output[..., 0].clip(0, 1) * np.maximum(output[..., 1], 0)
    return _project_physics(output, trajectory.actions[origin:origin + horizon], zero_injectors=True)[0]


def self_check():
    from types import SimpleNamespace
    class Forecaster:
        def predict_batch(self, histories, *, horizon, past_future_covariates, **_):
            for history, cov in zip(histories, past_future_covariates, strict=True):
                assert cov.shape[-1] == history.shape[-1] + horizon
                yield SimpleNamespace(forecast=np.repeat(history[:, -1:], horizon, axis=1))
    states = np.ones((12, 2, 3))
    states[:, 1] = [2, 3, 4]
    actions = np.ones((12, 2, 4)); actions[..., 0] = 100; actions[..., 3] = 70
    from timesoil.aios.interwell import WellConnectivity
    connectivity = WellConnectivity(('A', 'B'), np.array([[0, 1], [1, 0]]), np.ones((2, 3)), {})
    t = SimpleNamespace(states=states, actions=actions, well_ids=('A', 'B'))
    expected = np.repeat(states[5:6], 6, axis=0)
    for layout in ('joint', 'well', 'well_with_field_injection'):
        np.testing.assert_array_equal(forecast_layout(Forecaster(), t, 5, 6, 4, layout), expected)
        changed = states.copy(); changed[6:] = 1e9
        np.testing.assert_array_equal(forecast_layout(Forecaster(),
            SimpleNamespace(states=changed, actions=actions), 5, 6, 4, layout), expected)
        for target_mode in ('rates', 'oil_fraction'):
            np.testing.assert_allclose(forecast_layout(Forecaster(), t, 5, 6, 4, layout,
                target_mode=target_mode, connectivity=connectivity), expected, rtol=1e-6)
            np.testing.assert_allclose(forecast_layout(Forecaster(),
                SimpleNamespace(states=changed, actions=actions, well_ids=t.well_ids), 5, 6, 4, layout,
                target_mode=target_mode, connectivity=connectivity), expected, rtol=1e-6)
    print('Well identity, complete horizon and future-state leakage checks passed', flush=True)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--start', type=pd.Timestamp)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--target-mode', choices=('rates', 'oil_fraction'), default='rates')
    parser.add_argument('--connectivity', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.run or args.start is None or not args.horizon or args.horizon < 1 or not args.output:
        parser.error('run, start, positive horizon and new output directory required')
    csv = args.run / 'canonical/trajectory.csv'
    manifest = args.run / 'canonical/manifest.json'
    dataset = load_trajectory_dataset(csv, manifest=manifest)
    if len(dataset) != 1:
        raise ValueError('exactly one authenticated trajectory required')
    trajectory = dataset[0]
    connectivity = None
    if args.connectivity:
        from timesoil.aios.interwell import WellConnectivity
        connectivity = WellConnectivity.from_dict(json.loads(args.connectivity.read_text()))
        opm_manifest = json.loads((args.run / 'manifest.json').read_text())
        if connectivity.provenance['source_sha256'] != opm_manifest['source_sha256']:
            raise ValueError('connectivity belongs to a different source reservoir')
    origin = int(trajectory.dates.get_loc(args.start))
    context = min(128, origin)
    forecast_inputs(trajectory.states, trajectory.actions, origin, context, args.horizon)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('this scientific benchmark requires the allocated A100')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
        revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
    report = dict(schema='timesoil.timesfm-layout-benchmark/v1', model_revision=MODEL_REVISION,
        source_trajectory_sha256=sha256(csv.read_bytes()).hexdigest(),
        source_manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        well_count=len(trajectory.well_ids), context_months=context, horizon_months=args.horizon,
        target_mode=args.target_mode,
        connectivity_sha256=sha256(args.connectivity.read_bytes()).hexdigest() if args.connectivity else None,
        observation_cutoff=args.start.isoformat(), attention_backend='math',
        future_observations_used=False, calibrated=False, official_chdd=False, results=[])
    truth = trajectory.states[origin + 1:origin + args.horizon + 1]
    for layout in ('joint', 'well', 'well_with_field_injection'):
        torch.manual_seed(20260909)
        started = time.monotonic()
        prediction = forecast_layout(forecaster, trajectory, origin, args.horizon, context, layout,
                                     target_mode=args.target_mode, connectivity=connectivity)
        row = dict(layout=layout, seconds=time.monotonic() - started, **metrics(truth, prediction))
        report['results'].append(row)
        np.savez_compressed(args.output / f'{layout}.npz', prediction=prediction, truth=truth)
        print(json.dumps(row), flush=True)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
