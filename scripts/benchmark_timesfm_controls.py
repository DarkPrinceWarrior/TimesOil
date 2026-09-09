"""A100-only, pretrained TimesFM BHP ablation on a verified full-period OPM batch."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from benchmark_bhp_surrogate import START, END, MONTHS, management_window
from benchmark_horizons import forecast_blocks, self_check
from benchmark_timesfm3 import MODEL_REVISION, metrics
from timesoil.aios.track2 import MODEL_Z_SOURCE_SHA256, load_trajectory_dataset


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verified_batch(batch, expected_hash):
    if digest(batch / 'manifest.json') != expected_hash:
        raise ValueError('batch manifest hash mismatch')
    manifest = json.loads((batch / 'manifest.json').read_text())
    records = manifest['scenarios']
    if (manifest['official_source_sha256'] != MODEL_Z_SOURCE_SHA256
            or manifest['scenario_count'] != 10 or len(records) != 10
            or len({r['scenario_id'] for r in records}) != 10):
        raise ValueError('ten distinct official Model Z scenarios required')
    for record in records:
        for name in ('dataset', 'export_manifest', 'run_manifest', 'canonical_chdd'):
            path = (batch / record[name]).resolve()
            if not path.is_relative_to(batch.resolve()) or digest(path) != record[name + '_sha256']:
                raise ValueError(f'scenario artifact mismatch: {name}')
    full = load_trajectory_dataset(batch / 'dataset', manifest=batch / 'manifests')
    if not full.model_z_identity or {t.scenario_id for t in full} != {r['scenario_id'] for r in records}:
        raise ValueError('verified dataset does not match the batch')
    management_window(full)  # Existing complete-period and four-control contract.
    baseline = next(t for t in full if t.scenario_id == 'baseline')
    origin = int(baseline.dates.get_loc(START))
    if origin < 128 or len(baseline.well_ids) != 103:
        raise ValueError('128 historical months and 103 wells required')
    for item in full:
        if not item.dates.equals(baseline.dates) or item.well_ids != baseline.well_ids:
            raise ValueError('scenario grids differ')
        np.testing.assert_allclose(item.states[:origin + 1], baseline.states[:origin + 1], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(item.actions[:origin], baseline.actions[:origin])
    return full, origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--batch-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.batch or not args.batch_sha256 or not args.output:
        parser.error('batch, batch-sha256 and output required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trajectories, origin = verified_batch(args.batch, args.batch_sha256)
    print('Verified ten complete OPM trajectories with identical history.', flush=True)
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster

    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('requires the allocated A100 GPU')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    model = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
        revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
    report = {
        'schema': 'timesoil.timesfm-controls-benchmark/v1',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'batch_manifest_sha256': args.batch_sha256,
        'source_sha256': MODEL_Z_SOURCE_SHA256,
        'source_scenario_hashes': {t.scenario_id: t.content_hash for t in trajectories},
        'executed_sources': {name: digest(Path(__file__).with_name(name)) for name in
            ('benchmark_timesfm_controls.py', 'benchmark_horizons.py', 'benchmark_timesfm3.py')},
        'model_revision': MODEL_REVISION, 'timesfm_version': version('timesfm'),
        'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(0),
        'start': str(START.date()), 'end_exclusive': str(END.date()),
        'horizon_months': MONTHS, 'context_months': 128, 'well_count': 103,
        'locally_trained': False, 'is_new_optimization_result': False,
        'claim': 'Pretrained forecast ablation. Observed-update rows use fresh OPM states. '
                 'Rate and BHP changed together in OPM; no isolated BHP causal claim.',
        'results': [],
    }
    for trajectory in trajectories:
        truth = trajectory.states[origin + 1:origin + MONTHS + 1]
        outputs = {'truth': truth}
        for bhp in (False, True):
            item = trajectory if bhp else replace(trajectory, actions=trajectory.actions[..., :3])
            for name, block, observe in (('fixed_origin_direct', MONTHS, False),
                                         ('observed_update_block_6', 6, True)):
                torch.cuda.synchronize()
                tick = time.monotonic()
                prediction = forecast_blocks(model, item, origin, MONTHS, 128, block, observe=observe)
                torch.cuda.synchronize()
                key = f'{name}_bhp_{bhp}'
                outputs[key] = prediction
                row = dict(scenario_id=item.scenario_id, name=name, bhp_channel=bhp,
                    control_channels=515 if bhp else 412, new_observations_between_blocks=observe,
                    seconds=time.monotonic() - tick, **metrics(truth, prediction),
                    prefix_metrics={str(h): metrics(truth[:h], prediction[:h]) for h in (6, 23, 60, 120, 224)})
                report['results'].append(row)
                (args.output / 'metrics.partial.json').write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps({k: v for k, v in row.items() if k != 'prefix_metrics'}), flush=True)
        np.savez_compressed(args.output / f'{trajectory.scenario_id}.npz', **outputs)
    report['seconds_total'] = time.monotonic() - started
    report['complete'] = True
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
