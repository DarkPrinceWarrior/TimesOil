"""Evaluate frozen Google weights on an independently exported Model Z scenario set."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmark_bhp_surrogate import START, MONTHS
from benchmark_timesfm3 import MODEL_REVISION, metrics
from benchmark_timesfm_layouts import forecast_layout
from timesoil.aios.interwell import WellConnectivity
from timesoil.aios.track2 import MODEL_Z_SOURCE_SHA256, load_trajectory_dataset


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def interval_check(calibration_errors, test_errors):
    # ponytail: five fixed scenario groups; report empirical coverage, not exchangeability or deployment certification.
    radius = np.max(calibration_errors, axis=(0, 1, 2))
    covered = test_errors <= radius
    return {'nominal_group_coverage': .8, 'calibration_groups': len(calibration_errors),
        'radius_oil_tpd_liquid_tpd_pressure_bar': radius.tolist(),
        'test_pointwise_coverage': covered.mean(axis=(0, 1, 2)).tolist(),
        'test_whole_trajectory_coverage_by_target': covered.all(axis=(1, 2)).mean(axis=0).tolist(),
        'guaranteed_coverage_claimed': False,
        'limitation': 'Fixed intervention scenarios are not established as exchangeable; only three independent test groups.'}


def validate_split(manifest):
    splits = {
        'timesoil.frozen-forecast-evaluation-cases/v1': ([0, 1, 2, 4, 7], [3, 5, 6]),
        'timesoil.bhp-only-forecast-evaluation/v1': ([0, 1, 3, 4, 6], [2, 5, 7]),
    }
    if (manifest.get('complete') is not True or manifest.get('schema') not in splits
            or (manifest.get('calibration_cases'), manifest.get('test_cases')) != splits[manifest['schema']]
            or manifest.get('model_selection_allowed_on_test') is not False
            or [r['index'] for r in manifest['scenarios']] != list(range(8))):
        raise ValueError('frozen five-calibration/three-test scenario split required')


def self_check():
    errors = np.ones((5, 2, 2, 3)); errors[4, 0, 0] = [2, 3, 4]
    check = interval_check(errors, np.full((3, 2, 2, 3), [2, 4, 3]))
    assert check['radius_oil_tpd_liquid_tpd_pressure_bar'] == [2, 3, 4]
    assert check['test_whole_trajectory_coverage_by_target'] == [1, 0, 1]
    manifest = dict(schema='timesoil.bhp-only-forecast-evaluation/v1', complete=True,
        calibration_cases=[0, 1, 3, 4, 6], test_cases=[2, 5, 7], model_selection_allowed_on_test=False,
        scenarios=[{'index': i} for i in range(8)])
    validate_split(manifest)
    manifest['test_cases'] = [3, 5, 6]
    try:
        validate_split(manifest)
    except ValueError:
        pass
    else:
        raise AssertionError('changed scenario split accepted')
    print('Scenario-group interval and independent target coverage checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--batch-sha256')
    parser.add_argument('--head-report', type=Path)
    parser.add_argument('--head', type=Path)
    parser.add_argument('--connectivity', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not all((args.batch, args.batch_sha256, args.head_report, args.head, args.connectivity, args.output)):
        parser.error('batch, hash, frozen head/report, connectivity and output required')
    if digest(args.batch / 'manifest.json') != args.batch_sha256:
        raise ValueError('evaluation batch hash mismatch')
    manifest = json.loads((args.batch / 'manifest.json').read_text())
    training = json.loads(args.head_report.read_text())
    validate_split(manifest)
    if (training.get('complete') is not True or training['model_revision'] != MODEL_REVISION
            or digest(args.head) != training['checkpoint_sha256']
            or digest(args.connectivity) != training['connectivity_sha256']):
        raise ValueError('frozen model, training report and geology do not match')
    connectivity = WellConnectivity.from_dict(json.loads(args.connectivity.read_text()))
    if connectivity.provenance['source_sha256'] != MODEL_Z_SOURCE_SHA256:
        raise ValueError('official Model Z geology required')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'timesoil.independent-timesfm-evaluation/v1', 'batch_sha256': args.batch_sha256,
        'model_revision': MODEL_REVISION, 'head_sha256': training['checkpoint_sha256'],
        'training_report_sha256': digest(args.head_report), 'connectivity_sha256': digest(args.connectivity),
        'calibration_cases': manifest['calibration_cases'], 'test_cases': manifest['test_cases'],
        'model_selection_allowed_on_test': False, 'head_retraining_allowed_after_test': False,
        'modes_fixed_before_evaluation': ['trained_fixed_origin_224', 'pretrained_observed_update_1'],
        'head_validation_loss': training['validation_loss_best'], 'metrics': [], 'source_scenarios': {}}
    (args.output / 'protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster
    from timesfm_geology import StaticConditionedHead, load_selected_layer
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('requires allocated A100')
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
    model = forecaster.model
    original_head = model.output_head
    original_layer = model.transformer_stack.layers[-1]
    selected = torch.load(args.head, map_location='cuda', weights_only=True)
    selected_head = StaticConditionedHead(deepcopy(original_head), connectivity)
    weights = selected.get('output_head', selected)
    torch.testing.assert_close(weights['features'], selected_head.features, rtol=0, atol=0)
    selected_head.load_state_dict(weights)
    selected_layer = load_selected_layer(deepcopy(original_layer), selected_head, selected)
    errors = {'trained_fixed_origin_224': {}, 'pretrained_observed_update_1': {}}
    for record in manifest['scenarios']:
        index = record['index']
        root = Path(record['directory']).resolve()
        if root != (args.batch / f'candidate-{index:02d}').resolve():
            raise ValueError('evaluation scenario path mismatch')
        if (digest(root / 'trajectory.csv') != record['trajectory_sha256']
                or digest(root / 'manifest.json') != record['export_manifest_sha256']):
            raise ValueError('evaluation scenario hash mismatch')
        dataset = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')
        if len(dataset) != 1 or not dataset.model_z_identity or dataset[0].well_ids != connectivity.well_ids:
            raise ValueError('evaluation reservoir or well order mismatch')
        t = dataset[0]
        if t.content_hash in training['source_scenario_hashes'].values():
            raise ValueError('evaluation trajectory was already used during training/development')
        origin = int(t.dates.get_loc(START))
        if t.actions.shape[-1] != 4 or origin < 128 or len(t.states[origin + 1:origin + MONTHS + 1]) != MONTHS:
            raise ValueError('complete historical/control/target grid required')
        truth = t.states[origin + 1:origin + MONTHS + 1]
        report['source_scenarios'][str(index)] = t.content_hash
        outputs = {'truth': truth}
        for mode in errors:
            model.output_head = selected_head if mode.startswith('trained') else original_head
            model.transformer_stack.layers[-1] = selected_layer if mode.startswith('trained') else original_layer
            if mode.startswith('trained'):
                prediction = forecast_layout(forecaster, t, origin, MONTHS, 128, 'joint', connectivity=connectivity)
            else:
                prediction = np.concatenate([forecast_layout(forecaster, t, origin + offset, 1, 128,
                    'joint', connectivity=connectivity) for offset in range(MONTHS)])
            outputs[mode] = prediction
            errors[mode][index] = np.abs(prediction - truth)
            row = {'index': index, 'split': 'test' if index in manifest['test_cases'] else 'calibration',
                'mode': mode, **metrics(truth, prediction)}
            report['metrics'].append(row)
            print(json.dumps(row), flush=True)
        naive = t.states[origin:origin + MONTHS]
        report['metrics'].append({'index': index, 'mode': 'naive_observed_update_1', **metrics(truth, naive)})
        np.savez_compressed(args.output / f'candidate-{index:02d}.npz', **outputs)
        (args.output / 'report.partial.json').write_text(json.dumps(report, indent=2) + '\n')
    report['intervals'] = {mode: interval_check(np.stack([e[i] for i in manifest['calibration_cases']]),
        np.stack([e[i] for i in manifest['test_cases']])) for mode, e in errors.items()}
    report.update(complete=True, seconds=time.monotonic() - started, is_optimization_result=False,
        script_sha256=digest(Path(__file__)), independently_certified_for_surrogate_control=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
