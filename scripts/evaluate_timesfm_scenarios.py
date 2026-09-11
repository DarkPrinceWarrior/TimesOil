"""Evaluate frozen Google weights on an independently exported Model Z scenario set."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmark_bhp_surrogate import START, MONTHS
from timesoil.aios.interwell import WellConnectivity
from timesoil.aios.track2 import MODEL_Z_SOURCE_SHA256, load_trajectory_dataset
from timesfm_economics import (ECONOMIC_TARGETS, ECONOMIC_UNITS, economic_metrics,
    economic_targets, forecast_economic, observed_economic_history)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def interval_check(calibration_errors, test_errors):
    # ponytail: five fixed scenario groups; report empirical coverage, not exchangeability or deployment certification.
    calibration_errors, test_errors = np.asarray(calibration_errors), np.asarray(test_errors)
    targets = len(ECONOMIC_TARGETS)
    if (calibration_errors.ndim != 4 or test_errors.ndim != 4
            or calibration_errors.shape[0] != 5 or test_errors.shape[0] != 3
            or calibration_errors.shape[1:] != test_errors.shape[1:]
            or calibration_errors.shape[-1] != targets
            or not all(np.isfinite(x).all() and (x >= 0).all() and x.size
                       for x in (calibration_errors, test_errors))):
        raise ValueError('finite nonnegative errors for five calibration and three test trajectory groups required')
    radius = np.max(calibration_errors, axis=(0, 1, 2))
    covered = test_errors <= radius
    return {'nominal_group_coverage': .8, 'calibration_groups': len(calibration_errors),
        'radius_by_target': dict(zip(ECONOMIC_TARGETS, radius.tolist())),
        'units_by_target': dict(zip(ECONOMIC_TARGETS, ECONOMIC_UNITS)),
        'nominal_coverage_scope': 'Per target across one whole field trajectory; not joint across targets.',
        'test_pointwise_coverage': covered.mean(axis=(0, 1, 2)).tolist(),
        'test_whole_trajectory_coverage_by_target': covered.all(axis=(1, 2)).mean(axis=0).tolist(),
        'test_whole_trajectory_joint_coverage': float(covered.all(axis=(1, 2, 3)).mean()),
        'guaranteed_coverage_claimed': False,
        'limitation': 'Fixed intervention scenarios are not established as exchangeable; only three independent test groups.'}


def economic_evaluation_inputs(root, manifest, trajectory, origin, months, development_hashes):
    """Keep scoring truth separate from inference; economic development hashes are CSV hashes."""
    expected = manifest['outputs']['chdd_csv']['sha256']
    if expected in development_hashes.values():
        raise ValueError('economic evaluation CSV was already used during training/development')
    _, inputs = observed_economic_history(root, manifest, trajectory, origin)
    raw = (root / 'chdd.csv').read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError('economic scoring CSV hash mismatch')
    stamps = trajectory.dates[origin + 1:origin + months + 1].strftime('%Y-%m-%d')
    endpoints = set(stamps)
    truth = economic_targets((row for row in csv.DictReader(raw.decode('utf-8-sig').splitlines())
                              if row['DATA'] in endpoints), stamps, trajectory.well_ids)
    return inputs, truth, expected


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
    errors = np.ones((5, 2, 2, 9)); errors[4, 0, 0] = [2, 3, 4, 1, 1, 1, 1, 1, 1]
    held_out = np.full((3, 2, 2, 9), [2., 4., 3., 1., 1., 1., 1., 1., 1.])
    check = interval_check(errors, held_out)
    assert list(check['radius_by_target']) == list(ECONOMIC_TARGETS)
    assert list(check['radius_by_target'].values()) == [2, 3, 4, 1, 1, 1, 1, 1, 1]
    assert check['units_by_target'] == dict(zip(ECONOMIC_TARGETS, ECONOMIC_UNITS))
    assert check['test_whole_trajectory_coverage_by_target'] == [1, 0, 1] + [1] * 6
    for invalid in (held_out[..., :3], held_out * np.nan, -held_out, held_out[:2]):
        try:
            interval_check(errors, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid or three-target error groups accepted')
    manifest = dict(schema='timesoil.bhp-only-forecast-evaluation/v1', complete=True,
        calibration_cases=[0, 1, 3, 4, 6], test_cases=[2, 5, 7], model_selection_allowed_on_test=False,
        scenarios=[{'index': i} for i in range(8)])
    validate_split(manifest)
    for invalid in ({'test_cases': [3, 5, 6]}, {'schema': 'timesoil.model-y-forecast-evaluation/v1'}):
        try:
            validate_split({**manifest, **invalid})
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
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--calibration-only', action='store_true')
    split_group.add_argument('--test-only', action='store_true',
        help='Evaluate only held-out cases; do not reuse development cases for intervals')
    parser.add_argument('--fixed-origin-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not all((args.batch, args.batch_sha256, args.head_report, args.head, args.connectivity, args.output)):
        parser.error('batch, hash, frozen head/report, connectivity and output required')
    if not args.fixed_origin_only:
        parser.error('the nine-target economic evaluation never observes the candidate future; pass --fixed-origin-only')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from timesfm_geology import MODEL_REVISION, load_frozen_model
    if digest(args.batch / 'manifest.json') != args.batch_sha256:
        raise ValueError('evaluation batch hash mismatch')
    manifest = json.loads((args.batch / 'manifest.json').read_text())
    training = json.loads(args.head_report.read_text())
    if tuple(training.get('economic_targets') or ()) != ECONOMIC_TARGETS:
        raise ValueError('fixed-origin Model Z evaluation requires the nine canonical economic targets')
    validate_split(manifest)
    if (training.get('complete') is not True or training['model_revision'] != MODEL_REVISION
            or digest(args.head) != training['checkpoint_sha256']
            or digest(args.connectivity) != training['connectivity_sha256']):
        raise ValueError('frozen model, training report and geology do not match')
    connectivity = WellConnectivity.from_dict(json.loads(args.connectivity.read_text()))
    if connectivity.provenance['source_sha256'] != MODEL_Z_SOURCE_SHA256:
        raise ValueError('official reservoir geology required')
    months, start = MONTHS, START
    mode = 'trained_economic_fixed_origin_224'
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'timesoil.independent-timesfm-evaluation/v1', 'batch_sha256': args.batch_sha256,
        'model_revision': MODEL_REVISION, 'head_sha256': training['checkpoint_sha256'],
        'training_report_sha256': digest(args.head_report), 'connectivity_sha256': digest(args.connectivity),
        'calibration_cases': manifest['calibration_cases'], 'test_cases': manifest['test_cases'],
        'model_selection_allowed_on_test': False, 'head_retraining_allowed_after_test': False,
        'source_sha256': MODEL_Z_SOURCE_SHA256, 'horizon_months': months,
        'modes_fixed_before_evaluation': [mode], 'calibration_only': args.calibration_only,
        'test_only': args.test_only,
        'simulated_reference_future_used': False, 'candidate_future_observations_used_for_fixed_origin': False,
        'reference_manifest_sha256': None,
        'reference_correction_sha256': None,
        'head_validation_loss': training['validation_loss_best'], 'metrics': [], 'source_scenarios': {},
        'economic_targets': list(ECONOMIC_TARGETS), 'target_units': list(ECONOMIC_UNITS),
        'source_scenario_hash_semantics': 'canonical economic CSV SHA-256'}
    (args.output / 'protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster
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
    selected = torch.load(args.head, map_location='cuda', weights_only=True)
    report['checkpoint_requires_physical_reference'] = bool(selected.get('reference_manifest_sha256'))
    forecaster.model = load_frozen_model(forecaster.model, connectivity, selected)
    if forecaster.model.output_head.target_count != len(ECONOMIC_TARGETS):
        raise ValueError('checkpoint and economic evaluation target counts differ')
    errors = {}
    for record in manifest['scenarios']:
        index = record['index']
        if args.calibration_only and index not in manifest['calibration_cases']:
            continue
        if args.test_only and index not in manifest['test_cases']:
            continue
        root = Path(record['directory']).resolve()
        if root != (args.batch / f'candidate-{index:02d}').resolve():
            raise ValueError('evaluation scenario path mismatch')
        if (digest(root / 'trajectory.csv') != record['trajectory_sha256']
                or digest(root / 'manifest.json') != record['export_manifest_sha256']):
            raise ValueError('evaluation scenario hash mismatch')
        dataset = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')
        if (len(dataset) != 1 or not dataset.model_z_identity
                or json.loads((root / 'manifest.json').read_text())['provenance']['opm_source_sha256'] != MODEL_Z_SOURCE_SHA256
                or dataset[0].well_ids != connectivity.well_ids):
            raise ValueError('evaluation reservoir or well order mismatch')
        t = dataset[0]
        if t.content_hash in training['source_scenario_hashes'].values():
            raise ValueError('evaluation trajectory was already used during training/development')
        origin = int(t.dates.get_loc(start))
        context = min(128, origin)
        if t.actions.shape[-1] != 4 or origin < 128 or len(t.states[origin + 1:origin + months + 1]) != months:
            raise ValueError('complete historical/control/target grid required')
        economic_input, truth, scenario_hash = economic_evaluation_inputs(root,
            json.loads((root / 'manifest.json').read_text()), t, origin, months,
            training['source_scenario_hashes'])
        report['source_scenarios'][str(index)] = scenario_hash
        prediction = forecast_economic(forecaster, economic_input, origin, months, context, connectivity)
        errors[index] = np.abs(prediction - truth)
        row = {'index': index, 'split': 'test' if index in manifest['test_cases'] else 'calibration',
            'mode': mode, **economic_metrics(truth, prediction),
            'first_month': economic_metrics(truth[:1], prediction[:1])}
        report['metrics'].append(row)
        print(json.dumps(row), flush=True)
        np.savez_compressed(args.output / f'candidate-{index:02d}.npz', truth=truth, **{mode: prediction})
        (args.output / 'report.partial.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.test_only:
        assert set(report['source_scenarios']) == {str(i) for i in manifest['test_cases']}
    report['intervals'] = {} if args.calibration_only or args.test_only else {
        mode: interval_check(np.stack([errors[i] for i in manifest['calibration_cases']]),
                             np.stack([errors[i] for i in manifest['test_cases']]))}
    report.update(complete=True, seconds=time.monotonic() - started, is_optimization_result=False,
        script_sha256=digest(Path(__file__)), independently_certified_for_surrogate_control=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
