"""Fit a local BHP discrepancy correction using calibration scenarios only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from timesoil.aios.surrogate import _project_physics
from timesoil.aios.track2 import load_trajectory_dataset


def bhp_features(actions, reference, degree):
    if actions.shape != reference.shape or not np.array_equal(actions[..., :3], reference[..., :3]):
        raise ValueError('local BHP correction requires unchanged rates, roles and statuses; full OPM required')
    active = reference[..., 2] == 1
    producer, injector = active & (reference[..., 1] != 2), active & (reference[..., 1] == 2)
    if not producer.any() or not injector.any() or (reference[..., 3][active] <= 0).any():
        raise ValueError('known producer and injector BHP limits required')
    additions = actions[..., 3][producer] - reference[..., 3][producer]
    factors = actions[..., 3][injector] / reference[..., 3][injector]
    if not np.allclose(additions, additions[0], rtol=0, atol=1e-5) or not np.allclose(factors, factors[0], rtol=0, atol=1e-7):
        raise ValueError('heterogeneous BHP change outside calibrated region; full OPM required')
    x, y = float(additions[0] / 15), float((1 - factors[0]) / .1)
    # Convex hull of the five calibration interventions and the physical reference.
    if not (-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 - .5 * x + 1e-6):
        raise ValueError('BHP change outside calibrated convex hull; full OPM required')
    if degree not in (1, 2):
        raise ValueError('unsupported local correction degree')
    return np.asarray([x, y] if degree == 1 else [x, y, x*x, x*y, y*y])


def self_check():
    reference = np.asarray([[[100., 0., 1., 50.], [100., 2., 1., 300.]]])
    changed = reference.copy(); changed[0, 0, 3] += 15; changed[0, 1, 3] *= .95
    np.testing.assert_allclose(bhp_features(changed, reference, 2), [1, .5, 1, .5, .25])
    np.testing.assert_array_equal(bhp_features(reference, reference, 1), [0, 0])
    changed[0, 0, 3] += 5
    try:
        bhp_features(changed, reference, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-domain BHP intervention accepted')
    print('Local BHP identity, feature scale and OOD rejection passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--training-report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not all((args.pilot, args.batch, args.reference, args.training_report, args.output)):
        parser.error('pilot, batch, reference, training-report and output required')
    digest = lambda p: sha256(p.read_bytes()).hexdigest()
    pilot = json.loads((args.pilot / 'report.json').read_text())
    training = json.loads(args.training_report.read_text())
    batch = json.loads((args.batch / 'manifest.json').read_text())
    assert pilot['complete'] and pilot['calibration_only']
    assert pilot['batch_sha256'] == digest(args.batch / 'manifest.json')
    assert pilot['head_sha256'] == training['checkpoint_sha256']
    assert pilot['training_report_sha256'] == digest(args.training_report)
    assert pilot['reference_manifest_sha256'] == digest(args.reference / 'manifest.json')
    assert set(pilot['source_scenarios']) == {str(i) for i in batch['calibration_cases']}
    reference = load_trajectory_dataset(args.reference / 'trajectory.csv', manifest=args.reference / 'manifest.json')[0]
    assert reference.content_hash == pilot['reference_trajectory_sha256']
    origin = int(reference.dates.get_loc('2007-01-01'))
    months = 224
    reference_actions = reference.actions[origin:origin + months]
    actions, truths, anchors, physical = [], [], [], []
    array_hashes = {}
    for index in batch['calibration_cases']:
        root = args.batch / f'candidate-{index:02d}'
        trajectory = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')[0]
        assert trajectory.content_hash == pilot['source_scenarios'][str(index)]
        data = np.load(args.pilot / f'candidate-{index:02d}.npz', allow_pickle=False)
        array_hashes[str(index)] = digest(args.pilot / f'candidate-{index:02d}.npz')
        np.testing.assert_array_equal(data['truth'], trajectory.states[origin + 1:origin + months + 1])
        actions.append(trajectory.actions[origin:origin + months])
        truths.append(data['truth']); anchors.append(data['trained_reference_delta_224'])
        physical.append(data['reference_only_224'])
    truths, anchors = np.asarray(truths), np.asarray(anchors)
    residuals = (truths - anchors).reshape(len(truths), -1)
    scale = np.asarray(training['training_scale'])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'timesoil.local-timesfm-reference-correction/v1', 'degrees_evaluated': [1, 2],
        'selection': 'leave-one-calibration-scenario-out normalized absolute error',
        'head_sha256': training['checkpoint_sha256'], 'reference_manifest_sha256': pilot['reference_manifest_sha256'],
        'pilot_report_sha256': digest(args.pilot / 'report.json'), 'calibration_batch_sha256': pilot['batch_sha256'],
        'calibration_cases': batch['calibration_cases'], 'test_cases_used': [],
        'source_scenarios': pilot['source_scenarios'], 'cv': [], 'full_opm_required_outside_region': True,
        'pilot_array_sha256': array_hashes, 'training_report_sha256': digest(args.training_report),
        'uncorrected_delta_score': float(np.mean(np.abs(anchors - truths) / scale)),
        'physical_reference_score': float(np.mean(np.abs(np.asarray(physical) - truths) / scale)),
        'domain': 'same source, origin, rates, roles and statuses; uniform producer BHP addition x*15 bar and injector reduction y*0.1; 0<=x<=1, 0<=y<=1-x/2',
        'independent_accuracy_certified': False}
    (args.output / 'protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    for degree in (1, 2):
        design = np.stack([bhp_features(a, reference_actions, degree) for a in actions])
        predictions = []
        for i in range(len(design)):
            keep = np.arange(len(design)) != i
            coefficients = np.linalg.lstsq(design[keep], residuals[keep], rcond=None)[0]
            corrected = anchors[i] + (design[i] @ coefficients).reshape(truths[i].shape)
            predictions.append(_project_physics(corrected, actions[i], zero_injectors=True)[0])
        score = float(np.mean(np.abs(np.asarray(predictions) - truths) / scale))
        report['cv'].append({'degree': degree, 'score': score})
        # Ablation: fit the same discrepancy family without a Google response delta.
        physical_residuals = (truths - np.asarray(physical)).reshape(len(truths), -1)
        ablated = []
        for i in range(len(design)):
            keep = np.arange(len(design)) != i
            beta = np.linalg.lstsq(design[keep], physical_residuals[keep], rcond=None)[0]
            corrected = physical[i] + (design[i] @ beta).reshape(truths[i].shape)
            ablated.append(_project_physics(corrected, actions[i], zero_injectors=True)[0])
        report['cv'][-1]['without_google_delta_score'] = float(np.mean(np.abs(np.asarray(ablated) - truths) / scale))
    degree = min(report['cv'], key=lambda row: (row['score'], row['degree']))['degree']
    design = np.stack([bhp_features(a, reference_actions, degree) for a in actions])
    coefficients = np.linalg.lstsq(design, residuals, rcond=None)[0].reshape(len(design[0]), *truths.shape[1:])
    checkpoint = args.output / 'correction.npz'
    np.savez_compressed(checkpoint, coefficients=coefficients)
    report.update(complete=True, degree=degree, checkpoint_sha256=digest(checkpoint), script_sha256=digest(Path(__file__)))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
