"""Adapt only TimesFM 3.0's output head on whole, provenance-verified OPM scenarios."""

from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from benchmark_horizons import forecast_blocks
from benchmark_timesfm3 import MODEL_REVISION, forecast_inputs, metrics
from benchmark_timesfm_controls import verified_batch


def pinball_loss(prediction, target, scale, quantiles):
    error = (target[..., None] - prediction) / scale[..., None]
    return (error * quantiles).maximum(error * (quantiles - 1)).mean()


def project_quantiles(prediction, actions):
    """Differentiable version of the inference projection, for every quantile."""
    import torch
    values = prediction.reshape(1, actions.shape[1], 3, actions.shape[0], -1).clamp_min(0)
    active = ((actions[..., 2] == 1) & (actions[..., 1] != 2)).T[None, ..., None]
    oil, liquid, pressure = values.unbind(dim=2)
    return torch.stack([torch.minimum(oil, liquid) * active, liquid * active, pressure], dim=2).reshape_as(prediction)


def self_check():
    import torch
    predicted = torch.zeros((1, 1, 1, 2), requires_grad=True)
    loss = pinball_loss(predicted, torch.ones((1, 1, 1)), torch.ones((1, 1, 1)),
                        torch.tensor([.25, .75]))
    torch.testing.assert_close(loss, torch.tensor(.5))
    loss.backward()
    torch.testing.assert_close(predicted.grad.flatten(), torch.tensor([-.125, -.375]))
    actions = torch.tensor([[[100., 0, 1, 50], [100., 2, 1, 300.]]])
    values = torch.tensor([[[[4., 5.]], [[2., 3.]], [[-1., 5.]], [[8., 9.]], [[9., 10.]], [[100., 110.]]]], requires_grad=True)
    projected = project_quantiles(values, actions)
    torch.testing.assert_close(projected.flatten(), torch.tensor([2., 3., 2., 3., 0., 5., 0., 0., 0., 0., 100., 110.]))
    physical = torch.ones((1, 6, 1, 1))
    anchored = physical + projected - projected[..., :1]
    torch.testing.assert_close(anchored[..., :1], physical)
    anchored[..., :1].sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))
    print('normalized quantile loss and gradients verified', flush=True)


def verified_y_batch(batch, expected_hash):
    import pandas as pd
    from timesoil.aios.track2 import load_trajectory_dataset
    if sha256((batch / 'manifest.json').read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Model Y batch manifest hash mismatch')
    manifest = json.loads((batch / 'manifest.json').read_text())
    if (manifest.get('complete') is not True or manifest.get('start') != '2014-01-01'
            or manifest.get('horizon_months') != 23
            or manifest.get('train_cases') != [0, 1, 2, 4, 8]
            or manifest.get('validation_cases') != [5] or manifest.get('test_cases') != [3, 6, 7]
            or [r['index'] for r in manifest['scenarios']] != list(range(9))):
        raise ValueError('nine complete Model Y scenarios and the frozen split required')
    trajectories = []
    for record in manifest['scenarios']:
        root = Path(record['directory']).resolve()
        if root != (batch / f"candidate-{record['index']:02d}").resolve():
            raise ValueError('Model Y scenario directory mismatch')
        for name, key in [('trajectory.csv', 'trajectory_sha256'), ('manifest.json', 'export_manifest_sha256')]:
            if sha256((root / name).read_bytes()).hexdigest() != record[key]:
                raise ValueError('Model Y scenario artifact hash mismatch')
        metadata = json.loads((root / 'manifest.json').read_text())
        if metadata['provenance']['opm_source_sha256'] != manifest['official_source_sha256']:
            raise ValueError('Model Y scenarios have different reservoir sources')
        data = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')
        if len(data) != 1 or data[0].scenario_id != f"physical-sweep-{record['index']:02d}":
            raise ValueError('Model Y scenario identity mismatch')
        trajectories.append(data[0])
    baseline = trajectories[0]
    origin = int(baseline.dates.get_loc(pd.Timestamp(manifest['start'])))
    expected = pd.date_range('2014-01-01', periods=24, freq='MS')
    for item in trajectories:
        if (len(item.well_ids) != 49 or item.well_ids != baseline.well_ids or origin < 48
                or item.actions.shape[-1] != 4 or not item.dates.equals(baseline.dates)
                or not item.dates[origin:origin + 24].equals(expected)):
            raise ValueError('Model Y historical or management grid mismatch')
        np.testing.assert_allclose(item.states[:origin + 1], baseline.states[:origin + 1], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(item.actions[:origin], baseline.actions[:origin])
    return trajectories, origin


def verified_regime_calibration(batch, expected_hash, baseline, origin, *, model_y_source_sha256=None):
    from timesoil.aios.track2 import load_trajectory_dataset
    if sha256((batch / 'manifest.json').read_bytes()).hexdigest() != expected_hash:
        raise ValueError('regime calibration manifest hash mismatch')
    manifest = json.loads((batch / 'manifest.json').read_text())
    model_y = model_y_source_sha256 is not None
    allowed = ('timesoil.model-y-forecast-evaluation/v1',) if model_y else (
        'timesoil.bhp-only-forecast-evaluation/v1', 'timesoil.frozen-forecast-evaluation-cases/v1')
    if manifest.get('schema') not in allowed:
        raise ValueError('calibration schema does not match the training reservoir')
    bhp = model_y or manifest.get('schema') == 'timesoil.bhp-only-forecast-evaluation/v1'
    calibration, test = ([0, 1, 3, 4, 6], [2, 5, 7]) if bhp else ([0, 1, 2, 4, 7], [3, 5, 6])
    if (manifest.get('complete') is not True or manifest['calibration_cases'] != calibration
            or manifest['test_cases'] != test or manifest['model_selection_allowed_on_test'] is not False):
        raise ValueError('only the five previously designated calibration cases may augment training')
    extra = []
    for record in manifest['scenarios']:
        if record['index'] not in manifest['calibration_cases']:
            continue
        root = Path(record['directory']).resolve()
        if root != (batch / f"candidate-{record['index']:02d}").resolve():
            raise ValueError('regime calibration directory mismatch')
        for name, key in [('trajectory.csv', 'trajectory_sha256'), ('manifest.json', 'export_manifest_sha256')]:
            if sha256((root / name).read_bytes()).hexdigest() != record[key]:
                raise ValueError('regime calibration artifact hash mismatch')
        data = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')
        if (len(data) != 1 or not model_y and not data.model_z_identity
                or model_y and json.loads((root / 'manifest.json').read_text())['provenance']['opm_source_sha256'] != model_y_source_sha256):
            raise ValueError('regime calibration requires the authenticated training reservoir')
        item = data[0]
        prefix = 'forecast-validation' if model_y else 'bhp-only' if bhp else 'physical-sweep'
        if (item.well_ids != baseline.well_ids or not item.dates.equals(baseline.dates)
                or item.actions.shape != baseline.actions.shape
                or item.scenario_id != f"{prefix}-{record['index']:02d}"):
            raise ValueError('regime calibration grid differs from the training reservoir')
        np.testing.assert_allclose(item.states[:origin + 1], baseline.states[:origin + 1], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(item.actions[:origin], baseline.actions[:origin])
        extra.append(item)
    if len(extra) != 5:
        raise ValueError('five distinct calibration scenarios required')
    return extra


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--batch-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--connectivity', type=Path)
    parser.add_argument('--initial-head', type=Path)
    parser.add_argument('--initial-head-sha256')
    parser.add_argument('--unfreeze-last-layer', action='store_true')
    parser.add_argument('--unfreeze-backbone', action='store_true')
    parser.add_argument('--model-y', action='store_true')
    parser.add_argument('--condition-last-layer', action='store_true')
    parser.add_argument('--regime-calibration', type=Path)
    parser.add_argument('--regime-calibration-sha256')
    parser.add_argument('--bhp-calibration', type=Path)
    parser.add_argument('--bhp-calibration-sha256')
    parser.add_argument('--monthly-observed-training', action='store_true')
    parser.add_argument('--intervention-repeats', type=int, default=1, help='Repeat first intervention windows when training the monthly model')
    parser.add_argument('--model-y-calibration', type=Path)
    parser.add_argument('--model-y-calibration-sha256')
    parser.add_argument('--reference', type=Path, help='Verified physical reference for additive-response training')
    parser.add_argument('--reference-sha256')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.batch or not args.batch_sha256 or not args.output or not 1 <= args.epochs <= 100:
        parser.error('batch, batch-sha256, output and 1..100 epochs required')
    if not np.isfinite(args.learning_rate) or not 1e-7 <= args.learning_rate <= 1e-3:
        parser.error('learning-rate must be in [1e-7, 1e-3]')
    if bool(args.initial_head) != bool(args.initial_head_sha256):
        parser.error('initial-head and its SHA-256 must be supplied together')
    if args.condition_last_layer and not args.connectivity:
        parser.error('static last-layer conditioning requires verified connectivity')
    if args.monthly_observed_training and not args.model_y:
        parser.error('monthly observed training currently requires Model Y')
    if not 1 <= args.intervention_repeats <= 23 or args.intervention_repeats != 1 and not args.monthly_observed_training:
        parser.error('intervention repeats must be 1..23 and require monthly observed training')
    if (bool(args.model_y_calibration) != bool(args.model_y_calibration_sha256)
            or args.model_y_calibration and not args.model_y):
        parser.error('Model Y calibration requires its manifest hash and --model-y')
    if args.unfreeze_backbone and not args.condition_last_layer:
        parser.error('full-backbone adaptation requires static last-layer conditioning')
    if (bool(args.reference) != bool(args.reference_sha256)
            or args.reference and (args.model_y or not args.condition_last_layer)):
        parser.error('reference-response training requires Model Z, static conditioning and a manifest hash')
    if (bool(args.bhp_calibration) != bool(args.bhp_calibration_sha256)
            or args.bhp_calibration and args.model_y):
        parser.error('Model Z BHP calibration requires a paired manifest hash')
    if (bool(args.regime_calibration) != bool(args.regime_calibration_sha256)
            or args.regime_calibration and args.model_y):
        parser.error('Model Z regime calibration requires a paired manifest hash')
    args.unfreeze_last_layer |= args.condition_last_layer
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.model_y:
        trajectories, origin = verified_y_batch(args.batch, args.batch_sha256)
    else:
        trajectories, origin = verified_batch(args.batch, args.batch_sha256)
    horizon = 23 if args.model_y else 224
    training_horizon = 1 if args.monthly_observed_training else horizon
    offsets = [0] * args.intervention_repeats + list(range(1, horizon)) if args.monthly_observed_training else [0]
    context = min(128, origin)
    count = len(trajectories[0].well_ids)
    targets_count = count * 3
    if args.regime_calibration:
        baseline = next(t for t in trajectories if t.scenario_id == 'baseline')
        trajectories = list(trajectories) + verified_regime_calibration(
            args.regime_calibration, args.regime_calibration_sha256, baseline, origin)
    if args.bhp_calibration:
        baseline = next(t for t in trajectories if t.scenario_id == 'baseline')
        trajectories = list(trajectories) + verified_regime_calibration(
            args.bhp_calibration, args.bhp_calibration_sha256, baseline, origin)
    if args.model_y_calibration:
        trajectories = list(trajectories) + verified_regime_calibration(args.model_y_calibration,
            args.model_y_calibration_sha256, trajectories[0], origin,
            model_y_source_sha256=json.loads((args.batch / 'manifest.json').read_text())['official_source_sha256'])
    connectivity = None
    if args.connectivity:
        from timesoil.aios.interwell import WellConnectivity
        from timesfm_geology import StaticConditionedHead, geological_inputs, self_check as geology_check
        geology_check()
        connectivity = WellConnectivity.from_dict(json.loads(args.connectivity.read_text()))
        if connectivity.provenance['source_sha256'] != json.loads((args.batch / 'manifest.json').read_text())['official_source_sha256']:
            raise ValueError('connectivity belongs to another source reservoir')
    by_id = {t.scenario_id: t for t in trajectories}
    train_ids = ['baseline', 'perturbation-001', 'perturbation-002', 'perturbation-003',
                 'perturbation-005', 'perturbation-006']
    validation_ids = ['perturbation-009']
    test_ids = ['perturbation-004', 'perturbation-007', 'perturbation-008']
    if args.model_y:
        train_ids = [f'physical-sweep-{i:02d}' for i in (0, 1, 2, 4, 8)]
        validation_ids = ['physical-sweep-05']
        test_ids = [f'physical-sweep-{i:02d}' for i in (3, 6, 7)]
    if args.regime_calibration:
        train_ids += [f'physical-sweep-{i:02d}' for i in (0, 1, 2, 7)]
        validation_ids += ['physical-sweep-04']
    if args.bhp_calibration:
        train_ids += [f'bhp-only-{i:02d}' for i in (0, 1, 3, 6)]
        validation_ids += ['bhp-only-04']
    if args.model_y_calibration:
        train_ids += [f'forecast-validation-{i:02d}' for i in (0, 1, 3, 6)]
        validation_ids += ['forecast-validation-04']
    assert set(train_ids + validation_ids + test_ids) == set(by_id)
    assert len(train_ids + validation_ids + test_ids) == len(by_id)
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster
    from timesfm3.torch import cpm_revin_refine

    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('requires the allocated A100 GPU')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(20260909)
    torch.cuda.set_per_process_memory_fraction(.50 if args.unfreeze_backbone else .35)
    forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
        revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
    model = forecaster.model
    if connectivity is not None:
        model.output_head = StaticConditionedHead(model.output_head, connectivity)
    if args.initial_head:
        if sha256(args.initial_head.read_bytes()).hexdigest() != args.initial_head_sha256:
            raise ValueError('initial output-head hash mismatch')
        initial = torch.load(args.initial_head, map_location='cuda', weights_only=True)
        if initial.get('reference_manifest_sha256') not in (None, args.reference_sha256):
            raise ValueError('initial weights require their original physical reference')
        initial_weights = ({k.removeprefix('output_head.'): v for k, v in initial['full_model'].items()
                            if k.startswith('output_head.')} if 'full_model' in initial else initial.get('output_head', initial))
        if connectivity is not None:
            torch.testing.assert_close(initial_weights['features'], model.output_head.features, rtol=0, atol=0)
        model.output_head.load_state_dict(initial_weights)
    if args.condition_last_layer:
        from timesfm_geology import StaticConditionedLayer
        model.transformer_stack.layers[-1] = StaticConditionedLayer(model.transformer_stack.layers[-1], model.output_head)
    if args.initial_head and ('last_layer' in initial or 'full_model' in initial):
        if bool(initial.get('static_last_layer', False)) != args.condition_last_layer:
            raise ValueError('initial last-layer architecture differs from requested conditioning')
        if 'full_model' in initial:
            model.load_state_dict(initial['full_model'])
        elif args.condition_last_layer:
            torch.testing.assert_close(initial['last_layer']['features'], model.transformer_stack.layers[-1].features, rtol=0, atol=0)
        if 'last_layer' in initial:
            model.transformer_stack.layers[-1].load_state_dict(initial['last_layer'])
    model.requires_grad_(args.unfreeze_backbone)
    model.output_head.requires_grad_(True)
    if args.unfreeze_last_layer:
        model.transformer_stack.layers[-1].requires_grad_(True)
    if args.unfreeze_backbone:
        from functools import partial
        from torch.utils.checkpoint import checkpoint as checkpoint_layer
        for layer in model.transformer_stack.layers:
            layer.forward = partial(checkpoint_layer, layer.forward, use_reentrant=False)
    model.eval()  # Keep the frozen backbone's inference behavior during head adaptation.
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0)
    quantiles = torch.tensor(model.quantiles, device='cuda')
    train_truth = np.stack([by_id[i].states[origin + 1:origin + horizon + 1] for i in train_ids])
    feature_scale = np.maximum(np.abs(train_truth).mean(axis=(0, 1, 2)), 1.0)
    scale = torch.tensor(np.tile(feature_scale, count)[None, :, None], device='cuda', dtype=torch.float32)
    examples = {}
    train_keys = [(name, offset) for name in train_ids for offset in offsets]
    validation_keys = [(name, offset) for name in validation_ids for offset in offsets]
    assert not set(train_keys) & set(validation_keys)
    for name, offset in dict.fromkeys(train_keys + validation_keys):
        t = by_id[name]
        position = origin + offset
        if connectivity is None:
            target, cov = forecast_inputs(t.states, t.actions, position, context, training_horizon)
            target, cov = target.reshape(targets_count, context), cov[:, :-1].reshape(count * 5, context + training_horizon)
        else:
            target, cov = geological_inputs(t, position, context, training_horizon, connectivity)
        examples[name, offset] = (
            torch.tensor(target[None], device='cuda'),
            torch.tensor(cov[None], device='cuda', dtype=torch.float32),
            torch.tensor(t.states[position + 1:position + training_horizon + 1].transpose(1, 2, 0).reshape(1, targets_count, training_horizon),
                         device='cuda', dtype=torch.float32))
    reference = None
    if args.reference:
        from timesoil.aios.track2 import load_trajectory_dataset
        if sha256((args.reference / 'manifest.json').read_bytes()).hexdigest() != args.reference_sha256:
            raise ValueError('physical reference manifest hash mismatch')
        data = load_trajectory_dataset(args.reference / 'trajectory.csv', manifest=args.reference / 'manifest.json')
        if len(data) != 1 or not data.model_z_identity or data[0].well_ids != connectivity.well_ids:
            raise ValueError('physical reference reservoir or well inventory differs')
        reference = data[0]
        for t in trajectories:
            if not t.dates.equals(reference.dates):
                raise ValueError('physical reference temporal grid differs')
            np.testing.assert_allclose(t.states[:origin + 1], reference.states[:origin + 1], rtol=0, atol=1e-6)
            np.testing.assert_array_equal(t.actions[:origin], reference.actions[:origin])
        ref_target, ref_cov = geological_inputs(reference, origin, context, horizon, connectivity)
        ref_target = torch.tensor(ref_target[None], device='cuda')
        ref_cov = torch.tensor(ref_cov[None], device='cuda', dtype=torch.float32)
        ref_truth = torch.tensor(reference.states[origin + 1:origin + horizon + 1].transpose(1, 2, 0).reshape(1, targets_count, horizon, 1), device='cuda', dtype=torch.float32)
        ref_actions = torch.tensor(reference.actions[origin:origin + horizon], device='cuda')
        median = forecaster.config.median_quantile_index
    decode = type(model).decode.__wrapped__  # Same pinned decoder, with autograd enabled.
    refine = torch.no_grad()(cpm_revin_refine.cpm_iterative_revin_refine)

    def loss_for(name):
        target, cov, truth = examples[name]
        # ponytail: process-local patch for this single-threaded trainer; replace with a native 3.0 trainer when available.
        with patch.object(cpm_revin_refine, 'cpm_iterative_revin_refine', refine):
            prediction = decode(model, target, horizon=training_horizon, past_future_covariates=cov)[:, :targets_count]
            if reference is not None:
                reference_prediction = decode(model, ref_target, horizon=horizon, past_future_covariates=ref_cov)[:, :targets_count]
                actions = torch.tensor(by_id[name[0]].actions[origin:origin + horizon], device='cuda')
                reference_point = project_quantiles(reference_prediction, ref_actions)[..., median:median + 1]
                prediction = project_quantiles(ref_truth + project_quantiles(prediction, actions) - reference_point, actions)
        return pinball_loss(prediction, truth, scale, quantiles)

    target, cov, _ = examples[train_keys[0]]
    with torch.no_grad():
        torch.manual_seed(20260909)
        wrapped = model.decode(target, horizon=training_horizon, past_future_covariates=cov)
        torch.manual_seed(20260909)
        unwrapped = decode(model, target, horizon=training_horizon, past_future_covariates=cov)
        # Forecaster and loss consume targets only; predicted covariate outputs are discarded.
        parity_error = float((wrapped[:, :targets_count] - unwrapped[:, :targets_count]).abs().max())
        parity_scaled_error = float(((wrapped[:, :targets_count] - unwrapped[:, :targets_count]) / scale[..., None]).abs().max())
        torch.testing.assert_close(wrapped[:, :targets_count] / scale[..., None],
                                   unwrapped[:, :targets_count] / scale[..., None], rtol=0, atol=.001)
        best_loss = float(torch.stack([loss_for(key) for key in validation_keys]).mean())
    del wrapped, unwrapped
    checkpoint = args.output / ('full-model.pt' if args.unfreeze_backbone else
        'last-layer-and-head.pt' if args.unfreeze_last_layer else 'output-head.pt')
    def selected_weights():
        if args.unfreeze_backbone:
            weights = {'full_model': model.state_dict(), 'static_last_layer': args.condition_last_layer}
        elif args.unfreeze_last_layer:
            weights = {'output_head': model.output_head.state_dict(),
                    'last_layer': model.transformer_stack.layers[-1].state_dict(),
                    'static_last_layer': args.condition_last_layer}
        else:
            return model.output_head.state_dict()
        if args.reference:
            weights['reference_manifest_sha256'] = args.reference_sha256
        return weights
    torch.save(selected_weights(), checkpoint)
    report = dict(schema='timesoil.timesfm-head-adaptation/v1', model_revision=MODEL_REVISION,
        batch_manifest_sha256=args.batch_sha256, source_scenario_hashes={t.scenario_id: t.content_hash for t in trajectories},
        train_scenarios=train_ids, validation_scenarios=validation_ids, test_scenarios=test_ids,
        trained_component='TimesFM3Torch (complete)' if args.unfreeze_backbone else
            'TimesFM3Torch.output_head' + (' + transformer_stack.layers[-1]' if args.unfreeze_last_layer else ''),
        backbone_frozen=not args.unfreeze_last_layer,
        last_layer_trainable=args.unfreeze_last_layer,
        static_last_layer=args.condition_last_layer,
        all_other_backbone_parameters_frozen=not args.unfreeze_backbone,
        full_backbone_trainable=args.unfreeze_backbone, gradient_checkpointing=args.unfreeze_backbone,
        initial_head_sha256=args.initial_head_sha256,
        regime_calibration_manifest_sha256=args.regime_calibration_sha256,
        bhp_calibration_manifest_sha256=args.bhp_calibration_sha256,
        model_y_calibration_manifest_sha256=args.model_y_calibration_sha256,
        intervention_window_repeats=args.intervention_repeats,
        reference_manifest_sha256=args.reference_sha256,
        reference_trajectory_sha256=reference.content_hash if reference is not None else None,
        training_prediction='projected OPM(reference) + Google(candidate) - Google(reference)' if args.reference else 'Google absolute response',
        regime_calibration_reused_for_development=args.regime_calibration is not None,
        development_test_scenarios_previously_inspected=True,
        trainable_parameters=sum(p.numel() for p in trainable), learning_rate=args.learning_rate,
        attention_backend='math', decoder_target_quantile_parity_max_abs=parity_error,
        gradient_policy='stop gradients through iterative CPM-RevIN statistics; unchanged forward calculation',
        decoder_target_quantile_parity_max_scaled=parity_scaled_error,
        decoder_target_quantile_parity_atol_train_scale=.001,
        epochs_requested=args.epochs, horizon_months=horizon, context_months=context, control_channels=examples[train_keys[0]][1].shape[1],
        training_horizon_months=training_horizon, training_origin_offsets=offsets,
        monthly_observed_training=args.monthly_observed_training,
        train_example_count=len(train_keys), validation_example_count=len(validation_keys),
        connectivity_sha256=sha256(args.connectivity.read_bytes()).hexdigest() if args.connectivity else None,
        static_conditioning=connectivity is not None,
        static_feature_names=connectivity.provenance.get('static_feature_names', ['permeability', 'porosity', 'net_thickness']) if connectivity else [],
        training_scale=feature_scale.tolist(), best_epoch=0, validation_loss_before=best_loss,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        decoder_source_sha256=sha256(Path(inspect.getsourcefile(type(model))).read_bytes()).hexdigest(),
        independent_uncertainty_calibrated=False, is_new_optimization_result=False,
        epochs=[], test_results=[])

    def forecast(t, block=horizon, observe=False):
        if connectivity is None:
            return forecast_blocks(forecaster, t, origin, horizon, context, block, observe=observe)
        from benchmark_timesfm_layouts import forecast_layout
        results = []
        for offset in range(0, horizon, block):
            size = min(block, horizon - offset)
            if offset and not observe:
                raise ValueError('geological block forecast requires actual observed updates')
            results.append(forecast_layout(forecaster, t, origin + offset, size, context,
                                           'joint', connectivity=connectivity))
        prediction = np.concatenate(results)
        if reference is not None:
            from evaluate_timesfm_scenarios import reference_delta
            reference_prediction = forecast_layout(forecaster, reference, origin, horizon, context,
                                                  'joint', connectivity=connectivity)
            prediction = reference_delta(reference.states[origin + 1:origin + horizon + 1],
                reference_prediction, prediction, t.actions[origin:origin + horizon])
        return prediction

    evaluation_modes = [('observed_update_block_1', 1, True)] if args.monthly_observed_training else [
        ('fixed_origin_direct', horizon, False), ('observed_update_block_6', 6, True)]
    if reference is not None:
        evaluation_modes = [('reference_delta_direct', horizon, False)]
    for name in test_ids:
        t = by_id[name]
        initial_mode, block, observe = evaluation_modes[0]
        pred = forecast(t, block, observe)
        report['test_results'].append(dict(scenario_id=name, stage='initial_head' if args.initial_head else 'pretrained',
            name=initial_mode, **metrics(t.states[origin + 1:origin + horizon + 1], pred)))
    for epoch in range(1, args.epochs + 1):
        losses = []
        for index in np.random.default_rng(20260909 + epoch).permutation(len(train_keys)):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(train_keys[index])
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            loss.backward()
            if args.unfreeze_backbone and epoch == 1:
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.transformer_stack.layers[0].parameters()):
                    raise ValueError('first native transformer layer received no gradient')
            if args.unfreeze_last_layer and epoch == 1:
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.transformer_stack.layers[-1].parameters()):
                    raise ValueError('last native transformer layer received no gradient')
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            validation = float(torch.stack([loss_for(key) for key in validation_keys]).mean())
        if not np.isfinite(validation):
            raise ValueError('non-finite validation loss')
        if validation < best_loss:
            best_loss = validation
            report['best_epoch'] = epoch
            torch.save(selected_weights(), checkpoint)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_loss=validation)
        report['epochs'].append(row)
        (args.output / 'report.partial.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    assert frozen_versions == {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    selected = torch.load(checkpoint, map_location='cuda', weights_only=True)
    if args.unfreeze_backbone:
        model.load_state_dict(selected['full_model'])
    else:
        model.output_head.load_state_dict(selected['output_head'] if args.unfreeze_last_layer else selected)
    if args.unfreeze_last_layer and not args.unfreeze_backbone:
        model.transformer_stack.layers[-1].load_state_dict(selected['last_layer'])
    for name in test_ids:
        t = by_id[name]
        for mode, block, observe in evaluation_modes:
            pred = forecast(t, block, observe)
            row = dict(scenario_id=name, stage='selected_head', name=mode,
                **metrics(t.states[origin + 1:origin + horizon + 1], pred))
            report['test_results'].append(row)
            print(json.dumps(row), flush=True)
        if connectivity is not None:
            mode, block, observe = evaluation_modes[0]
            full = forecast(t, block, observe)
            model.output_head.disabled = True
            if args.condition_last_layer:
                model.transformer_stack.layers[-1].disabled = True
            ablated = forecast(t, block, observe)
            model.output_head.disabled = False
            if args.condition_last_layer:
                model.transformer_stack.layers[-1].disabled = False
            report['test_results'].append(dict(scenario_id=name, stage='static_conditioning_disabled',
                name=mode, **metrics(t.states[origin + 1:origin + horizon + 1], ablated),
                prediction_max_abs_change=float(np.abs(full - ablated).max())))
    report.update(complete=True, validation_loss_best=best_loss,
                  checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                  seconds_total=time.monotonic() - started)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
