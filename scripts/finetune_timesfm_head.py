"""Adapt TimesFM 3.0 to the nine economic targets on whole, provenance-verified OPM scenarios."""

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

from benchmark_bhp_surrogate import MONTHS, START, management_window
from timesoil.aios.track2 import MODEL_Z_SOURCE_SHA256, load_trajectory_dataset


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verified_batch(batch, expected_hash):
    if digest(batch / 'manifest.json') != expected_hash:
        raise ValueError('batch manifest hash mismatch')
    manifest = json.loads((batch / 'manifest.json').read_text())
    records = manifest['scenarios']
    count = manifest['scenario_count']
    if (manifest['official_source_sha256'] != MODEL_Z_SOURCE_SHA256
            or type(count) is not int or count < 3 or len(records) != count
            or len({r['scenario_id'] for r in records}) != count
            or 'baseline' not in {r['scenario_id'] for r in records}):
        raise ValueError('at least three distinct official scenarios including baseline required')
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


def pinball_loss(prediction, target, scale, quantiles):
    error = (target[..., None] - prediction) / scale[..., None]
    return (error * quantiles).maximum(error * (quantiles - 1)).mean()


def training_feature_scale(train_truth, retained_scale=None):
    """Keep frozen checkpoint units when continuing on a new development batch."""
    scale = (np.maximum(np.abs(train_truth).mean(axis=(0, 1, 2)), 1.0)
             if retained_scale is None else np.asarray(retained_scale, dtype=float))
    if scale.shape != (train_truth.shape[-1],) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('finite positive scales matching the training targets required')
    return scale


def self_check():
    import torch
    truth = np.array([[[[2., 4., 6.]]]])
    np.testing.assert_array_equal(training_feature_scale(truth), [2., 4., 6.])
    np.testing.assert_array_equal(training_feature_scale(truth, [1., 3., 5.]), [1., 3., 5.])
    for invalid in ([1., 0., 3.], [1., float('nan'), 3.], [1., 2.]):
        try:
            training_feature_scale(truth, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid retained normalization accepted')
    predicted = torch.zeros((1, 1, 1, 2), requires_grad=True)
    loss = pinball_loss(predicted, torch.ones((1, 1, 1)), torch.ones((1, 1, 1)),
                        torch.tensor([.25, .75]))
    torch.testing.assert_close(loss, torch.tensor(.5))
    loss.backward()
    torch.testing.assert_close(predicted.grad.flatten(), torch.tensor([-.125, -.375]))
    print('normalized quantile loss and gradients verified', flush=True)


def verified_regime_calibration(batch, expected_hash, baseline, origin):
    if digest(batch / 'manifest.json') != expected_hash:
        raise ValueError('regime calibration manifest hash mismatch')
    manifest = json.loads((batch / 'manifest.json').read_text())
    allowed = ('timesoil.bhp-only-forecast-evaluation/v1', 'timesoil.frozen-forecast-evaluation-cases/v1')
    if manifest.get('schema') not in allowed:
        raise ValueError('calibration schema does not match the training reservoir')
    bhp = manifest.get('schema') == 'timesoil.bhp-only-forecast-evaluation/v1'
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
            if digest(root / name) != record[key]:
                raise ValueError('regime calibration artifact hash mismatch')
        data = load_trajectory_dataset(root / 'trajectory.csv', manifest=root / 'manifest.json')
        if len(data) != 1 or not data.model_z_identity:
            raise ValueError('regime calibration requires the authenticated training reservoir')
        item = data[0]
        prefix = 'bhp-only' if bhp else 'physical-sweep'
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
    parser.add_argument('--unfreeze-backbone', action='store_true')
    parser.add_argument('--condition-last-layer', action='store_true')
    parser.add_argument('--condition-first-layer', action='store_true')
    parser.add_argument('--cold-start-normalization', action='store_true')
    parser.add_argument('--retain-initial-scale', action='store_true',
        help='Keep authenticated initial checkpoint normalization on a new development batch')
    parser.add_argument('--regime-calibration', type=Path)
    parser.add_argument('--regime-calibration-sha256')
    parser.add_argument('--bhp-calibration', type=Path)
    parser.add_argument('--bhp-calibration-sha256')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--economic-targets', action='store_true',
        help='Train the nine canonical economic outputs on the verified Model Z development batch')
    parser.add_argument('--precise-variate-softmax', action='store_true',
        help='Use FP64 softmax in native manual variate attention; persist this inference setting')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.batch or not args.batch_sha256 or not args.output or not 1 <= args.epochs <= 100:
        parser.error('batch, batch-sha256, output and 1..100 epochs required')
    if not np.isfinite(args.learning_rate) or not 1e-7 <= args.learning_rate <= 1e-3:
        parser.error('learning-rate must be in [1e-7, 1e-3]')
    if not args.economic_targets:
        parser.error('only the nine canonical economic targets are trainable; pass --economic-targets')
    if not args.connectivity or not args.condition_last_layer:
        parser.error('economic training requires verified connectivity and static last-layer conditioning')
    if not args.unfreeze_backbone:
        parser.error('only full-backbone adaptation is supported; pass --unfreeze-backbone')
    if bool(args.initial_head) != bool(args.initial_head_sha256):
        parser.error('initial-head and its SHA-256 must be supplied together')
    if args.retain_initial_scale and not (args.cold_start_normalization and args.initial_head):
        parser.error('retaining normalization requires cold-start normalization and initial weights')
    if bool(args.bhp_calibration) != bool(args.bhp_calibration_sha256):
        parser.error('Model Z BHP calibration requires a paired manifest hash')
    if bool(args.regime_calibration) != bool(args.regime_calibration_sha256):
        parser.error('Model Z regime calibration requires a paired manifest hash')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    from timesfm_economics import (ECONOMIC_TARGETS, economic_metrics, forecast_economic,
                                   load_economic_trajectories)
    trajectories, origin = verified_batch(args.batch, args.batch_sha256)
    horizon = MONTHS
    context = min(128, origin)
    count = len(trajectories[0].well_ids)
    targets_count = count * len(ECONOMIC_TARGETS)
    if args.regime_calibration:
        baseline = next(t for t in trajectories if t.scenario_id == 'baseline')
        trajectories = list(trajectories) + verified_regime_calibration(
            args.regime_calibration, args.regime_calibration_sha256, baseline, origin)
    if args.bhp_calibration:
        baseline = next(t for t in trajectories if t.scenario_id == 'baseline')
        trajectories = list(trajectories) + verified_regime_calibration(
            args.bhp_calibration, args.bhp_calibration_sha256, baseline, origin)
    trajectories = load_economic_trajectories(args.batch, trajectories, origin,
        extra_batches=[path for path in (args.regime_calibration, args.bhp_calibration) if path is not None])
    from timesoil.aios.interwell import WellConnectivity
    from timesfm_geology import (MODEL_REVISION, StaticConditionedHead, StaticConditionedLayer,
        enable_cold_start_normalization, enable_precise_variate_softmax, geological_inputs,
        self_check as geology_check)
    geology_check()
    connectivity = WellConnectivity.from_dict(json.loads(args.connectivity.read_text()))
    if connectivity.provenance['source_sha256'] != json.loads((args.batch / 'manifest.json').read_text())['official_source_sha256']:
        raise ValueError('connectivity belongs to another source reservoir')
    by_id = {t.scenario_id: t for t in trajectories}
    train_ids = ['baseline', 'perturbation-001', 'perturbation-002', 'perturbation-003',
                 'perturbation-005', 'perturbation-006']
    validation_ids = ['perturbation-009']
    test_ids = ['perturbation-004', 'perturbation-007', 'perturbation-008']
    if args.regime_calibration:
        train_ids += [f'physical-sweep-{i:02d}' for i in (0, 1, 2, 7)]
        validation_ids += ['physical-sweep-04']
    if args.bhp_calibration:
        train_ids += [f'bhp-only-{i:02d}' for i in (0, 1, 3, 6)]
        validation_ids += ['bhp-only-04']
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
    torch.cuda.set_per_process_memory_fraction(.60 if args.precise_variate_softmax else .50)
    forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
        revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
    model = forecaster.model
    model.output_head = StaticConditionedHead(model.output_head, connectivity)
    if args.initial_head:
        if digest(args.initial_head) != args.initial_head_sha256:
            raise ValueError('initial output-head hash mismatch')
        initial = torch.load(args.initial_head, map_location='cuda', weights_only=True)
        if initial.get('economic_targets') != list(ECONOMIC_TARGETS):
            raise ValueError('initial checkpoint target schema differs')
        if 'full_model' not in initial:
            raise ValueError('a complete full-backbone initial checkpoint is required')
        if 'cold_start_scale' in initial and not args.cold_start_normalization:
            raise ValueError('initial weights require cold-start normalization')
        if initial.get('reference_manifest_sha256') is not None:
            raise ValueError('initial weights require their original physical reference')
        initial_weights = {k.removeprefix('output_head.'): v for k, v in initial['full_model'].items()
                           if k.startswith('output_head.')}
        torch.testing.assert_close(initial_weights['features'], model.output_head.features, rtol=0, atol=0)
        model.output_head.load_state_dict(initial_weights)
    model.transformer_stack.layers[-1] = StaticConditionedLayer(model.transformer_stack.layers[-1], model.output_head)
    if args.initial_head and initial.get('static_first_layer', False):
        if not args.condition_first_layer:
            raise ValueError('initial weights require first-layer conditioning')
        model.transformer_stack.layers[0] = StaticConditionedLayer(model.transformer_stack.layers[0], model.output_head)
    if args.initial_head:
        if bool(initial.get('static_last_layer', False)) != args.condition_last_layer:
            raise ValueError('initial last-layer architecture differs from requested conditioning')
        model.load_state_dict(initial['full_model'])
    if args.condition_first_layer and not isinstance(model.transformer_stack.layers[0], StaticConditionedLayer):
        model.transformer_stack.layers[0] = StaticConditionedLayer(model.transformer_stack.layers[0], model.output_head)
    model.requires_grad_(True)
    if args.initial_head and initial.get('precise_variate_softmax', False) and not args.precise_variate_softmax:
        raise ValueError('initial weights require precise variate softmax')
    if args.precise_variate_softmax:
        enable_precise_variate_softmax(model)
    model.output_head.requires_grad_(True)
    model.transformer_stack.layers[-1].requires_grad_(True)
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
    if args.retain_initial_scale and 'cold_start_scale' not in initial:
        raise ValueError('initial checkpoint has no normalization to retain')
    feature_scale = training_feature_scale(train_truth, initial['cold_start_scale'] if args.retain_initial_scale else None)
    if args.cold_start_normalization:
        enable_cold_start_normalization(model, feature_scale)
        if args.initial_head and 'cold_start_scale' in initial:
            np.testing.assert_array_equal(initial['cold_start_scale'], feature_scale)
    scale = torch.tensor(np.tile(feature_scale, count)[None, :, None], device='cuda', dtype=torch.float32)
    examples = {}
    assert not set(train_ids) & set(validation_ids)
    for name in dict.fromkeys(train_ids + validation_ids):
        t = by_id[name]
        target, cov = geological_inputs(t, origin, context, horizon, connectivity)
        examples[name] = (
            torch.tensor(target[None], device='cuda'),
            torch.tensor(cov[None], device='cuda', dtype=torch.float32),
            torch.tensor(t.states[origin + 1:origin + horizon + 1].transpose(1, 2, 0).reshape(1, targets_count, horizon),
                         device='cuda', dtype=torch.float32))
    decode = type(model).decode.__wrapped__  # Same pinned decoder, with autograd enabled.
    refine = torch.no_grad()(cpm_revin_refine.cpm_iterative_revin_refine)

    def loss_for(name):
        target, cov, truth = examples[name]
        # ponytail: process-local patch for this single-threaded trainer; replace with a native 3.0 trainer when available.
        with patch.object(cpm_revin_refine, 'cpm_iterative_revin_refine', refine):
            prediction = decode(model, target, horizon=horizon, past_future_covariates=cov)[:, :targets_count]
        return pinball_loss(prediction, truth, scale, quantiles)

    target, cov, _ = examples[train_ids[0]]
    with torch.no_grad():
        torch.manual_seed(20260909)
        wrapped = model.decode(target, horizon=horizon, past_future_covariates=cov)
        torch.manual_seed(20260909)
        unwrapped = decode(model, target, horizon=horizon, past_future_covariates=cov)
        # Forecaster and loss consume targets only; predicted covariate outputs are discarded.
        parity_error = float((wrapped[:, :targets_count] - unwrapped[:, :targets_count]).abs().max())
        parity_scaled_error = float(((wrapped[:, :targets_count] - unwrapped[:, :targets_count]) / scale[..., None]).abs().max())
        torch.testing.assert_close(wrapped[:, :targets_count] / scale[..., None],
                                   unwrapped[:, :targets_count] / scale[..., None], rtol=0, atol=.001)
        best_loss = float(torch.stack([loss_for(key) for key in validation_ids]).mean())
    del wrapped, unwrapped
    checkpoint = args.output / 'full-model.pt'
    def selected_weights():
        weights = {'full_model': model.state_dict(), 'static_last_layer': args.condition_last_layer,
                   'static_first_layer': args.condition_first_layer}
        if args.cold_start_normalization:
            weights['cold_start_scale'] = model.cold_start_scale
        weights['economic_targets'] = list(ECONOMIC_TARGETS)
        if args.precise_variate_softmax:
            weights['precise_variate_softmax'] = True
        return weights
    torch.save(selected_weights(), checkpoint)
    report = dict(schema='timesoil.timesfm-head-adaptation/v1', model_revision=MODEL_REVISION,
        economic_targets=list(ECONOMIC_TARGETS),
        batch_manifest_sha256=args.batch_sha256, source_scenario_hashes={t.scenario_id: t.content_hash for t in trajectories},
        train_scenarios=train_ids, validation_scenarios=validation_ids, test_scenarios=test_ids,
        trained_component='TimesFM3Torch (complete)',
        backbone_frozen=False, last_layer_trainable=True,
        static_last_layer=args.condition_last_layer,
        static_first_layer=args.condition_first_layer,
        all_other_backbone_parameters_frozen=False,
        full_backbone_trainable=True, gradient_checkpointing=True,
        initial_head_sha256=args.initial_head_sha256,
        cold_start_scale=getattr(model, 'cold_start_scale', None),
        normalization_scale_source='initial_checkpoint' if args.retain_initial_scale else 'current_training_scenarios',
        cold_start_source_sha256=digest(Path(inspect.getsourcefile(enable_cold_start_normalization)))
            if args.cold_start_normalization else None,
        regime_calibration_manifest_sha256=args.regime_calibration_sha256,
        bhp_calibration_manifest_sha256=args.bhp_calibration_sha256,
        training_prediction='Google absolute response',
        regime_calibration_reused_for_development=args.regime_calibration is not None,
        development_test_scenarios_previously_inspected=True,
        trainable_parameters=sum(p.numel() for p in trainable), learning_rate=args.learning_rate,
        attention_backend='manual_variate_fp64_softmax_math_sequence' if args.precise_variate_softmax else 'math',
        precise_variate_softmax=args.precise_variate_softmax,
        decoder_target_quantile_parity_max_abs=parity_error,
        gradient_policy='stop gradients through iterative CPM-RevIN statistics; unchanged forward calculation',
        decoder_target_quantile_parity_max_scaled=parity_scaled_error,
        decoder_target_quantile_parity_atol_train_scale=.001,
        epochs_requested=args.epochs, horizon_months=horizon, context_months=context,
        control_channels=examples[train_ids[0]][1].shape[1],
        train_example_count=len(train_ids), validation_example_count=len(validation_ids),
        connectivity_sha256=digest(args.connectivity),
        static_conditioning=True,
        static_feature_names=connectivity.provenance.get('static_feature_names', ['permeability', 'porosity', 'net_thickness']),
        training_scale=feature_scale.tolist(), best_epoch=0, validation_loss_before=best_loss,
        script_sha256=digest(Path(__file__)),
        decoder_source_sha256=digest(Path(inspect.getsourcefile(type(model)))),
        independent_uncertainty_calibrated=False, is_new_optimization_result=False,
        epochs=[], test_results=[])

    def forecast(t):
        return forecast_economic(forecaster, t, origin, horizon, context, connectivity)

    for name in test_ids:
        t = by_id[name]
        pred = forecast(t)
        report['test_results'].append(dict(scenario_id=name, stage='initial_head' if args.initial_head else 'pretrained',
            name='economic_direct', **economic_metrics(t.states[origin + 1:origin + horizon + 1], pred)))
    for epoch in range(1, args.epochs + 1):
        losses = []
        for index in np.random.default_rng(20260909 + epoch).permutation(len(train_ids)):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(train_ids[index])
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            loss.backward()
            if epoch == 1:
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.transformer_stack.layers[0].parameters()):
                    raise ValueError('first native transformer layer received no gradient')
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.transformer_stack.layers[-1].parameters()):
                    raise ValueError('last native transformer layer received no gradient')
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            validation = float(torch.stack([loss_for(key) for key in validation_ids]).mean())
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
    model.load_state_dict(selected['full_model'])
    for name in test_ids:
        t = by_id[name]
        pred = forecast(t)
        row = dict(scenario_id=name, stage='selected_head', name='economic_direct',
            **economic_metrics(t.states[origin + 1:origin + horizon + 1], pred))
        report['test_results'].append(row)
        print(json.dumps(row), flush=True)
        full = forecast(t)
        model.output_head.disabled = True
        model.transformer_stack.layers[-1].disabled = True
        if args.condition_first_layer:
            model.transformer_stack.layers[0].disabled = True
        ablated = forecast(t)
        model.output_head.disabled = False
        model.transformer_stack.layers[-1].disabled = False
        if args.condition_first_layer:
            model.transformer_stack.layers[0].disabled = False
        report['test_results'].append(dict(scenario_id=name, stage='static_conditioning_disabled',
            name='economic_direct', **economic_metrics(t.states[origin + 1:origin + horizon + 1], ablated),
            prediction_max_abs_change=float(np.abs(full - ablated).max())))
    report.update(complete=True, validation_loss_best=best_loss,
                  checkpoint_sha256=digest(checkpoint),
                  seconds_total=time.monotonic() - started)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
