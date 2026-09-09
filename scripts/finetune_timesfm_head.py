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


def self_check():
    import torch
    predicted = torch.zeros((1, 1, 1, 2), requires_grad=True)
    loss = pinball_loss(predicted, torch.ones((1, 1, 1)), torch.ones((1, 1, 1)),
                        torch.tensor([.25, .75]))
    torch.testing.assert_close(loss, torch.tensor(.5))
    loss.backward()
    torch.testing.assert_close(predicted.grad.flatten(), torch.tensor([-.125, -.375]))
    print('normalized quantile loss and gradients verified', flush=True)


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
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trajectories, origin = verified_batch(args.batch, args.batch_sha256)
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
    torch.cuda.set_per_process_memory_fraction(.35)
    forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
        revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
    model = forecaster.model
    if connectivity is not None:
        model.output_head = StaticConditionedHead(model.output_head, connectivity)
    if args.initial_head:
        if sha256(args.initial_head.read_bytes()).hexdigest() != args.initial_head_sha256:
            raise ValueError('initial output-head hash mismatch')
        initial = torch.load(args.initial_head, map_location='cuda', weights_only=True)
        if connectivity is not None:
            torch.testing.assert_close(initial['features'], model.output_head.features, rtol=0, atol=0)
        model.output_head.load_state_dict(initial)
    model.requires_grad_(False)
    model.output_head.requires_grad_(True)
    if args.unfreeze_last_layer:
        model.transformer_stack.layers[-1].requires_grad_(True)
    model.eval()  # Keep the frozen backbone's inference behavior during head adaptation.
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0)
    quantiles = torch.tensor(model.quantiles, device='cuda')
    train_truth = np.stack([by_id[i].states[origin + 1:origin + 225] for i in train_ids])
    feature_scale = np.maximum(np.abs(train_truth).mean(axis=(0, 1, 2)), 1.0)
    scale = torch.tensor(np.tile(feature_scale, 103)[None, :, None], device='cuda', dtype=torch.float32)
    examples = {}
    for name in train_ids + validation_ids:
        t = by_id[name]
        if connectivity is None:
            target, cov = forecast_inputs(t.states, t.actions, origin, 128, 224)
            target, cov = target.reshape(309, 128), cov[:, :-1].reshape(515, 352)
        else:
            target, cov = geological_inputs(t, origin, 128, 224, connectivity)
        examples[name] = (
            torch.tensor(target[None], device='cuda'),
            torch.tensor(cov[None], device='cuda', dtype=torch.float32),
            torch.tensor(t.states[origin + 1:origin + 225].transpose(1, 2, 0).reshape(1, 309, 224),
                         device='cuda', dtype=torch.float32))
    decode = type(model).decode.__wrapped__  # Same pinned decoder, with autograd enabled.
    refine = torch.no_grad()(cpm_revin_refine.cpm_iterative_revin_refine)

    def loss_for(name):
        target, cov, truth = examples[name]
        # ponytail: process-local patch for this single-threaded trainer; replace with a native 3.0 trainer when available.
        with patch.object(cpm_revin_refine, 'cpm_iterative_revin_refine', refine):
            prediction = decode(model, target, horizon=224, past_future_covariates=cov)[:, :309]
        return pinball_loss(prediction, truth, scale, quantiles)

    target, cov, _ = examples[train_ids[0]]
    with torch.no_grad():
        torch.manual_seed(20260909)
        wrapped = model.decode(target, horizon=224, past_future_covariates=cov)
        torch.manual_seed(20260909)
        unwrapped = decode(model, target, horizon=224, past_future_covariates=cov)
        # Forecaster and loss consume targets only; predicted covariate outputs are discarded.
        parity_error = float((wrapped[:, :309] - unwrapped[:, :309]).abs().max())
        parity_scaled_error = float(((wrapped[:, :309] - unwrapped[:, :309]) / scale[..., None]).abs().max())
        torch.testing.assert_close(wrapped[:, :309] / scale[..., None],
                                   unwrapped[:, :309] / scale[..., None], rtol=0, atol=.001)
        best_loss = float(loss_for(validation_ids[0]))
    del wrapped, unwrapped
    checkpoint = args.output / ('last-layer-and-head.pt' if args.unfreeze_last_layer else 'output-head.pt')
    def selected_weights():
        if args.unfreeze_last_layer:
            return {'output_head': model.output_head.state_dict(),
                    'last_layer': model.transformer_stack.layers[-1].state_dict()}
        return model.output_head.state_dict()
    torch.save(selected_weights(), checkpoint)
    report = dict(schema='timesoil.timesfm-head-adaptation/v1', model_revision=MODEL_REVISION,
        batch_manifest_sha256=args.batch_sha256, source_scenario_hashes={t.scenario_id: t.content_hash for t in trajectories},
        train_scenarios=train_ids, validation_scenarios=validation_ids, test_scenarios=test_ids,
        trained_component='TimesFM3Torch.output_head' + (' + transformer_stack.layers[-1]' if args.unfreeze_last_layer else ''),
        backbone_frozen=not args.unfreeze_last_layer,
        last_layer_trainable=args.unfreeze_last_layer,
        all_other_backbone_parameters_frozen=True,
        initial_head_sha256=args.initial_head_sha256,
        trainable_parameters=sum(p.numel() for p in trainable), learning_rate=args.learning_rate,
        attention_backend='math', decoder_target_quantile_parity_max_abs=parity_error,
        gradient_policy='stop gradients through iterative CPM-RevIN statistics; unchanged forward calculation',
        decoder_target_quantile_parity_max_scaled=parity_scaled_error,
        decoder_target_quantile_parity_atol_train_scale=.001,
        epochs_requested=args.epochs, horizon_months=224, context_months=128, control_channels=examples[train_ids[0]][1].shape[1],
        connectivity_sha256=sha256(args.connectivity.read_bytes()).hexdigest() if args.connectivity else None,
        static_conditioning=connectivity is not None,
        static_feature_names=connectivity.provenance.get('static_feature_names', ['permeability', 'porosity', 'net_thickness']) if connectivity else [],
        training_scale=feature_scale.tolist(), best_epoch=0, validation_loss_before=best_loss,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        decoder_source_sha256=sha256(Path(inspect.getsourcefile(type(model))).read_bytes()).hexdigest(),
        independent_uncertainty_calibrated=False, is_new_optimization_result=False,
        epochs=[], test_results=[])

    def forecast(t, block=224, observe=False):
        if connectivity is None:
            return forecast_blocks(forecaster, t, origin, 224, 128, block, observe=observe)
        from benchmark_timesfm_layouts import forecast_layout
        results = []
        for offset in range(0, 224, block):
            size = min(block, 224 - offset)
            if offset and not observe:
                raise ValueError('geological block forecast requires actual observed updates')
            results.append(forecast_layout(forecaster, t, origin + offset, size, 128,
                                           'joint', connectivity=connectivity))
        return np.concatenate(results)

    for name in test_ids:
        t = by_id[name]
        pred = forecast(t)
        report['test_results'].append(dict(scenario_id=name, stage='initial_head' if args.initial_head else 'pretrained',
            name='fixed_origin_direct', **metrics(t.states[origin + 1:origin + 225], pred)))
    for epoch in range(1, args.epochs + 1):
        losses = []
        for name in np.random.default_rng(20260909 + epoch).permutation(train_ids):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(name)
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            loss.backward()
            if args.unfreeze_last_layer and epoch == 1:
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.transformer_stack.layers[-1].parameters()):
                    raise ValueError('last native transformer layer received no gradient')
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            validation = float(loss_for(validation_ids[0]))
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
    model.output_head.load_state_dict(selected['output_head'] if args.unfreeze_last_layer else selected)
    if args.unfreeze_last_layer:
        model.transformer_stack.layers[-1].load_state_dict(selected['last_layer'])
    for name in test_ids:
        t = by_id[name]
        for mode, block, observe in [('fixed_origin_direct', 224, False), ('observed_update_block_6', 6, True)]:
            pred = forecast(t, block, observe)
            row = dict(scenario_id=name, stage='selected_head', name=mode,
                **metrics(t.states[origin + 1:origin + 225], pred))
            report['test_results'].append(row)
            print(json.dumps(row), flush=True)
        if connectivity is not None:
            full = forecast(t)
            model.output_head.disabled = True
            ablated = forecast(t)
            model.output_head.disabled = False
            report['test_results'].append(dict(scenario_id=name, stage='static_conditioning_disabled',
                name='fixed_origin_direct', **metrics(t.states[origin + 1:origin + 225], ablated),
                prediction_max_abs_change=float(np.abs(full - ablated).max())))
    report.update(complete=True, validation_loss_best=best_loss,
                  checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                  seconds_total=time.monotonic() - started)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
