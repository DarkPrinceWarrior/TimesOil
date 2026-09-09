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
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.batch or not args.batch_sha256 or not args.output or not 1 <= args.epochs <= 100:
        parser.error('batch, batch-sha256, output and 1..100 epochs required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trajectories, origin = verified_batch(args.batch, args.batch_sha256)
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
    model.requires_grad_(False)
    model.output_head.requires_grad_(True)
    model.eval()  # Keep the frozen backbone's inference behavior during head adaptation.
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    trainable = list(model.output_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0)
    quantiles = torch.tensor(model.quantiles, device='cuda')
    train_truth = np.stack([by_id[i].states[origin + 1:origin + 225] for i in train_ids])
    feature_scale = np.maximum(np.abs(train_truth).mean(axis=(0, 1, 2)), 1.0)
    scale = torch.tensor(np.tile(feature_scale, 103)[None, :, None], device='cuda', dtype=torch.float32)
    examples = {}
    for name in train_ids + validation_ids:
        t = by_id[name]
        target, cov = forecast_inputs(t.states, t.actions, origin, 128, 224)
        examples[name] = (
            torch.tensor(target.reshape(1, 309, 128), device='cuda'),
            torch.tensor(cov[:, :-1].reshape(1, 515, 352), device='cuda'),
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
    checkpoint = args.output / 'output-head.pt'
    torch.save(model.output_head.state_dict(), checkpoint)
    report = dict(schema='timesoil.timesfm-head-adaptation/v1', model_revision=MODEL_REVISION,
        batch_manifest_sha256=args.batch_sha256, source_scenario_hashes={t.scenario_id: t.content_hash for t in trajectories},
        train_scenarios=train_ids, validation_scenarios=validation_ids, test_scenarios=test_ids,
        trained_component='TimesFM3Torch.output_head', backbone_frozen=True,
        trainable_parameters=sum(p.numel() for p in trainable), learning_rate=1e-5,
        attention_backend='math', decoder_target_quantile_parity_max_abs=parity_error,
        gradient_policy='stop gradients through iterative CPM-RevIN statistics; unchanged forward calculation',
        decoder_target_quantile_parity_max_scaled=parity_scaled_error,
        decoder_target_quantile_parity_atol_train_scale=.001,
        epochs_requested=args.epochs, horizon_months=224, context_months=128, control_channels=515,
        training_scale=feature_scale.tolist(), best_epoch=0, validation_loss_before=best_loss,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        decoder_source_sha256=sha256(Path(inspect.getsourcefile(type(model))).read_bytes()).hexdigest(),
        independent_uncertainty_calibrated=False, is_new_optimization_result=False,
        epochs=[], test_results=[])
    for name in test_ids:
        t = by_id[name]
        pred = forecast_blocks(forecaster, t, origin, 224, 128, 224)
        report['test_results'].append(dict(scenario_id=name, stage='pretrained',
            name='fixed_origin_direct', **metrics(t.states[origin + 1:origin + 225], pred)))
    for epoch in range(1, args.epochs + 1):
        losses = []
        for name in np.random.default_rng(20260909 + epoch).permutation(train_ids):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(name)
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            loss.backward()
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
            torch.save(model.output_head.state_dict(), checkpoint)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_loss=validation)
        report['epochs'].append(row)
        (args.output / 'report.partial.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    assert frozen_versions == {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    model.output_head.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
    for name in test_ids:
        t = by_id[name]
        for mode, block, observe in [('fixed_origin_direct', 224, False), ('observed_update_block_6', 6, True)]:
            pred = forecast_blocks(forecaster, t, origin, 224, 128, block, observe=observe)
            row = dict(scenario_id=name, stage='selected_head', name=mode,
                **metrics(t.states[origin + 1:origin + 225], pred))
            report['test_results'].append(row)
            print(json.dumps(row), flush=True)
    report.update(complete=True, validation_loss_best=best_loss,
                  checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                  seconds_total=time.monotonic() - started)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
