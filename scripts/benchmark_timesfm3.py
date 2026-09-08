"""Offline TimesFM 3 benchmark on the frozen KT2 OPM snapshot; run on A100.

This produces forecast metrics, not official CHDD or a replacement surrogate.
Install TimesFM only in a separate experiment environment (see KT3_PREFLIGHT.md).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from timesoil.aios.surrogate import Track2Surrogate, _project_physics
from timesoil.aios.track2 import trajectory_from_frame
from timesoil.metrics import rmse, wape


METRICS_SHA256 = "d0e2a42468b10d0d0ec48442f1b8a7bc4450857537087d3b517abdbc36530de4"
MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
HORIZON = 6


def forecast_inputs(states, actions, origin, context_length):
    """Align action[t] with state[t+1]; never expose future target states."""
    if context_length < 1 or origin < 1 or origin + HORIZON >= len(states):
        raise ValueError("invalid forecast window")
    start = max(1, origin + 1 - context_length)
    targets = states[start : origin + 1].transpose(1, 2, 0).astype(np.float32)
    controls = actions[start - 1 : origin + HORIZON]
    rates = np.stack(
        [np.where(controls[..., 1] == code, controls[..., 0], 0.0)
         * controls[..., 2] for code in range(3)], axis=-1
    )
    field_injection = np.broadcast_to(rates[..., 2].sum(axis=1)[:, None], rates.shape[:2])
    covariates = np.concatenate(
        [rates, controls[..., 2:3], field_injection[..., None]], axis=-1
    ).transpose(1, 2, 0).astype(np.float32)
    return targets, covariates


def self_check():
    """Regression check for the one-month control lag and future-target leakage."""
    states = np.arange(40 * 2 * 3, dtype=float).reshape(40, 2, 3)
    actions = np.zeros_like(states)
    actions[..., 0] = np.arange(40)[:, None] + 1
    actions[:, 0, 1], actions[:, 1, 1] = 1, 2
    actions[..., 2] = 1
    target, cov = forecast_inputs(states, actions, 20, 12)
    assert target.shape == (2, 3, 12) and cov.shape == (2, 5, 18)
    np.testing.assert_array_equal(target[0], states[9:21, 0].T)
    np.testing.assert_array_equal(cov[0, 1, :12], actions[8:20, 0, 0])
    np.testing.assert_array_equal(cov[0, 1, 12:], actions[20:26, 0, 0])
    changed = states.copy()
    changed[21:] = -12345
    np.testing.assert_array_equal(forecast_inputs(changed, actions, 20, 12)[0], target)
    other = actions.copy()
    other[20:26, 1, 0] *= 2
    _, other_cov = forecast_inputs(states, other, 20, 12)
    np.testing.assert_array_equal(other_cov[0, :4], cov[0, :4])
    np.testing.assert_array_equal(other_cov[0, 4, 12:], 2 * cov[0, 4, 12:])
    print("forecast alignment / leakage / field injection checks passed", flush=True)


def metrics(truth, prediction):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid forecast")
    return {
        "oil_wape": wape(truth[..., 0], prediction[..., 0]),
        "liquid_wape": wape(truth[..., 1], prediction[..., 1]),
        "pressure_rmse_bar": rmse(truth[..., 2], prediction[..., 2]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.bundle or not args.output or min(args.windows, args.context, args.batch_size) < 1:
        parser.error("positive window/context/batch sizes and bundle/output required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = (args.bundle / "training/metrics.json").read_bytes()
    if sha256(data).hexdigest() != METRICS_SHA256:
        raise ValueError("frozen KT2 metrics hash mismatch")
    reference = json.loads(data)
    trajectories, sources = {}, []
    for record in reference["scenario_batch"]["scenarios"]:
        scenario = record["scenario_id"]
        csv_path = args.bundle / f"scenario-runs/dataset/{scenario}.csv"
        for path, expected in (
            (csv_path, record["dataset_sha256"]),
            (args.bundle / f"scenario-runs/manifests/{scenario}.json", record["export_manifest_sha256"]),
        ):
            digest = sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"snapshot hash mismatch: {path}")
            sources.append({"path": str(path), "sha256": digest})
        trajectory = trajectory_from_frame(pd.read_csv(csv_path))
        if trajectory.scenario_id != scenario:
            raise ValueError("scenario identity mismatch")
        trajectories[scenario] = trajectory
    train_ids, test_ids = reference["train_scenarios"], reference["test_scenarios"]
    if set(train_ids) & set(test_ids) or set(train_ids + test_ids) != set(trajectories):
        raise ValueError("invalid scenario split")
    print("Refitting CRM+LightGBM on the original eight training scenarios", flush=True)
    surrogate = Track2Surrogate.fit([trajectories[s] for s in train_ids], seed=20260831)
    cases, truths, controls, contexts, covariates, crm, persistence = [], [], [], [], [], [], []
    sensitivity = []
    for scenario in test_ids:
        t = trajectories[scenario]
        origins = [o for o in range(0, len(t.dates) - HORIZON, HORIZON) if o >= args.context]
        if len(origins) < args.windows:
            raise ValueError("not enough complete windows with the requested history")
        for origin in origins[-args.windows:]:
            action = t.actions[origin : origin + HORIZON]
            target, cov = forecast_inputs(t.states, t.actions, origin, args.context)
            contexts.extend(target)
            covariates.extend(cov)
            truth = t.states[origin + 1 : origin + HORIZON + 1]
            truths.append(truth)
            controls.append(action)
            predicted = surrogate.rollout(t.states[origin], action).mean
            crm.append(predicted)
            persistence.append(np.repeat(t.states[origin][None], HORIZON, axis=0))
            changed = action.copy()
            injection = action[..., 1] == 2
            changed[..., 0][injection] *= 1.2
            perturbed = surrogate.rollout(t.states[origin], changed).mean
            producers = ~injection
            sensitivity.append(float(np.max(np.abs(perturbed[..., 0][producers] - predicted[..., 0][producers]))))
            cases.append({"scenario_id": scenario, "origin": origin,
                          "state_date": str(t.dates[origin].date()),
                          "target_dates": [str(d.date()) for d in t.dates[origin + 1 : origin + HORIZON + 1]],
                          "well_ids": list(t.well_ids)})
    truth, actions = np.stack(truths), np.stack(controls)
    predictions = {"persistence": np.stack(persistence), "crm_lightgbm": np.stack(crm)}
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment must run on an A100 GPU")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.35)
    forecaster = TimesFM3Evaluator(ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch", revision=MODEL_REVISION,
        per_core_batch_size=args.batch_size, device="cuda",
    ))
    joint_contexts = list(np.stack(contexts).reshape(len(cases), -1, args.context))
    joint_covariates = list(np.stack(covariates)[:, :4].reshape(len(cases), -1, args.context + HORIZON))
    timings = {}
    for name, inputs, cov, batch_size in (
        ("timesfm3_per_well_history", contexts, None, args.batch_size),
        ("timesfm3_joint_history", joint_contexts, None, 1),
        ("timesfm3_joint_controls", joint_contexts, joint_covariates, 1),
    ):
        forecaster.config = replace(forecaster.config, per_core_batch_size=batch_size)
        print(f"Forecasting {name}: {len(inputs)} windows, {inputs[0].shape[0]} targets", flush=True)
        begin = time.monotonic()
        forecast = list(forecaster.predict_batch(
            inputs, horizon=HORIZON, past_future_covariates=cov,
            use_symmetric_averaging=False, make_positive=True,
        ))
        raw = np.stack([f.forecast for f in forecast]).reshape(len(cases), -1, 3, HORIZON)
        predictions[name] = _project_physics(raw.transpose(0, 3, 1, 2), actions)[0]
        timings[name] = round(time.monotonic() - begin, 3)
        print(json.dumps({"model": name, **metrics(truth, predictions[name]), "seconds": timings[name]}), flush=True)
    report = {
        "schema": "timesoil.timesfm3-offline-benchmark/v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "gpu": torch.cuda.get_device_name(0),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION, "frozen_metrics_sha256": METRICS_SHA256,
        "packages": {n: version(n) for n in ("timesfm", "torch", "numpy", "pandas", "lightgbm")},
        "input_sources": sources, "train_scenarios": train_ids, "test_scenarios": test_ids,
        "cases": cases, "horizon": HORIZON, "context": args.context,
        "metrics": {name: metrics(truth, p) for name, p in predictions.items()},
        "per_scenario": {s: {name: metrics(truth[[c["scenario_id"] == s for c in cases]],
                                                    p[[c["scenario_id"] == s for c in cases]])
                              for name, p in predictions.items()} for s in test_ids},
        "crm_max_producer_oil_change_tpd_after_injection_plus20pct": max(sensitivity),
        "inference_seconds": timings, "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "official_chdd_evaluated": False, "new_opm_replay": False,
        "limitations": [
            "Offline hash-verified snapshot; not a new OPM extraction receipt.",
            f"Last {args.windows} complete six-month windows per held-out scenario, not the full KT2 test set.",
            "TimesFM receives observed history; CRM uses the current state and eight training scenarios.",
            "Joint modes forecast 309 well targets; controls mode sees 412 planned rate/status channels.",
            "No explicit geological connectivity features; attention alone does not prove causal validity.",
            "No tuning on test outcomes, no independently calibrated uncertainty, no NPV improvement claim.",
        ],
    }
    np.savez_compressed(args.output / "predictions.npz", truth=truth, actions=actions, **predictions)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
