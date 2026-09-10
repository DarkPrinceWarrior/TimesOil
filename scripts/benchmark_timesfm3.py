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
PAST_FIELDS = ("WLPR", "WWIR", "BHP", "WEFF", "WOMT_Diff", "WLPT_Diff", "WWIT_Diff")


def past_inputs(observations, origin, context_length):
    """Observed covariates stop at the forecast origin, including actual outages."""
    if origin + 1 < context_length or origin >= len(observations):
        raise ValueError("incomplete observed covariate window")
    return observations[origin + 1 - context_length:origin + 1].transpose(1, 2, 0).astype(np.float32)


def forecast_inputs(states, actions, origin, context_length, horizon=HORIZON, *, target_count=3):
    """Align action[t] with state[t+1]; retain BHP when present and exclude future targets."""
    states, actions = np.asarray(states), np.asarray(actions)
    if (target_count not in (3, 9) or states.ndim != 3 or states.shape[-1] != target_count or actions.ndim != 3
            or actions.shape[:2] != states.shape[:2] or actions.shape[-1] not in (3, 4)):
        raise ValueError("forecast requires aligned states and three/four-feature actions")
    if context_length < 1 or horizon < 1 or origin < 1 or origin + horizon >= len(states):
        raise ValueError("invalid forecast window")
    start = max(1, origin + 1 - context_length)
    targets = states[start : origin + 1].transpose(1, 2, 0).astype(np.float32)
    controls = actions[start - 1 : origin + horizon]
    if (not np.isfinite(targets).all() or not np.isfinite(controls).all()
            or (controls[..., 0] < 0).any()
            or not np.isin(controls[..., 1], (0, 1, 2)).all()
            or not np.isin(controls[..., 2], (0, 1)).all()
            or (controls[..., 3:] < 0).any()):
        raise ValueError("invalid observed states or planned forecast controls")
    rates = np.stack(
        [np.where(controls[..., 1] == code, controls[..., 0], 0.0)
         * controls[..., 2] for code in range(3)], axis=-1
    )
    field_injection = np.broadcast_to(rates[..., 2].sum(axis=1)[:, None], rates.shape[:2])
    covariates = np.concatenate(
        [rates, controls[..., 2:], field_injection[..., None]], axis=-1
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
    bhp_actions = np.concatenate([actions, np.full((*actions.shape[:2], 1), 70.0)], axis=-1)
    bhp_actions[20:26, 0, 3] = 90.0
    bhp_target, bhp_cov = forecast_inputs(states, bhp_actions, 20, 12)
    assert bhp_cov.shape == (2, 6, 18)
    np.testing.assert_array_equal(bhp_target, target)
    np.testing.assert_array_equal(bhp_cov[:, :4], cov[:, :-1])
    np.testing.assert_array_equal(bhp_cov[:, -1], cov[:, -1])
    np.testing.assert_array_equal(bhp_cov[0, 4, :12], 70.0)
    np.testing.assert_array_equal(bhp_cov[0, 4, 12:], 90.0)
    invalid_bhp = bhp_actions.copy()
    invalid_bhp[20, 0, 3] = np.nan
    try:
        forecast_inputs(states, invalid_bhp, 20, 12)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite BHP accepted")
    observed = past_inputs(states, 20, 12)
    np.testing.assert_array_equal(past_inputs(changed, 20, 12), observed)
    from types import SimpleNamespace
    class StepForecaster:
        def predict_batch(self, histories, **kwargs):
            assert kwargs["horizon"] == 1
            for h in histories:
                yield SimpleNamespace(forecast=h[:, -1:] + 1)
    step_actions = np.array([[[[100, 1, 1]], [[100, 1, 1]], [[100, 1, 1]]]], dtype=float)
    rolled = recursive_forecast(StepForecaster(), [np.array([[1, 2], [10, 11], [100, 101]])],
                                [np.zeros((4, 5))], step_actions)
    np.testing.assert_array_equal(rolled[0, :, 0, 0], [3, 4, 5])
    step_actions[0, 1, 0, 1] = 2
    converted = recursive_forecast(StepForecaster(), [np.array([[1, 2], [10, 11], [100, 101]])],
                                  [np.zeros((4, 5))], step_actions)
    np.testing.assert_array_equal(converted[0, :, 0, 0], [3, 0, 1])
    assert converted[0, 1, 0, 1] == 0 and converted[0, 1, 0, 2] > 0
    print("forecast alignment / leakage / field injection / BHP channel checks passed", flush=True)


def metrics(truth, prediction):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid forecast")
    return {
        "oil_wape": wape(truth[..., 0], prediction[..., 0]),
        "liquid_wape": wape(truth[..., 1], prediction[..., 1]),
        "pressure_rmse_bar": rmse(truth[..., 2], prediction[..., 2]),
    }


def recursive_forecast(forecaster, histories, covariates, actions, block_size=1):
    """Advance on predictions only; no future observations enter this rollout."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    context = histories[0].shape[-1]
    histories = [h.copy() for h in histories]
    steps = []
    for step in range(0, actions.shape[1], block_size):
        horizon = min(block_size, actions.shape[1] - step)
        forecasts = list(forecaster.predict_batch(
            [h[:, -context:] for h in histories], horizon=horizon,
            past_future_covariates=[c[:, step:step + context + horizon] for c in covariates],
            use_symmetric_averaging=False, make_positive=True, return_quantiles=True,
        ))
        raw = np.stack([f.forecast for f in forecasts]).reshape(len(histories), -1, 3, horizon).transpose(0, 3, 1, 2)
        projected = _project_physics(raw, actions[:, step:step + horizon], zero_injectors=True)[0]
        steps.append(projected)
        histories = [np.concatenate([h, p.transpose(1, 2, 0).reshape(-1, horizon)], axis=1)
                     for h, p in zip(histories, projected, strict=True)]
    return np.concatenate(steps, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--full-field", action="store_true", help="Use Forecaster directly; retain every target and control channel")
    parser.add_argument("--past-covariates", action="store_true", help="Add hash-verified historical CHDD observations")
    parser.add_argument("--recursive", action="store_true", help="Also test one-month steps fed only their own predictions")
    parser.add_argument("--start-date", type=pd.Timestamp, help="First forecast origin, e.g. 2007-01-01")
    parser.add_argument("--all-windows", action="store_true", help="Evaluate every complete six-month window from the selected start")
    parser.add_argument("--interwell-model", type=Path, help="Verified v5 artifact supplying geology for the eight-scenario refit")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.bundle or not args.output or min(args.windows, args.context, args.batch_size, args.horizon) < 1:
        parser.error("positive window/context/batch sizes and bundle/output required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = (args.bundle / "training/metrics.json").read_bytes()
    if sha256(data).hexdigest() != METRICS_SHA256:
        raise ValueError("frozen KT2 metrics hash mismatch")
    reference = json.loads(data)
    trajectories, sources, observed = {}, [], {}
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
        if args.past_covariates and scenario in reference["test_scenarios"]:
            path = args.bundle / f"scenario-runs/{scenario}/canonical/chdd.csv"
            raw = path.read_bytes()
            proof = json.loads((args.bundle / f"scenario-runs/manifests/{scenario}.json").read_text())
            if sha256(raw).hexdigest() != proof["outputs"]["chdd_csv"]["sha256"]:
                raise ValueError(f"historical covariate hash mismatch: {path}")
            frame = pd.read_csv(path, dtype={"well": str})
            frame["DATA"] = pd.to_datetime(frame["DATA"])
            grid = pd.MultiIndex.from_product([trajectory.dates, trajectory.well_ids], names=["DATA", "well"])
            values = frame.set_index(["DATA", "well"]).reindex(grid)[list(PAST_FIELDS)].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("observed covariates require a complete finite date/well grid")
            observed[scenario] = values.reshape(len(trajectory.dates), len(trajectory.well_ids), -1)
            sources.append({"path": str(path), "sha256": sha256(raw).hexdigest()})
    train_ids, test_ids = reference["train_scenarios"], reference["test_scenarios"]
    if set(train_ids) & set(test_ids) or set(train_ids + test_ids) != set(trajectories):
        raise ValueError("invalid scenario split")
    print("Refitting CRM+LightGBM on the original eight training scenarios", flush=True)
    connectivity = None
    if args.interwell_model:
        connectivity = Track2Surrogate.load(args.interwell_model).baseline.connectivity
        if connectivity is None:
            raise ValueError("interwell artifact has no geological connectivity")
        sources.append({"path": str(args.interwell_model / "manifest.json"),
                        "sha256": sha256((args.interwell_model / "manifest.json").read_bytes()).hexdigest()})
    surrogate = Track2Surrogate.fit(
        [trajectories[s] for s in train_ids], seed=20260831, connectivity=connectivity
    )
    cases, truths, controls, contexts, covariates, crm, persistence = [], [], [], [], [], [], []
    sensitivity = []
    past_covariates = []
    for scenario in test_ids:
        t = trajectories[scenario]
        if args.start_date is None:
            origins = [o for o in range(0, len(t.dates) - args.horizon, args.horizon) if o >= args.context]
        else:
            first = max(args.context, int(t.dates.searchsorted(args.start_date)))
            origins = list(range(first, len(t.dates) - args.horizon, args.horizon))
            if args.all_windows and origins and origins[-1] != len(t.dates) - args.horizon - 1:
                origins.append(len(t.dates) - args.horizon - 1)
        if len(origins) < args.windows:
            raise ValueError("not enough complete windows with the requested history")
        for origin in origins if args.all_windows else origins[-args.windows:]:
            action = t.actions[origin : origin + args.horizon]
            target, cov = forecast_inputs(t.states, t.actions, origin, args.context, args.horizon)
            contexts.extend(target)
            covariates.extend(cov)
            if args.past_covariates:
                past_covariates.append(past_inputs(observed[scenario], origin, args.context).reshape(-1, args.context))
            truth = t.states[origin + 1 : origin + args.horizon + 1]
            truths.append(truth)
            controls.append(action)
            predicted = surrogate.rollout(t.states[origin], action).mean
            crm.append(predicted)
            persistence.append(np.repeat(t.states[origin][None], args.horizon, axis=0))
            changed = action.copy()
            injection = action[..., 1] == 2
            changed[..., 0][injection] *= 1.2
            perturbed = surrogate.rollout(t.states[origin], changed).mean
            producers = ~injection
            sensitivity.append(float(np.max(np.abs(perturbed[..., 0][producers] - predicted[..., 0][producers]))))
            cases.append({"scenario_id": scenario, "origin": origin,
                          "state_date": str(t.dates[origin].date()),
                          "target_dates": [str(d.date()) for d in t.dates[origin + 1 : origin + args.horizon + 1]],
                          "well_ids": list(t.well_ids)})
    truth, actions = np.stack(truths), np.stack(controls)
    predictions = {"persistence": np.stack(persistence), "crm_lightgbm": np.stack(crm)}
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator, TimesFM3Forecaster
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment must run on an A100 GPU")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.35)
    forecast_class = TimesFM3Forecaster if args.full_field else TimesFM3Evaluator
    forecaster = forecast_class(ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch", revision=MODEL_REVISION,
        per_core_batch_size=args.batch_size, device="cuda",
    ))
    joint_contexts = list(np.stack(contexts).reshape(len(cases), -1, args.context))
    joint_covariates = list(np.stack(covariates)[:, :-1].reshape(len(cases), -1, args.context + args.horizon))
    timings = {}
    variants = [
        ("timesfm3_per_well_history", contexts, None, None, args.batch_size),
        ("timesfm3_joint_history", joint_contexts, None, None, 1),
        ("timesfm3_joint_controls", joint_contexts, joint_covariates, None, 1),
    ]
    if args.past_covariates:
        variants.append(("timesfm3_joint_all_covariates", joint_contexts, joint_covariates, past_covariates, 1))
    for name, inputs, cov, past_cov, batch_size in variants:
        forecaster.config = replace(forecaster.config, per_core_batch_size=batch_size)
        print(f"Forecasting {name}: {len(inputs)} windows, {inputs[0].shape[0]} targets", flush=True)
        begin = time.monotonic()
        forecast = list(forecaster.predict_batch(
            inputs, horizon=args.horizon, past_future_covariates=cov,
            past_only_covariates=past_cov,
            use_symmetric_averaging=False, make_positive=True, return_quantiles=True,
        ))
        raw = np.stack([f.forecast for f in forecast]).reshape(len(cases), -1, 3, args.horizon)
        predictions[name] = _project_physics(raw.transpose(0, 3, 1, 2), actions, zero_injectors=True)[0]
        timings[name] = round(time.monotonic() - begin, 3)
        print(json.dumps({"model": name, **metrics(truth, predictions[name]), "seconds": timings[name]}), flush=True)
    if args.recursive:
        forecaster.config = replace(forecaster.config, per_core_batch_size=1)
        begin = time.monotonic()
        name = "timesfm3_recursive_joint_controls"
        predictions[name] = recursive_forecast(forecaster, joint_contexts, joint_covariates, actions)
        timings[name] = round(time.monotonic() - begin, 3)
        print(json.dumps({"model": name, **metrics(truth, predictions[name]), "seconds": timings[name]}), flush=True)
    report = {
        "schema": "timesoil.timesfm3-offline-benchmark/v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "gpu": torch.cuda.get_device_name(0),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION, "frozen_metrics_sha256": METRICS_SHA256,
        "timesfm_interface": forecast_class.__name__,
        "all_control_channels_retained": args.full_field,
        "past_only_fields": list(PAST_FIELDS) if args.past_covariates else [],
        "requested_start_date": str(args.start_date) if args.start_date is not None else None,
        "crm_interwell_geology": connectivity is not None,
        "packages": {n: version(n) for n in ("timesfm", "torch", "numpy", "pandas", "lightgbm")},
        "input_sources": sources, "train_scenarios": train_ids, "test_scenarios": test_ids,
        "cases": cases, "horizon": args.horizon, "context": args.context,
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
            (f"All complete {args.horizon}-month windows from requested start; final window may overlap to cover the last report."
             if args.all_windows else f"Last {args.windows} complete {args.horizon}-month windows per held-out scenario, not the full KT2 test set."),
            "TimesFM receives observed history; CRM uses the current state and eight training scenarios.",
            ("Joint controls retain all 309 well targets and 412 planned rate/status channels."
             if args.full_field else
             "Evaluator chunks targets and samples 31 of 412 control channels; not full-field joint inference."),
            "No explicit geological connectivity features; attention alone does not prove causal validity.",
            "Known planned closures are status covariates; no unprovided failure schedule or water quota is invented.",
            "No tuning on test outcomes, no independently calibrated uncertainty, no NPV improvement claim.",
        ],
    }
    np.savez_compressed(args.output / "predictions.npz", truth=truth, actions=actions, **predictions)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
