"""Compare fixed-origin and observed-update forecasts without mixing their information budgets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmark_timesfm3 import MODEL_REVISION, forecast_inputs, metrics, recursive_forecast
from timesoil.aios.surrogate import _project_physics
from timesoil.aios.track2 import trajectory_from_frame


def forecast_blocks(forecaster, trajectory, origin, horizon, context, block, *, observe=False):
    """Observations are enabled only for the explicitly separate rolling-origin benchmark."""
    targets, cov = forecast_inputs(trajectory.states, trajectory.actions, origin, context, horizon)
    length = targets.shape[-1]
    actions = trajectory.actions[origin:origin + horizon]
    if not observe:
        return recursive_forecast(
            forecaster, [targets.reshape(-1, length)],
            [cov[:, :-1].reshape(-1, length + horizon)], actions[None], block_size=block,
        )[0]
    output = []
    for step in range(0, horizon, block):
        count = min(block, horizon - step)
        history, future_controls = forecast_inputs(
            trajectory.states, trajectory.actions, origin + step, length, count,
        )
        predicted = recursive_forecast(
            forecaster, [history.reshape(-1, length)],
            [future_controls[:, :-1].reshape(-1, length + count)],
            actions[None, step:step + count], block_size=count,
        )
        output.append(predicted[0])
    return np.concatenate(output)


def self_check():
    from types import SimpleNamespace

    class Forecaster:
        expected_channels = 8

        def predict_batch(self, histories, *, horizon, **kwargs):
            assert kwargs["past_future_covariates"][0].shape[0] == self.expected_channels
            for history in histories:
                yield SimpleNamespace(forecast=history[:, -1:] + np.arange(1, horizon + 1))

    states = np.ones((10, 2, 3))
    actions = np.ones_like(states)
    actions[..., 0] = 100
    actions[:, 1, 1] = 2
    t = SimpleNamespace(states=states, actions=actions)
    forecast = forecast_blocks(Forecaster(), t, 3, 5, 3, 2)
    np.testing.assert_array_equal(forecast[:, 0, 0], [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(forecast[:, 1, :2], 0)
    changed = states.copy()
    changed[4:] = 40
    other = SimpleNamespace(states=changed, actions=actions)
    np.testing.assert_array_equal(forecast, forecast_blocks(Forecaster(), other, 3, 5, 3, 2))
    observed = forecast_blocks(Forecaster(), other, 3, 5, 3, 2, observe=True)
    assert observed[2, 0, 0] == 41 and observed[0, 0, 0] == 2
    Forecaster.expected_channels = 10
    with_bhp = SimpleNamespace(states=states, actions=np.concatenate(
        [actions, np.full((*actions.shape[:2], 1), 70.0)], axis=-1))
    np.testing.assert_array_equal(forecast_blocks(Forecaster(), with_bhp, 3, 5, 3, 2), forecast)
    forecast_blocks(Forecaster(), with_bhp, 3, 5, 3, 2, observe=True)
    print("fixed-origin leakage, state carry, role and incomplete-final-block checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--start", type=pd.Timestamp)
    parser.add_argument("--horizons", type=int, nargs="+", default=[23, 224])
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.run or args.start is None or not args.output or min(args.horizons + args.blocks + [args.context]) < 1:
        parser.error("run, start, output and positive horizons/blocks/context required")
    csv_path = args.run / "canonical/trajectory.csv"
    raw = csv_path.read_bytes()
    export = json.loads((args.run / "canonical/manifest.json").read_text())
    if sha256(raw).hexdigest() != export["outputs"]["track2_csv"]["sha256"]:
        raise ValueError("canonical trajectory hash mismatch")
    trajectory = trajectory_from_frame(pd.read_csv(csv_path))
    origin = int(trajectory.dates.get_loc(args.start))
    if origin < 1 or origin + max(args.horizons) >= len(trajectory.dates):
        raise ValueError("requested horizon is not fully covered by the source trajectory")
    context = min(args.context, origin)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this benchmark requires the allocated A100 GPU")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    model = TimesFM3Forecaster(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
        revision=MODEL_REVISION, per_core_batch_size=1, device="cuda"))
    report = {
        "schema": "timesoil.horizon-benchmark/v1", "as_of": "2026-09-09",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_trajectory_sha256": sha256(raw).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "forecast_helper_sha256": sha256(Path(__file__).with_name("benchmark_timesfm3.py").read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION, "context_months": context,
        "observation_cutoff": str(args.start.date()), "well_count": len(trajectory.well_ids),
        "future_controls": "source schedule assumed known; retrospective training-archive experiment",
        "learned_on_future_local_states": False, "official_chdd": False,
        "claim": "Forecast accuracy only; observed-update rows are not autonomous policy rollouts or causal evidence.",
        "results": [],
    }
    for horizon in args.horizons:
        truth = trajectory.states[origin + 1:origin + horizon + 1]
        actions = trajectory.actions[origin:origin + horizon]
        variants = [("fixed_origin_direct", horizon, False)]
        variants += [(f"fixed_origin_block_{b}", b, False) for b in args.blocks if b != horizon]
        variants += [("observed_update_block_6", 6, True)]
        outputs = {"truth": truth, "actions": actions}
        outputs["persistence"] = _project_physics(
            np.repeat(trajectory.states[origin][None], horizon, axis=0), actions, zero_injectors=True,
        )[0]
        for name, block, observe in variants:
            started = time.monotonic()
            prediction = forecast_blocks(model, trajectory, origin, horizon, context, block, observe=observe)
            outputs[name] = prediction
            row = {
                "name": name, "horizon_months": horizon, "block_months": block,
                "new_observations_between_blocks": observe,
                "seconds": time.monotonic() - started, **metrics(truth, prediction),
                "end_exclusive": str(trajectory.dates[origin + horizon].date()),
                "prefix_metrics": {str(h): metrics(truth[:h], prediction[:h])
                                   for h in [1, 3, 6, 12, 23, 60, 120, 224] if h <= horizon},
            }
            report["results"].append(row)
            (args.output / "metrics.partial.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "prefix_metrics"}), flush=True)
        report["results"].append({"name": "persistence", "horizon_months": horizon,
                                  "new_observations_between_blocks": False, **metrics(truth, outputs["persistence"])})
        np.savez_compressed(args.output / f"predictions-{horizon}.npz", **outputs)
    report["complete"] = True
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
