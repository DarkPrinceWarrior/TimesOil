from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from timesoil.aios.opm_chdd import _scheduled_controls
from timesoil.aios.scenario_generation import ScenarioGeneratorConfig, generate_control_scenarios
from timesoil.aios.surrogate import BHP_ACTION_FEATURES, ScenarioTrajectory, Track2Surrogate
from timesoil.aios.track2 import _action_cube, _trajectory_controls


def test_pressure_reaches_training_prediction_artifact_and_scenarios(tmp_path):
    rng = np.random.default_rng(9)
    trajectories = []
    for index in range(6):
        actions = np.zeros((24, 3, 4))
        actions[..., :3] = [400, 1, 1]
        actions[..., 3] = rng.uniform(60, 180, (24, 3))
        states = np.empty((24, 3, 3))
        states[0] = [70, 140, 240]
        states[1:, :, 1] = (300 - actions[:-1, :, 3]) / 2
        states[1:, :, 0] = states[1:, :, 1] / 2
        states[1:, :, 2] = 240
        trajectories.append(ScenarioTrajectory(str(index), "synthetic_check", pd.date_range(
            "2014-01-01", periods=24, freq="MS"), ("1", "2", "3"), states, actions))
    model = Track2Surrogate.fit(trajectories, ensemble_size=3, n_estimators=60)
    assert model.action_features == BHP_ACTION_FEATURES
    low, high = trajectories[0].actions[0].copy(), trajectories[0].actions[0].copy()
    low[:, 3], high[:, 3] = 70, 170
    assert np.all(model.step(trajectories[0].states[0], low).mean[:, 0]
                  > model.step(trajectories[0].states[0], high).mean[:, 0])
    model.save(tmp_path / "model")
    restored = Track2Surrogate.load(tmp_path / "model")
    assert restored.action_features == BHP_ACTION_FEATURES
    np.testing.assert_array_equal(model.rollout(trajectories[0].states[0], trajectories[0].actions).mean,
                                  restored.rollout(trajectories[0].states[0], trajectories[0].actions).mean)
    controls = _trajectory_controls(trajectories[0])
    months = tuple(t.date() for t in trajectories[0].dates)
    np.testing.assert_array_equal(_action_cube(controls, months, ("1", "2", "3"), BHP_ACTION_FEATURES),
                                  trajectories[0].actions)
    scenarios = generate_control_scenarios(controls, ScenarioGeneratorConfig(bhp_perturbation_fraction=.1))
    assert all(b.bhp_limit >= a.bhp_limit for a, b in zip(controls, scenarios[1].actions))
    legacy = [replace(t, actions=t.actions[..., :3]) for t in trajectories]
    old = Track2Surrogate.fit(legacy, ensemble_size=2, n_estimators=3)
    old.save(tmp_path / "old")
    old = Track2Surrogate.load(tmp_path / "old")
    with pytest.raises(ValueError, match="action"):
        old.step(legacy[0].states[0], low)
    invalid = trajectories[0].actions.copy()
    invalid[-1, 0, 3] = -1
    with pytest.raises(ValueError, match="invalid"):
        model.rollout(trajectories[0].states[0], invalid)
    invalid[-1, 0, 3] = 9000
    assert model.rollout(trajectories[0].states[0], invalid).ood[-1]


def test_schedule_pressure_updates_are_chronological():
    text = """SCHEDULE
WCONPROD
'1' 'OPEN' 'LRAT' 3* 100 1* 50 /
/
DATES
1 FEB 2014 /
/
WELTARG
'1' 'BHP' 70 /
/
"""
    summary = [(date(2014, month, 1), {"1": {"WLPR": 0, "WWIR": 0}}, {}) for month in (1, 2)]
    result = _scheduled_controls(text, date(2014, 1, 1), summary, ("1",))["1"]
    assert [c.bhp_limit for c in result] == [50, 70]
    assert [c.value for c in result] == [100, 100]
