from datetime import date

import numpy as np
import pytest

from timesoil.aios.interwell import WellConnectivity, _components
from timesoil.aios.surrogate import ScenarioTrajectory, Track2Surrogate


def test_interwell_roles_months_barriers_and_portable_forecast(tmp_path):
    wells = ("P1", "P2", "I1", "I2")
    geometry = WellConnectivity(wells, np.array([[0, 0, 3, 1], [0, 0, 1, 3],
                                               [3, 1, 0, 0], [1, 3, 0, 0]]),
                                np.tile([100, .2, 10], (4, 1)), {})
    state = np.array([[5, 10, 200], [5, 10, 190], [0, 0, 210], [0, 0, 220.]])
    action = np.array([[100, 1, 1], [100, 1, 1], [40, 2, 1], [20, 2, 1.]])
    features = geometry.features(state, action)
    np.testing.assert_allclose(features[:, 0], [35, 25, 0, 0])
    switched = action.copy()
    switched[0], switched[2] = [40, 2, 1], [100, 1, 1]
    assert geometry.features(state, switched)[2, 0] == 40
    stacked = geometry.features(np.tile(state, (2, 1)), np.concatenate([action, switched]))
    np.testing.assert_array_equal(stacked[:4], features)
    np.testing.assert_array_equal(stacked[4:], geometry.features(state, switched))
    stopped = action.copy()
    stopped[2, 2] = 0
    assert geometry.features(state, stopped)[:, 0].sum() == 20
    with pytest.raises(ValueError, match="complete ordered"):
        geometry.features(state[:3], action[:3])
    active = np.ones((1, 1, 3), bool)
    permeability = np.ones_like(active, float)
    px = permeability.copy()
    px[0, 0, 1] = 0
    labels = _components(active, px, permeability, permeability, permeability)
    assert len(set(labels.ravel())) == 3

    dates = tuple(date(2000 + month // 12, month % 12 + 1, 1) for month in range(24))
    trajectories = []
    for scenario in range(3):
        actions = np.tile(action, (24, 1, 1))
        actions[:, 2, 0] += np.arange(24) * (scenario + 1)
        states = [state.copy()]
        for current in actions[:-1]:
            following = states[-1].copy()
            allocated = geometry.features(following, current)[:, 0]
            following[:2, 1] = .9 * following[:2, 1] + .1 * allocated[:2]
            following[:2, 0] = following[:2, 1] / 2
            states.append(following)
        trajectories.append(ScenarioTrajectory(f"s{scenario}", "test", dates, wells,
                                                np.array(states), actions))
    model = Track2Surrogate.fit(trajectories, connectivity=geometry, ensemble_size=2, n_estimators=3)
    base = model.rollout(state, np.tile(action, (6, 1, 1)))
    changed = action.copy()
    changed[2:, 0] = [20, 40]
    prediction = model.rollout(state, np.tile(changed, (6, 1, 1)))
    assert np.max(np.abs(base.mean[:, :2, 0] - prediction.mean[:, :2, 0])) > .01
    np.testing.assert_array_equal(prediction.mean[:, 2:, :2], 0)
    model.save(tmp_path)
    loaded = Track2Surrogate.load(tmp_path)
    np.testing.assert_array_equal(loaded.rollout(state, np.tile(changed, (6, 1, 1))).mean, prediction.mean)
