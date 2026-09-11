import numpy as np
import pytest

from timesoil.aios.interwell import WellConnectivity


def _geometry() -> WellConnectivity:
    return WellConnectivity(
        ("P1", "P2", "I1", "I2"),
        np.array([[0, 0, 3, 1], [0, 0, 1, 3], [3, 1, 0, 0], [1, 3, 0, 0]]),
        np.tile([100, .2, 10], (4, 1)),
        {},
    )


def test_interwell_allocation_follows_roles_status_and_complete_fields():
    geometry = _geometry()
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


def test_from_dict_restores_an_equivalent_connectivity():
    geometry = _geometry()
    restored = WellConnectivity.from_dict({
        "well_ids": list(geometry.well_ids),
        "weights": geometry.weights.tolist(),
        "static": geometry.static.tolist(),
        "provenance": {"method": "test"},
    })
    assert restored.well_ids == geometry.well_ids
    np.testing.assert_array_equal(restored.weights, geometry.weights)
    np.testing.assert_array_equal(restored.static, geometry.static)
    assert restored.provenance == {"method": "test"}

    state = np.array([[5, 10, 200], [5, 10, 190], [0, 0, 210], [0, 0, 220.]])
    action = np.array([[100, 1, 1], [100, 1, 1], [40, 2, 1], [20, 2, 1.]])
    np.testing.assert_array_equal(
        restored.features(state, action), geometry.features(state, action)
    )

    with pytest.raises(ValueError, match="symmetric"):
        WellConnectivity.from_dict({
            "well_ids": ["P1", "P2"],
            "weights": [[0, 1], [2, 0]],
            "static": [[100, .2, 10], [100, .2, 10]],
            "provenance": {},
        })
