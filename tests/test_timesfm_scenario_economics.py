import csv
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evaluate_timesfm_scenarios import economic_evaluation_inputs, interval_check
from timesfm_economics import ECONOMIC_TARGETS, ECONOMIC_UNITS
from timesoil.aios.economics import CHDD_FIELDS


def test_economic_evaluation_excludes_future_inputs_and_development_csv(tmp_path):
    dates = pd.date_range('2007-01-01', periods=4, freq='MS')
    rows = [dict.fromkeys(CHDD_FIELDS, 0.) for _ in dates]
    for i, row in enumerate(rows):
        row.update(DATA=dates[i].date().isoformat(), well='P', WOMR=float(i + 1),
                   WOMT_Diff=float(10 * (i + 1)), WLPT_Diff=float(20 * (i + 1)))
    path = tmp_path / 'chdd.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CHDD_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    expected = sha256(path.read_bytes()).hexdigest()
    manifest = {'outputs': {'chdd_csv': {'sha256': expected}}}
    actions = np.broadcast_to([100., 1., 1., 50.], (4, 1, 4)).copy()
    trajectory = SimpleNamespace(dates=dates, well_ids=('P',), actions=actions,
                                 states=np.full((4, 1, 3), 999999.))
    inputs, truth, actual = economic_evaluation_inputs(tmp_path, manifest, trajectory, 1, 2, {})
    assert actual == expected and truth.shape == (2, 1, 9)
    np.testing.assert_array_equal(truth[:, 0, 6], [30., 40.])
    np.testing.assert_array_equal(inputs.states[:2, 0, 0], [1., 2.])
    assert np.isnan(inputs.states[2:]).all()
    np.testing.assert_array_equal(inputs.actions, actions)
    with pytest.raises(ValueError, match='already used'):
        economic_evaluation_inputs(tmp_path, manifest, trajectory, 1, 2, {'developed': expected})
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        economic_evaluation_inputs(tmp_path, manifest, trajectory, 1, 2, {})


def test_nine_target_intervals_preserve_units_and_group_coverage():
    calibration = np.ones((5, 2, 2, 9))
    calibration[4, 1, 1, 8] = 2.
    held_out = np.ones((3, 2, 2, 9))
    held_out[2, 1, 1, 8] = 3.
    report = interval_check(calibration, held_out)
    assert list(report['radius_by_target']) == list(ECONOMIC_TARGETS)
    assert report['radius_by_target']['WWIT_Diff'] == 2.
    assert report['units_by_target'] == dict(zip(ECONOMIC_TARGETS, ECONOMIC_UNITS))
    assert report['test_whole_trajectory_coverage_by_target'] == [1.] * 8 + [2 / 3]
    assert report['test_whole_trajectory_joint_coverage'] == 2 / 3
    assert not report['guaranteed_coverage_claimed']
    assert 'radius_oil_tpd_liquid_tpd_pressure_bar' not in report
    # Three physical channels are no longer a supported forecast shape.
    for invalid in [held_out[..., :3], held_out * np.nan, -held_out, held_out[:2]]:
        with pytest.raises(ValueError, match='trajectory groups'):
            interval_check(calibration, invalid)
    with pytest.raises(ValueError, match='trajectory groups'):
        interval_check(calibration[..., :3], held_out[..., :3])
