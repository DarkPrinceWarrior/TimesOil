"""Economic target mapping must preserve volumes, pump rates and conversion history."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from timesoil.aios.economics import CHDD_FIELDS

spec = importlib.util.spec_from_file_location('timesfm_economics',
    Path(__file__).resolve().parents[1] / 'scripts/timesfm_economics.py')
economics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(economics)


def test_economic_targets_roundtrip_and_reject_incomplete_forecasts(tmp_path):
    def row(stamp, well, **updates):
        return {'DATA': stamp, 'well': well, **dict.fromkeys(CHDD_FIELDS[2:], 0.), **updates}

    history = [row('2007-01-01', 'a', WOMT=50., WLPT=90., WOMR=2., WLPR=3., WEFF=1.),
               row('2007-01-01', 'b')]
    future = [row('2007-02-01', 'a', WOMT=50., WLPT=90., WWIR=10., WWIT=123., WWIT_Diff=123., BHP=280., WEFF=1.),
              row('2007-02-01', 'b', WOMT=7., WLPT=11., WOMT_Diff=7., WLPT_Diff=11., WOMR=2., WLPR=3., WEFF=1.),
              row('2007-03-01', 'a', WOMT=50., WLPT=90., WWIR=15., WWIT=144., WWIT_Diff=21., BHP=280., WEFF=1.),
              row('2007-03-01', 'b', WOMT=12., WLPT=19., WOMT_Diff=5., WLPT_Diff=8., WOMR=1., WLPR=2., WEFF=1.)]
    stamps, wells = ['2007-02-01', '2007-03-01'], ['b', 'a']
    values = economics.economic_targets(reversed(future), stamps, wells)
    pristine = deepcopy(history)
    result = economics.forecast_chdd_rows(history, stamps, wells, values)
    assert result == history + future
    assert history == pristine
    assert result[2]['WWIT'] == 123.  # Not endpoint injection rate × 31 days.
    assert result[3]['WOMT_Diff'] == 7.  # Not endpoint oil rate × 31 days.
    for bad in (values[..., :3], values * np.nan, -np.ones_like(values)):
        with pytest.raises(ValueError):
            economics.forecast_chdd_rows(history, stamps, wells, bad)
    efficiency = values.copy(); efficiency[..., 5] = 1.01
    with pytest.raises(ValueError, match='WEFF'):
        economics.forecast_chdd_rows(history, stamps, wells, efficiency)
    with pytest.raises(ValueError, match='without future'):
        economics.forecast_chdd_rows(history + future, stamps, wells, values)
    with pytest.raises(ValueError, match='every well'):
        economics.forecast_chdd_rows([row('2006-12-01', 'a'), history[1]], stamps, wells, values)
    with pytest.raises(ValueError, match='complete'):
        economics.economic_targets(future[:-1], stamps, wells)
    with pytest.raises(ValueError, match='consecutive'):
        economics.economic_targets(future, ['2007-02-01', '2007-04-01'], wells)
    actions = np.array([[[10., 0., 1., 70.], [100., 2., 1., 280.]],
                        [[10., 0., 0., 70.], [100., 2., 1., 280.]]])
    forecast = np.ones((2, 2, 9)); forecast[..., 8] = 123.; forecast[..., 2] = 5.
    projected = economics.project_economic_forecast(forecast, actions)
    assert projected[0, 1, 8] == 123. and projected[0, 1, 2] == 5.
    assert not projected[0, 1, [0, 1, 6, 7]].any()
    assert not projected[0, 0, [2, 8]].any()
    assert not projected[1, 0, [0, 1, 2, 6, 7, 8]].any()
    import csv
    from hashlib import sha256
    from types import SimpleNamespace
    import pandas as pd

    path = tmp_path / 'chdd.csv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=CHDD_FIELDS)
        writer.writeheader(); writer.writerows(history + future)
    manifest = {'outputs': {'chdd_csv': {'sha256': sha256(path.read_bytes()).hexdigest()}}}
    trajectory = SimpleNamespace(states=np.ones((3, 2, 3)), actions=np.ones((3, 2, 4)),
        dates=pd.date_range('2007-01-01', periods=3, freq='MS'), well_ids=('a', 'b'))
    observed, inference = economics.observed_economic_history(tmp_path, manifest, trajectory, 0)
    assert observed == history and np.isnan(inference.states[1:]).all()
    np.testing.assert_array_equal(inference.states[:1],
        economics.economic_targets(history, ['2007-01-01'], trajectory.well_ids))
