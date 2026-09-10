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


def test_forecast_limits_use_surface_units_groups_and_control_months():
    from datetime import date
    from timesoil.aios.operating_constraints import parse_constraints, check_observed

    start, end = date(2007, 1, 1), date(2007, 2, 1)
    wells = ['P', 'I', 'I2']
    prediction = np.zeros((2, 3, 9))
    prediction[:, 0, 0] = 8  # Oil mass is not the missing oil surface-volume rate.
    prediction[:, 0, 1] = 100
    prediction[:, 1:, 2] = 20
    prediction[:, :, 4] = [70, 280, 280]
    stamps = ['2007-02-01', '2007-03-01']
    def rules(**limits):
        return parse_constraints([dict(start=str(start), end=str(start), wells=wells, limits=limits)],
                                 wells=wells, start=start, end=end)
    checked = rules(max_liquid_m3d=100, min_injection_m3d=40, max_injection_m3d=40,
                    min_bhp_bar=70, max_bhp_bar=280)
    assert not economics.economic_constraint_violations(prediction, stamps, wells, checked)
    for key, value in [('max_liquid_m3d', 99), ('min_injection_m3d', 41),
                       ('max_injection_m3d', 39), ('min_bhp_bar', 71), ('max_bhp_bar', 279)]:
        errors = economics.economic_constraint_violations(prediction, stamps, wells, rules(**{key:value}))
        assert len(errors) == 1 and key in errors[0] and '2007-01-01' in errors[0]
    later = prediction.copy(); later[1, 0, 1] = 1000
    assert not economics.economic_constraint_violations(later, stamps, wells, checked)
    for key in ('max_oil_m3d', 'max_watercut', 'max_monthly_water_deficit_m3',
                'min_monthly_voidage_replacement', 'max_monthly_voidage_replacement'):
        with pytest.raises(ValueError, match='lack required vectors'):
            economics.economic_constraint_violations(prediction, stamps, wells, rules(**{key:1}))
    outage = parse_constraints([dict(start=str(start), end=str(start), wells=['P'], unavailable=True)],
                               wells=wells, start=start, end=end)
    assert 'unavailable' in economics.economic_constraint_violations(prediction, stamps, wells, outage)[0]
    stopped = prediction.copy(); stopped[:, 0, :3] = 0
    assert not economics.economic_constraint_violations(stopped, stamps, wells, outage)
    rows = {well:dict(WLPR=0, WWIR=0, WBHP=0) for well in wells}
    with pytest.raises(ValueError, match='required vectors'):
        check_observed(rules(max_watercut=1), start, rows)
    with pytest.raises(ValueError, match='required vectors'):
        check_observed(outage, start, rows)
    rows['P']['WOPR'] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        check_observed(checked, start, rows)


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
    import json
    extra = tmp_path / 'development'; extra.mkdir()
    exported = extra / 'candidate-00'; exported.mkdir()
    (exported / 'chdd.csv').write_bytes(path.read_bytes())
    (exported / 'manifest.json').write_text(json.dumps({'outputs': {'chdd_csv': {
        'name':'chdd.csv', 'sha256':sha256(path.read_bytes()).hexdigest()}}}))
    (extra / 'manifest.json').write_text(json.dumps({
        'schema':'timesoil.frozen-forecast-evaluation-cases/v1', 'calibration_cases':[0],
        'scenarios':[{'index':0, 'directory':str(exported),
            'export_manifest_sha256':sha256((exported / 'manifest.json').read_bytes()).hexdigest()},
            {'index':1, 'directory':'unread-held-out-case'}]}))
    (tmp_path / 'manifest.json').write_text(json.dumps({'scenarios':[{
        'scenario_id':'baseline', 'canonical_chdd':'chdd.csv',
        'canonical_chdd_sha256':sha256(path.read_bytes()).hexdigest()}]}))
    physical = [SimpleNamespace(**vars(trajectory), scenario_id=name)
                for name in ('baseline', 'physical-sweep-00')]
    loaded = economics.load_economic_trajectories(tmp_path, physical, 0, extra_batches=[extra])
    assert len(loaded) == 2 and loaded[0].states.shape == (3, 2, 9)
    np.testing.assert_array_equal(loaded[0].states, loaded[1].states)
    (exported / 'chdd.csv').write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='CSV hash'):
        economics.load_economic_trajectories(tmp_path, physical, 0, extra_batches=[extra])
