"""Economic target mapping must preserve volumes, pump rates and conversion history."""

from copy import deepcopy
from datetime import date
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from timesoil.aios.economics import CHDD_FIELDS

spec = importlib.util.spec_from_file_location('timesfm_economics',
    Path(__file__).resolve().parents[1] / 'scripts/timesfm_economics.py')
economics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(economics)


def test_forecast_checks_own_rate_bhp_role_and_shut_controls():
    from datetime import date
    from dataclasses import replace
    from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
    from timesoil.aios.operating_constraints import own_control_constraints

    month = date(2007, 1, 1)
    controls = [ControlAction(month, 'P', WellRole.PRODUCER, WellStatus.OPEN,
                             ControlTarget.LIQUID_RATE, 25., 50.),
                ControlAction(month, 'I', WellRole.INJECTOR, WellStatus.OPEN,
                             ControlTarget.WATER_INJECTION_RATE, 40., 300.),
                ControlAction(month, 'S', WellRole.PRODUCER, WellStatus.SHUT,
                             ControlTarget.OIL_RATE, 0., 50.)]
    rules = own_control_constraints(controls)
    forecast = np.zeros((1, 3, 9))
    forecast[0, 0, [1, 4]] = [20., 60.]
    forecast[0, 1, [2, 4]] = [30., 280.]
    def violations(values):
        return economics.economic_constraint_violations(values, ['2007-02-01'], ['P', 'I', 'S'], rules)
    assert not violations(forecast)
    for well, column, value, expected in [(0, 1, 26., 'max_liquid_m3d'),
            (1, 2, 41., 'max_injection_m3d'), (0, 4, 49., 'min_bhp_bar'),
            (1, 4, 301., 'max_bhp_bar'), (0, 2, 1., 'max_injection_m3d'),
            (1, 1, 1., 'max_liquid_m3d'), (2, 0, 1., 'unavailable')]:
        bad = forecast.copy(); bad[0, well, column] = value
        assert expected in violations(bad)[0]
    stopped = forecast.copy(); stopped[..., :3] = 0.; stopped[..., 4] = 0.
    assert not violations(stopped)
    unsupported = own_control_constraints([replace(controls[0], target=ControlTarget.OIL_RATE)])
    with pytest.raises(ValueError, match='max_oil_m3d'):
        economics.validate_economic_constraints(unsupported)


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


def _nine_target_forecast():
    """Three months of one producer and one injector, in the nine economic targets."""
    from timesoil.aios.operating_constraints import parse_constraints

    wells = ['P', 'I']
    stamps = ['2007-02-01', '2007-03-01', '2007-04-01']
    values = np.zeros((3, 2, 9))
    values[:, 0, [0, 1, 3, 4, 5]] = [8., 100., 120., 60., 1.]
    values[:, 1, [2, 3, 4, 5]] = [50., 130., 280., 1.]
    # 800 kg/m3 oil: 80 t of oil is 100 m3; 1000 kg/m3 water: 1 t of water is 1 m3.
    values[:, 0, 6] = 80.
    values[:, 0, 7] = [3080., 2780., 3080.]
    values[:, 1, 8] = [3100., 2800., 6200.]
    densities = {well: {'oil_kg_m3': 800., 'water_kg_m3': 1000.} for well in wells}
    return values, stamps, wells, densities, parse_constraints


def test_derived_field_vectors_convert_mass_with_export_densities_only():
    values, stamps, wells, densities, _ = _nine_target_forecast()
    derived = economics.derived_field_vectors(values, stamps, wells, densities,
                                              oil_fvf=1.1, water_fvf=1.05, vrr_window=3)
    np.testing.assert_allclose(derived['WOPT_DELTA'][:, 0], 100.)
    np.testing.assert_allclose(derived['WWPT_DELTA'][:, 0], [3000., 2700., 3000.])
    np.testing.assert_allclose(derived['WLPT_DELTA'][:, 0], [3100., 2800., 3100.])
    np.testing.assert_allclose(derived['WVPT_DELTA'][:, 0], [3260., 2945., 3260.])
    np.testing.assert_allclose(derived['WVIT_DELTA'][:, 1], [3255., 2940., 6510.])
    assert not derived['WVPT_DELTA'][:, 1].any() and not derived['WVIT_DELTA'][:, 0].any()
    field = derived['field']
    assert field['months'] == ['2007-01-01', '2007-02-01', '2007-03-01']
    np.testing.assert_allclose(field['liquid_m3d'], 100.)          # 3100/31, 2800/28, 3100/31
    np.testing.assert_allclose(field['injection_m3d'], [100., 100., 200.])
    np.testing.assert_allclose(field['water_m3d'], [3000 / 31, 2700 / 28, 3000 / 31])
    np.testing.assert_allclose(field['vrr3'], [3255 / 3260, 6195 / 6205, 12705 / 9465])
    assert derived['formation_volume_factors'] == {'oil': 1.1, 'water': 1.05}
    unit = economics.derived_field_vectors(values, stamps, wells, densities)
    np.testing.assert_allclose(unit['WVPT_DELTA'][:, 0], unit['WLPT_DELTA'][:, 0])
    manifest = {'conversion': {'density_by_well': {'P': {'oil_kg_m3': 800, 'water_kg_m3': 1000,
                                                         'provenance': 'deck'}}}}
    assert economics.export_densities(manifest, ['P'])['P'] == {'oil_kg_m3': 800., 'water_kg_m3': 1000., 'method': 'well_surface_density'}
    with pytest.raises(ValueError, match='misses positive per-well densities'):
        economics.export_densities(manifest, wells)
    with pytest.raises(ValueError, match='misses positive per-well densities'):
        economics.export_densities({'conversion': {'density_by_well': {'P': {'oil_kg_m3': 0,
                                                                            'water_kg_m3': 1000}}}}, ['P'])
    with pytest.raises(ValueError, match='explicit density'):
        economics.derived_field_vectors(values, stamps, wells, {'P': densities['P']})
    for bad in ({'oil_fvf': 0}, {'water_fvf': float('nan')}, {'vrr_window': 0}, {'vrr_window': 3.0}):
        with pytest.raises(ValueError):
            economics.derived_field_vectors(values, stamps, wells, densities, **bad)


def test_forecast_gates_k1_k2_k4_k5_need_the_derived_volume_increments():
    values, stamps, wells, densities, parse_constraints = _nine_target_forecast()
    derived = economics.derived_field_vectors(values, stamps, wells, densities,
                                              oil_fvf=1.1, water_fvf=1.05, vrr_window=3)

    def rules(window=1, status='hard', **limits):
        return parse_constraints([dict(start='2007-01-01', end='2007-03-01', wells=wells,
                                       limits=limits, window_months=window, status=status)],
                                 wells=wells, start=date(2007, 1, 1), end=date(2007, 3, 1))

    def verdicts(**kwargs):
        return economics.economic_constraint_verdicts(values, stamps, wells, rules(**kwargs), derived=derived)

    def violations(**kwargs):
        return economics.economic_constraint_violations(values, stamps, wells, rules(**kwargs), derived=derived)

    # Water rules stay unsupported until the derived vectors are actually supplied.
    for key in ('max_monthly_liquid_m3d', 'max_monthly_injection_m3d', 'max_monthly_water_deficit_m3',
                'max_window_voidage_replacement'):
        with pytest.raises(ValueError, match='lack required vectors'):
            economics.economic_constraint_violations(values, stamps, wells, rules(**{key: 1}))
    assert not violations(max_monthly_liquid_m3d=150)                       # K1: 100 m3/day every month
    assert len(violations(max_monthly_liquid_m3d=90)) == 3
    assert [v.month.isoformat() for v in verdicts(max_monthly_injection_m3d=150) if not v.ok] == ['2007-03-01']
    k4 = [v for v in verdicts(max_window_voidage_replacement=1.15, window=3) if not v.ok]
    assert [v.month.isoformat() for v in k4] == ['2007-03-01']
    assert k4[0].worst_value == pytest.approx(12705 / 9465) and k4[0].margin < 0
    assert len(violations(max_monthly_water_deficit_m3=0)) == 3             # K5: injection above produced water
    assert [v.month.isoformat() for v in verdicts(max_monthly_water_deficit_m3=150) if not v.ok] == ['2007-03-01']
    diagnostic = verdicts(max_monthly_water_deficit_m3=0, status='diagnostic')
    assert sum(not v.ok for v in diagnostic) == 3
    assert not economics.economic_constraint_violations(
        values, stamps, wells, rules(max_monthly_water_deficit_m3=0, status='diagnostic'), derived=derived)
    with pytest.raises(ValueError, match='do not match the forecast'):
        economics.economic_constraint_verdicts(values, stamps, ['I', 'P'], rules(max_monthly_liquid_m3d=1),
                                               derived=derived)


def test_export_densities_uses_connection_densities_for_multi_pvt_wells():
    import pytest
    from timesfm_economics import export_densities
    manifest = {'conversion': {
        'density_by_well': {'1': {'oil_kg_m3': 850.0, 'water_kg_m3': 1000.0}},
        'connection_density_by_well': {'2': {'connection_count': 3, 'oil_kg_m3': [840.0, 860.0], 'water_kg_m3': [1000.0]},
                                       '3': {'connection_count': 1, 'oil_kg_m3': [], 'water_kg_m3': []}}}}
    got = export_densities(manifest, ['1', '2'])
    assert got['1'] == {'oil_kg_m3': 850.0, 'water_kg_m3': 1000.0, 'method': 'well_surface_density'}
    assert got['2']['method'] == 'connection_mean' and got['2']['oil_kg_m3'] == 850.0 and got['2']['water_kg_m3'] == 1000.0
    with pytest.raises(ValueError, match="densities for: 3"):
        export_densities(manifest, ['1', '2', '3'])
