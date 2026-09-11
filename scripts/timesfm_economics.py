"""Lossless economic forecast targets; endpoint rates never stand in for monthly volumes."""

from __future__ import annotations

import argparse
import calendar
import csv
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from timesoil.aios.economics import (
    CHDD_FIELDS, CHDDEconomicsAdapter, normalize_chdd_rows, opm_management_rows,
)


# These nine outputs preserve every independent input to the official calculator.
# THP is the canonical export's WBP9 diagnostic, not tubing-head pressure.
ECONOMIC_TARGETS = ('WOMR', 'WLPR', 'WWIR', 'THP', 'BHP', 'WEFF',
                    'WOMT_Diff', 'WLPT_Diff', 'WWIT_Diff')
ECONOMIC_UNITS = ('t/day', 'm3/day', 'm3/day', 'bar', 'bar', 'fraction',
                  't/month', 't/month', 'm3/month')
TOTAL_DIFFS = (('WOMT', 'WOMT_Diff'), ('WLPT', 'WLPT_Diff'), ('WWIT', 'WWIT_Diff'))


def _monthly_grid(timestamps, well_ids):
    stamps = tuple(date.fromisoformat(str(value)) for value in timestamps)
    wells = tuple(well_ids)
    if not stamps or any(stamp.day != 1 for stamp in stamps):
        raise ValueError('monthly report endpoints required')
    if any(right != (left.replace(day=28) + timedelta(days=4)).replace(day=1)
           for left, right in zip(stamps, stamps[1:])):
        raise ValueError('report endpoints must be consecutive and ordered')
    if not wells or any(not isinstance(well, str) or not well for well in wells) or len(set(wells)) != len(wells):
        raise ValueError('unique nonempty string well IDs required')
    return stamps, wells


def economic_targets(records, timestamps, well_ids):
    """Extract explicit endpoint and elapsed-month targets in the requested well order."""
    stamps, wells = _monthly_grid(timestamps, well_ids)
    rows = normalize_chdd_rows(records)
    grid = {(str(row['DATA']), str(row['well'])): row for row in rows}
    expected = {(stamp.isoformat(), well) for stamp in stamps for well in wells}
    if set(grid) != expected:
        raise ValueError('economic targets require the exact complete date × well grid')
    targets = np.array([[[grid[stamp.isoformat(), well][field] for field in ECONOMIC_TARGETS]
                         for well in wells] for stamp in stamps], dtype=np.float64)
    _validate_targets(targets, len(stamps), len(wells))
    return targets


def _validate_targets(values, months, wells):
    if (values.shape != (months, wells, len(ECONOMIC_TARGETS))
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('nine finite nonnegative economic targets per well-month required')
    if (values[..., ECONOMIC_TARGETS.index('WEFF')] > 1).any():
        raise ValueError('WEFF must be a fraction in [0, 1]')


def forecast_chdd_rows(history, timestamps, well_ids, predictions):
    """Append forecasts to observed history; integrate predicted increments only.

    History must end at the forecast origin. Neither future physical rows nor
    planned rates are accepted as substitutes for a missing forecast channel.
    """
    stamps, wells = _monthly_grid(timestamps, well_ids)
    values = np.asarray(predictions, dtype=np.float64)
    _validate_targets(values, len(stamps), len(wells))
    rows = normalize_chdd_rows(history)
    origin = (stamps[0] - timedelta(days=1)).replace(day=1).isoformat()
    if max(str(row['DATA']) for row in rows) != origin:
        raise ValueError('history must end exactly at the forecast origin, without future rows')
    if {str(row['well']) for row in rows} != set(wells):
        raise ValueError('forecast and historical well sets differ')
    last = {str(row['well']): row for row in rows if row['DATA'] == origin}
    if set(last) != set(wells):
        raise ValueError('history needs every well at the forecast origin')
    totals = {well: {field: float(last[well][field]) for field, _ in TOTAL_DIFFS} for well in wells}
    for month, stamp in enumerate(stamps):
        for column, well in enumerate(wells):
            row = dict(zip(ECONOMIC_TARGETS, values[month, column].tolist()))
            for field, increment in TOTAL_DIFFS:
                totals[well][field] += row[increment]
                row[field] = totals[well][field]
            rows.append({'DATA': stamp.isoformat(), 'well': well, **row})
    return normalize_chdd_rows(rows)


def load_economic_trajectories(batch, trajectories, origin, *, extra_batches=()):
    """Use canonical CSVs from the already verified development batch, retaining its split."""
    from types import SimpleNamespace

    records = {row['scenario_id']: row for row in json.loads((batch / 'manifest.json').read_text())['scenarios']}
    paths = {name: ((batch / row['canonical_chdd']).resolve(), row['canonical_chdd_sha256'], batch)
             for name, row in records.items()}
    for extra in extra_batches:
        manifest = json.loads((extra / 'manifest.json').read_text())
        prefix = 'bhp-only' if manifest['schema'] == 'timesoil.bhp-only-forecast-evaluation/v1' else 'physical-sweep'
        for record in manifest['scenarios']:
            if record['index'] not in manifest['calibration_cases']:
                continue
            root = (extra / f"candidate-{record['index']:02d}").resolve()
            if root != Path(record['directory']).resolve() or sha256((root / 'manifest.json').read_bytes()).hexdigest() != record['export_manifest_sha256']:
                raise ValueError('economic development export identity changed')
            exported = json.loads((root / 'manifest.json').read_text())['outputs']['chdd_csv']
            name = f"{prefix}-{record['index']:02d}"
            if name in paths:
                raise ValueError('duplicate economic development scenario')
            paths[name] = ((root / exported['name']).resolve(), exported['sha256'], root)
    result = []
    for trajectory in trajectories:
        path, expected_hash, root = paths[trajectory.scenario_id]
        if not path.is_relative_to(root.resolve()):
            raise ValueError('economic target CSV escapes verified batch')
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != expected_hash:
            raise ValueError('economic target CSV hash mismatch')
        states = economic_targets(csv.DictReader(raw.decode('utf-8-sig').splitlines()),
                                  trajectory.dates.strftime('%Y-%m-%d'), trajectory.well_ids)
        result.append(SimpleNamespace(scenario_id=trajectory.scenario_id, dates=trajectory.dates,
            well_ids=trajectory.well_ids, actions=trajectory.actions, states=states,
            content_hash=expected_hash))
    baseline = next(t for t in result if t.scenario_id == 'baseline')
    for trajectory in result:
        np.testing.assert_allclose(trajectory.states[:origin + 1], baseline.states[:origin + 1], rtol=0, atol=1e-6)
    return result


def observed_economic_history(canonical, manifest, trajectory, origin):
    """Build inference inputs with all post-origin economic observations replaced by NaN."""
    from types import SimpleNamespace

    raw = (canonical / 'chdd.csv').read_bytes()
    if sha256(raw).hexdigest() != manifest['outputs']['chdd_csv']['sha256']:
        raise ValueError('observed economic history hash mismatch')
    cutoff = trajectory.dates[origin].date().isoformat()
    history = normalize_chdd_rows(row for row in csv.DictReader(raw.decode('utf-8-sig').splitlines())
                                   if row['DATA'] <= cutoff)
    states = np.full((*trajectory.states.shape[:2], len(ECONOMIC_TARGETS)), np.nan)
    states[:origin + 1] = economic_targets(history, trajectory.dates[:origin + 1].strftime('%Y-%m-%d'),
                                          trajectory.well_ids)
    return history, SimpleNamespace(states=states, actions=trajectory.actions, dates=trajectory.dates,
                                    well_ids=trajectory.well_ids)


def project_economic_forecast(prediction, actions):
    """Monthly control roles mask inactive outputs; actual injection remains a model output."""
    values = np.asarray(prediction, dtype=float).copy()
    if values.shape != (*actions.shape[:2], len(ECONOMIC_TARGETS)) or not np.isfinite(values).all():
        raise ValueError('economic forecast must cover the complete control grid')
    values = np.maximum(values, 0)
    producing = (actions[..., 1] != 2) & (actions[..., 2] == 1)
    injecting = (actions[..., 1] == 2) & (actions[..., 2] == 1)
    values[..., [0, 1, 6, 7]] *= producing[..., None]
    values[..., [2, 8]] *= injecting[..., None]
    values[..., 5] = values[..., 5].clip(0, 1)
    values[..., 6] = np.minimum(values[..., 6], values[..., 7])
    return values


def forecast_economic(forecaster, trajectory, origin, horizon, context, connectivity):
    from timesfm_geology import geological_inputs

    if forecaster.model.output_head.target_count != len(ECONOMIC_TARGETS):
        raise ValueError('nine-target economic checkpoint required')
    targets, covariates = geological_inputs(trajectory, origin, context, horizon, connectivity)
    prediction, = forecaster.predict_batch([targets], horizon=horizon,
        past_future_covariates=[covariates], use_symmetric_averaging=False,
        make_positive=True, return_quantiles=False)
    values = prediction.forecast.reshape(len(trajectory.well_ids), len(ECONOMIC_TARGETS), horizon).transpose(2, 0, 1)
    return project_economic_forecast(values, trajectory.actions[origin:origin + horizon])


def economic_metrics(truth, prediction):
    error = np.abs(truth - prediction)
    return {field: {'mae': float(error[..., i].mean()),
                   'wape': float(error[..., i].sum() / np.abs(truth[..., i]).sum())
                       if np.abs(truth[..., i]).sum() > 0 else None}
            for i, field in enumerate(ECONOMIC_TARGETS)}


_FORECAST_ENDPOINT_LIMITS = {'max_liquid_m3d', 'min_injection_m3d', 'max_injection_m3d',
                             'min_bhp_bar', 'max_bhp_bar'}
# Reachable only once derived_field_vectors has supplied the monthly volume increments.
_FORECAST_DERIVED_LIMITS = {'max_monthly_liquid_m3d', 'max_monthly_injection_m3d',
                            'max_monthly_water_deficit_m3',
                            'min_monthly_voidage_replacement', 'max_monthly_voidage_replacement',
                            'min_window_voidage_replacement', 'max_window_voidage_replacement',
                            'min_window_water_replacement', 'max_window_water_replacement'}
_DERIVED_VECTORS = ('WLPT', 'WWPT', 'WWIT', 'WVPT', 'WVIT')


def export_densities(manifest, well_ids):
    """Per-well surface densities from the canonical export manifest; never a field default."""
    table = (manifest.get('conversion') or {}).get('density_by_well') or {}
    result, missing = {}, []
    for well in well_ids:
        row = table.get(well)
        oil = row.get('oil_kg_m3') if isinstance(row, dict) else None
        water = row.get('water_kg_m3') if isinstance(row, dict) else None
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) or v <= 0 for v in (oil, water)):
            missing.append(well)
            continue
        result[well] = {'oil_kg_m3': float(oil), 'water_kg_m3': float(water)}
    if missing:
        raise ValueError('canonical export manifest misses positive per-well densities for: '
                         + ', '.join(sorted(missing)))
    return result


def _positive(value, name, *, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not 0 < value < high):
        raise ValueError(f'{name} must be a finite positive number below {high}')
    return float(value)


def derived_field_vectors(prediction, timestamps, well_ids, densities, *,
                          oil_fvf=1.0, water_fvf=1.0, vrr_window=3):
    """Monthly surface and reservoir volumes implied by the nine forecast targets.

    WOMT_Diff and WLPT_Diff are masses; they become volumes only through the
    canonical export's per-well density map. Bo and Bw are explicit arguments:
    no formation volume factor is ever assumed silently in the field aggregates.
    """
    stamps, wells = _monthly_grid(timestamps, well_ids)
    values = np.asarray(prediction, dtype=np.float64)
    _validate_targets(values, len(stamps), len(wells))
    oil_fvf = _positive(oil_fvf, 'oil_fvf', high=10)
    water_fvf = _positive(water_fvf, 'water_fvf', high=10)
    if isinstance(vrr_window, bool) or not isinstance(vrr_window, int) or not 1 <= vrr_window <= 12:
        raise ValueError('vrr_window must be an integer number of months in [1, 12]')
    missing = [well for well in wells if well not in densities]
    if missing:
        raise ValueError('derived field vectors require an explicit density for: ' + ', '.join(missing))
    oil_density = np.array([_positive(densities[w]['oil_kg_m3'], f'density {w} oil', high=2000)
                            for w in wells]) / 1000.0
    water_density = np.array([_positive(densities[w]['water_kg_m3'], f'density {w} water', high=2000)
                              for w in wells]) / 1000.0
    months = [(stamp - timedelta(days=1)).replace(day=1) for stamp in stamps]
    days = np.array([calendar.monthrange(m.year, m.month)[1] for m in months], dtype=np.float64)
    oil_mass = values[..., ECONOMIC_TARGETS.index('WOMT_Diff')]
    liquid_mass = values[..., ECONOMIC_TARGETS.index('WLPT_Diff')]
    water_mass = np.maximum(liquid_mass - oil_mass, 0.0)
    oil = oil_mass / oil_density
    water = water_mass / water_density
    injection = values[..., ECONOMIC_TARGETS.index('WWIT_Diff')]
    derived = {'months': [month.isoformat() for month in months], 'well_ids': list(wells),
               'WOPT_DELTA': oil, 'WWPT_DELTA': water, 'WLPT_DELTA': oil + water,
               'WWIT_DELTA': injection,
               'WVPT_DELTA': oil * oil_fvf + water * water_fvf, 'WVIT_DELTA': injection * water_fvf,
               'formation_volume_factors': {'oil': oil_fvf, 'water': water_fvf},
               'vrr_window_months': vrr_window,
               'density_units': 'kg/m3 surface density from the canonical export manifest'}
    produced = derived['WVPT_DELTA'].sum(axis=1)
    injected = derived['WVIT_DELTA'].sum(axis=1)
    window = [(max(0, index - vrr_window + 1), index + 1) for index in range(len(months))]
    derived['field'] = {
        'months': derived['months'],
        'liquid_m3d': (derived['WLPT_DELTA'].sum(axis=1) / days).tolist(),
        'injection_m3d': (injection.sum(axis=1) / days).tolist(),
        'water_m3d': (water.sum(axis=1) / days).tolist(),
        'reservoir_production_m3': produced.tolist(),
        'reservoir_injection_m3': injected.tolist(),
        'vrr3': [float(injected[lo:hi].sum() / produced[lo:hi].sum())
                 if produced[lo:hi].sum() > 1e-12 else None for lo, hi in window],
        'window_months': vrr_window}
    return derived


def validate_economic_constraints(rules, *, derived=False):
    supported = _FORECAST_ENDPOINT_LIMITS | (_FORECAST_DERIVED_LIMITS if derived else set())
    unsupported = {key for rule in rules for key, _ in rule.limits} - supported
    if unsupported:
        raise ValueError('economic forecasts lack required vectors for: ' + ', '.join(sorted(unsupported)))


def economic_constraint_verdicts(predictions, timestamps, well_ids, rules, *, derived=None):
    """Structured verdicts for K1, K2, K4 and K5 on the forecast, aligned to control months."""
    from timesoil.aios.operating_constraints import evaluate_observed, window_months

    validate_economic_constraints(rules, derived=derived is not None)
    stamps, wells = _monthly_grid(timestamps, well_ids)
    values = np.asarray(predictions, dtype=float)
    _validate_targets(values, len(stamps), len(wells))
    window = window_months(rules) or 1
    if derived is not None:
        months = [(stamp - timedelta(days=1)).replace(day=1).isoformat() for stamp in stamps]
        if list(derived['months']) != months or list(derived['well_ids']) != list(wells):
            raise ValueError('derived field vectors do not match the forecast well-month grid')
    verdicts = []
    for index, (stamp, forecast) in enumerate(zip(stamps, values, strict=True)):
        month = (stamp - timedelta(days=1)).replace(day=1)
        rows = {well: dict(WOMR=float(row[0]), WLPR=float(row[1]),
                          WWIR=float(row[2]), WBHP=float(row[4]))
                for well, row in zip(wells, forecast, strict=True)}
        if derived is not None:
            low = max(0, index - window + 1)
            for column, well in enumerate(wells):
                for vector in _DERIVED_VECTORS:
                    series = derived[f'{vector}_DELTA']
                    rows[well][f'{vector}_DELTA'] = float(series[index, column])
                    rows[well][f'{vector}_WINDOW'] = float(series[low:index + 1, column].sum())
        verdicts.extend(evaluate_observed(rules, month, rows))
    return tuple(verdicts)


def economic_constraint_violations(predictions, timestamps, well_ids, rules, *, derived=None):
    """Messages of the violated hard rules; diagnostic rules are reported, never fatal."""
    from timesoil.aios.operating_constraints import failures

    return [verdict.message for verdict in
            failures(economic_constraint_verdicts(predictions, timestamps, well_ids, rules, derived=derived))]


def verify_roundtrip(canonical, manifest_sha256, output, start, end):
    """Prove the target representation against authenticated physical data, not forecast skill."""
    raw_manifest = (canonical / 'manifest.json').read_bytes()
    raw_csv = (canonical / 'chdd.csv').read_bytes()
    manifest = json.loads(raw_manifest)
    if (sha256(raw_manifest).hexdigest() != manifest_sha256
            or sha256(raw_csv).hexdigest() != manifest['outputs']['chdd_csv']['sha256']):
        raise ValueError('canonical economics input hash mismatch')
    rows = normalize_chdd_rows(csv.DictReader(raw_csv.decode('utf-8-sig').splitlines()))
    if not start < end or start.day != 1 or end.day != 1:
        raise ValueError('ordered monthly management boundaries required')
    history = [row for row in rows if str(row['DATA']) <= start.isoformat()]
    future = [row for row in rows if start.isoformat() < str(row['DATA']) <= end.isoformat()]
    timestamps = sorted({str(row['DATA']) for row in future})
    wells = sorted({str(row['well']) for row in rows})
    if not timestamps or timestamps[-1] != end.isoformat():
        raise ValueError('archive does not cover the complete requested forecast period')
    targets = economic_targets(future, timestamps, wells)
    rebuilt = forecast_chdd_rows(history, timestamps, wells, targets)
    original = [row for row in rows if str(row['DATA']) <= end.isoformat()]
    if [(r['DATA'], r['well']) for r in rebuilt] != [(r['DATA'], r['well']) for r in original]:
        raise AssertionError('roundtrip changed date/well alignment')
    errors = {field: max(abs(float(a[field]) - float(b[field])) for a, b in zip(original, rebuilt))
              for field in CHDD_FIELDS[2:]}
    np.testing.assert_allclose([[row[field] for field in CHDD_FIELDS[2:]] for row in rebuilt],
                               [[row[field] for field in CHDD_FIELDS[2:]] for row in original],
                               rtol=1e-12, atol=1e-8)
    output.mkdir(parents=True, exist_ok=False)
    calculator = CHDDEconomicsAdapter()
    results = [calculator.calculate(opm_management_rows(data, (start, end)),
                    start_year=start.year, output_dir=output / name,
                    charge_initial_pump=False, management_period=(start, end))
               for name, data in [('original', original), ('reconstructed', rebuilt)]]
    np.testing.assert_allclose(results[0].total_chdd_m, results[1].total_chdd_m, rtol=1e-12, atol=1e-8)
    report = {'schema': 'timesoil.economic-target-roundtrip/v1',
        'export_manifest_sha256': manifest_sha256, 'csv_sha256': sha256(raw_csv).hexdigest(),
        'targets': list(ECONOMIC_TARGETS), 'units': list(ECONOMIC_UNITS),
        'months': len(timestamps), 'wells': len(wells), 'start': start.isoformat(), 'end_exclusive': end.isoformat(),
        'max_absolute_error_by_field': errors,
        'original_chdd_m': results[0].total_chdd_m, 'reconstructed_chdd_m': results[1].total_chdd_m,
        'economics_manifests_sha256': [sha256(result.manifest_path.read_bytes()).hexdigest() for result in results],
        'new_opm_runs': 0, 'forecast_accuracy_tested': False,
        'scope': 'Lossless economic target representation; physical truth used only for this roundtrip check.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=date.fromisoformat, required=True)
    parser.add_argument('--end-exclusive', type=date.fromisoformat, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_roundtrip(args.canonical, args.manifest_sha256,
          args.output, args.start, args.end_exclusive)), flush=True)


if __name__ == '__main__':
    main()
