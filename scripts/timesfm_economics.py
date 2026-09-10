"""Lossless economic forecast targets; endpoint rates never stand in for monthly volumes."""

import argparse
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
