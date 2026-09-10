"""Audit archived forecasts and original physics without feeding results into selection."""
import argparse
import ast
import csv
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from timesoil.aios.workflow import CycleRequest


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(values, controls, stamps, wells):
    assert values.shape == (len(stamps), len(wells), 9) and np.isfinite(values).all()
    grid = {(row['month'], row['well']): row for row in controls}
    assert len(grid) == len(controls) == len(stamps) * len(wells)
    result = {}
    for i, stamp in enumerate(stamps):
        endpoint = date.fromisoformat(stamp)
        assert endpoint.day == 1
        month = endpoint.replace(year=endpoint.year - (endpoint.month == 1),
                                 month=endpoint.month - 1 or 12).isoformat()
        for j, well in enumerate(wells):
            control = grid[month, well]
            if control['status'] != 'OPEN':
                continue
            liquid, injection, bhp = values[i, j, [1, 2, 4]]
            checks = {}
            if control['target'] == 'LRAT':
                checks['liquid_above_lrat_m3d'] = liquid - control['value']
            if control['target'] == 'WRAT':
                checks['injection_above_wrat_m3d'] = injection - control['value']
            if max(liquid, injection) > 1e-6 and control.get('bhp_limit', 0) > 0:
                if control['role'] == 'producer':
                    checks['producer_below_bhp_bar'] = control['bhp_limit'] - bhp
                else:
                    checks['injector_above_bhp_bar'] = bhp - control['bhp_limit']
            for key, excess in checks.items():
                entry = result.setdefault(key, dict(checked=0, count=0, maximum_excess=0., examples=[]))
                entry['checked'] += 1
                if excess <= 1e-3:
                    continue
                entry['count'] += 1
                entry['maximum_excess'] = max(entry['maximum_excess'], float(excess))
                if len(entry['examples']) < 3:
                    entry['examples'].append(dict(control_month=month, endpoint=stamp, well=well,
                        excess=float(excess), liquid_m3d=float(liquid), injection_m3d=float(injection),
                        bhp_bar=float(bhp), control=control))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('search', type=Path)
    parser.add_argument('canonical', type=Path)
    parser.add_argument('manifest_sha256')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = args.canonical / 'manifest.json'
    assert digest(manifest) == args.manifest_sha256
    data = json.loads(manifest.read_text())
    csv_path = args.canonical / 'chdd.csv'
    assert digest(csv_path) == data['outputs']['chdd_csv']['sha256']
    ledger = json.loads((args.search / 'candidates.json').read_text())
    assert isinstance(ledger, list)
    report = dict(source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        canonical_manifest_sha256=digest(manifest), canonical_csv_sha256=digest(csv_path),
        scope='Diagnostic only; no candidate selection, model fitting or new OPM.',
        tolerance=1e-3, bhp_requires_flow_above=1e-6, forecasts=[])
    for candidate in ledger:
        index = candidate['id']
        request_path = args.search / f'request-{index:02d}.json'
        request = json.loads(request_path.read_text())
        assert CycleRequest.from_mapping(request).controls_sha256 == candidate['controls_sha256']
        controls = request['controls']
        path = args.search / f'forecast-{index:02d}.npz'
        assert digest(path) == candidate['forecast_sha256']
        with np.load(path, allow_pickle=False) as archive:
            values, wells = archive['prediction'], archive['well_ids'].tolist()
            assert archive['targets'].tolist() == ['WOMR', 'WLPR', 'WWIR', 'THP', 'BHP', 'WEFF',
                                                   'WOMT_Diff', 'WLPT_Diff', 'WWIT_Diff']
            # Old authenticated archives contain object timestamps; derive dates from their sealed controls.
            months = sorted({row['month'] for row in controls})
            stamps = [(pd.Timestamp(month) + pd.offsets.MonthBegin(1)).date().isoformat() for month in months]
            try:
                stored = archive['timestamps'].tolist()
            except ValueError as error:
                assert 'Object arrays' in str(error)
                timestamp_encoding = 'legacy_object_array_not_loaded'
            else:
                assert stored == stamps
                timestamp_encoding = 'safe_unicode'
        report['forecasts'].append(dict(id=index, forecast_sha256=digest(path),
            request_sha256=digest(request_path), timestamp_encoding=timestamp_encoding,
            controls=audit(values, controls, stamps, wells)))
        if index == 0:
            original = json.loads((args.search.parent / 'request.json').read_text())
            assert CycleRequest.from_mapping(original).controls_sha256 == candidate['controls_sha256']
            with csv_path.open() as stream:
                physical = {(row['DATA'], row['well']): row for row in csv.DictReader(stream)}
            truth = np.array([[[float(physical[stamp, well][field]) for field in
                ['WOMR', 'WLPR', 'WWIR', 'THP', 'BHP', 'WEFF', 'WOMT_Diff', 'WLPT_Diff', 'WWIT_Diff']]
                for well in wells] for stamp in stamps])
            report['original_physics'] = audit(truth, controls, stamps, wells)
    # Execute the actual producer's serialization statement, including its pandas timestamp input.
    producer = Path('scripts/propose_track2_policies.py')
    statements = [node for node in ast.walk(ast.parse(producer.read_text())) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                  and node.value.func.attr == 'savez_compressed']
    assert len(statements) == 1
    sample = args.output / 'serialization-check.npz'
    namespace = dict(np=np, forecast_path=sample, prediction=values,
        timestamps=pd.DatetimeIndex(stamps).strftime('%Y-%m-%d'),
        trajectory=SimpleNamespace(well_ids=wells),
        ECONOMIC_TARGETS=['WOMR', 'WLPR', 'WWIR', 'THP', 'BHP', 'WEFF', 'WOMT_Diff', 'WLPT_Diff', 'WWIT_Diff'])
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(producer), 'exec'), namespace)
    with np.load(sample, allow_pickle=False) as archive:
        for key in archive.files:
            assert not archive[key].dtype.hasobject
        assert archive['timestamps'].tolist() == stamps
        np.testing.assert_array_equal(archive['prediction'], values)
    report['serialization_check'] = dict(passed=True, producer_sha256=digest(producer),
        archive_sha256=digest(sample), predictions_unchanged=True, allow_pickle=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(forecasts=len(ledger), original_physics=report['original_physics'],
                         serialization_check=report['serialization_check'])), flush=True)


if __name__ == '__main__':
    main()
