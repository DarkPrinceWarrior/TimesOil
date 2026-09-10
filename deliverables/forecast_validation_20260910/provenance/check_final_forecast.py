"""Evaluate the already executed sealed graph; never fit or select another graph."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_forecast_control_bounds import audit, digest
from timesfm_economics import economic_targets, economic_metrics
from track2_final_selection import verify_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('seal_sha256')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    seal, request, _ = verify_selection(args.run / 'search', args.seal_sha256)
    final = args.run / 'final'
    final_audit = json.loads((final / 'final-audit.json').read_text())
    assert final_audit['selection_seal_sha256'] == args.seal_sha256
    assert final_audit['selected_before_opm'] and not final_audit['reselection_after_opm']
    assert final_audit['search_opm_calls'] == 0 and final_audit['final_opm_calls'] == 1
    receipt_path = final / 'selected/full-cycle-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['request_sha256'] == request.request_sha256
    assert receipt['controls']['canonical_schedule_sha256'] == seal['controls_sha256']
    assert receipt['artifacts']['exact_opm_input_schedule']['sha256'] == seal['schedule_overlay_sha256']
    paths = {}
    for key in ['canonical_chdd_csv', 'canonical_export_manifest', 'exact_opm_input_schedule']:
        artifact = receipt['artifacts'][key]
        path = final / 'selected' / artifact['path']
        assert digest(path) == artifact['sha256']
        paths[key] = path
    manifest = json.loads(paths['canonical_export_manifest'].read_text())
    assert digest(paths['canonical_chdd_csv']) == manifest['outputs']['chdd_csv']['sha256']
    selected = seal['selected_id']
    proposed = json.loads((args.run / 'search' / seal['selected_request']).read_text())
    months = sorted({row['month'] for row in proposed['controls']})
    stamps = [(pd.Timestamp(month) + pd.offsets.MonthBegin(1)).date().isoformat() for month in months]
    with np.load(args.run / 'search' / f'forecast-{selected:02d}.npz', allow_pickle=False) as archive:
        prediction, wells = archive['prediction'], archive['well_ids'].tolist()
    with paths['canonical_chdd_csv'].open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    # economic_targets requires exactly the managed endpoints, never pre-management history.
    endpoints = set(stamps)
    truth = economic_targets([row for row in rows if row['DATA'] in endpoints], stamps, wells)
    assert truth.shape == prediction.shape == (seal['horizon_months'], len(wells), 9)
    report = dict(scope='Final selected graph evaluation only; no fitting, calibration or reselection.',
        selection_seal_sha256=args.seal_sha256, final_audit_sha256=digest(final / 'final-audit.json'),
        full_cycle_receipt_sha256=digest(receipt_path),
        canonical_csv_sha256=digest(paths['canonical_chdd_csv']),
        forecast_sha256=digest(args.run / 'search' / f'forecast-{selected:02d}.npz'),
        months=len(stamps), wells=len(wells), search_opm_calls=0, final_opm_calls=1,
        new_opm_calls_for_this_check=0, independent_coverage_certification=False,
        forecast_metrics=economic_metrics(truth, prediction),
        physical_own_control_bounds=audit(truth, proposed['controls'], stamps, wells),
        forecast_own_control_bounds=audit(prediction, proposed['controls'], stamps, wells))
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(months=len(stamps), wells=len(wells),
        forecast_metrics=report['forecast_metrics'], physical_violations={
            k: v['count'] for k, v in report['physical_own_control_bounds'].items()})), flush=True)


if __name__ == '__main__':
    main()
