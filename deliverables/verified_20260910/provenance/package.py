"""Package existing verified calculations; never run or modify reservoir models."""
from pathlib import Path
from hashlib import sha256
import json
import math
import shutil
import sys

R = Path('/root/projects/TimesOil/results/audit-20260909')
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=False)
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
audit = read(R / 'physical-only-monthly-y-20260910/full-audit.json')
selection = read('/root/projects/TimesOil/results/audit-20260909/timesfm-early-identity-z-policy-20260910/selection.json')
assert selection['agent_review_approved'] and audit['approved_monthly_reviews'] == 23
assert audit['surrogate_used'] is False
manifest = {'schema': 'timesoil.verified-delivery/v1', 'files': {}, 'cases': {},
            'scope': 'Provided training archives; physical results, not universal forecast certification.'}

def copy(source, destination, expected=None):
    actual = digest(source)
    if expected is not None:
        assert actual == expected, (str(source), 'hash mismatch')
    target = OUT / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert digest(target) == actual
    manifest['files'][destination] = {'sha256': actual, 'source': str(source), 'bytes': target.stat().st_size}

def economics(name, root, folder, result_hash, report_hash, expected_chdd):
    result = root / folder / 'result.json'
    copy(result, name + '/official-result.json', result_hash)
    copy(root / folder / 'report.xlsx', name + '/official-report.xlsx', report_hash)
    data = read(result)
    rows = data['fieldMonthly']
    assert len({r['month'] for r in rows}) == len(rows)
    assert math.isclose(data['summary']['totalChddM'], expected_chdd, abs_tol=1e-7)
    assert math.isclose(sum(row['chddM'] for row in rows), expected_chdd, abs_tol=1e-7)
    assert math.isclose(rows[-1]['cumulativeChddM'], expected_chdd, abs_tol=1e-7)
    nominal = ['oilOpexM', 'liquidOpexM', 'injectionOpexM', 'fundOpexM', 'pumpOperationM',
               'startStopCostM', 'conversionOpexM', 'propertyTaxM', 'deductionsM',
               'profitTaxM', 'depreciationM', 'pumpCapexM']
    manifest['cases'][name] = {'months': len(rows), 'start': data['startDate'], 'last_month': data['maxDate'],
        'official_chdd_m': expected_chdd, 'assumptions': data['assumptions'], 'summary': data['summary'],
        'nominal_totals_m': {key: sum(row[key] for row in rows) for key in nominal},
        'nominal_totals_are_not_discounted_contributions': True}

for label, root, expected in [('model_y/baseline', Path(audit['baseline_run']), audit['baseline_chdd_m']),
                              ('model_y/selected', Path(audit['final_run']), audit['candidate_chdd_m'])]:
    lineage = read(root / 'lineage.json')
    records = {row['path']: row['sha256'] for row in lineage['artifacts']}
    folder = 'planning-economics' if label.endswith('/baseline') else 'economics'
    economics(label, root, folder, records[folder + '/result.json'], records[folder + '/report.xlsx'], expected)
    if label.endswith('/selected'):
        opm = read(root / 'manifest.json')
        schedule = next(a for a in opm['artifacts'] if a['sha256'] == lineage['schedule_overlay']['output_sha256'])
        copy(root / schedule['path'], label + '/full-opm-schedule.inc', schedule['sha256'])
        copy(root / 'canonical/chdd.csv', label + '/physical-chdd-input.csv', records['canonical/chdd.csv'])
        copy(R / 'physical-only-monthly-y-20260910/full-audit.json', 'model_y/full-audit.json')

z_new = next(pair['candidate'] for pair in selection['comparisons'] if pair['candidate']['run'] == str(R / 'timesfm-early-identity-z-policy-20260910/cycles/candidate'))
for label, record in [('model_z/baseline', selection['baseline']), ('model_z/selected', selection['selected']),
                       ('model_z/timesfm_candidate', z_new)]:
    root = Path(record['run'])
    assert digest(root / 'full-cycle-receipt.json') == record['receipt_sha256']
    receipt = read(root / 'full-cycle-receipt.json')
    artifacts = receipt['artifacts']
    folder = str(Path(artifacts['economics_result']['path']).parent)
    economics(label, root, folder, artifacts['economics_result']['sha256'], artifacts['economics_report']['sha256'], record['chdd_m'])
    if label.endswith(('/selected', '/timesfm_candidate')):
        for key, name in [('exact_opm_input_schedule', 'full-opm-schedule.inc'), ('canonical_chdd_csv', 'physical-chdd-input.csv')]:
            a = artifacts[key]
            copy(root / a['path'], label + '/' + name, a['sha256'])

for model in ['model_y', 'model_z']:
    base, selected = (manifest['cases'][model + '/' + kind] for kind in ['baseline', 'selected'])
    assert base['assumptions'] == selected['assumptions']
    assert base['months'] == selected['months']
    selected['uplift_percent'] = 100 * (selected['official_chdd_m'] / base['official_chdd_m'] - 1)
    assert selected['uplift_percent'] >= 15
base, candidate = (manifest['cases']['model_z/' + kind] for kind in ['baseline', 'timesfm_candidate'])
assert base['assumptions'] == candidate['assumptions'] and base['months'] == candidate['months']
candidate['uplift_percent'] = 100 * (candidate['official_chdd_m'] / base['official_chdd_m'] - 1)
assert candidate['uplift_percent'] >= 15
copy(Path(__file__), 'provenance/package.py')
copy(Path('/root/projects/TimesOil/results/audit-20260909/timesfm-early-identity-z-policy-20260910/selection.json'), 'model_z/full-comparison.json')
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: {'chdd_m': v['official_chdd_m'], 'months': v['months']} for k, v in manifest['cases'].items()}, indent=2))
