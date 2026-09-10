"""Verify this delivery offline using only Python's standard library."""
from hashlib import sha256
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['schema'] == 'timesoil.verified-delivery/v1'
for name, record in manifest['files'].items():
    path = root / name
    assert not path.is_symlink() and path.resolve().is_relative_to(root)
    assert path.stat().st_size == record['bytes']
    assert sha256(path.read_bytes()).hexdigest() == record['sha256'], name
for name, record in manifest['cases'].items():
    result = json.loads((root / name / 'official-result.json').read_text())
    rows = result['fieldMonthly']
    start = int(result['startDate'][:4]) * 12 + int(result['startDate'][5:7]) - 1
    expected = [f'{(start+i)//12:04d}-{(start+i)%12+1:02d}' for i in range(record['months'])]
    assert [row['month'] for row in rows] == expected, name
    for actual in [sum(r['chddM'] for r in rows), rows[-1]['cumulativeChddM'], result['summary']['totalChddM']]:
        assert math.isclose(actual, record['official_chdd_m'], rel_tol=1e-10, abs_tol=1e-7), name
for model in ['model_y', 'model_z']:
    base, selected = [manifest['cases'][model + '/' + label] for label in ['baseline', 'selected']]
    assert base['assumptions'] == selected['assumptions']
    assert base['months'] == selected['months']
    uplift = 100 * (selected['official_chdd_m'] / base['official_chdd_m'] - 1)
    assert math.isclose(uplift, selected['uplift_percent']) and uplift >= 15
    print(f'{model}: {selected["months"]} months, CHDD uplift {uplift:.6f}%')
print(f'All {len(manifest["files"])} artifact hashes and official monthly totals verified.')
