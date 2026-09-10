"""Audit completed sensitivity pair; run on A100 with its artifact directory."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def main(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    assert protocol['complete'] and (root / 'exit').read_text().strip() == '0'
    scenarios, histories, settings, receipts = [], [], [], []
    for record in protocol['scenarios']:
        run = root / record['label']
        raw = json.loads((run / 'manifest.json').read_text())
        assert raw['status'] == 'success'
        canonical = json.loads((run / 'canonical/manifest.json').read_text())
        assert canonical['provenance']['opm_run_manifest_sha256'] == digest(run / 'manifest.json')
        for name in canonical['outputs'].values():
            assert Path(name['name']).name == name['name']
            assert digest(run / 'canonical' / name['name']) == name['sha256']
        econ = json.loads((run / 'economics-2007/manifest.json').read_text())
        for artifact, key in [('input', 'input_sha256'), ('result', 'result_sha256'), ('effective_norms', 'norms_sha256')]:
            assert digest(run / 'economics-2007' / econ['artifacts'][artifact]) == econ[key]
        settings.append({k: econ[k] for k in ('calculator_sha256', 'norms_sha256', 'start_year', 'assumption_overrides')})
        settings[-1]['period'] = {k: v for k, v in econ['management_period'].items() if k not in ('total_chdd_m', 'profitability_index')}
        with (run / 'canonical/trajectory.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        scope = {(r['date'], r['well']) for r in rows}
        assert len(scope) == len(rows) == 38213
        assert len({r['date'] for r in rows}) == 371 and len({r['well'] for r in rows}) == 103
        histories.append(sorted((r['date'], r['well'], r['oil_tpd'], r['liquid_tpd'], r['pressure_bar'])
                                for r in rows if r['date'] <= '2007-01-01'))
        assert record['official_chdd_m'] == econ['management_period']['total_chdd_m']
        scenarios.append(dict(label=record['label'], official_chdd_m=record['official_chdd_m'],
                              management_summary=econ['summary'], report_dates=371, wells=103))
        for rel in ('manifest.json', 'canonical/manifest.json', 'economics-2007/manifest.json'):
            receipts.append(dict(path=f"{record['label']}/{rel}", sha256=digest(run / rel)))
    assert settings[0] == settings[1]
    assert histories[0] == histories[1], 'different physical history before first control'
    baseline, candidate = [s['official_chdd_m'] for s in scenarios]
    uplift = 100 * (candidate - baseline) / baseline
    assert uplift == protocol['uplift_percent']
    proof = dict(complete=True, experiment='PINCHREG-to-PINCH2 plus opm-grid935; not tNavigator certification',
                 source_commit=protocol['resume']['source_commit'], protocol_sha256=digest(root / 'protocol.json'),
                 script_sha256=digest(Path(__file__)), scenarios=scenarios,
                 settings_equal=True, physical_history_through_origin_equal=True,
                 history_rows=len(histories[0]), uplift_percent=uplift,
                 target_15_percent_met=uplift >= 15, tnavigator_equivalence_proven=False,
                 receipts=receipts)
    (root / 'pair-audit.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
