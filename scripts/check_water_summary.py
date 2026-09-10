"""Replay a sealed OPM baseline with reservoir volume outputs, preserving its physics."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

from timesoil.aios.opm import OPM_IMAGE, OpmFlowRunner, build_summary_overlay, verify_summary_extraction
from timesoil.aios.opm_chdd import REQUIRED_VECTORS, _deck_text, _eclipse_date, _read_summary, _single_record


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'prior', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--normalize-model-y', action='store_true')
    parser.add_argument('--cpu-affinity', default='30-45')
    args = parser.parse_args()
    prior = json.loads((args.prior / 'manifest.json').read_text())
    assert prior['status'] == 'success' and prior['image_reference'] == OPM_IMAGE
    assert prior['source_sha256'] == digest(args.source)
    runner = OpmFlowRunner(timeout_seconds=7200, mpi_processes=16, threads_per_process=1, cpu_affinity=args.cpu_affinity)
    options = {'normalize_model_y': True} if args.normalize_model_y else {}
    prepared = runner.prepare(args.source, args.output, deck=prior['deck'], **options)
    inputs = {a['path'][6:]: a for a in prior['artifacts'] if a['path'].startswith('input/')}
    assert set(inputs) == {p.relative_to(prepared.input_dir).as_posix() for p in prepared.input_dir.rglob('*') if p.is_file()}
    for rel, item in inputs.items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        path = args.prior / 'input' / rel
        assert not path.is_symlink() and digest(path) == item['sha256'] and path.stat().st_size == item['bytes']
        shutil.copyfile(path, prepared.input_dir / rel)
    prepared.summary_overlay_path.write_text(build_summary_overlay(prepared.connection_wells))
    changed = [rel for rel, item in inputs.items() if digest(prepared.input_dir / rel) != item['sha256']]
    assert changed == [prepared.summary_overlay_path.relative_to(prepared.input_dir).as_posix()]
    result = runner._run_prepared(prepared, parsing_strictness='low')
    report, extraction = runner.extract_summary_report(result, args.output / 'summary-report.txt')
    verify_summary_extraction(report, extraction, result.manifest_path)
    verify_summary_extraction(args.prior / 'summary-report.txt', args.prior / 'summary-extraction.json', args.prior / 'manifest.json')
    text, _ = _deck_text(result.deck_path.parent)
    start = _eclipse_date(_single_record(text, 'START'), 'START')
    observed, _ = _read_summary(report, start_date=start)
    original, _ = _read_summary(args.prior / 'summary-report.txt', start_date=start)
    assert len(observed) == len(original)
    max_difference = 0.0
    for (stamp, wells, connections), (old_stamp, old_wells, old_connections) in zip(observed, original, strict=True):
        assert stamp == old_stamp and wells.keys() == old_wells.keys()
        assert connections == old_connections, 'connection outputs changed'
        for well, values in wells.items():
            assert 'WVPT' in values and 'WVIT' in values
            for vector in REQUIRED_VECTORS:
                max_difference = max(max_difference, abs(values[vector] - old_wells[well][vector]))
    assert max_difference <= 1e-6, f'existing summary changed: {max_difference}'
    monthly = []
    for (stamp, before, _), (_, after, _) in zip(observed, observed[1:]):
        totals = {v: sum(after[w][v] - before[w][v] for w in before) for v in ('WVPT', 'WVIT', 'WWPT', 'WWIT')}
        assert all(v >= -1e-6 for v in totals.values())
        monthly.append(dict(month=str(stamp), **totals,
                            water_deficit_m3=max(0, totals['WWIT'] - totals['WWPT']),
                            voidage_replacement=totals['WVIT'] / totals['WVPT'] if totals['WVPT'] > 0 else None))
    table = args.output / 'monthly-field-water.csv'
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(monthly[0]))
        writer.writeheader()
        writer.writerows(monthly)
    proof = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 script_sha256=digest(Path(__file__)), source_sha256=digest(args.source),
                 prior_manifest_sha256=digest(args.prior / 'manifest.json'),
                 run_manifest_sha256=digest(result.manifest_path), changed_inputs=changed,
                 report_dates=len(observed), wells=len(observed[0][1]),
                 required_vector_max_abs_difference=max_difference, all_connections_unchanged=True,
                 optional_vectors=['WVPT', 'WVIT'], unit_system=prepared.unit_system,
                 monthly_field_water_sha256=digest(table), complete=True)
    (args.output / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
