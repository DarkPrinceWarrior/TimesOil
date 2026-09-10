"""Measure PINCHREG -> PINCH 2m sensitivity on sealed Model Z input snapshots.

This is a physics compatibility experiment, not tNavigator certification.
No production source archive, schedule, checkpoint or historical receipt is edited.
"""

import argparse
import csv
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from timesoil.aios.economics import CHDDEconomicsAdapter, opm_management_rows
from timesoil.aios.opm import OpmFlowRunner
from timesoil.aios.opm_chdd import export_opm_chdd
from timesoil.aios.track2 import MODEL_Z_SOURCE_SHA256

GRID = 'Model_Z/Model_Z_grid.inc'
GRID_SHA = 'c2bc700bd6cba9ea6dceba3349618a1d7098cd91abc96bed0d57d84c6d3fb708'
START, END = date(2007, 1, 1), date(2025, 9, 1)
BLOCK = re.compile(rb'(?m)^PINCHREG\r?\n-- thickness controlling_opt max.empty_gap calc.opt account.opt\r?\n 2 /\r?\n 2 /\r?\n/\r?\n?\Z')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def convert_grid(raw):
    """Accept only this archived uniform region block, retaining every other byte."""
    if sha256(raw).hexdigest() != GRID_SHA:
        raise ValueError('unexpected Model Z grid bytes')
    matches = list(BLOCK.finditer(raw))
    if len(matches) != 1:
        raise ValueError('expected one terminal PINCHREG block')
    match = matches[0]
    newline = b'\r\n' if b'\r\n' in match[0] else b'\n'
    return raw[:match.start()] + newline.join([b'PINCH', b' 2 GAP 1.0E20 TOPBOT TOP /', b''])


def self_check(source):
    from zipfile import ZipFile
    assert digest(source) == MODEL_Z_SOURCE_SHA256
    with ZipFile(source) as archive:
        raw = archive.read(GRID)
    changed = convert_grid(raw)
    offset = BLOCK.search(raw).start()
    assert changed[:offset] == raw[:offset]
    assert b'PINCHREG' not in changed
    assert changed[offset:].splitlines() == [b'PINCH', b' 2 GAP 1.0E20 TOPBOT TOP /']
    try:
        convert_grid(raw.replace(b' 2 /', b' 3 /', 1))
    except ValueError:
        pass
    else:
        raise AssertionError('changed region settings accepted')
    print('exact source, byte preservation and changed-region rejection passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--candidate-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check(args.source)
    if args.self_check:
        return
    if not all((args.baseline_run, args.candidate_run, args.output)):
        parser.error('baseline-run, candidate-run and output required')
    economics = CHDDEconomicsAdapter.from_env()
    economics.normative_profile()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='timesoil.model-z-pinch-sensitivity/v1', complete=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=digest(Path(__file__)), source_sha256=digest(args.source),
        hypothesis='Absent PINCHNUM defaults to region 1; replace its 2m criteria with global PINCH.',
        tnavigator_equivalence_proven=False, scenarios=[])
    target = args.output / 'protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    runner = OpmFlowRunner(timeout_seconds=7200, mpi_processes=16, threads_per_process=1, cpu_affinity='14-29')
    for label, prior in [('baseline', args.baseline_run), ('candidate', args.candidate_run)]:
        started = time.monotonic()
        manifest = json.loads((prior / 'manifest.json').read_text())
        assert manifest['source_sha256'] == MODEL_Z_SOURCE_SHA256 and manifest['status'] == 'success'
        assert manifest['image_reference'] == runner.get_provenance().split('image=')[1]
        inputs = {r['path'][6:]: r for r in manifest['artifacts'] if r['path'].startswith('input/')}
        actual = {p.relative_to(prior / 'input').as_posix() for p in (prior / 'input').rglob('*') if p.is_file()}
        assert set(inputs) == actual and GRID in inputs
        for rel, item in inputs.items():
            assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
            p = prior / 'input' / rel
            assert not p.is_symlink() and digest(p) == item['sha256'] and p.stat().st_size == item['bytes']
            if p.suffix.lower() in {'.data', '.inc'}:
                clean = re.sub(rb'--[^\r\n]*', b'', p.read_bytes())
                assert not re.search(rb'\bPINCHNUM\b', clean), 'region assignment needs a separate proof'
                assert not re.search(rb'(?m)^\s*PINCH\s*$', clean), 'pre-existing global PINCH'
        prepared = runner.prepare(args.source, args.output / label, deck=manifest['deck'])
        assert {p.relative_to(prepared.input_dir).as_posix() for p in prepared.input_dir.rglob('*') if p.is_file()} == actual
        for rel in inputs:
            shutil.copyfile(prior / 'input' / rel, prepared.input_dir / rel)
        grid = prepared.input_dir / GRID
        grid.write_bytes(convert_grid(grid.read_bytes()))
        changed = [rel for rel, item in inputs.items() if digest(prepared.input_dir / rel) != item['sha256']]
        assert changed == [GRID]
        record = dict(label=label, prior_run=str(prior), prior_manifest_sha256=digest(prior / 'manifest.json'),
            changed_files=changed, original_grid_sha256=GRID_SHA, transformed_grid_sha256=digest(grid),
            input_file_count=len(inputs), run=str(prepared.run_dir), complete=False)
        protocol['scenarios'].append(record)
        target.write_text(json.dumps(protocol, indent=2) + '\n')
        result = runner._run_prepared(prepared, parsing_strictness='low')
        assert 'PINCHREG: keyword not supported' not in result.stdout_path.read_text()
        report, extraction = runner.extract_summary_report(result, prepared.run_dir / 'summary-report.txt')
        out = prepared.run_dir / 'canonical'
        export_opm_chdd(report, out / 'chdd.csv', out / 'trajectory.csv', out / 'manifest.json',
            scenario_id=f'pinch2-sensitivity-{label}', source_model='Model_Z_PINCH2_sensitivity',
            opm_run_manifest=result.manifest_path, summary_extraction_manifest=extraction,
            deck_dir=result.deck_path.parent, include_bhp=True)
        with (out / 'chdd.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        previous = json.loads((prior / 'economics-2007/manifest.json').read_text())
        charge = previous['assumption_overrides'].get('chargeInitialPump')
        value = economics.calculate(opm_management_rows(rows, (START, END)), start_year=2007,
            output_dir=prepared.run_dir / 'economics-2007', charge_initial_pump=charge,
            management_period=(START, END))
        current = json.loads(value.manifest_path.read_text())
        for key in ('calculator_sha256', 'norms_sha256', 'management_period', 'assumption_overrides'):
            assert current[key] == previous[key], f'economics drift: {key}'
        record.update(complete=True, official_chdd_m=value.total_chdd_m, seconds=time.monotonic()-started)
        target.write_text(json.dumps(protocol, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    baseline, candidate = [s['official_chdd_m'] for s in protocol['scenarios']]
    protocol.update(complete=True, uplift_percent=100*(candidate-baseline)/baseline)
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    (args.output / 'exit').write_text('0\n')


if __name__ == '__main__':
    main()
