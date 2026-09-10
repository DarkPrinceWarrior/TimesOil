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
from timesoil.aios.opm import OPM_IMAGE, OpmFlowRunner, verify_summary_extraction
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


def compare_economics_settings(current, previous):
    """Results may change; calendar, calculator and all assumptions may not."""
    for key in ('calculator_sha256', 'norms_sha256', 'norms_source_sha256',
                'start_year', 'assumption_overrides'):
        assert current[key] == previous[key], f'economics drift: {key}'
    results = {'total_chdd_m', 'profitability_index'}
    settings = lambda item: {k: v for k, v in item['management_period'].items() if k not in results}
    assert settings(current) == settings(previous), 'economics drift: management_period'


def resume_baseline(run, prior, record, runner):
    """Authenticate the completed physical run before continuing the interrupted pair."""
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'success' and manifest['returncode'] == 0
    assert manifest['source_sha256'] == MODEL_Z_SOURCE_SHA256
    assert manifest['image_reference'] == OPM_IMAGE
    assert record['experimental_engine'] == runner.get_provenance()
    assert record['prior_run'] == str(prior) and record['run'] == str(run)
    assert record['prior_manifest_sha256'] == digest(prior / 'manifest.json')
    for item in manifest['artifacts']:
        rel = Path(item['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        path = run / rel
        assert path.is_file() and not path.is_symlink()
        assert digest(path) == item['sha256'] and path.stat().st_size == item['bytes']
    original = json.loads((prior / 'manifest.json').read_text())
    before = {a['path']: a['sha256'] for a in original['artifacts'] if a['path'].startswith('input/')}
    after = {a['path']: a['sha256'] for a in manifest['artifacts'] if a['path'].startswith('input/')}
    assert set(before) == set(after)
    assert [p for p in before if before[p] != after[p]] == ['input/' + GRID]
    assert after['input/' + GRID] == sha256(convert_grid((prior / 'input' / GRID).read_bytes())).hexdigest()
    assert record['transformed_grid_sha256'] == after['input/' + GRID]
    verify_summary_extraction(run / 'summary-report.txt', run / 'summary-extraction.json', manifest_path)
    canonical = json.loads((run / 'canonical/manifest.json').read_text())
    provenance = canonical['provenance']
    assert provenance['opm_run_manifest_sha256'] == digest(manifest_path)
    assert provenance['summary_extraction_manifest_sha256'] == digest(run / 'summary-extraction.json')
    for key, name in [('chdd_csv', 'chdd.csv'), ('track2_csv', 'trajectory.csv')]:
        assert canonical['outputs'][key]['name'] == name
        assert canonical['outputs'][key]['sha256'] == digest(run / 'canonical' / name)
    current = json.loads((run / 'economics-2007/manifest.json').read_text())
    for field, artifact in [('input_sha256', 'input'), ('result_sha256', 'result'), ('norms_sha256', 'effective_norms')]:
        name = current['artifacts'][artifact]
        assert Path(name).name == name
        assert current[field] == digest(run / 'economics-2007' / name)
    compare_economics_settings(current, json.loads((prior / 'economics-2007/manifest.json').read_text()))
    record.update(complete=True, official_chdd_m=current['management_period']['total_chdd_m'],
                  resumed_without_physical_rerun=True,
                  opm_manifest_sha256=digest(manifest_path),
                  economics_manifest_sha256=digest(run / 'economics-2007/manifest.json'))


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
    parser.add_argument('--resume-baseline', action='store_true', help='Resume only a completed baseline with no candidate yet')
    args = parser.parse_args()
    self_check(args.source)
    if args.self_check:
        return
    if not all((args.baseline_run, args.candidate_run, args.output)):
        parser.error('baseline-run, candidate-run and output required')
    economics = CHDDEconomicsAdapter.from_env()
    economics.normative_profile()
    args.output.mkdir(parents=True, exist_ok=args.resume_baseline)
    protocol = dict(schema='timesoil.model-z-pinch-sensitivity/v1', complete=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=digest(Path(__file__)), source_sha256=digest(args.source),
        hypothesis='Absent PINCHNUM defaults to region 1; replace its 2m criteria with global PINCH.',
        tnavigator_equivalence_proven=False, scenarios=[])
    target = args.output / 'protocol.json'
    if args.resume_baseline:
        saved = json.loads(target.read_text())
        assert saved['schema'] == protocol['schema'] and not saved['complete']
        assert saved['source_sha256'] == digest(args.source)
        assert len(saved['scenarios']) == 1 and saved['scenarios'][0]['label'] == 'baseline'
        assert not (args.output / 'candidate').exists()
        backup = args.output / 'protocol-before-resume.json'
        with backup.open('x') as stream:
            stream.write(target.read_text())
        saved['resume'] = {k: protocol[k] for k in ('source_commit', 'script_sha256')}
        protocol = saved
    else:
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
        if args.resume_baseline and label == 'baseline':
            resume_baseline(args.output / label, prior, protocol['scenarios'][0], runner)
            target.write_text(json.dumps(protocol, indent=2) + '\n')
            print(json.dumps(protocol['scenarios'][0]), flush=True)
            continue
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
        compare_economics_settings(current, previous)
        record.update(complete=True, official_chdd_m=value.total_chdd_m, seconds=time.monotonic()-started)
        target.write_text(json.dumps(protocol, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    baseline, candidate = [s['official_chdd_m'] for s in protocol['scenarios']]
    protocol.update(complete=True, uplift_percent=100*(candidate-baseline)/baseline)
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    (args.output / 'exit').write_text('0\n')


if __name__ == '__main__':
    main()
