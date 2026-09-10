"""Evaluate full remaining-period Model Y policies with the existing OPM backend."""

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

from run_track1_mpc import _next_month, build_backend, load_config
from timesoil.aios.contracts import ControlTarget, WellRole, WellStatus
from timesoil.aios.schedule import ScheduleCompiler


def scale(actions, producer, injector):
    return tuple(replace(action, value=min(
        action.value * (injector if action.role is WellRole.INJECTOR else producer),
        500.0 if action.target is ControlTarget.LIQUID_RATE else float('inf'),
    )) if action.status is WellStatus.OPEN else action for action in actions)


def evidence(root):
    lineage = json.loads((root / 'lineage.json').read_text())
    for entry in lineage['artifacts']:
        path = root / entry['path']
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        assert sha256(path.read_bytes()).hexdigest() == entry['sha256']
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'success' and manifest['returncode'] == 0
    for entry in manifest['artifacts']:
        path = root / entry['path']
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        assert sha256(path.read_bytes()).hexdigest() == entry['sha256']
    rows = list(csv.DictReader((root / 'canonical/chdd.csv').open()))
    economics = json.loads((root / 'planning-economics/result.json').read_text())
    norms = json.loads((root / 'planning-economics/manifest.json').read_text())
    assert economics['summary']['totalChddM'] == lineage['planning']['total_chdd_m']
    return lineage, manifest, rows, economics, norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--forecast-validation', type=Path, help='Authenticated BHP reference export for eight independent scenarios')
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    config = replace(config, opm_runs_dir=args.output / 'opm')
    compiler = ScheduleCompiler()
    original = tuple(action for month in sorted(config.candidates) for action in config.candidates[month][0])
    assert scale(original, 1, 1) == original
    assert all(a.value <= 500 for a in scale(original, 100, 1) if a.target is ControlTarget.LIQUID_RATE)
    print('Identity and liquid-rate cap checks passed', flush=True)
    results, baseline = [], None
    designs = [(p, i, 0, 1) for p, i in [(1, 1), (2, 1), (3, 1), (4, 1), (2, .5), (2, 1.5), (3, 2), (1, .5), (1, 2)]]
    if args.forecast_validation:
        from timesoil.aios.track2 import load_trajectory_dataset
        from timesoil.aios.opm_chdd import export_opm_chdd
        reference = args.forecast_validation
        dataset = load_trajectory_dataset(reference / 'trajectory.csv', manifest=reference / 'manifest.json')
        assert len(dataset) == 1
        metadata = json.loads((reference / 'manifest.json').read_text())
        assert metadata['provenance']['opm_source_sha256'] == config.source_sha256
        with (reference / 'trajectory.csv').open() as stream:
            limits = {(r['date'], r['well']): float(r['bhp_limit']) for r in csv.DictReader(stream)}
        designs = [(1.5, .75, 5, 1), (2.5, 1.25, 10, .95), (3.25, 1.6, 20, .9),
                   (3.5, 1.75, 0, .9), (2.25, .8, 15, 1), (1.75, 1.1, 10, .85),
                   (2.75, 1.8, 10, .95), (2.8, 1.9, 20, .9)]
        evaluation = {'schema': 'timesoil.model-y-forecast-evaluation/v1', 'source_sha256': config.source_sha256,
            'calibration_cases': [0, 1, 3, 4, 6], 'test_cases': [2, 5, 7], 'designs': designs,
            'model_selection_allowed_on_test': False, 'scenarios': [], 'complete': False,
            'reference_manifest_sha256': sha256((reference / 'manifest.json').read_bytes()).hexdigest()}
        (args.output / 'protocol.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    for index, (producer, injector, producer_bhp_add, injector_bhp_factor) in enumerate(designs):
        started = time.monotonic()
        controls = compiler.validate(config.case, scale(original, producer, injector))
        if args.forecast_validation:
            pressure_controls = []
            for action in controls:
                if action.status is WellStatus.OPEN:
                    limit = limits[action.month.isoformat(), action.well]
                    assert limit > 0
                    pressure_controls.append(replace(action, bhp_limit=(limit * injector_bhp_factor
                        if action.role is WellRole.INJECTOR else limit + producer_bhp_add)))
                else:
                    pressure_controls.append(action)
            assert tuple(replace(a, bhp_limit=b.bhp_limit) for a, b in zip(pressure_controls, controls, strict=True)) == controls
            controls = compiler.validate(config.case, pressure_controls)
        (args.output / f'controls-{index:02d}.json').write_text(json.dumps([a.to_dict() for a in controls], indent=2) + '\n')
        current = tuple(a for a in controls if a.month == config.initial_state.month)
        tail = tuple(a for a in controls if a.month > config.initial_state.month)
        entry = {'index': index, 'producer_scale': producer, 'injector_scale': injector,
                 'scope': 'Full-period physical planning experiment; future states are not committed to the monthly controller.'}
        try:
            result = build_backend(config).run_from_restart(config.case, config.initial_state, current, planning_tail=tail)
            root = config.opm_runs_dir / result.trajectory.run_id
            item = evidence(root)
            if baseline is None:
                assert index == 0
                baseline = item
            for key in ('source_sha256',):
                assert baseline[0][key] == item[0][key]
            for key in ('image_reference', 'source_sha256', 'deck_sha256'):
                assert baseline[1][key] == item[1][key]
            # The backend writes the actual relative schedule path to its overlay manifest.
            schedule = 'input/' + json.loads((root / 'schedule-overlay.json').read_text())['schedule']
            inputs = [{e['path']: e['sha256'] for e in data[1]['artifacts']
                       if e['path'].startswith('input/') and e['path'] != schedule} for data in (baseline, item)]
            assert inputs[0] == inputs[1], 'non-schedule simulator input changed'
            for key in ('calculator_sha256', 'norms_source_sha256', 'norms_sha256', 'assumption_overrides', 'start_year'):
                assert baseline[4][key] == item[4][key]
            assert baseline[3]['assumptions'] == item[3]['assumptions']
            start, end = config.case.start.isoformat(), result.planning_end.isoformat()
            history = [{(r['DATA'], r['well']): r for r in data[2] if r['DATA'] <= start} for data in (baseline, item)]
            assert history[0] == history[1], 'pre-control history changed'
            expected = {(a.month.isoformat(), a.well) for a in controls}
            managed = [r for r in item[2] if start < r['DATA'] <= end]
            assert len(managed) == len(expected)
            assert {(r['DATA'], r['well']) for r in managed} == {
                (_next_month(a.month).isoformat(), a.well) for a in controls}
            maximum = max(float(r['WLPR']) for r in managed)
            assert maximum <= config.case.max_liquid_rate + 1e-6
            value, base = item[3]['summary']['totalChddM'], baseline[3]['summary']['totalChddM']
            entry.update(run=str(root), total_chdd_m=value, baseline_chdd_m=base,
                         uplift_percent=(value / base - 1) * 100, max_liquid_m3d=maximum,
                         months=len({a.month for a in controls}), wells=len(current), artifacts_verified=True)
            if args.forecast_validation:
                exported = args.output / f'candidate-{index:02d}'
                export_opm_chdd(root / 'summary-report.txt', exported / 'chdd.csv', exported / 'trajectory.csv',
                    exported / 'manifest.json', scenario_id=f'forecast-validation-{index:02d}', source_model='model_y_opm',
                    opm_run_manifest=root / 'manifest.json', summary_extraction_manifest=root / 'summary-extraction.json',
                    deck_dir=root / 'input', include_bhp=True)
                evaluation['scenarios'].append({'index': index, 'run': str(root.resolve()), 'directory': str(exported.resolve()),
                    'trajectory_sha256': sha256((exported / 'trajectory.csv').read_bytes()).hexdigest(),
                    'export_manifest_sha256': sha256((exported / 'manifest.json').read_bytes()).hexdigest()})
                (args.output / 'manifest.json').write_text(json.dumps(evaluation, indent=2) + '\n')
        except Exception as error:
            entry['error'] = f'{type(error).__name__}: {error}'
        entry['seconds'] = time.monotonic() - started
        results.append(entry)
        (args.output / 'completion.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(entry), flush=True)
        if index == 0 and 'error' in entry:
            raise RuntimeError('baseline failed verification')
    if args.forecast_validation:
        assert [r['index'] for r in evaluation['scenarios']] == list(range(8))
        evaluation['complete'] = True
        (args.output / 'manifest.json').write_text(json.dumps(evaluation, indent=2) + '\n')


if __name__ == '__main__':
    main()
