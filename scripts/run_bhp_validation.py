"""Generate independently held-out full-period BHP interventions with fixed rate controls."""

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

from benchmark_bhp_surrogate import START, END, MONTHS
from timesoil.aios.contracts import WellRole, WellStatus
from timesoil.aios.economics import CHDDEconomicsAdapter, opm_management_rows
from timesoil.aios.opm import OpmFlowRunner
from timesoil.aios.opm_chdd import export_opm_chdd
from timesoil.aios.operating_constraints import check_controls, check_summary, parse_constraints
from timesoil.aios.schedule_overlay import apply_schedule_overlay
from timesoil.aios.track2 import MODEL_Z_SOURCE_SHA256, load_trajectory_dataset
from timesoil.aios.workflow import CycleRequest, _source_control_inventory, _validate_source_well_scope, _physical_control_evidence

TRANSITION_DESIGNS = [(0, .92), (30, 1), (22.5, .96), (10, .97),
                      (20, .94), (12.5, .89), (30, .88), (27.5, .93)]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def pressure_controls(controls, limits, producer_add, injector_factor):
    output = []
    for action in controls:
        updated = action
        if action.status is WellStatus.OPEN:
            original = limits[action.month, action.well]
            if not original > 0:
                raise ValueError('open well has no positive authenticated BHP limit')
            bhp = original + producer_add if action.role is WellRole.PRODUCER else original * injector_factor
            updated = replace(action, bhp_limit=bhp)
        assert replace(updated, bhp_limit=action.bhp_limit) == action
        output.append(updated)
    return tuple(output)


def self_check():
    from datetime import date
    from timesoil.aios.contracts import ControlAction, ControlTarget
    month = date(2007, 1, 1)
    controls = (ControlAction(month, 'P', WellRole.PRODUCER, WellStatus.OPEN, ControlTarget.LIQUID_RATE, 100),
        ControlAction(month, 'I', WellRole.INJECTOR, WellStatus.OPEN, ControlTarget.WATER_INJECTION_RATE, 200),
        ControlAction(month, 'F', WellRole.PRODUCER, WellStatus.SHUT, ControlTarget.LIQUID_RATE, 0))
    changed = pressure_controls(controls, {(month, 'P'): 50, (month, 'I'): 300}, 15, .9)
    assert [a.bhp_limit for a in changed] == [65, 270, None]
    assert [a.value for a in changed] == [100, 200, 0]
    train = [TRANSITION_DESIGNS[i] for i in (0, 1, 3, 6)]
    held_out = [TRANSITION_DESIGNS[i] for i in (2, 4, 5, 7)]
    assert TRANSITION_DESIGNS[4][0] > 0  # Validation must exercise producer BHP tightening.
    for coordinate in (0, 1):
        assert all(min(d[coordinate] for d in train) <= d[coordinate] <= max(x[coordinate] for x in train)
                   for d in held_out)
    assert len(set(TRANSITION_DESIGNS)) == 8
    print('BHP-only intervention, fixed rates and inactive-well preservation passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--reference', type=Path, help='Authenticated export directory with four action channels')
    parser.add_argument('--output', type=Path)
    designs_group = parser.add_mutually_exclusive_group()
    designs_group.add_argument('--local-reference-evaluation', action='store_true')
    designs_group.add_argument('--transition-coverage', action='store_true',
        help='Cover producer BHP transitions in both development training and validation')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not all((args.request, args.reference, args.output)):
        parser.error('request, reference and output required')
    request = CycleRequest.from_mapping(json.loads(args.request.read_text()))
    data = load_trajectory_dataset(args.reference / 'trajectory.csv', manifest=args.reference / 'manifest.json')
    if len(data) != 1 or not data.model_z_identity or data[0].actions.shape[-1] != 4:
        raise ValueError('authenticated BHP history for Model Z required')
    if digest(request.source) != MODEL_Z_SOURCE_SHA256 or request.horizon_months != MONTHS:
        raise ValueError('official reservoir and full 224-month period required')
    t = data[0]
    limits = {(d.date(), well): float(t.actions[i, j, 3]) for i, d in enumerate(t.dates)
              for j, well in enumerate(t.well_ids)}
    args.output.mkdir(parents=True, exist_ok=False)
    designs = [(5, 1), (15, 1), (30, 1), (0, .95), (0, .9), (0, .85), (15, .95), (30, .9)]
    if args.local_reference_evaluation:
        designs = [(2.5, .975), (7.5, .975), (10, .965), (2.5, .94),
                   (5, .925), (12.5, .96), (10, .98), (2.5, .92)]
    elif args.transition_coverage:
        designs = TRANSITION_DESIGNS
    manifest = {'schema': 'timesoil.bhp-only-forecast-evaluation/v1', 'source_sha256': MODEL_Z_SOURCE_SHA256,
        'incumbent_request_sha256': digest(args.request), 'reference_export_sha256': digest(args.reference / 'manifest.json'),
        'calibration_cases': [0, 1, 3, 4, 6], 'test_cases': [2, 5, 7], 'designs': designs,
        'model_selection_allowed_on_test': False, 'rate_status_and_role_controls_fixed': True,
        'local_reference_evaluation': args.local_reference_evaluation,
        'transition_coverage': args.transition_coverage,
        'scenarios': [], 'complete': False, 'script_sha256': digest(Path(__file__))}
    (args.output / 'protocol.json').write_text(json.dumps(manifest, indent=2) + '\n')
    runner = OpmFlowRunner(timeout_seconds=7200, mpi_processes=16, threads_per_process=1, cpu_affinity='14-29')
    for index, (producer_add, injector_factor) in enumerate(designs):
        started = time.monotonic()
        controls = pressure_controls(request.controls, limits, producer_add, injector_factor)
        case = replace(request, scenario_id=f'bhp-only-{index:02d}', controls=controls)
        run = args.output / 'opm' / f'candidate-{index:02d}'
        prepared = runner.prepare(case.source, run, deck=case.deck)
        assert prepared.source_sha256 == MODEL_Z_SOURCE_SHA256
        schedule = prepared.input_dir / case.schedule_relative_path
        source = schedule.read_text()
        months = sorted({a.month for a in controls})
        if months[0] != START.date() or months[-1].year != 2025 or months[-1].month != 8:
            raise ValueError('unexpected intervention period')
        inventory = _source_control_inventory(source, months)
        _validate_source_well_scope(controls, inventory, allow_conversion_to_injection=case.context.get('constraints', {}).get('allow_conversion_to_injection', False))
        constraints = parse_constraints(case.context.get('operating_constraints', []), wells=t.well_ids,
            start=months[0], end=months[-1])
        check_controls(constraints, controls)
        overlay = apply_schedule_overlay(source, controls, known_wells=t.well_ids, end_exclusive=END.date())
        assert overlay.controls_sha256 == case.controls_sha256 and overlay.action_count == 103 * MONTHS
        schedule.write_text(overlay.text)
        result = runner._run_prepared(prepared, parsing_strictness=case.parsing_strictness)
        report, extraction = runner.extract_summary_report(result, run / 'summary-report.txt')
        check_summary(constraints, report, deck_dir=result.deck_path.parent, months=months, unit_system=prepared.unit_system)
        exported = args.output / f'candidate-{index:02d}'
        export_opm_chdd(report, exported / 'chdd.csv', exported / 'trajectory.csv', exported / 'manifest.json',
            scenario_id=case.scenario_id, source_model=case.source_model, opm_run_manifest=result.manifest_path,
            summary_extraction_manifest=extraction, deck_dir=result.deck_path.parent, include_bhp=True)
        with (exported / 'chdd.csv').open() as f:
            rows = list(csv.DictReader(f))
        physical = _physical_control_evidence(rows, controls)
        economics = CHDDEconomicsAdapter.from_env().calculate(opm_management_rows(rows, (START.date(), END.date())),
            start_year=2007, output_dir=run / 'economics', charge_initial_pump=case.charge_initial_pump,
            management_period=(START.date(), END.date()))
        record = {'index': index, 'run': str(run.resolve()), 'directory': str(exported.resolve()),
            'trajectory_sha256': digest(exported / 'trajectory.csv'), 'export_manifest_sha256': digest(exported / 'manifest.json'),
            'controls_sha256': case.controls_sha256, 'physical_control_evidence': physical,
            'official_chdd_m': economics.total_chdd_m, 'seconds': time.monotonic() - started}
        manifest['scenarios'].append(record)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    manifest['complete'] = True
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
