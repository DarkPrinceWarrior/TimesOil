"""Seal forecast CHDD selection, then allow one immutable graph's final OPM verification."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

from timesoil.aios.economics import CHDDEconomicsAdapter
from timesoil.aios.workflow import CycleRequest
from timesfm_economics import ECONOMIC_TARGETS


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def checked_file(root, name, expected=None):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('selection artifact escapes its directory')
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f'selection artifact changed: {name}')
    return path, actual


def best_forecast(candidates):
    if not candidates or [row['id'] for row in candidates] != list(range(len(candidates))):
        raise ValueError('complete ordered candidate ledger required')
    for row in candidates:
        value = row.get('forecast_chdd_m')
        if (type(row['id']) is not int or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or tuple(row.get('economic_targets', ())) != ECONOMIC_TARGETS
                or row.get('is_official_chdd') is not False
                or type(row.get('forecast_eligible')) is not bool):
            raise ValueError('complete economic forecasts required; screening margin is not CHDD')
    eligible = [row for row in candidates if row['forecast_eligible']]
    if not eligible:
        raise ValueError('no forecast satisfies the supported constraints')
    return max(eligible, key=lambda row: (row['forecast_chdd_m'], -row['id']))


def seal_forecast_selection(root):
    root = root.resolve()
    if (root / 'selection-before-opm.json').exists() or (root / 'final-verification-attempt.json').exists():
        raise FileExistsError('selection or final verification already exists')
    receipt = json.loads((root / 'proposal-receipt.json').read_text())
    if (receipt.get('economic_selection') is not True or receipt.get('search_opm_calls') != 0
            or receipt.get('reference_manifest_sha256') is not None
            or receipt.get('reference_correction_sha256') is not None
            or receipt.get('final_chdd_computed') is not False):
        raise ValueError('forecast-only search without physical future references required')
    candidates = json.loads((root / 'candidates.json').read_text())
    selected = best_forecast(candidates)
    files = {}
    for name in ('proposal-receipt.json', 'candidates.json'):
        files[name] = checked_file(root, name)[1]
    for pattern in ('planning-context-*.json', 'policy-attempts-*.json', 'rejected-plan-*.json'):
        for path in sorted(root.glob(pattern)):
            files[path.name] = checked_file(root, path.name)[1]
    profile = receipt['normative_profile']
    for row in candidates:
        if row['trained_head_sha256'] != receipt['head_sha256'] or row['training_report_sha256'] != receipt['head_report_sha256']:
            raise ValueError('candidates use different frozen models')
        request_name = f"request-{row['id']:02d}.json"
        path, files[request_name] = checked_file(root, request_name)
        request = CycleRequest.from_mapping(json.loads(path.read_text()), base_dir=root)
        if request.controls_sha256 != row['controls_sha256'] or request.horizon_months != receipt['horizon_months']:
            raise ValueError('candidate request disagrees with forecast controls or period')
        name = f"forecast-{row['id']:02d}.npz"
        files[name] = checked_file(root, name, row['forecast_sha256'])[1]
        directory = row['forecast_economics_directory']
        name = f'{directory}/manifest.json'
        path, files[name] = checked_file(root, name, row['forecast_economics_manifest_sha256'])
        economics = json.loads(path.read_text())
        if (economics['management_period']['total_chdd_m'] != row['forecast_chdd_m']
                or economics['norms_source_sha256'] != profile['source_sha256']['Нормативы_ЧДД.xlsx']
                or any(value != profile['source_sha256'][key] for key, value in economics['calculator_sha256'].items())
                or economics['assumption_overrides'] != ({'chargeInitialPump': request.charge_initial_pump}
                    if request.charge_initial_pump is not None else {})):
            raise ValueError('forecast score, norms or official calculator changed')
        for artifact, hash_key in [('input', 'input_sha256'), ('result', 'result_sha256')]:
            name = f"{directory}/{economics['artifacts'][artifact]}"
            files[name] = checked_file(root, name, economics[hash_key])[1]
    for index, candidate_id in enumerate(receipt['agent_proposal_ids']):
        name = f'agent-{index:02d}.json'
        path, files[name] = checked_file(root, name)
        plan = json.loads(path.read_text())
        if len(plan['decisions']) != 3 or not all(d['approved'] for d in plan['decisions']) or candidate_id not in range(len(candidates)):
            raise ValueError('each agent proposal requires three planning approvals')
    request_name = f"request-{selected['id']:02d}.json"
    request = CycleRequest.from_mapping(json.loads((root / request_name).read_text()), base_dir=root)
    baseline = candidates[0]['forecast_chdd_m']
    seal = {'schema': 'timesoil.forecast-selected-final-opm/v1',
        'sealed_at_utc': datetime.now(timezone.utc).isoformat(), 'selected_id': selected['id'],
        'selected_request': request_name, 'selected_request_sha256': request.request_sha256,
        'controls_sha256': request.controls_sha256, 'schedule_overlay_sha256': selected['schedule_overlay_sha256'],
        'source_sha256': digest(request.source), 'selection_metric': 'forecast_chdd_m',
        'forecast_chdd_m': selected['forecast_chdd_m'], 'baseline_forecast_chdd_m': baseline,
        'forecast_uplift_percent': 100 * (selected['forecast_chdd_m'] / baseline - 1) if baseline > 0 else None,
        'search_opm_calls': 0, 'maximum_final_opm_calls': 1, 'reselection_after_opm_allowed': False,
        'horizon_months': receipt['horizon_months'], 'head_sha256': receipt['head_sha256'],
        'normative_profile': profile, 'files': files,
        'independently_certified_accuracy': False, 'physical_uplift_verified': False}
    path = root / 'selection-before-opm.json'
    with path.open('x') as handle:
        json.dump(seal, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    path.chmod(0o444)
    return {'selection': str(path), 'sha256': digest(path), 'selected_id': selected['id'],
            'forecast_chdd_m': selected['forecast_chdd_m']}


def verify_selection(root, expected_hash):
    path, _ = checked_file(root, 'selection-before-opm.json', expected_hash)
    seal = json.loads(path.read_text())
    if (seal['schema'] != 'timesoil.forecast-selected-final-opm/v1' or seal['search_opm_calls'] != 0
            or seal['maximum_final_opm_calls'] != 1 or seal['reselection_after_opm_allowed'] is not False):
        raise ValueError('invalid final-only selection contract')
    for name, expected in seal['files'].items():
        checked_file(root, name, expected)
    selected = best_forecast(json.loads((root / 'candidates.json').read_text()))
    if selected['id'] != seal['selected_id'] or seal['selected_request'] != f"request-{selected['id']:02d}.json":
        raise ValueError('sealed graph does not maximize eligible forecast CHDD')
    request_path, _ = checked_file(root, seal['selected_request'])
    request = CycleRequest.from_mapping(json.loads(request_path.read_text()), base_dir=root)
    if (request.request_sha256 != seal['selected_request_sha256'] or digest(request.source) != seal['source_sha256']
            or request.controls_sha256 != seal['controls_sha256']):
        raise ValueError('sealed request, source or graph changed')
    return seal, request, request_path


def run_final_verification(root, expected_hash, baseline, output):
    root, output = root.resolve(), output.resolve()
    seal, request, request_path = verify_selection(root, expected_hash)
    current_profile = CHDDEconomicsAdapter.from_env().normative_profile(charge_initial_pump=request.charge_initial_pump)
    if current_profile != seal['normative_profile']:
        raise ValueError('final economics differ from forecast selection norms')
    if output.exists():
        raise FileExistsError('final verification output already exists')
    command = [sys.executable, '-m', 'timesoil.aios.cli', 'full-cycle', str(request_path),
               '--runs-dir', str(output), '--run-id', 'selected', '--timeout', '7200']
    # This receipt is the at-most-once barrier, including failed or interrupted attempts.
    with (root / 'final-verification-attempt.json').open('x') as handle:
        json.dump({'seal_sha256': expected_hash, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                   'command': command, 'maximum_opm_calls': 1}, handle, indent=2)
        handle.write('\n')
    output.mkdir()
    with (output / 'full-cycle.log').open('x') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    (output / 'full-cycle.exit').write_text(str(code) + '\n')
    verify_selection(root, expected_hash)
    if code:
        raise RuntimeError('final cycle failed; this selection cannot launch a second OPM attempt')
    from compare_track2_cycles import compare
    audit = compare(baseline, output / 'selected', expected_months=seal['horizon_months'])
    final = json.loads((output / 'selected/full-cycle-receipt.json').read_text())
    if (final['request_sha256'] != request.request_sha256
            or final['controls']['canonical_schedule_sha256'] != seal['controls_sha256']
            or final['artifacts']['exact_opm_input_schedule']['sha256'] != seal['schedule_overlay_sha256']):
        raise ValueError('final OPM did not execute the sealed graph')
    audit.update(selection_seal_sha256=expected_hash, selection_metric='forecast_chdd_m',
        selected_before_opm=True, search_opm_calls=0, final_opm_calls=1,
        reselection_after_opm=False, forecast_chdd_m=seal['forecast_chdd_m'],
        forecast_error_m=final['economics']['total_chdd_m'] - seal['forecast_chdd_m'])
    (output / 'final-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seal-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run_final_verification(args.selection, args.seal_sha256, args.baseline, args.output)))
