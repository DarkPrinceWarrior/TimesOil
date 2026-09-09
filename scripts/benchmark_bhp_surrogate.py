"""Train a BHP-aware Model Z surrogate on whole held-out 224-month OPM scenarios."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from timesoil.aios.interwell import WellConnectivity
from timesoil.aios.opm_chdd import export_opm_chdd
from timesoil.aios.surrogate import ScenarioTrajectory
from timesoil.aios.track2 import (
    MODEL_Z_SOURCE_SHA256, _VerifiedTrajectoryDataset, fit_track2_surrogate,
    load_trajectory_dataset,
)

START = pd.Timestamp('2007-01-01')
END = pd.Timestamp('2025-09-01')
MONTHS = 224


def management_window(trajectories):
    """Keep the initial state plus every transition in the management period."""
    expected = pd.date_range(START, END, freq='MS')
    selected = []
    for item in trajectories:
        positions = np.flatnonzero((item.dates >= START) & (item.dates <= END))
        if not item.dates[positions].equals(expected):
            raise ValueError('each scenario must cover all 224 management months')
        if item.actions.shape[-1] != 4:
            raise ValueError('BHP training requires four action features')
        selected.append(replace(item, dates=item.dates[positions],
                                states=item.states[positions], actions=item.actions[positions]))
    if isinstance(trajectories, _VerifiedTrajectoryDataset):
        if trajectories.scenario_hashes != tuple(t.content_hash for t in trajectories):
            raise ValueError('verified dataset changed before window selection')
        return _VerifiedTrajectoryDataset(selected, model_z_identity=trajectories.model_z_identity)
    return selected


def self_check():
    dates = pd.date_range('2006-12-01', END, freq='MS')
    states = np.ones((len(dates), 1, 3))
    actions = np.ones((len(dates), 1, 4))
    actions[..., 1] = 1
    actions[..., 3] = 70
    t = ScenarioTrajectory('check', 'synthetic', dates, ('1',), states, actions)
    selected = management_window([t])[0]
    assert len(selected.dates) == MONTHS + 1 and selected.dates[0] == START
    np.testing.assert_array_equal(selected.states[0], states[1])
    assert len(t.dates) == MONTHS + 2
    for invalid in (replace(t, dates=dates[:-1], states=states[:-1], actions=actions[:-1]),
                    replace(t, actions=actions[..., :3])):
        try:
            management_window([invalid])
        except ValueError:
            pass
        else:
            raise AssertionError('incomplete period or missing BHP accepted')
    print('224-month alignment and missing-BHP checks passed', flush=True)


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.reference_batch / 'baseline'
    run = json.loads((base / 'manifest.json').read_text())
    if run['source_sha256'] != MODEL_Z_SOURCE_SHA256:
        raise ValueError('reference baseline must be official Model Z')
    deck_dir = base / 'input' / 'Model_Z'
    exports = args.output / 'reference'
    export_opm_chdd(
        base / 'summary-report.txt', exports / 'chdd.csv', exports / 'baseline.csv',
        exports / 'manifest.json', scenario_id='baseline', source_model='model_z_opm',
        opm_run_manifest=base / 'manifest.json',
        summary_extraction_manifest=base / 'summary-extraction.json',
        deck_dir=deck_dir, unit_system='METRIC', include_bhp=True,
    )
    frame = pd.read_csv(exports / 'baseline.csv', dtype={'well': str})
    selected = frame.loc[(frame.date >= str(START.date())) & (frame.date < str(END.date()))]
    if selected.date.nunique() != MONTHS or len(selected) != MONTHS * 103:
        raise ValueError('baseline must contain 224 complete months of 103 wells')
    controls = args.output / 'baseline-controls.csv'
    selected.to_csv(controls, index=False)
    bundle = args.output / 'scenario-bundle'
    subprocess.run([sys.executable, str(Path(__file__).with_name('generate_track2_scenarios.py')),
                    str(controls), str(deck_dir / 'Model_Z_sch.inc'), str(bundle),
                    '--scenario-count', '10', '--seed', '20260909',
                    '--perturbation-fraction', '0.15', '--bhp-perturbation-fraction', '0.15'], check=True)
    plan = dict(start=str(START.date()), end_exclusive=str(END.date()), months=MONTHS,
                historical_controls_preserved=True, scenario_index_sha256=digest(bundle / 'index.json'),
                reference_manifest_sha256=digest(base / 'manifest.json'),
                baseline_chdd_sha256=digest(exports / 'chdd.csv'),
                source_sha256=MODEL_Z_SOURCE_SHA256, source=str(args.source),
                script_sha256=digest(Path(__file__)))
    write_json(args.output / 'plan.json', plan)
    subprocess.run([sys.executable, str(Path(__file__).with_name('run_track2_scenarios.py')),
                    str(args.source), str(bundle), str(args.output / 'scenario-runs'),
                    '--source-sha256', MODEL_Z_SOURCE_SHA256,
                    '--scenario-index-sha256', plan['scenario_index_sha256'],
                    '--baseline-chdd-sha256', plan['baseline_chdd_sha256'],
                    '--schedule-relative-path', 'Model_Z/Model_Z_sch.inc',
                    '--deck', 'Model_Z/Model_Z.data', '--parsing-strictness', 'low',
                    '--timeout-seconds', '7200', '--include-bhp'], check=True)


def train(args):
    batch = args.output / 'scenario-runs'
    plan = json.loads((args.output / 'plan.json').read_text())
    manifest = json.loads((batch / 'manifest.json').read_text())
    if (manifest['official_source_sha256'] != MODEL_Z_SOURCE_SHA256
            or manifest['scenario_index_sha256'] != plan['scenario_index_sha256']
            or digest(args.output / 'scenario-bundle/index.json') != plan['scenario_index_sha256']
            or manifest['scenario_count'] != 10):
        raise ValueError('scenario batch does not match the BHP experiment plan')
    for record in manifest['scenarios']:
        for name in ('dataset', 'export_manifest', 'run_manifest', 'canonical_chdd'):
            path = (batch / record[name]).resolve()
            if not path.is_relative_to(batch.resolve()) or digest(path) != record[name + '_sha256']:
                raise ValueError(f'scenario artifact mismatch: {name}')
    full = load_trajectory_dataset(batch / 'dataset', manifest=batch / 'manifests')
    trajectories = management_window(full)
    if not trajectories.model_z_identity or len(trajectories) != 10:
        raise ValueError('ten verified official Model Z trajectories required')
    initial = trajectories[0].states[0]
    for item in trajectories[1:]:
        np.testing.assert_allclose(item.states[0], initial, rtol=1e-6, atol=1e-6,
                                   err_msg='scenarios changed physical history before January 2007')
    connectivity = WellConnectivity.from_source(args.source, trajectories[0].well_ids)
    result = fit_track2_surrogate(trajectories, test_fraction=0.3, ensemble_size=5,
                                  n_estimators=160, horizon=MONTHS, seed=20260909,
                                  conformal_level=None, connectivity=connectivity)
    assert set(result.train_ids).isdisjoint(result.test_ids)
    model_dir = args.output / 'surrogate'
    model_manifest = result.model.save(model_dir)
    report = {**result.report(), 'period': plan, 'artifact_hash': model_manifest['artifact_hash'],
              'batch_manifest_sha256': digest(batch / 'manifest.json'),
              'full_source_scenario_hashes': {t.scenario_id: t.content_hash for t in full},
              'action_features': ['control_value', 'control_target_code', 'status', 'bhp_limit'],
              'final_model_refit_on_test': False, 'future_observations_used_in_rollout': False,
              'is_new_optimization_result': False}
    write_json(args.output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--reference-batch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not args.source or not args.output or (not args.train_only and not args.reference_batch):
        parser.error('source, output and reference-batch required')
    args.source, args.output = args.source.absolute(), args.output.absolute()
    if not args.train_only:
        args.reference_batch = args.reference_batch.absolute()
        prepare(args)
    train(args)


if __name__ == '__main__':
    main()
