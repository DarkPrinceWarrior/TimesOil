"""Google forecasts conditioned only on verified history and planned field controls."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.aios.interwell import WellConnectivity
from timesoil.aios.opm import OpmGdmBackend, _source_digest
from timesoil.aios.track2 import load_trajectory_dataset

MODEL_REVISION = '43046b85ec22d584a13f8098c2ed39c889e129c2'


def state_array(state, wells):
    by_well = {w.well: w for w in state.wells}
    if set(by_well) != set(wells) or len(by_well) != len(state.wells):
        raise ValueError('forecast observation requires the exact field inventory')
    result = np.array([[by_well[w].oil_rate, by_well[w].liquid_rate, by_well[w].bhp] for w in wells])
    if not np.isfinite(result).all():
        raise ValueError('non-finite observed state')
    return result


class TimesFMPlanning:
    def __init__(self, settings, case, state, backend):
        if not isinstance(backend, OpmGdmBackend):
            raise ValueError('forecast observations require the physical OPM backend')
        self.backend, self.case = backend, case
        source_sha256 = _source_digest(backend.source)
        self.source_sha256 = source_sha256
        backend._authenticated_history(case, state)
        self.last_restart_ref = state.restart_ref
        self.controller_state = state_array(state, tuple(sorted(w.well for w in state.wells)))
        self.files = {}
        for name in ('history', 'geology'):
            path = Path(settings[name])
            digest = sha256(path.read_bytes()).hexdigest()
            if digest != settings[name + '_sha256']:
                raise ValueError(f'forecast {name} hash mismatch')
            self.files[str(path)] = digest
        csv = Path(settings['history'])
        manifest = csv.parent / 'manifest.json'
        dataset = load_trajectory_dataset(csv, manifest=manifest)
        if len(dataset) != 1:
            raise ValueError('forecast history requires one verified trajectory')
        trajectory = dataset[0]
        metadata = json.loads(manifest.read_text())
        if metadata['provenance']['opm_source_sha256'] != source_sha256:
            raise ValueError('forecast history belongs to another reservoir')
        self.files[str(manifest)] = sha256(manifest.read_bytes()).hexdigest()
        self.geology = WellConnectivity.from_dict(json.loads(Path(settings['geology']).read_text()))
        if self.geology.provenance['source_sha256'] != source_sha256:
            raise ValueError('forecast geology belongs to another reservoir')
        self.wells = tuple(trajectory.well_ids)
        if self.wells != self.geology.well_ids:
            raise ValueError('history and geology well order differ')
        origin = int(trajectory.dates.get_loc(pd.Timestamp(state.month)))
        if origin < 1:
            raise ValueError('forecast requires observed history before the control origin')
        self.dates = trajectory.dates
        frame = pd.read_csv(csv, dtype={'well': str}, parse_dates=['date'])
        if 'bhp_limit' not in frame:
            raise ValueError('forecast history requires exported requested BHP limits')
        self.source_bhp = frame.pivot(index='date', columns='well', values='bhp_limit').reindex(
            index=self.dates, columns=self.wells).to_numpy(float)
        if not np.isfinite(self.source_bhp).all() or (self.source_bhp < 0).any():
            raise ValueError('invalid requested BHP history')
        self.states = trajectory.states[:origin + 1].copy()
        self.actions = np.concatenate([trajectory.actions[:origin, :, :3],
                                       self.source_bhp[:origin, :, None]], axis=-1)
        self.month = state.month
        self.origin = origin
        self.initial_origin = origin
        self.last_predictions = {}
        self.last_diagnostics = {}
        self.observed_errors = []
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from timesfm3 import ModelConfig, TimesFM3Forecaster
        if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('forecast must run on the allocated A100')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.15)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.manual_seed(20260909)
        self.forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
            revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
        self.calibration = self.calibrate()
        self.provenance = {'model_revision': MODEL_REVISION, 'input_hashes': self.files,
            'source_sha256': source_sha256, 'well_count': len(self.wells),
            'target_units': ['oil tonnes/day', 'liquid tonnes/day', 'reservoir WBP9 bar'],
            'controller_units': ['oil surface m3/day', 'liquid surface m3/day', 'well BHP bar'],
            'initial_observation_cutoff': str(state.month), 'future_observations_used': False,
            'uncertainty_calibrated': False, 'static_head_trained': False,
            'one_month_historical_calibration': self.calibration,
            'forecast_authorizes_control_without_opm': False,
            'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}

    def forecast_arrays(self, states, history_actions, future):
        length = min(128, len(history_actions))
        if length < 1 or len(states) != len(history_actions) + 1:
            raise ValueError('forecast history states/actions are misaligned')
        past = states[-length:].transpose(1, 2, 0).reshape(-1, length).astype(np.float32)
        actions = np.concatenate([history_actions[-length:], future])
        rates = np.stack([np.where(actions[..., 1] == code, actions[..., 0], 0) * actions[..., 2]
                          for code in range(3)], axis=-1)
        own = np.concatenate([rates, actions[..., 2:]], axis=-1).transpose(1, 2, 0).reshape(-1, len(actions))
        allocated = self.geology.features(np.zeros((len(actions) * len(self.wells), 3)),
            actions.reshape(-1, 4))[:, 0].reshape(len(actions), len(self.wells)).T
        cov = np.concatenate([own, allocated]).astype(np.float32)
        prediction = next(self.forecaster.predict_batch([past], horizon=len(future),
            past_future_covariates=[cov], return_quantiles=False, make_positive=True,
            use_symmetric_averaging=False)).forecast.reshape(len(self.wells), 3, len(future)).transpose(2, 0, 1)
        if not np.isfinite(prediction).all():
            raise ValueError('non-finite Google forecast')
        prediction = np.maximum(prediction, 0)
        prediction[..., 0] = np.minimum(prediction[..., 0], prediction[..., 1])
        prediction[(future[..., 2] == 0) | (future[..., 1] == 2), :2] = 0
        return prediction

    def calibrate(self):
        if self.origin < 48:
            raise ValueError('historical calibration requires at least 48 observed monthly transitions')
        residuals, naive, truth, dates = [], [], [], []
        for origin in range(self.origin - 36, self.origin):
            prediction = self.forecast_arrays(self.states[:origin + 1], self.actions[:origin],
                                               self.actions[origin:origin + 1])[0]
            residuals.append(np.abs(prediction - self.states[origin + 1]))
            naive.append(np.abs(self.states[origin] - self.states[origin + 1]))
            truth.append(self.states[origin + 1])
            dates.append(str(self.dates[origin + 1].date()))
        residuals, naive, truth = map(np.asarray, (residuals, naive, truth))
        # ponytail: 24 monthly residuals per well; longer histories and blocked calibration are needed for reliable coverage under regime shifts.
        rank = min(23, int(np.ceil(25 * .9)) - 1)
        self.interval_radius = np.sort(residuals[:24], axis=0)[rank]
        test, target = residuals[24:], truth[24:]
        self.control_min = self.actions.min(axis=0)
        self.control_max = self.actions.max(axis=0)
        return {'horizon_months': 1, 'nominal_coverage': .9,
            'calibration_dates': dates[:24], 'test_dates': dates[24:],
            'test_wape_oil_liquid': (test[..., :2].sum(axis=(0, 1)) /
                np.maximum(np.abs(target[..., :2]).sum(axis=(0, 1)), 1e-9)).tolist(),
            'naive_test_wape_oil_liquid': (naive[24:, ..., :2].sum(axis=(0, 1)) /
                np.maximum(np.abs(target[..., :2]).sum(axis=(0, 1)), 1e-9)).tolist(),
            'test_coverage_oil_liquid_pressure': np.mean(test <= self.interval_radius, axis=(0, 1)).tolist(),
            'test_pressure_rmse_bar': float(np.sqrt(np.mean(test[..., 2] ** 2))),
            'scope': 'Chronological historical check, including zero production for injectors; not a coverage guarantee under new control regimes or over the full remaining horizon.'}

    def controls_array(self, controls):
        rows = {(a.month, a.well): a for a in controls}
        months = sorted({a.month for a in controls})
        if len(rows) != len(controls) or len(rows) != len(months) * len(self.wells):
            raise ValueError('forecast controls must contain each well exactly once per month')
        output = []
        for month in months:
            index = int(self.dates.get_loc(pd.Timestamp(month)))
            values = []
            for j, well in enumerate(self.wells):
                action = rows[month, well]
                values.append([action.value, {'ORAT': 0, 'LRAT': 1, 'WRAT': 2}[action.target.value],
                    int(action.status.value == 'OPEN'),
                    self.source_bhp[index, j] if action.bhp_limit is None else action.bhp_limit])
            output.append(values)
        result = np.array(output, dtype=float)
        if not np.isfinite(result).all() or (result[..., (0, 3)] < 0).any():
            raise ValueError('invalid planned forecast controls')
        return months, result

    def geology_context(self):
        names = self.geology.provenance.get('static_feature_names', ['perm_md', 'poro', 'net_thickness_m'])
        values = self.geology.provenance.get('static_features', self.geology.static.tolist())
        rows = []
        for i, well in enumerate(self.wells):
            weights = self.geology.weights[i]
            neighbors = [(self.wells[j], round(float(weights[j]), 6))
                         for j in np.argsort(-weights)[:5] if weights[j] > 0]
            rows.append([well, [round(float(v), 6) for v in values[i]], neighbors])
        return {'columns': ['well', 'static_features', 'five_strongest_geological_neighbors'],
            'static_feature_names': names, 'rows': rows,
            'all_links_used_in_forecast': True,
            'limitations': self.geology.provenance['limitations']}

    def predict(self, state, controls):
        if state.month != self.month:
            raise ValueError('forecast state is not the last committed observation')
        np.testing.assert_allclose(state_array(state, self.wells), self.controller_state, rtol=0, atol=1e-5)
        months, future = self.controls_array(controls)
        expected = [d.date() for d in self.dates[self.origin:self.origin + len(months)]]
        if months != expected or not months or months[0] != state.month:
            raise ValueError('forecast controls must span consecutive months from the current state')
        prediction = self.forecast_arrays(self.states, self.actions, future)
        one_month = prediction[0] if len(months) == 1 else self.forecast_arrays(self.states, self.actions, future[:1])[0]
        ood = np.any((future < self.control_min - 1e-6) | (future > self.control_max + 1e-6), axis=-1)
        key = sha256(future.tobytes()).hexdigest()
        first_month_key = sha256(future[0].tobytes()).hexdigest()
        self.last_predictions[first_month_key] = one_month.copy()
        self.last_diagnostics[first_month_key] = {
            'model': 'Google TimesFM 3', 'revision': MODEL_REVISION,
            'training': 'Official pretrained foundation model; task-specific head tuning was not applied to Model Y.',
            'historical_validation_and_one_month_uq': self.calibration,
            'full_remaining_horizon_uq_calibrated': False,
            'ood_control_well_months': int(ood.sum()),
            'ood_wells': [w for i, w in enumerate(self.wells) if ood[:, i].any()],
            'scope': 'Forecast proposed a hypothesis; physical OPM and official full-period CHDD select the control. No autonomous surrogate certification.'}
        days = np.array([pd.Timestamp(d).days_in_month for d in months])
        volumes = (prediction[..., :2] * days[:, None, None]).sum(axis=0)
        return {'model_revision': MODEL_REVISION, 'observation_cutoff': str(state.month),
            'months': len(months), 'well_count': len(self.wells), 'controls_sha256': key,
            'future_observations_used': False, 'official_chdd': False, 'uncertainty_calibrated': False,
            'one_month_historical_calibration': self.calibration,
            'ood_control_well_months': int(ood.sum()),
            'ood_wells': [w for i, w in enumerate(self.wells) if ood[:, i].any()],
            'columns': ['well', 'next_oil_tpd', 'next_liquid_tpd', 'next_reservoir_pressure_bar',
                        'estimated_remaining_oil_tonnes', 'estimated_remaining_liquid_tonnes', 'terminal_reservoir_pressure_bar'],
            'rows': [[well, *np.round(one_month[i], 6).tolist(), *np.round(volumes[i], 3).tolist(),
                      round(float(prediction[-1, i, 2]), 6)] for i, well in enumerate(self.wells)],
            'one_month_intervals': {'columns': ['well', 'lower_oil_tpd_liquid_tpd_reservoir_bar', 'upper_oil_tpd_liquid_tpd_reservoir_bar'],
                'rows': [[w, np.maximum(one_month[i] - self.interval_radius[i], 0).round(6).tolist(),
                          (one_month[i] + self.interval_radius[i]).round(6).tolist()] for i, w in enumerate(self.wells)],
                'scope': 'One-step historical calibration; full-horizon forecast has no calibrated interval.'},
            'observed_month_errors': self.observed_errors[-3:],
            'decision_rule': 'Forecast supports a hypothesis; select only using full remaining OPM and official CHDD.'}

    def diagnostics(self, controls):
        _, values = self.controls_array(controls)
        key = sha256(values[0].tobytes()).hexdigest()
        if key not in self.last_diagnostics:
            raise ValueError('selected controls have no forecast diagnostic record')
        return self.last_diagnostics[key]

    def committed_model_state(self, state):
        path, _ = self.backend._parse_restart_ref(state.restart_ref)
        lineage = json.loads(path.read_text())
        if (lineage.get('source_sha256') != self.source_sha256
                or lineage.get('prior_restart_ref') != self.last_restart_ref
                or lineage.get('next_state') != self.backend._state_value(state)):
            raise ValueError('forecast observation differs from the committed physical lineage')
        self.backend._verify_lineage_artifacts(path.parent, lineage['artifacts'])
        def artifact(purpose):
            matches = [a for a in lineage['artifacts'] if a['purpose'] == purpose]
            if len(matches) != 1:
                raise ValueError('missing or duplicate committed forecast artifact')
            return path.parent / matches[0]['path']
        data = load_trajectory_dataset(artifact('canonical_trajectory'), manifest=artifact('canonical_export_manifest'))
        if len(data) != 1 or data[0].well_ids != self.wells:
            raise ValueError('committed forecast observation grid differs')
        # A planning run may contain a future tail. Read only the newly committed observation.
        result = data[0].states[int(data[0].dates.get_loc(pd.Timestamp(state.month)))].copy()
        self.last_restart_ref = state.restart_ref
        return result

    def observe(self, state, controls):
        months, actions = self.controls_array(controls)
        if months != [self.month] or pd.Timestamp(state.month) != self.dates[self.origin + 1]:
            raise ValueError('observations must commit exactly the next month')
        observed = self.committed_model_state(state)
        prior = self.last_predictions.get(sha256(actions[0].tobytes()).hexdigest())
        if prior is not None:
            error = np.abs(prior - observed)
            self.observed_errors.append({'control_month': str(self.month),
                'oil_wape': float(error[:, 0].sum() / max(np.abs(observed[:, 0]).sum(), 1e-9)),
                'liquid_wape': float(error[:, 1].sum() / max(np.abs(observed[:, 1]).sum(), 1e-9)),
                'pressure_rmse_bar': float(np.sqrt(np.mean(error[:, 2] ** 2)))})
        self.states = np.concatenate([self.states, observed[None]])
        self.actions = np.concatenate([self.actions, actions])
        self.controller_state = state_array(state, self.wells)
        self.origin += 1
        self.month = state.month
        self.last_predictions.clear()
        self.last_diagnostics.clear()

    def verify_files(self):
        for path, digest in self.files.items():
            if sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('forecast input changed during the controller run')
        if sha256(Path(__file__).read_bytes()).hexdigest() != self.provenance['script_sha256']:
            raise ValueError('forecast implementation changed during the controller run')


def self_check():
    from types import SimpleNamespace as Obj
    from datetime import date
    from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
    planner = TimesFMPlanning.__new__(TimesFMPlanning)
    planner.wells = ('p', 'i')
    planner.dates = pd.date_range('2014-01-01', periods=5, freq='MS')
    planner.month, planner.origin = date(2014, 3, 1), 2
    planner.source_bhp = np.full((5, 2), 70.)
    planner.states = np.tile([[1., 2, 3], [0., 0, 4]], (3, 1, 1))
    planner.controller_state = np.array([[10., 20, 30], [0., 0, 40]])
    planner.committed_model_state = lambda state: np.array([[2., 3, 4], [0., 0, 5]])
    planner.actions = np.tile([[100., 1, 1, 70], [10., 2, 1, 70]], (2, 1, 1))
    planner.geology = WellConnectivity(planner.wells, [[0, 1], [1, 0]], [[10, .2, 2], [20, .1, 3]], {})
    planner.last_predictions, planner.observed_errors = {}, []
    planner.last_diagnostics = {}
    planner.calibration = {'scope': 'synthetic self-check'}
    planner.interval_radius = np.zeros((2, 3))
    planner.control_min, planner.control_max = planner.actions.min(axis=0), planner.actions.max(axis=0)
    captured = []
    class Forecaster:
        def predict_batch(self, histories, *, horizon, past_future_covariates, **kwargs):
            captured.append((histories[0].copy(), past_future_covariates[0].copy()))
            assert histories[0].shape == (6, 2)
            assert past_future_covariates[0].shape == (12, 2 + horizon)
            yield Obj(forecast=np.ones((6, horizon)))
    planner.forecaster = Forecaster()
    state = Obj(month=planner.month, wells=tuple(Obj(well=w, oil_rate=values[0], liquid_rate=values[1], bhp=values[2])
        for w, values in zip(planner.wells, planner.controller_state, strict=True)))
    controls = tuple(ControlAction(date(2014, m, 1), w,
        WellRole.PRODUCER if w == 'p' else WellRole.INJECTOR, WellStatus.OPEN,
        ControlTarget.LIQUID_RATE if w == 'p' else ControlTarget.WATER_INJECTION_RATE,
        100. if w == 'p' else 20.) for m in (3, 4) for w in planner.wells)
    result = planner.predict(state, controls)
    assert result['months'] == 2 and result['well_count'] == 2
    np.testing.assert_array_equal(captured[0][0], np.repeat([[1], [2], [3], [0], [0], [4]], 2, axis=1))
    np.testing.assert_array_equal(captured[0][1][-2], [10, 10, 20, 20])
    observed = Obj(month=date(2014, 4, 1), wells=state.wells)
    planner.observe(observed, controls[:2])
    assert len(planner.states) == 4 and len(planner.actions) == 3
    assert len(planner.observed_errors) == 1
    assert planner.observed_errors[0]['oil_wape'] == .5
    np.testing.assert_array_equal(planner.states[-1], [[2, 3, 4], [0, 0, 5]])
    np.testing.assert_array_equal(planner.controller_state, [[10, 20, 30], [0, 0, 40]])
    try:
        planner.observe(observed, controls[:2])
    except ValueError:
        pass
    else:
        raise AssertionError('a repeated observation was committed twice')
    print('Forecast control alignment, full field allocation and single-month feedback checks passed', flush=True)


if __name__ == '__main__':
    self_check()
