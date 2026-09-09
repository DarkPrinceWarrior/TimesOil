"""Google forecasts conditioned only on verified history and planned field controls."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from timesoil.aios.interwell import WellConnectivity
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
    def __init__(self, settings, state, source_sha256):
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
        np.testing.assert_allclose(self.states[-1], state_array(state, self.wells), rtol=0, atol=1e-5)
        self.month = state.month
        self.origin = origin
        self.initial_origin = origin
        self.last_predictions = {}
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
        self.provenance = {'model_revision': MODEL_REVISION, 'input_hashes': self.files,
            'source_sha256': source_sha256, 'well_count': len(self.wells),
            'initial_observation_cutoff': str(state.month), 'future_observations_used': False,
            'uncertainty_calibrated': False, 'static_head_trained': False,
            'forecast_authorizes_control_without_opm': False,
            'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}

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
        np.testing.assert_allclose(state_array(state, self.wells), self.states[-1], rtol=0, atol=1e-5)
        months, future = self.controls_array(controls)
        expected = [d.date() for d in self.dates[self.origin:self.origin + len(months)]]
        if months != expected or not months or months[0] != state.month:
            raise ValueError('forecast controls must span consecutive months from the current state')
        length = min(128, len(self.actions))
        past = self.states[-length:].transpose(1, 2, 0).reshape(-1, length).astype(np.float32)
        actions = np.concatenate([self.actions[-length:], future])
        rates = np.stack([np.where(actions[..., 1] == code, actions[..., 0], 0) * actions[..., 2]
                          for code in range(3)], axis=-1)
        own = np.concatenate([rates, actions[..., 2:]], axis=-1).transpose(1, 2, 0).reshape(-1, len(actions))
        allocated = self.geology.features(np.zeros((len(actions) * len(self.wells), 3)),
            actions.reshape(-1, 4))[:, 0].reshape(len(actions), len(self.wells)).T
        cov = np.concatenate([own, allocated]).astype(np.float32)
        prediction = next(self.forecaster.predict_batch([past], horizon=len(months),
            past_future_covariates=[cov], return_quantiles=False, make_positive=True,
            use_symmetric_averaging=False)).forecast.reshape(len(self.wells), 3, len(months)).transpose(2, 0, 1)
        if not np.isfinite(prediction).all():
            raise ValueError('non-finite Google forecast')
        prediction = np.maximum(prediction, 0)
        prediction[..., 0] = np.minimum(prediction[..., 0], prediction[..., 1])
        prediction[(future[..., 2] == 0) | (future[..., 1] == 2), :2] = 0
        key = sha256(future.tobytes()).hexdigest()
        self.last_predictions[sha256(future[0].tobytes()).hexdigest()] = prediction[0].copy()
        days = np.array([pd.Timestamp(d).days_in_month for d in months])
        volumes = (prediction[..., :2] * days[:, None, None]).sum(axis=0)
        return {'model_revision': MODEL_REVISION, 'observation_cutoff': str(state.month),
            'months': len(months), 'well_count': len(self.wells), 'controls_sha256': key,
            'future_observations_used': False, 'official_chdd': False, 'uncertainty_calibrated': False,
            'columns': ['well', 'next_oil_m3d', 'next_liquid_m3d', 'next_bhp_bar',
                        'remaining_oil_volume_m3', 'remaining_liquid_volume_m3', 'terminal_bhp_bar'],
            'rows': [[well, *np.round(prediction[0, i], 6).tolist(), *np.round(volumes[i], 3).tolist(),
                      round(float(prediction[-1, i, 2]), 6)] for i, well in enumerate(self.wells)],
            'observed_month_errors': self.observed_errors[-3:],
            'decision_rule': 'Forecast supports a hypothesis; select only using full remaining OPM and official CHDD.'}

    def observe(self, state, controls):
        months, actions = self.controls_array(controls)
        if months != [self.month] or pd.Timestamp(state.month) != self.dates[self.origin + 1]:
            raise ValueError('observations must commit exactly the next month')
        observed = state_array(state, self.wells)
        prior = self.last_predictions.get(sha256(actions[0].tobytes()).hexdigest())
        if prior is not None:
            error = np.abs(prior - observed)
            self.observed_errors.append({'control_month': str(self.month),
                'oil_wape': float(error[:, 0].sum() / max(np.abs(observed[:, 0]).sum(), 1e-9)),
                'liquid_wape': float(error[:, 1].sum() / max(np.abs(observed[:, 1]).sum(), 1e-9)),
                'pressure_rmse_bar': float(np.sqrt(np.mean(error[:, 2] ** 2)))})
        self.states = np.concatenate([self.states, observed[None]])
        self.actions = np.concatenate([self.actions, actions])
        self.origin += 1
        self.month = state.month
        self.last_predictions.clear()

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
    planner.states = np.ones((3, 2, 3))
    planner.actions = np.tile([[100., 1, 1, 70], [10., 2, 1, 70]], (2, 1, 1))
    planner.geology = WellConnectivity(planner.wells, [[0, 1], [1, 0]], [[10, .2, 2], [20, .1, 3]], {})
    planner.last_predictions, planner.observed_errors = {}, []
    captured = []
    class Forecaster:
        def predict_batch(self, histories, *, horizon, past_future_covariates, **kwargs):
            captured.append((histories[0].copy(), past_future_covariates[0].copy()))
            assert histories[0].shape == (6, 2)
            assert past_future_covariates[0].shape == (12, 2 + horizon)
            yield Obj(forecast=np.ones((6, horizon)))
    planner.forecaster = Forecaster()
    state = Obj(month=planner.month, wells=tuple(Obj(well=w, oil_rate=1., liquid_rate=1., bhp=1.) for w in planner.wells))
    controls = tuple(ControlAction(date(2014, m, 1), w,
        WellRole.PRODUCER if w == 'p' else WellRole.INJECTOR, WellStatus.OPEN,
        ControlTarget.LIQUID_RATE if w == 'p' else ControlTarget.WATER_INJECTION_RATE,
        100. if w == 'p' else 20.) for m in (3, 4) for w in planner.wells)
    result = planner.predict(state, controls)
    assert result['months'] == 2 and result['well_count'] == 2
    np.testing.assert_array_equal(captured[0][0], 1)
    np.testing.assert_array_equal(captured[0][1][-2], [10, 10, 20, 20])
    observed = Obj(month=date(2014, 4, 1), wells=state.wells)
    planner.observe(observed, controls[:2])
    assert len(planner.states) == 4 and len(planner.actions) == 3
    assert len(planner.observed_errors) == 1
    assert planner.observed_errors[0]['oil_wape'] == .5  # Injector forecast is physically projected to zero.
    try:
        planner.observe(observed, controls[:2])
    except ValueError:
        pass
    else:
        raise AssertionError('a repeated observation was committed twice')
    print('Forecast control alignment, full field allocation and single-month feedback checks passed', flush=True)


if __name__ == '__main__':
    self_check()
