"""Stateful Track 2 surrogate with a CRM-inspired physical baseline.

The model operates on complete scenarios.  A control at month ``t`` predicts
the state at ``t + 1``; multi-step forecasts feed predictions back as state.
LightGBM only learns the residual left by the deterministic physical baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import lightgbm as lgb
import numpy as np
import pandas as pd


SCHEMA_VERSION = 2
STATE_FEATURES = ("oil_tpd", "liquid_tpd", "pressure_bar")
ACTION_FEATURES = ("control_value", "control_target_code", "status")
# ponytail: artifact-count safety cap; raise with the training contract if larger ensembles are needed.
_MAX_ARTIFACT_ENSEMBLE_SIZE = 64
RESIDUAL_FEATURES = (
    "oil_tpd",
    "liquid_tpd",
    "pressure_bar",
    "control_value",
    "target_orat",
    "target_lrat",
    "target_wrat",
    "status",
    "oil_fraction",
    "watercut",
    "injection_minus_liquid",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ValueError(f"{label} changed while it was read")
    return data


def _validated_surrogate_artifact(
    manifest: dict[str, Any], model_bytes: bytes
) -> tuple[dict[str, str], dict[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported surrogate manifest schema: {manifest.get('schema_version')}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or not all(
        isinstance(name, str)
        and name not in {"", ".", ".."}
        and "/" not in name
        and "\\" not in name
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for name, digest in files.items()
    ):
        raise ValueError("surrogate manifest files must map basenames to SHA-256 strings")
    files = dict(files)
    if manifest.get("artifact_hash") != sha256(_canonical_json(files)).hexdigest():
        raise ValueError("invalid surrogate artifact hash")
    if "model.json" not in files:
        raise ValueError("surrogate artifact must list model.json")
    if sha256(model_bytes).hexdigest() != files["model.json"]:
        raise ValueError("surrogate artifact file failed hash check: model.json")
    try:
        metadata = json.loads(model_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("surrogate model.json is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("surrogate model.json must be a JSON object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported surrogate schema: {metadata.get('schema_version')}")
    ensemble_size = metadata.get("ensemble_size")
    if (
        not isinstance(ensemble_size, int)
        or isinstance(ensemble_size, bool)
        or not 2 <= ensemble_size <= _MAX_ARTIFACT_ENSEMBLE_SIZE
    ):
        raise ValueError("surrogate ensemble_size is outside the supported range")
    expected = {"model.json"} | {
        f"member_{member:02d}_{target}.txt"
        for member in range(ensemble_size)
        for target in STATE_FEATURES
    }
    if set(files) != expected:
        raise ValueError("surrogate artifact file set does not match model.json")
    return files, metadata


@dataclass(frozen=True)
class ScenarioTrajectory:
    """One indivisible simulator trajectory used for grouped splitting.

    ``states`` has shape ``[month, well, 3]`` and ``actions`` has shape
    ``[month, well, 3]``. Target codes are ORAT=0, LRAT=1 and WRAT=2. The
    action at index ``t`` drives ``states[t + 1]``.
    ``source_model`` must explicitly identify simulator provenance.
    """

    scenario_id: str
    source_model: str
    dates: pd.DatetimeIndex
    well_ids: tuple[str, ...]
    states: np.ndarray
    actions: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dates = pd.DatetimeIndex(self.dates)
        states = np.asarray(self.states, dtype=float)
        actions = np.asarray(self.actions, dtype=float)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)

        if not self.scenario_id or not self.source_model:
            raise ValueError("scenario_id and source_model are required")
        if len(dates) < 2 or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(f"scenario {self.scenario_id!r}: dates must be unique and increasing")
        periods = dates.to_period("M").astype(int)
        if not np.all(np.diff(periods) == 1):
            raise ValueError(f"scenario {self.scenario_id!r}: monthly trajectory has gaps")
        expected_states = (len(dates), len(self.well_ids), len(STATE_FEATURES))
        expected_actions = (len(dates), len(self.well_ids), len(ACTION_FEATURES))
        if states.shape != expected_states or actions.shape != expected_actions:
            raise ValueError(
                f"scenario {self.scenario_id!r}: expected states {expected_states} and "
                f"actions {expected_actions}, got {states.shape} and {actions.shape}"
            )
        if len(set(self.well_ids)) != len(self.well_ids):
            raise ValueError(f"scenario {self.scenario_id!r}: duplicate wells")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"scenario {self.scenario_id!r}: non-finite values")
        oil, liquid = states[..., 0], states[..., 1]
        if np.any(oil < 0) or np.any(liquid < oil) or np.any(states[..., 2] < 0):
            raise ValueError(f"scenario {self.scenario_id!r}: physical state invariants violated")
        if (
            np.any(actions[..., 0] < 0)
            or not np.isin(actions[..., 1], (0.0, 1.0, 2.0)).all()
            or not np.isin(actions[..., 2], (0.0, 1.0)).all()
        ):
            raise ValueError(f"scenario {self.scenario_id!r}: invalid control target/value/status")

    @property
    def content_hash(self) -> str:
        digest = sha256()
        digest.update(_canonical_json({
            "scenario_id": self.scenario_id,
            "source_model": self.source_model,
            "dates": [value.isoformat() for value in self.dates],
            "well_ids": self.well_ids,
            "metadata": self.metadata,
        }))
        digest.update(np.asarray(self.states, dtype="<f8").tobytes())
        digest.update(np.asarray(self.actions, dtype="<f8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class StepPrediction:
    mean: np.ndarray
    std: np.ndarray
    interval_half_width: np.ndarray
    ood_score: float
    ood: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RolloutPrediction:
    mean: np.ndarray
    std: np.ndarray
    interval_half_width: np.ndarray
    ood_score: np.ndarray
    ood: np.ndarray
    reasons: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class OptimizerPrediction:
    mean: np.ndarray
    std: np.ndarray
    interval_half_width: np.ndarray
    ood_score: np.ndarray
    ood: np.ndarray
    accepted: np.ndarray


@runtime_checkable
class StatefulSurrogate(Protocol):
    """Protocol consumed by the optimizer-facing Track 2 world model."""

    def step(self, state: np.ndarray, action: np.ndarray) -> StepPrediction: ...

    def rollout(self, initial_state: np.ndarray, actions: np.ndarray) -> RolloutPrediction: ...

    def predict(self, initial_state: np.ndarray, action_sequences: np.ndarray) -> OptimizerPrediction: ...


@dataclass(frozen=True)
class PhysicalBaseline:
    """Small CRM/material-balance transition used before learned residuals."""

    liquid_decay: float
    injection_gain: float
    oil_fraction_retention: float
    pressure_balance_gain: float
    pressure_mean_reversion: float
    pressure_center: float

    @classmethod
    def fit(cls, trajectories: list[ScenarioTrajectory]) -> PhysicalBaseline:
        previous = np.concatenate([item.states[:-1].reshape(-1, 3) for item in trajectories])
        following = np.concatenate([item.states[1:].reshape(-1, 3) for item in trajectories])
        actions = np.concatenate([
            item.actions[:-1].reshape(-1, len(ACTION_FEATURES)) for item in trajectories
        ])
        active = actions[:, 2] > 0.5
        if active.sum() < 2:
            raise ValueError("training trajectories contain too few active well transitions")
        injection = np.where(actions[:, 1] == 2.0, actions[:, 0], 0.0)

        liq_x = np.column_stack([previous[:, 1], injection])
        liq_coef, *_ = np.linalg.lstsq(liq_x[active], following[active, 1], rcond=None)
        liquid_decay = float(np.clip(liq_coef[0], 0.0, 1.2))
        injection_gain = float(np.clip(liq_coef[1], 0.0, None))

        prev_fraction = np.divide(
            previous[:, 0], previous[:, 1], out=np.zeros(len(previous)), where=previous[:, 1] > 1e-9
        )
        next_fraction = np.divide(
            following[:, 0], following[:, 1], out=np.zeros(len(following)), where=following[:, 1] > 1e-9
        )
        fraction_mask = active & (prev_fraction > 1e-5)
        retention = (
            np.median(next_fraction[fraction_mask] / prev_fraction[fraction_mask])
            if fraction_mask.any()
            else 1.0
        )

        pressure_center = float(np.median(previous[:, 2]))
        pressure_x = np.column_stack([
            injection - previous[:, 1],
            pressure_center - previous[:, 2],
        ])
        pressure_coef, *_ = np.linalg.lstsq(
            pressure_x[active], following[active, 2] - previous[active, 2], rcond=None
        )
        return cls(
            liquid_decay=liquid_decay,
            injection_gain=injection_gain,
            oil_fraction_retention=float(np.clip(retention, 0.0, 1.2)),
            pressure_balance_gain=float(np.clip(pressure_coef[0], 0.0, None)),
            pressure_mean_reversion=float(np.clip(pressure_coef[1], 0.0, 1.0)),
            pressure_center=pressure_center,
        )

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = np.asarray(state, float)
        action = np.asarray(action, float)
        control, target, status = action.T
        active = status >= 0.5
        injection = np.where(target == 2.0, control, 0.0)
        liquid = self.liquid_decay * state[:, 1] + self.injection_gain * injection
        fraction = np.divide(
            state[:, 0], state[:, 1], out=np.zeros(len(state)), where=state[:, 1] > 1e-9
        )
        oil = liquid * np.clip(fraction * self.oil_fraction_retention, 0.0, 1.0)
        pressure = (
            state[:, 2]
            + self.pressure_balance_gain * (injection - state[:, 1])
            + self.pressure_mean_reversion * (self.pressure_center - state[:, 2])
        )
        result = np.column_stack([oil, liquid, np.maximum(pressure, 0.0)])
        result[~active, :2] = 0.0
        return result

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def _residual_features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    oil, liquid, pressure = np.asarray(state, float).T
    control, target, status = np.asarray(action, float).T
    injection = np.where(target == 2.0, control, 0.0)
    fraction = np.divide(oil, liquid, out=np.zeros_like(oil), where=liquid > 1e-9)
    return np.column_stack([
        oil,
        liquid,
        pressure,
        control,
        target == 0.0,
        target == 1.0,
        target == 2.0,
        status,
        fraction,
        1.0 - fraction,
        injection - liquid,
    ])


def _project_physics(raw: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
    projected = np.asarray(raw, float).copy()
    scale = np.maximum(np.abs(projected[..., 1]), 1.0)
    violations = (
        (projected[..., 0] < -1e-6)
        | (projected[..., 1] < -1e-6)
        | (projected[..., 0] > projected[..., 1] + 0.05 * scale)
        | (projected[..., 2] < -1e-6)
    )
    projected[..., 1] = np.maximum(projected[..., 1], 0.0)
    projected[..., 0] = np.clip(projected[..., 0], 0.0, projected[..., 1])
    projected[..., 2] = np.maximum(projected[..., 2], 0.0)
    inactive = np.asarray(action)[..., 2] < 0.5
    projected[..., 0] = np.where(inactive, 0.0, projected[..., 0])
    projected[..., 1] = np.where(inactive, 0.0, projected[..., 1])
    return projected, float(np.mean(violations))


class Track2Surrogate:
    """Ensemble residual surrogate with stateful rollout and OOD gating."""

    def __init__(
        self,
        baseline: PhysicalBaseline,
        boosters: list[list[lgb.Booster]],
        feature_min: np.ndarray,
        feature_max: np.ndarray,
        feature_scale: np.ndarray,
        state_scale: np.ndarray,
        residual_scale: np.ndarray,
        *,
        seed: int,
        n_estimators: int,
        ood_feature_margin: float,
        ood_disagreement: float,
        conformal_level: float | None = None,
        conformal_scale: np.ndarray | None = None,
        conformal_floor_fraction: float = 0.01,
        training_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.baseline = baseline
        self.boosters = boosters
        self.feature_min = np.asarray(feature_min, float)
        self.feature_max = np.asarray(feature_max, float)
        self.feature_scale = np.asarray(feature_scale, float)
        self.state_scale = np.asarray(state_scale, float)
        self.residual_scale = np.asarray(residual_scale, float)
        self.seed = seed
        self.n_estimators = n_estimators
        self.ood_feature_margin = float(ood_feature_margin)
        self.ood_disagreement = float(ood_disagreement)
        self.conformal_level = (
            None if conformal_level is None else float(conformal_level)
        )
        self.conformal_scale = (
            None if conformal_scale is None else np.asarray(conformal_scale, float)
        )
        self.conformal_floor_fraction = float(conformal_floor_fraction)
        self.training_metadata = training_metadata or {}
        if (
            self.feature_min.shape != (len(RESIDUAL_FEATURES),)
            or self.feature_max.shape != (len(RESIDUAL_FEATURES),)
            or self.feature_scale.shape != (len(RESIDUAL_FEATURES),)
            or self.state_scale.shape != (len(STATE_FEATURES),)
            or self.residual_scale.shape != (len(STATE_FEATURES),)
            or (
                self.conformal_scale is not None
                and self.conformal_scale.shape != (len(STATE_FEATURES),)
            )
        ):
            raise ValueError("surrogate parameter shape mismatch")
        arrays = (
            self.feature_min,
            self.feature_max,
            self.feature_scale,
            self.state_scale,
            self.residual_scale,
            *(() if self.conformal_scale is None else (self.conformal_scale,)),
        )
        if not all(np.isfinite(value).all() for value in arrays) or not np.isfinite(
            [
                self.ood_feature_margin,
                self.ood_disagreement,
                self.conformal_floor_fraction,
            ]
        ).all():
            raise ValueError("surrogate parameters must be finite")
        if (
            np.any(self.feature_min > self.feature_max)
            or np.any(self.feature_scale <= 0)
            or np.any(self.state_scale <= 0)
            or np.any(self.residual_scale < 0)
            or np.any(self.residual_scale > 1)
            or self.ood_feature_margin <= 0
            or self.ood_disagreement <= 0
            or self.conformal_floor_fraction <= 0
            or (self.conformal_level is None) != (self.conformal_scale is None)
            or (
                self.conformal_level is not None
                and not 0.0 < self.conformal_level < 1.0
            )
            or (
                self.conformal_scale is not None
                and np.any(self.conformal_scale <= 0)
            )
        ):
            raise ValueError("surrogate scales, bounds, or OOD thresholds are invalid")

    @property
    def ensemble_size(self) -> int:
        return len(self.boosters)

    @property
    def is_calibrated(self) -> bool:
        return self.conformal_scale is not None

    def _interval_half_width(self, std: np.ndarray) -> np.ndarray:
        if self.conformal_scale is None:
            return np.asarray(std, float)
        floor = self.state_scale * self.conformal_floor_fraction
        return np.maximum(np.asarray(std, float), floor) * self.conformal_scale

    def _apply_conformal_calibration(
        self,
        *,
        level: float,
        scale: np.ndarray,
        floor_fraction: float,
    ) -> None:
        scale = np.asarray(scale, float)
        if (
            not 0.0 < level < 1.0
            or scale.shape != (len(STATE_FEATURES),)
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or not np.isfinite(floor_fraction)
            or floor_fraction <= 0
        ):
            raise ValueError("invalid conformal calibration")
        self.conformal_level = float(level)
        self.conformal_scale = scale
        self.conformal_floor_fraction = float(floor_fraction)

    @classmethod
    def fit(
        cls,
        trajectories: list[ScenarioTrajectory],
        *,
        ensemble_size: int = 5,
        n_estimators: int = 160,
        seed: int = 42,
        ood_feature_margin: float = 3.0,
        ood_disagreement: float = 0.5,
    ) -> Track2Surrogate:
        if not trajectories:
            raise ValueError("at least one training scenario is required")
        if not 2 <= ensemble_size <= _MAX_ARTIFACT_ENSEMBLE_SIZE:
            raise ValueError(
                f"ensemble_size must be between 2 and {_MAX_ARTIFACT_ENSEMBLE_SIZE}"
            )
        well_ids = trajectories[0].well_ids
        if any(item.well_ids != well_ids for item in trajectories[1:]):
            raise ValueError("all training scenarios must use the same ordered wells")

        baseline = PhysicalBaseline.fit(trajectories)
        transitions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for item in trajectories:
            state = item.states[:-1].reshape(-1, len(STATE_FEATURES))
            action = item.actions[:-1].reshape(-1, len(ACTION_FEATURES))
            following = item.states[1:].reshape(-1, len(STATE_FEATURES))
            transitions[item.scenario_id] = (
                _residual_features(state, action), following - baseline.predict(state, action)
            )

        all_x = np.concatenate([value[0] for value in transitions.values()])
        all_states = np.concatenate([item.states.reshape(-1, 3) for item in trajectories])
        feature_scale = np.maximum(np.std(all_x, axis=0), 1e-6)
        state_scale = np.maximum(np.median(np.abs(all_states), axis=0), 1.0)
        rng = np.random.default_rng(seed)
        scenario_ids = np.array(sorted(transitions))
        boosters: list[list[lgb.Booster]] = []
        sampled_scenarios: list[set[str]] = []
        for member in range(ensemble_size):
            sampled = rng.choice(scenario_ids, size=len(scenario_ids), replace=True)
            sampled_scenarios.append(set(sampled.tolist()))
            member_x = np.concatenate([transitions[str(key)][0] for key in sampled])
            member_y = np.concatenate([transitions[str(key)][1] for key in sampled])
            targets: list[lgb.Booster] = []
            for target in range(len(STATE_FEATURES)):
                params = {
                    "objective": "regression_l1",
                    "learning_rate": 0.04,
                    "num_leaves": 15,
                    "min_data_in_leaf": 12,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.9,
                    "bagging_freq": 1,
                    "verbosity": -1,
                    "num_threads": 1,
                    "deterministic": True,
                    "force_col_wise": True,
                    "seed": seed + member * 17 + target,
                }
                targets.append(lgb.train(
                    params,
                    lgb.Dataset(member_x, label=member_y[:, target], feature_name=list(RESIDUAL_FEATURES)),
                    num_boost_round=n_estimators,
                ))
            boosters.append(targets)

        oob_truth, oob_prediction = [], []
        for scenario_id, (features, residual) in transitions.items():
            available = [
                member for member, sampled in enumerate(sampled_scenarios)
                if scenario_id not in sampled
            ]
            if not available:
                continue
            predictions = np.stack([
                np.column_stack([booster.predict(features, num_threads=1) for booster in boosters[member]])
                for member in available
            ])
            oob_truth.append(residual)
            oob_prediction.append(predictions.mean(axis=0))
        residual_scale = np.zeros(len(STATE_FEATURES))
        if oob_truth:
            truth, prediction = np.concatenate(oob_truth), np.concatenate(oob_prediction)
            grid = np.linspace(0.0, 1.0, 21)
            for target in range(len(STATE_FEATURES)):
                losses = [np.abs(truth[:, target] - alpha * prediction[:, target]).mean() for alpha in grid]
                residual_scale[target] = grid[int(np.argmin(losses))]

        return cls(
            baseline,
            boosters,
            all_x.min(axis=0),
            all_x.max(axis=0),
            feature_scale,
            state_scale,
            residual_scale,
            seed=seed,
            n_estimators=n_estimators,
            ood_feature_margin=ood_feature_margin,
            ood_disagreement=ood_disagreement,
            training_metadata={
                "scenario_ids": sorted(transitions),
                "scenario_hashes": {item.scenario_id: item.content_hash for item in trajectories},
                "source_models": sorted({item.source_model for item in trajectories}),
            },
        )

    def _raw_member(self, member: int, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        features = _residual_features(state, action)
        residual = np.column_stack([
            booster.predict(features, num_threads=1) for booster in self.boosters[member]
        ])
        return self.baseline.predict(state, action) + residual * self.residual_scale

    def _diagnose(
        self,
        features: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        violation_fraction: float,
    ) -> tuple[float, bool, tuple[str, ...]]:
        below = np.maximum(self.feature_min - features, 0.0) / self.feature_scale
        above = np.maximum(features - self.feature_max, 0.0) / self.feature_scale
        feature_score = float(np.max(np.maximum(below, above)))
        scale_floor = np.maximum(
            self.state_scale, self.feature_scale[: len(STATE_FEATURES)]
        )
        disagreement = float(np.max(std / np.maximum(np.abs(mean), scale_floor)))
        reasons: list[str] = []
        if feature_score > self.ood_feature_margin:
            reasons.append("control_or_state_outside_training_range")
        if disagreement > self.ood_disagreement:
            reasons.append("ensemble_disagreement")
        if violation_fraction > 0.2:
            reasons.append("physical_projection_excess")
        score = max(
            feature_score / max(self.ood_feature_margin, 1e-9),
            disagreement / max(self.ood_disagreement, 1e-9),
            violation_fraction / 0.2,
        )
        return score, bool(reasons), tuple(reasons)

    def step(self, state: np.ndarray, action: np.ndarray) -> StepPrediction:
        state, action = self._validate_step(state, action)
        raw = np.stack([self._raw_member(member, state, action) for member in range(self.ensemble_size)])
        projected, violation = _project_physics(
            raw, np.broadcast_to(action, raw.shape[:-1] + (len(ACTION_FEATURES),))
        )
        mean, std = projected.mean(axis=0), projected.std(axis=0, ddof=1)
        score, ood, reasons = self._diagnose(_residual_features(state, action), mean, std, violation)
        return StepPrediction(
            mean, std, self._interval_half_width(std), score, ood, reasons
        )

    def rollout(self, initial_state: np.ndarray, actions: np.ndarray) -> RolloutPrediction:
        initial_state = np.asarray(initial_state, float)
        actions = np.asarray(actions, float)
        if actions.ndim != 3 or actions.shape[1:] != (len(initial_state), len(ACTION_FEATURES)):
            raise ValueError("actions must have shape [horizon, wells, 3]")
        self._validate_step(initial_state, actions[0])
        member_states = np.repeat(initial_state[None, ...], self.ensemble_size, axis=0)
        means, stds, scores, flags, reasons = [], [], [], [], []
        for action in actions:
            member_features = np.concatenate([
                _residual_features(member_states[member], action)
                for member in range(self.ensemble_size)
            ])
            raw = np.stack([
                self._raw_member(member, member_states[member], action)
                for member in range(self.ensemble_size)
            ])
            member_states, violation = _project_physics(
                raw, np.broadcast_to(action, raw.shape[:-1] + (len(ACTION_FEATURES),))
            )
            mean, std = member_states.mean(axis=0), member_states.std(axis=0, ddof=1)
            score, ood, why = self._diagnose(member_features, mean, std, violation)
            means.append(mean)
            stds.append(std)
            scores.append(score)
            flags.append(ood)
            reasons.append(why)
        std = np.stack(stds)
        return RolloutPrediction(
            np.stack(means),
            std,
            self._interval_half_width(std),
            np.asarray(scores),
            np.asarray(flags),
            tuple(reasons),
        )

    def predict(self, initial_state: np.ndarray, action_sequences: np.ndarray) -> OptimizerPrediction:
        """Batch prediction used by optimizers; ``accepted`` is the OOD gate."""
        sequences = np.asarray(action_sequences, float)
        if sequences.ndim != 4:
            raise ValueError("action_sequences must have shape [candidate, horizon, wells, 3]")
        rollouts = [self.rollout(initial_state, candidate) for candidate in sequences]
        flags = np.stack([item.ood for item in rollouts])
        return OptimizerPrediction(
            mean=np.stack([item.mean for item in rollouts]),
            std=np.stack([item.std for item in rollouts]),
            interval_half_width=np.stack(
                [item.interval_half_width for item in rollouts]
            ),
            ood_score=np.stack([item.ood_score for item in rollouts]),
            ood=flags,
            accepted=~flags.any(axis=1),
        )

    @staticmethod
    def _validate_step(state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state, action = np.asarray(state, float), np.asarray(action, float)
        if state.ndim != 2 or state.shape[1] != len(STATE_FEATURES):
            raise ValueError("state must have shape [wells, 3]")
        if action.shape != (len(state), len(ACTION_FEATURES)):
            raise ValueError("action must have shape [wells, 3]")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("state and action must be finite")
        if (
            np.any(action[:, 0] < 0)
            or not np.isin(action[:, 1], (0.0, 1.0, 2.0)).all()
            or not np.isin(action[:, 2], (0.0, 1.0)).all()
        ):
            raise ValueError("control value/target/status is invalid")
        return state, action

    def save(self, directory: Path | str) -> dict[str, Any]:
        """Save portable LightGBM text models and a verified manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "state_features": STATE_FEATURES,
            "action_features": ACTION_FEATURES,
            "residual_features": RESIDUAL_FEATURES,
            "ensemble_size": self.ensemble_size,
            "n_estimators": self.n_estimators,
            "seed": self.seed,
            "ood_feature_margin": self.ood_feature_margin,
            "ood_disagreement": self.ood_disagreement,
            "physical_baseline": self.baseline.as_dict(),
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "state_scale": self.state_scale.tolist(),
            "residual_scale": self.residual_scale.tolist(),
            "conformal_level": self.conformal_level,
            "conformal_scale": (
                None if self.conformal_scale is None else self.conformal_scale.tolist()
            ),
            "conformal_floor_fraction": self.conformal_floor_fraction,
            "training": self.training_metadata,
        }
        metadata_path = directory / "model.json"
        metadata_path.write_bytes(_canonical_json(metadata) + b"\n")
        paths = [metadata_path]
        for member, targets in enumerate(self.boosters):
            for target, booster in zip(STATE_FEATURES, targets, strict=True):
                path = directory / f"member_{member:02d}_{target}.txt"
                booster.save_model(path)
                paths.append(path)
        hashes = {path.name: _sha256_file(path) for path in sorted(paths)}
        manifest = {
            "artifact_type": "timesoil_track2_surrogate",
            "schema_version": SCHEMA_VERSION,
            "files": hashes,
            "artifact_hash": sha256(_canonical_json(hashes)).hexdigest(),
        }
        (directory / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        return manifest

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> Track2Surrogate:
        directory = Path(directory)
        manifest_bytes = _read_regular_bytes(
            directory / "manifest.json", "surrogate manifest"
        )
        if (
            expected_manifest_sha256 is not None
            and sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
        ):
            raise ValueError("surrogate manifest failed pinned hash check")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("surrogate manifest is not valid UTF-8 JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError("surrogate manifest must be a JSON object")
        model_bytes = _read_regular_bytes(
            directory / "model.json", "surrogate artifact file model.json"
        )
        hashes, metadata = _validated_surrogate_artifact(manifest, model_bytes)
        artifact_bytes: dict[str, bytes] = {}
        for name, expected in hashes.items():
            path = directory / name
            data = model_bytes if name == "model.json" else _read_regular_bytes(
                path, f"surrogate artifact file {name}"
            )
            if sha256(data).hexdigest() != expected:
                raise ValueError(f"surrogate artifact file failed hash check: {name}")
            artifact_bytes[name] = data
        try:
            booster_text = {
                name: data.decode("utf-8")
                for name, data in artifact_bytes.items()
                if name != "model.json"
            }
        except UnicodeError as exc:
            raise ValueError("surrogate booster is not valid UTF-8 text") from exc
        if tuple(metadata.get("state_features", ())) != STATE_FEATURES:
            raise ValueError("state feature contract mismatch")
        if tuple(metadata.get("action_features", ())) != ACTION_FEATURES:
            raise ValueError("action feature contract mismatch")
        if tuple(metadata.get("residual_features", ())) != RESIDUAL_FEATURES:
            raise ValueError("residual feature contract mismatch")
        boosters = [
            [
                lgb.Booster(
                    model_str=booster_text[f"member_{member:02d}_{target}.txt"]
                )
                for target in STATE_FEATURES
            ]
            for member in range(metadata["ensemble_size"])
        ]
        return cls(
            PhysicalBaseline(**metadata["physical_baseline"]),
            boosters,
            np.asarray(metadata["feature_min"]),
            np.asarray(metadata["feature_max"]),
            np.asarray(metadata["feature_scale"]),
            np.asarray(metadata["state_scale"]),
            np.asarray(metadata["residual_scale"]),
            seed=int(metadata["seed"]),
            n_estimators=int(metadata["n_estimators"]),
            ood_feature_margin=float(metadata["ood_feature_margin"]),
            ood_disagreement=float(metadata["ood_disagreement"]),
            conformal_level=metadata.get("conformal_level"),
            conformal_scale=metadata.get("conformal_scale"),
            conformal_floor_fraction=float(
                metadata.get("conformal_floor_fraction", 0.01)
            ),
            training_metadata=metadata.get("training", {}),
        )
