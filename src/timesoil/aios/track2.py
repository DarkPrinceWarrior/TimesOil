"""Scenario dataset, training and evaluation pipeline for Track 2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from math import ceil, fsum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import rmse, wape
from .contracts import ControlAction, ControlTarget, WellRole, WellStatus
from .opm import OPM_IMAGE_DIGEST, OpmSummaryError, verify_summary_extraction
from .scenario_generation import ScenarioGeneratorConfig, generate_control_scenarios
from .schedule_overlay import _canonical_schedule
from .surrogate import (
    ACTION_FEATURES,
    STATE_FEATURES,
    PhysicalBaseline,
    ScenarioTrajectory,
    Track2Surrogate,
)


CANONICAL_COLUMNS = (
    "scenario_id",
    "source_model",
    "date",
    "well",
    *STATE_FEATURES,
    "control_value",
    "control_target",
    "status",
)
CONTROL_TARGET_CODES = {"ORAT": 0.0, "LRAT": 1.0, "WRAT": 2.0}
MODEL_Z_SOURCE_SHA256 = "4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa"
MODEL_Z_SCENARIO_INDEX_SHA256 = (
    "69697fede3bafe9fd50f7ba568a7aaec3d2f98a9726fde94595feea82f10e317"
)
MODEL_Z_SCENARIO_ACTIONS_SHA256 = (
    ("baseline", "826a77efad0c8dc848b3e26f1da70ea00ce34b19e30cf69df574348899ef4c85"),
    ("perturbation-001", "4b5a1cdef9c00eb633b91d1acf0750f11afc0a5b56a0359d6bcd1cde952c3cb4"),
    ("perturbation-002", "951a90621ab3705fcd0cbe97ae5ddfe4866cb570a31589f014fe68038ebd1b53"),
    ("perturbation-003", "8f298be7162158bc785933be1b922e7358e4ebdae251311c7a2af381ab1031dd"),
    ("perturbation-004", "876c355b40ad8222a96bbb1c2104b538e3808795970ad41dd6247f9ecd603c43"),
    ("perturbation-005", "1fa828a1b4a646eb609ca9d29dd34f7b75142ee3c22d0fca71773c508ebf8de6"),
    ("perturbation-006", "562659552e8609871de4b973e68e889ce670ed793d927200b8f909453f8c42a4"),
    ("perturbation-007", "a5b9aead89a8f35a38b88734b23bb89a4277c2577062fb8d3c002c53ebd210ff"),
    ("perturbation-008", "b2a4ab007987076a41d7873af7b5d0d9ccc77f7fb73767812a333780a83edbe5"),
    ("perturbation-009", "ff914a5c5cf47b3568e4f01c345ce65c9d8ce1e3053159cd033ec42337d761c8"),
)
MODEL_Z_SCENARIO_EXPORTED_ACTIONS_SHA256 = (
    ("baseline", "826a77efad0c8dc848b3e26f1da70ea00ce34b19e30cf69df574348899ef4c85"),
    ("perturbation-001", "b560f191784d7f56c207e0c80a2a98aa57ac1d5c7244bd73c48e888bd15597e5"),
    ("perturbation-002", "ff81b7fc4644926be9f0b0865be4b4def30b0b58c3938e2048eb8098c7667c5f"),
    ("perturbation-003", "e9a28dc2b2f9835aa5d363f42c842575d3c9bbe1b549e5941ae56de677cb441c"),
    ("perturbation-004", "3a32f181e8ed550a60bdc8f03cdf75184048ee8ef2400d3d07e908a50441c369"),
    ("perturbation-005", "ab9d1b23a3f3ae9fb467cac0b111e089d6d84df7d6df3da2a1d4e4962e463bad"),
    ("perturbation-006", "cdddb50851ccbc0762833aebd8da7fd846aa21a3261ab81d8c3fe68288425b47"),
    ("perturbation-007", "f07434f25a44537166113d3424082da1c1b69e78febea0e5d81ba25d99674566"),
    ("perturbation-008", "553467f9590a7f48537a96d14b749a406ad9e365755c85594d8932ca6f30aadf"),
    ("perturbation-009", "d059a5832beee263957566ce06f7c79de2e64862a6358a1c39a6ad21884a1ee9"),
)
_MODEL_Z_SOURCE_SHA256 = MODEL_Z_SOURCE_SHA256
MAX_TRACK2_SEARCH_CANDIDATES = 500


class _VerifiedTrajectoryDataset(list[ScenarioTrajectory]):
    def __init__(
        self, trajectories: list[ScenarioTrajectory], *, model_z_identity: bool
    ) -> None:
        super().__init__(trajectories)
        self.scenario_hashes = tuple(item.content_hash for item in trajectories)
        self.model_z_identity = model_z_identity


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _linked_file(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is required")
    path = (base / value).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path


def _verify_run_artifact(
    manifest_path: Path,
    artifacts: Any,
    relative_path: Any,
    expected_digest: str,
    label: str,
) -> None:
    if not isinstance(artifacts, list) or not isinstance(relative_path, str):
        raise ValueError(f"OPM {label} artifact contract is invalid")
    matches = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and (item["path"] == relative_path or item["path"].endswith(f"/{relative_path}"))
        and item.get("sha256") == expected_digest
    ]
    if len(matches) != 1:
        raise ValueError(f"OPM {label} artifact hash is missing or ambiguous")
    artifact_name = matches[0]["path"]
    artifact = Path(artifact_name)
    if artifact.is_absolute() or ".." in artifact.parts:
        raise ValueError(f"OPM {label} artifact path is unsafe")
    artifact_path = _linked_file(manifest_path.parent, artifact_name, f"OPM {label}")
    if _sha256_file(artifact_path) != expected_digest:
        raise ValueError(f"OPM {label} artifact hash mismatch")


def _verify_export_manifest(
    dataset_path: Path,
    frame: pd.DataFrame,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    summary_run: Any = None,
) -> bool:
    if manifest.get("schema_version") != 1 or manifest.get("generator") != "timesoil.aios.opm_chdd":
        raise ValueError("unsupported Track 2 export manifest")
    try:
        provenance = manifest["provenance"]
        source = manifest["source"]
        scenario = manifest["scenario"]
        output = manifest["outputs"]["track2_csv"]
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete Track 2 export manifest") from exc
    if not all(isinstance(item, dict) for item in (provenance, source, scenario, output)):
        raise ValueError("invalid Track 2 export manifest objects")

    if output.get("name") != dataset_path.name:
        raise ValueError("Track 2 dataset filename does not match its manifest")
    if output.get("row_count") != len(frame) or isinstance(output.get("row_count"), bool):
        raise ValueError("Track 2 dataset row count does not match its manifest")
    if _sha256_file(dataset_path) != _digest(output.get("sha256"), "Track 2 dataset"):
        raise ValueError("Track 2 dataset hash mismatch")
    scenario_ids = frame["scenario_id"].astype(str).unique()
    source_models = frame["source_model"].astype(str).unique()
    if (
        len(scenario_ids) != 1
        or len(source_models) != 1
        or scenario.get("scenario_id") != scenario_ids[0]
        or scenario.get("source_model") != source_models[0]
    ):
        raise ValueError("Track 2 scenario does not match its manifest")

    summary_path = _linked_file(manifest_path.parent, source.get("summary_csv"), "OPM summary CSV")
    if _sha256_file(summary_path) != _digest(source.get("summary_csv_sha256"), "OPM summary CSV"):
        raise ValueError("OPM summary CSV hash mismatch")
    _digest(source.get("deck_sha256"), "OPM source deck")

    run_manifest_path = _linked_file(
        manifest_path.parent, provenance.get("opm_run_manifest"), "OPM run manifest"
    )
    if _sha256_file(run_manifest_path) != _digest(
        provenance.get("opm_run_manifest_sha256"), "OPM run manifest"
    ):
        raise ValueError("OPM run manifest hash mismatch")
    extraction_manifest_path = _linked_file(
        manifest_path.parent,
        provenance.get("summary_extraction_manifest"),
        "OPM summary extraction manifest",
    )
    if _sha256_file(extraction_manifest_path) != _digest(
        provenance.get("summary_extraction_manifest_sha256"),
        "OPM summary extraction manifest",
    ):
        raise ValueError("OPM summary extraction manifest hash mismatch")
    try:
        run_manifest = verify_summary_extraction(
            summary_path,
            extraction_manifest_path,
            run_manifest_path,
            _summary_run=summary_run,
        )
    except OpmSummaryError as exc:
        raise ValueError(f"unverified OPM summary extraction: {exc}") from exc
    if (
        run_manifest.get("schema") != "timesoil.aios.opm-run/v1"
        or run_manifest.get("status") != "success"
        or run_manifest.get("returncode") != 0
        or isinstance(run_manifest.get("returncode"), bool)
        or run_manifest.get("image_digest") != OPM_IMAGE_DIGEST
    ):
        raise ValueError("OPM run provenance is not a successful pinned-image run")
    run_source_digest = _digest(run_manifest.get("source_sha256"), "OPM case source")
    if run_source_digest != _digest(
        provenance.get("opm_source_sha256"), "exported OPM case source"
    ):
        raise ValueError("OPM case source hash does not match its export manifest")
    summary_contract = run_manifest.get("summary_contract")
    if not isinstance(summary_contract, dict):
        raise ValueError("OPM summary contract is missing")
    deck_digest = _digest(run_manifest.get("deck_sha256"), "OPM deck")
    overlay_digest = _digest(summary_contract.get("overlay_sha256"), "OPM summary overlay")
    _verify_run_artifact(
        run_manifest_path, run_manifest.get("artifacts"), run_manifest.get("deck"), deck_digest, "deck"
    )
    _verify_run_artifact(
        run_manifest_path,
        run_manifest.get("artifacts"),
        summary_contract.get("overlay"),
        overlay_digest,
        "summary overlay",
    )
    return run_source_digest == _MODEL_Z_SOURCE_SHA256 and source_models[0] == "model_z_opm"


@dataclass(frozen=True)
class TrainingRun:
    model: Track2Surrogate
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    physical_train_metrics: dict[str, float | int]
    physical_test_metrics: dict[str, float | int]
    calibration_metrics: dict[str, Any] | None
    dataset_hash: str
    source_models: tuple[str, ...]
    model_z_ready: bool

    def report(self) -> dict[str, Any]:
        return {
            "dataset_contract": "scenario_trajectory_v1",
            "dataset_hash": self.dataset_hash,
            "source_models": self.source_models,
            "model_z_ready": self.model_z_ready,
            "pipeline_proof_only": not self.model_z_ready,
            "architecture": "CRM/material-balance baseline + LightGBM residual ensemble",
            "training_objective": "per-output L1 residual with train-only OOB shrinkage",
            "uncertainty": (
                "scenario-level LOSO cross-conformal intervals"
                if self.calibration_metrics is not None
                else "ensemble standard deviation; not conformally calibrated"
            ),
            "train_scenarios": self.train_ids,
            "test_scenarios": self.test_ids,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "physical_baseline_train_metrics": self.physical_train_metrics,
            "physical_baseline_test_metrics": self.physical_test_metrics,
            "conformal_calibration": self.calibration_metrics,
        }


def trajectory_from_frame(frame: pd.DataFrame) -> ScenarioTrajectory:
    """Validate one canonical long-frame scenario and convert it to arrays."""
    missing = set(CANONICAL_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"trajectory columns missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("trajectory is empty")
    scenario_ids = frame["scenario_id"].astype(str).unique()
    source_models = frame["source_model"].astype(str).unique()
    if len(scenario_ids) != 1 or len(source_models) != 1:
        raise ValueError("one frame must contain one scenario_id and one source_model")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["well"] = data["well"].astype(str)
    if data.duplicated(["date", "well"]).any():
        raise ValueError(f"scenario {scenario_ids[0]!r}: duplicate date × well")
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    wells = tuple(sorted(data["well"].unique()))
    if len(data) != len(dates) * len(wells):
        raise ValueError(f"scenario {scenario_ids[0]!r}: date × well grid is incomplete")

    def cube(columns: tuple[str, ...]) -> np.ndarray:
        matrices = [
            data.pivot(index="date", columns="well", values=column).reindex(index=dates, columns=wells)
            for column in columns
        ]
        return np.stack([matrix.to_numpy(float) for matrix in matrices], axis=-1)

    targets = data["control_target"].astype(str).str.upper().map(CONTROL_TARGET_CODES)
    if targets.isna().any():
        invalid = sorted(data.loc[targets.isna(), "control_target"].astype(str).unique())
        raise ValueError(f"unsupported control_target values: {invalid}")
    data["control_target_code"] = targets
    actions = np.stack([
        data.pivot(index="date", columns="well", values=column)
        .reindex(index=dates, columns=wells)
        .to_numpy(float)
        for column in ACTION_FEATURES
    ], axis=-1)
    return ScenarioTrajectory(
        scenario_id=str(scenario_ids[0]),
        source_model=str(source_models[0]),
        dates=dates,
        well_ids=wells,
        states=cube(STATE_FEATURES),
        actions=actions,
        metadata={"contract": "canonical_long_frame_v1"},
    )


def load_trajectory_dataset(
    path: Path | str,
    *,
    manifest: Path | str,
    _summary_run: Any = None,
) -> list[ScenarioTrajectory]:
    """Load canonical trajectories with mandatory export provenance."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) + sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no CSV/Parquet trajectories in {path}")
    frames: list[pd.DataFrame] = []
    for file in files:
        frames.append(pd.read_parquet(file) if file.suffix.lower() == ".parquet" else pd.read_csv(file))
    model_z_identities: list[bool] = []
    manifest_path = Path(manifest)
    manifest_files = sorted(manifest_path.glob("*.json")) if manifest_path.is_dir() else [manifest_path]
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for candidate in manifest_files:
        value = _json_object(candidate, "Track 2 export manifest")
        if value.get("generator") != "timesoil.aios.opm_chdd":
            continue
        try:
            name = value["outputs"]["track2_csv"]["name"]
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete Track 2 export manifest") from exc
        if not isinstance(name, str) or name in manifests:
            raise ValueError("duplicate or invalid Track 2 dataset manifest")
        manifests[name] = (candidate, value)
    for file, frame in zip(files, frames, strict=True):
        if file.name not in manifests:
            raise ValueError(f"no provenance manifest for Track 2 dataset: {file.name}")
        candidate, value = manifests[file.name]
        model_z_identities.append(
            _verify_export_manifest(
                file,
                frame,
                candidate,
                value,
                summary_run=_summary_run,
            )
        )
    data = pd.concat(frames, ignore_index=True)
    trajectories = [trajectory_from_frame(group) for _, group in data.groupby("scenario_id", sort=True)]
    return _VerifiedTrajectoryDataset(
        trajectories, model_z_identity=all(model_z_identities)
    )


def split_scenarios(
    trajectories: list[ScenarioTrajectory], *, test_fraction: float = 0.25, seed: int = 42
) -> tuple[list[ScenarioTrajectory], list[ScenarioTrajectory]]:
    """Deterministic group split: complete scenarios never cross the boundary."""
    if len(trajectories) < 2:
        raise ValueError("at least two scenarios are required for train/test split")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    ids = [item.scenario_id for item in trajectories]
    if len(set(ids)) != len(ids):
        raise ValueError("scenario_id values must be unique")
    ordered = sorted(trajectories, key=lambda item: item.scenario_id)
    rng = np.random.default_rng(seed)
    test_count = min(len(ordered) - 1, max(1, round(len(ordered) * test_fraction)))
    test_indices = set(rng.choice(len(ordered), size=test_count, replace=False).tolist())
    train = [item for index, item in enumerate(ordered) if index not in test_indices]
    test = [item for index, item in enumerate(ordered) if index in test_indices]
    return train, test


def evaluate_rollouts(
    model: Track2Surrogate, trajectories: list[ScenarioTrajectory], *, horizon: int = 6
) -> dict[str, float | int]:
    """Closed-loop rollout metrics; origins are non-overlapping within scenario."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    truths, predictions, standard_deviations, half_widths = [], [], [], []
    rollout_oil, rollout_liquid, ood_steps = [], [], []
    for item in trajectories:
        item_horizon = min(horizon, len(item.dates) - 1)
        for origin in range(0, len(item.dates) - item_horizon, item_horizon):
            end = origin + item_horizon
            result = model.rollout(item.states[origin], item.actions[origin:end])
            truth = item.states[origin + 1 : end + 1]
            truths.append(truth)
            predictions.append(result.mean)
            standard_deviations.append(result.std)
            half_widths.append(result.interval_half_width)
            rollout_oil.append(wape(truth[..., 0], result.mean[..., 0]))
            rollout_liquid.append(wape(truth[..., 1], result.mean[..., 1]))
            ood_steps.extend(result.ood.tolist())
    if not truths:
        raise ValueError("no evaluable rollout windows")
    truth = np.concatenate([item.reshape(-1, 3) for item in truths])
    prediction = np.concatenate([item.reshape(-1, 3) for item in predictions])
    standard_deviation = np.concatenate(
        [item.reshape(-1, 3) for item in standard_deviations]
    )
    half_width = np.concatenate([item.reshape(-1, 3) for item in half_widths])
    covered = np.abs(truth - prediction) <= half_width
    raw_covered = np.abs(truth - prediction) <= 1.645 * standard_deviation
    metrics: dict[str, float | int] = {
        "oil_wape": wape(truth[:, 0], prediction[:, 0]),
        "liquid_wape": wape(truth[:, 1], prediction[:, 1]),
        "pressure_rmse_bar": rmse(truth[:, 2], prediction[:, 2]),
        "mean_rollout_oil_wape": float(np.mean(rollout_oil)),
        "mean_rollout_liquid_wape": float(np.mean(rollout_liquid)),
        "raw_ensemble_90_coverage": float(np.mean(raw_covered)),
        "ood_step_rate": float(np.mean(ood_steps)),
        "rollouts": len(truths),
        "points": len(truth),
        "horizon_months": horizon,
    }
    if model.is_calibrated:
        metrics.update({
            "conformal_interval_coverage": float(np.mean(covered)),
            "oil_interval_coverage": float(np.mean(covered[:, 0])),
            "liquid_interval_coverage": float(np.mean(covered[:, 1])),
            "pressure_interval_coverage": float(np.mean(covered[:, 2])),
            "oil_mean_interval_width_tpd": float(2.0 * np.mean(half_width[:, 0])),
            "liquid_mean_interval_width_tpd": float(2.0 * np.mean(half_width[:, 1])),
            "pressure_mean_interval_width_bar": float(2.0 * np.mean(half_width[:, 2])),
        })
    return metrics


def _conformal_rank(scenario_count: int, level: float) -> int:
    if not 0.0 < level < 1.0:
        raise ValueError("conformal_level must be between 0 and 1")
    minimum = ceil(1.0 / (1.0 - level) - 1e-12)
    if scenario_count < minimum:
        raise ValueError(
            f"strict {level:.0%} scenario-level conformal calibration requires "
            f"at least {minimum} whole scenarios; got {scenario_count}"
        )
    rank = ceil((scenario_count + 1) * level - 1e-12)
    if rank > scenario_count:
        raise ValueError("requested conformal level has no finite scenario-level quantile")
    return rank


def _loso_conformal_calibration(
    trajectories: list[ScenarioTrajectory],
    *,
    level: float,
    ensemble_size: int,
    n_estimators: int,
    horizon: int,
    seed: int,
    floor_fraction: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Leakage-free LOSO scores; one simultaneous score per whole scenario/target."""

    ordered = sorted(trajectories, key=lambda item: item.scenario_id)
    rank = _conformal_rank(len(ordered), level)
    scenario_scores: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    base_widths: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for fold, holdout in enumerate(ordered):
        training = [item for item in ordered if item.scenario_id != holdout.scenario_id]
        model = Track2Surrogate.fit(
            training,
            ensemble_size=ensemble_size,
            n_estimators=n_estimators,
            seed=seed + 1009 * (fold + 1),
        )
        fold_errors, fold_widths, fold_truths, fold_predictions = [], [], [], []
        item_horizon = min(horizon, len(holdout.dates) - 1)
        for origin in range(0, len(holdout.dates) - item_horizon, item_horizon):
            end = origin + item_horizon
            result = model.rollout(
                holdout.states[origin], holdout.actions[origin:end]
            )
            truth = holdout.states[origin + 1 : end + 1]
            fold_errors.append(np.abs(truth - result.mean))
            fold_truths.append(truth)
            fold_predictions.append(result.mean)
            fold_widths.append(
                np.maximum(result.std, model.state_scale * floor_fraction)
            )
        absolute_error = np.concatenate(
            [item.reshape(-1, len(STATE_FEATURES)) for item in fold_errors]
        )
        base_width = np.concatenate(
            [item.reshape(-1, len(STATE_FEATURES)) for item in fold_widths]
        )
        scenario_scores.append(np.max(absolute_error / base_width, axis=0))
        errors.append(absolute_error)
        base_widths.append(base_width)
        truths.append(np.concatenate(
            [item.reshape(-1, len(STATE_FEATURES)) for item in fold_truths]
        ))
        predictions.append(np.concatenate(
            [item.reshape(-1, len(STATE_FEATURES)) for item in fold_predictions]
        ))

    scores = np.stack(scenario_scores)
    scale = np.sort(scores, axis=0)[rank - 1]
    absolute_error = np.concatenate(errors)
    half_width = np.concatenate(base_widths) * scale
    covered = absolute_error <= half_width
    group_covered = scores <= scale
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    report = {
        "method": "scenario_loso_max_normalized_residual",
        "nominal_coverage": level,
        "scenario_count": len(ordered),
        "scenario_ids": [item.scenario_id for item in ordered],
        "quantile_rank": rank,
        "finite_sample_denominator": len(ordered) + 1,
        "simultaneous_within_scenario_per_target": True,
        "independent_validation": False,
        "independent_validation_note": (
            "coverage below is calibration-fold empirical coverage, not an independent claim"
        ),
        "floor_fraction_of_state_scale": floor_fraction,
        "scale_by_target": dict(zip(STATE_FEATURES, scale.tolist(), strict=True)),
        "calibration_group_coverage_by_target": dict(
            zip(STATE_FEATURES, group_covered.mean(axis=0).tolist(), strict=True)
        ),
        "calibration_point_coverage_by_target": dict(
            zip(STATE_FEATURES, covered.mean(axis=0).tolist(), strict=True)
        ),
        "loso_oil_wape": wape(truth[:, 0], prediction[:, 0]),
        "loso_liquid_wape": wape(truth[:, 1], prediction[:, 1]),
        "loso_pressure_rmse_bar": rmse(truth[:, 2], prediction[:, 2]),
        "mean_interval_width_by_target": dict(
            zip(STATE_FEATURES, (2.0 * half_width.mean(axis=0)).tolist(), strict=True)
        ),
        "median_interval_width_by_target": dict(
            zip(STATE_FEATURES, (2.0 * np.median(half_width, axis=0)).tolist(), strict=True)
        ),
        "scenario_scores": {
            item.scenario_id: dict(zip(STATE_FEATURES, score.tolist(), strict=True))
            for item, score in zip(ordered, scores, strict=True)
        },
    }
    return scale, report


def evaluate_physical_baseline(
    baseline: PhysicalBaseline, trajectories: list[ScenarioTrajectory], *, horizon: int = 6
) -> dict[str, float | int]:
    """Same closed-loop metric for the CRM/material-balance component alone."""
    truths, predictions, rollout_oil, rollout_liquid = [], [], [], []
    for item in trajectories:
        item_horizon = min(horizon, len(item.dates) - 1)
        for origin in range(0, len(item.dates) - item_horizon, item_horizon):
            end = origin + item_horizon
            state = item.states[origin]
            forecast = []
            for action in item.actions[origin:end]:
                state = baseline.predict(state, action)
                forecast.append(state)
            prediction = np.stack(forecast)
            truth = item.states[origin + 1 : end + 1]
            truths.append(truth)
            predictions.append(prediction)
            rollout_oil.append(wape(truth[..., 0], prediction[..., 0]))
            rollout_liquid.append(wape(truth[..., 1], prediction[..., 1]))
    truth = np.concatenate([item.reshape(-1, 3) for item in truths])
    prediction = np.concatenate([item.reshape(-1, 3) for item in predictions])
    return {
        "oil_wape": wape(truth[:, 0], prediction[:, 0]),
        "liquid_wape": wape(truth[:, 1], prediction[:, 1]),
        "pressure_rmse_bar": rmse(truth[:, 2], prediction[:, 2]),
        "mean_rollout_oil_wape": float(np.mean(rollout_oil)),
        "mean_rollout_liquid_wape": float(np.mean(rollout_liquid)),
        "rollouts": len(truths),
        "points": len(truth),
        "horizon_months": horizon,
    }


def fit_track2_surrogate(
    trajectories: list[ScenarioTrajectory],
    *,
    test_fraction: float = 0.25,
    ensemble_size: int = 5,
    n_estimators: int = 160,
    horizon: int = 6,
    seed: int = 42,
    conformal_level: float | None = None,
) -> TrainingRun:
    if conformal_level is not None:
        _conformal_rank(len(trajectories), conformal_level)
    train, test = split_scenarios(trajectories, test_fraction=test_fraction, seed=seed)
    evaluation_model = Track2Surrogate.fit(
        train, ensemble_size=ensemble_size, n_estimators=n_estimators, seed=seed
    )
    scenario_hashes = {item.scenario_id: item.content_hash for item in sorted(trajectories, key=lambda x: x.scenario_id)}
    dataset_hash = sha256(
        "".join(f"{key}:{value};" for key, value in scenario_hashes.items()).encode()
    ).hexdigest()
    train_metrics = evaluate_rollouts(evaluation_model, train, horizon=horizon)
    test_metrics = evaluate_rollouts(evaluation_model, test, horizon=horizon)
    physical_train_metrics = evaluate_physical_baseline(evaluation_model.baseline, train, horizon=horizon)
    physical_test_metrics = evaluate_physical_baseline(evaluation_model.baseline, test, horizon=horizon)
    calibration_metrics = None
    if conformal_level is not None:
        scale, calibration_metrics = _loso_conformal_calibration(
            trajectories,
            level=conformal_level,
            ensemble_size=ensemble_size,
            n_estimators=n_estimators,
            horizon=horizon,
            seed=seed,
        )
        model = Track2Surrogate.fit(
            trajectories,
            ensemble_size=ensemble_size,
            n_estimators=n_estimators,
            seed=seed,
        )
        model._apply_conformal_calibration(
            level=conformal_level, scale=scale, floor_fraction=0.01
        )
    else:
        model = evaluation_model
    source_models = tuple(sorted({item.source_model for item in trajectories}))
    model_z_ready = isinstance(trajectories, _VerifiedTrajectoryDataset) and (
        trajectories.model_z_identity
        and trajectories.scenario_hashes == tuple(item.content_hash for item in trajectories)
    )
    run = TrainingRun(
        model=model,
        train_ids=tuple(item.scenario_id for item in train),
        test_ids=tuple(item.scenario_id for item in test),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        physical_train_metrics=physical_train_metrics,
        physical_test_metrics=physical_test_metrics,
        calibration_metrics=calibration_metrics,
        dataset_hash=dataset_hash,
        source_models=source_models,
        model_z_ready=model_z_ready,
    )
    model.training_metadata.update(run.report())
    model.training_metadata["scenario_hashes"] = scenario_hashes
    return run


@dataclass(frozen=True)
class Track2SearchCandidate:
    """One surrogate-ranked schedule; never a simulator certificate."""

    candidate_id: str
    actions: tuple[ControlAction, ...]
    actions_sha256: str
    wells_schedule: str
    wells_schedule_sha256: str
    proxy_score: float
    predicted_oil_tonnes: float
    oil_uncertainty_tonnes: float
    injected_water_m3: float
    max_ood_score: float


@dataclass(frozen=True)
class Track2ScheduleSearch:
    """Bounded search receipt. Certification requires later OPM and CHDD receipts."""

    baseline_scenario_id: str
    start_date: date
    horizon_months: int
    requested_candidates: int
    accepted: tuple[Track2SearchCandidate, ...]
    rejected_ood: tuple[str, ...]
    selected: Track2SearchCandidate
    certified: bool = False


def _trajectory_controls(
    trajectory: ScenarioTrajectory,
    start_index: int = 0,
    horizon: int | None = None,
) -> tuple[ControlAction, ...]:
    target_codes = {
        0.0: ControlTarget.OIL_RATE,
        1.0: ControlTarget.LIQUID_RATE,
        2.0: ControlTarget.WATER_INJECTION_RATE,
    }
    actions: list[ControlAction] = []
    stop = None if horizon is None else start_index + horizon
    for offset, timestamp in enumerate(trajectory.dates[start_index:stop]):
        month = timestamp.date()
        for well_index, well in enumerate(trajectory.well_ids):
            value, code, status_code = trajectory.actions[start_index + offset, well_index]
            target = target_codes[float(code)]
            role = (
                WellRole.INJECTOR
                if target is ControlTarget.WATER_INJECTION_RATE
                else WellRole.PRODUCER
            )
            status = WellStatus.OPEN if status_code == 1.0 else WellStatus.SHUT
            actions.append(
                ControlAction(month, well, role, status, target, float(value))
            )
    return tuple(actions)


def _window_controls(
    trajectory: ScenarioTrajectory, start_index: int
) -> tuple[ControlAction, ...]:
    return _trajectory_controls(trajectory, start_index, 6)


def _action_cube(
    actions: tuple[ControlAction, ...],
    months: tuple[date, ...],
    wells: tuple[str, ...],
) -> np.ndarray:
    target_codes = {
        ControlTarget.OIL_RATE: 0.0,
        ControlTarget.LIQUID_RATE: 1.0,
        ControlTarget.WATER_INJECTION_RATE: 2.0,
    }
    indexed = {(item.month, item.well): item for item in actions}
    if len(indexed) != len(months) * len(wells):
        raise ValueError("candidate controls do not cover the complete month × well grid")
    cube = np.empty((len(months), len(wells), len(ACTION_FEATURES)), dtype=float)
    for month_index, month in enumerate(months):
        for well_index, well in enumerate(wells):
            try:
                action = indexed[month, well]
            except KeyError:
                raise ValueError(
                    f"candidate controls are missing {month.isoformat()} × {well}"
                ) from None
            cube[month_index, well_index] = (
                action.value,
                target_codes[action.target],
                1.0 if action.status is WellStatus.OPEN else 0.0,
            )
    return cube


def _injection_totals(actions: np.ndarray) -> np.ndarray:
    injected = (actions[..., 1] == 2.0) & (actions[..., 2] == 1.0)
    return np.asarray(
        [fsum(float(value) for value in row) for row in np.where(injected, actions[..., 0], 0.0)],
        dtype=float,
    )


def _validated_rollout(result: Any, *, horizon: int, wells: int) -> None:
    mean = np.asarray(result.mean, dtype=float)
    std = np.asarray(result.std, dtype=float)
    half_width = np.asarray(result.interval_half_width, dtype=float)
    scores = np.asarray(result.ood_score, dtype=float)
    ood = np.asarray(result.ood, dtype=bool)
    if (
        mean.shape != (horizon, wells, 3)
        or std.shape != mean.shape
        or half_width.shape != mean.shape
    ):
        raise ValueError("surrogate rollout shape disagrees with the search contract")
    if scores.shape != (horizon,) or ood.shape != (horizon,):
        raise ValueError("surrogate OOD output shape disagrees with the search contract")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or not np.isfinite(half_width).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("surrogate rollout contains non-finite values")
    if np.any(std < 0) or np.any(half_width <= 0):
        raise ValueError("surrogate rollout contains negative uncertainty")
    oil, liquid, pressure = mean[..., 0], mean[..., 1], mean[..., 2]
    if np.any(oil < 0) or np.any(liquid < oil) or np.any(pressure < 0):
        raise ValueError("surrogate rollout violates physical state invariants")


def search_track2_schedule(
    model: Track2Surrogate,
    trajectory: ScenarioTrajectory,
    *,
    start_index: int,
    candidate_count: int = 32,
    seed: int = 20260831,
    perturbation_fraction: float = 0.05,
    liquid_rate_scale: float = 1.0,
    uncertainty_weight: float = 1.0,
    injection_cost_equivalent: float = 0.01,
) -> Track2ScheduleSearch:
    """Rank six-month schedules with a risk-adjusted proxy, never as final CHDD."""

    if (
        isinstance(candidate_count, bool)
        or not 4 <= candidate_count <= MAX_TRACK2_SEARCH_CANDIDATES
    ):
        raise ValueError(
            f"candidate_count must be in [4, {MAX_TRACK2_SEARCH_CANDIDATES}]"
        )
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if start_index + 6 > len(trajectory.dates):
        raise ValueError("trajectory does not contain a complete six-month action window")
    if not np.isfinite(uncertainty_weight) or uncertainty_weight < 0:
        raise ValueError("uncertainty_weight must be finite and non-negative")
    if not np.isfinite(liquid_rate_scale) or liquid_rate_scale <= 0:
        raise ValueError("liquid_rate_scale must be finite and positive")
    if not np.isfinite(injection_cost_equivalent) or injection_cost_equivalent < 0:
        raise ValueError("injection_cost_equivalent must be finite and non-negative")

    baseline_actions = _window_controls(trajectory, start_index)
    generated = generate_control_scenarios(
        baseline_actions,
        ScenarioGeneratorConfig(
            scenario_count=candidate_count,
            seed=seed,
            perturbation_fraction=perturbation_fraction,
            liquid_rate_scale=liquid_rate_scale,
            perturb_injection=False,
        ),
    )
    months = tuple(
        timestamp.date()
        for timestamp in trajectory.dates[start_index : start_index + 6]
    )
    days = np.asarray([pd.Timestamp(month).days_in_month for month in months], float)
    baseline_cube = _action_cube(generated[0].actions, months, trajectory.well_ids)
    baseline_injectors = baseline_cube[..., 1] == 2.0
    accepted: list[Track2SearchCandidate] = []
    rejected_ood: list[str] = []

    for scenario in generated:
        cube = _action_cube(scenario.actions, months, trajectory.well_ids)
        candidate_injectors = cube[..., 1] == 2.0
        same_injectors = np.array_equal(candidate_injectors, baseline_injectors)
        same_injection_controls = np.array_equal(
            cube[baseline_injectors], baseline_cube[baseline_injectors]
        )
        if not same_injectors or not same_injection_controls:
            raise ValueError("candidate changed the baseline injection controls")
        rollout = model.rollout(trajectory.states[start_index], cube)
        _validated_rollout(rollout, horizon=6, wells=len(trajectory.well_ids))
        if np.asarray(rollout.ood, dtype=bool).any():
            if scenario.scenario_id == "baseline":
                raise ValueError("baseline schedule is outside the surrogate domain")
            rejected_ood.append(scenario.scenario_id)
            continue

        oil_tonnes = float(np.sum(np.asarray(rollout.mean)[..., 0] * days[:, None]))
        oil_uncertainty = float(
            np.sum(np.asarray(rollout.interval_half_width)[..., 0] * days[:, None])
        )
        injected_water = float(np.sum(_injection_totals(cube) * days))
        proxy_score = (
            oil_tonnes
            - uncertainty_weight * oil_uncertainty
            - injection_cost_equivalent * injected_water
        )
        if not np.isfinite(proxy_score):
            raise ValueError("candidate proxy score is not finite")
        schedule = _canonical_schedule(scenario.actions)
        accepted.append(
            Track2SearchCandidate(
                candidate_id=scenario.scenario_id,
                actions=scenario.actions,
                actions_sha256=scenario.sha256,
                wells_schedule=schedule,
                wells_schedule_sha256=sha256(schedule.encode()).hexdigest(),
                proxy_score=proxy_score,
                predicted_oil_tonnes=oil_tonnes,
                oil_uncertainty_tonnes=oil_uncertainty,
                injected_water_m3=injected_water,
                max_ood_score=float(np.max(np.asarray(rollout.ood_score))),
            )
        )

    if len(accepted) < 2:
        raise ValueError("fewer than two physically valid in-domain schedules remain")
    if accepted[0].candidate_id != "baseline":
        raise RuntimeError("baseline schedule was not retained")
    selected = min(accepted, key=lambda item: (-item.proxy_score, item.candidate_id))
    return Track2ScheduleSearch(
        baseline_scenario_id=trajectory.scenario_id,
        start_date=months[0],
        horizon_months=6,
        requested_candidates=candidate_count,
        accepted=tuple(accepted),
        rejected_ood=tuple(rejected_ood),
        selected=selected,
    )
