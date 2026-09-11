"""Canonical Track 2 scenario dataset loading with mandatory OPM provenance."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .opm import OPM_IMAGE_DIGEST, OpmSummaryError, verify_summary_extraction
from .surrogate import (
    ACTION_FEATURES,
    BHP_ACTION_FEATURES,
    STATE_FEATURES,
    ScenarioTrajectory,
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
TRAINING_SOURCE_SHA256 = "4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa"


def _case_source_sha256() -> str:
    """The archive every gate pins to: the training deck unless a case archive is declared.

    ``TIMESOIL_CASE_SOURCE_SHA256`` carries the SHA-256 of the organizers' case archive so
    that the same strict equality checks run against the case instead of the training deck.
    The value is validated, never trusted blindly, and the checks themselves stay strict.
    """
    value = os.environ.get("TIMESOIL_CASE_SOURCE_SHA256", TRAINING_SOURCE_SHA256).strip().lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("TIMESOIL_CASE_SOURCE_SHA256 must be a 64-character hex SHA-256")
    return value


MODEL_Z_SOURCE_SHA256 = _case_source_sha256()
_MODEL_Z_SOURCE_SHA256 = MODEL_Z_SOURCE_SHA256


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
        for column in (BHP_ACTION_FEATURES if "bhp_limit" in data else ACTION_FEATURES)
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
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no CSV trajectories in {path}")
    frames = [pd.read_csv(file) for file in files]
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
