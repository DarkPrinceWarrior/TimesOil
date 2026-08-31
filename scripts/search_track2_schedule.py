#!/usr/bin/env python3
"""Select a six-month Model Z schedule with the surrogate, then replay it once."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from hashlib import sha256
import io
import json
from math import ceil
import os
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from typing import Any

import timesoil.aios.contracts as contracts_module
import timesoil.aios.economics as economics_module
import timesoil.aios.opm as opm_module
import timesoil.aios.opm_chdd as opm_chdd_module
import timesoil.aios.scenario_generation as scenario_generation_module
import timesoil.aios.schedule_overlay as schedule_overlay_module
import timesoil.aios.surrogate as surrogate_module
import timesoil.aios.track2 as track2_module
from timesoil.aios.contracts import (
    ControlAction,
    ControlTarget,
    WellRole,
    WellStatus,
)
from timesoil.aios.economics import CHDDEconomicsAdapter
from timesoil.aios.opm import (
    OpmFlowRunner,
    OpmSummaryError,
    _source_digest,
    _validated_summary_artifacts,
)
from timesoil.aios.opm_chdd import export_opm_chdd
from timesoil.aios.scenario_generation import _actions_sha256
from timesoil.aios.schedule_overlay import apply_schedule_overlay
from timesoil.aios.surrogate import Track2Surrogate, _validated_surrogate_artifact
from timesoil.aios.track2 import (
    MODEL_Z_SOURCE_SHA256,
    MAX_TRACK2_SEARCH_CANDIDATES,
    Track2SearchCandidate,
    load_trajectory_dataset,
    search_track2_schedule,
)


_EXECUTION_SOURCE_PATHS = {
    "scripts.search_track2_schedule": Path(__file__).absolute(),
    "timesoil.aios.contracts": Path(contracts_module.__file__).absolute(),
    "timesoil.aios.scenario_generation": Path(
        scenario_generation_module.__file__
    ).absolute(),
    "timesoil.aios.track2": Path(track2_module.__file__).absolute(),
    "timesoil.aios.surrogate": Path(surrogate_module.__file__).absolute(),
    "timesoil.aios.schedule_overlay": Path(schedule_overlay_module.__file__).absolute(),
    "timesoil.aios.opm": Path(opm_module.__file__).absolute(),
    "timesoil.aios.opm_chdd": Path(opm_chdd_module.__file__).absolute(),
    "timesoil.aios.economics": Path(economics_module.__file__).absolute(),
}
_ECONOMICS_CALCULATOR_ARTIFACT_KEYS = {
    "РАСЧЕТ_ЧДД.py": "economics_calculator_main",
    "chdd_model.py": "economics_calculator_model",
    "excel_io.py": "economics_calculator_excel_io",
}
FileSnapshot = tuple[bytes, tuple[int, int, int, int, int, int]]
_MAX_REGULAR_FILE_BYTES = 64 * 1024**2
_MAX_SUMMARY_REPORT_BYTES = 128 * 1024**2
EconomicsSourceSnapshot = dict[str, tuple[Path, FileSnapshot]]


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _regular_snapshot(
    path: Path, label: str, *, limit: int = _MAX_REGULAR_FILE_BYTES
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    absolute = path.absolute()
    directory_fd: int | None = None
    file_fd: int | None = None
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(
            absolute.anchor,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec,
        )
        for part in absolute.parts[1:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            absolute.name,
            os.O_RDONLY | os.O_NOFOLLOW | close_on_exec,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if not S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if before.st_size > limit:
            raise ValueError(f"{label} exceeds {limit} bytes")
        identity = lambda item: (
            item.st_mode,
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

        def read_pass() -> bytes:
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(file_fd, min(1024**2, remaining))
                if not chunk or len(chunk) > remaining:
                    raise ValueError(f"{label} changed while it was read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(file_fd, 1):
                raise ValueError(f"{label} changed while it was read")
            return b"".join(chunks)

        first = read_pass()
        middle = os.fstat(file_fd)
        os.lseek(file_fd, 0, os.SEEK_SET)
        second = read_pass()
        after = os.fstat(file_fd)
        if (
            first != second
            or identity(before) != identity(middle)
            or identity(middle) != identity(after)
        ):
            raise ValueError(f"{label} changed while it was read")
        return second, identity(after)
    except OSError as exc:
        raise ValueError(f"{label} must be a regular non-symlink file") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _regular_bytes(
    path: Path, label: str, *, limit: int = _MAX_REGULAR_FILE_BYTES
) -> bytes:
    return _regular_snapshot(path, label, limit=limit)[0]


def _verify_regular_snapshot(
    path: Path,
    label: str,
    expected: tuple[bytes, tuple[int, int, int, int, int, int]],
) -> None:
    if _regular_snapshot(path, label) != expected:
        raise ValueError(f"{label} changed during load or search")


def _artifact(
    path: Path, label: str, *, limit: int = _MAX_REGULAR_FILE_BYTES
) -> dict[str, str]:
    data = _regular_bytes(path, label, limit=limit)
    return {"path": str(path.absolute()), "sha256": _sha256(data)}


def _execution_snapshot() -> dict[str, dict[str, str]]:
    return {
        name: _artifact(path, f"executed Python source {name}")
        for name, path in _EXECUTION_SOURCE_PATHS.items()
    }


def _verify_execution_snapshot(expected: dict[str, dict[str, str]]) -> None:
    if _execution_snapshot() != expected:
        raise RuntimeError("executed Python source changed during operation")


def _json_object_data(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = _regular_bytes(path, label, limit=4 * 1024**2)
    value = _json_object_data(data, label)
    return value, data


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _surrogate_snapshot(
    directory: Path, *, expected_files: dict[str, str] | None = None
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, tuple[bytes, str, tuple[int, int, int, int, int, int]]],
]:
    def capture(
        path: Path, label: str
    ) -> tuple[bytes, str, tuple[int, int, int, int, int, int]]:
        data, identity = _regular_snapshot(path, label)
        return data, _sha256(data), identity

    manifest_path = directory / "manifest.json"
    manifest_capture = capture(manifest_path, "surrogate manifest")
    manifest_bytes = manifest_capture[0]
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("surrogate manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("surrogate manifest must be a JSON object")
    model_capture = capture(
        directory / "model.json", "surrogate artifact file model.json"
    )
    files, _ = _validated_surrogate_artifact(manifest, model_capture[0])
    if expected_files is not None and files != expected_files:
        raise ValueError("surrogate artifact changed during load")

    snapshot = {
        "manifest.json": manifest_capture,
        "model.json": model_capture,
    }
    for name, expected in sorted(files.items()):
        if name == "model.json":
            continue
        relative = _safe_relative(name, "surrogate manifest file")
        data, digest, status = capture(
            directory.joinpath(*relative.parts), f"surrogate artifact file {name}"
        )
        if digest != expected:
            raise ValueError(f"surrogate artifact file failed hash check: {name}")
        snapshot[name] = data, digest, status
    return manifest, manifest_bytes, snapshot


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _safe_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def _candidate(candidate: Track2SearchCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "actions_sha256": candidate.actions_sha256,
        "wells_schedule_sha256": candidate.wells_schedule_sha256,
        "proxy_score": candidate.proxy_score,
        "predicted_oil_tonnes": candidate.predicted_oil_tonnes,
        "oil_uncertainty_tonnes": candidate.oil_uncertainty_tonnes,
        "injected_water_m3": candidate.injected_water_m3,
        "max_ood_score": candidate.max_ood_score,
        "selection_only": True,
        "certified": False,
    }


def _action(action: ControlAction) -> dict[str, str | float]:
    return {
        "month": action.month.isoformat(),
        "well": action.well,
        "role": action.role.value,
        "status": action.status.value,
        "target": action.target.value,
        "value": action.value,
    }


def _lineage_actions(value: object) -> tuple[ControlAction, ...]:
    fields = {"month", "well", "role", "status", "target", "value"}
    if not isinstance(value, list) or not value:
        raise ValueError("search lineage selected_actions must be a non-empty list")
    actions: list[ControlAction] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"search lineage action {index} has invalid fields")
        raw_value = item["value"]
        if (
            any(
                not isinstance(item[name], str)
                for name in ("month", "well", "role", "status", "target")
            )
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
        ):
            raise ValueError(f"search lineage action {index} has invalid field values")
        try:
            actions.append(
                ControlAction(
                    date.fromisoformat(item["month"]),
                    item["well"],
                    WellRole(item["role"]),
                    WellStatus(item["status"]),
                    ControlTarget(item["target"]),
                    float(raw_value),
                )
            )
        except ValueError as exc:
            raise ValueError(f"search lineage action {index} is invalid") from exc
    return tuple(actions)


def _authenticated_opm_input(
    result: Any,
    prepared: Any,
    prepared_schedule: Path,
    expected_sha256: str,
    expected_deck: PurePosixPath,
) -> tuple[dict[str, Any], str]:
    manifest_bytes = _regular_bytes(result.manifest_path, "OPM run manifest")
    manifest_sha256 = _sha256(manifest_bytes)
    if manifest_sha256 != result.manifest_sha256:
        raise ValueError("OPM run manifest hash disagrees with runner result")
    sidecar = _regular_bytes(
        result.manifest_path.with_name("manifest.sha256"), "OPM run manifest hash"
    )
    if sidecar != f"{manifest_sha256}  manifest.json\n".encode("ascii"):
        raise ValueError("OPM run manifest hash sidecar mismatch")
    try:
        manifest, _ = _validated_summary_artifacts(result.manifest_path)
    except OpmSummaryError as exc:
        raise ValueError("OPM run manifest failed authentication") from exc
    if manifest.get("source_sha256") != MODEL_Z_SOURCE_SHA256:
        raise ValueError("OPM run manifest source identity mismatch")

    expected_deck_path = prepared.input_dir.joinpath(*expected_deck.parts).absolute()
    if (
        prepared.deck_path.absolute() != expected_deck_path
        or result.deck_path.absolute() != expected_deck_path
        or manifest.get("deck") != str(expected_deck)
    ):
        raise ValueError("OPM run deck path disagrees with pinned search deck")
    deck_sha256 = _sha256(_regular_bytes(result.deck_path, "exact OPM input deck"))
    if manifest.get("deck_sha256") != deck_sha256:
        raise ValueError("OPM run deck hash mismatch")

    try:
        relative = prepared_schedule.relative_to(result.run_dir).as_posix()
    except ValueError as exc:
        raise ValueError("prepared OPM input schedule is outside the run directory") from exc
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    matches = [item for item in artifacts if item.get("path") == relative]
    if len(matches) != 1:
        raise ValueError("OPM run manifest misses exact prepared input schedule artifact")
    record = matches[0]
    schedule_bytes = _regular_bytes(prepared_schedule, "exact OPM input schedule")
    byte_count = record.get("bytes")
    if (
        record.get("sha256") != expected_sha256
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count != len(schedule_bytes)
        or _sha256(schedule_bytes) != expected_sha256
    ):
        raise ValueError("OPM run manifest exact input schedule hash mismatch")
    if _sha256(_regular_bytes(result.manifest_path, "OPM run manifest")) != manifest_sha256:
        raise ValueError("OPM run manifest changed during authentication")
    return manifest, manifest_sha256


def _manifest_relative(path: Path, manifest_path: Path) -> str:
    return Path(
        os.path.relpath(path.resolve(), manifest_path.resolve().parent)
    ).as_posix()


def _authenticated_export(
    expected: object,
    manifest_path: Path,
    report: Path,
    extraction: Path,
    opm_manifest: Path,
    chdd_csv: Path,
    trajectory_csv: Path,
) -> str:
    manifest, manifest_bytes = _json_object(
        manifest_path, "canonical export manifest"
    )
    if (
        not isinstance(expected, dict)
        or manifest != expected
        or manifest.get("schema_version") != 1
        or manifest.get("generator") != "timesoil.aios.opm_chdd"
    ):
        raise ValueError("canonical export manifest failed authentication")
    provenance = manifest.get("provenance")
    source = manifest.get("source")
    outputs = manifest.get("outputs")
    if not all(isinstance(value, dict) for value in (provenance, source, outputs)):
        raise ValueError("canonical export manifest is incomplete")
    assert isinstance(provenance, dict) and isinstance(source, dict)
    assert isinstance(outputs, dict)
    if (
        provenance.get("opm_run_manifest")
        != _manifest_relative(opm_manifest, manifest_path)
        or provenance.get("opm_run_manifest_sha256")
        != _sha256(_regular_bytes(opm_manifest, "linked OPM run manifest"))
        or provenance.get("summary_extraction_manifest")
        != _manifest_relative(extraction, manifest_path)
        or provenance.get("summary_extraction_manifest_sha256")
        != _sha256(_regular_bytes(extraction, "linked SUMMARY extraction manifest"))
        or source.get("summary_csv") != _manifest_relative(report, manifest_path)
        or source.get("summary_csv_sha256")
        != _sha256(
            _regular_bytes(
                report,
                "linked SUMMARY report",
                limit=_MAX_SUMMARY_REPORT_BYTES,
            )
        )
    ):
        raise ValueError("canonical export manifest input lineage mismatch")
    for name, path in (("chdd_csv", chdd_csv), ("track2_csv", trajectory_csv)):
        record = outputs.get(name)
        if (
            path.absolute().parent != manifest_path.absolute().parent
            or not isinstance(record, dict)
            or record.get("name") != path.name
            or record.get("sha256")
            != _sha256(_regular_bytes(path, f"canonical export {name}"))
        ):
            raise ValueError(f"canonical export manifest {name} link mismatch")
    return _sha256(manifest_bytes)


def _economics_source_paths(adapter: Any) -> dict[str, Path]:
    if set(_ECONOMICS_CALCULATOR_ARTIFACT_KEYS) != set(
        economics_module._CALCULATOR_FILES
    ):
        raise RuntimeError("official CHDD calculator file contract changed")
    try:
        norms_path = Path(adapter.norms_path).absolute()
        chdd_dir = Path(adapter.chdd_dir).absolute()
    except (AttributeError, TypeError) as exc:
        raise ValueError("official CHDD adapter source paths are invalid") from exc
    return {
        "economics_norms_source": norms_path,
        **{
            key: chdd_dir / name
            for name, key in _ECONOMICS_CALCULATOR_ARTIFACT_KEYS.items()
        },
    }


def _economics_source_snapshot(adapter: Any) -> EconomicsSourceSnapshot:
    return {
        key: (path, _regular_snapshot(path, key.replace("_", " ")))
        for key, path in _economics_source_paths(adapter).items()
    }


def _authenticated_economics(
    adapter: Any,
    result: Any,
    source_chdd: Path,
    profile: dict[str, Any],
    source_snapshot: EconomicsSourceSnapshot,
) -> tuple[str, dict[str, dict[str, str]]]:
    if _economics_source_snapshot(adapter) != source_snapshot:
        raise ValueError("official CHDD source files changed during calculation")
    manifest_path = Path(result.manifest_path).absolute()
    if (
        not hasattr(result, "output_dir")
        or Path(result.output_dir).absolute() != manifest_path.parent
        or manifest_path.name != "manifest.json"
    ):
        raise ValueError("official CHDD economics manifest path mismatch")
    manifest, manifest_bytes = _json_object(
        manifest_path, "official CHDD economics manifest"
    )
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        "input": "input.csv",
        "result": "result.json",
        "report": "report.xlsx",
        "effective_norms": "norms-effective.xlsx",
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("adapter") != "timesoil.aios.CHDDEconomicsAdapter"
        or manifest.get("start_year") != 2007
        or manifest.get("fields") != list(economics_module.CHDD_FIELDS)
        or manifest.get("assumption_overrides")
        != {"chargeInitialPump": profile.get("charge_initial_pump")}
        or profile.get("name") != "operational_sunk_assets"
        or profile.get("charge_initial_pump") is not False
        or artifacts != expected_artifacts
    ):
        raise ValueError("official CHDD economics manifest failed authentication")
    root = manifest_path.parent
    input_path = root / expected_artifacts["input"]
    result_path = root / expected_artifacts["result"]
    report_path = root / expected_artifacts["report"]
    effective_norms = root / expected_artifacts["effective_norms"]
    effective_snapshot = _regular_snapshot(effective_norms, "effective CHDD norms")
    source_norms = source_snapshot["economics_norms_source"][1][0]
    calculator_hashes = {
        name: _sha256(source_snapshot[key][1][0])
        for name, key in _ECONOMICS_CALCULATOR_ARTIFACT_KEYS.items()
    }
    model_source = source_snapshot["economics_calculator_model"][1][0]
    try:
        model_text = model_source.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("official CHDD model source must be UTF-8") from exc
    version_match = economics_module._VERSION_RE.search(model_text)
    if (
        manifest.get("norms_source_sha256") != _sha256(source_norms)
        or manifest.get("norms_sha256") != _sha256(effective_snapshot[0])
        or manifest.get("calculator_sha256") != calculator_hashes
        or manifest.get("calculator_version")
        != (version_match.group(1) if version_match else "unknown")
    ):
        raise ValueError("official CHDD economics provenance mismatch")
    input_bytes = _regular_bytes(input_path, "official CHDD input")
    input_records = _records(input_path)
    if (
        manifest.get("input_sha256") != _sha256(input_bytes)
        or economics_module.normalize_chdd_rows(input_records)
        != economics_module.normalize_chdd_rows(_records(source_chdd))
    ):
        raise ValueError("official CHDD input does not match canonical CHDD")
    raw_result, result_bytes = _json_object(result_path, "official CHDD result")
    validated = economics_module._validated_result(raw_result, expected_start_year=2007)
    if (
        manifest.get("result_sha256") != _sha256(result_bytes)
        or manifest.get("summary") != raw_result.get("summary")
        or manifest.get("row_count") != len(input_records)
        or validated["total_chdd_m"] != result.total_chdd_m
    ):
        raise ValueError("official CHDD result lineage mismatch")
    links = {
        "economics_input": _artifact(input_path, "official CHDD input"),
        "economics_result": _artifact(result_path, "official CHDD result"),
        "economics_report": _artifact(report_path, "official CHDD report"),
        "economics_effective_norms": {
            "path": str(effective_norms.absolute()),
            "sha256": _sha256(effective_snapshot[0]),
        },
        **{
            key: {"path": str(path.absolute()), "sha256": _sha256(snapshot[0])}
            for key, (path, snapshot) in source_snapshot.items()
        },
    }
    if (
        _economics_source_snapshot(adapter) != source_snapshot
        or _regular_snapshot(effective_norms, "effective CHDD norms")
        != effective_snapshot
    ):
        raise ValueError("official CHDD source files changed during authentication")
    return _sha256(manifest_bytes), links


def _validated_training(
    model: Track2Surrogate,
    metrics: dict[str, Any],
    trajectories: list[Any],
    model_manifest: dict[str, Any],
    model_manifest_sha256: str,
) -> None:
    if not getattr(trajectories, "model_z_identity", False):
        raise ValueError("trajectory provenance is not verified as official Model Z OPM")
    if metrics.get("model_z_ready") is not True or metrics.get("pipeline_proof_only") is not False:
        raise ValueError("training metrics do not certify a Model Z dataset")
    if metrics.get("source_models") != ["model_z_opm"]:
        raise ValueError("training metrics source_models must be exactly model_z_opm")
    training = model.training_metadata
    if training.get("model_z_ready") is not True:
        raise ValueError("surrogate artifact is not linked to verified Model Z training")
    if training.get("source_models") != ["model_z_opm"]:
        raise ValueError("surrogate artifact source_models must be exactly model_z_opm")
    if training.get("dataset_hash") != metrics.get("dataset_hash"):
        raise ValueError("surrogate artifact and metrics dataset hashes disagree")
    if metrics.get("surrogate_artifact_hash") != model_manifest.get("artifact_hash"):
        raise ValueError("training metrics and surrogate artifact hashes disagree")
    if metrics.get("surrogate_manifest_sha256") != model_manifest_sha256:
        raise ValueError("training metrics and surrogate manifest hashes disagree")
    calibration = metrics.get("conformal_calibration")
    artifact_calibration = training.get("conformal_calibration")
    calibration_ids = (
        calibration.get("scenario_ids") if isinstance(calibration, dict) else None
    )
    if (
        not model.is_calibrated
        or not isinstance(calibration, dict)
        or calibration != artifact_calibration
        or calibration.get("method") != "scenario_loso_max_normalized_residual"
        or calibration.get("nominal_coverage") != model.conformal_level
        or model.conformal_level is None
        or model.conformal_level < 0.9
        or isinstance(calibration.get("scenario_count"), bool)
        or not isinstance(calibration.get("scenario_count"), int)
        or calibration["scenario_count"] < 10
        or not isinstance(calibration_ids, list)
        or len(calibration_ids) != calibration["scenario_count"]
        or len(set(calibration_ids)) != len(calibration_ids)
        or set(calibration_ids) != set(training.get("scenario_ids", []))
        or calibration.get("quantile_rank")
        != ceil((calibration["scenario_count"] + 1) * model.conformal_level - 1e-12)
        or calibration.get("independent_validation") is not False
    ):
        raise ValueError("surrogate lacks valid whole-scenario 90% LOSO calibration")


def _replay_argv(
    source: Path,
    search_dir: Path,
    output: Path,
    deck: PurePosixPath,
    schedule: PurePosixPath,
    timeout_seconds: float,
    parsing_strictness: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/search_track2_schedule.py",
        "replay",
        str(source),
        str(search_dir),
        str(output),
        "--deck",
        str(deck),
        "--schedule-relative-path",
        str(schedule),
        "--timeout-seconds",
        str(timeout_seconds),
        "--parsing-strictness",
        parsing_strictness,
    ]


def _search(args: argparse.Namespace) -> Path:
    execution_sources = _execution_snapshot()
    model_dir = args.model.resolve()
    dataset = args.dataset.resolve()
    export_manifest = args.export_manifest.resolve()
    metrics_path = args.metrics.resolve()
    source = args.source.resolve()
    schedule = args.schedule.resolve()
    output = args.output.resolve()
    schedule_relative = _safe_relative(
        args.schedule_relative_path, "schedule-relative-path"
    )
    deck_relative = _safe_relative(args.deck, "deck")
    if schedule.name != schedule_relative.name:
        raise ValueError("schedule file name disagrees with schedule-relative-path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory {output}")
    if _source_digest(source) != MODEL_Z_SOURCE_SHA256:
        raise ValueError("Model Z source SHA-256 mismatch")

    schedule_bytes = _regular_bytes(schedule, "Model Z schedule")
    try:
        schedule_text = schedule_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Model Z schedule must be UTF-8") from exc
    dataset_snapshot = _regular_snapshot(dataset, "Model Z trajectory dataset")
    export_snapshot = _regular_snapshot(export_manifest, "Model Z export manifest")
    dataset_bytes = dataset_snapshot[0]
    manifest_bytes = export_snapshot[0]
    metrics, metrics_bytes = _json_object(metrics_path, "Track 2 metrics")
    model_manifest, model_manifest_bytes, model_snapshot = _surrogate_snapshot(model_dir)
    model = Track2Surrogate.load(model_dir)
    try:
        _, _, loaded_model_snapshot = _surrogate_snapshot(
            model_dir, expected_files=model_manifest["files"]
        )
    except (OSError, ValueError) as exc:
        raise ValueError("surrogate artifact changed during load") from exc
    if loaded_model_snapshot != model_snapshot:
        raise ValueError("surrogate artifact changed during load")
    trajectories = load_trajectory_dataset(dataset, manifest=export_manifest)
    _verify_regular_snapshot(
        dataset, "Model Z trajectory dataset", dataset_snapshot
    )
    _verify_regular_snapshot(
        export_manifest, "Model Z export manifest", export_snapshot
    )
    _validated_training(
        model, metrics, trajectories, model_manifest, _sha256(model_manifest_bytes)
    )
    matches = [item for item in trajectories if item.scenario_id == args.scenario_id]
    if len(matches) != 1:
        raise ValueError("scenario-id must identify exactly one verified trajectory")
    trajectory = matches[0]
    scenario_hashes = model.training_metadata.get("scenario_hashes")
    if not isinstance(scenario_hashes, dict) or trajectory.scenario_id not in scenario_hashes:
        raise ValueError("selected trajectory is absent from surrogate training metadata")
    if scenario_hashes[trajectory.scenario_id] != trajectory.content_hash:
        raise ValueError("selected trajectory hash disagrees with surrogate training metadata")
    start = date.fromisoformat(args.start_date)
    if start.day != 1:
        raise ValueError("start-date must be the first day of a month")
    indices = [index for index, value in enumerate(trajectory.dates) if value.date() == start]
    if len(indices) != 1:
        raise ValueError("start-date is absent or duplicated in the trajectory")

    result = search_track2_schedule(
        model,
        trajectory,
        start_index=indices[0],
        candidate_count=args.candidate_count,
        seed=args.seed,
        perturbation_fraction=args.perturbation_fraction,
        liquid_rate_scale=getattr(args, "liquid_rate_scale", 1.0),
        uncertainty_weight=args.uncertainty_weight,
        injection_cost_equivalent=args.injection_cost_equivalent,
    )
    selected = result.selected
    overlay = apply_schedule_overlay(
        schedule_text, selected.actions, known_wells=trajectory.well_ids
    )
    if overlay.controls_sha256 != selected.wells_schedule_sha256:
        raise RuntimeError("selected schedule and full overlay controls disagree")

    replay_argv = _replay_argv(
        source,
        output,
        output.parent / f"{output.name}-final-opm",
        deck_relative,
        schedule_relative,
        args.timeout_seconds,
        args.parsing_strictness,
    )
    candidates = [_candidate(item) for item in result.accepted]
    manifest = {
        "schema": "timesoil.aios.track2-surrogate-search/v1",
        "selection_only": True,
        "certified": False,
        "model_z_source_sha256": MODEL_Z_SOURCE_SHA256,
        "baseline_scenario_id": trajectory.scenario_id,
        "start_date": result.start_date.isoformat(),
        "horizon_months": result.horizon_months,
        "score": {
            "kind": "risk_adjusted_oil_minus_injection_proxy",
            "official_chdd": False,
            "uncertainty": "scenario_loso_conformal_half_width",
            "nominal_coverage": model.conformal_level,
            "uncertainty_weight": args.uncertainty_weight,
            "injection_cost_equivalent": args.injection_cost_equivalent,
        },
        "search": {
            "seed": args.seed,
            "perturbation_fraction": args.perturbation_fraction,
            "liquid_rate_scale": getattr(args, "liquid_rate_scale", 1.0),
            "requested_candidates": result.requested_candidates,
            "accepted_candidates": len(result.accepted),
            "rejected_ood": list(result.rejected_ood),
            "candidate_cap": MAX_TRACK2_SEARCH_CANDIDATES,
            "gates": [
                "unchanged_monthly_total_injection",
                "surrogate_ood",
                "physical_state_constraints",
                "final_opm_and_chdd_replay",
            ],
        },
        "candidates": candidates,
        "selected": _candidate(selected),
        "inputs": {
            "dataset_sha256": _sha256(dataset_bytes),
            "export_manifest_sha256": _sha256(manifest_bytes),
            "metrics_sha256": _sha256(metrics_bytes),
            "surrogate_artifact_hash": model_manifest["artifact_hash"],
            "surrogate_manifest_sha256": _sha256(model_manifest_bytes),
            "source_schedule_sha256": _sha256(schedule_bytes),
            "schedule_relative_path": str(schedule_relative),
            "deck_relative_path": str(deck_relative),
        },
        "artifacts": {
            "wells_schedule": "wells_schedule.inc",
            "wells_schedule_sha256": selected.wells_schedule_sha256,
            "modified_schedule": schedule.name,
            "modified_schedule_sha256": overlay.sha256,
            "lineage": "lineage.json",
        },
        "schedule_transformation": {
            "source_schedule_sha256": overlay.source_sha256,
            "controls_sha256": overlay.controls_sha256,
            "output_schedule_sha256": overlay.sha256,
        },
        "execution_sources": execution_sources,
        "final_replay_argv": replay_argv,
        "missing_certification_evidence": [
            "successful_opm_run_manifest",
            "authenticated_summary_extraction",
            "canonical_chdd_export_manifest",
            "official_chdd_receipt",
        ],
    }
    lineage = {
        "schema": "timesoil.aios.track2-search-lineage/v1",
        "selected_candidate_id": selected.candidate_id,
        "selected_actions_sha256": selected.actions_sha256,
        "baseline_trajectory_sha256": trajectory.content_hash,
        "model_z_source_sha256": MODEL_Z_SOURCE_SHA256,
        "deck_relative_path": str(deck_relative),
        "input_hashes": manifest["inputs"],
        "selected_actions": [_action(item) for item in selected.actions],
        "certified": False,
    }
    lineage_bytes = _json_bytes(lineage)
    manifest["artifacts"]["lineage_sha256"] = _sha256(lineage_bytes)
    manifest_bytes_out = _json_bytes(manifest)
    _verify_regular_snapshot(
        dataset, "Model Z trajectory dataset", dataset_snapshot
    )
    _verify_regular_snapshot(
        export_manifest, "Model Z export manifest", export_snapshot
    )
    output.mkdir(parents=True)
    _write_new(output / "wells_schedule.inc", selected.wells_schedule.encode())
    _write_new(output / schedule.name, overlay.text.encode())
    _write_new(output / "lineage.json", lineage_bytes)
    _verify_execution_snapshot(execution_sources)
    _write_new(output / "manifest.json", manifest_bytes_out)
    _write_new(output / "manifest.sha256", (_sha256(manifest_bytes_out) + "\n").encode())
    return output / "manifest.json"


def _records(path: Path) -> list[dict[str, str]]:
    data = _regular_bytes(path, "CSV input")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("CSV input must be UTF-8") from exc
    return list(csv.DictReader(io.StringIO(text, newline="")))


def _replay(args: argparse.Namespace) -> Path:
    execution_sources = _execution_snapshot()
    source = args.source.resolve()
    search_dir = args.search_dir.resolve()
    output = args.output.resolve()
    search, search_bytes = _json_object(search_dir / "manifest.json", "search manifest")
    sidecar = _regular_bytes(search_dir / "manifest.sha256", "search manifest hash")
    if sidecar != (_sha256(search_bytes) + "\n").encode():
        raise ValueError("search manifest hash sidecar mismatch")
    if search.get("schema") != "timesoil.aios.track2-surrogate-search/v1":
        raise ValueError("unsupported search manifest")
    if search.get("certified") is not False or search.get("selection_only") is not True:
        raise ValueError("search manifest has unsafe certification state")
    if search.get("model_z_source_sha256") != MODEL_Z_SOURCE_SHA256:
        raise ValueError("search manifest Model Z source identity mismatch")
    if _source_digest(source) != MODEL_Z_SOURCE_SHA256:
        raise ValueError("Model Z source SHA-256 mismatch")
    inputs = search.get("inputs")
    artifacts = search.get("artifacts")
    selected = search.get("selected")
    if not isinstance(inputs, dict) or not isinstance(artifacts, dict):
        raise ValueError("search manifest inputs or artifacts are invalid")
    schedule_relative = _safe_relative(
        args.schedule_relative_path, "schedule-relative-path"
    )
    deck_relative = _safe_relative(args.deck, "deck")
    expected_relative = inputs.get("schedule_relative_path")
    if str(schedule_relative) != expected_relative:
        raise ValueError("replay schedule-relative-path disagrees with search manifest")
    if str(deck_relative) != inputs.get("deck_relative_path"):
        raise ValueError("replay deck disagrees with search manifest")
    if search.get("final_replay_argv") != _replay_argv(
        source,
        search_dir,
        output,
        deck_relative,
        schedule_relative,
        args.timeout_seconds,
        args.parsing_strictness,
    ):
        raise ValueError("replay arguments disagree with search manifest")
    wells_relative = _safe_relative(
        artifacts.get("wells_schedule"), "wells schedule artifact"
    )
    wells_schedule = _regular_bytes(
        search_dir / Path(*wells_relative.parts), "selected wells schedule"
    )
    wells_schedule_sha256 = _sha256(wells_schedule)
    if wells_schedule_sha256 != artifacts.get("wells_schedule_sha256"):
        raise ValueError("selected wells schedule hash mismatch")
    if not isinstance(selected, dict):
        raise ValueError("search manifest selected candidate is invalid")
    baseline_id = search.get("baseline_scenario_id")
    candidate_id = selected.get("candidate_id")
    search_config = search.get("search")
    if (
        not isinstance(baseline_id, str)
        or not baseline_id
        or not isinstance(candidate_id, str)
        or not candidate_id
        or not isinstance(search_config, dict)
        or isinstance(search_config.get("seed"), bool)
        or not isinstance(search_config.get("seed"), int)
    ):
        raise ValueError("search manifest scenario, candidate, or seed is invalid")
    start_raw = search.get("start_date")
    horizon_months = search.get("horizon_months")
    if (
        not isinstance(start_raw, str)
        or isinstance(horizon_months, bool)
        or not isinstance(horizon_months, int)
        or horizon_months < 1
    ):
        raise ValueError("search manifest horizon is invalid")
    start = date.fromisoformat(start_raw)
    if start.day != 1:
        raise ValueError("search manifest horizon must start on the first day of a month")
    end_index = start.year * 12 + start.month - 1 + horizon_months
    end_exclusive = date(end_index // 12, end_index % 12 + 1, 1)
    modified_relative = _safe_relative(
        artifacts.get("modified_schedule"), "modified schedule artifact"
    )
    if len(modified_relative.parts) != 1:
        raise ValueError("modified schedule artifact must be a basename")
    selected_schedule = _regular_bytes(
        search_dir / Path(*modified_relative.parts), "selected full schedule overlay"
    )
    selected_schedule_sha256 = _sha256(selected_schedule)
    if selected_schedule_sha256 != artifacts.get("modified_schedule_sha256"):
        raise ValueError("selected full schedule overlay hash mismatch")
    lineage_relative = _safe_relative(artifacts.get("lineage"), "lineage artifact")
    lineage_path = search_dir.joinpath(*lineage_relative.parts)
    lineage_snapshot = _regular_snapshot(
        lineage_path, "search lineage", limit=4 * 1024**2
    )
    lineage = lineage_snapshot[0]
    lineage_value = _json_object_data(lineage, "search lineage")
    if _sha256(lineage) != artifacts.get("lineage_sha256"):
        raise ValueError("search lineage hash mismatch")
    if lineage_value.get("schema") != "timesoil.aios.track2-search-lineage/v1":
        raise ValueError("unsupported search lineage")
    if (
        lineage_value.get("selected_candidate_id") != selected.get("candidate_id")
        or lineage_value.get("selected_actions_sha256") != selected.get("actions_sha256")
        or lineage_value.get("model_z_source_sha256") != MODEL_Z_SOURCE_SHA256
        or lineage_value.get("deck_relative_path") != str(deck_relative)
        or lineage_value.get("input_hashes") != inputs
    ):
        raise ValueError("search lineage disagrees with search manifest")
    actions = _lineage_actions(lineage_value.get("selected_actions"))
    actions_sha256 = _actions_sha256(actions)
    if (
        lineage_value.get("selected_actions_sha256") != actions_sha256
        or selected.get("actions_sha256") != actions_sha256
    ):
        raise ValueError("selected actions canonical hash mismatch")
    canonical_wells_schedule = schedule_overlay_module._canonical_schedule(
        actions
    ).encode()
    if (
        selected.get("wells_schedule_sha256") != wells_schedule_sha256
        or canonical_wells_schedule != wells_schedule
    ):
        raise ValueError("selected wells schedule disagrees with selected candidate")

    runner = OpmFlowRunner(timeout_seconds=args.timeout_seconds)
    prepared = runner.prepare(source, output, deck=str(deck_relative))
    prepared_schedule = prepared.input_dir / Path(*schedule_relative.parts)
    source_schedule = _regular_bytes(prepared_schedule, "prepared Model Z schedule")
    if _sha256(source_schedule) != inputs.get("source_schedule_sha256"):
        raise ValueError("prepared Model Z schedule disagrees with search input")
    try:
        source_text = source_schedule.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("source schedule must be UTF-8") from exc
    replay_overlay = apply_schedule_overlay(
        source_text,
        actions,
        known_wells={item.well for item in actions},
    )
    transformation = {
        "source_schedule_sha256": replay_overlay.source_sha256,
        "controls_sha256": replay_overlay.controls_sha256,
        "output_schedule_sha256": replay_overlay.sha256,
    }
    if (
        replay_overlay.controls_sha256 != wells_schedule_sha256
        or search.get("schedule_transformation") != transformation
        or replay_overlay.text.encode() != selected_schedule
    ):
        raise ValueError("selected overlay fails source plus controls transformation proof")
    prepared_schedule.write_bytes(selected_schedule)
    result = runner._run_prepared(
        prepared, parsing_strictness=args.parsing_strictness
    )
    opm_manifest, opm_manifest_sha256 = _authenticated_opm_input(
        result,
        prepared,
        prepared_schedule,
        selected_schedule_sha256,
        deck_relative,
    )
    report, extraction = runner.extract_summary_report(
        result, result.run_dir / "summary-report.txt"
    )
    canonical = result.run_dir / "canonical"
    chdd_csv = canonical / "chdd.csv"
    trajectory_csv = canonical / "trajectory.csv"
    export_manifest = canonical / "manifest.json"
    exported = export_opm_chdd(
        report,
        chdd_csv,
        trajectory_csv,
        export_manifest,
        scenario_id="track2-selected-final-replay",
        source_model="model_z_opm",
        opm_run_manifest=result.manifest_path,
        summary_extraction_manifest=extraction,
        deck_dir=result.deck_path.parent,
        unit_system=prepared.unit_system,
    )
    export_manifest_sha256 = _authenticated_export(
        exported,
        export_manifest,
        report,
        extraction,
        result.manifest_path,
        chdd_csv,
        trajectory_csv,
    )
    economics_profile = {
        "name": "operational_sunk_assets",
        "charge_initial_pump": False,
        "semantics": (
            "existing ESPs on 2007-01-01 are sunk assets and are not charged again"
        ),
    }
    adapter = CHDDEconomicsAdapter.from_env()
    economics_source_snapshot = _economics_source_snapshot(adapter)
    economics = adapter.calculate(
        _records(chdd_csv),
        start_year=2007,
        output_dir=result.run_dir / "economics-2007",
        charge_initial_pump=False,
    )
    economics_manifest_sha256, economics_artifacts = _authenticated_economics(
        adapter,
        economics,
        chdd_csv,
        economics_profile,
        economics_source_snapshot,
    )
    simulator_fields = (
        "image",
        "image_digest",
        "image_reference",
        "deck",
        "deck_sha256",
    )
    if any(not isinstance(opm_manifest.get(name), str) for name in simulator_fields):
        raise ValueError("OPM run manifest simulator identity is incomplete")
    receipt_artifacts = {
        "search_manifest": _artifact(search_dir / "manifest.json", "search manifest"),
        "lineage": _artifact(lineage_path, "search lineage"),
        "submitted_wells_schedule": _artifact(
            search_dir / Path(*wells_relative.parts), "submitted wells schedule"
        ),
        "replay_overlay": _artifact(
            search_dir / Path(*modified_relative.parts), "selected replay overlay"
        ),
        "opm_run_manifest": _artifact(result.manifest_path, "OPM run manifest"),
        "exact_opm_input_schedule": _artifact(
            prepared_schedule, "exact OPM input schedule"
        ),
        "summary_report": _artifact(
            report, "SUMMARY report", limit=_MAX_SUMMARY_REPORT_BYTES
        ),
        "summary_extraction": _artifact(extraction, "SUMMARY extraction manifest"),
        "canonical_export_manifest": _artifact(
            export_manifest, "canonical export manifest"
        ),
        "chdd_csv": _artifact(chdd_csv, "canonical CHDD CSV"),
        "trajectory_csv": _artifact(trajectory_csv, "canonical trajectory CSV"),
        "economics_manifest": _artifact(
            economics.manifest_path, "official CHDD economics manifest"
        ),
        **economics_artifacts,
    }
    if (
        receipt_artifacts["search_manifest"]["sha256"] != _sha256(search_bytes)
        or receipt_artifacts["lineage"]["sha256"] != _sha256(lineage)
        or receipt_artifacts["lineage"]["sha256"]
        != artifacts.get("lineage_sha256")
        or receipt_artifacts["opm_run_manifest"]["sha256"] != opm_manifest_sha256
        or receipt_artifacts["submitted_wells_schedule"]["sha256"]
        != wells_schedule_sha256
        or receipt_artifacts["replay_overlay"]["sha256"]
        != selected_schedule_sha256
        or receipt_artifacts["exact_opm_input_schedule"]["sha256"]
        != selected_schedule_sha256
        or receipt_artifacts["canonical_export_manifest"]["sha256"]
        != export_manifest_sha256
        or receipt_artifacts["economics_manifest"]["sha256"]
        != economics_manifest_sha256
    ):
        raise ValueError("final replay artifacts changed before receipt")
    receipt = {
        "schema": "timesoil.aios.track2-final-replay/v2",
        "complete": True,
        "organizer_certified": False,
        "model_z_source_sha256": MODEL_Z_SOURCE_SHA256,
        "scenario": {
            "baseline_id": baseline_id,
            "selected_candidate_id": candidate_id,
            "replay_id": "track2-selected-final-replay",
        },
        "search_seed": search_config["seed"],
        "horizon": {
            "start_inclusive": start.isoformat(),
            "end_exclusive": end_exclusive.isoformat(),
            "months": horizon_months,
        },
        "run_ids": {
            "opm": f"opm-{opm_manifest_sha256[:24]}",
            "economics": (
                "economics-"
                f"{receipt_artifacts['economics_manifest']['sha256'][:24]}"
            ),
        },
        "schedule_transformation": transformation,
        "simulator": {name: opm_manifest[name] for name in simulator_fields},
        "artifacts": receipt_artifacts,
        "execution_sources": execution_sources,
        "total_chdd_m": economics.total_chdd_m,
        "start_year": 2007,
        "economics_profile": economics_profile,
    }
    receipt_path = result.run_dir / "final-replay-receipt.json"
    _verify_regular_snapshot(lineage_path, "search lineage", lineage_snapshot)
    if (
        _authenticated_export(
            exported,
            export_manifest,
            report,
            extraction,
            result.manifest_path,
            chdd_csv,
            trajectory_csv,
        )
        != export_manifest_sha256
        or _authenticated_economics(
            adapter,
            economics,
            chdd_csv,
            economics_profile,
            economics_source_snapshot,
        )
        != (economics_manifest_sha256, economics_artifacts)
    ):
        raise ValueError("certification artifacts changed before receipt")
    _verify_execution_snapshot(execution_sources)
    _write_new(receipt_path, _json_bytes(receipt))
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search six-month Model Z schedules and replay the selected one once."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("model", type=Path)
    search.add_argument("dataset", type=Path)
    search.add_argument("export_manifest", type=Path)
    search.add_argument("metrics", type=Path)
    search.add_argument("source", type=Path)
    search.add_argument("schedule", type=Path)
    search.add_argument("output", type=Path)
    search.add_argument("--scenario-id", required=True)
    search.add_argument("--start-date", required=True)
    search.add_argument("--candidate-count", type=int, default=32)
    search.add_argument("--seed", type=int, default=20260831)
    search.add_argument("--perturbation-fraction", type=float, default=0.05)
    search.add_argument("--liquid-rate-scale", type=float, default=1.0)
    search.add_argument("--uncertainty-weight", type=float, default=1.0)
    search.add_argument("--injection-cost-equivalent", type=float, default=0.01)
    search.add_argument("--deck", default="Model_Z/Model_Z.data")
    search.add_argument(
        "--schedule-relative-path", default="Model_Z/Model_Z_sch.inc"
    )
    search.add_argument("--timeout-seconds", type=float, default=3600.0)
    search.add_argument("--parsing-strictness", choices=("strict", "low"), default="low")

    replay = commands.add_parser("replay")
    replay.add_argument("source", type=Path)
    replay.add_argument("search_dir", type=Path)
    replay.add_argument("output", type=Path)
    replay.add_argument("--deck", default="Model_Z/Model_Z.data")
    replay.add_argument(
        "--schedule-relative-path", default="Model_Z/Model_Z_sch.inc"
    )
    replay.add_argument("--timeout-seconds", type=float, default=3600.0)
    replay.add_argument("--parsing-strictness", choices=("strict", "low"), default="low")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _search(args) if args.command == "search" else _replay(args)
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
