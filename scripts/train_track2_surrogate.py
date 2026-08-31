#!/usr/bin/env python3
"""Train and persist the Track 2 stateful surrogate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any

from timesoil.aios import surrogate as _surrogate_module
from timesoil.aios import track2 as _track2_module
from timesoil.aios.scenario_generation import _actions_sha256
from timesoil.aios.track2 import (
    CANONICAL_COLUMNS,
    MODEL_Z_SCENARIO_ACTIONS_SHA256,
    MODEL_Z_SCENARIO_EXPORTED_ACTIONS_SHA256,
    MODEL_Z_SCENARIO_INDEX_SHA256,
    MODEL_Z_SOURCE_SHA256,
    fit_track2_surrogate,
    load_trajectory_dataset,
    _trajectory_controls,
)


ROOT = Path(__file__).resolve().parents[1]
_MAX_SOURCE_BYTES = 4 * 1024**2
_MAX_BATCH_ARTIFACT_BYTES = 128 * 1024**2
_SCENARIO_RUN_SCHEMA = "timesoil.aios.track2-scenario-run/v2"
_TRAINING_LINEAGE_SCHEMA = "timesoil.aios.track2-training-lineage/v1"
_MODEL_Z_SCENARIO_INDEX_SHA256 = MODEL_Z_SCENARIO_INDEX_SHA256
_EXPECTED_SCENARIO_ACTIONS_SHA256 = dict(MODEL_Z_SCENARIO_ACTIONS_SHA256)
_EXPECTED_EXPORTED_ACTIONS_SHA256 = dict(
    MODEL_Z_SCENARIO_EXPORTED_ACTIONS_SHA256
)
_EXPECTED_SCENARIO_IDS = tuple(_EXPECTED_SCENARIO_ACTIONS_SHA256)


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink path component is forbidden: {current}")


def _read_regular(path: Path, label: str, *, limit: int = _MAX_SOURCE_BYTES) -> bytes:
    def read_once(descriptor: int, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024**2))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining or os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)

    absolute = Path(os.path.abspath(path))
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        directory_fds.append(os.open(absolute.anchor, directory_flags))
        for part in absolute.parts[1:-1]:
            directory_fds.append(
                os.open(part, directory_flags, dir_fd=directory_fds[-1])
            )
        file_fd = os.open(
            absolute.name or ".", os.O_RDONLY | no_follow, dir_fd=directory_fds[-1]
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if before.st_size > limit:
            raise ValueError(f"{label} exceeds {limit} bytes")
        first = read_once(file_fd, before.st_size)
        middle = os.fstat(file_fd)
        os.lseek(file_fd, 0, os.SEEK_SET)
        second = read_once(file_fd, before.st_size)
        after = os.fstat(file_fd)
        identity = lambda item: (
            item.st_mode,
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if not (
            identity(before) == identity(middle) == identity(after)
            and first == second
        ):
            raise ValueError(f"{label} changed while it was read")
        return first
    finally:
        for descriptor in ([file_fd] if file_fd is not None else []) + list(
            reversed(directory_fds)
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _snapshot_executed_sources() -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for module_file in (__file__, _track2_module.__file__, _surrogate_module.__file__):
        if not isinstance(module_file, str) or not module_file:
            raise ValueError("executed source module has no __file__")
        path = Path(module_file).absolute()
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(_read_regular(path, "executed source")).hexdigest(),
            }
        )
    return sources


def _verify_executed_sources(expected: list[dict[str, str]]) -> None:
    if _snapshot_executed_sources() != expected:
        raise RuntimeError("executed source changed during surrogate training")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validated_batch_lineage(
    batch_manifest: Path,
    expected_scenario_index_sha256: str,
    dataset: Path,
    export_manifests: Path,
) -> dict[str, object]:
    expected_index = _digest(
        expected_scenario_index_sha256, "expected scenario index SHA-256"
    )
    if expected_index != _MODEL_Z_SCENARIO_INDEX_SHA256:
        raise ValueError("scenario index SHA-256 is not the frozen Track 2 index")

    batch_path = Path(os.path.abspath(batch_manifest))
    batch_root = batch_path.parent
    dataset_dir = Path(os.path.abspath(dataset))
    manifest_dir = Path(os.path.abspath(export_manifests))
    if batch_path.name != "manifest.json":
        raise ValueError("scenario batch manifest path must end in manifest.json")
    if dataset_dir != batch_root / "dataset" or manifest_dir != batch_root / "manifests":
        raise ValueError("dataset and export manifests must belong to the batch manifest")

    batch_bytes = _read_regular(batch_path, "scenario batch manifest")
    try:
        batch = json.loads(
            batch_bytes.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("scenario batch manifest must be UTF-8 JSON") from exc
    if not isinstance(batch, dict):
        raise ValueError("scenario batch manifest must be a JSON object")
    if set(batch) != {
        "schema",
        "executed_sources",
        "official_source_sha256",
        "scenario_index_sha256",
        "scenario_count",
        "sequential",
        "scenarios",
        "training",
    }:
        raise ValueError("scenario batch manifest fields are not canonical")
    if batch.get("schema") != _SCENARIO_RUN_SCHEMA:
        raise ValueError("scenario batch manifest schema is not the conformal v2 contract")
    if batch.get("official_source_sha256") != MODEL_Z_SOURCE_SHA256:
        raise ValueError("scenario batch manifest is not bound to official Model Z")
    if batch.get("scenario_index_sha256") != expected_index:
        raise ValueError("scenario batch manifest has the wrong scenario index SHA-256")
    if (
        batch.get("scenario_count") != len(_EXPECTED_SCENARIO_IDS)
        or isinstance(batch.get("scenario_count"), bool)
        or batch.get("sequential") is not True
    ):
        raise ValueError("scenario batch must contain ten sequential runs")
    records = batch.get("scenarios")
    if not isinstance(records, list) or len(records) != len(_EXPECTED_SCENARIO_IDS):
        raise ValueError("scenario batch records are incomplete")
    scenario_ids = tuple(
        record.get("scenario_id") if isinstance(record, dict) else None
        for record in records
    )
    if scenario_ids != _EXPECTED_SCENARIO_IDS:
        raise ValueError("scenario batch IDs are not the frozen canonical set")

    expected_dataset_names = {f"{scenario_id}.csv" for scenario_id in scenario_ids}
    expected_manifest_names = {f"{scenario_id}.json" for scenario_id in scenario_ids}
    actual_dataset_names = {
        path.name
        for pattern in ("*.csv", "*.parquet")
        for path in dataset_dir.glob(pattern)
    }
    actual_manifest_names = {path.name for path in manifest_dir.glob("*.json")}
    if actual_dataset_names != expected_dataset_names:
        raise ValueError("scenario batch dataset file set is not canonical")
    if actual_manifest_names != expected_manifest_names:
        raise ValueError("scenario batch export manifest file set is not canonical")

    authenticated: list[dict[str, str]] = []
    for scenario_id, record in zip(scenario_ids, records, strict=True):
        if not isinstance(scenario_id, str) or not isinstance(record, dict):
            raise ValueError("scenario batch record is invalid")
        if set(record) != {
            "scenario_id",
            "actions_sha256",
            "run_manifest",
            "run_manifest_sha256",
            "summary_report_sha256",
            "summary_extraction_sha256",
            "canonical_chdd",
            "canonical_chdd_sha256",
            "dataset",
            "dataset_sha256",
            "export_manifest",
            "export_manifest_sha256",
        }:
            raise ValueError(f"scenario {scenario_id}: record fields are not canonical")
        run_manifest_relative = f"{scenario_id}/manifest.json"
        summary_report_relative = f"{scenario_id}/summary-report.txt"
        summary_extraction_relative = f"{scenario_id}/summary-extraction.json"
        canonical_chdd_relative = f"{scenario_id}/canonical/chdd.csv"
        dataset_relative = f"dataset/{scenario_id}.csv"
        manifest_relative = f"manifests/{scenario_id}.json"
        if record.get("run_manifest") != run_manifest_relative:
            raise ValueError(f"scenario {scenario_id}: run manifest path is not canonical")
        if record.get("canonical_chdd") != canonical_chdd_relative:
            raise ValueError(f"scenario {scenario_id}: CHDD path is not canonical")
        if record.get("dataset") != dataset_relative:
            raise ValueError(f"scenario {scenario_id}: dataset path is not canonical")
        if record.get("export_manifest") != manifest_relative:
            raise ValueError(f"scenario {scenario_id}: export manifest path is not canonical")
        actions_sha256 = _digest(
            record.get("actions_sha256"), f"scenario {scenario_id} actions"
        )
        if actions_sha256 != _EXPECTED_SCENARIO_ACTIONS_SHA256[scenario_id]:
            raise ValueError(f"scenario {scenario_id}: actions SHA-256 mismatch")
        dataset_sha256 = _digest(
            record.get("dataset_sha256"), f"scenario {scenario_id} dataset"
        )
        export_manifest_sha256 = _digest(
            record.get("export_manifest_sha256"),
            f"scenario {scenario_id} export manifest",
        )
        run_manifest_sha256 = _digest(
            record.get("run_manifest_sha256"), f"scenario {scenario_id} run manifest"
        )
        summary_report_sha256 = _digest(
            record.get("summary_report_sha256"), f"scenario {scenario_id} summary report"
        )
        summary_extraction_sha256 = _digest(
            record.get("summary_extraction_sha256"),
            f"scenario {scenario_id} summary extraction",
        )
        canonical_chdd_sha256 = _digest(
            record.get("canonical_chdd_sha256"), f"scenario {scenario_id} CHDD"
        )
        dataset_bytes = _read_regular(
            batch_root / dataset_relative,
            f"scenario {scenario_id} dataset",
            limit=_MAX_BATCH_ARTIFACT_BYTES,
        )
        export_manifest_bytes = _read_regular(
            batch_root / manifest_relative,
            f"scenario {scenario_id} export manifest",
            limit=_MAX_BATCH_ARTIFACT_BYTES,
        )
        linked_artifacts = (
            (run_manifest_relative, run_manifest_sha256, "run manifest"),
            (summary_report_relative, summary_report_sha256, "summary report"),
            (
                summary_extraction_relative,
                summary_extraction_sha256,
                "summary extraction",
            ),
            (canonical_chdd_relative, canonical_chdd_sha256, "CHDD"),
        )
        for relative, expected_sha256, label in linked_artifacts:
            if sha256(
                _read_regular(
                    batch_root / relative,
                    f"scenario {scenario_id} {label}",
                    limit=_MAX_BATCH_ARTIFACT_BYTES,
                )
            ).hexdigest() != expected_sha256:
                raise ValueError(f"scenario {scenario_id}: {label} SHA-256 mismatch")
        if sha256(dataset_bytes).hexdigest() != dataset_sha256:
            raise ValueError(f"scenario {scenario_id}: dataset SHA-256 mismatch")
        if sha256(export_manifest_bytes).hexdigest() != export_manifest_sha256:
            raise ValueError(f"scenario {scenario_id}: export manifest SHA-256 mismatch")
        try:
            exported = json.loads(
                export_manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_object
            )
            provenance = exported["provenance"]
            source = exported["source"]
            scenario = exported["scenario"]
            outputs = exported["outputs"]
            track2_output = outputs["track2_csv"]
            chdd_output = outputs["chdd_csv"]
        except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"scenario {scenario_id}: export manifest is incomplete"
            ) from exc
        if (
            exported.get("generator") != "timesoil.aios.opm_chdd"
            or not all(
                isinstance(item, dict)
                for item in (provenance, source, scenario, outputs, track2_output, chdd_output)
            )
            or scenario.get("scenario_id") != scenario_id
            or scenario.get("source_model") != "model_z_opm"
            or provenance.get("opm_source_sha256") != MODEL_Z_SOURCE_SHA256
            or provenance.get("opm_run_manifest") != f"../{run_manifest_relative}"
            or provenance.get("opm_run_manifest_sha256") != run_manifest_sha256
            or provenance.get("summary_extraction_manifest")
            != f"../{summary_extraction_relative}"
            or provenance.get("summary_extraction_manifest_sha256")
            != summary_extraction_sha256
            or source.get("summary_csv") != f"../{summary_report_relative}"
            or source.get("summary_csv_sha256") != summary_report_sha256
            or track2_output.get("name") != f"{scenario_id}.csv"
            or track2_output.get("sha256") != dataset_sha256
            or chdd_output.get("name") != "chdd.csv"
            or chdd_output.get("sha256") != canonical_chdd_sha256
        ):
            raise ValueError(f"scenario {scenario_id}: export provenance disagrees with batch")
        authenticated.append(
            {
                "scenario_id": scenario_id,
                "actions_sha256": actions_sha256,
                "dataset_sha256": dataset_sha256,
                "export_manifest_sha256": export_manifest_sha256,
            }
        )
    if len({item["actions_sha256"] for item in authenticated}) != len(authenticated):
        raise ValueError("scenario batch action hashes must be unique")

    return {
        "schema": _TRAINING_LINEAGE_SCHEMA,
        "scenario_run_schema": _SCENARIO_RUN_SCHEMA,
        "batch_manifest_sha256": sha256(batch_bytes).hexdigest(),
        "official_source_sha256": MODEL_Z_SOURCE_SHA256,
        "scenario_index_sha256": expected_index,
        "scenario_ids": list(scenario_ids),
        "scenarios": authenticated,
    }


def _verify_trajectory_actions(trajectories: list[Any]) -> None:
    trajectories_by_id = {item.scenario_id: item for item in trajectories}
    if tuple(trajectories_by_id) != _EXPECTED_SCENARIO_IDS:
        raise ValueError("loaded trajectories are not the frozen canonical scenario set")
    for scenario_id, expected_actions_sha256 in _EXPECTED_EXPORTED_ACTIONS_SHA256.items():
        if _actions_sha256(
            _trajectory_controls(trajectories_by_id[scenario_id])
        ) != expected_actions_sha256:
            raise ValueError(f"scenario {scenario_id}: loaded controls disagree with index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="canonical OPM CSV/Parquet file or directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="OPM/CHDD export manifest file, or directory of manifests",
    )
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        required=True,
        help="authenticated timesoil.aios.track2-scenario-run/v2 manifest",
    )
    parser.add_argument(
        "--scenario-index-sha256",
        required=True,
        help="exact frozen Track 2 scenario index SHA-256",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "track2_surrogate")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conformal-level", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    executed_sources = _snapshot_executed_sources()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory {output}")
    scenario_batch = _validated_batch_lineage(
        args.batch_manifest,
        args.scenario_index_sha256,
        args.dataset,
        args.manifest,
    )
    trajectories = load_trajectory_dataset(args.dataset, manifest=args.manifest)
    _verify_trajectory_actions(trajectories)
    run = fit_track2_surrogate(
        trajectories,
        test_fraction=args.test_fraction,
        ensemble_size=args.ensemble_size,
        n_estimators=args.n_estimators,
        horizon=args.horizon,
        seed=args.seed,
        conformal_level=args.conformal_level,
    )
    if _validated_batch_lineage(
        args.batch_manifest,
        args.scenario_index_sha256,
        args.dataset,
        args.manifest,
    ) != scenario_batch:
        raise RuntimeError("scenario batch changed during surrogate training")
    run.model.training_metadata["scenario_batch"] = scenario_batch
    _verify_executed_sources(executed_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    report = run.report()
    report["scenario_batch"] = scenario_batch
    report["canonical_columns"] = CANONICAL_COLUMNS
    report["executed_sources"] = executed_sources
    manifest_path = output / "model" / "manifest.json"
    manifest = run.model.save(manifest_path.parent)
    report["surrogate_artifact_hash"] = manifest["artifact_hash"]
    report["surrogate_manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    if _validated_batch_lineage(
        args.batch_manifest,
        args.scenario_index_sha256,
        args.dataset,
        args.manifest,
    ) != scenario_batch:
        raise RuntimeError("scenario batch changed before training receipt")
    _verify_executed_sources(executed_sources)
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "model_artifact_hash": manifest["artifact_hash"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
