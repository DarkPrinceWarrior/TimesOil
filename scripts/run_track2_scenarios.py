#!/usr/bin/env python3
"""Run generated Model Z Track 2 scenarios through the authenticated OPM path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from timesoil.aios import opm as _opm_module
from timesoil.aios import opm_chdd as _opm_chdd_module
from timesoil.aios.opm import OpmFlowRunner, _sha256_file, _source_digest
from timesoil.aios.opm_chdd import _deck_text, export_opm_chdd


_DIGEST = re.compile(r"[0-9a-f]{64}")
_SCENARIO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MAX_JSON_BYTES = 4 * 1024**2
_MAX_ARTIFACT_BYTES = 64 * 1024**2
_MAX_SOURCE_BYTES = 4 * 1024**2
_MAX_SCENARIOS = 1_000
_EXPECTED_SCENARIO_SETS = {
    4: ("baseline", *(f"perturbation-{index:03d}" for index in range(1, 4))),
    10: ("baseline", *(f"perturbation-{index:03d}" for index in range(1, 10))),
}
_EXPECTED_SCENARIO_IDS = _EXPECTED_SCENARIO_SETS[4]


@dataclass(frozen=True, slots=True)
class ScenarioInput:
    scenario_id: str
    manifest_sha256: str
    actions_sha256: str
    modified_schedule: bytes
    modified_schedule_sha256: str
    controls_sha256: str
    schedule_mode: str


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink path component is forbidden: {current}")


def _read_regular(path: Path, label: str, *, limit: int) -> bytes:
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
    for module_file in (__file__, _opm_module.__file__, _opm_chdd_module.__file__):
        if not isinstance(module_file, str) or not module_file:
            raise ValueError("executed source module has no __file__")
        path = Path(module_file).absolute()
        data = _read_regular(path, "executed source", limit=_MAX_SOURCE_BYTES)
        sources.append({"path": str(path), "sha256": _sha256(data)})
    return sources


def _verify_executed_sources(expected: list[dict[str, str]]) -> None:
    if _snapshot_executed_sources() != expected:
        raise RuntimeError("executed source changed during scenario run")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read_regular(path, label, limit=_MAX_JSON_BYTES)
    try:
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, data


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys mismatch: expected {sorted(expected)}, got {sorted(value)}"
        )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def _artifact(
    root: Path, scenario_id: str, description: dict[str, Any], label: str
) -> tuple[bytes, str]:
    _exact_keys(description, {"path", "sha256", "size_bytes"}, label)
    relative = _relative(description["path"], f"{label} path")
    path = root / scenario_id / Path(*relative.parts)
    data = _read_regular(path, label, limit=_MAX_ARTIFACT_BYTES)
    expected_hash = _digest(description["sha256"], f"{label} sha256")
    size = description["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label} size_bytes must be a non-negative integer")
    if len(data) != size or _sha256(data) != expected_hash:
        raise ValueError(f"{label} hash or size mismatch")
    return data, expected_hash


def _load_bundle(root: Path) -> tuple[dict[str, Any], bytes, tuple[ScenarioInput, ...]]:
    _reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("scenario bundle must be a regular non-symlink directory")
    index, index_bytes = _json_file(root / "index.json", "scenario index")
    _exact_keys(
        index,
        {"schema_version", "generator", "inputs", "scenario_count", "scenarios"},
        "scenario index",
    )
    if index["schema_version"] != 1 or index["generator"] != "scripts/generate_track2_scenarios.py":
        raise ValueError("unsupported scenario index generator or schema")
    inputs = index["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("scenario index inputs must be an object")
    _exact_keys(inputs, {"baseline_csv", "schedule_include"}, "scenario index inputs")
    for name, item in inputs.items():
        if not isinstance(item, dict):
            raise ValueError(f"scenario index input {name} must be an object")
        _exact_keys(item, {"name", "sha256"}, f"scenario index input {name}")
        input_name = _relative(item["name"], f"scenario index input {name} name")
        if len(input_name.parts) != 1:
            raise ValueError(f"scenario index input {name} name must be a basename")
        _digest(item["sha256"], f"scenario index input {name} sha256")

    entries = index["scenarios"]
    count = index["scenario_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= _MAX_SCENARIOS
        or not isinstance(entries, list)
        or len(entries) != count
    ):
        raise ValueError("scenario_count must match a non-empty bounded scenarios list")

    scenarios: list[ScenarioInput] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries):
        label = f"scenario index entry {position}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be an object")
        _exact_keys(
            entry,
            {
                "scenario_id",
                "manifest",
                "manifest_sha256",
                "actions_sha256",
                "generator_parameters",
            },
            label,
        )
        scenario_id = entry["scenario_id"]
        if (
            not isinstance(scenario_id, str)
            or _SCENARIO_ID.fullmatch(scenario_id) is None
            or scenario_id in seen
        ):
            raise ValueError(f"unsafe or duplicate scenario_id: {scenario_id!r}")
        seen.add(scenario_id)
        manifest_relative = _relative(entry["manifest"], f"{label} manifest")
        if manifest_relative != PurePosixPath(scenario_id, "manifest.json"):
            raise ValueError(f"{label} manifest must be {scenario_id}/manifest.json")
        manifest, manifest_bytes = _json_file(
            root / Path(*manifest_relative.parts), f"scenario {scenario_id} manifest"
        )
        manifest_sha = _digest(entry["manifest_sha256"], f"{label} manifest_sha256")
        actions_sha = _digest(entry["actions_sha256"], f"{label} actions_sha256")
        if _sha256(manifest_bytes) != manifest_sha:
            raise ValueError(f"scenario {scenario_id} manifest hash mismatch")
        _exact_keys(
            manifest,
            {
                "schema_version",
                "scenario_id",
                "generator_parameters",
                "inputs",
                "controls",
                "artifacts",
                "overlay",
            },
            f"scenario {scenario_id} manifest",
        )
        if manifest["schema_version"] != 1 or manifest["scenario_id"] != scenario_id:
            raise ValueError(f"scenario {scenario_id} manifest identity mismatch")
        if manifest["generator_parameters"] != entry["generator_parameters"]:
            raise ValueError(f"scenario {scenario_id} generator parameters mismatch")

        manifest_inputs = manifest["inputs"]
        if not isinstance(manifest_inputs, dict):
            raise ValueError(f"scenario {scenario_id} inputs must be an object")
        _exact_keys(
            manifest_inputs,
            {"baseline_csv", "schedule_include", "known_wells"},
            f"scenario {scenario_id} inputs",
        )
        for name in ("baseline_csv", "schedule_include"):
            if manifest_inputs[name] != inputs[name]:
                raise ValueError(f"scenario {scenario_id} input {name} disagrees with index")
        known_wells = manifest_inputs["known_wells"]
        if (
            not isinstance(known_wells, list)
            or not known_wells
            or any(not isinstance(item, str) or not item for item in known_wells)
            or len(known_wells) != len(set(known_wells))
        ):
            raise ValueError(f"scenario {scenario_id} known_wells is invalid")

        controls = manifest["controls"]
        if not isinstance(controls, dict):
            raise ValueError(f"scenario {scenario_id} controls must be an object")
        _exact_keys(
            controls,
            {"action_count", "actions_sha256", "months"},
            f"scenario {scenario_id} controls",
        )
        if controls["actions_sha256"] != actions_sha:
            raise ValueError(f"scenario {scenario_id} actions hash mismatch")
        if (
            isinstance(controls["action_count"], bool)
            or not isinstance(controls["action_count"], int)
            or controls["action_count"] < 1
            or not isinstance(controls["months"], list)
            or not controls["months"]
        ):
            raise ValueError(f"scenario {scenario_id} controls metadata is invalid")

        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError(f"scenario {scenario_id} artifacts must be an object")
        _exact_keys(
            artifacts,
            {"modified_schedule", "wells_schedule"},
            f"scenario {scenario_id} artifacts",
        )
        modified, modified_sha = _artifact(
            root,
            scenario_id,
            artifacts["modified_schedule"],
            f"scenario {scenario_id} modified schedule",
        )
        _, controls_sha = _artifact(
            root,
            scenario_id,
            artifacts["wells_schedule"],
            f"scenario {scenario_id} wells schedule",
        )
        if artifacts["modified_schedule"]["path"] != inputs["schedule_include"]["name"]:
            raise ValueError(f"scenario {scenario_id} modified schedule name mismatch")
        if artifacts["wells_schedule"]["path"] != "wells_schedule.inc":
            raise ValueError(f"scenario {scenario_id} wells_schedule path mismatch")

        overlay = manifest["overlay"]
        if not isinstance(overlay, dict):
            raise ValueError(f"scenario {scenario_id} overlay must be an object")
        _exact_keys(
            overlay,
            {
                "source_sha256",
                "controls_sha256",
                "output_sha256",
                "mode",
                "action_count",
                "action_months",
                "truncated_after",
            },
            f"scenario {scenario_id} overlay",
        )
        expected_mode = "identity" if scenario_id == "baseline" else "full"
        if (
            overlay["mode"] != expected_mode
            or overlay["truncated_after"] is not None
            or overlay["source_sha256"] != inputs["schedule_include"]["sha256"]
            or overlay["controls_sha256"] != controls_sha
            or overlay["output_sha256"] != modified_sha
            or overlay["action_count"] != controls["action_count"]
            or overlay["action_months"] != controls["months"]
            or (expected_mode == "identity" and modified_sha != inputs["schedule_include"]["sha256"])
        ):
            raise ValueError(f"scenario {scenario_id} schedule provenance mismatch")
        scenarios.append(
            ScenarioInput(
                scenario_id,
                manifest_sha,
                actions_sha,
                modified,
                modified_sha,
                controls_sha,
                expected_mode,
            )
        )
    return index, index_bytes, tuple(scenarios)


def _write_new_json(path: Path, value: Any) -> None:
    data = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _run_batch(args: argparse.Namespace) -> Path:
    executed_sources = _snapshot_executed_sources()
    source = args.source.absolute()
    bundle = args.scenario_bundle.absolute()
    output = args.output_dir.absolute()
    official_sha = _digest(args.source_sha256, "official source SHA-256")
    expected_index_sha = _digest(
        args.scenario_index_sha256, "scenario index SHA-256"
    )
    baseline_chdd_sha = _digest(
        args.baseline_chdd_sha256, "baseline CHDD SHA-256"
    )
    schedule_relative = _relative(
        args.schedule_relative_path, "schedule-relative-path"
    )
    _reject_symlink_components(source)
    if source.is_symlink() or not source.exists():
        raise ValueError("source must exist and must not be a symlink")
    if _source_digest(source) != official_sha:
        raise ValueError("official source SHA-256 mismatch")
    index, index_bytes, scenarios = _load_bundle(bundle)
    index_sha = _sha256(index_bytes)
    if index_sha != expected_index_sha:
        raise ValueError("scenario index SHA-256 disagrees with authenticated input")
    scenario_ids = tuple(scenario.scenario_id for scenario in scenarios)
    if scenario_ids != _EXPECTED_SCENARIO_SETS.get(len(scenario_ids)):
        raise ValueError(
            "scenario bundle must contain the authenticated Track 2 v2 scenario set "
            "or v3 conformal 10-scenario set"
        )
    conformal_batch = len(scenario_ids) == 10
    schedule_input_sha = _digest(
        index["inputs"]["schedule_include"]["sha256"],
        "schedule input SHA-256",
    )
    if PurePosixPath(index["inputs"]["schedule_include"]["name"]).name != schedule_relative.name:
        raise ValueError("documented schedule path name disagrees with scenario bundle")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory {output}")
    parent = output.parent
    _reject_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(parent)
    output.mkdir()

    runner = OpmFlowRunner(timeout_seconds=args.timeout_seconds)
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        run_dir = output / scenario.scenario_id
        prepared = runner.prepare(source, run_dir, deck=args.deck)
        if prepared.source_sha256 != official_sha:
            raise RuntimeError("prepared source SHA-256 disagrees with official source")
        schedule_path = prepared.input_dir / Path(*schedule_relative.parts)
        _reject_symlink_components(schedule_path)
        original = _read_regular(
            schedule_path, "prepared schedule include", limit=_MAX_ARTIFACT_BYTES
        )
        if _sha256(original) != schedule_input_sha:
            raise ValueError("prepared schedule include hash disagrees with scenario source")
        _, used_deck_files = _deck_text(prepared.deck_path.parent)
        if schedule_path.resolve() not in set(used_deck_files):
            raise ValueError("documented schedule include is not reachable from selected deck")
        schedule_path.write_bytes(scenario.modified_schedule)
        if _sha256_file(schedule_path) != scenario.modified_schedule_sha256:
            raise RuntimeError("prepared scenario schedule hash mismatch after replacement")
        binding = prepared.run_dir / "scenario-binding.json"
        _write_new_json(
            binding,
            {
                "schema": "timesoil.aios.track2-scenario-binding/v1",
                "scenario_id": scenario.scenario_id,
                "official_source_sha256": official_sha,
                "scenario_index_sha256": index_sha,
                "scenario_manifest_sha256": scenario.manifest_sha256,
                "actions_sha256": scenario.actions_sha256,
                "controls_sha256": scenario.controls_sha256,
                "schedule": {
                    "path": str(schedule_relative),
                    "source_sha256": schedule_input_sha,
                    "output_sha256": scenario.modified_schedule_sha256,
                    "mode": scenario.schedule_mode,
                    "full_overlay": scenario.schedule_mode == "full",
                },
            },
        )
        result = runner._run_prepared(
            prepared, parsing_strictness=args.parsing_strictness
        )
        report, extraction = runner.extract_summary_report(
            result, result.run_dir / "summary-report.txt"
        )
        chdd = result.run_dir / "canonical" / "chdd.csv"
        track2 = output / "dataset" / f"{scenario.scenario_id}.csv"
        export_manifest = output / "manifests" / f"{scenario.scenario_id}.json"
        export_opm_chdd(
            report,
            chdd,
            track2,
            export_manifest,
            scenario_id=scenario.scenario_id,
            source_model="model_z_opm",
            opm_run_manifest=result.manifest_path,
            summary_extraction_manifest=extraction,
            deck_dir=result.deck_path.parent,
            unit_system=prepared.unit_system,
        )
        chdd_sha = _sha256_file(chdd)
        if scenario.scenario_id == "baseline" and chdd_sha != baseline_chdd_sha:
            raise RuntimeError("identity baseline CHDD disagrees with authenticated reference")
        records.append(
            {
                "scenario_id": scenario.scenario_id,
                "actions_sha256": scenario.actions_sha256,
                "run_manifest": str(result.manifest_path.relative_to(output)),
                "run_manifest_sha256": _sha256_file(result.manifest_path),
                "summary_report_sha256": _sha256_file(report),
                "summary_extraction_sha256": _sha256_file(extraction),
                "canonical_chdd": str(chdd.relative_to(output)),
                "canonical_chdd_sha256": chdd_sha,
                "dataset": str(track2.relative_to(output)),
                "dataset_sha256": _sha256_file(track2),
                "export_manifest": str(export_manifest.relative_to(output)),
                "export_manifest_sha256": _sha256_file(export_manifest),
            }
        )

    batch_manifest = output / "manifest.json"
    _verify_executed_sources(executed_sources)
    _write_new_json(
        batch_manifest,
        {
            "schema": (
                "timesoil.aios.track2-scenario-run/v2"
                if conformal_batch
                else "timesoil.aios.track2-scenario-run/v1"
            ),
            "executed_sources": executed_sources,
            "official_source_sha256": official_sha,
            "scenario_index_sha256": index_sha,
            "scenario_count": len(records),
            "sequential": True,
            "scenarios": records,
            "training": {
                "dataset": "dataset",
                "manifests": "manifests",
                "argv": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_track2_surrogate.py",
                    "--dataset",
                    str(output / "dataset"),
                    "--manifest",
                    str(output / "manifests"),
                    "--output",
                    str(output / "surrogate"),
                    "--test-fraction",
                    "0.25",
                    "--ensemble-size",
                    "5",
                    "--n-estimators",
                    "160",
                    "--horizon",
                    "6",
                    "--seed",
                    "20260831",
                    *(
                        ["--conformal-level", "0.9"]
                        if conformal_batch
                        else []
                    ),
                ],
            },
        },
    )
    return batch_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a verified generated Model Z scenario bundle sequentially in OPM."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("scenario_bundle", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--scenario-index-sha256", required=True)
    parser.add_argument("--baseline-chdd-sha256", required=True)
    parser.add_argument("--schedule-relative-path", required=True)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--parsing-strictness", choices=("strict", "low"), default="strict"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = _run_batch(args)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"completed verified Track 2 scenarios: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
