#!/usr/bin/env python3
"""Assemble verified full-cycle runs into a Track 2 scenario batch for fine-tuning.

The batch directory is the directory the runs were *executed* into. OPM summary
extraction records the absolute bind mount of ``<run>/output`` inside its own
extraction manifest, so a run cannot be relocated without invalidating the chain
``verify_summary_extraction`` replays. The assembler therefore never moves a run:
it adds ``dataset/``, ``manifests/`` and ``manifest.json`` next to the existing run
directories and refuses to overwrite any of the three.

Layout expected in ``BATCH_DIR`` (one ``timesoil-aios full-cycle`` run per
subdirectory, the incumbent run carrying ``scenario_id == "baseline"``)::

    <run>/full-cycle-receipt.json
    <run>/manifest.json  summary-report.txt  summary-extraction.json
    <run>/canonical/{chdd.csv,trajectory.csv,manifest.json}

Layout written by this script, accepted by
``scripts/finetune_timesfm_head.py::verified_batch``::

    dataset/<scenario_id>.csv        hard link to <run>/canonical/trajectory.csv
    manifests/<scenario_id>.json     <run>/canonical/manifest.json, re-anchored
    manifest.json                    the batch manifest
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

import numpy as np
import pandas as pd

from benchmark_bhp_surrogate import START
from timesoil.aios.opm import _sha256_file
from timesoil.aios.surrogate import STATE_FEATURES
from timesoil.aios.track2 import MODEL_Z_SOURCE_SHA256

RECEIPT_NAME = "full-cycle-receipt.json"
RECEIPT_SCHEMA = "timesoil.aios.track2-full-cycle/v1"
BATCH_SCHEMA = "timesoil.aios.track2-scenario-run/v2"
BASELINE_ID = "baseline"
RESERVED_NAMES = frozenset({"dataset", "manifests"})
RECEIPT_ARTIFACTS = (
    "opm_run_manifest",
    "summary_report",
    "summary_extraction_manifest",
    "canonical_export_manifest",
    "canonical_chdd_csv",
    "canonical_trajectory_csv",
)
# Export-manifest links resolved against <run>/canonical, re-anchored to <batch>/manifests.
EXPORT_LINKS = (
    ("provenance", "opm_run_manifest"),
    ("provenance", "summary_extraction_manifest"),
    ("source", "summary_csv"),
)
_ACTION_COLUMNS = ("control_value", "control_target", "status")
_LABEL_COLUMNS = frozenset({"date", "well", "control_target"})
_HISTORY_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class _Run:
    directory: Path
    scenario_id: str
    actions_sha256: str
    artifacts: dict[str, tuple[Path, str]]
    export_manifest: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


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


def _within(root: Path, base: Path, value: Any, label: str) -> Path:
    """Resolve a relative POSIX link and require a regular file inside ``root``."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must not be absolute: {value!r}")
    path = (base / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the run directory: {value!r}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path


def _load_run(directory: Path) -> _Run:
    """Authenticate one full-cycle run against its own receipt."""
    label = directory.name
    receipt_path = directory / RECEIPT_NAME
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError(f"{label}: no {RECEIPT_NAME}; the run is incomplete")
    receipt = _json_object(receipt_path, f"{label} full-cycle receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"{label}: unsupported full-cycle receipt schema")
    if receipt.get("complete") is not True:
        raise ValueError(f"{label}: full-cycle receipt is not a complete production run")
    source = _digest(receipt.get("source_sha256"), f"{label} source_sha256")
    if source != MODEL_Z_SOURCE_SHA256:
        raise ValueError(
            f"{label}: run source {source} is not the pinned case archive "
            f"{MODEL_Z_SOURCE_SHA256}"
        )
    controls = receipt.get("controls")
    if not isinstance(controls, dict):
        raise ValueError(f"{label}: full-cycle receipt has no controls evidence")
    actions_sha256 = _digest(
        controls.get("canonical_actions_sha256"), f"{label} canonical_actions_sha256"
    )

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{label}: full-cycle receipt has no artifact inventory")
    resolved: dict[str, tuple[Path, str]] = {}
    for name in RECEIPT_ARTIFACTS:
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"{label}: receipt artifact {name} is missing")
        path = _within(directory, directory, entry.get("path"), f"{label} {name}")
        expected = _digest(entry.get("sha256"), f"{label} {name} sha256")
        if _sha256_file(path) != expected:
            raise ValueError(f"{label}: {name} does not match the receipt hash")
        resolved[name] = (path, expected)

    export_path, _ = resolved["canonical_export_manifest"]
    export = _json_object(export_path, f"{label} canonical export manifest")
    scenario = export.get("scenario")
    outputs = export.get("outputs")
    if not isinstance(scenario, dict) or not isinstance(outputs, dict):
        raise ValueError(f"{label}: canonical export manifest is incomplete")
    track2 = outputs.get("track2_csv")
    if not isinstance(track2, dict):
        raise ValueError(f"{label}: canonical export manifest has no track2_csv output")
    scenario_id = scenario.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id or "/" in scenario_id or scenario_id in {".", ".."}:
        raise ValueError(f"{label}: canonical export manifest has no usable scenario_id")
    if scenario.get("source_model") != "model_z_opm":
        raise ValueError(f"{label}: scenario source_model is not model_z_opm")
    if track2.get("sha256") != resolved["canonical_trajectory_csv"][1]:
        raise ValueError(f"{label}: canonical trajectory hash disagrees with its export manifest")
    return _Run(directory, scenario_id, actions_sha256, resolved, export)


def _reanchored(run: _Run, manifests_dir: Path, dataset_name: str) -> dict[str, Any]:
    """Rename the exported dataset and re-point its links at the batch manifests dir."""
    value = deepcopy(run.export_manifest)
    value["outputs"]["track2_csv"]["name"] = dataset_name
    canonical = run.artifacts["canonical_export_manifest"][0].parent
    for section, key in EXPORT_LINKS:
        target = _within(
            run.directory, canonical, value[section][key], f"{run.scenario_id} {key}"
        )
        value[section][key] = Path(
            os.path.relpath(target, manifests_dir.resolve())
        ).as_posix()
    return value


def _history(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The pre-origin states (through START) and actions (before START) of one dataset."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "well"], kind="stable").reset_index(drop=True)
    states = frame.loc[frame["date"] <= START, ["date", "well", *STATE_FEATURES]]
    actions = frame.loc[frame["date"] < START, ["date", "well", *_ACTION_COLUMNS]]
    return states.reset_index(drop=True), actions.reset_index(drop=True)


def _history_differs(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return True
    labels = [column for column in left.columns if column in _LABEL_COLUMNS]
    if not left[labels].astype(str).equals(right[labels].astype(str)):
        return True
    numeric = [column for column in left.columns if column not in _LABEL_COLUMNS]
    return not np.allclose(
        left[numeric].to_numpy(float),
        right[numeric].to_numpy(float),
        rtol=0.0,
        atol=_HISTORY_TOLERANCE,
        equal_nan=True,
    )


def _link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _executed_sources() -> list[dict[str, str]]:
    from timesoil.aios import opm, opm_chdd, track2

    files = [__file__, opm.__file__, opm_chdd.__file__, track2.__file__]
    records = []
    for module_file in files:
        if not isinstance(module_file, str) or not module_file:
            raise ValueError("executed source module has no __file__")
        path = Path(module_file).absolute()
        records.append({"path": str(path), "sha256": _sha256_file(path)})
    return records


def _ordered_runs(batch: Path, selected: list[str]) -> list[_Run]:
    if selected:
        names = selected
        if len(set(names)) != len(names):
            raise ValueError("--run was given the same directory twice")
        directories = []
        for name in names:
            directory = batch / name
            if "/" in name or name in {".", ".."} or not directory.is_dir():
                raise ValueError(f"--run {name!r} is not a directory inside the batch")
            directories.append(directory)
    else:
        directories = sorted(
            item
            for item in batch.iterdir()
            if item.is_dir()
            and not item.is_symlink()
            and item.name not in RESERVED_NAMES
            and (item / RECEIPT_NAME).is_file()
        )
    if not directories:
        raise ValueError(f"no full-cycle runs with a {RECEIPT_NAME} in {batch}")
    runs = [_load_run(directory) for directory in directories]
    identifiers = [run.scenario_id for run in runs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate scenario_id across runs: {sorted(identifiers)}")
    if BASELINE_ID not in set(identifiers):
        raise ValueError(
            f"no run carries scenario_id {BASELINE_ID!r}; the incumbent full-cycle run "
            f"must be requested with that scenario_id (found: {sorted(identifiers)})"
        )
    return sorted(runs, key=lambda run: (run.scenario_id != BASELINE_ID, run.scenario_id))


def _assemble(batch: Path, runs: list[_Run]) -> Path:
    dataset_dir = batch / "dataset"
    manifests_dir = batch / "manifests"
    batch_manifest = batch / "manifest.json"
    for target in (dataset_dir, manifests_dir, batch_manifest):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite {target}")

    histories = {
        run.scenario_id: _history(run.artifacts["canonical_trajectory_csv"][0])
        for run in runs
    }
    baseline = histories[BASELINE_ID]
    for run in runs:
        if run.scenario_id == BASELINE_ID:
            continue
        states, actions = histories[run.scenario_id]
        if _history_differs(states, baseline[0]) or _history_differs(actions, baseline[1]):
            raise ValueError(
                f"{run.scenario_id}: history before {START.date()} differs from the baseline run"
            )

    dataset_dir.mkdir(exist_ok=False)
    manifests_dir.mkdir(exist_ok=False)
    records: list[dict[str, Any]] = []
    for run in runs:
        dataset_name = f"{run.scenario_id}.csv"
        dataset_path = dataset_dir / dataset_name
        manifest_path = manifests_dir / f"{run.scenario_id}.json"
        _link(run.artifacts["canonical_trajectory_csv"][0], dataset_path)
        with manifest_path.open("xb") as stream:
            stream.write(_canonical_json(_reanchored(run, manifests_dir, dataset_name)))
        run_manifest, run_manifest_sha = run.artifacts["opm_run_manifest"]
        chdd, chdd_sha = run.artifacts["canonical_chdd_csv"]
        records.append({
            "scenario_id": run.scenario_id,
            "run_directory": run.directory.name,
            "dataset": f"dataset/{dataset_name}",
            "dataset_sha256": _sha256_file(dataset_path),
            "export_manifest": f"manifests/{run.scenario_id}.json",
            "export_manifest_sha256": _sha256_file(manifest_path),
            "source_export_manifest_sha256": run.artifacts["canonical_export_manifest"][1],
            "run_manifest": os.path.relpath(run_manifest, batch),
            "run_manifest_sha256": run_manifest_sha,
            "canonical_chdd": os.path.relpath(chdd, batch),
            "canonical_chdd_sha256": chdd_sha,
            "actions_sha256": run.actions_sha256,
            "summary_extraction_sha256": run.artifacts["summary_extraction_manifest"][1],
            "summary_report_sha256": run.artifacts["summary_report"][1],
        })

    index = _canonical_json([
        {"scenario_id": record["scenario_id"], "actions_sha256": record["actions_sha256"]}
        for record in records
    ])
    with batch_manifest.open("xb") as stream:
        stream.write(_canonical_json({
            "schema": BATCH_SCHEMA,
            "assembler": "scripts/assemble_scenario_batch.py",
            # No OPM ran here: the scenarios were simulated by timesoil-aios full-cycle.
            "execution_mode": "external-full-cycle-runs",
            "executed_sources": _executed_sources(),
            "official_source_sha256": MODEL_Z_SOURCE_SHA256,
            "scenario_index_sha256": sha256(index).hexdigest(),
            "scenario_index_mode": "assembled-from-full-cycle-receipts",
            "scenario_count": len(records),
            "scenarios": records,
        }))
    return batch_manifest


def _dry_run(batch: Path, runs: list[_Run]) -> None:
    for target in ("dataset", "manifests", "manifest.json"):
        state = "EXISTS (assembly would refuse)" if (batch / target).exists() else "new"
        print(f"{target}: {state}")
    print(f"scenario_count: {len(runs)}")
    for run in runs:
        print(
            f"{run.scenario_id}\t{run.directory.name}\t"
            f"dataset/{run.scenario_id}.csv\tmanifests/{run.scenario_id}.json\t"
            f"{os.path.relpath(run.artifacts['opm_run_manifest'][0], batch)}\t"
            f"{os.path.relpath(run.artifacts['canonical_chdd_csv'][0], batch)}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch_dir", type=Path,
                        help="directory holding the executed full-cycle run directories")
    parser.add_argument("--run", action="append", default=[], metavar="NAME",
                        help="restrict assembly to this run subdirectory (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the scenarios and target paths without writing anything")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    batch = args.batch_dir.absolute()
    try:
        if batch.is_symlink() or not batch.is_dir():
            raise ValueError(f"batch directory must be a regular directory: {batch}")
        runs = _ordered_runs(batch, args.run)
        if args.dry_run:
            _dry_run(batch, runs)
            return 0
        manifest = _assemble(batch, runs)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"assembled Track 2 scenario batch: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
