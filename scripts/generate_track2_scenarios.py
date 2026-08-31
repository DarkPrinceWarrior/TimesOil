#!/usr/bin/env python3
"""Prepare deterministic Track 2 schedule inputs for subsequent real OPM runs."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from timesoil.aios.scenario_generation import (
    ScenarioGeneratorConfig,
    generate_control_scenarios,
    load_control_csv,
    load_control_records,
)
from timesoil.aios.schedule_overlay import _canonical_schedule, apply_schedule_overlay
from timesoil.aios.track2 import CANONICAL_COLUMNS


_WELL = re.compile(r"[A-Za-z0-9_.-]+")
_CONTROL_KEYWORDS = {"WCONPROD", "WCONINJE"}
_WELL_RECORD = re.compile(r"\s*'([^']+)'(?:\s|$)")
_CONTROL_FIELDS = ("date", "well", "control_value", "control_target", "status")
_BASELINE_NAME = "model-z-baseline-controls.csv"


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _regular_file(path: Path, label: str) -> bytes:
    _reject_symlink_components(path.absolute())
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _baseline_controls(path: Path, source: bytes) -> tuple[bytes, tuple[Any, ...], str]:
    try:
        reader = csv.DictReader(StringIO(source.decode("utf-8"), newline=""), strict=True)
        header = reader.fieldnames
        if header is None or len(header) != len(set(header)):
            raise ValueError("baseline CSV has an empty or duplicate header")
        fields = set(header)
        if fields == set(_CONTROL_FIELDS):
            return source, load_control_csv(path), path.name
        if fields != set(CANONICAL_COLUMNS):
            raise ValueError(
                f"CSV header must contain exactly {_CONTROL_FIELDS} or {CANONICAL_COLUMNS}"
            )
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("baseline CSV must be valid UTF-8 CSV") from exc

    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError("canonical baseline CSV is empty or has incomplete rows")
    scenario_ids = {row["scenario_id"] for row in rows}
    if len(scenario_ids) != 1 or not next(iter(scenario_ids)).strip():
        raise ValueError("canonical baseline CSV must contain exactly one scenario_id")
    if {row["source_model"] for row in rows} != {"model_z_opm"}:
        raise ValueError("canonical baseline CSV source_model must be model_z_opm")

    projected: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=1):
        try:
            status = float(row["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"canonical baseline row {number} status must be 0 or 1") from exc
        if status not in (0.0, 1.0):
            raise ValueError(f"canonical baseline row {number} status must be 0 or 1")
        projected.append(
            {
                **{field: row[field] for field in _CONTROL_FIELDS[:-1]},
                "status": "OPEN" if status == 1.0 else "SHUT",
            }
        )

    output = StringIO(newline="")
    writer = csv.DictWriter(output, _CONTROL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in projected:
        writer.writerow(
            {
                "date": row["date"],
                "well": row["well"],
                "control_value": row["control_value"],
                "control_target": row["control_target"],
                "status": row["status"],
            }
        )
    return output.getvalue().encode("utf-8"), load_control_records(projected), _BASELINE_NAME


def _schedule_wells(source: str) -> tuple[str, ...]:
    wells: set[str] = set()
    block: str | None = None
    for raw in source.splitlines():
        line = raw.split("--", 1)[0].strip()
        upper = line.upper()
        if upper in _CONTROL_KEYWORDS:
            block = upper
        elif line == "/":
            block = None
        elif block and line:
            match = _WELL_RECORD.match(line)
            if not match or not _WELL.fullmatch(match.group(1)):
                raise ValueError(f"unsafe or malformed {block} well record")
            wells.add(match.group(1))
    if not wells:
        raise ValueError("schedule contains no WCONPROD/WCONINJE wells")
    return tuple(sorted(wells))


def _known_wells(source: str, explicit: Path | None) -> tuple[str, ...]:
    if explicit is None:
        return _schedule_wells(source)
    text = _regular_file(explicit, "known-wells file").decode("utf-8")
    wells = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not wells or len(wells) != len(set(wells)) or any(
        not _WELL.fullmatch(well) for well in wells
    ):
        raise ValueError("known-wells file is empty, duplicated, or unsafe")
    return tuple(sorted(wells))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink path component is forbidden: {current}")


def _atomic_new_file(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _build_outputs(
    baseline_path: Path,
    schedule_path: Path,
    known_wells_path: Path | None,
    config: ScenarioGeneratorConfig,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    baseline_bytes = _regular_file(baseline_path, "baseline CSV")
    projected_baseline, baseline_actions, baseline_name = _baseline_controls(
        baseline_path, baseline_bytes
    )
    baseline_description = {
        "name": baseline_name,
        "sha256": _sha256(projected_baseline),
    }
    schedule_bytes = _regular_file(schedule_path, "schedule include")
    if schedule_path.suffix.lower() != ".inc" or schedule_path.name == "wells_schedule.inc":
        raise ValueError("schedule input must be a non-reserved .inc file")
    try:
        source = schedule_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("schedule include must be UTF-8") from exc
    wells = _known_wells(source, known_wells_path)
    scenarios = generate_control_scenarios(baseline_actions, config)
    if _regular_file(baseline_path, "baseline CSV") != baseline_bytes:
        raise ValueError("baseline CSV changed while scenarios were generated")

    outputs: dict[Path, bytes] = {}
    index_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        controls = _canonical_schedule(scenario.actions).encode()
        controls_sha = _sha256(controls)
        months = sorted({action.month.isoformat() for action in scenario.actions})
        if scenario.scenario_id == "baseline":
            modified = schedule_bytes
            overlay_provenance = {
                "source_sha256": _sha256(schedule_bytes),
                "controls_sha256": controls_sha,
                "output_sha256": _sha256(schedule_bytes),
                "mode": "identity",
                "action_count": len(scenario.actions),
                "action_months": months,
                "truncated_after": None,
            }
        else:
            overlay = apply_schedule_overlay(source, scenario.actions, known_wells=wells)
            if controls_sha != overlay.controls_sha256:
                raise RuntimeError("canonical controls hash disagrees with schedule overlay")
            modified = overlay.text.encode()
            overlay_provenance = overlay.provenance
        base = Path(scenario.scenario_id)
        modified_path = base / schedule_path.name
        controls_path = base / "wells_schedule.inc"
        manifest_path = base / "manifest.json"
        manifest = {
            "schema_version": 1,
            "scenario_id": scenario.scenario_id,
            "generator_parameters": dict(scenario.generator_parameters),
            "inputs": {
                "baseline_csv": {
                    **baseline_description,
                },
                "schedule_include": {
                    "name": schedule_path.name,
                    "sha256": _sha256(schedule_bytes),
                },
                "known_wells": list(wells),
            },
            "controls": {
                "action_count": len(scenario.actions),
                "actions_sha256": scenario.sha256,
                "months": months,
            },
            "artifacts": {
                "modified_schedule": {
                    "path": schedule_path.name,
                    "sha256": _sha256(modified),
                    "size_bytes": len(modified),
                },
                "wells_schedule": {
                    "path": "wells_schedule.inc",
                    "sha256": _sha256(controls),
                    "size_bytes": len(controls),
                },
            },
            "overlay": overlay_provenance,
        }
        manifest_bytes = _json(manifest)
        outputs[modified_path] = modified
        outputs[controls_path] = controls
        outputs[manifest_path] = manifest_bytes
        index_scenarios.append(
            {
                "scenario_id": scenario.scenario_id,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_bytes),
                "actions_sha256": scenario.sha256,
                "generator_parameters": dict(scenario.generator_parameters),
            }
        )

    index = {
        "schema_version": 1,
        "generator": "scripts/generate_track2_scenarios.py",
        "inputs": {
            "baseline_csv": baseline_description,
            "schedule_include": {"name": schedule_path.name, "sha256": _sha256(schedule_bytes)},
        },
        "scenario_count": len(scenarios),
        "scenarios": index_scenarios,
    }
    outputs[Path("index.json")] = _json(index)
    return outputs, index


def _publish(output_dir: Path, outputs: dict[Path, bytes]) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory {output_dir}")
    parent = output_dir.parent.absolute()
    _reject_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(parent)
    output_dir.mkdir()
    for relative, data in sorted(outputs.items(), key=lambda item: str(item[0])):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe output path: {relative}")
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_new_file(destination, data)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Track 2 controls and modified schedules; does not run OPM."
    )
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("schedule_include", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--known-wells-file", type=Path)
    parser.add_argument("--scenario-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--perturbation-fraction", type=float, default=0.15)
    parser.add_argument("--liquid-rate-scale", type=float, default=1.0)
    parser.add_argument("--monthly-liquid-rate-cap", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = ScenarioGeneratorConfig(
            scenario_count=args.scenario_count,
            seed=args.seed,
            perturbation_fraction=args.perturbation_fraction,
            liquid_rate_scale=args.liquid_rate_scale,
            monthly_liquid_rate_cap=args.monthly_liquid_rate_cap,
        )
        outputs, index = _build_outputs(
            args.baseline_csv,
            args.schedule_include,
            args.known_wells_file,
            config,
        )
        _publish(args.output_dir, outputs)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"generated {index['scenario_count']} scenarios: {args.output_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
