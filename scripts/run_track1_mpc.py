#!/usr/bin/env python3
"""Run deterministic Track 1 monthly MPC through the production OPM backend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Any

from timesoil.aios.contracts import (
    Case,
    ControlAction,
    ControlTarget,
    State,
    WellRole,
    WellState,
    WellStatus,
)
from timesoil.aios.opm import OpmFlowRunner, OpmGdmBackend, _source_digest
from timesoil.aios.track1 import Candidate, GdmBackend, MonthlyMPC, Track1Result


_SCHEMA = "timesoil.aios.track1-mpc-input/v1"
_RESULT_SCHEMA = "timesoil.aios.track1-mpc-result/v1"
_MANIFEST_SCHEMA = "timesoil.aios.track1-mpc-manifest/v1"
_MAX_CONFIG_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RunConfig:
    case: Case
    initial_state: State
    candidates: dict[date, tuple[Candidate, ...]]
    source: Path
    opm_runs_dir: Path
    deck: str | None
    schedule_include: str | None
    normalize_model_y: bool
    parsing_strictness: str
    density_map: Path | None
    source_model: str | None
    timeout_seconds: float
    config_sha256: str
    source_sha256: str
    input_sha256: str

    @property
    def run_id(self) -> str:
        return f"track1-{self.input_sha256[:24]}"


def _json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _script_source_contract(proof_script: Path | None = None) -> dict[str, dict[str, str]]:
    paths = {"run_track1_mpc.py": Path(__file__).absolute()}
    if proof_script is not None:
        paths["run_model_y_track1_proof.py"] = proof_script.absolute()
    contract: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if path.name != name:
            raise ValueError(f"{name} provenance path has wrong filename: {path}")
        _reject_symlink_components(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} provenance path must be a regular non-symlink file")
        contract[name] = {"path": str(path), "sha256": _digest(path.read_bytes())}
    return contract


def _verify_script_source_contract(contract: dict[str, dict[str, str]]) -> None:
    names = set(contract)
    if "run_track1_mpc.py" not in names or not names <= {
        "run_track1_mpc.py",
        "run_model_y_track1_proof.py",
    }:
        raise RuntimeError("Track 1 script source contract has invalid entries")
    for name, item in contract.items():
        path = Path(item["path"])
        if path.name != name or (
            name == "run_track1_mpc.py" and path != Path(__file__).absolute()
        ):
            raise RuntimeError(f"{name} source contract path differs from executable")
        _reject_symlink_components(path)
        if (
            path.is_symlink()
            or not path.is_file()
            or _digest(path.read_bytes()) != item["sha256"]
        ):
            raise RuntimeError(f"{name} changed while Track 1 was running")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(
    value: Any,
    label: str,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = required - value.keys()
    unexpected = value.keys() - required - optional
    if missing or unexpected:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string array")
    return tuple(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _month(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    if result.day != 1:
        raise ValueError(f"{label} must be the first day of a month")
    return result


def _next_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


def _case(value: Any) -> Case:
    item = _object(
        value,
        "case",
        {"case_id", "start", "end", "economics_start", "producers", "injectors"},
        {"max_liquid_rate"},
    )
    max_rate = item.get("max_liquid_rate", 500.0)
    return Case(
        case_id=_string(item["case_id"], "case.case_id"),
        start=_month(item["start"], "case.start"),
        end=_month(item["end"], "case.end"),
        economics_start=_month(item["economics_start"], "case.economics_start"),
        producers=_strings(item["producers"], "case.producers"),
        injectors=_strings(item["injectors"], "case.injectors"),
        max_liquid_rate=_number(max_rate, "case.max_liquid_rate"),
    )


def _well_state(value: Any, index: int) -> WellState:
    label = f"initial_state.wells[{index}]"
    item = _object(
        value,
        label,
        {"well", "role", "active"},
        {"oil_rate", "liquid_rate", "injection_rate", "bhp"},
    )
    if not isinstance(item["active"], bool):
        raise ValueError(f"{label}.active must be boolean")
    bhp = item.get("bhp")
    return WellState(
        well=_string(item["well"], f"{label}.well"),
        role=WellRole(item["role"]),
        active=item["active"],
        oil_rate=_number(item.get("oil_rate", 0.0), f"{label}.oil_rate"),
        liquid_rate=_number(item.get("liquid_rate", 0.0), f"{label}.liquid_rate"),
        injection_rate=_number(item.get("injection_rate", 0.0), f"{label}.injection_rate"),
        bhp=None if bhp is None else _number(bhp, f"{label}.bhp"),
    )


def _state(value: Any) -> State:
    item = _object(value, "initial_state", {"case_id", "month", "restart_ref", "wells"})
    if not isinstance(item["wells"], list):
        raise ValueError("initial_state.wells must be an array")
    return State(
        case_id=_string(item["case_id"], "initial_state.case_id"),
        month=_month(item["month"], "initial_state.month"),
        restart_ref=_string(item["restart_ref"], "initial_state.restart_ref"),
        wells=tuple(_well_state(well, index) for index, well in enumerate(item["wells"])),
    )


def _action(value: Any, month: date, label: str) -> ControlAction:
    item = _object(value, label, {"well", "role", "status", "target", "value"})
    return ControlAction(
        month=month,
        well=_string(item["well"], f"{label}.well"),
        role=WellRole(item["role"]),
        status=WellStatus(item["status"]),
        target=ControlTarget(item["target"]),
        value=_number(item["value"], f"{label}.value"),
    )


def _candidates(value: Any, case: Case) -> dict[date, tuple[Candidate, ...]]:
    if not isinstance(value, dict):
        raise ValueError("candidates must be an object keyed by month")
    result: dict[date, tuple[Candidate, ...]] = {}
    for month_text, options in value.items():
        month = _month(month_text, "candidate month")
        if not isinstance(options, list) or not options:
            raise ValueError(f"candidates.{month_text} must contain options")
        parsed: list[Candidate] = []
        for option_index, option in enumerate(options):
            if not isinstance(option, list) or not option:
                raise ValueError(f"candidates.{month_text}[{option_index}] must be non-empty")
            parsed.append(
                tuple(
                    _action(
                        action,
                        month,
                        f"candidates.{month_text}[{option_index}][{action_index}]",
                    )
                    for action_index, action in enumerate(option)
                )
            )
        result[month] = tuple(parsed)

    expected: set[date] = set()
    current = case.start
    while current <= case.end:
        expected.add(current)
        current = _next_month(current)
    if result.keys() != expected:
        raise ValueError(
            "candidate months must exactly match case horizon; "
            f"expected={[item.isoformat() for item in sorted(expected)]}, "
            f"actual={[item.isoformat() for item in sorted(result)]}"
        )
    return result


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink path component is forbidden: {current}")


def load_config(path: Path) -> RunConfig:
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("config must be a regular non-symlink file")
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("config exceeds 16 MiB limit")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("config must be unique-key UTF-8 JSON") from exc
    root = _object(payload, "config", {"schema", "case", "initial_state", "candidates", "opm"})
    if root["schema"] != _SCHEMA:
        raise ValueError(f"config.schema must equal {_SCHEMA}")
    case = _case(root["case"])
    state = _state(root["initial_state"])
    if state.case_id != case.case_id or state.month != case.start:
        raise ValueError("initial_state must match case id and start month")
    opm = _object(
        root["opm"],
        "opm",
        {"source"},
        {
            "runs_dir",
            "deck",
            "schedule_include",
            "normalize_model_y",
            "parsing_strictness",
            "density_map",
            "source_model",
            "timeout_seconds",
        },
    )
    source = (path.absolute().parent / _string(opm["source"], "opm.source")).absolute()
    _reject_symlink_components(source)
    if source.is_symlink() or not (source.is_file() or source.is_dir()):
        raise ValueError("opm.source must be a regular non-symlink file or directory")
    runs_dir = (
        path.absolute().parent
        / _string(opm.get("runs_dir", "opm-runs"), "opm.runs_dir")
    ).absolute()
    _reject_symlink_components(runs_dir)
    normalize_model_y = opm.get("normalize_model_y", False)
    if not isinstance(normalize_model_y, bool):
        raise ValueError("opm.normalize_model_y must be boolean")
    parsing_strictness = opm.get("parsing_strictness", "strict")
    if parsing_strictness not in {"strict", "low"}:
        raise ValueError("opm.parsing_strictness must be 'strict' or 'low'")
    deck = opm.get("deck")
    schedule_include = opm.get("schedule_include")
    source_model = opm.get("source_model")
    for value, label in (
        (deck, "opm.deck"),
        (schedule_include, "opm.schedule_include"),
        (source_model, "opm.source_model"),
    ):
        if value is not None:
            _string(value, label)
    density_map_raw = opm.get("density_map")
    density_map = None
    if density_map_raw is not None:
        density_map = (
            path.absolute().parent / _string(density_map_raw, "opm.density_map")
        ).absolute()
        _reject_symlink_components(density_map)
        if density_map.is_symlink() or not density_map.is_file():
            raise ValueError("opm.density_map must be a regular non-symlink file")
    timeout = _number(opm.get("timeout_seconds", 3600.0), "opm.timeout_seconds")
    config_sha256 = _digest(_json(payload))
    source_sha256 = _source_digest(source)
    input_sha256 = _digest(
        _json({"config_sha256": config_sha256, "source_sha256": source_sha256})
    )
    return RunConfig(
        case=case,
        initial_state=state,
        candidates=_candidates(root["candidates"], case),
        source=source,
        opm_runs_dir=runs_dir,
        deck=deck,
        schedule_include=schedule_include,
        normalize_model_y=normalize_model_y,
        parsing_strictness=parsing_strictness,
        density_map=density_map,
        source_model=source_model,
        timeout_seconds=timeout,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        input_sha256=input_sha256,
    )


def build_backend(config: RunConfig) -> OpmGdmBackend:
    """Only production backend factory exposed by this operator command."""

    return OpmGdmBackend(
        OpmFlowRunner(timeout_seconds=config.timeout_seconds),
        config.source,
        runs_dir=config.opm_runs_dir,
        deck=config.deck,
        schedule_include=config.schedule_include,
        normalize_model_y=config.normalize_model_y,
        parsing_strictness=config.parsing_strictness,
        density_map=config.density_map,
        source_model=config.source_model,
    )


def _action_payload(action: ControlAction) -> dict[str, Any]:
    return {
        "month": action.month.isoformat(),
        "well": action.well,
        "role": action.role.value,
        "status": action.status.value,
        "target": action.target.value,
        "value": action.value,
    }


def _state_payload(state: State) -> dict[str, Any]:
    return {
        "case_id": state.case_id,
        "month": state.month.isoformat(),
        "restart_ref": state.restart_ref,
        "wells": [
            {
                "well": well.well,
                "role": well.role.value,
                "active": well.active,
                "oil_rate": well.oil_rate,
                "liquid_rate": well.liquid_rate,
                "injection_rate": well.injection_rate,
                "bhp": well.bhp,
            }
            for well in state.wells
        ],
    }


def execute(
    config: RunConfig,
    backend: GdmBackend,
    *,
    script_source_contract: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    source_contract = script_source_contract or _script_source_contract()
    result: Track1Result = MonthlyMPC(backend).run(
        config.case,
        config.initial_state,
        lambda state: config.candidates[state.month],
    )
    if _source_digest(config.source) != config.source_sha256:
        raise RuntimeError("OPM source changed while Track 1 was running")
    _verify_script_source_contract(source_contract)
    payload = {
        "schema": _RESULT_SCHEMA,
        "run_id": config.run_id,
        "input_sha256": config.input_sha256,
        "config_sha256": config.config_sha256,
        "source_sha256": config.source_sha256,
        "script_source_contract": source_contract,
        "case_id": config.case.case_id,
        "schedule": {
            "sha256": result.schedule.sha256,
            "text": result.schedule.text,
            "actions": [_action_payload(action) for action in result.schedule.actions],
        },
        "evidence": {
            "backend_provenance": result.evidence.backend_provenance,
            "trajectories": [
                {
                    "run_id": trajectory.run_id,
                    "month": trajectory.month.isoformat(),
                    "simulator": trajectory.simulator,
                    "certified": trajectory.certified,
                    "chdd_complete": trajectory.chdd_complete,
                    "invariant_violations": list(trajectory.invariant_violations),
                    "actions": [_action_payload(action) for action in trajectory.actions],
                    "next_state": _state_payload(trajectory.next_state),
                }
                for trajectory in result.evidence.trajectories
            ],
            "step_economics": [
                {
                    "run_id": item.run_id,
                    "start_date": item.start_date.isoformat(),
                    "npv_million_rub": item.npv_million_rub,
                    "complete": item.complete,
                }
                for item in result.evidence.step_economics
            ],
        },
    }
    result_bytes = _json(payload)
    schedule_bytes = result.schedule.text.encode("utf-8")
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "run_id": config.run_id,
        "input_sha256": config.input_sha256,
        "config_sha256": config.config_sha256,
        "source_sha256": config.source_sha256,
        "script_source_contract": source_contract,
        "backend_provenance": result.evidence.backend_provenance,
        "schedule_sha256": result.schedule.sha256,
        "trajectory_run_ids": [item.run_id for item in result.evidence.trajectories],
        "artifacts": {
            "result": {
                "path": "result.json",
                "bytes": len(result_bytes),
                "sha256": _digest(result_bytes),
            },
            "schedule": {
                "path": "wells_schedule.inc",
                "bytes": len(schedule_bytes),
                "sha256": _digest(schedule_bytes),
            },
        },
    }
    manifest_bytes = _json(manifest)
    manifest_sha256 = _digest(manifest_bytes)
    outputs = {
        Path("result.json"): result_bytes,
        Path("wells_schedule.inc"): schedule_bytes,
        Path("manifest.json"): manifest_bytes,
        Path("manifest.sha256"): f"{manifest_sha256}  manifest.json\n".encode("ascii"),
    }
    return outputs, {
        "run_id": config.run_id,
        "result_sha256": manifest["artifacts"]["result"]["sha256"],
        "manifest_sha256": manifest_sha256,
    }


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
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def publish(runs_dir: Path, run_id: str, outputs: dict[Path, bytes]) -> Path:
    parent = runs_dir.absolute()
    _reject_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(parent)
    destination = parent / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite run directory {destination}")
    for relative in outputs:
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError(f"unsafe output path: {relative}")
    destination.mkdir()
    for relative, data in sorted(outputs.items(), key=lambda item: str(item[0])):
        _atomic_new_file(destination / relative, data)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--proof-script", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        source_contract = _script_source_contract(args.proof_script)
        config = load_config(args.config)
        outputs, summary = execute(
            config,
            build_backend(config),
            script_source_contract=source_contract,
        )
        run_dir = publish(args.runs_dir, config.run_id, outputs)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({**summary, "run_dir": str(run_dir)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
