#!/usr/bin/env python3
"""Run deterministic Track 1 monthly MPC through the production OPM backend."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
from datetime import date
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from time import monotonic
from typing import Any

from timesoil.aios.agents import AgentRole, AgentWorkflow, ToolDefinition, ToolRegistry
from timesoil.aios.llm import LLMConfig, TatneftLLMClient
from timesoil.aios.economics import CHDDEconomicsAdapter
from timesoil.aios.operating_constraints import check_controls, parse_constraints

from timesoil.aios.contracts import (
    Case,
    ControlAction,
    ControlTarget,
    Economics,
    State,
    Trajectory,
    WellRole,
    WellState,
    WellStatus,
)
from timesoil.aios.opm import OpmFlowRunner, OpmGdmBackend, _source_digest
from timesoil.aios.schedule import ScheduleCompiler
from timesoil.aios.track1 import Candidate, GdmBackend, GdmResult, MonthlyMPC, Track1Result


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
    forecast: dict[str, str] | None = None

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
        {"max_liquid_rate", "allow_conversion_to_injection", "operating_constraints"},
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
        allow_conversion_to_injection=item.get("allow_conversion_to_injection", False),
        operating_constraints=parse_constraints(
            item.get("operating_constraints", []), wells=(*item["producers"], *item["injectors"]),
            start=_month(item["start"], "case.start"), end=_month(item["end"], "case.end"),
        ),
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
    item = _object(value, label, {"well", "role", "status", "target", "value"}, {"bhp_limit"})
    return ControlAction(
        month=month,
        well=_string(item["well"], f"{label}.well"),
        role=WellRole(item["role"]),
        status=WellStatus(item["status"]),
        target=ControlTarget(item["target"]),
        value=_number(item["value"], f"{label}.value"),
        bhp_limit=None if item.get("bhp_limit") is None else _number(item["bhp_limit"], f"{label}.bhp_limit"),
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
    root = _object(payload, "config", {"schema", "case", "initial_state", "candidates", "opm"}, {"forecast"})
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
    forecast = None
    if 'forecast' in root:
        forecast = dict(_object(root['forecast'], 'forecast',
            {'history', 'history_sha256', 'geology', 'geology_sha256'}))
        for name in ('history', 'geology'):
            file = (path.absolute().parent / _string(forecast[name], 'forecast.' + name)).absolute()
            _reject_symlink_components(file)
            if not file.is_file() or _digest(file.read_bytes()) != forecast[name + '_sha256']:
                raise ValueError('forecast input hash mismatch: ' + name)
            forecast[name] = str(file)
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
        forecast=forecast,
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
    return action.to_dict()


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


def _propose_controls(case: Case, baseline: Candidate, updates: Any) -> Candidate:
    controls = {action.well: action for action in baseline}
    if len(controls) != len(baseline) or set(controls) != set(case.producers + case.injectors):
        raise ValueError("full-field baseline must contain every well exactly once")
    if not isinstance(updates, list):
        raise ValueError("updates must be an array")
    seen = set()
    for index, value in enumerate(updates):
        item = _object(value, f"updates[{index}]", {"well", "status", "target", "value"}, {"role", "bhp_limit"})
        well = _string(item["well"], "update well")
        if well not in controls or well in seen:
            raise ValueError("unknown or duplicate well update")
        seen.add(well)
        controls[well] = replace(
            controls[well], status=WellStatus(item["status"]),
            target=ControlTarget(item["target"]), value=_number(item["value"], "control value"),
            role=WellRole(item.get("role", controls[well].role)),
            bhp_limit=(controls[well].bhp_limit if "bhp_limit" not in item
                       else _number(item["bhp_limit"], "control bhp_limit")),
        )
        if controls[well].role is WellRole.INJECTOR and case.role_of(well) is WellRole.PRODUCER and controls[well].bhp_limit is None:
            raise ValueError("conversion to injection requires an explicit BHP ceiling")
    candidate = ScheduleCompiler().validate(case, controls.values())
    check_controls(case.operating_constraints, candidate)
    return candidate


def _continuation_tail(config: RunConfig, state: State, candidate: Candidate) -> Candidate:
    """Carry controls forward; only explicit source-calendar changes override them."""
    current = {a.well: a for a in candidate}
    source_previous = {a.well: a for a in config.candidates[state.month][0]}
    output = []
    month = _next_month(state.month)
    while month <= config.case.end:
        source = {a.well: a for a in config.candidates[month][0]}
        current = {
            well: replace(current[well], month=month)
            if (replace(action, month=source_previous[well].month) == source_previous[well]
                or current[well].role is WellRole.INJECTOR and action.role is WellRole.PRODUCER)
            else action for well, action in source.items()
        }
        output.extend(current.values())
        source_previous = source
        month = _next_month(month)
    return ScheduleCompiler().validate(config.case, output)


def _resume_steps(config: RunConfig, backend: OpmGdmBackend, path: Path):
    """Load only the contiguous approved prefix, authenticated against physical receipts."""
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    steps, accepted = [], []
    state, plan, prefix_end = config.initial_state, None, 0
    for index, record in enumerate(records):
        if record["phase"] == "planning":
            plan = record["agent"]
        if record["phase"] != "terminal_month_review":
            continue
        review = record["agent"]
        if not all(d["approved"] for d in review["decisions"]):
            break
        context = review["context"]
        if (not plan or not all(d["approved"] for d in plan["decisions"])
                or plan["context"]["state"] != _state_payload(state)
                or context["source_sha256"] != config.source_sha256
                or record["month"] != state.month.isoformat()
                or not review["decisions"][-1]["tool_evidence"]):
            raise ValueError("resume journal does not match the approved state chain")
        item = context["trajectory"]
        trajectory = Trajectory(**{**item, "month": _month(item["month"], "resume month"),
            "actions": backend._actions_value(item["actions"]),
            "next_state": _state(item["next_state"]),
            "invariant_violations": tuple(item["invariant_violations"])})
        def economics(value):
            return Economics(**{**value, "start_date": _month(value["start_date"], "economics start")})
        step = GdmResult(trajectory, economics(context["economics"]),
            economics(context["planning_economics"]),
            _month(context["planning_end_exclusive"], "planning end"))
        MonthlyMPC._check_result(config.case, state, trajectory.actions, step)
        lineage_path, _ = backend._parse_restart_ref(trajectory.next_state.restart_ref)
        backend._verify_opm_manifest(lineage_path.parent / "manifest.json", baseline=False)
        lineage = json.loads(lineage_path.read_text())
        if (lineage["prior_restart_ref"] != state.restart_ref
                or lineage["input_state"] != backend._state_value(state)
                or lineage["economics"]["total_chdd_m"] != step.economics.npv_million_rub
                or lineage["planning"]["total_chdd_m"] != step.planning_economics.npv_million_rub
                or lineage["planning"]["end_exclusive"] != step.planning_end.isoformat()
                or lineage["planning"]["future_states_committed"]):
            raise ValueError("resume journal disagrees with physical lineage")
        accepted.extend(trajectory.actions)
        if backend._authenticated_history(config.case, trajectory.next_state) != tuple(accepted):
            raise ValueError("resume controls differ from authenticated physical history")
        steps.append(step)
        state, prefix_end = trajectory.next_state, index + 1
    if not steps:
        raise ValueError("resume journal has no approved contiguous prefix")
    return steps, records[:prefix_end], {"path": str(path.resolve()), "sha256": _digest(raw),
        "approved_months": len(steps), "remaining_records_preserved_in_source": len(records) - prefix_end,
        "physical_lineage_verified": True}


def execute(
    config: RunConfig,
    backend: GdmBackend,
    *,
    script_source_contract: dict[str, dict[str, str]] | None = None,
    agent: bool = False,
    agent_log: Path | None = None,
    full_field: bool = False,
    lifecycle: bool = False,
    resume_log: Path | None = None,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    started = monotonic()
    if full_field and (not agent or any(len(options) != 1 for options in config.candidates.values())):
        raise ValueError("full-field mode requires --agent and exactly one baseline per month")
    if lifecycle and not full_field:
        raise ValueError("lifecycle mode requires full-field agent proposals")
    if config.forecast and not lifecycle:
        raise ValueError('Google forecast assistance requires full-field lifecycle verification')
    if agent and config.case.economics_start != config.case.start:
        raise ValueError("agent economics_start must equal the AIOS management start")
    source_contract = script_source_contract or _script_source_contract()
    agent_records: list[dict[str, Any]] = []
    planning = None
    llm_config = LLMConfig.from_env() if agent else None
    normative_profile = (
        (backend.economics or CHDDEconomicsAdapter.from_env()).normative_profile()
        if agent and isinstance(backend, OpmGdmBackend) else None
    )
    feedback: list[dict[str, Any]] = []
    previous_controls: dict[str, ControlAction] = {}
    completed_steps, resume_receipt = (), None
    forecast = None
    if config.forecast:
        from timesfm_planning import TimesFMPlanning
        forecast = TimesFMPlanning(config.forecast, config.initial_state, config.source_sha256)

    def record(item: dict[str, Any]) -> None:
        agent_records.append(item)
        if agent_log is not None:
            with agent_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    if resume_log is not None:
        if not lifecycle or not isinstance(backend, OpmGdmBackend):
            raise ValueError("resume requires lifecycle mode and the physical OPM backend")
        completed_steps, saved_records, resume_receipt = _resume_steps(config, backend, resume_log)
        for item in saved_records:
            record(item)
        for step in completed_steps:
            if forecast:
                forecast.observe(step.trajectory.next_state, step.trajectory.actions)
            previous_controls.update({a.well: a for a in step.trajectory.actions})
            feedback.append({"month": step.trajectory.month.isoformat(),
                "cumulative_chdd_m": step.economics.npv_million_rub,
                "controls": [_action_payload(a) for a in step.trajectory.actions],
                "state": _state_payload(step.trajectory.next_state)})

    async def choose(state: State) -> Any:
        options = tuple(ScheduleCompiler().validate(config.case, option) for option in config.candidates[state.month])
        if full_field and feedback:
            source_previous = {a.well: a for a in config.candidates[date.fromisoformat(feedback[-1]["month"])][0]}
            options = (tuple(
                replace(previous_controls[a.well], month=state.month)
                if (replace(a, month=source_previous[a.well].month) == source_previous[a.well]
                    or previous_controls[a.well].role is WellRole.INJECTOR and a.role is WellRole.PRODUCER)
                else a for a in options[0]
            ),)
        context = {
            "track": 1, "phase": "planning", "surrogate_used": False,
            "source_sha256": config.source_sha256,
            "state": _state_payload(state),
            "operating_constraints": [rule.to_dict() for rule in config.case.operating_constraints],
            "candidates": [[_action_payload(a) for a in option] for option in options],
            "selection_policy": "Choose one candidate for full OPM and official CHDD; numerical results are not known yet. Candidate index is zero-based. No global optimality claim.",
        }
        context["horizon_protocol"] = {
            "observation_cutoff_inclusive": state.month.isoformat(),
            "commit_months": 1,
            "economic_start": config.case.economics_start.isoformat(),
            "economic_end_exclusive": _next_month(config.case.end).isoformat(),
            "selection": "compare proposal and incumbent by full remaining OPM economics" if lifecycle else "one proposed candidate; no lookahead comparison",
            "future_calendar": "provided source schedule, assumed known for this training experiment",
            "future_observed_states_available": False,
        }
        if forecast:
            context['surrogate_used'] = True
            context['geology'] = forecast.geology_context()
            context['google_forecast'] = forecast.predict(state, options[0] + _continuation_tail(config, state, options[0]))
            context['forecast_claim_limits'] = 'Google forecast assists the proposal. Uncertainty is not independently calibrated; every candidate is checked by full remaining OPM and official CHDD before committing one month.'
        if full_field:
            context.update({
                "selection_policy": "Optimize official CHDD by proposing rate and OPEN/SHUT updates for ANY well in the complete baseline. Baseline carries forward approved agent controls, except explicit changes in the source calendar. Empty updates means keep these controls. No preselected well subset or percentage bounds. Exactly one propose_controls call. Every month is validated by full OPM and official CHDD. Use the verified inventory; never infer extra wells from gaps in numeric IDs.",
                "verified_inventory": {"well_count": len(config.case.producers) + len(config.case.injectors),
                    "producers": [a.well for a in options[0] if a.role is WellRole.PRODUCER],
                    "injectors": [a.well for a in options[0] if a.role is WellRole.INJECTOR]},
                "state_units": {"oil_rate": "surface m3/day, NOT tonnes/day", "liquid_rate": "surface m3/day", "injection_rate": "surface m3/day", "bhp": "bar"},
                "constraints": {"max_producer_liquid_m3d": config.case.max_liquid_rate,
                    "status_changes": "OPEN/SHUT are permitted control decisions, not immutable source data; SHUT requires value=0",
                    "pressure": "bhp_limit may tighten original BHP bounds; never relax them. New conversions require an explicit injection BHP ceiling.",
                    "water_quota": "no additional numeric quota supplied in this archive; do not invent one",
                    "availability": "respect source completions; no drilling or unprovided repair assumptions",
                    "roles": "producer-to-injector conversion allowed; reverse conversion forbidden" if config.case.allow_conversion_to_injection else "roles fixed by this case contract",
                    "allow_conversion_to_injection": config.case.allow_conversion_to_injection},
                "economics": {"official_normative_profile": normative_profile,
                    "horizon_end": config.case.end.isoformat(),
                    "objective": "Preserve profitable oil, avoid uneconomic water production and needless pump/status changes; evaluate tradeoffs over the full remaining horizon."},
                "prior_rate_columns": ["well", "active", "oil_rate", "liquid_rate", "injection_rate", "bhp"],
                "prior_months": [{"month": f["month"], "cumulative_chdd_m": f["cumulative_chdd_m"],
                    "well_rates": [[w[k] for k in ("well", "active", "oil_rate", "liquid_rate", "injection_rate", "bhp")]
                                   for w in f["state"]["wells"]]} for f in feedback[-3:]],
            })

        def select(arguments: Any, _: Any) -> dict[str, Any]:
            index = arguments["index"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(options):
                raise ValueError("candidate index outside configured options")
            return {"selected_index": index, "constraint_check": "configured candidate; deterministic validation precedes OPM"}

        tool = ToolDefinition(
            "select_candidate", "Select one configured candidate index for the current month.",
            {"type": "object", "properties": {"index": {"type": "integer"}},
             "required": ["index"], "additionalProperties": False}, select,
        )
        proposed: list[Candidate] = []
        if full_field:
            def propose(arguments: Any, _: Any) -> dict[str, Any]:
                candidate = _propose_controls(config.case, options[0], arguments["updates"])
                proposed.append(candidate)
                output = {"well_count": len(candidate), "inventory_matches_case": True,
                        "controls_validated": True, "status_changes_permitted": True,
                        "missing_wells": [], "extra_wells": [],
                        "controls": [_action_payload(a) for a in candidate],
                        "schedule_sha256": ScheduleCompiler().compile(config.case, candidate).sha256}
                if forecast:
                    output['google_forecast'] = forecast.predict(state, candidate + _continuation_tail(config, state, candidate))
                return output
            tool = ToolDefinition("propose_controls", "Propose updates to any well; retain other controls. SHUT requires value=0; OPEN LRAT <=500; injectors use WRAT. Optional role changes require case permission; new injection needs bhp_limit (bar). Optional bhp_limit tightens the original pressure bound.",
                {"type": "object", "properties": {"updates": {"type": "array", "items": {
                    "type": "object", "properties": {"well": {"type": "string"},
                        "status": {"type": "string", "enum": ["OPEN", "SHUT"]},
                        "target": {"type": "string", "enum": ["ORAT", "LRAT", "WRAT"]},
                        "role": {"type": "string", "enum": ["producer", "injector"]},
                        "bhp_limit": {"type": "number"},
                        "value": {"type": "number"}},
                    "required": ["well", "status", "target", "value"], "additionalProperties": False}}},
                 "required": ["updates"], "additionalProperties": False}, propose)
        registry = ToolRegistry((tool,))
        async with TatneftLLMClient(llm_config) as client:
            workflow = AgentWorkflow(client, registry,
                role_tools={AgentRole.PLANNER: (tool.name,)},
                required_tools={AgentRole.PLANNER: (tool.name,)})
            for attempt in range(2):
                try:
                    plan = await workflow.run_plan(context)
                    if all(d.approved for d in plan.decisions) or not full_field or attempt:
                        break
                    record({"phase": "rejected_planning", "month": state.month.isoformat(), "agent": asdict(plan)})
                    proposed.clear()
                    context["previous_rejection"] = plan.decisions[-1].summary
                    context["repair_instruction"] = "Recheck against explicit control permissions and tool validation. OPEN/SHUT changes are allowed hypotheses. Correct or withdraw your proposal if needed; do not approve unsupported controls. Call propose_controls once."
                except ValueError as exc:
                    if not full_field or attempt:
                        raise
                    proposed.clear()
                    record({"phase": "invalid_proposal", "month": state.month.isoformat(),
                            "error": str(exc), "simulator_executed": False})
                    context["previous_validation_error"] = str(exc)
                    context["repair_instruction"] = "Correct the invalid proposal and call propose_controls once. SHUT must have value=0. No invalid control was executed."
        selections = plan.decisions[-1].tool_evidence
        record({"phase": "planning", "month": state.month.isoformat(), "agent": asdict(plan)})
        if not all(decision.approved for decision in plan.decisions):
            raise RuntimeError("agent rejected planning; see agent decision log")
        if len(selections) != 1 or selections[0].tool != tool.name:
            raise RuntimeError("planner must make exactly one control proposal")
        if full_field:
            if len(proposed) != 1:
                raise RuntimeError("planner must propose one complete field schedule")
            return plan, (options[0], proposed[0]) if lifecycle else (proposed[0],)
        return plan, (options[selections[0].output["selected_index"]],)

    def candidates(state: State) -> Any:
        nonlocal planning
        if not agent:
            return config.candidates[state.month]
        planning, selected = asyncio.run(choose(state))
        return selected

    async def review(result: Any) -> None:
        verified_schedule = ScheduleCompiler().compile(config.case, result.trajectory.actions)
        next_wells = result.trajectory.next_state.wells
        context = {
            "track": 1, "phase": "terminal_month_review", "surrogate_used": False,
            "source_sha256": config.source_sha256,
            "trajectory": asdict(result.trajectory), "economics": asdict(result.economics),
            "planning_economics": None if result.planning_economics is None else asdict(result.planning_economics),
            "planning_end_exclusive": result.planning_end,
            "selection_policy": "best full-horizon OPM CHDD among incumbent and agent proposal; only this month committed" if lifecycle else "single agent proposal",
            "provenance": {"backend": backend.get_provenance(), "planning_run_id": planning.run_id,
                "controls_sha256": verified_schedule.sha256,
                "source_sha256": config.source_sha256,
                "verified_state_receipt": result.trajectory.next_state.restart_ref},
            "verified_constraints": {"controls_validated_by": "ScheduleCompiler",
                "result_checked_by": "MonthlyMPC._check_result",
                "well_count": len(next_wells),
                "inventory_matches_case": {w.well for w in next_wells} == set(config.case.producers + config.case.injectors),
                "actual_max_liquid_m3d": max((w.liquid_rate for w in next_wells), default=0),
                "allowed_max_liquid_m3d": config.case.max_liquid_rate,
                "invariant_violations": list(result.trajectory.invariant_violations)},
            "claim_limits": "No surrogate used; UQ/OOD not applicable. No NPV improvement or global optimality claim. Deterministic MPC gates already passed.",
        }
        if forecast:
            context.update(surrogate_used=True, forecast_provenance=forecast.provenance,
                claim_limits='Google assisted the proposal; the selected full remaining trajectory, constraints and official CHDD were verified by OPM. Uncalibrated forecast uncertainty prevents autonomous surrogate certification, not an audit of completed physical calculations. No global optimality claim.')
        # Dates belong to the typed simulator result, not to model-generated data.
        context = json.loads(json.dumps(context, default=str, allow_nan=False))
        evidence_tool = ToolDefinition(
            "verify_month_evidence", "Read deterministic checks and provenance for this completed simulator month.",
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            lambda _arguments, _context: {
                **context,
                "verified": True,
                "constraints_checked_before_simulation": True,
                "constraints": context["verified_constraints"],
            },
        )
        async with TatneftLLMClient(llm_config) as client:
            workflow = AgentWorkflow(client, ToolRegistry((evidence_tool,)),
                role_tools={AgentRole.CRITIC: (evidence_tool.name,)},
                required_tools={AgentRole.CRITIC: (evidence_tool.name,)},
            )
            for attempt in range(2):
                reviewed = await workflow.run_critic(planning, context)
                if reviewed.critic_approved or attempt:
                    break
                record({"phase": "rejected_month_review", "month": result.trajectory.month.isoformat(),
                        "agent": asdict(reviewed)})
                context["previous_rejection"] = reviewed.decisions[-1].summary
                context["review_instruction"] = (
                    "Recheck the rejection against the complete evidence tool: selected simulator trajectory, "
                    "constraints, official economics, provenance and claim limits. Approve only if supported. "
                    "No global optimality is claimed. Respect the explicit claim limits and distinguish completed physical verification from surrogate certification."
                )
        record({"phase": "terminal_month_review", "month": result.trajectory.month.isoformat(), "agent": asdict(reviewed)})
        if not reviewed.critic_approved:
            raise RuntimeError("critic rejected simulated month; see agent decision log")
        if forecast:
            forecast.observe(result.trajectory.next_state, result.trajectory.actions)
            record({'phase': 'forecast_observation_check', 'month': result.trajectory.month.isoformat(),
                    'observed_errors': forecast.observed_errors[-1:]})
        previous_controls.update({a.well: a for a in result.trajectory.actions})
        feedback.append({"month": result.trajectory.month.isoformat(),
                         "cumulative_chdd_m": result.economics.npv_million_rub,
                         "controls": [_action_payload(a) for a in result.trajectory.actions],
                         "state": _state_payload(result.trajectory.next_state)})

    result: Track1Result = MonthlyMPC(
        backend,
        planning_tail=(lambda state, candidate: _continuation_tail(config, state, candidate)) if lifecycle else None,
        parallel_candidates=lifecycle,
    ).run(
        config.case,
        config.initial_state,
        candidates,
        on_step=(lambda step: asyncio.run(review(step))) if agent else None,
        completed_steps=completed_steps,
    )
    if _source_digest(config.source) != config.source_sha256:
        raise RuntimeError("OPM source changed while Track 1 was running")
    _verify_script_source_contract(source_contract)
    if forecast:
        forecast.verify_files()
    payload = {
        "schema": _RESULT_SCHEMA,
        "run_id": config.run_id,
        "input_sha256": config.input_sha256,
        "config_sha256": config.config_sha256,
        "source_sha256": config.source_sha256,
        "script_source_contract": source_contract,
        "case_id": config.case.case_id,
        "horizon_protocol": {
            "economic_start": config.case.economics_start.isoformat(),
            "economic_end_exclusive": _next_month(config.case.end).isoformat(),
            "commit_months": 1,
            "planning": "full_remaining_period" if lifecycle else "one_month",
            "candidate_workers": 2 if lifecycle else 1,
            "future_states_committed": False,
            "final_chdd": "last cumulative economics, never sum cumulative monthly values",
        },
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
    if agent:
        payload["agent"] = {"model": llm_config.model, "base_url": llm_config.base_url,
                            "elapsed_seconds": monotonic() - started,
                            "selection_policy": "incumbent versus agent proposal, full-horizon OPM, commit one month then critic" if lifecycle else ("all-well agent proposals, full OPM then critic" if full_field else "one configured candidate per month, full OPM then critic"),
                            "records": agent_records}
        if forecast:
            payload['agent']['google_forecast'] = {**forecast.provenance, 'observed_errors': forecast.observed_errors}
        if resume_receipt is not None:
            payload["agent"]["resume"] = resume_receipt
            payload["agent"]["elapsed_seconds_scope"] = "current invocation; historical resumed calls excluded"
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
    parser.add_argument("--agent", action="store_true", help="Use external Qwen to select one candidate per month, then review each full OPM/CHDD result")
    parser.add_argument("--full-field", action="store_true", help="Allow Qwen to propose validated controls for every well")
    parser.add_argument("--lifecycle", action="store_true", help="Compare incumbent and proposal over the full remaining period; commit only one month")
    parser.add_argument("--resume-log", type=Path, help="Resume a lifecycle run from a physically verified approved journal prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        source_contract = _script_source_contract(args.proof_script)
        config = load_config(args.config)
        agent_log = None
        if args.agent:
            args.runs_dir.mkdir(parents=True, exist_ok=True)
            _reject_symlink_components(args.runs_dir.absolute())
            agent_log = args.runs_dir / f"{config.run_id}-agent.jsonl"
            with agent_log.open("x", encoding="utf-8"):
                pass
        outputs, summary = execute(
            config,
            build_backend(config),
            script_source_contract=source_contract,
            agent=args.agent,
            agent_log=agent_log,
            full_field=args.full_field,
            lifecycle=args.lifecycle,
            resume_log=args.resume_log,
        )
        run_dir = publish(args.runs_dir, config.run_id, outputs)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({**summary, "run_dir": str(run_dir)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
