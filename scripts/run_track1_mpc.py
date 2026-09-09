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
from timesoil.aios.schedule import ScheduleCompiler
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


def _propose_controls(case: Case, baseline: Candidate, updates: Any) -> Candidate:
    controls = {action.well: action for action in baseline}
    if len(controls) != len(baseline) or set(controls) != set(case.producers + case.injectors):
        raise ValueError("full-field baseline must contain every well exactly once")
    if not isinstance(updates, list):
        raise ValueError("updates must be an array")
    seen = set()
    for index, value in enumerate(updates):
        item = _object(value, f"updates[{index}]", {"well", "status", "target", "value"})
        well = _string(item["well"], "update well")
        if well not in controls or well in seen:
            raise ValueError("unknown or duplicate well update")
        seen.add(well)
        controls[well] = replace(
            controls[well], status=WellStatus(item["status"]),
            target=ControlTarget(item["target"]), value=_number(item["value"], "control value"),
        )
    return ScheduleCompiler().validate(case, controls.values())


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
            if replace(action, month=source_previous[well].month) == source_previous[well]
            else action for well, action in source.items()
        }
        output.extend(current.values())
        source_previous = source
        month = _next_month(month)
    return ScheduleCompiler().validate(config.case, output)


def execute(
    config: RunConfig,
    backend: GdmBackend,
    *,
    script_source_contract: dict[str, dict[str, str]] | None = None,
    agent: bool = False,
    agent_log: Path | None = None,
    full_field: bool = False,
    lifecycle: bool = False,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    started = monotonic()
    if full_field and (not agent or any(len(options) != 1 for options in config.candidates.values())):
        raise ValueError("full-field mode requires --agent and exactly one baseline per month")
    if lifecycle and not full_field:
        raise ValueError("lifecycle mode requires full-field agent proposals")
    if agent and config.case.economics_start != config.case.start:
        raise ValueError("agent economics_start must equal the AIOS management start")
    source_contract = script_source_contract or _script_source_contract()
    agent_records: list[dict[str, Any]] = []
    planning = None
    llm_config = LLMConfig.from_env() if agent else None
    feedback: list[dict[str, Any]] = []
    previous_controls: dict[str, ControlAction] = {}

    def record(item: dict[str, Any]) -> None:
        agent_records.append(item)
        if agent_log is not None:
            with agent_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    async def choose(state: State) -> Any:
        options = tuple(ScheduleCompiler().validate(config.case, option) for option in config.candidates[state.month])
        if full_field and feedback:
            source_previous = {a.well: a for a in config.candidates[date.fromisoformat(feedback[-1]["month"])][0]}
            options = (tuple(
                replace(previous_controls[a.well], month=state.month)
                if replace(a, month=source_previous[a.well].month) == source_previous[a.well]
                else a for a in options[0]
            ),)
        context = {
            "track": 1, "phase": "planning", "surrogate_used": False,
            "source_sha256": config.source_sha256,
            "state": _state_payload(state),
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
        if full_field:
            context.update({
                "selection_policy": "Optimize official CHDD by proposing rate and OPEN/SHUT updates for ANY well in the complete baseline. Baseline carries forward approved agent controls, except explicit changes in the source calendar. Empty updates means keep these controls. No preselected well subset or percentage bounds. Exactly one propose_controls call. Every month is validated by full OPM and official CHDD. Use the verified inventory; never infer extra wells from gaps in numeric IDs.",
                "verified_inventory": {"well_count": len(config.case.producers) + len(config.case.injectors),
                    "producers": list(config.case.producers), "injectors": list(config.case.injectors)},
                "state_units": {"oil_rate": "surface m3/day, NOT tonnes/day", "liquid_rate": "surface m3/day", "injection_rate": "surface m3/day", "bhp": "bar"},
                "constraints": {"max_producer_liquid_m3d": config.case.max_liquid_rate,
                    "pressure": "source schedule BHP bounds remain enforced by OPM",
                    "water_quota": "no additional numeric quota supplied in this archive; do not invent one",
                    "availability": "respect source completions; no drilling or unprovided repair assumptions",
                    "roles": "role fixed by this case contract; conversion is not implemented in this controller"},
                "economics": {"oil_rub_per_t": 28000, "oil_deductions_rub_per_t": 19600,
                    "oil_opex_rub_per_t": 40, "liquid_opex_rub_per_t": 100,
                    "injection_opex_rub_per_m3": 30, "active_well_m_per_year": 1,
                    "stop_or_start_m": 1, "pump_change_operation_m": 1.8,
                    "pump_capex_m": "0.55 to 8.05 by type; switching across size bands incurs CAPEX",
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
                return {"well_count": len(candidate), "inventory_matches_case": True,
                        "missing_wells": [], "extra_wells": [],
                        "controls": [_action_payload(a) for a in candidate],
                        "schedule_sha256": ScheduleCompiler().compile(config.case, candidate).sha256}
            tool = ToolDefinition("propose_controls", "Propose updates to any well; retain complete baseline for other wells. SHUT requires value=0; OPEN LRAT must be <=500; injectors use WRAT.",
                {"type": "object", "properties": {"updates": {"type": "array", "items": {
                    "type": "object", "properties": {"well": {"type": "string"},
                        "status": {"type": "string", "enum": ["OPEN", "SHUT"]},
                        "target": {"type": "string", "enum": ["ORAT", "LRAT", "WRAT"]},
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
                    break
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
                    "No improvement is claimed; surrogate UQ/OOD is inapplicable because no surrogate was used."
                )
        record({"phase": "terminal_month_review", "month": result.trajectory.month.isoformat(), "agent": asdict(reviewed)})
        if not reviewed.critic_approved:
            raise RuntimeError("critic rejected simulated month; see agent decision log")
        previous_controls.update({a.well: a for a in result.trajectory.actions})
        feedback.append({"month": result.trajectory.month.isoformat(),
                         "cumulative_chdd_m": result.economics.npv_million_rub,
                         "controls": [_action_payload(a) for a in result.trajectory.actions],
                         "state": _state_payload(result.trajectory.next_state)})

    result: Track1Result = MonthlyMPC(
        backend,
        planning_tail=(lambda state, candidate: _continuation_tail(config, state, candidate)) if lifecycle else None,
    ).run(
        config.case,
        config.initial_state,
        candidates,
        on_step=(lambda step: asyncio.run(review(step))) if agent else None,
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
        "horizon_protocol": {
            "economic_start": config.case.economics_start.isoformat(),
            "economic_end_exclusive": _next_month(config.case.end).isoformat(),
            "commit_months": 1,
            "planning": "full_remaining_period" if lifecycle else "one_month",
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
        )
        run_dir = publish(args.runs_dir, config.run_id, outputs)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({**summary, "run_dir": str(run_dir)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
