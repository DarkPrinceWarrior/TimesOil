"""One fail-closed operator path from Qwen decision to OPM/CHDD receipt."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .agents import (
    AgentLLM,
    AgentRole,
    AgentState,
    AgentWorkflow,
    ToolDefinition,
    ToolRegistry,
    WorkflowError,
)
from .contracts import ControlAction, ControlTarget, WellRole, WellStatus
from .economics import (
    CHDD_FIELDS,
    CHDDEconomicsAdapter,
    EconomicResult,
    normalize_chdd_rows,
)
from .llm import APPROVED_BASE_URL, APPROVED_MODEL, TatneftLLMClient
from .opm import OPM_IMAGE, OpmFlowRunner
from .opm_chdd import export_opm_chdd
from .schedule_overlay import _canonical_schedule, apply_schedule_overlay
from .tools import GROUNDED_ROLE_TOOLS, build_grounded_tool_registry


VERIFY_FULL_CONTROLS = "verify_full_controls"
VERIFY_TERMINAL_EVIDENCE = "verify_terminal_evidence"
_EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api[_-]?key|authorization|cookie|credential|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)
_MAX_ACTIONS = 10_000


class CycleError(RuntimeError):
    """The deterministic full-cycle boundary rejected an input or artifact."""


class CycleRejected(CycleError):
    """A planning role rejected controls before simulator execution."""


@dataclass(frozen=True, slots=True)
class CycleRequest:
    context: Mapping[str, Any]
    controls: tuple[ControlAction, ...]
    source: Path
    deck: str
    schedule_relative_path: PurePosixPath
    scenario_id: str
    source_model: str
    start_year: int
    parsing_strictness: str = "strict"
    normalize_model_y: bool = False
    density_map: Path | None = None
    charge_initial_pump: bool | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, base_dir: str | Path = "."
    ) -> CycleRequest:
        required = {
            "context",
            "controls",
            "source",
            "deck",
            "schedule_relative_path",
            "scenario_id",
            "source_model",
            "start_year",
        }
        allowed = required | {
            "parsing_strictness",
            "normalize_model_y",
            "density_map",
            "charge_initial_pump",
        }
        missing = required - set(value)
        unknown = set(value) - allowed
        if missing or unknown:
            raise CycleError("cycle request keys are invalid")
        context = value["context"]
        if not isinstance(context, Mapping):
            raise CycleError("context must be a JSON object")
        normalized_context = _json_object(context, "context")
        _reject_sensitive_keys(normalized_context)
        controls = _controls(value["controls"])
        _validate_full_horizon(controls)
        root = Path(base_dir).resolve()
        source = _path(value["source"], root, "source")
        density_raw = value.get("density_map")
        density_map = (
            None if density_raw is None else _path(density_raw, root, "density_map")
        )
        deck = _text(value["deck"], "deck", 512)
        schedule_relative_path = _relative(
            value["schedule_relative_path"], "schedule_relative_path"
        )
        scenario_id = _identifier(value["scenario_id"], "scenario_id")
        source_model = _text(value["source_model"], "source_model", 128)
        start_year = value["start_year"]
        if isinstance(start_year, bool) or not isinstance(start_year, int) or not 1900 <= start_year <= 9999:
            raise CycleError("start_year must be a four-digit integer")
        parsing = value.get("parsing_strictness", "strict")
        if parsing not in {"strict", "low"}:
            raise CycleError("parsing_strictness must be strict or low")
        normalize = value.get("normalize_model_y", False)
        charge = value.get("charge_initial_pump")
        if not isinstance(normalize, bool) or charge is not None and not isinstance(charge, bool):
            raise CycleError("cycle boolean options are invalid")
        if normalize and parsing != "low":
            raise CycleError("normalize_model_y requires explicit low parsing")
        return cls(
            context=normalized_context,
            controls=controls,
            source=source,
            deck=deck,
            schedule_relative_path=schedule_relative_path,
            scenario_id=scenario_id,
            source_model=source_model,
            start_year=start_year,
            parsing_strictness=parsing,
            normalize_model_y=normalize,
            density_map=density_map,
            charge_initial_pump=charge,
        )

    @property
    def controls_sha256(self) -> str:
        return sha256(_canonical_schedule(self.controls).encode()).hexdigest()

    @property
    def actions_sha256(self) -> str:
        return sha256(_json_bytes([_action(item) for item in self.controls])).hexdigest()

    @property
    def request_sha256(self) -> str:
        return sha256(
            _json_bytes(
                {
                    "context": self.context,
                    "controls_sha256": self.controls_sha256,
                    "actions_sha256": self.actions_sha256,
                    "source": str(self.source),
                    "deck": self.deck,
                    "schedule_relative_path": str(self.schedule_relative_path),
                    "scenario_id": self.scenario_id,
                    "source_model": self.source_model,
                    "start_year": self.start_year,
                    "parsing_strictness": self.parsing_strictness,
                    "normalize_model_y": self.normalize_model_y,
                    "density_map": str(self.density_map) if self.density_map else None,
                    "charge_initial_pump": self.charge_initial_pump,
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CycleResult:
    run_id: str
    run_dir: Path
    receipt_path: Path
    receipt_sha256: str
    critic_approved: bool


class FullCycleWorkflow:
    """Run supplied controls only after Qwen planning, then ask Qwen to review evidence."""

    def __init__(
        self,
        llm: AgentLLM,
        *,
        runner: OpmFlowRunner,
        economics: CHDDEconomicsAdapter,
        registry: ToolRegistry | None = None,
        exporter: Callable[..., dict[str, Any]] = export_opm_chdd,
    ) -> None:
        if (
            not isinstance(llm, TatneftLLMClient)
            or llm.config.base_url.rstrip("/") != APPROVED_BASE_URL
            or llm.config.model != APPROVED_MODEL
        ):
            raise CycleError("full cycle requires the approved external Tatneft Qwen client")
        self._llm = llm
        self._runner = runner
        self._economics = economics
        self._registry = registry or build_grounded_tool_registry()
        self._exporter = exporter

    async def run(
        self, request: CycleRequest, run_dir: str | Path, *, run_id: str
    ) -> CycleResult:
        if not _ID.fullmatch(run_id) or run_id in {".", ".."}:
            raise CycleError("run_id violates the bounded contract")
        destination = Path(run_dir).resolve()
        if destination.exists():
            raise FileExistsError(f"run directory already exists: {destination}")
        controls_evidence = _controls_evidence(request)
        terminal_evidence: dict[str, Any] = {"available": False}
        registry = self._registry.extended(
            (
                ToolDefinition(
                    VERIFY_FULL_CONTROLS,
                    "Return a hash-backed digest of every validated monthly control.",
                    _EMPTY_SCHEMA,
                    lambda _arguments, _context: controls_evidence,
                ),
                ToolDefinition(
                    VERIFY_TERMINAL_EVIDENCE,
                    "Return authenticated terminal OPM, SUMMARY, export and CHDD evidence.",
                    _EMPTY_SCHEMA,
                    lambda _arguments, _context: terminal_evidence,
                ),
            )
        )
        role_tools = {role: tuple(names) for role, names in GROUNDED_ROLE_TOOLS.items()}
        role_tools[AgentRole.PLANNER] += (VERIFY_FULL_CONTROLS,)
        role_tools[AgentRole.CRITIC] += (VERIFY_TERMINAL_EVIDENCE,)
        agents = AgentWorkflow(
            self._llm,
            registry,
            role_tools=role_tools,
            required_tools={
                AgentRole.PLANNER: (VERIFY_FULL_CONTROLS,),
                AgentRole.CRITIC: (VERIFY_TERMINAL_EVIDENCE,),
            },
        )
        planning_context = _agent_context(request, controls_evidence)
        planning = await agents.run_plan(planning_context, run_id=run_id)
        if not all(decision.approved for decision in planning.decisions):
            raise CycleRejected("Qwen planning rejected controls before OPM execution")

        prepared = self._runner.prepare(
            request.source,
            destination,
            deck=request.deck,
            normalize_model_y=request.normalize_model_y,
        )
        schedule_path = _inside(
            prepared.input_dir,
            request.schedule_relative_path,
            "prepared schedule",
        )
        source_schedule = _regular_bytes(schedule_path, "prepared schedule")
        try:
            source_text = source_schedule.decode("utf-8")
        except UnicodeError as exc:
            raise CycleError("prepared schedule must be UTF-8") from exc
        overlay = apply_schedule_overlay(
            source_text,
            request.controls,
            known_wells={action.well for action in request.controls},
        )
        if (
            overlay.action_count != len(request.controls)
            or overlay.controls_sha256 != request.controls_sha256
        ):
            raise CycleError("schedule overlay disagrees with full controls digest")
        schedule_path.write_text(overlay.text, encoding="utf-8")
        result = self._runner._run_prepared(
            prepared, parsing_strictness=request.parsing_strictness
        )
        if _sha256_file(result.manifest_path) != result.manifest_sha256:
            raise CycleError("OPM run manifest changed after execution")
        report, extraction = self._runner.extract_summary_report(
            result, result.run_dir / "summary-report.txt"
        )
        canonical = result.run_dir / "canonical"
        chdd_csv = canonical / "chdd.csv"
        trajectory_csv = canonical / "trajectory.csv"
        export_manifest = canonical / "manifest.json"
        exported = self._exporter(
            report,
            chdd_csv,
            trajectory_csv,
            export_manifest,
            scenario_id=request.scenario_id,
            source_model=request.source_model,
            opm_run_manifest=result.manifest_path,
            summary_extraction_manifest=extraction,
            deck_dir=result.deck_path.parent,
            density_map=request.density_map,
            unit_system=prepared.unit_system,
        )
        export_sha256 = _authenticate_export(
            exported,
            export_manifest,
            report,
            extraction,
            result.manifest_path,
            chdd_csv,
            trajectory_csv,
        )
        economics = self._economics.calculate(
            _csv_rows(chdd_csv),
            start_year=request.start_year,
            output_dir=result.run_dir / f"economics-{request.start_year}",
            charge_initial_pump=request.charge_initial_pump,
        )
        economics_sha256 = _authenticate_economics(
            economics, chdd_csv, request.start_year
        )
        terminal_evidence.update(
            {
                "available": True,
                "controls": controls_evidence,
                "schedule": {
                    "action_count": overlay.action_count,
                    "input_sha256": _sha256_file(schedule_path),
                    "overlay_sha256": overlay.sha256,
                },
                "opm": {
                    "complete": True,
                    "gdm_executed": True,
                    "image": OPM_IMAGE,
                    "pinned_image_verified": True,
                    "manifest_sha256": result.manifest_sha256,
                },
                "summary": {
                    "authenticated": True,
                    "report_sha256": _sha256_file(report),
                    "extraction_manifest_sha256": _sha256_file(extraction),
                },
                "export": {"authenticated": True, "manifest_sha256": export_sha256},
                "economics": {
                    "official_chdd_complete": True,
                    "manifest_sha256": economics_sha256,
                    "total_chdd_m": economics.total_chdd_m,
                    "profitability_index": economics.profitability_index,
                },
            }
        )
        final_state = await agents.run_critic(
            planning, _terminal_context(planning_context, terminal_evidence)
        )
        receipt = _receipt(
            request,
            final_state,
            result.run_dir,
            schedule_path,
            result.manifest_path,
            report,
            extraction,
            export_manifest,
            chdd_csv,
            trajectory_csv,
            economics,
            terminal_evidence,
            prepared.source_sha256,
        )
        receipt_path = result.run_dir / "full-cycle-receipt.json"
        receipt_bytes = _json_bytes(receipt, indent=2)
        _write_immutable(receipt_path, receipt_bytes)
        return CycleResult(
            run_id=run_id,
            run_dir=result.run_dir,
            receipt_path=receipt_path,
            receipt_sha256=sha256(receipt_bytes).hexdigest(),
            critic_approved=final_state.critic_approved,
        )


def _controls(value: Any) -> tuple[ControlAction, ...]:
    if not isinstance(value, list) or not value or len(value) > _MAX_ACTIONS:
        raise CycleError("controls must be a non-empty bounded array")
    fields = {"month", "well", "role", "status", "target", "value"}
    actions: list[ControlAction] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise CycleError("control fields are invalid")
        target = raw["value"]
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise CycleError("control value must be numeric")
        try:
            actions.append(
                ControlAction(
                    month=date.fromisoformat(str(raw["month"])),
                    well=str(raw["well"]),
                    role=WellRole(raw["role"]),
                    status=WellStatus(raw["status"]),
                    target=ControlTarget(raw["target"]),
                    value=float(target),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CycleError("control contract is invalid") from exc
    return tuple(sorted(actions, key=lambda item: (item.month, item.well, item.role.value)))


def _validate_full_horizon(actions: Sequence[ControlAction]) -> None:
    months = sorted({action.month for action in actions})
    if len(months) != 6:
        raise CycleError("full-cycle controls must cover exactly six months")
    expected = [date((months[0].year * 12 + months[0].month - 1 + i) // 12,
                     (months[0].month - 1 + i) % 12 + 1, 1) for i in range(6)]
    if months != expected:
        raise CycleError("full-cycle controls must cover six consecutive months")
    wells = {action.well for action in actions if action.month == months[0]}
    if not wells or any(
        {action.well for action in actions if action.month == month} != wells
        for month in months
    ):
        raise CycleError("every control month must cover the same wells")
    if len(actions) != len(months) * len(wells):
        raise CycleError("controls contain duplicate monthly well actions")
    if any(
        action.role is WellRole.PRODUCER
        and action.target is ControlTarget.LIQUID_RATE
        and action.value > 500.0
        for action in actions
    ):
        raise CycleError("producer liquid control exceeds 500 m3/day")


def _controls_evidence(request: CycleRequest) -> dict[str, Any]:
    months = sorted({action.month for action in request.controls})
    return {
        "verified": True,
        "complete_six_month_horizon": True,
        "action_count": len(request.controls),
        "well_count": len({action.well for action in request.controls}),
        "months": [month.isoformat() for month in months],
        "canonical_schedule_sha256": request.controls_sha256,
        "canonical_actions_sha256": request.actions_sha256,
    }


def _agent_context(
    request: CycleRequest, controls_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    value = _json_object(request.context, "context")
    value.pop("candidate_controls", None)
    value.pop("controls", None)
    facts = value.get("facts", {})
    constraints = value.get("constraints", {})
    if not isinstance(facts, dict) or not isinstance(constraints, dict):
        raise CycleError("context facts and constraints must be objects")
    facts.pop("bounded_controls", None)
    constraints.pop("agent_tool_action_cap", None)
    value["case_id"] = request.scenario_id
    value["facts"] = {**facts, "full_controls": dict(controls_evidence)}
    value["constraints"] = {
        **constraints,
        "candidate_controls_are_full_six_month_schedule": True,
        "candidate_controls_scope": "exact_full_six_month_operator_input",
        "full_controls_validated_by_agent_tool": True,
        "full_controls_hash_tool": VERIFY_FULL_CONTROLS,
        "qwen_recommendation_only": True,
    }
    return value


def _terminal_context(
    context: Mapping[str, Any], terminal_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    value = _json_object(context, "planning context")
    facts = dict(value.get("facts", {}))
    facts["terminal_evidence"] = dict(terminal_evidence)
    value["facts"] = facts
    value["readiness"] = {
        "full_controls_verified": True,
        "opm_complete": True,
        "gdm_executed": True,
        "summary_authenticated": True,
        "export_authenticated": True,
        "official_chdd_complete": True,
    }
    return value


def _authenticate_export(
    expected: Mapping[str, Any],
    manifest_path: Path,
    report: Path,
    extraction: Path,
    opm_manifest: Path,
    chdd_csv: Path,
    trajectory_csv: Path,
) -> str:
    manifest, raw = _json_file(manifest_path, "canonical export manifest")
    if manifest != expected or manifest.get("schema_version") != 1 or manifest.get("generator") != "timesoil.aios.opm_chdd":
        raise CycleError("canonical export manifest failed authentication")
    provenance = manifest.get("provenance")
    source = manifest.get("source")
    outputs = manifest.get("outputs")
    if not all(isinstance(item, dict) for item in (provenance, source, outputs)):
        raise CycleError("canonical export manifest is incomplete")
    assert isinstance(provenance, dict) and isinstance(source, dict) and isinstance(outputs, dict)
    if (
        _manifest_link(manifest_path, provenance.get("opm_run_manifest")) != opm_manifest.resolve()
        or provenance.get("opm_run_manifest_sha256") != _sha256_file(opm_manifest)
        or _manifest_link(manifest_path, provenance.get("summary_extraction_manifest")) != extraction.resolve()
        or provenance.get("summary_extraction_manifest_sha256") != _sha256_file(extraction)
        or _manifest_link(manifest_path, source.get("summary_csv")) != report.resolve()
        or source.get("summary_csv_sha256") != _sha256_file(report)
    ):
        raise CycleError("canonical export input lineage mismatch")
    for name, path in (("chdd_csv", chdd_csv), ("track2_csv", trajectory_csv)):
        record = outputs.get(name)
        if not isinstance(record, dict) or record.get("name") != path.name or record.get("sha256") != _sha256_file(path):
            raise CycleError(f"canonical export {name} failed authentication")
    return sha256(raw).hexdigest()


def _authenticate_economics(
    result: EconomicResult, source_chdd: Path, start_year: int
) -> str:
    manifest, raw = _json_file(result.manifest_path, "economics manifest")
    artifacts = manifest.get("artifacts")
    if (
        result.manifest_path.resolve().parent != result.output_dir.resolve()
        or manifest.get("schema_version") != 1
        or manifest.get("adapter") != "timesoil.aios.CHDDEconomicsAdapter"
        or manifest.get("start_year") != start_year
        or not isinstance(artifacts, dict)
    ):
        raise CycleError("economics manifest failed authentication")
    input_path = _inside(result.output_dir, _relative(artifacts.get("input"), "economics input"), "economics input")
    result_path = _inside(result.output_dir, _relative(artifacts.get("result"), "economics result"), "economics result")
    if (
        manifest.get("input_sha256") != _sha256_file(input_path)
        or manifest.get("result_sha256") != _sha256_file(result_path)
        or normalize_chdd_rows(_csv_rows(input_path))
        != normalize_chdd_rows(_csv_rows(source_chdd))
    ):
        raise CycleError("economics input or result lineage mismatch")
    calculated, _ = _json_file(result_path, "economics result")
    summary = calculated.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("totalChddM") != result.total_chdd_m
        or summary.get("profitabilityIndex") != result.profitability_index
        or manifest.get("summary") != summary
    ):
        raise CycleError("economics result summary mismatch")
    return sha256(raw).hexdigest()


def _receipt(
    request: CycleRequest,
    state: AgentState,
    run_dir: Path,
    schedule: Path,
    opm_manifest: Path,
    report: Path,
    extraction: Path,
    export_manifest: Path,
    chdd_csv: Path,
    trajectory_csv: Path,
    economics: EconomicResult,
    terminal_evidence: Mapping[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    artifacts = {
        "exact_opm_input_schedule": _artifact(schedule, run_dir),
        "opm_run_manifest": _artifact(opm_manifest, run_dir),
        "summary_report": _artifact(report, run_dir),
        "summary_extraction_manifest": _artifact(extraction, run_dir),
        "canonical_export_manifest": _artifact(export_manifest, run_dir),
        "canonical_chdd_csv": _artifact(chdd_csv, run_dir),
        "canonical_trajectory_csv": _artifact(trajectory_csv, run_dir),
        "economics_manifest": _artifact(economics.manifest_path, run_dir),
    }
    economics_manifest, _ = _json_file(economics.manifest_path, "economics manifest")
    for name, relative in economics_manifest["artifacts"].items():
        artifacts[f"economics_{name}"] = _artifact(economics.output_dir / relative, run_dir)
    return {
        "schema": "timesoil.aios.full-cycle/v1",
        "complete": True,
        "critic_approved": state.critic_approved,
        "approval_is_recommendation_only": True,
        "local_model_used": False,
        "run_id": state.run_id,
        "request_sha256": request.request_sha256,
        "source_sha256": source_sha256,
        "controls": _controls_evidence(request),
        "agent": {
            "provider": "Tatneft LiteLLM",
            "endpoint": APPROVED_BASE_URL,
            "model": APPROVED_MODEL,
            "decisions": [_decision(decision) for decision in state.decisions],
        },
        "terminal_evidence": dict(terminal_evidence),
        "economics": {
            "start_year": request.start_year,
            "total_chdd_m": economics.total_chdd_m,
            "profitability_index": economics.profitability_index,
        },
        "artifacts": artifacts,
    }


def _decision(value: Any) -> dict[str, Any]:
    return {
        "role": value.role.value,
        "summary": value.summary,
        "recommendation": value.recommendation,
        "evidence": list(value.evidence),
        "approved": value.approved,
        "tools": [
            {"call_id": item.call_id, "tool": item.tool, "output": dict(item.output)}
            for item in value.tool_evidence
        ],
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    data = _regular_bytes(path, "receipt artifact")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CycleError("receipt artifact is outside the run directory") from exc
    return {"path": relative, "bytes": len(data), "sha256": sha256(data).hexdigest()}


def _action(value: ControlAction) -> dict[str, Any]:
    return {
        "month": value.month.isoformat(),
        "well": value.well,
        "role": value.role.value,
        "status": value.status.value,
        "target": value.target.value,
        "value": value.value,
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    data = _regular_bytes(path, "canonical CHDD CSV")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise CycleError("canonical CHDD CSV must be UTF-8") from exc
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != CHDD_FIELDS:
        raise CycleError("canonical CHDD CSV fields are invalid")
    rows = [dict(row) for row in reader]
    if not rows:
        raise CycleError("canonical CHDD CSV is empty")
    return rows


def _json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        normalized = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CycleError(f"{label} must contain finite JSON data") from exc
    if not isinstance(normalized, dict):
        raise CycleError(f"{label} must be a JSON object")
    return normalized


def _json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CycleError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CycleError(f"{label} must be a JSON object")
    return value, raw


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CycleError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _sha256_file(path: Path) -> str:
    return sha256(_regular_bytes(path, "hashed artifact")).hexdigest()


def _manifest_link(manifest: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CycleError("manifest link must be a relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute():
        raise CycleError("manifest link must be a relative POSIX path")
    return (manifest.parent / Path(*relative.parts)).resolve()


def _inside(root: Path, relative: PurePosixPath, label: str) -> Path:
    path = (root.resolve() / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CycleError(f"{label} escapes its root") from exc
    return path


def _relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise CycleError(f"{label} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise CycleError(f"{label} must be a safe relative POSIX path")
    return path


def _path(value: Any, root: Path, label: str) -> Path:
    text = _text(value, label, 4096)
    path = Path(text).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value or len(value) > limit:
        raise CycleError(f"{label} is invalid")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label, 64)
    if not _ID.fullmatch(text):
        raise CycleError(f"{label} is invalid")
    return text


def _reject_sensitive_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str) or _SENSITIVE_KEY.search(key):
                raise CycleError("context contains a forbidden sensitive key")
            _reject_sensitive_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_sensitive_keys(nested)


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)
