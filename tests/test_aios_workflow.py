from __future__ import annotations

import asyncio
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, Mock, patch

import pytest

from timesoil.aios.economics import CHDD_FIELDS, EconomicResult
from timesoil.aios.llm import (
    APPROVED_MODEL,
    ChatMessage,
    ExternalQwenClient,
    LLMResponse,
    ToolCall,
)
from timesoil.aios.workflow import CycleError, CycleRequest, FullCycleWorkflow
from timesoil.aios import cli, workflow


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _execution_binding(*, commit: str = "f" * 40, marker: str = "1" * 64) -> dict[str, Any]:
    return {
        "verified": True,
        "git_commit": commit,
        "git_scoped_sources_clean": True,
        "source_count": 1,
        "source_map_sha256": marker,
        "sources": {
            "src/timesoil/aios/workflow.py": {"bytes": 1, "sha256": marker}
        },
    }


def _request(root: Path) -> dict[str, Any]:
    controls = []
    for month in range(1, 7):
        controls.extend(
            {
                "month": f"2007-{month:02d}-01",
                "well": f"P{well}",
                "role": "producer",
                "status": "OPEN",
                "target": "LRAT",
                "value": 100.0,
            }
            for well in range(1, 72)
        )
        controls.extend(
            {
                "month": f"2007-{month:02d}-01",
                "well": f"I{well}",
                "role": "injector",
                "status": "OPEN",
                "target": "WRAT",
                "value": 90.0,
            }
            for well in range(1, 33)
        )
    return {
        "context": {
            "track": 2,
            "facts": {"bounded_controls": {"action_count": 412}},
            "constraints": {
                "agent_tool_action_cap": 512,
                "candidate_controls_are_full_six_month_schedule": False,
            },
        },
        "controls": controls,
        "source": str(root / "source"),
        "deck": "CASE.DATA",
        "schedule_relative_path": "schedule.inc",
        "scenario_id": "selected",
        "source_model": "model_z_opm",
        "start_year": 2007,
        "charge_initial_pump": False,
    }


class _LLM(ExternalQwenClient):
    def __init__(self) -> None:
        self.config = SimpleNamespace(model=APPROVED_MODEL)
        self.forced_tools: list[str] = []
        self.states: list[str] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        reasoning: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        del reasoning, tools, max_tokens, timeout_seconds
        self.states.append(messages[-1].content)
        calls: tuple[ToolCall, ...] = ()
        if isinstance(tool_choice, Mapping):
            name = str(tool_choice["function"]["name"])
            self.forced_tools.append(name)
            calls = (ToolCall(f"call-{len(self.forced_tools)}", name, {}),)
        return LLMResponse("bounded", None, "tool_calls" if calls else "stop", calls)

    async def structured(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        del messages, schema_name, max_tokens, timeout_seconds
        role = schema["properties"]["role"]["const"]
        return (
            {
                "role": role,
                "summary": f"checked {role}",
                "recommendation": "continue" if role != "critic" else "reject",
                "evidence": ["tool evidence"],
                "approved": role != "critic",
            },
            LLMResponse("{}", None, "stop"),
        )


class _Runner:
    def __init__(self) -> None:
        self.executions = 0

    def prepare(
        self,
        source: Path,
        run_dir: Path,
        *,
        deck: str,
    ) -> Any:
        del source, deck
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True)
        producers = "".join(
            f" 'P{well}' 'OPEN' 'LRAT' /\n" for well in range(1, 72)
        )
        injectors = "".join(
            f" 'I{well}' 'WATER' 'OPEN' 'RATE' /\n" for well in range(1, 33)
        )
        (input_dir / "schedule.inc").write_text(
            "DATES\n  1 JAN 2007 /\n/\n"
            f"WCONPROD\n{producers}/\n"
            f"WCONINJE\n{injectors}/\n"
            "DATES\n  1 FEB 2007 /\n/\n"
            "DATES\n  1 MAR 2007 /\n/\n"
            "DATES\n  1 APR 2007 /\n/\n"
            "DATES\n  1 MAY 2007 /\n/\n"
            "DATES\n  1 JUN 2007 /\n/\n",
            encoding="utf-8",
        )
        deck_path = input_dir / "CASE.DATA"
        deck_path.write_text("METRIC\n", encoding="utf-8")
        return SimpleNamespace(
            input_dir=input_dir,
            run_dir=run_dir,
            deck_path=deck_path,
            unit_system="METRIC",
            source_sha256="1" * 64,
        )

    def _run_prepared(self, prepared: Any, *, parsing_strictness: str) -> Any:
        self.executions += 1
        assert parsing_strictness == "strict"
        manifest = prepared.run_dir / "manifest.json"
        manifest.write_text('{"image":"pinned"}\n', encoding="utf-8")
        return SimpleNamespace(
            run_dir=prepared.run_dir,
            deck_path=prepared.deck_path,
            manifest_path=manifest,
            manifest_sha256=_sha(manifest),
        )

    def extract_summary_report(self, result: Any, report: Path) -> tuple[Path, Path]:
        report.write_text("SUMMARY\n", encoding="utf-8")
        extraction = result.run_dir / "summary-extraction.json"
        extraction.write_text('{"verified":true}\n', encoding="utf-8")
        return report, extraction


class _FailingRunner(_Runner):
    def _run_prepared(self, prepared: Any, *, parsing_strictness: str) -> Any:
        self.executions += 1
        assert parsing_strictness == "strict"
        (prepared.run_dir / "execution-started").write_text("yes\n", encoding="utf-8")
        raise RuntimeError("simulated OPM failure")


class _PlannerRejectingLLM(_LLM):
    async def structured(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], LLMResponse]:
        decision, response = await super().structured(*args, **kwargs)
        decision["approved"] = False
        return decision, response


def _exporter(
    report: Path,
    chdd_csv: Path,
    trajectory_csv: Path,
    manifest_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    chdd_csv.parent.mkdir(parents=True)
    with chdd_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CHDD_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {"DATA": "2007-01-01", "well": "P1", **{name: 1 for name in CHDD_FIELDS[2:]}}
        )
    trajectory_csv.write_text("scenario_id\nselected\n", encoding="utf-8")
    opm_manifest = Path(kwargs["opm_run_manifest"])
    extraction = Path(kwargs["summary_extraction_manifest"])
    relative = lambda path: Path(os.path.relpath(path, manifest_path.parent)).as_posix()
    manifest = {
        "schema_version": 1,
        "generator": "timesoil.aios.opm_chdd",
        "provenance": {
            "opm_run_manifest": relative(opm_manifest),
            "opm_run_manifest_sha256": _sha(opm_manifest),
            "summary_extraction_manifest": relative(extraction),
            "summary_extraction_manifest_sha256": _sha(extraction),
        },
        "source": {
            "summary_csv": relative(report),
            "summary_csv_sha256": _sha(report),
        },
        "outputs": {
            "chdd_csv": {"name": chdd_csv.name, "sha256": _sha(chdd_csv)},
            "track2_csv": {"name": trajectory_csv.name, "sha256": _sha(trajectory_csv)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class _Economics:
    def calculate(
        self,
        records: list[dict[str, str]],
        *,
        start_year: int,
        output_dir: Path,
        charge_initial_pump: bool | None,
    ) -> EconomicResult:
        assert start_year == 2007 and charge_initial_pump is False
        output_dir.mkdir()
        input_path = output_dir / "input.csv"
        with input_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(CHDD_FIELDS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        result_path = output_dir / "result.json"
        summary = {"totalChddM": 12.5, "profitabilityIndex": 1.2}
        result_path.write_text(json.dumps({"summary": summary}) + "\n", encoding="utf-8")
        (output_dir / "report.xlsx").write_bytes(b"xlsx")
        manifest = {
            "schema_version": 1,
            "adapter": "timesoil.aios.CHDDEconomicsAdapter",
            "start_year": 2007,
            "input_sha256": _sha(input_path),
            "result_sha256": _sha(result_path),
            "summary": summary,
            "artifacts": {"input": "input.csv", "result": "result.json", "report": "report.xlsx"},
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return EconomicResult(
            total_chdd_m=12.5,
            profitability_index=1.2,
            start_date="2007-01-01",
            max_date="2007-01-01",
            diagnostics={},
            output_dir=output_dir,
            manifest_path=manifest_path,
        )


def test_full_cycle_keeps_full_digest_and_real_critic_decision(tmp_path: Path) -> None:
    llm = _LLM()
    request = CycleRequest.from_mapping(_request(tmp_path))
    binding = _execution_binding()
    with patch(
        "timesoil.aios.workflow._capture_execution_source_binding",
        return_value=binding,
    ) as capture:
        result = asyncio.run(
            FullCycleWorkflow(
                llm,
                runner=_Runner(),
                economics=_Economics(),
                exporter=_exporter,
            ).run(request, tmp_path / "run", run_id="cycle-test")
        )

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert llm.forced_tools == ["verify_full_controls", "verify_terminal_evidence"]
    assert all("bounded_controls" not in state for state in llm.states)
    assert all("agent_tool_action_cap" not in state for state in llm.states)
    assert receipt["controls"]["action_count"] == 618
    assert receipt["controls"]["complete_six_month_horizon"] is True
    assert receipt["controls"]["source_well_count"] == 103
    assert receipt["controls"]["source_pre_control_action_count"] == 0
    assert len(receipt["controls"]["source_control_inventory_sha256"]) == 64
    assert receipt["schema"] == "timesoil.aios.track2-full-cycle/v1"
    assert receipt["experimental"] is True
    assert receipt["complete"] is False
    assert receipt["organizer_certified"] is False
    assert receipt["dependency_mode"] == "injected_test"
    assert receipt["external_qwen_used"] is False
    assert receipt["agent"]["transport"] == "injected_test"
    assert receipt["agent"]["model"] is None
    assert "endpoint" not in receipt["agent"]
    assert receipt["terminal_evidence"]["available"] is False
    assert receipt["terminal_evidence"]["opm"] == {
        "complete": False,
        "simulator_executed": False,
        "pinned_image_verified": False,
    }
    assert receipt["terminal_evidence"]["summary"]["authenticated"] is False
    assert receipt["terminal_evidence"]["export"]["authenticated"] is False
    assert (
        receipt["terminal_evidence"]["economics"]["official_chdd_complete"]
        is False
    )
    assert receipt["economics"] == {"official_chdd_complete": False}
    assert receipt["claim_limits"] and not any(receipt["claim_limits"].values())
    assert receipt["execution_source_binding"] == {
        **binding,
        "unchanged_after_execution": True,
    }
    assert capture.call_count == 2
    assert receipt["critic_approved"] is False
    assert receipt["agent"]["decisions"][-1]["approved"] is False
    assert result.receipt_path.stat().st_mode & 0o777 == 0o444
    assert _sha(result.receipt_path) == result.receipt_sha256


def test_only_internal_cli_factory_with_owned_client_enables_production_claims() -> None:
    class _QwenSubclass(ExternalQwenClient):
        pass

    def qwen(kind: type[ExternalQwenClient], *, owns: bool = True) -> ExternalQwenClient:
        client = object.__new__(kind)
        client.config = SimpleNamespace(model=APPROVED_MODEL)
        client._owns_client = owns
        return client

    runner = object.__new__(workflow.OpmFlowRunner)
    economics = object.__new__(workflow.CHDDEconomicsAdapter)
    client = qwen(ExternalQwenClient)
    direct = FullCycleWorkflow(client, runner=runner, economics=economics)
    assert direct._dependency_mode == "injected_test"

    with patch.object(
        workflow.CHDDEconomicsAdapter, "from_env", return_value=economics
    ) as from_env:
        production = FullCycleWorkflow._from_cli(client, timeout_seconds=123.0)
    assert production._dependency_mode == "production_cli"
    assert type(production._runner) is workflow.OpmFlowRunner
    assert production._runner.timeout_seconds == 123.0
    assert production._runner.docker_executable == "docker"
    assert production._economics is economics
    assert production._exporter is workflow.export_opm_chdd
    from_env.assert_called_once_with()

    with pytest.raises(CycleError, match="exact owned Qwen"):
        FullCycleWorkflow._from_cli(qwen(_QwenSubclass), timeout_seconds=1.0)
    with pytest.raises(CycleError, match="exact owned Qwen"):
        FullCycleWorkflow._from_cli(
            qwen(ExternalQwenClient, owns=False), timeout_seconds=1.0
        )


@pytest.mark.parametrize(
    "after",
    (
        _execution_binding(marker="2" * 64),
        _execution_binding(commit="e" * 40),
    ),
    ids=("source-drift", "commit-drift"),
)
def test_full_cycle_rejects_execution_binding_drift(
    tmp_path: Path, after: dict[str, Any]
) -> None:
    request = CycleRequest.from_mapping(_request(tmp_path))
    with patch(
        "timesoil.aios.workflow._capture_execution_source_binding",
        side_effect=(_execution_binding(), after),
    ), pytest.raises(CycleError, match="changed during full cycle"):
        asyncio.run(
            FullCycleWorkflow(
                _LLM(),
                runner=_Runner(),
                economics=_Economics(),
                exporter=_exporter,
            ).run(request, tmp_path / "run", run_id="cycle-drift")
        )

    assert not (tmp_path / "run" / "full-cycle-receipt.json").exists()


def test_execution_binding_requires_clean_tracked_sources(tmp_path: Path) -> None:
    paths = ("source.py", "Нормативы.xlsx")
    for name in paths:
        (tmp_path / name).write_text(name, encoding="utf-8")
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "add", "--", *paths), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Scorp test",
            "-c",
            "user.email=scorp@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )
    with patch.object(workflow, "_PROJECT_ROOT", tmp_path), patch.object(
        workflow, "_EXECUTION_SOURCE_PATHS", paths
    ):
        binding = workflow._capture_execution_source_binding()
        assert binding["git_scoped_sources_clean"] is True
        assert binding["source_count"] == 2
        assert set(binding["sources"]) == set(paths)

        (tmp_path / "unrelated.md").write_text("dirty but out of scope", encoding="utf-8")
        assert workflow._capture_execution_source_binding() == binding

        (tmp_path / paths[0]).write_text("changed", encoding="utf-8")
        with pytest.raises(CycleError, match="not clean"):
            workflow._capture_execution_source_binding()


def test_execution_binding_fails_closed_on_missing_source_or_git(tmp_path: Path) -> None:
    with patch.object(workflow, "_PROJECT_ROOT", tmp_path), patch.object(
        workflow, "_EXECUTION_SOURCE_PATHS", ("missing.py",)
    ), pytest.raises(CycleError, match="regular non-symlink"):
        workflow._capture_execution_source_binding()

    (tmp_path / "source.py").write_text("pass\n", encoding="utf-8")
    with patch.object(workflow, "_PROJECT_ROOT", tmp_path), patch.object(
        workflow, "_EXECUTION_SOURCE_PATHS", ("source.py",)
    ), pytest.raises(CycleError, match="git provenance"):
        workflow._capture_execution_source_binding()


def test_cycle_request_rejects_partial_scope_and_sensitive_context(tmp_path: Path) -> None:
    partial = _request(tmp_path)
    partial["controls"] = partial["controls"][:-103]
    with pytest.raises(CycleError, match="six months"):
        CycleRequest.from_mapping(partial)

    sensitive = _request(tmp_path)
    sensitive["context"] = {"api_key": "must-not-cross-boundary"}
    with pytest.raises(CycleError, match="sensitive"):
        CycleRequest.from_mapping(sensitive)

    with pytest.raises(CycleError, match="external Qwen"):
        FullCycleWorkflow(Mock(), runner=_Runner(), economics=_Economics())

    wrong_source = _request(tmp_path)
    wrong_source["source_model"] = "unverified_source"
    with pytest.raises(CycleError, match="Model Z"):
        CycleRequest.from_mapping(wrong_source)


def test_direct_cycle_request_rejects_one_month_before_prepare_or_qwen(
    tmp_path: Path,
) -> None:
    valid = CycleRequest.from_mapping(_request(tmp_path))
    with pytest.raises(CycleError, match="six months"):
        CycleRequest(
            context=valid.context,
            controls=valid.controls[:103],
            source=valid.source,
            deck=valid.deck,
            schedule_relative_path=valid.schedule_relative_path,
            scenario_id=valid.scenario_id,
            source_model=valid.source_model,
            start_year=valid.start_year,
        )


@pytest.mark.parametrize("scope", ("missing", "unknown"))
def test_full_cycle_rejects_controls_outside_source_well_scope_before_execution(
    tmp_path: Path, scope: str
) -> None:
    raw = _request(tmp_path)
    if scope == "missing":
        raw["controls"] = [
            control for control in raw["controls"] if control["well"] != "P71"
        ]
    else:
        raw["controls"].extend(
            {
                "month": f"2007-{month:02d}-01",
                "well": "UNKNOWN",
                "role": "producer",
                "status": "OPEN",
                "target": "LRAT",
                "value": 1.0,
            }
            for month in range(1, 7)
        )
    request = CycleRequest.from_mapping(raw)
    llm = _LLM()
    runner = _Runner()
    run_dir = tmp_path / "run"

    with patch(
        "timesoil.aios.workflow._capture_execution_source_binding",
        return_value=_execution_binding(),
    ), pytest.raises(CycleError, match="every prepared source schedule well"):
        asyncio.run(
            FullCycleWorkflow(
                llm,
                runner=runner,
                economics=_Economics(),
                exporter=_exporter,
            ).run(request, run_dir, run_id=f"scope-{scope}")
        )

    assert llm.states == []
    assert runner.executions == 0
    assert not run_dir.exists()


def test_full_cycle_rejects_source_role_swap_before_execution(tmp_path: Path) -> None:
    raw = _request(tmp_path)
    for control in raw["controls"]:
        if control["well"] == "P71":
            control.update(role="injector", target="WRAT")
        elif control["well"] == "I32":
            control.update(role="producer", target="LRAT")
    request = CycleRequest.from_mapping(raw)
    llm = _LLM()
    runner = _Runner()
    run_dir = tmp_path / "run"

    with patch(
        "timesoil.aios.workflow._capture_execution_source_binding",
        return_value=_execution_binding(),
    ), pytest.raises(CycleError, match="source schedule well and role"):
        asyncio.run(
            FullCycleWorkflow(
                llm,
                runner=runner,
                economics=_Economics(),
                exporter=_exporter,
            ).run(request, run_dir, run_id="scope-role-swap")
        )

    assert llm.states == []
    assert runner.executions == 0
    assert not run_dir.exists()


def test_full_cycle_cleans_prepared_tree_after_qwen_rejection(tmp_path: Path) -> None:
    request = CycleRequest.from_mapping(_request(tmp_path))
    runner = _Runner()
    run_dir = tmp_path / "run"

    with patch(
        "timesoil.aios.workflow._capture_execution_source_binding",
        return_value=_execution_binding(),
    ), pytest.raises(CycleError, match="planning rejected"):
        asyncio.run(
            FullCycleWorkflow(
                _PlannerRejectingLLM(),
                runner=runner,
                economics=_Economics(),
                exporter=_exporter,
            ).run(request, run_dir, run_id="planning-rejected")
        )

    assert runner.executions == 0
    assert not run_dir.exists()


def test_pre_execution_cleanup_refuses_replaced_directory(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    owned = destination / "owned"
    owned.mkdir(parents=True)
    (owned / "artifact").write_text("owned\n", encoding="utf-8")
    prepared = SimpleNamespace(run_dir=destination)
    identity = workflow._verify_prepared_destination(prepared, destination)
    original = tmp_path / "original-run"
    marker = destination / "replacement"
    real_rmtree = workflow.shutil.rmtree

    def swap_after_fd_binding(path: str, *, dir_fd: int) -> None:
        assert path == "."
        destination.rename(original)
        destination.mkdir()
        marker.write_text("preserve\n", encoding="utf-8")
        real_rmtree(path, dir_fd=dir_fd)

    with patch.object(
        workflow.shutil, "rmtree", side_effect=swap_after_fd_binding
    ), pytest.raises(CycleError, match="identity changed during cleanup"):
        workflow._cleanup_prepared_destination(prepared, destination, identity)

    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert list(original.iterdir()) == []


def test_pre_execution_cleanup_requires_symlink_safe_rmtree(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    marker = destination / "preserved"
    marker.write_text("yes\n", encoding="utf-8")
    prepared = SimpleNamespace(run_dir=destination)
    identity = workflow._verify_prepared_destination(prepared, destination)

    with patch.object(
        workflow.shutil.rmtree, "avoids_symlink_attacks", False
    ), pytest.raises(CycleError, match="safe prepared run directory cleanup"):
        workflow._cleanup_prepared_destination(prepared, destination, identity)

    assert marker.read_text(encoding="utf-8") == "yes\n"


def test_full_cycle_preserves_tree_after_opm_execution_starts(tmp_path: Path) -> None:
    request = CycleRequest.from_mapping(_request(tmp_path))
    runner = _FailingRunner()
    run_dir = tmp_path / "run"

    with patch(
        "timesoil.aios.workflow._capture_execution_source_binding",
        return_value=_execution_binding(),
    ), pytest.raises(RuntimeError, match="simulated OPM failure"):
        asyncio.run(
            FullCycleWorkflow(
                _LLM(),
                runner=runner,
                economics=_Economics(),
                exporter=_exporter,
            ).run(request, run_dir, run_id="opm-failure")
        )

    assert runner.executions == 1
    assert (run_dir / "execution-started").read_text(encoding="utf-8") == "yes\n"
    assert not (run_dir / "full-cycle-receipt.json").exists()


def test_source_control_inventory_uses_latest_role_for_each_month() -> None:
    source = (
        "DATES\n 1 JAN 2007 /\n/\n"
        "WCONPROD\n 'P1' 'OPEN' 'LRAT' /\n/\n"
        "DATES\n 1 FEB 2007 /\n/\n"
        "WCONINJE\n 'P1' 'WATER' 'OPEN' 'RATE' /\n/\n"
        "DATES\n 1 MAR 2007 /\n/\n"
    )

    inventory = workflow._source_control_inventory(
        source, (workflow.date(2007, 1, 1), workflow.date(2007, 2, 1))
    )

    january = inventory[workflow.date(2007, 1, 1)]["P1"]
    february = inventory[workflow.date(2007, 2, 1)]["P1"]
    assert (january.role, january.pre_control, january.first_control_month) == (
        workflow.WellRole.PRODUCER,
        False,
        workflow.date(2007, 1, 1),
    )
    assert (february.role, february.pre_control, february.first_control_month) == (
        workflow.WellRole.INJECTOR,
        False,
        workflow.date(2007, 1, 1),
    )


def test_source_control_inventory_allows_only_shut_zero_before_first_wcon() -> None:
    source = (
        "DATES\n 1 JAN 2007 /\n/\n"
        "DATES\n 1 FEB 2007 /\n/\n"
        "WCONINJE\n 'I1' 'WATER' 'OPEN' 'RATE' /\n/\n"
        "DATES\n 1 MAR 2007 /\n/\n"
    )
    january = workflow.date(2007, 1, 1)
    february = workflow.date(2007, 2, 1)
    inventory = workflow._source_control_inventory(source, (january, february))

    assert (
        inventory[january]["I1"].role,
        inventory[january]["I1"].pre_control,
        inventory[january]["I1"].first_control_month,
    ) == (workflow.WellRole.INJECTOR, True, february)
    assert not inventory[february]["I1"].pre_control

    shut = workflow.ControlAction(
        january,
        "I1",
        workflow.WellRole.INJECTOR,
        workflow.WellStatus.SHUT,
        workflow.ControlTarget.WATER_INJECTION_RATE,
        0.0,
    )
    open_early = workflow.ControlAction(
        january,
        "I1",
        workflow.WellRole.INJECTOR,
        workflow.WellStatus.OPEN,
        workflow.ControlTarget.WATER_INJECTION_RATE,
        0.0,
    )
    workflow._validate_source_well_scope((shut,), {january: inventory[january]})
    with pytest.raises(CycleError, match="first source WCON"):
        workflow._validate_source_well_scope(
            (open_early,), {january: inventory[january]}
        )


def test_source_control_inventory_rejects_non_water_injector() -> None:
    source = (
        "DATES\n 1 JAN 2007 /\n/\n"
        "WCONINJE\n 'I1' 'GAS' 'OPEN' 'RATE' /\n/\n"
        "DATES\n 1 FEB 2007 /\n/\n"
    )
    with pytest.raises(CycleError, match="non-WATER"):
        workflow._source_control_inventory(source, (workflow.date(2007, 1, 1),))


def test_single_cli_command_emits_receipt_and_preserves_critic_rejection(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    parsed = Mock(request_sha256="a" * 64)
    result = SimpleNamespace(
        run_id="cycle-fixed",
        run_dir=tmp_path / "runs" / "cycle-fixed",
        receipt_path=tmp_path / "runs" / "cycle-fixed" / "full-cycle-receipt.json",
        receipt_sha256="b" * 64,
        critic_approved=False,
    )
    output = io.StringIO()
    with patch(
        "timesoil.aios.cli.CycleRequest.from_mapping", return_value=parsed
    ), patch(
        "timesoil.aios.cli._run_full_cycle", AsyncMock(return_value=result)
    ) as execute, patch("sys.stdout", output):
        code = cli.main(
            [
                "full-cycle",
                str(request_path),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--run-id",
                "cycle-fixed",
            ]
        )

    assert code == 2
    assert json.loads(output.getvalue())["critic_approved"] is False
    execute.assert_awaited_once()
