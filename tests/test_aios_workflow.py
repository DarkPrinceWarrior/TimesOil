from __future__ import annotations

import asyncio
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock, Mock, patch

import pytest

from timesoil.aios.economics import CHDD_FIELDS, EconomicResult
from timesoil.aios.llm import (
    APPROVED_BASE_URL,
    APPROVED_MODEL,
    ChatMessage,
    LLMResponse,
    TatneftLLMClient,
    ToolCall,
)
from timesoil.aios.workflow import CycleError, CycleRequest, FullCycleWorkflow
from timesoil.aios import cli


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


class _LLM(TatneftLLMClient):
    def __init__(self) -> None:
        self.config = SimpleNamespace(base_url=APPROVED_BASE_URL, model=APPROVED_MODEL)
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
    def prepare(
        self,
        source: Path,
        run_dir: Path,
        *,
        deck: str,
        normalize_model_y: bool,
    ) -> Any:
        del source, deck, normalize_model_y
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True)
        (input_dir / "schedule.inc").write_text(
            "DATES\n  1 JAN 2007 /\n/\n"
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
    assert receipt["terminal_evidence"]["opm"]["gdm_executed"] is True
    assert receipt["terminal_evidence"]["summary"]["authenticated"] is True
    assert receipt["economics"]["total_chdd_m"] == 12.5
    assert receipt["critic_approved"] is False
    assert receipt["agent"]["decisions"][-1]["approved"] is False
    assert result.receipt_path.stat().st_mode & 0o777 == 0o444
    assert _sha(result.receipt_path) == result.receipt_sha256


def test_cycle_request_rejects_partial_scope_and_sensitive_context(tmp_path: Path) -> None:
    partial = _request(tmp_path)
    partial["controls"] = partial["controls"][:-103]
    with pytest.raises(CycleError, match="six months"):
        CycleRequest.from_mapping(partial)

    sensitive = _request(tmp_path)
    sensitive["context"] = {"api_key": "must-not-cross-boundary"}
    with pytest.raises(CycleError, match="sensitive"):
        CycleRequest.from_mapping(sensitive)

    with pytest.raises(CycleError, match="external Tatneft Qwen"):
        FullCycleWorkflow(Mock(), runner=_Runner(), economics=_Economics())


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
