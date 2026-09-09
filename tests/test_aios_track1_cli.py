from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

from timesoil.aios.agents import ROLE_ORDER
from timesoil.aios.llm import LLMResponse, ToolCall
from timesoil.aios.opm import OpmGdmBackend
from timesoil.aios.track1 import DeterministicGdmBackend


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_track1_mpc.py"
SPEC = importlib.util.spec_from_file_location("timesoil_track1_cli_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


class _AgentClient:
    selected_index = 1
    rejected_role: str | None = None

    def __init__(self, _: Any) -> None:
        pass

    async def __aenter__(self) -> _AgentClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    async def chat(self, _: Any, **kwargs: Any) -> LLMResponse:
        if kwargs.get("tool_choice") == {"type": "function", "function": {"name": "verify_month_evidence"}}:
            return LLMResponse("verify", None, "tool_calls", (ToolCall("verify-1", "verify_month_evidence", {}),))
        calls = (
            (ToolCall("select-1", "select_candidate", {"index": self.selected_index}),)
            if kwargs.get("tools")
            else ()
        )
        return LLMResponse("preliminary", None, "tool_calls" if calls else "stop", calls)

    async def structured(
        self, _: Any, *, schema: dict[str, Any], **__: Any
    ) -> tuple[dict[str, Any], LLMResponse]:
        role = schema["properties"]["role"]["const"]
        return (
            {
                "role": role,
                "summary": f"summary {role}",
                "recommendation": f"recommendation {role}",
                "evidence": ["deterministic evidence"],
                "approved": role != self.rejected_role,
            },
            LLMResponse("{}", None, "stop"),
        )


def _raises(error: type[BaseException], match: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except error as exc:
        assert match in str(exc)
    else:
        raise AssertionError(f"{error.__name__} was not raised")


def _payload() -> dict[str, object]:
    return {
        "schema": "timesoil.aios.track1-mpc-input/v1",
        "case": {
            "case_id": "model-y-cli-test",
            "start": "2014-01-01",
            "end": "2014-01-01",
            "economics_start": "2014-01-01",
            "producers": ["P1"],
            "injectors": ["I1"],
        },
        "initial_state": {
            "case_id": "model-y-cli-test",
            "month": "2014-01-01",
            "restart_ref": "restart-0",
            "wells": [],
        },
        "candidates": {
            "2014-01-01": [
                [
                    {
                        "well": "P1",
                        "role": "producer",
                        "status": "OPEN",
                        "target": "LRAT",
                        "value": 140.0,
                    },
                    {
                        "well": "I1",
                        "role": "injector",
                        "status": "OPEN",
                        "target": "WRAT",
                        "value": 100.0,
                    },
                ],
                [
                    {
                        "well": "P1",
                        "role": "producer",
                        "status": "OPEN",
                        "target": "LRAT",
                        "value": 100.0,
                    },
                    {
                        "well": "I1",
                        "role": "injector",
                        "status": "OPEN",
                        "target": "WRAT",
                        "value": 100.0,
                    },
                ],
            ]
        },
        "opm": {
            "source": "model.DATA",
            "runs_dir": "candidate-runs",
            "deck": "MODEL.DATA",
            "schedule_include": "schedule.inc",
            "normalize_model_y": True,
            "parsing_strictness": "low",
            "source_model": "Model Y",
            "timeout_seconds": 30.0,
        },
    }


def _config(tmp_path: Path, *, compact: bool = False) -> Any:
    (tmp_path / "model.DATA").write_text("RUNSPEC\nEND\n", encoding="utf-8")
    path = tmp_path / "track1.json"
    path.write_text(
        json.dumps(_payload(), separators=(",", ":") if compact else None),
        encoding="utf-8",
    )
    return cli.load_config(path)


def test_cli_contract_produces_deterministic_hashed_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    compact_dir = tmp_path / "compact"
    compact_dir.mkdir()
    compact = _config(compact_dir, compact=True)
    assert compact.input_sha256 == config.input_sha256
    assert compact.run_id == config.run_id
    (compact_dir / "model.DATA").write_text("RUNSPEC\nTITLE\nchanged /\nEND\n", encoding="utf-8")
    changed = cli.load_config(compact_dir / "track1.json")
    assert changed.config_sha256 == config.config_sha256
    assert changed.source_sha256 != config.source_sha256
    assert changed.run_id != config.run_id
    backend = cli.build_backend(config)
    assert isinstance(backend, OpmGdmBackend)
    assert backend.runs_dir == tmp_path / "candidate-runs"
    assert backend.schedule_include == "schedule.inc"
    assert backend.normalize_model_y is True
    assert backend.parsing_strictness == "low"

    first, summary = cli.execute(config, DeterministicGdmBackend())
    second, second_summary = cli.execute(config, DeterministicGdmBackend())
    assert first == second
    assert summary == second_summary

    run_dir = cli.publish(tmp_path / "runs", config.run_id, first)
    result = (run_dir / "result.json").read_bytes()
    schedule = (run_dir / "wells_schedule.inc").read_bytes()
    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["artifacts"]["result"]["sha256"] == sha256(result).hexdigest()
    assert manifest["artifacts"]["schedule"]["sha256"] == sha256(schedule).hexdigest()
    assert schedule.decode() == json.loads(result)["schedule"]["text"]
    assert summary["manifest_sha256"] == sha256(manifest_bytes).hexdigest()
    assert (run_dir / "manifest.sha256").read_text(encoding="ascii") == (
        f"{summary['manifest_sha256']}  manifest.json\n"
    )
    assert json.loads(result)["evidence"]["step_economics"][0]["npv_million_rub"] == 130.0
    assert all(not (path.stat().st_mode & 0o222) for path in run_dir.iterdir())

    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    _raises(
        FileExistsError,
        "refusing to overwrite",
        lambda: cli.publish(tmp_path / "runs", config.run_id, first),
    )
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


def test_cli_executes_exact_six_month_horizon(tmp_path: Path) -> None:
    payload = _payload()
    case = payload["case"]
    candidates = payload["candidates"]
    assert isinstance(case, dict) and isinstance(candidates, dict)
    case["end"] = "2014-06-01"
    january = candidates["2014-01-01"]
    payload["candidates"] = {
        f"2014-{month:02d}-01": january for month in range(1, 7)
    }
    (tmp_path / "model.DATA").write_text("RUNSPEC\nEND\n", encoding="utf-8")
    config_path = tmp_path / "track1.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    outputs, _ = cli.execute(
        cli.load_config(config_path), DeterministicGdmBackend()
    )
    result = json.loads(outputs[Path("result.json")])

    assert [item["month"] for item in result["evidence"]["trajectories"]] == [
        f"2014-{month:02d}-01" for month in range(1, 7)
    ]
    assert sorted({item["month"] for item in result["schedule"]["actions"]}) == [
        f"2014-{month:02d}-01" for month in range(1, 7)
    ]


def test_cli_rejects_duplicate_keys_and_symlink_paths(tmp_path: Path) -> None:
    source = tmp_path / "model.DATA"
    source.write_text("RUNSPEC\nEND\n", encoding="utf-8")
    malformed = tmp_path / "duplicate.json"
    malformed.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    _raises(ValueError, "duplicate JSON key", lambda: cli.load_config(malformed))

    config_path = tmp_path / "track1.json"
    config_path.write_text(json.dumps(_payload()), encoding="utf-8")
    linked_config = tmp_path / "linked.json"
    linked_config.symlink_to(config_path)
    _raises(ValueError, "symlink path component", lambda: cli.load_config(linked_config))

    linked_source = tmp_path / "linked.DATA"
    linked_source.symlink_to(source)
    linked_payload = _payload()
    linked_payload["opm"] = {"source": "linked.DATA"}
    linked_source_config = tmp_path / "linked-source.json"
    linked_source_config.write_text(json.dumps(linked_payload), encoding="utf-8")
    _raises(
        ValueError,
        "symlink path component",
        lambda: cli.load_config(linked_source_config),
    )

    config = cli.load_config(config_path)
    outputs, _ = cli.execute(config, DeterministicGdmBackend())
    target = tmp_path / "target-runs"
    target.mkdir()
    linked_runs = tmp_path / "linked-runs"
    linked_runs.symlink_to(target, target_is_directory=True)
    _raises(
        ValueError,
        "symlink path component",
        lambda: cli.publish(linked_runs, config.run_id, outputs),
    )
    assert not tuple(target.iterdir())


def test_cli_fails_closed_if_source_changes_during_run(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class MutatingBackend(DeterministicGdmBackend):
        def run_from_restart(self, case, state, actions):  # type: ignore[no-untyped-def]
            config.source.write_text("RUNSPEC\nTITLE\nmutated /\nEND\n", encoding="utf-8")
            return super().run_from_restart(case, state, actions)

    _raises(
        RuntimeError,
        "OPM source changed",
        lambda: cli.execute(config, MutatingBackend()),
    )
    assert not (tmp_path / "runs").exists()


def test_cli_manifest_pins_and_rechecks_executed_scripts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    proof_script = tmp_path / "run_model_y_track1_proof.py"
    proof_script.write_text("# immutable proof source\n", encoding="utf-8")
    contract = cli._script_source_contract(proof_script)

    outputs, _ = cli.execute(
        config,
        DeterministicGdmBackend(),
        script_source_contract=contract,
    )

    manifest = json.loads(outputs[Path("manifest.json")])
    result = json.loads(outputs[Path("result.json")])
    assert manifest["script_source_contract"] == contract
    assert result["script_source_contract"] == contract
    assert contract["run_model_y_track1_proof.py"]["sha256"] == sha256(
        proof_script.read_bytes()
    ).hexdigest()

    proof_script.write_text("# changed proof source\n", encoding="utf-8")
    _raises(
        RuntimeError,
        "run_model_y_track1_proof.py changed",
        lambda: cli.execute(
            config,
            DeterministicGdmBackend(),
            script_source_contract=contract,
        ),
    )


def test_agent_mode_selects_one_candidate_and_records_each_month(
    tmp_path: Path, monkeypatch: Any
) -> None:
    payload = _payload()
    case = payload["case"]
    candidates = payload["candidates"]
    assert isinstance(case, dict) and isinstance(candidates, dict)
    case["end"] = "2014-02-01"
    candidates["2014-02-01"] = candidates["2014-01-01"]
    (tmp_path / "model.DATA").write_text("RUNSPEC\nEND\n", encoding="utf-8")
    config_path = tmp_path / "track1.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    secret = "test-secret-must-not-leak"
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setattr(cli, "TatneftLLMClient", _AgentClient)
    _AgentClient.selected_index = 1
    _AgentClient.rejected_role = None

    agent_log = tmp_path / "agent.jsonl"
    outputs, summary = cli.execute(
        cli.load_config(config_path),
        DeterministicGdmBackend(),
        agent=True,
        agent_log=agent_log,
    )
    result = json.loads(outputs[Path("result.json")])
    trajectories = result["evidence"]["trajectories"]
    records = result["agent"]["records"]

    assert [
        (item["month"], item["next_state"]["month"]) for item in trajectories
    ] == [("2014-01-01", "2014-02-01"), ("2014-02-01", "2014-03-01")]
    assert {
        item["value"]
        for item in result["schedule"]["actions"]
        if item["well"] == "P1"
    } == {100.0}
    assert [item["phase"] for item in records] == [
        "planning",
        "terminal_month_review",
        "planning",
        "terminal_month_review",
    ]
    assert all(
        tuple(decision["role"] for decision in item["agent"]["decisions"])
        == tuple(role.value for role in ROLE_ORDER)
        for item in records
        if item["phase"] == "terminal_month_review"
    )
    assert secret not in outputs[Path("result.json")].decode()
    assert secret not in agent_log.read_text(encoding="utf-8")
    assert secret not in json.dumps(summary)


def test_full_field_agent_can_change_both_roles_outside_the_candidate_bank(tmp_path: Path, monkeypatch: Any) -> None:
    from dataclasses import replace
    config = _config(tmp_path)
    january = config.candidates[config.case.start][0]
    months = [config.case.start.replace(month=m) for m in (1, 2, 3)]
    config = replace(config, case=replace(config.case, end=months[-1]), candidates={
        month: (tuple(replace(a, month=month, value=120 if month.month == 3 and a.well == "I1" else a.value)
                      for a in january),) for month in months})
    updates = [{"well": "P1", "status": "SHUT", "target": "LRAT", "value": 0.0},
               {"well": "I1", "status": "OPEN", "target": "WRAT", "value": 90.0}]

    class FullFieldClient(_AgentClient):
        rejected_role = None
        proposals = 0
        reviews = 0

        async def structured(self, messages: Any, **kwargs: Any):
            decision, response = await super().structured(messages, **kwargs)
            if decision["role"] == "critic":
                decision["approved"] = self.reviews > 0
                type(self).reviews += 1
            return decision, response

        async def chat(self, _: Any, **kwargs: Any) -> LLMResponse:
            if kwargs.get("tool_choice") == {"type": "function", "function": {"name": "verify_month_evidence"}}:
                return await super().chat(_, **kwargs)
            proposal = ([{**updates[0], "value": 5.0}] if self.proposals == 0
                        else updates if self.proposals == 1 else [])
            calls = ((ToolCall("propose-1", "propose_controls", {"updates": proposal}),)
                     if kwargs.get("tools") else ())
            if calls:
                type(self).proposals += 1
            return LLMResponse("preliminary", None, "tool_calls" if calls else "stop", calls)

    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setattr(cli, "TatneftLLMClient", FullFieldClient)
    outputs, _ = cli.execute(config, DeterministicGdmBackend(), agent=True, full_field=True)
    result = json.loads(outputs[Path("result.json")])
    actions = result["schedule"]["actions"]
    assert result["agent"]["records"][0]["phase"] == "invalid_proposal"
    assert FullFieldClient.reviews == 4
    rejected = [r for r in result["agent"]["records"] if r["phase"] == "rejected_month_review"]
    assert len(rejected) == 1 and not rejected[0]["agent"]["decisions"][-1]["approved"]
    assert not result["agent"]["records"][0]["simulator_executed"]
    assert [a["status"] for a in actions if a["well"] == "P1"] == ["SHUT"] * 3
    assert [a["value"] for a in actions if a["well"] == "I1"] == [90, 90, 120]
    assert all(r["agent"]["context"]["verified_inventory"]["well_count"] == 2
               for r in result["agent"]["records"] if r["phase"] == "planning")
    assert result["agent"]["records"][-1]["phase"] == "terminal_month_review"
    review = result["agent"]["records"][-1]["agent"]["context"]
    assert review["verified_constraints"]["inventory_matches_case"]
    assert review["verified_constraints"]["well_count"] == 2
    assert review["provenance"]["verified_state_receipt"] == result["evidence"]["trajectories"][-1]["next_state"]["restart_ref"]
    tool_evidence = result["agent"]["records"][-1]["agent"]["decisions"][-1]["tool_evidence"]
    assert len(tool_evidence) == 1 and tool_evidence[0]["tool"] == "verify_month_evidence"
    assert tool_evidence[0]["output"]["provenance"] == review["provenance"]
    assert tool_evidence[0]["output"]["constraints"] == review["verified_constraints"]
    assert tool_evidence[0]["output"]["trajectory"] == review["trajectory"]
    assert tool_evidence[0]["output"]["claim_limits"] == review["claim_limits"]
    assert tool_evidence[0]["output"]["planning_economics"] == review["planning_economics"]
    baseline = config.candidates[config.case.start][0]
    tail = cli._continuation_tail(config, config.initial_state, cli._propose_controls(config.case, baseline, updates))
    assert [a.status.value for a in tail if a.well == "P1"] == ["SHUT", "SHUT"]
    assert [a.value for a in tail if a.well == "I1"] == [90, 120]
    _raises(ValueError, "unknown or duplicate", lambda: cli._propose_controls(config.case, baseline, updates * 2))
    _raises(ValueError, "exceeds", lambda: cli._propose_controls(config.case, baseline, [{**updates[0], "status": "OPEN", "value": 501}]))
    _raises(ValueError, "every well", lambda: cli._propose_controls(config.case, baseline[:1], []))


def test_agent_mode_fails_closed_on_invalid_choice_and_critic_rejection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setattr(cli, "TatneftLLMClient", _AgentClient)
    config = _config(tmp_path)

    _AgentClient.selected_index = 2
    _AgentClient.rejected_role = None
    _raises(
        RuntimeError,
        "candidate index outside configured options",
        lambda: cli.execute(config, DeterministicGdmBackend(), agent=True),
    )

    from dataclasses import replace
    earlier_economics = replace(
        config, case=replace(config.case, economics_start=config.case.start.replace(year=2013))
    )
    _raises(
        ValueError,
        "economics_start must equal",
        lambda: cli.execute(earlier_economics, DeterministicGdmBackend(), agent=True),
    )

    _AgentClient.selected_index = 1
    _AgentClient.rejected_role = "critic"
    _raises(
        RuntimeError,
        "critic rejected simulated month",
        lambda: cli.execute(config, DeterministicGdmBackend(), agent=True),
    )
