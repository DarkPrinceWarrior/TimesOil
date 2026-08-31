from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

from timesoil.aios.opm import OpmGdmBackend
from timesoil.aios.track1 import DeterministicGdmBackend


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_track1_mpc.py"
SPEC = importlib.util.spec_from_file_location("timesoil_track1_cli_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


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
