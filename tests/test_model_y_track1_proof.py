from __future__ import annotations

from datetime import date
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from timesoil.aios.contracts import Case
from timesoil.aios.opm import (
    OPM_EXPORT_VECTORS,
    OPM_IMAGE,
    OPM_IMAGE_DIGEST,
    _summary_mapping,
    _source_digest,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_model_y_track1_proof.py"
SPEC = importlib.util.spec_from_file_location("timesoil_model_y_track1_proof_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
proof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proof
SPEC.loader.exec_module(proof)


def test_proof_requires_complete_export_vector_contract() -> None:
    assert proof.MASS_VECTORS == frozenset(OPM_EXPORT_VECTORS)
    assert proof.TARGET_PROFILES == {2007: "organizer_reference", 2014: "canonical"}
    vectors = proof._validated_manifest_vectors(
        {"vectors": _summary_mapping("METRIC")}
    )
    assert proof.MASS_VECTORS <= vectors
    with pytest.raises(RuntimeError, match="summary vector contract"):
        proof._validated_manifest_vectors({"vectors": list(vectors)})
    with pytest.raises(RuntimeError, match="summary vector contract"):
        proof._validated_manifest_vectors({"vectors": {"WOPR": {"unit": "SM3/DAY"}}})


def test_proof_horizon_is_exactly_six_months() -> None:
    assert proof._month_range(proof.HORIZON_START, proof.HORIZON_END) == tuple(
        date(2014, month, 1) for month in range(1, 7)
    )
    assert proof.HORIZON_MONTH_COUNT == 6


def test_organizer_reference_mass_and_pump_semantics(
    tmp_path: Path,
) -> None:
    workbook = (
        SCRIPT.parents[1]
        / "docs/hackathon/chdd/reference_baselines"
        / "Расчет ЧДД через OPM Flow Model_Y.xlsx"
    )
    reference = proof._organizer_reference(workbook)
    assert reference["pump_changes"] == 58
    assert reference["pump_operation_m"] == pytest.approx(104.4)
    assert reference["rebased_chdd_2014_m"] == pytest.approx(1082.233695354093)
    masses = reference["annual_mass_kt"]
    assert isinstance(masses, dict)
    records = [
        {
            "DATA": f"{year}-01-01",
            "WOMT_Diff": str(values["oil"] * 1000),
            "WLPT_Diff": str(values["liquid"] * 1000),
        }
        for year, values in masses.items()
    ]
    result_dir = tmp_path / "economics"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "summary": {"pumpChanges": 58},
                "annual": [{"pumpOperationM": 104.4}],
            }
        ),
        encoding="utf-8",
    )

    evidence = proof._assert_organizer_reference_parity(records, result_dir, workbook)

    assert evidence["pump_changes"] == 58
    assert evidence["pump_operation_m"] == pytest.approx(104.4)
    assert evidence["workbook"] == str(workbook)


def _result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


def test_guard_rejects_any_running_opm_2026_04_container() -> None:
    with pytest.raises(RuntimeError, match="another OPM 2026.04 container"):
        proof._assert_no_running_opm(
            _run=lambda *_args, **_kwargs: _result(
                "openporousmedia/opmreleases:2026.04_amd64 timesoil-model-z\n"
            )
        )


def test_guard_allows_unrelated_containers_and_fails_on_docker_error() -> None:
    proof._assert_no_running_opm(
        _run=lambda *_args, **_kwargs: _result("postgres:17 production-db\n")
    )
    with pytest.raises(RuntimeError, match="cannot verify"):
        proof._assert_no_running_opm(
            _run=lambda *_args, **_kwargs: _result("", returncode=1)
        )


def test_resume_baseline_reauthenticates_existing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    manifest = {
        "deck": "MODEL_Y/MODEL_Y.DATA",
        "summary_contract": {"overlay": "MODEL_Y/_TIMESOIL_SUMMARY.INC"},
        "command": ["docker", "run"],
        "warnings": [],
    }
    manifest_path = baseline / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = sha256(manifest_path.read_bytes()).hexdigest()
    (baseline / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )
    verified: list[tuple[Path, bool]] = []
    replayed: list[tuple[Path, Path, Path]] = []
    constructed: list[tuple[Path, dict[str, object]]] = []
    source = tmp_path / "custom-source.zip"

    class Verifier:
        def __init__(self, _runner, configured_source: Path, **kwargs) -> None:
            constructed.append((configured_source, kwargs))

        def _verify_opm_manifest(self, path: Path, *, baseline: bool) -> None:
            verified.append((path, baseline))

    monkeypatch.setattr(proof, "OpmGdmBackend", Verifier)
    monkeypatch.setattr(
        proof,
        "verify_summary_extraction",
        lambda report, extraction, run_manifest, **_kwargs: replayed.append(
            (report, extraction, run_manifest)
        ),
    )

    result, report, extraction = proof._resume_baseline(
        proof.OpmFlowRunner(docker_executable="docker"),
        baseline=baseline,
        candidates=tmp_path / "custom-candidates",
        source=source,
        deck="CUSTOM/MODEL.DATA",
        schedule_relative_path="CUSTOM/schedule.inc",
    )

    assert result.manifest_sha256 == digest
    assert result.deck_path == baseline / "input/MODEL_Y/MODEL_Y.DATA"
    assert verified == [(manifest_path, True)]
    assert replayed == [(report, extraction, manifest_path)]
    assert constructed == [
        (
            source,
            {
                "runs_dir": tmp_path / "custom-candidates",
                "deck": "CUSTOM/MODEL.DATA",
                "schedule_include": "CUSTOM/schedule.inc",
                "normalize_model_y": True,
                "parsing_strictness": "low",
                "source_model": "Model Y",
            },
        )
    ]


def test_cli_uses_explicit_fresh_host_paths_and_current_sibling_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "inputs/Model Y.zip"
    source.parent.mkdir()
    source.write_bytes(b"official-fixture")
    baseline = tmp_path / "authenticated-baseline"
    baseline.mkdir()
    workbook = tmp_path / "reference.xlsx"
    workbook.write_bytes(b"workbook-fixture")
    output_root = tmp_path / "fresh-output"
    common = [
        "--resume-baseline",
        "--source",
        str(source),
        "--baseline-dir",
        str(baseline),
        "--reference-workbook",
        str(workbook),
        "--deck",
        "CUSTOM/MODEL.DATA",
        "--schedule-relative-path",
        "CUSTOM/schedule.inc",
        "--output-root",
        str(output_root),
    ]
    monkeypatch.setattr(proof, "_assert_no_running_opm", lambda: None)
    with pytest.raises(RuntimeError, match="Model Y source SHA-256 mismatch"):
        proof.main(common)

    captured: dict[str, object] = {}
    controllers: list[Path] = []
    real_contract = proof._proof_source_contract

    class StopAfterPathThreading(RuntimeError):
        pass

    def source_contract(controller: Path) -> dict[str, object]:
        controllers.append(controller)
        return real_contract(controller)

    def resume(_runner, **kwargs):
        captured.update(kwargs)
        raise StopAfterPathThreading

    monkeypatch.setattr(proof, "SOURCE_SHA256", sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(proof, "_proof_source_contract", source_contract)
    monkeypatch.setattr(proof, "_resume_baseline", resume)
    with pytest.raises(StopAfterPathThreading):
        proof.main(common)

    assert controllers == [SCRIPT.with_name("run_track1_mpc.py")]
    current_contract = real_contract(controllers[0])
    assert current_contract["controller_execution_mode"] == "current-source-sibling"
    assert current_contract["copied_controller_bytes_exact"] is False
    assert captured["baseline"] == baseline
    assert captured["source"] == source
    assert captured["deck"] == "CUSTOM/MODEL.DATA"
    assert captured["schedule_relative_path"] == "CUSTOM/schedule.inc"
    configured_candidates = captured["candidates"]
    assert isinstance(configured_candidates, Path)
    assert configured_candidates.is_relative_to(output_root)


def test_staged_controller_source_contract_pins_executed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_scripts = tmp_path / "source/scripts"
    staged_scripts = tmp_path / "staged/scripts"
    source_scripts.mkdir(parents=True)
    staged_scripts.mkdir(parents=True)
    proof_script = source_scripts / "run_model_y_track1_proof.py"
    controller_source = source_scripts / "run_track1_mpc.py"
    controller = staged_scripts / "run_track1_mpc.py"
    proof_script.write_text("# proof\n", encoding="utf-8")
    controller_source.write_text("# controller\n", encoding="utf-8")
    controller.write_bytes(controller_source.read_bytes())
    monkeypatch.setattr(proof, "__file__", str(proof_script))

    contract = proof._proof_source_contract(controller)

    execution = contract["execution"]
    assert isinstance(execution, dict)
    assert execution["run_model_y_track1_proof.py"]["sha256"] == sha256(
        proof_script.read_bytes()
    ).hexdigest()
    assert execution["run_track1_mpc.py"]["sha256"] == sha256(
        controller.read_bytes()
    ).hexdigest()
    assert contract["copied_controller_bytes_exact"] is True
    proof._verify_proof_source_contract(contract)

    controller.write_text("# changed copy\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from source bytes"):
        proof._verify_proof_source_contract(contract)


def test_resume_uses_fresh_output_root_without_writing_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    immutable = tmp_path / "immutable-baseline"
    immutable.mkdir()
    (immutable / "canonical").mkdir()
    (immutable / "economics-2014-canonical").mkdir()
    monkeypatch.setattr(proof, "BASELINE", immutable)
    output_root = tmp_path / "resume-output"

    baseline, candidates, final, config, artifacts = proof._output_paths(
        output_root, resume_baseline=True
    )

    assert baseline == immutable
    assert candidates.is_relative_to(output_root)
    assert final.is_relative_to(output_root)
    assert config.is_relative_to(output_root)
    assert artifacts.is_relative_to(output_root)
    assert artifacts != immutable
    assert set(immutable.iterdir()) == {
        immutable / "canonical",
        immutable / "economics-2014-canonical",
    }
    for unsafe in (immutable, immutable / "child", immutable.parent):
        with pytest.raises(RuntimeError, match="overlaps immutable baseline"):
            proof._output_paths(unsafe, resume_baseline=True)

    identifiers = iter([type("Id", (), {"hex": "one"})(), type("Id", (), {"hex": "two"})()])
    monkeypatch.setattr(proof, "ROOT", tmp_path / "stage/track1-v1")
    monkeypatch.setattr(proof, "uuid4", lambda: next(identifiers))
    automatic_1 = proof._output_paths(None, resume_baseline=True)
    automatic_2 = proof._output_paths(None, resume_baseline=True)
    assert automatic_1[0] == automatic_2[0] == immutable
    assert all(path.is_relative_to(tmp_path / "timesoil-track1-one") for path in automatic_1[1:])
    assert all(path.is_relative_to(tmp_path / "timesoil-track1-two") for path in automatic_2[1:])
    assert automatic_1[1:] != automatic_2[1:]
    assert not (tmp_path / "timesoil-track1-one").exists()
    assert not (tmp_path / "timesoil-track1-two").exists()

    controller_tree = proof.ROOT / "scripts"
    for unsafe in (controller_tree, controller_tree / "child", proof.ROOT):
        with pytest.raises(RuntimeError, match="overlaps staged controller tree"):
            proof._output_paths(unsafe, resume_baseline=True)
    model_source = tmp_path / "source/model-y.zip"
    model_source.parent.mkdir()
    model_source.write_bytes(b"source")
    monkeypatch.setattr(proof, "SOURCE", model_source)
    with pytest.raises(RuntimeError, match="overlaps Model Y source"):
        proof._output_paths(model_source, resume_baseline=True)
    with pytest.raises(RuntimeError, match="overlaps proof source tree"):
        proof._output_paths(SCRIPT.parents[1] / "unsafe-output", resume_baseline=True)
    target = tmp_path / "symlink-target"
    target.mkdir()
    linked = tmp_path / "symlink-output"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink path component"):
        proof._output_paths(linked / "child", resume_baseline=True)
    assert not (target / "child").exists()


def test_resume_rejects_manifest_mutation_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    manifest_path = baseline / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "deck": "MODEL_Y/MODEL_Y.DATA",
                "summary_contract": {"overlay": "MODEL_Y/_TIMESOIL_SUMMARY.INC"},
                "command": ["docker", "run"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    digest = sha256(manifest_path.read_bytes()).hexdigest()
    (baseline / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )

    class MutatingVerifier:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def _verify_opm_manifest(self, path: Path, *, baseline: bool) -> None:
            assert baseline is True
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(proof, "BASELINE", baseline)
    monkeypatch.setattr(proof, "OpmGdmBackend", MutatingVerifier)
    monkeypatch.setattr(proof, "verify_summary_extraction", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="changed during verification"):
        proof._resume_baseline(proof.OpmFlowRunner(docker_executable="docker"))


def _write_track1_bundle(root: Path) -> tuple[Path, dict[str, object]]:
    run_dir = root / "track1-input"
    run_dir.mkdir()
    contract: dict[str, object] = {
        "run_track1_mpc.py": {"path": "/staged/run_track1_mpc.py", "sha256": "1" * 64},
        "run_model_y_track1_proof.py": {
            "path": "/source/run_model_y_track1_proof.py",
            "sha256": "2" * 64,
        },
    }
    schedule = "DATES\n  1 JAN 2014 /\n/\n"
    run_id = "opm-full-replay-1234567890abcdef12345678"
    result = {
        "schema": "timesoil.aios.track1-mpc-result/v1",
        "run_id": "track1-input",
        "input_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "source_sha256": "5" * 64,
        "script_source_contract": contract,
        "case_id": "case",
        "schedule": {
            "sha256": sha256(schedule.encode()).hexdigest(),
            "text": schedule,
            "actions": [],
        },
        "evidence": {
            "backend_provenance": "OPM pinned",
            "trajectories": [
                {
                    "run_id": run_id,
                    "month": "2014-01-01",
                    "simulator": "OPM pinned",
                    "certified": True,
                    "chdd_complete": True,
                    "invariant_violations": [],
                    "actions": [],
                    "next_state": {"restart_ref": "pending"},
                }
            ],
            "step_economics": [
                {
                    "run_id": run_id,
                    "start_date": "2014-01-01",
                    "npv_million_rub": 1.0,
                    "complete": True,
                }
            ],
        },
    }
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    schedule_bytes = schedule.encode()
    (run_dir / "result.json").write_bytes(result_bytes)
    (run_dir / "wells_schedule.inc").write_bytes(schedule_bytes)
    manifest = {
        "schema": "timesoil.aios.track1-mpc-manifest/v1",
        "run_id": result["run_id"],
        "input_sha256": result["input_sha256"],
        "config_sha256": result["config_sha256"],
        "source_sha256": result["source_sha256"],
        "script_source_contract": contract,
        "backend_provenance": "OPM pinned",
        "schedule_sha256": sha256(schedule_bytes).hexdigest(),
        "trajectory_run_ids": [run_id],
        "artifacts": {
            "result": {
                "path": "result.json",
                "bytes": len(result_bytes),
                "sha256": sha256(result_bytes).hexdigest(),
            },
            "schedule": {
                "path": "wells_schedule.inc",
                "bytes": len(schedule_bytes),
                "sha256": sha256(schedule_bytes).hexdigest(),
            },
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (run_dir / "manifest.json").write_bytes(manifest_bytes)
    (run_dir / "manifest.sha256").write_text(
        f"{sha256(manifest_bytes).hexdigest()}  manifest.json\n", encoding="ascii"
    )
    return run_dir, contract


def _resign_track1_manifest(run_dir: Path) -> None:
    result_bytes = (run_dir / "result.json").read_bytes()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["artifacts"]["result"].update(
        bytes=len(result_bytes), sha256=sha256(result_bytes).hexdigest()
    )
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (run_dir / "manifest.json").write_bytes(manifest_bytes)
    (run_dir / "manifest.sha256").write_text(
        f"{sha256(manifest_bytes).hexdigest()}  manifest.json\n", encoding="ascii"
    )


def test_track1_bundle_verifier_rejects_hash_and_cross_field_tampering(tmp_path: Path) -> None:
    run_dir, contract = _write_track1_bundle(tmp_path)
    proof._verify_track1_run(run_dir, contract)

    sidecar = run_dir / "manifest.sha256"
    valid_sidecar = sidecar.read_bytes()
    sidecar.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="sidecar is invalid"):
        proof._verify_track1_run(run_dir, contract)
    sidecar.write_bytes(valid_sidecar)

    result_path = run_dir / "result.json"
    valid_result = result_path.read_bytes()
    result_path.write_bytes(valid_result + b" ")
    with pytest.raises(RuntimeError, match="result artifact hash mismatch"):
        proof._verify_track1_run(run_dir, contract)
    result_path.write_bytes(valid_result)

    result = json.loads(result_path.read_text())
    result["run_id"] = "tampered-but-resigned"
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    _resign_track1_manifest(run_dir)
    with pytest.raises(RuntimeError, match="run_id differs from manifest"):
        proof._verify_track1_run(run_dir, contract)


def test_track1_consumed_artifact_snapshots_reject_late_mutation(tmp_path: Path) -> None:
    run_dir, contract = _write_track1_bundle(tmp_path)
    _, _, _, snapshots = proof._verify_track1_run(run_dir, contract)

    for name in ("result.json", "wells_schedule.inc"):
        path = run_dir / name
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(RuntimeError, match="bytes or stat changed"):
            proof._assert_file_snapshot(path, snapshots[path], name)
        path.write_bytes(original)
        _, _, _, snapshots = proof._verify_track1_run(run_dir, contract)


def _write_selected_lineage(root: Path) -> tuple[Path, str, str, bytes]:
    candidates = root / "candidates"
    run_id = "opm-full-replay-1234567890abcdef12345678"
    selected = candidates / run_id
    canonical = selected / "canonical"
    canonical.mkdir(parents=True)
    chdd_bytes = b"DATA,CHDD\n2014-01-01,1.0\n"
    (canonical / "chdd.csv").write_bytes(chdd_bytes)
    lineage = {
        "schema": "timesoil.aios.track1-opm-lineage/v1",
        "status": "certified",
        "run_id": run_id,
        "artifacts": [
            {
                "purpose": "canonical_chdd",
                "path": "canonical/chdd.csv",
                "sha256": sha256(chdd_bytes).hexdigest(),
            }
        ],
    }
    lineage_path = selected / "lineage.json"
    lineage_bytes = (json.dumps(lineage, sort_keys=True) + "\n").encode()
    lineage_path.write_bytes(lineage_bytes)
    lineage_sha256 = sha256(lineage_bytes).hexdigest()
    lineage_path.with_suffix(".sha256").write_text(
        f"{lineage_sha256}  lineage.json\n", encoding="ascii"
    )
    return candidates, run_id, f"{lineage_path}#sha256={lineage_sha256}", chdd_bytes


def _signed_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    digest = sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return f"{path.resolve()}#sha256={digest}"


def _opm_manifest(run: Path, source_sha256: str) -> str:
    output = run / "output"
    output.mkdir(parents=True)
    primary = output / "MODEL.SMSPEC"
    companion = output / "MODEL.UNSMRY"
    primary.write_bytes(b"smspec")
    companion.write_bytes(b"unsmry")
    artifacts = [
        {
            "path": path.relative_to(run).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in (primary, companion)
    ]
    return _signed_json(
        run / "manifest.json",
        {
            "schema": "timesoil.aios.opm-run/v1",
            "status": "success",
            "returncode": 0,
            "source_sha256": source_sha256,
            "image_reference": OPM_IMAGE,
            "image_digest": OPM_IMAGE_DIGEST,
            "artifacts": artifacts,
        },
    )


def _write_full_selected_lineage(
    root: Path,
) -> tuple[Path, Path, Case, dict[str, object], Path, str]:
    source = root / "source"
    source.mkdir()
    (source / "MODEL.DATA").write_text(
        "RUNSPEC\nMETRIC\nSUMMARY\nSCHEDULE\nINCLUDE\n 'schedule.inc' /\nEND\n",
        encoding="utf-8",
    )
    (source / "schedule.inc").write_text(
        "DATES\n 1 JAN 2014 /\n/\nDATES\n 1 FEB 2014 /\n/\nEND\n",
        encoding="utf-8",
    )
    source_sha256 = _source_digest(source)
    baseline_ref = _opm_manifest(root / "prior/baseline", source_sha256)
    candidates = root / "candidates"
    run_id = "opm-full-replay-1234567890abcdef12345678"
    selected = candidates / run_id
    selected_manifest_ref = _opm_manifest(selected, source_sha256)
    selected_manifest = Path(selected_manifest_ref.split("#sha256=", 1)[0])
    canonical = selected / "canonical"
    canonical.mkdir()
    chdd = canonical / "chdd.csv"
    chdd.write_bytes(b"DATA,CHDD\n2014-01-01,1.0\n")
    case = proof.Case(
        "case",
        date(2014, 1, 1),
        date(2014, 1, 1),
        date(2014, 1, 1),
        ("P1",),
        ("I1",),
    )
    actions = (
        proof.ControlAction(
            case.start,
            "I1",
            proof.WellRole.INJECTOR,
            proof.WellStatus.OPEN,
            proof.ControlTarget.WATER_INJECTION_RATE,
            100.0,
        ),
        proof.ControlAction(
            case.start,
            "P1",
            proof.WellRole.PRODUCER,
            proof.WellStatus.OPEN,
            proof.ControlTarget.LIQUID_RATE,
            75.0,
        ),
    )
    pending_state = proof.State(
        case.case_id,
        date(2014, 2, 1),
        "pending",
        (
            proof.WellState("I1", proof.WellRole.INJECTOR, True, 0, 0, 100, 170),
            proof.WellState("P1", proof.WellRole.PRODUCER, True, 20, 75, 0, 150),
        ),
    )
    verifier = proof.OpmGdmBackend(
        proof.OpmFlowRunner(),
        source,
        runs_dir=candidates,
        deck="MODEL.DATA",
        schedule_include="schedule.inc",
        normalize_model_y=True,
        parsing_strictness="low",
        source_model="Model Y",
    )
    lineage_ref = _signed_json(
        selected / "lineage.json",
        {
            "schema": verifier._LINEAGE_SCHEMA,
            "status": "certified",
            "provenance": {
                "mode": "full-replay",
                "simulator": verifier.runner.get_provenance(),
                "binary_restart": False,
            },
            "run_id": run_id,
            "case_id": case.case_id,
            "source_sha256": source_sha256,
            "prior_restart_ref": baseline_ref,
            "step_actions": [verifier._action_value(action) for action in actions],
            "accepted_actions": [verifier._action_value(action) for action in actions],
            "next_state": verifier._state_value(pending_state),
            "artifacts": [
                {
                    "purpose": "opm_run_manifest",
                    "path": "manifest.json",
                    "sha256": sha256(selected_manifest.read_bytes()).hexdigest(),
                },
                {
                    "purpose": "canonical_chdd",
                    "path": "canonical/chdd.csv",
                    "sha256": sha256(chdd.read_bytes()).hexdigest(),
                },
            ],
        },
    )
    state = proof.State(
        pending_state.case_id,
        pending_state.month,
        lineage_ref,
        pending_state.wells,
    )
    trajectory: dict[str, object] = {
        "run_id": run_id,
        "month": case.start.isoformat(),
        "simulator": verifier.get_provenance(),
        "actions": [verifier._action_value(action) for action in actions],
        "next_state": proof._state_json(state),
    }
    return source, candidates, case, trajectory, chdd, baseline_ref


def test_selected_lineage_is_authenticated_before_chdd_copy(tmp_path: Path) -> None:
    candidates, run_id, restart_ref, chdd_bytes = _write_selected_lineage(tmp_path)
    lineage_path, lineage_sha256, verified_chdd, snapshots = proof._verified_selected_lineage(
        candidates, run_id, restart_ref
    )
    assert lineage_path.name == "lineage.json"
    assert restart_ref.endswith(lineage_sha256)
    assert verified_chdd == chdd_bytes

    chdd_path = candidates / run_id / "canonical/chdd.csv"
    chdd_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bytes or stat changed"):
        proof._assert_file_snapshot(chdd_path, snapshots[chdd_path], "selected CHDD")
    with pytest.raises(RuntimeError, match="canonical CHDD SHA-256 mismatch"):
        proof._verified_selected_lineage(candidates, run_id, restart_ref)

    chdd_path.write_bytes(chdd_bytes)
    lineage_path.write_bytes(lineage_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="lineage SHA-256 mismatch"):
        proof._verified_selected_lineage(candidates, run_id, restart_ref)

    with pytest.raises(RuntimeError, match="not SHA-256 authenticated"):
        proof._verified_selected_lineage(candidates, run_id, str(lineage_path))


def test_selected_history_requires_full_authenticated_opm_lineage(
    tmp_path: Path,
) -> None:
    source, candidates, case, trajectory, chdd, baseline_ref = (
        _write_full_selected_lineage(tmp_path)
    )

    state, history = proof._certify_selected_history(
        proof.OpmFlowRunner(),
        case,
        candidates,
        [trajectory],
        trajectory["actions"],
        baseline_ref,
        source=source,
        deck="MODEL.DATA",
        schedule_relative_path="schedule.inc",
    )
    next_state = trajectory["next_state"]
    assert isinstance(next_state, dict)
    assert state.restart_ref == next_state["restart_ref"]
    assert len(history) == 2

    six_month_case = proof.Case(
        case.case_id,
        case.start,
        date(2014, 6, 1),
        case.economics_start,
        case.producers,
        case.injectors,
    )
    with pytest.raises(RuntimeError, match="trajectory count differs"):
        proof._certify_selected_history(
            proof.OpmFlowRunner(),
            six_month_case,
            candidates,
            [trajectory],
            trajectory["actions"],
            baseline_ref,
            source=source,
            deck="MODEL.DATA",
            schedule_relative_path="schedule.inc",
        )

    run_id = str(trajectory["run_id"])
    minimal_ref = _signed_json(
        candidates / run_id / "lineage.json",
        {
            "schema": "timesoil.aios.track1-opm-lineage/v1",
            "status": "certified",
            "run_id": run_id,
            "artifacts": [
                {
                    "purpose": "canonical_chdd",
                    "path": "canonical/chdd.csv",
                    "sha256": sha256(chdd.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    next_state["restart_ref"] = minimal_ref
    with pytest.raises(RuntimeError, match="identity or provenance is invalid"):
        proof._certify_selected_history(
            proof.OpmFlowRunner(),
            case,
            candidates,
            [trajectory],
            trajectory["actions"],
            baseline_ref,
            source=source,
            deck="MODEL.DATA",
            schedule_relative_path="schedule.inc",
        )
