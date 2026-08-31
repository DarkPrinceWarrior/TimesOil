from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from timesoil.aios.agents import AgentRole, AgentState, RoleDecision, WorkflowError
from timesoil.aios import api as api_module
from timesoil.aios.api import app, get_agent_workflow, get_chdd_adapter, get_runs_dir
from timesoil.aios.economics import CHDD_FIELDS, EconomicResult


ROOT = Path(__file__).resolve().parents[1]


def _chdd_row() -> dict[str, Any]:
    return {
        "DATA": "2015-01-01",
        "well": "well-1",
        **{field: 1.0 for field in CHDD_FIELDS[2:]},
    }


def test_health_and_capabilities_do_not_expose_secret(monkeypatch) -> None:
    secret = "do-not-return-this-key"
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setenv("LLM_BASE_URL", "https://qwen.example/v1")

    with TestClient(app) as client:
        health = client.get("/health")
        capabilities = client.get("/v1/capabilities")

    assert health.json() == {"status": "ok"}
    payload = capabilities.json()
    assert payload["qwen"] == {
        "model": "qwen3.6-35b-a3b",
        "configured": True,
        "connectivity_verified": False,
    }
    assert set(payload) == {"qwen", "track2", "chdd"}
    assert payload["track2"] == {
        "component_available": True,
        "certified": False,
        "model_z_trained": False,
    }
    assert payload["chdd"]["component_available"] is True
    assert secret not in health.text + capabilities.text


def test_model_z_capability_requires_pinned_verified_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    expected_hash = "a" * 64
    calls: list[tuple[Path, str | None]] = []

    def load(directory: Path, *, expected_manifest_sha256: str | None = None):
        calls.append((directory, expected_manifest_sha256))
        return SimpleNamespace(
            training_metadata={
                "model_z_ready": True,
                "pipeline_proof_only": False,
                "source_models": ["model_z_opm"],
                "dataset_hash": "verified-extra-evidence",
            }
        )

    monkeypatch.setenv("MODEL_Z_SURROGATE_DIR", str(tmp_path))
    monkeypatch.setattr(api_module, "MODEL_Z_SURROGATE_MANIFEST_SHA256", expected_hash)
    monkeypatch.setattr(api_module.Track2Surrogate, "load", load)
    assert api_module._model_z_trained() is True
    assert calls == [(tmp_path, expected_hash)]


def test_bundled_model_z_artifact_matches_pin(monkeypatch) -> None:
    artifact = ROOT / "deliverables/track2_model_z/surrogate/model"
    manifest = artifact / "manifest.json"
    assert sha256(manifest.read_bytes()).hexdigest() == (
        api_module.MODEL_Z_SURROGATE_MANIFEST_SHA256
    )

    model = api_module.Track2Surrogate.load(
        artifact,
        expected_manifest_sha256=api_module.MODEL_Z_SURROGATE_MANIFEST_SHA256,
    )
    assert model.training_metadata["model_z_ready"] is True
    assert model.training_metadata["pipeline_proof_only"] is False
    assert model.training_metadata["source_models"] == ["model_z_opm"]

    monkeypatch.setenv("MODEL_Z_SURROGATE_DIR", str(artifact))
    assert api_module._model_z_trained() is True

    summary = json.loads(
        (ROOT / "deliverables/track2_model_z/model_z_v4_summary.json").read_text()
    )
    assert summary["surrogate"]["model_manifest_sha256"] == (
        api_module.MODEL_Z_SURROGATE_MANIFEST_SHA256
    )
    assert summary["final_replay"]["complete"] is True
    assert summary["final_replay"]["improvement_over_operational_baseline"] is False
    assert summary["final_replay"]["candidate_accepted"] is False
    assert summary["search"]["operational_decision"] == "retain_authenticated_baseline"
    assert summary["organizer_certified"] is False


def test_compose_secret_bootstrap_drops_privileges() -> None:
    compose = (ROOT / "compose.yaml").read_text()
    for expected in (
        'user: "0:0"',
        "- DAC_READ_SEARCH",
        "- SETGID",
        "- SETUID",
        "setpriv --reuid=10001 --regid=10001 --init-groups",
        "--bounding-set=-all --inh-caps=-all --ambient-caps=-all",
    ):
        assert expected in compose


def test_model_z_capability_fails_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODEL_Z_SURROGATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        api_module, "MODEL_Z_SURROGATE_MANIFEST_SHA256", "a" * 64
    )

    for value in (
        SimpleNamespace(
            training_metadata={
                "model_z_ready": True,
                "pipeline_proof_only": True,
                "source_models": ["model_z_opm"],
            }
        ),
        ValueError("corrupt artifact"),
    ):
        def load(*_args, **_kwargs):
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(api_module.Track2Surrogate, "load", load)
        assert api_module._model_z_trained() is False


def test_model_z_capability_revalidates_after_artifact_failure(
    monkeypatch, tmp_path: Path
) -> None:
    ready = SimpleNamespace(
        training_metadata={
            "model_z_ready": True,
            "pipeline_proof_only": False,
            "source_models": ["model_z_opm"],
        }
    )
    results = iter((ready, ValueError("artifact disappeared")))

    def load(*_args, **_kwargs):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setenv("MODEL_Z_SURROGATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        api_module, "MODEL_Z_SURROGATE_MANIFEST_SHA256", "a" * 64
    )
    monkeypatch.setattr(api_module.Track2Surrogate, "load", load)

    assert api_module._model_z_trained() is True
    assert api_module._model_z_trained() is False


def test_agent_experiment_uses_dependency_and_filters_internal_data() -> None:
    class FakeWorkflow:
        async def run(self, context: dict[str, Any]) -> AgentState:
            assert context == {"track": 2}
            return AgentState(
                run_id="fake-run",
                context=context,
                decisions=(
                    RoleDecision(
                        role=AgentRole.COORDINATOR,
                        summary="checked",
                        recommendation="continue",
                        evidence=("fixture",),
                        approved=True,
                    ),
                ),
            )

    app.dependency_overrides[get_agent_workflow] = lambda: FakeWorkflow()
    try:
        response = TestClient(app).post(
            "/v1/experiments/agents",
            json={"context": {"track": 2}},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "fake-run",
        "complete": False,
        "critic_approved": False,
        "decisions": [
            {
                "role": "coordinator",
                "summary": "checked",
                "recommendation": "continue",
                "evidence": ["fixture"],
                "approved": True,
                "tools": [],
            }
        ],
    }
    assert "reasoning" not in response.text


def test_agent_experiment_fails_closed_without_qwen_env(monkeypatch) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    response = TestClient(app).post(
        "/v1/experiments/agents",
        json={"context": {"track": 2}},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Qwen3.6 is not configured"}


def test_agent_experiment_returns_bounded_error() -> None:
    class RejectingWorkflow:
        async def run(self, _context: dict[str, Any]) -> AgentState:
            raise WorkflowError("internal detail must stay hidden")

    app.dependency_overrides[get_agent_workflow] = lambda: RejectingWorkflow()
    try:
        response = TestClient(app).post(
            "/v1/experiments/agents", json={"context": {"track": 2}}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {
        "detail": "agent context or response violated the bounded contract"
    }
    assert "internal detail" not in response.text


def test_chdd_uses_typed_records_and_server_runs_dir(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    class FakeCHDD:
        def calculate(
            self,
            records: list[dict[str, Any]],
            *,
            start_year: int,
            output_dir: Path,
        ) -> EconomicResult:
            calls.update(records=records, start_year=start_year, output_dir=output_dir)
            return EconomicResult(
                total_chdd_m=12.5,
                profitability_index=1.2,
                start_date="2015-01-01",
                max_date="2015-12-01",
                diagnostics={"rows": 1},
                output_dir=output_dir,
                manifest_path=output_dir / "manifest.json",
            )

    server_runs = tmp_path / "server-runs"
    app.dependency_overrides[get_chdd_adapter] = lambda: FakeCHDD()
    app.dependency_overrides[get_runs_dir] = lambda: server_runs
    try:
        response = TestClient(app).post(
            "/v1/economics/chdd",
            json={"records": [_chdd_row()], "start_year": 2015},
        )
        rejected_path = TestClient(app).post(
            "/v1/economics/chdd",
            json={
                "records": [_chdd_row()],
                "start_year": 2015,
                "output_dir": "client-choice",
            },
        )
        missing_year = TestClient(app).post(
            "/v1/economics/chdd",
            json={"records": [_chdd_row()]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_chdd_m"] == 12.5
    assert "output_dir" not in payload
    assert calls["start_year"] == 2015
    assert set(calls["records"][0]) == set(CHDD_FIELDS)
    assert Path(calls["output_dir"]).parent == server_runs
    assert Path(calls["output_dir"]).name == payload["run_id"]
    assert rejected_path.status_code == 422
    assert missing_year.status_code == 422


def test_chdd_rejects_incomplete_or_untyped_records(tmp_path: Path) -> None:
    app.dependency_overrides[get_chdd_adapter] = lambda: object()
    app.dependency_overrides[get_runs_dir] = lambda: tmp_path
    try:
        incomplete = _chdd_row()
        incomplete.pop("WWIT_Diff")
        missing_field = TestClient(app).post(
            "/v1/economics/chdd",
            json={"records": [incomplete], "start_year": 2015},
        )
        untyped = _chdd_row()
        untyped["WLPT"] = "1.0"
        wrong_type = TestClient(app).post(
            "/v1/economics/chdd",
            json={"records": [untyped], "start_year": 2015},
        )
    finally:
        app.dependency_overrides.clear()

    assert missing_field.status_code == 422
    assert wrong_type.status_code == 422
