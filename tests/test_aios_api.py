from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from timesoil.aios.agents import AgentRole, AgentState, RoleDecision, WorkflowError
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
    monkeypatch.setenv("LLM_BASE_URL", "https://litellm.tatneft.guru/v1")

    with TestClient(app) as client:
        health = client.get("/health")
        capabilities = client.get("/v1/capabilities")

    assert health.json() == {"status": "ok"}
    payload = capabilities.json()
    assert payload["qwen"] == {
        "model": "qwen3.8-27b",
        "configured": True,
        "connectivity_verified": False,
    }
    assert set(payload) == {"qwen", "track2", "chdd"}
    assert payload["track2"] == {"component_available": True, "certified": False}
    assert payload["chdd"]["component_available"] is True
    assert secret not in health.text + capabilities.text


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
    assert response.json() == {"detail": "Qwen is not configured"}


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
