from __future__ import annotations

from fastapi.testclient import TestClient

from timesoil.aios.api import app


def test_operator_page_exposes_accessible_same_origin_controls() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    assert '<html lang="ru">' in page
    assert '<label for="context">' in page
    assert 'aria-labelledby="result-title"' in page
    assert 'id="run" type="submit"' in page
    assert 'role="status" aria-live="polite"' in page
    assert 'requestJson("/health")' in page
    assert 'requestJson("/v1/capabilities")' in page
    assert 'requestJson("/v1/experiments/agents"' in page
    assert "http://" not in page
    assert "https://" not in page


def test_operator_page_renders_remote_data_as_text_and_states_certification_limits() -> None:
    response = TestClient(app).get("/")
    page = response.text

    assert "innerHTML" not in page
    assert ".textContent" in page
    assert ".replaceChildren()" in page
    assert "не сертифицирован" in page
    assert "Model Z" in page
    assert "это не сертификат результата" in page
    assert "LLM_API_KEY" not in page
    assert response.headers["cache-control"] == "no-store"
    assert "connect-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
