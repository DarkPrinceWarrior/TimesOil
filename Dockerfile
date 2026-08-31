FROM ghcr.io/astral-sh/uv:0.11.25@sha256:1e3808aa9023d0980e7c15b1fa7c1ac16ff35925780cf5c459858b2d693f01a9 AS uv

FROM python:3.13-slim-bookworm@sha256:c45a22ea000adfd9cda29364bbe7edd23001ce5cc2ad15857cfbf7766943b9ca

COPY --from=uv /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

RUN useradd --create-home --uid 10001 timesoil

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY docs/hackathon/chdd/CHDD_PYTHON/РАСЧЕТ_ЧДД.py \
    docs/hackathon/chdd/CHDD_PYTHON/chdd_model.py \
    docs/hackathon/chdd/CHDD_PYTHON/excel_io.py \
    ./docs/hackathon/chdd/CHDD_PYTHON/
COPY docs/hackathon/chdd/CHDD_PYTHON/input/Нормативы_ЧДД.xlsx \
    ./docs/hackathon/chdd/CHDD_PYTHON/input/Нормативы_ЧДД.xlsx

RUN uv sync --locked --no-dev --no-editable \
    && mkdir -p /app/runs \
    && chown timesoil:timesoil /app/runs

USER timesoil

EXPOSE 8000
VOLUME ["/app/runs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "timesoil.aios.api:app", "--host", "0.0.0.0", "--port", "8000"]
