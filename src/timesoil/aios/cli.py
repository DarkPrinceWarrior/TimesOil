"""Minimal operator CLI for TimesOil AIOS."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import csv
from importlib import import_module
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, TextIO
from uuid import uuid4

from .agents import AgentWorkflow
from .api import AgentExperimentRequest, capabilities, get_runs_dir, run_agent_experiment
from .economics import CHDD_FIELDS, CHDDEconomicsAdapter
from .llm import LLMConfig, TatneftLLMClient
from .opm import OPM_IMAGE, OpmFlowRunner
from .tools import GROUNDED_ROLE_TOOLS, build_grounded_tool_registry
from .workflow import CycleRequest, CycleResult, FullCycleWorkflow


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class CLIError(RuntimeError):
    """Safe operator-facing error; secrets and model reasoning are omitted."""


def _json(value: Mapping[str, Any], *, stream: TextIO | None = None) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream or sys.stdout)


def _run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "run-id must contain only letters, digits, ._- and be <=64 chars"
        )
    return value


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return port


def _year(value: str) -> int:
    year = int(value)
    if not 1900 <= year <= 9999:
        raise argparse.ArgumentTypeError("start-year must be a four-digit year")
    return year


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if not 0 < timeout <= 7 * 24 * 3600:
        raise argparse.ArgumentTypeError("timeout must be in (0, 604800] seconds")
    return timeout


def _destination(runs_dir: Path | None, run_id: str) -> Path:
    configured = runs_dir if runs_dir is not None else get_runs_dir()
    raw_root = configured.expanduser()
    if raw_root.is_symlink():
        raise CLIError("runs directory cannot be a symbolic link")
    root = raw_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise CLIError("runs directory is not a directory")
    destination = (root / run_id).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        raise CLIError("run-id escapes the runs directory") from None
    if destination.exists():
        raise CLIError(f"run already exists: {run_id}")
    return destination


def _read_json_object(source: str) -> dict[str, Any]:
    try:
        if source == "-":
            value = json.load(sys.stdin)
        else:
            path = Path(source).expanduser()
            if path.is_symlink() or not path.is_file():
                raise CLIError("context must be a regular JSON file or '-' for stdin")
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise CLIError("context is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CLIError("agent context must be a JSON object")
    return value


def _read_chdd_csv(source: str) -> list[dict[str, str]]:
    if source == "-":
        context = nullcontext(sys.stdin)
    else:
        path = Path(source).expanduser()
        if path.is_symlink() or not path.is_file():
            raise CLIError("CHDD input must be a regular CSV file or '-' for stdin")
        context = path.open(encoding="utf-8-sig", newline="")
    with context as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CHDD_FIELDS:
            raise CLIError("CHDD CSV header must contain the canonical 14 fields in order")
        rows = [dict(row) for row in reader]
    if not rows:
        raise CLIError("CHDD CSV contains no records")
    return rows


async def _qwen_experiment(context: dict[str, Any]) -> dict[str, Any]:
    config = LLMConfig.from_env()
    async with TatneftLLMClient(config) as client:
        workflow = AgentWorkflow(
            client,
            build_grounded_tool_registry(),
            role_tools=GROUNDED_ROLE_TOOLS,
        )
        response = await run_agent_experiment(AgentExperimentRequest(context=context), workflow)
    return response.model_dump(mode="json")


async def _run_full_cycle(
    request: CycleRequest,
    destination: Path,
    *,
    run_id: str,
    timeout_seconds: float,
) -> CycleResult:
    config = LLMConfig.from_env()
    async with TatneftLLMClient(config) as client:
        workflow = FullCycleWorkflow(
            client,
            runner=OpmFlowRunner(timeout_seconds=timeout_seconds),
            economics=CHDDEconomicsAdapter.from_env(),
        )
        return await workflow.run(request, destination, run_id=run_id)


def _opm_runtime_ready() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        completed = subprocess.run(
            [docker, "image", "inspect", OPM_IMAGE],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _doctor(_: argparse.Namespace) -> int:
    report = capabilities().model_dump(mode="json")
    report["track1"]["runtime_ready"] = _opm_runtime_ready()
    _json(report)
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .api import app

    uvicorn = import_module("uvicorn")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _agent_experiment(args: argparse.Namespace) -> int:
    context = _read_json_object(args.context)
    try:
        result = asyncio.run(_qwen_experiment(context))
    except Exception as exc:
        raise CLIError("Qwen agent experiment failed") from exc
    _json(result)
    return 0


def _full_cycle(args: argparse.Namespace) -> int:
    raw = _read_json_object(args.request)
    base_dir = Path.cwd() if args.request == "-" else Path(args.request).expanduser().resolve().parent
    try:
        request = CycleRequest.from_mapping(raw, base_dir=base_dir)
        run_id = args.run_id or f"cycle-{request.request_sha256[:16]}"
        destination = _destination(args.runs_dir, run_id)
        result = asyncio.run(
            _run_full_cycle(
                request,
                destination,
                run_id=run_id,
                timeout_seconds=args.timeout,
            )
        )
    except Exception as exc:
        raise CLIError("full cycle failed; no unverified success receipt was emitted") from exc
    _json(
        {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "receipt": str(result.receipt_path),
            "receipt_sha256": result.receipt_sha256,
            "critic_approved": result.critic_approved,
        }
    )
    return 0 if result.critic_approved else 2


def _chdd(args: argparse.Namespace) -> int:
    rows = _read_chdd_csv(args.csv)
    run_id = args.run_id or uuid4().hex
    destination = _destination(args.runs_dir, run_id)
    try:
        result = CHDDEconomicsAdapter.from_env().calculate(
            rows, start_year=args.start_year, output_dir=destination
        )
    except Exception as exc:
        raise CLIError("official CHDD calculation failed") from exc
    _json(
        {
            "run_id": run_id,
            "total_chdd_m": result.total_chdd_m,
            "profitability_index": result.profitability_index,
            "start_date": result.start_date,
            "max_date": result.max_date,
            "diagnostics": dict(result.diagnostics),
            "output_dir": str(result.output_dir),
            "manifest": str(result.manifest_path),
        }
    )
    return 0


def _opm_baseline(args: argparse.Namespace) -> int:
    if args.normalize_model_y and not args.low:
        raise CLIError("--normalize-model-y requires explicit --low")
    run_id = args.run_id or uuid4().hex
    destination = _destination(args.runs_dir, run_id)
    runner = OpmFlowRunner(timeout_seconds=args.timeout)
    try:
        result = runner.run(
            args.source,
            destination,
            deck=args.deck,
            parsing_strictness="low" if args.low else "strict",
            normalize_model_y=args.normalize_model_y,
        )
        summary_report, summary_extraction = runner.extract_summary_report(
            result, result.run_dir / "summary-report.txt"
        )
    except Exception as exc:
        raise CLIError("OPM baseline failed; inspect its run logs and manifest") from exc
    _json(
        {
            "run_id": run_id,
            "run_dir": str(result.run_dir),
            "manifest": str(result.manifest_path),
            "manifest_sha256": result.manifest_sha256,
            "provenance": runner.get_provenance(),
            "warnings": list(result.warnings),
            "summary_report": str(summary_report),
            "summary_extraction_manifest": str(summary_extraction),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timesoil-aios",
        description="TimesOil AIOS operator CLI",
        epilog="Track 2 training stays in scripts/train_track2_surrogate.py.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="print secret-free component readiness JSON")
    doctor.set_defaults(handler=_doctor)

    serve = commands.add_parser("serve", help="serve the fixed FastAPI application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8000)
    serve.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info"),
        default="info",
    )
    serve.set_defaults(handler=_serve)

    agent = commands.add_parser("agent-experiment", help="run the fixed Qwen agent workflow")
    agent.add_argument("context", nargs="?", default="-", help="JSON object file or '-' for stdin")
    agent.set_defaults(handler=_agent_experiment)

    cycle = commands.add_parser(
        "full-cycle",
        help="run Qwen plan, pinned OPM, authenticated export, CHDD and immutable receipt",
    )
    cycle.add_argument("request", nargs="?", default="-", help="cycle request JSON or '-' for stdin")
    cycle.add_argument("--runs-dir", type=Path)
    cycle.add_argument("--run-id", type=_run_id)
    cycle.add_argument("--timeout", type=_positive_timeout, default=3600.0)
    cycle.set_defaults(handler=_full_cycle)

    chdd = commands.add_parser("chdd", help="run official CHDD for a canonical 14-field CSV")
    chdd.add_argument("csv", help="canonical CSV file or '-' for stdin")
    chdd.add_argument("--start-year", type=_year, required=True)
    chdd.add_argument("--runs-dir", type=Path)
    chdd.add_argument("--run-id", type=_run_id)
    chdd.set_defaults(handler=_chdd)

    opm = commands.add_parser("opm-baseline", help="run digest-pinned OPM Flow baseline")
    opm.add_argument("source", type=Path, help="case ZIP, directory, or standalone .DATA deck")
    opm.add_argument("--deck", help="relative .DATA path when source contains multiple decks")
    opm.add_argument("--runs-dir", type=Path)
    opm.add_argument("--run-id", type=_run_id)
    opm.add_argument("--timeout", type=_positive_timeout, default=3600.0)
    opm.add_argument("--low", "--allow-low-parsing", action="store_true")
    opm.add_argument("--normalize-model-y", action="store_true")
    opm.set_defaults(handler=_opm_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except CLIError as exc:
        _json({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 1
    except Exception as exc:
        _json({"ok": False, "error": type(exc).__name__}, stream=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
