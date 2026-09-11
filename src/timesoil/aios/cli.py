"""Minimal operator CLI for TimesOil AIOS."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, TextIO

from .agents import AgentWorkflow
from .api import AgentExperimentRequest, capabilities, get_runs_dir, run_agent_experiment
from .llm import ExternalQwenClient, LLMConfig
from .opm import OPM_IMAGE
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


async def _qwen_experiment(context: dict[str, Any]) -> dict[str, Any]:
    config = LLMConfig.from_env()
    async with ExternalQwenClient(config) as client:
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
    async with ExternalQwenClient(config) as client:
        workflow = FullCycleWorkflow._from_cli(
            client, timeout_seconds=timeout_seconds
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
    report["track2"]["runtime_ready"] = _opm_runtime_ready()
    _json(report)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timesoil-aios",
        description="TimesOil AIOS operator CLI",
        epilog=(
            "Track 2 search stays in scripts/propose_track2_policies.py; sealing and the "
            "single final verification in scripts/track2_final_selection.py."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="print secret-free component readiness JSON")
    doctor.set_defaults(handler=_doctor)

    agent = commands.add_parser("agent-experiment", help="run the fixed Qwen agent workflow")
    agent.add_argument("context", nargs="?", default="-", help="JSON object file or '-' for stdin")
    agent.set_defaults(handler=_agent_experiment)

    cycle = commands.add_parser(
        "full-cycle",
        help="experimental fail-closed Qwen, OPM, export and CHDD cycle",
    )
    cycle.add_argument("request", nargs="?", default="-", help="cycle request JSON or '-' for stdin")
    cycle.add_argument("--runs-dir", type=Path)
    cycle.add_argument("--run-id", type=_run_id)
    cycle.add_argument("--timeout", type=_positive_timeout, default=3600.0)
    cycle.set_defaults(handler=_full_cycle)
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
