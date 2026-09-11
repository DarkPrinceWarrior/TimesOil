from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from timesoil.aios import cli
from timesoil.aios.opm import OPM_IMAGE
from timesoil.aios.tools import GROUNDED_ROLE_TOOLS


class CLITest(unittest.TestCase):
    def test_cli_exposes_only_the_submission_subcommands(self) -> None:
        commands = next(
            action.choices
            for action in cli.build_parser()._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            set(commands), {"doctor", "agent-experiment", "full-cycle"}
        )

    def test_doctor_is_json_and_never_prints_secrets(self) -> None:
        secret = "api-key-must-not-leak"
        output = io.StringIO()
        with patch.dict(
            "os.environ",
        {"LLM_API_KEY": secret, "LLM_BASE_URL": "https://litellm.tatneft.guru/v1"},
            clear=True,
        ), patch(
            "timesoil.aios.cli.shutil.which", return_value=None
        ), patch("sys.stdout", output):
            self.assertEqual(cli.main(["doctor"]), 0)

        report = json.loads(output.getvalue())
        self.assertNotIn(secret, output.getvalue())
        self.assertEqual(
            report["qwen"],
            {
        "model": "qwen3.8-27b",
                "configured": True,
                "connectivity_verified": False,
            },
        )
        self.assertEqual(set(report), {"qwen", "track2", "chdd"})
        self.assertFalse(report["track2"]["runtime_ready"])

    def test_doctor_inspects_exact_pinned_image_without_shell(self) -> None:
        output = io.StringIO()
        completed = subprocess.CompletedProcess([], 0, "[]", "")
        with patch("timesoil.aios.cli.shutil.which", return_value="docker-bin"), patch(
            "timesoil.aios.cli.subprocess.run", return_value=completed
        ) as run, patch("sys.stdout", output):
            self.assertEqual(cli.main(["doctor"]), 0)

        self.assertTrue(json.loads(output.getvalue())["track2"]["runtime_ready"])
        self.assertEqual(
            run.call_args.args[0],
            ["docker-bin", "image", "inspect", OPM_IMAGE],
        )
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["timeout"], 5.0)

    def test_doctor_fails_closed_when_image_inspect_times_out(self) -> None:
        output = io.StringIO()
        with patch("timesoil.aios.cli.shutil.which", return_value="docker"), patch(
            "timesoil.aios.cli.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["docker"], 5),
        ), patch("sys.stdout", output):
            self.assertEqual(cli.main(["doctor"]), 0)

        self.assertFalse(json.loads(output.getvalue())["track2"]["runtime_ready"])

    def test_agent_experiment_reads_stdin_and_outputs_no_reasoning(self) -> None:
        result = {
            "run_id": "run",
            "complete": True,
            "critic_approved": False,
            "decisions": [],
        }
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO('{"case":"model-z"}')), patch(
            "sys.stdout", output
        ), patch(
            "timesoil.aios.cli._qwen_experiment", AsyncMock(return_value=result)
        ) as experiment:
            self.assertEqual(cli.main(["agent-experiment"]), 0)

        experiment.assert_awaited_once_with({"case": "model-z"})
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertNotIn("reasoning", output.getvalue())
        self.assertNotIn("api_key", output.getvalue())

    def test_qwen_cli_uses_the_grounded_request_scoped_registry(self) -> None:
        client = Mock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        response = SimpleNamespace(model_dump=Mock(return_value={"complete": True}))
        workflow = Mock()

        with patch("timesoil.aios.cli.LLMConfig.from_env", return_value=Mock()), patch(
            "timesoil.aios.cli.ExternalQwenClient", return_value=client
        ), patch("timesoil.aios.cli.AgentWorkflow", return_value=workflow) as factory, patch(
            "timesoil.aios.cli.run_agent_experiment",
            AsyncMock(return_value=response),
        ):
            result = asyncio.run(cli._qwen_experiment({"case": "model-z"}))

        registry = factory.call_args.args[1]
        self.assertEqual(registry.names, frozenset(sum(GROUNDED_ROLE_TOOLS.values(), ())))
        self.assertEqual(factory.call_args.kwargs["role_tools"], GROUNDED_ROLE_TOOLS)
        self.assertEqual(result, {"complete": True})

    def test_full_cycle_uses_internal_production_factory(self) -> None:
        client = Mock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        request = Mock()
        destination = Path("full-cycle-test")
        expected = Mock()
        workflow = Mock()
        workflow.run = AsyncMock(return_value=expected)

        with patch("timesoil.aios.cli.LLMConfig.from_env", return_value=Mock()), patch(
            "timesoil.aios.cli.ExternalQwenClient", return_value=client
        ), patch.object(
            cli.FullCycleWorkflow, "_from_cli", return_value=workflow
        ) as factory:
            result = asyncio.run(
                cli._run_full_cycle(
                    request,
                    destination,
                    run_id="production-cycle",
                    timeout_seconds=321.0,
                )
            )

        factory.assert_called_once_with(client, timeout_seconds=321.0)
        workflow.run.assert_awaited_once_with(
            request, destination, run_id="production-cycle"
        )
        self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
