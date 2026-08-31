from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from timesoil.aios import cli
from timesoil.aios.economics import CHDD_FIELDS
from timesoil.aios.opm import OPM_IMAGE
from timesoil.aios.tools import GROUNDED_ROLE_TOOLS


def _csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CHDD_FIELDS))
        writer.writeheader()
        writer.writerow(
            {
                "DATA": "2014-01-01",
                "well": "P1",
                **{field: "1" for field in CHDD_FIELDS[2:]},
            }
        )


class CLITest(unittest.TestCase):
    def test_doctor_is_json_and_never_prints_secrets(self) -> None:
        secret = "api-key-must-not-leak"
        output = io.StringIO()
        with patch.dict("os.environ", {"LLM_API_KEY": secret}, clear=False), patch(
            "timesoil.aios.cli.shutil.which", return_value=None
        ), patch("sys.stdout", output):
            self.assertEqual(cli.main(["doctor"]), 0)

        report = json.loads(output.getvalue())
        self.assertNotIn(secret, output.getvalue())
        self.assertEqual(
            report["qwen"],
            {
                "model": "qwen3.6-35b-a3b",
                "configured": True,
                "connectivity_verified": False,
            },
        )
        self.assertIn("component_available", report["track1"])
        self.assertIn("certified", report["track1"])
        self.assertIn("model_z_trained", report["track2"])
        self.assertFalse(report["track1"]["runtime_ready"])

    def test_doctor_inspects_exact_pinned_image_without_shell(self) -> None:
        output = io.StringIO()
        completed = subprocess.CompletedProcess([], 0, "[]", "")
        with patch("timesoil.aios.cli.shutil.which", return_value="/usr/bin/docker"), patch(
            "timesoil.aios.cli.subprocess.run", return_value=completed
        ) as run, patch("sys.stdout", output):
            self.assertEqual(cli.main(["doctor"]), 0)

        self.assertTrue(json.loads(output.getvalue())["track1"]["runtime_ready"])
        self.assertEqual(
            run.call_args.args[0],
            ["/usr/bin/docker", "image", "inspect", OPM_IMAGE],
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

        self.assertFalse(json.loads(output.getvalue())["track1"]["runtime_ready"])

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
            "timesoil.aios.cli.TatneftLLMClient", return_value=client
        ), patch("timesoil.aios.cli.AgentWorkflow", return_value=workflow) as factory, patch(
            "timesoil.aios.cli.run_agent_experiment",
            AsyncMock(return_value=response),
        ):
            result = asyncio.run(cli._qwen_experiment({"case": "model-z"}))

        registry = factory.call_args.args[1]
        self.assertEqual(registry.names, frozenset(sum(GROUNDED_ROLE_TOOLS.values(), ())))
        self.assertEqual(factory.call_args.kwargs["role_tools"], GROUNDED_ROLE_TOOLS)
        self.assertEqual(result, {"complete": True})

    def test_chdd_requires_canonical_csv_and_writes_below_runs_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            runs = root / "runs"
            _csv(source)
            adapter = Mock()
            adapter.calculate.return_value = SimpleNamespace(
                total_chdd_m=12.0,
                profitability_index=1.2,
                start_date="2014-01-01",
                max_date="2015-01-01",
                diagnostics={},
                output_dir=runs / "safe-run",
                manifest_path=runs / "safe-run" / "manifest.json",
            )
            output = io.StringIO()
            with patch(
                "timesoil.aios.cli.CHDDEconomicsAdapter.from_env", return_value=adapter
            ), patch("sys.stdout", output):
                code = cli.main(
                    [
                        "chdd",
                        str(source),
                        "--start-year",
                        "2014",
                        "--runs-dir",
                        str(runs),
                        "--run-id",
                        "safe-run",
                    ]
                )

            self.assertEqual(code, 0)
            records = adapter.calculate.call_args.args[0]
            self.assertEqual(tuple(records[0]), CHDD_FIELDS)
            destination = adapter.calculate.call_args.kwargs["output_dir"]
            self.assertEqual(destination, (runs / "safe-run").resolve())
            self.assertEqual(json.loads(output.getvalue())["run_id"], "safe-run")

    def test_chdd_rejects_noncanonical_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.csv"
            source.write_text("DATA,well\n2014-01-01,P1\n")
            error = io.StringIO()
            with patch("sys.stderr", error):
                code = cli.main(["chdd", str(source), "--start-year", "2014"])
            self.assertEqual(code, 1)
            self.assertIn("canonical 14 fields", error.getvalue())

    def test_opm_baseline_uses_safe_destination_and_explicit_low(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "case.zip"
            source.write_bytes(b"zip")
            runner = Mock()
            runner.get_provenance.return_value = "OPM pinned"
            runner.run.return_value = SimpleNamespace(
                run_dir=root / "runs" / "baseline",
                manifest_path=root / "runs" / "baseline" / "manifest.json",
                manifest_sha256="a" * 64,
                warnings=("PINCHREG",),
            )
            runner.extract_summary_report.return_value = (
                root / "runs" / "baseline" / "summary-report.txt",
                root / "runs" / "baseline" / "summary-extraction.json",
            )
            output = io.StringIO()
            with patch("timesoil.aios.cli.OpmFlowRunner", return_value=runner), patch(
                "sys.stdout", output
            ):
                code = cli.main(
                    [
                        "opm-baseline",
                        str(source),
                        "--runs-dir",
                        str(root / "runs"),
                        "--run-id",
                        "baseline",
                        "--low",
                        "--normalize-model-y",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(
                runner.run.call_args.kwargs["parsing_strictness"], "low"
            )
            self.assertTrue(runner.run.call_args.kwargs["normalize_model_y"])
            runner.extract_summary_report.assert_called_once()
            self.assertEqual(json.loads(output.getvalue())["warnings"], ["PINCHREG"])

    def test_model_y_normalization_requires_explicit_low(self) -> None:
        error = io.StringIO()
        with patch("sys.stderr", error):
            code = cli.main(["opm-baseline", "case.zip", "--normalize-model-y"])
        self.assertEqual(code, 1)
        self.assertIn("requires explicit --low", error.getvalue())

    def test_serve_invokes_fixed_app_without_subprocess(self) -> None:
        with patch("uvicorn.run") as run:
            self.assertEqual(cli.main(["serve", "--port", "9000"]), 0)
        self.assertEqual(run.call_args.kwargs["port"], 9000)
        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
