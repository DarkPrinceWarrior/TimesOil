from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from timesoil.aios.opm import (
    OPM_IMAGE,
    OPM_IMAGE_DIGEST,
    OPM_EXPORT_VECTORS,
    OpmError,
    OpmFlowRunner,
    OpmSummaryError,
    OpmTimeoutError,
    build_summary_overlay,
    _canonical_summary_selection,
    verify_summary_extraction,
)


_MINIMAL_DECK = """RUNSPEC
METRIC
SUMMARY
SCHEDULE
WELSPECS
 'P1' 'G' 1 1 /
/
"""

_CONNECTION_VECTORS = {
    "COFR", "CWFR", "COPR", "COPT", "CWPR", "CWPT",
    "COIT", "CWIR", "CWIT",
}


class OpmFlowRunnerTest(unittest.TestCase):
    def test_summary_selection_ignores_only_irrelevant_duplicates(self) -> None:
        available = [
            "1",
            "1",
            "TIME",
            *(
                f"{vector}:P1:1,1,1"
                if vector in _CONNECTION_VECTORS
                else f"{vector}:P1"
                for vector in OPM_EXPORT_VECTORS
            ),
            "GGOR:GROUP",
            "GGOR:GROUP",
        ]

        selected = _canonical_summary_selection(available)

        self.assertNotIn("1", selected)
        self.assertNotIn("GGOR:GROUP", selected)
        with self.assertRaisesRegex(OpmSummaryError, "duplicate canonical"):
            _canonical_summary_selection([*available, selected[-1]])

    def test_summary_overlay_requests_required_vectors_for_all_wells(self) -> None:
        overlay = build_summary_overlay(("B-1H", "A-2", "B-1H"))

        self.assertIn("DATE\n", overlay)
        vectors = (
            "WLPR",
            "WLPT",
            "WOPR",
            "WOPT",
            "WOIR",
            "WOIT",
            "WWPR",
            "WWPT",
            "WWIR",
            "WWIT",
            "WBHP",
            "WBP9",
            "WEFF",
        )
        for vector in vectors:
            self.assertIn(f"{vector}\n/\n", overlay)
        connection_records = " 'A-2' /\n 'B-1H' /\n/\n"
        for vector in _CONNECTION_VECTORS:
            self.assertIn(f"{vector}\n{connection_records}", overlay)
            self.assertNotIn(f"{vector}\n/\n", overlay)
        self.assertNotIn("WOMT", overlay)

    def test_prepare_discovers_welspecs_across_include_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "schedule").mkdir(parents=True)
            (source / "MODEL.DATA").write_text(
                "RUNSPEC\nMETRIC\nSUMMARY\nSCHEDULE\n"
                "INCLUDE\n 'schedule/wells.inc' /\n"
            )
            (source / "schedule" / "wells.inc").write_text(
                "WELSPECS\n 'B-1H' 'G' 1 1 /\n 'A-2' 'G' 2 2 /\n/\n"
            )

            prepared = OpmFlowRunner().prepare(source, root / "run")

            overlay = prepared.summary_overlay_path.read_text()
            self.assertIn("COFR\n 'A-2' /\n 'B-1H' /\n/\n", overlay)
            self.assertEqual(prepared.connection_wells, ("A-2", "B-1H"))

    def test_prepare_fails_closed_without_welspecs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MODEL.DATA"
            source.write_text("RUNSPEC\nMETRIC\nSUMMARY\nSCHEDULE\n")

            with self.assertRaisesRegex(OpmError, "at least one WELSPECS well"):
                OpmFlowRunner().prepare(source, root / "run")
            self.assertFalse((root / "run").exists())

    def test_prepare_rejects_zip_slip_without_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "case.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("MODEL.DATA", "RUNSPEC\n")
                zipped.writestr("../escaped", "owned")

            with self.assertRaisesRegex(OpmError, "unsafe ZIP member"):
                OpmFlowRunner().prepare(archive, root / "run")

            self.assertFalse((root / "escaped").exists())
            self.assertFalse((root / "run").exists())

    @patch("timesoil.aios.opm.subprocess.run")
    def test_run_is_pinned_logged_and_does_not_modify_source(self, mocked_run) -> None:
        mocked_run.return_value = subprocess.CompletedProcess([], 0, "flow-out\n", "flow-err\n")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MODEL.DATA"
            original = _MINIMAL_DECK.encode()
            source.write_bytes(original)

            result = OpmFlowRunner(timeout_seconds=12).run(source, root / "run")

            self.assertEqual(source.read_bytes(), original)
            self.assertIn(OPM_IMAGE, result.command)
            self.assertNotIn("--parsing-strictness=low", result.command)
            self.assertEqual(result.stdout_path.read_text(), "flow-out\n")
            self.assertEqual(result.stderr_path.read_text(), "flow-err\n")
            self.assertIn("_TIMESOIL_SUMMARY.INC", result.deck_path.read_text())
            overlay = result.summary_overlay_path.read_text()
            self.assertIn("DATE\n", overlay)
            self.assertIn("WBP9\n/", overlay)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["image_digest"], OPM_IMAGE_DIGEST)
            self.assertEqual(manifest["warnings"], [])
            summary = manifest["summary_contract"]
            self.assertEqual(summary["unit_system"], "METRIC")
            self.assertEqual(summary["vectors"]["WOPR"]["unit"], "SM3/DAY")
            self.assertEqual(summary["vectors"]["WOPR"]["chdd_field"], "WOMR")
            self.assertEqual(summary["vectors"]["WEFF"]["unit"], "dimensionless")
            self.assertEqual(summary["vectors"]["WEFF"]["chdd_field"], "WEFF")
            self.assertIn("signed net", summary["vectors"]["COFR"]["quantity"])
            self.assertEqual(
                summary["vectors"]["COPR"]["quantity"],
                "connection surface oil production rate",
            )
            self.assertIn("aggregate", summary["vectors"]["CWPT"]["transform"])
            self.assertEqual(summary["connection_wells"], ["P1"])
            self.assertFalse(summary["postprocessor_certified"])
            self.assertIn("WOMT", summary["unsupported_direct_vectors"])
            self.assertNotIn("WEFF", summary["unsupported_direct_vectors"])
            self.assertEqual(len(result.manifest_sha256), 64)
            kwargs = mocked_run.call_args.kwargs
            self.assertFalse(kwargs["shell"])
            self.assertEqual(kwargs["timeout"], 12)

    @patch("timesoil.aios.opm.subprocess.run")
    def test_low_parser_mode_records_a_generic_accurate_warning(self, mocked_run) -> None:
        mocked_run.return_value = subprocess.CompletedProcess(
            [], 0, "ok", "unsupported keyword warning"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MODEL.DATA"
            source.write_text(_MINIMAL_DECK)

            result = OpmFlowRunner().run(
                source, root / "run", parsing_strictness="low"
            )

            self.assertIn("--parsing-strictness=low", result.command)
            self.assertEqual(len(result.warnings), 1)
            manifest = json.loads(result.manifest_path.read_text())
            warning = manifest["warnings"][0]
            self.assertIn("--parsing-strictness=low explicitly enabled", warning)
            self.assertIn("stdout.log and stderr.log", warning)
            self.assertNotIn("PINCHREG", warning)
            self.assertNotIn("Model Z", warning)

    @patch("timesoil.aios.opm.subprocess.run")
    def test_timeout_is_logged_and_manifested(self, mocked_run) -> None:
        mocked_run.side_effect = [
            subprocess.TimeoutExpired(
                ["docker"], 3, output=b"partial-out", stderr=b"partial-err"
            ),
            subprocess.CompletedProcess([], 0, "container-id\n", ""),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MODEL.DATA"
            source.write_text(_MINIMAL_DECK)

            with self.assertRaises(OpmTimeoutError):
                OpmFlowRunner(timeout_seconds=3).run(source, root / "run")

            self.assertEqual((root / "run" / "stdout.log").read_text(), "partial-out")
            manifest = json.loads((root / "run" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "timeout")
            self.assertEqual(manifest["container"]["timeout_cleanup"], "removed")
            container_name = manifest["container"]["name"]
            run_command = mocked_run.call_args_list[0].args[0]
            cleanup_command = mocked_run.call_args_list[1].args[0]
            self.assertEqual(
                run_command[run_command.index("--name") + 1], container_name
            )
            self.assertEqual(
                cleanup_command,
                ["docker", "container", "rm", "--force", container_name],
            )
            self.assertFalse(mocked_run.call_args_list[1].kwargs["shell"])
            self.assertEqual(mocked_run.call_args_list[1].kwargs["timeout"], 30.0)

    @patch("timesoil.aios.opm.subprocess.run")
    def test_summary_uses_list_and_report_options_and_checks_errors(self, mocked_run) -> None:
        mocked_run.side_effect = [
            subprocess.CompletedProcess([], 0, "ok", ""),
            subprocess.CompletedProcess([], 0, "TIME FOPR\n", ""),
            subprocess.CompletedProcess([], 0, "TIME FOPR\n0 1\n", ""),
            subprocess.CompletedProcess([], 2, "", "missing vector"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MODEL.DATA"
            source.write_text(_MINIMAL_DECK)
            runner = OpmFlowRunner()
            result = runner.run(source, root / "run")
            (result.output_dir / "MODEL.SMSPEC").write_bytes(b"smspec")

            self.assertEqual(runner.list_summary_vectors(result), ("TIME", "FOPR"))
            self.assertIn("0 1", runner.extract_summary(result, ("FOPR",)))
            with self.assertRaisesRegex(OpmSummaryError, "exit code 2"):
                runner.extract_summary(result, ("BAD",))

            list_command = mocked_run.call_args_list[1].args[0]
            extract_command = mocked_run.call_args_list[2].args[0]
            self.assertLess(list_command.index("-l"), list_command.index("/output/MODEL.SMSPEC"))
            self.assertLess(
                extract_command.index("-r"),
                extract_command.index("/output/MODEL.SMSPEC"),
            )
            self.assertFalse(mocked_run.call_args_list[2].kwargs["shell"])

    @patch("timesoil.aios.opm.subprocess.run")
    def test_summary_report_records_raw_artifact_derivation(self, mocked_run) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MODEL.DATA"
            source.write_text(_MINIMAL_DECK)

            def execute(command, **_kwargs):
                if "flow" in command:
                    output = root / "run" / "output"
                    (output / "MODEL.SMSPEC").write_bytes(b"smspec")
                    (output / "MODEL.UNSMRY").write_bytes(b"unsmry")
                    return subprocess.CompletedProcess(command, 0, "ok", "")
                if "-l" in command:
                    vectors = [
                        "TIME",
                        "YEARS",
                        *(
                            f"{vector}:P1"
                            for vector in OPM_EXPORT_VECTORS
                            if vector not in _CONNECTION_VECTORS
                        ),
                        *(
                            f"{vector}:P1:1"
                            for vector in OPM_EXPORT_VECTORS
                            if vector in _CONNECTION_VECTORS
                        ),
                    ]
                    return subprocess.CompletedProcess(
                        command, 0, " ".join(vectors) + "\n", ""
                    )
                report = b"TIME WLPR:P1\n0 1\n"
                return subprocess.CompletedProcess(
                    command,
                    0,
                    report.decode() if _kwargs.get("text") else report,
                    "" if _kwargs.get("text") else b"",
                )

            mocked_run.side_effect = execute
            runner = OpmFlowRunner()
            result = runner.run(source, root / "run")

            report, extraction = runner.extract_summary_report(
                result, result.run_dir / "summary-report.txt"
            )

            self.assertEqual(report.read_text(), "TIME WLPR:P1\n0 1\n")
            proof = json.loads(extraction.read_text())
            self.assertEqual(
                proof["vector_selection"],
                {
                    "mode": "filtered-summary-list",
                    "available": proof["vector_selection"]["available"],
                    "available_sha256": proof["vector_selection"][
                        "available_sha256"
                    ],
                    "selected": proof["vector_selection"]["available"],
                    "required": list(OPM_EXPORT_VECTORS),
                },
            )
            self.assertEqual(
                [item["path"] for item in proof["raw_summary_artifacts"]],
                ["output/MODEL.SMSPEC", "output/MODEL.UNSMRY"],
            )
            self.assertEqual(
                verify_summary_extraction(report, extraction, result.manifest_path)[
                    "status"
                ],
                "success",
            )
            list_command = mocked_run.call_args_list[1].args[0]
            summary_command = mocked_run.call_args_list[2].args[0]
            replay_command = mocked_run.call_args_list[3].args[0]
            self.assertEqual(list_command[-3:], ["summary", "-l", "/output/MODEL.SMSPEC"])
            self.assertEqual(
                summary_command,
                proof["commands"]["report"],
            )
            self.assertEqual(replay_command, summary_command)
            self.assertEqual(
                summary_command[-(3 + len(proof["vector_selection"]["selected"])) :],
                [
                    "summary",
                    "-r",
                    "/output/MODEL.SMSPEC",
                    *proof["vector_selection"]["selected"],
                ],
            )
            self.assertFalse(mocked_run.call_args_list[2].kwargs["shell"])
            self.assertFalse(mocked_run.call_args_list[3].kwargs["shell"])
            self.assertEqual(mocked_run.call_args_list[3].kwargs["timeout"], 900.0)

if __name__ == "__main__":
    unittest.main()
