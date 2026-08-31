from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from timesoil.aios.track2 import CANONICAL_COLUMNS, load_trajectory_dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_track2_scenarios.py"
SPEC = importlib.util.spec_from_file_location("run_track2_scenarios", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _fixture(
    root: Path, scenario_ids: tuple[str, ...] = MODULE._EXPECTED_SCENARIO_IDS
) -> tuple[Path, Path, dict[str, bytes]]:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    source.mkdir()
    original = b"DATES\n 1 'JAN' 2026 /\n/\n"
    (source / "MODEL_Z.DATA").write_text(
        "RUNSPEC\nSCHEDULE\nINCLUDE\n 'schedule.inc' /\nEND\n",
        encoding="utf-8",
    )
    (source / "schedule.inc").write_bytes(original)

    bundle = root / "bundle"
    schedule_sha = _digest(original)
    modified_by_id: dict[str, bytes] = {}
    entries: list[dict[str, object]] = []
    for index, scenario_id in enumerate(scenario_ids):
        scenario = bundle / scenario_id
        scenario.mkdir(parents=True)
        mode = "identity" if scenario_id == "baseline" else "full"
        modified = (
            original
            if mode == "identity"
            else original + f"-- TIMESOIL AIOS OVERRIDE {scenario_id}\n".encode()
        )
        controls = f"-- canonical controls {scenario_id}\n".encode()
        modified_by_id[scenario_id] = modified
        (scenario / "schedule.inc").write_bytes(modified)
        (scenario / "wells_schedule.inc").write_bytes(controls)
        actions_sha = _digest(controls)
        parameters = {"index": index, "seed": 20260831}
        manifest = {
            "schema_version": 1,
            "scenario_id": scenario_id,
            "generator_parameters": parameters,
            "inputs": {
                "baseline_csv": {"name": "baseline.csv", "sha256": "b" * 64},
                "schedule_include": {
                    "name": "schedule.inc",
                    "sha256": schedule_sha,
                },
                "known_wells": ["1"],
            },
            "controls": {
                "action_count": 1,
                "actions_sha256": actions_sha,
                "months": ["2026-01-01"],
            },
            "artifacts": {
                "modified_schedule": {
                    "path": "schedule.inc",
                    "sha256": _digest(modified),
                    "size_bytes": len(modified),
                },
                "wells_schedule": {
                    "path": "wells_schedule.inc",
                    "sha256": actions_sha,
                    "size_bytes": len(controls),
                },
            },
            "overlay": {
                "source_sha256": schedule_sha,
                "controls_sha256": actions_sha,
                "output_sha256": _digest(modified),
                "mode": mode,
                "action_count": 1,
                "action_months": ["2026-01-01"],
                "truncated_after": None,
            },
        }
        manifest_bytes = _json(manifest)
        (scenario / "manifest.json").write_bytes(manifest_bytes)
        entries.append(
            {
                "scenario_id": scenario_id,
                "manifest": f"{scenario_id}/manifest.json",
                "manifest_sha256": _digest(manifest_bytes),
                "actions_sha256": actions_sha,
                "generator_parameters": parameters,
            }
        )
    index = {
        "schema_version": 1,
        "generator": "scripts/generate_track2_scenarios.py",
        "inputs": {
            "baseline_csv": {"name": "baseline.csv", "sha256": "b" * 64},
            "schedule_include": {
                "name": "schedule.inc",
                "sha256": schedule_sha,
            },
        },
        "scenario_count": len(entries),
        "scenarios": entries,
    }
    (bundle / "index.json").write_bytes(_json(index))
    return source, bundle, modified_by_id


class _FakeRunner:
    events: list[str] = []
    expected_schedules: dict[str, bytes] = {}

    def __init__(self, *, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def prepare(
        self, source: Path, run_dir: Path, *, deck: Path | None = None
    ) -> SimpleNamespace:
        self.events.append("prepare")
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        shutil.copytree(source, input_dir)
        output_dir.mkdir()
        return SimpleNamespace(
            source_sha256=MODULE._source_digest(source),
            input_dir=input_dir,
            output_dir=output_dir,
            run_dir=run_dir,
            deck_path=input_dir / "MODEL_Z.DATA",
            unit_system="METRIC",
        )

    def _run_prepared(
        self, prepared: SimpleNamespace, *, parsing_strictness: str
    ) -> SimpleNamespace:
        self.events.append("run")
        assert parsing_strictness == "strict"
        scenario_id = prepared.run_dir.name
        assert (
            prepared.input_dir / "schedule.inc"
        ).read_bytes() == self.expected_schedules[scenario_id]
        binding = json.loads((prepared.run_dir / "scenario-binding.json").read_text())
        assert binding["scenario_id"] == scenario_id
        manifest = prepared.run_dir / "manifest.json"
        manifest.write_text('{"status":"success"}\n', encoding="utf-8")
        return SimpleNamespace(
            run_dir=prepared.run_dir,
            deck_path=prepared.deck_path,
            manifest_path=manifest,
            manifest_sha256=MODULE._sha256_file(manifest),
        )

    def extract_summary_report(
        self, result: SimpleNamespace, report_path: Path
    ) -> tuple[Path, Path]:
        self.events.append("extract")
        report_path.write_text("summary\n", encoding="utf-8")
        extraction = result.run_dir / "summary-extraction.json"
        extraction.write_text('{"verified":true}\n', encoding="utf-8")
        return report_path, extraction


def _export(
    report: Path,
    chdd: Path,
    track2: Path,
    manifest: Path,
    **kwargs: object,
) -> dict[str, object]:
    _FakeRunner.events.append("export")
    scenario_id = str(kwargs["scenario_id"])
    assert kwargs["source_model"] == "model_z_opm"
    chdd.parent.mkdir(parents=True)
    chdd.write_text("chdd\n", encoding="utf-8")
    track2.parent.mkdir(parents=True, exist_ok=True)
    with track2.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        for month in ("2026-01-01", "2026-02-01"):
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "source_model": "model_z_opm",
                    "date": month,
                    "well": "1",
                    "oil_tpd": 10.0,
                    "liquid_tpd": 12.0,
                    "pressure_bar": 200.0,
                    "control_value": 15.0,
                    "control_target": "ORAT",
                    "status": 1,
                }
            )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(
        _json(
            {
                "schema_version": 1,
                "generator": "timesoil.aios.opm_chdd",
                "scenario": {
                    "scenario_id": scenario_id,
                    "source_model": "model_z_opm",
                },
                "outputs": {"track2_csv": {"name": track2.name}},
            }
        )
    )
    return {"complete": True}


def _args(source: Path, bundle: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source=source,
        scenario_bundle=bundle,
        output_dir=output,
        source_sha256=MODULE._source_digest(source),
        scenario_index_sha256=MODULE._sha256_file(bundle / "index.json"),
        baseline_chdd_sha256=_digest(b"chdd\n"),
        schedule_relative_path="schedule.inc",
        deck=None,
        timeout_seconds=60.0,
        parsing_strictness="strict",
    )


class Track2ScenarioRunnerTest(unittest.TestCase):
    def test_ten_scenario_batch_emits_versioned_conformal_training_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario_ids = MODULE._EXPECTED_SCENARIO_SETS[10]
            source, bundle, modified = _fixture(root, scenario_ids)
            output = root / "runs"
            _FakeRunner.events = []
            _FakeRunner.expected_schedules = modified
            with (
                patch.object(MODULE, "OpmFlowRunner", _FakeRunner),
                patch.object(MODULE, "export_opm_chdd", _export),
            ):
                batch_manifest = MODULE._run_batch(_args(source, bundle, output))

            receipt = json.loads(batch_manifest.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], "timesoil.aios.track2-scenario-run/v2")
            self.assertEqual(receipt["scenario_count"], 10)
            self.assertEqual(
                receipt["training"]["argv"][-2:], ["--conformal-level", "0.9"]
            )

    def test_read_regular_rejects_component_swap_to_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root / "component"
            component.mkdir()
            (component / "source.py").write_bytes(b"trusted source\n")
            attacker = root / "attacker"
            attacker.mkdir()
            (attacker / "source.py").write_bytes(b"attacker source\n")
            moved = root / "moved-component"
            real_open = MODULE.os.open
            swapped = False

            def swapping_open(
                path: str | bytes,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal swapped
                if path == component.name and dir_fd is not None and not swapped:
                    component.rename(moved)
                    component.symlink_to(attacker, target_is_directory=True)
                    swapped = True
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with patch.object(MODULE.os, "open", side_effect=swapping_open):
                with self.assertRaises(OSError):
                    MODULE._read_regular(
                        component / "source.py", "source", limit=1024
                    )

            self.assertTrue(swapped)

    def test_torn_source_read_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, modified = _fixture(root)
            output = root / "runs"
            runner_source = root / "run_track2_scenarios.py"
            runner_source.write_bytes(b"A" * (2 * 1024**2))
            frozen = runner_source.stat()
            target = (frozen.st_dev, frozen.st_ino)
            real_fstat = MODULE.os.fstat
            real_read = MODULE.os.read
            mutated = False

            def frozen_fstat(descriptor: int) -> object:
                result = real_fstat(descriptor)
                return frozen if (result.st_dev, result.st_ino) == target else result

            def mutating_read(descriptor: int, size: int) -> bytes:
                nonlocal mutated
                chunk = real_read(descriptor, size)
                result = real_fstat(descriptor)
                if not mutated and (result.st_dev, result.st_ino) == target and chunk:
                    runner_source.write_bytes(b"B" * (2 * 1024**2))
                    mutated = True
                return chunk

            _FakeRunner.events = []
            _FakeRunner.expected_schedules = modified
            with (
                patch.object(MODULE, "__file__", str(runner_source)),
                patch.object(MODULE.os, "fstat", side_effect=frozen_fstat),
                patch.object(MODULE.os, "read", side_effect=mutating_read),
                patch.object(MODULE, "OpmFlowRunner", _FakeRunner),
                patch.object(MODULE, "export_opm_chdd", _export),
            ):
                with self.assertRaisesRegex(ValueError, "changed while it was read"):
                    MODULE._run_batch(_args(source, bundle, output))

            self.assertTrue(mutated)
            self.assertFalse(output.exists())
            self.assertFalse((output / "manifest.json").exists())

    def test_verified_bundle_runs_existing_pipeline_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario_ids = MODULE._EXPECTED_SCENARIO_IDS
            source, bundle, modified = _fixture(root, scenario_ids)
            output = root / "runs"
            _FakeRunner.events = []
            _FakeRunner.expected_schedules = modified
            with (
                patch.object(MODULE, "OpmFlowRunner", _FakeRunner),
                patch.object(MODULE, "export_opm_chdd", _export),
            ):
                batch_manifest = MODULE._run_batch(_args(source, bundle, output))

            self.assertEqual(
                _FakeRunner.events,
                ["prepare", "run", "extract", "export"] * len(scenario_ids),
            )
            receipt = json.loads(batch_manifest.read_text(encoding="utf-8"))
            self.assertEqual(receipt["scenario_count"], len(scenario_ids))
            self.assertTrue(receipt["sequential"])
            expected_sources = (
                Path(MODULE.__file__).absolute(),
                Path(MODULE._opm_module.__file__).absolute(),
                Path(MODULE._opm_chdd_module.__file__).absolute(),
            )
            self.assertEqual(
                receipt["executed_sources"],
                [
                    {"path": str(path), "sha256": _digest(path.read_bytes())}
                    for path in expected_sources
                ],
            )
            self.assertEqual(
                [item["scenario_id"] for item in receipt["scenarios"]],
                list(scenario_ids),
            )
            receipt_by_id = {
                item["scenario_id"]: item for item in receipt["scenarios"]
            }
            for scenario_id in scenario_ids:
                self.assertEqual(
                    receipt_by_id[scenario_id]["dataset"],
                    f"dataset/{scenario_id}.csv",
                )
                self.assertEqual(
                    receipt_by_id[scenario_id]["export_manifest"],
                    f"manifests/{scenario_id}.json",
                )
                self.assertEqual(
                    receipt_by_id[scenario_id]["canonical_chdd"],
                    f"{scenario_id}/canonical/chdd.csv",
                )
                self.assertEqual(
                    (output / scenario_id / "input/schedule.inc").read_bytes(),
                    modified[scenario_id],
                )
                self.assertTrue((output / "dataset" / f"{scenario_id}.csv").is_file())
                export_manifest = output / "manifests" / f"{scenario_id}.json"
                self.assertTrue(export_manifest.is_file())
                self.assertEqual(
                    json.loads(export_manifest.read_text())["outputs"]["track2_csv"][
                        "name"
                    ],
                    f"{scenario_id}.csv",
                )
            trajectories = load_trajectory_dataset(output / "dataset")
            self.assertEqual(
                [trajectory.scenario_id for trajectory in trajectories],
                list(scenario_ids),
            )
            self.assertEqual(receipt["training"]["dataset"], "dataset")
            self.assertEqual(receipt["training"]["manifests"], "manifests")
            self.assertEqual(
                receipt["training"]["argv"][-10:],
                [
                    "--test-fraction",
                    "0.25",
                    "--ensemble-size",
                    "5",
                    "--n-estimators",
                    "160",
                    "--horizon",
                    "6",
                    "--seed",
                    "20260831",
                ],
            )

    def test_incomplete_v2_scenario_set_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, _ = _fixture(root, ("baseline", "perturbation-001"))
            output = root / "runs"
            _FakeRunner.events = []

            with self.assertRaisesRegex(ValueError, "Track 2 v2 scenario set"):
                MODULE._run_batch(_args(source, bundle, output))

            self.assertFalse(output.exists())
            self.assertEqual(_FakeRunner.events, [])

    def test_tampered_artifact_fails_before_output_or_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, _ = _fixture(root)
            (bundle / "baseline/schedule.inc").write_bytes(b"tampered")
            output = root / "runs"
            _FakeRunner.events = []
            with patch.object(MODULE, "OpmFlowRunner", _FakeRunner):
                with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                    MODULE._run_batch(_args(source, bundle, output))
            self.assertFalse(output.exists())
            self.assertEqual(_FakeRunner.events, [])

    def test_unauthenticated_scenario_index_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, _ = _fixture(root)
            args = _args(source, bundle, root / "runs")
            args.scenario_index_sha256 = "0" * 64
            _FakeRunner.events = []

            with self.assertRaisesRegex(ValueError, "authenticated input"):
                MODULE._run_batch(args)

            self.assertFalse(args.output_dir.exists())
            self.assertEqual(_FakeRunner.events, [])

    def test_identity_baseline_must_match_authenticated_chdd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, modified = _fixture(root)
            output = root / "runs"
            args = _args(source, bundle, output)
            args.baseline_chdd_sha256 = "0" * 64
            _FakeRunner.events = []
            _FakeRunner.expected_schedules = modified
            with (
                patch.object(MODULE, "OpmFlowRunner", _FakeRunner),
                patch.object(MODULE, "export_opm_chdd", _export),
            ):
                with self.assertRaisesRegex(RuntimeError, "identity baseline CHDD"):
                    MODULE._run_batch(args)

            self.assertEqual(_FakeRunner.events, ["prepare", "run", "extract", "export"])
            self.assertFalse((output / "manifest.json").exists())

    def test_source_mutation_rejects_terminal_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, modified = _fixture(root)
            output = root / "runs"
            runner_source = root / "run_track2_scenarios.py"
            runner_source.write_bytes(b"original source\n")
            _FakeRunner.events = []
            _FakeRunner.expected_schedules = modified

            def mutating_export(*args: object, **kwargs: object) -> None:
                _export(*args, **kwargs)
                runner_source.write_bytes(b"mutated source\n")

            with (
                patch.object(MODULE, "__file__", str(runner_source)),
                patch.object(MODULE, "OpmFlowRunner", _FakeRunner),
                patch.object(MODULE, "export_opm_chdd", mutating_export),
            ):
                with self.assertRaisesRegex(RuntimeError, "executed source changed"):
                    MODULE._run_batch(_args(source, bundle, output))

            self.assertFalse((output / "manifest.json").exists())

    def test_path_escape_symlink_and_overwrite_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, bundle, _ = _fixture(root)

            index_path = bundle / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["scenarios"][0]["manifest"] = "../manifest.json"
            index_path.write_bytes(_json(index))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                MODULE._run_batch(_args(source, bundle, root / "escape-runs"))

            shutil.rmtree(bundle)
            _, real_bundle, _ = _fixture(root / "second")
            linked = root / "linked-bundle"
            linked.symlink_to(real_bundle, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                MODULE._run_batch(_args(source, linked, root / "linked-runs"))

            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                MODULE._run_batch(_args(source, real_bundle, output))


if __name__ == "__main__":
    unittest.main()
