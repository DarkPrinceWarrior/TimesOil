from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Callable
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_track2_surrogate.py"
SPEC = importlib.util.spec_from_file_location("train_track2_surrogate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _batch_lineage() -> dict[str, object]:
    scenarios = [
        {
            "scenario_id": scenario_id,
            "actions_sha256": MODULE._EXPECTED_SCENARIO_ACTIONS_SHA256[scenario_id],
            "dataset_sha256": sha256(f"dataset:{scenario_id}".encode()).hexdigest(),
            "export_manifest_sha256": sha256(
                f"manifest:{scenario_id}".encode()
            ).hexdigest(),
        }
        for scenario_id in MODULE._EXPECTED_SCENARIO_IDS
    ]
    return {
        "schema": MODULE._TRAINING_LINEAGE_SCHEMA,
        "scenario_run_schema": MODULE._SCENARIO_RUN_SCHEMA,
        "batch_manifest_sha256": "b" * 64,
        "official_source_sha256": MODULE.MODEL_Z_SOURCE_SHA256,
        "scenario_index_sha256": MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
        "scenario_ids": list(MODULE._EXPECTED_SCENARIO_IDS),
        "scenarios": scenarios,
    }


class _FakeModel:
    def __init__(self, mutate: Callable[[], object] | None = None) -> None:
        self._mutate = mutate
        self.training_metadata: dict[str, object] = {}

    def save(self, directory: Path) -> dict[str, str]:
        directory.mkdir(parents=True)
        manifest = {"artifact_hash": "a" * 64}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if self._mutate is not None:
            self._mutate()
        return manifest


class _FakeRun:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def report(self) -> dict[str, object]:
        return {"status": "ok"}


def _args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=Path("dataset.csv"),
        manifest=Path("manifest.json"),
        batch_manifest=Path("batch-manifest.json"),
        scenario_index_sha256=MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
        output=output,
        test_fraction=0.25,
        ensemble_size=2,
        n_estimators=4,
        horizon=2,
        seed=42,
        conformal_level=0.9,
    )


class TrainTrack2SurrogateSourceContractTest(unittest.TestCase):
    def test_read_regular_rejects_final_swap_to_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_bytes(b"trusted source\n")
            attacker = root / "attacker.py"
            attacker.write_bytes(b"attacker source\n")
            moved = root / "moved-source.py"
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
                if path == source.name and dir_fd is not None and not swapped:
                    source.rename(moved)
                    source.symlink_to(attacker)
                    swapped = True
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with patch.object(MODULE.os, "open", side_effect=swapping_open):
                with self.assertRaises(OSError):
                    MODULE._read_regular(source, "source")

            self.assertTrue(swapped)

    def test_torn_source_read_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "surrogate"
            trainer_source = root / "train_track2_surrogate.py"
            trainer_source.write_bytes(b"A" * (2 * 1024**2))
            frozen = trainer_source.stat()
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
                    trainer_source.write_bytes(b"B" * (2 * 1024**2))
                    mutated = True
                return chunk

            with (
                patch.object(MODULE, "__file__", str(trainer_source)),
                patch.object(MODULE.os, "fstat", side_effect=frozen_fstat),
                patch.object(MODULE.os, "read", side_effect=mutating_read),
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(
                    MODULE,
                    "_validated_batch_lineage",
                    return_value=_batch_lineage(),
                ),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(MODULE, "_verify_trajectory_actions"),
                patch.object(
                    MODULE,
                    "fit_track2_surrogate",
                    return_value=_FakeRun(_FakeModel()),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "changed while it was read"):
                    MODULE.main()

            self.assertTrue(mutated)
            self.assertFalse(output.exists())

    def test_metrics_bind_executed_sources_and_model_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "surrogate"
            model = _FakeModel()
            with (
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(
                    MODULE,
                    "_validated_batch_lineage",
                    return_value=_batch_lineage(),
                ),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(MODULE, "_verify_trajectory_actions"),
                patch.object(
                    MODULE,
                    "fit_track2_surrogate",
                    return_value=_FakeRun(model),
                ),
            ):
                self.assertEqual(MODULE.main(), 0)

            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            expected_paths = (
                Path(MODULE.__file__).absolute(),
                Path(MODULE._track2_module.__file__).absolute(),
                Path(MODULE._surrogate_module.__file__).absolute(),
            )
            self.assertEqual(
                metrics["executed_sources"],
                [
                    {
                        "path": str(path),
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in expected_paths
                ],
            )
            manifest = output / "model" / "manifest.json"
            self.assertEqual(metrics["surrogate_artifact_hash"], "a" * 64)
            self.assertEqual(
                metrics["surrogate_manifest_sha256"],
                sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(metrics["scenario_batch"], _batch_lineage())
            self.assertEqual(model.training_metadata["scenario_batch"], _batch_lineage())

    def test_mutation_before_metrics_write_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "surrogate"
            trainer_source = root / "train_track2_surrogate.py"
            trainer_source.write_bytes(b"original source\n")
            model = _FakeModel(
                lambda: trainer_source.write_bytes(b"mutated source\n")
            )
            with (
                patch.object(MODULE, "__file__", str(trainer_source)),
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(
                    MODULE,
                    "_validated_batch_lineage",
                    return_value=_batch_lineage(),
                ),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(MODULE, "_verify_trajectory_actions"),
                patch.object(
                    MODULE,
                    "fit_track2_surrogate",
                    return_value=_FakeRun(model),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "executed source changed"):
                    MODULE.main()

            self.assertFalse((output / "metrics.json").exists())

    def test_mutation_inside_fit_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "surrogate"
            trainer_source = root / "train_track2_surrogate.py"
            trainer_source.write_bytes(b"original source\n")

            def mutating_fit(*args: object, **kwargs: object) -> _FakeRun:
                trainer_source.write_bytes(b"mutated source\n")
                return _FakeRun(_FakeModel())

            with (
                patch.object(MODULE, "__file__", str(trainer_source)),
                patch.object(MODULE, "parse_args", return_value=_args(output)),
                patch.object(
                    MODULE,
                    "_validated_batch_lineage",
                    return_value=_batch_lineage(),
                ),
                patch.object(MODULE, "load_trajectory_dataset", return_value=[]),
                patch.object(MODULE, "_verify_trajectory_actions"),
                patch.object(MODULE, "fit_track2_surrogate", side_effect=mutating_fit),
            ):
                with self.assertRaisesRegex(RuntimeError, "executed source changed"):
                    MODULE.main()

            self.assertFalse(output.exists())

    def test_batch_lineage_authenticates_exact_canonical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            manifests = root / "manifests"
            dataset.mkdir()
            manifests.mkdir()
            records = []
            for scenario_id in MODULE._EXPECTED_SCENARIO_IDS:
                dataset_path = dataset / f"{scenario_id}.csv"
                manifest_path = manifests / f"{scenario_id}.json"
                scenario_dir = root / scenario_id
                canonical_dir = scenario_dir / "canonical"
                canonical_dir.mkdir(parents=True)
                run_manifest = scenario_dir / "manifest.json"
                summary_report = scenario_dir / "summary-report.txt"
                summary_extraction = scenario_dir / "summary-extraction.json"
                canonical_chdd = canonical_dir / "chdd.csv"
                run_manifest.write_bytes(f"run:{scenario_id}".encode())
                summary_report.write_bytes(f"summary:{scenario_id}".encode())
                summary_extraction.write_bytes(f"extraction:{scenario_id}".encode())
                canonical_chdd.write_bytes(f"chdd:{scenario_id}".encode())
                dataset_path.write_bytes(f"dataset:{scenario_id}".encode())
                run_sha = sha256(run_manifest.read_bytes()).hexdigest()
                summary_sha = sha256(summary_report.read_bytes()).hexdigest()
                extraction_sha = sha256(summary_extraction.read_bytes()).hexdigest()
                chdd_sha = sha256(canonical_chdd.read_bytes()).hexdigest()
                dataset_sha = sha256(dataset_path.read_bytes()).hexdigest()
                manifest_path.write_text(
                    json.dumps(
                        {
                            "generator": "timesoil.aios.opm_chdd",
                            "provenance": {
                                "opm_source_sha256": MODULE.MODEL_Z_SOURCE_SHA256,
                                "opm_run_manifest": f"../{scenario_id}/manifest.json",
                                "opm_run_manifest_sha256": run_sha,
                                "summary_extraction_manifest": (
                                    f"../{scenario_id}/summary-extraction.json"
                                ),
                                "summary_extraction_manifest_sha256": extraction_sha,
                            },
                            "source": {
                                "summary_csv": f"../{scenario_id}/summary-report.txt",
                                "summary_csv_sha256": summary_sha,
                            },
                            "scenario": {
                                "scenario_id": scenario_id,
                                "source_model": "model_z_opm",
                            },
                            "outputs": {
                                "track2_csv": {
                                    "name": f"{scenario_id}.csv",
                                    "sha256": dataset_sha,
                                },
                                "chdd_csv": {"name": "chdd.csv", "sha256": chdd_sha},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                records.append(
                    {
                        "scenario_id": scenario_id,
                        "actions_sha256": MODULE._EXPECTED_SCENARIO_ACTIONS_SHA256[
                            scenario_id
                        ],
                        "run_manifest": f"{scenario_id}/manifest.json",
                        "run_manifest_sha256": run_sha,
                        "summary_report_sha256": summary_sha,
                        "summary_extraction_sha256": extraction_sha,
                        "canonical_chdd": f"{scenario_id}/canonical/chdd.csv",
                        "canonical_chdd_sha256": chdd_sha,
                        "dataset": f"dataset/{scenario_id}.csv",
                        "dataset_sha256": dataset_sha,
                        "export_manifest": f"manifests/{scenario_id}.json",
                        "export_manifest_sha256": sha256(
                            manifest_path.read_bytes()
                        ).hexdigest(),
                    }
                )
            batch = root / "manifest.json"
            payload = {
                "schema": MODULE._SCENARIO_RUN_SCHEMA,
                "executed_sources": {},
                "official_source_sha256": MODULE.MODEL_Z_SOURCE_SHA256,
                "scenario_index_sha256": MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
                "scenario_count": 10,
                "sequential": True,
                "scenarios": records,
                "training": {},
            }
            batch.write_text(json.dumps(payload), encoding="utf-8")

            lineage = MODULE._validated_batch_lineage(
                batch,
                MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
                dataset,
                manifests,
            )
            self.assertEqual(
                lineage["batch_manifest_sha256"], sha256(batch.read_bytes()).hexdigest()
            )
            self.assertEqual(
                [item["scenario_id"] for item in lineage["scenarios"]],
                list(MODULE._EXPECTED_SCENARIO_IDS),
            )

            payload["scenarios"][0]["actions_sha256"] = "f" * 64
            batch.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "actions SHA-256 mismatch"):
                MODULE._validated_batch_lineage(
                    batch,
                    MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
                    dataset,
                    manifests,
                )
            payload["scenarios"][0]["actions_sha256"] = (
                MODULE._EXPECTED_SCENARIO_ACTIONS_SHA256["baseline"]
            )

            payload["scenarios"][0]["dataset_sha256"] = "0" * 64
            batch.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset SHA-256 mismatch"):
                MODULE._validated_batch_lineage(
                    batch,
                    MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
                    dataset,
                    manifests,
                )

            payload["scenarios"][0]["dataset_sha256"] = sha256(
                (dataset / "baseline.csv").read_bytes()
            ).hexdigest()
            payload["scenarios"][0]["export_manifest_sha256"] = "0" * 64
            batch.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "export manifest SHA-256 mismatch"):
                MODULE._validated_batch_lineage(
                    batch,
                    MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
                    dataset,
                    manifests,
                )

            payload["scenarios"][0]["export_manifest_sha256"] = sha256(
                (manifests / "baseline.json").read_bytes()
            ).hexdigest()
            payload["scenarios"][0]["scenario_id"] = "scenario-000"
            batch.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "IDs are not the frozen canonical set"):
                MODULE._validated_batch_lineage(
                    batch,
                    MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
                    dataset,
                    manifests,
                )

    def test_batch_lineage_rejects_non_frozen_index_and_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "not the frozen Track 2 index"):
            MODULE._validated_batch_lineage(
                Path("missing.json"),
                "0" * 64,
                Path("dataset"),
                Path("manifests"),
            )

    def test_batch_lineage_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset").mkdir()
            (root / "manifests").mkdir()
            batch = root / "manifest.json"
            batch.write_text('{"schema":"bad","schema":"duplicate"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                MODULE._validated_batch_lineage(
                    batch,
                    MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
                    root / "dataset",
                    root / "manifests",
                )

    def test_exported_control_hashes_are_checked_after_six_decimal_rendering(self) -> None:
        trajectories = [
            type("Trajectory", (), {"scenario_id": scenario_id})()
            for scenario_id in MODULE._EXPECTED_SCENARIO_IDS
        ]
        expected = iter(MODULE._EXPECTED_EXPORTED_ACTIONS_SHA256.values())
        with (
            patch.object(MODULE, "_trajectory_controls", return_value=()),
            patch.object(MODULE, "_actions_sha256", side_effect=expected),
        ):
            MODULE._verify_trajectory_actions(trajectories)

        with (
            patch.object(MODULE, "_trajectory_controls", return_value=()),
            patch.object(MODULE, "_actions_sha256", return_value="0" * 64),
        ):
            with self.assertRaisesRegex(ValueError, "loaded controls disagree"):
                MODULE._verify_trajectory_actions(trajectories)


if __name__ == "__main__":
    unittest.main()
