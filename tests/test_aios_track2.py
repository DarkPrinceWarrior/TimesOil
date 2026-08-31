from __future__ import annotations

from hashlib import sha256
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import timesoil.aios.surrogate as surrogate_module
from timesoil.aios.opm import OPM_EXPORT_VECTORS, OPM_IMAGE, OPM_IMAGE_DIGEST
from timesoil.aios.surrogate import ScenarioTrajectory, StatefulSurrogate, Track2Surrogate
from timesoil.aios.track2 import (
    fit_track2_surrogate,
    load_trajectory_dataset,
    trajectory_from_frame,
)


_CONNECTION_VECTORS = {
    "COFR", "CWFR", "COPR", "COPT", "CWPR", "CWPT",
    "COIT", "CWIR", "CWIT",
}


_MODEL_Z_SOURCE_SHA256 = "4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa"
_MODEL_Y_SOURCE_SHA256 = "261591b458084eaaf8c86a601e68d3bdc6e91fed9f0117fdcbe58cfca4eb882e"


def _scenarios(
    count: int = 4, *, source_model: str = "model_z_opm"
) -> list[ScenarioTrajectory]:
    dates = pd.date_range("2010-01-01", periods=18, freq="MS")
    wells = ("P1", "P2", "P3")
    scenarios = []
    for scenario in range(count):
        states = np.zeros((len(dates), len(wells), 3))
        actions = np.zeros((len(dates), len(wells), 3))
        states[0, :, 1] = np.array([90.0, 70.0, 50.0]) * (1 + 0.02 * scenario)
        states[0, :, 0] = states[0, :, 1] * np.array([0.7, 0.6, 0.5])
        states[0, :, 2] = 240.0
        actions[..., 0] = 35.0 + scenario * 4 + np.arange(len(dates))[:, None] * 0.3
        actions[..., 1] = 2.0
        actions[..., 2] = 1.0
        for month in range(len(dates) - 1):
            states[month + 1, :, 1] = 0.97 * states[month, :, 1] + 0.08 * actions[month, :, 0]
            fraction = states[month, :, 0] / states[month, :, 1] * 0.995
            states[month + 1, :, 0] = states[month + 1, :, 1] * fraction
            states[month + 1, :, 2] = states[month, :, 2] + 0.01 * (
                actions[month, :, 0] - states[month, :, 1]
            )
        scenarios.append(ScenarioTrajectory(
            scenario_id=f"scenario-{scenario}",
            source_model=source_model,
            dates=dates,
            well_ids=wells,
            states=states,
            actions=actions,
        ))
    return scenarios


def _frame(item: ScenarioTrajectory) -> pd.DataFrame:
    rows = []
    for month, date in enumerate(item.dates):
        for well, well_id in enumerate(item.well_ids):
            rows.append({
                "scenario_id": item.scenario_id,
                "source_model": item.source_model,
                "date": date,
                "well": well_id,
                "oil_tpd": item.states[month, well, 0],
                "liquid_tpd": item.states[month, well, 1],
                "pressure_bar": item.states[month, well, 2],
                "control_value": item.actions[month, well, 0],
                "control_target": "WRAT",
                "status": item.actions[month, well, 2],
            })
    return pd.DataFrame(rows)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _summary_replay(report: bytes):
    def replay(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, report, b"")

    return replay


def _write_csvs(root: Path, *, source_model: str = "model_z_opm") -> Path:
    data = root / "data"
    data.mkdir()
    for item in _scenarios(source_model=source_model):
        _frame(item).to_csv(data / f"{item.scenario_id}.csv", index=False)
    return data


def _write_provenance(
    root: Path,
    data: Path,
    *,
    source_sha256: str = _MODEL_Z_SOURCE_SHA256,
) -> Path:
    proof = root / "proof"
    run_input = root / "run" / "input"
    run_output = root / "run" / "output"
    proof.mkdir()
    run_input.mkdir(parents=True)
    run_output.mkdir()
    summary = root / "run" / "summary.txt"
    summary.write_text("verified OPM summary\n", encoding="utf-8")
    deck = run_input / "CASE.DATA"
    overlay = run_input / "_TIMESOIL_SUMMARY.INC"
    deck.write_text("RUNSPEC\n", encoding="ascii")
    overlay.write_text("DATE\n/\n", encoding="ascii")
    smspec = run_output / "CASE.SMSPEC"
    unsmry = run_output / "CASE.UNSMRY"
    smspec.write_bytes(b"smspec")
    unsmry.write_bytes(b"unsmry")
    raw_records = [
        {
            "path": path.relative_to(root / "run").as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_hash(path),
        }
        for path in (smspec, unsmry)
    ]
    available = [
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
    run_manifest = root / "run" / "manifest.json"
    run_manifest.write_text(json.dumps({
        "schema": "timesoil.aios.opm-run/v1",
        "status": "success",
        "returncode": 0,
        "image_reference": OPM_IMAGE,
        "image_digest": OPM_IMAGE_DIGEST,
        "source_sha256": source_sha256,
        "deck": deck.name,
        "deck_sha256": _file_hash(deck),
        "summary_contract": {
            "overlay": overlay.name,
            "overlay_sha256": _file_hash(overlay),
        },
        "artifacts": [
            {"path": f"input/{deck.name}", "sha256": _file_hash(deck)},
            {"path": f"input/{overlay.name}", "sha256": _file_hash(overlay)},
            *raw_records,
        ],
    }, sort_keys=True), encoding="utf-8")
    extraction = root / "run" / "summary-extraction.json"
    extraction.write_text(
        json.dumps(
            {
                "schema": "timesoil.aios.opm-summary-extraction/v1",
                "run_manifest": {
                    "path": run_manifest.name,
                    "sha256": _file_hash(run_manifest),
                },
                "image": {"reference": OPM_IMAGE, "digest": OPM_IMAGE_DIGEST},
                "raw_summary_artifacts": raw_records,
                "summary_input": "output/CASE.SMSPEC",
                "commands": {
                    "list": [
                        "docker", "run", "--rm", "--network=none", "--user",
                        f"{os.getuid()}:{os.getgid()}", "--mount",
                        f"type=bind,src={run_output.resolve()},dst=/output,readonly",
                        OPM_IMAGE, "summary", "-l", "/output/CASE.SMSPEC",
                    ],
                    "report": [
                        "docker", "run", "--rm", "--network=none", "--user",
                        f"{os.getuid()}:{os.getgid()}", "--mount",
                        f"type=bind,src={run_output.resolve()},dst=/output,readonly",
                        OPM_IMAGE, "summary", "-r", "/output/CASE.SMSPEC",
                        *available,
                    ],
                },
                "shell": False,
                "report_steps_only": True,
                "vector_selection": {
                    "mode": "filtered-summary-list",
                    "available": available,
                    "available_sha256": sha256(
                        json.dumps(available, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "selected": available,
                    "required": list(OPM_EXPORT_VECTORS),
                },
                "output_report": {
                    "path": summary.name,
                    "bytes": summary.stat().st_size,
                    "sha256": _file_hash(summary),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for dataset in sorted(data.glob("*.csv")):
        frame = pd.read_csv(dataset)
        scenario_id = str(frame["scenario_id"].iloc[0])
        source_model = str(frame["source_model"].iloc[0])
        manifest = {
            "schema_version": 1,
            "generator": "timesoil.aios.opm_chdd",
            "provenance": {
                "opm_run_manifest": "../run/manifest.json",
                "opm_run_manifest_sha256": _file_hash(run_manifest),
                "opm_source_sha256": source_sha256,
                "summary_extraction_manifest": "../run/summary-extraction.json",
                "summary_extraction_manifest_sha256": _file_hash(extraction),
            },
            "source": {
                "summary_csv": "../run/summary.txt",
                "summary_csv_sha256": _file_hash(summary),
                "deck_sha256": _file_hash(deck),
            },
            "scenario": {"scenario_id": scenario_id, "source_model": source_model},
            "outputs": {
                "track2_csv": {
                    "name": dataset.name,
                    "row_count": len(frame),
                    "sha256": _file_hash(dataset),
                }
            },
        }
        (proof / f"{scenario_id}.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
    return proof


class Track2Tests(unittest.TestCase):
    def test_strict_scenario_conformal_requires_ten_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 10 whole scenarios"):
            fit_track2_surrogate(
                _scenarios(),
                ensemble_size=2,
                n_estimators=1,
                horizon=3,
                seed=7,
                conformal_level=0.9,
            )

    def test_loso_conformal_calibration_reports_coverage_and_widths(self) -> None:
        run = fit_track2_surrogate(
            _scenarios(10),
            ensemble_size=2,
            n_estimators=1,
            horizon=3,
            seed=7,
            conformal_level=0.9,
        )
        calibration = run.calibration_metrics
        self.assertIsNotNone(calibration)
        assert calibration is not None
        self.assertEqual(calibration["scenario_count"], 10)
        self.assertEqual(calibration["quantile_rank"], 10)
        self.assertFalse(calibration["independent_validation"])
        self.assertEqual(
            set(calibration["mean_interval_width_by_target"]),
            {"oil_tpd", "liquid_tpd", "pressure_bar"},
        )
        prediction = run.model.rollout(
            _scenarios(10)[0].states[0], _scenarios(10)[0].actions[:3]
        )
        self.assertTrue(np.all(prediction.interval_half_width > 0))

    def test_scenario_split_rollout_and_ood_gate(self) -> None:
        run = fit_track2_surrogate(_scenarios(), ensemble_size=3, n_estimators=20, horizon=4, seed=7)
        self.assertTrue(set(run.train_ids).isdisjoint(run.test_ids))
        self.assertFalse(run.model_z_ready)
        self.assertIsInstance(run.model, StatefulSurrogate)
        self.assertLess(run.test_metrics["liquid_wape"], 0.2)

        item = _scenarios()[0]
        prediction = run.model.rollout(item.states[0], item.actions[:4])
        self.assertEqual(prediction.mean.shape, (4, 3, 3))
        self.assertTrue(np.all(prediction.mean[..., 1] >= prediction.mean[..., 0]))
        self.assertTrue(np.all(prediction.mean >= 0))

        candidates = np.stack([
            item.actions[:4], item.actions[:4] * np.array([1000.0, 1.0, 1.0])
        ])
        result = run.model.predict(item.states[0], candidates)
        self.assertEqual(result.accepted.shape, (2,))
        self.assertFalse(result.accepted[1])

    def test_ood_disagreement_uses_training_scale_floor(self) -> None:
        model = fit_track2_surrogate(
            _scenarios(), ensemble_size=3, n_estimators=12, horizon=3, seed=7
        ).model
        features = (model.feature_min + model.feature_max) / 2
        mean = np.zeros(3)
        scale_floor = np.maximum(model.state_scale, model.feature_scale[:3])

        score, ood, reasons = model._diagnose(
            features, mean, 0.49 * scale_floor, 0.0
        )
        self.assertFalse(ood)
        self.assertLess(score, 1.0)
        self.assertNotIn("ensemble_disagreement", reasons)

        score, ood, reasons = model._diagnose(
            features, mean, 0.51 * scale_floor, 0.0
        )
        self.assertTrue(ood)
        self.assertGreater(score, 1.0)
        self.assertIn("ensemble_disagreement", reasons)

        _, ood, reasons = model._diagnose(features, mean, np.zeros(3), 0.201)
        self.assertTrue(ood)
        self.assertIn("physical_projection_excess", reasons)

    def test_model_artifact_roundtrip_and_hash_check(self) -> None:
        run = fit_track2_surrogate(_scenarios(), ensemble_size=2, n_estimators=12, horizon=3)
        with TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            manifest = run.model.save(model_dir)
            loaded = Track2Surrogate.load(model_dir)
            item = _scenarios()[0]
            np.testing.assert_allclose(
                loaded.rollout(item.states[0], item.actions[:3]).mean,
                run.model.rollout(item.states[0], item.actions[:3]).mean,
            )
            self.assertEqual(len(manifest["artifact_hash"]), 64)

            target = model_dir / "member_00_oil_tpd.txt"
            original_target_bytes = target.read_bytes()
            replacement_bytes = (model_dir / "member_00_liquid_tpd.txt").read_bytes()
            self.assertNotEqual(original_target_bytes, replacement_bytes)
            original_read = surrogate_module._read_regular_bytes
            swapped = False

            def swapping_read(path: Path, label: str) -> bytes:
                nonlocal swapped
                data = original_read(path, label)
                if path == target and not swapped:
                    replacement = target.with_suffix(".swap")
                    replacement.write_bytes(replacement_bytes)
                    replacement.replace(target)
                    swapped = True
                return data

            with patch.object(surrogate_module, "_read_regular_bytes", swapping_read):
                stable = Track2Surrogate.load(model_dir)
            self.assertTrue(swapped)
            self.assertEqual(
                stable.boosters[0][0].model_to_string(),
                run.model.boosters[0][0].model_to_string(),
            )
            np.testing.assert_allclose(
                stable.rollout(item.states[0], item.actions[:3]).mean,
                run.model.rollout(item.states[0], item.actions[:3]).mean,
            )
            target.write_bytes(original_target_bytes)

            manifest_path = model_dir / "manifest.json"
            original_files = dict(manifest["files"])
            dummy = model_dir / "dummy.bin"
            dummy.write_bytes(b"dummy")
            nested = model_dir / "nested"
            nested.mkdir()
            (nested / "dummy.bin").write_bytes(b"nested")
            member = next(name for name in original_files if name.startswith("member_"))

            def write_manifest(files: dict[str, str]) -> None:
                forged = {
                    **manifest,
                    "files": files,
                    "artifact_hash": sha256(
                        json.dumps(
                            files,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                }
                manifest_path.write_text(json.dumps(forged), encoding="utf-8")

            cases = {
                "dummy_unlisted_model": {"dummy.bin": _file_hash(dummy)},
                "nested": {
                    **original_files,
                    "nested/dummy.bin": _file_hash(nested / "dummy.bin"),
                },
                "omission": {
                    name: digest for name, digest in original_files.items() if name != member
                },
                "extra": {**original_files, "dummy.bin": _file_hash(dummy)},
            }
            for label, files in cases.items():
                with self.subTest(label=label):
                    write_manifest(files)
                    with self.assertRaisesRegex(ValueError, "artifact file set|basename|model.json"):
                        Track2Surrogate.load(model_dir)

            target.write_bytes(b"\xff")
            invalid_text_files = {**original_files, target.name: _file_hash(target)}
            write_manifest(invalid_text_files)
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                Track2Surrogate.load(model_dir)
            target.write_bytes(original_target_bytes)

            model_path = model_dir / "model.json"
            original_model_bytes = model_path.read_bytes()
            one_member_metadata = json.loads(original_model_bytes)
            one_member_metadata["ensemble_size"] = 1
            model_path.write_text(
                json.dumps(
                    one_member_metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            one_member_files = {
                name: digest
                for name, digest in original_files.items()
                if name == "model.json" or name.startswith("member_00_")
            }
            one_member_files["model.json"] = _file_hash(model_path)
            write_manifest(one_member_files)
            with self.assertRaisesRegex(ValueError, "ensemble_size"):
                Track2Surrogate.load(model_dir)
            model_path.write_bytes(original_model_bytes)
            write_manifest(original_files)

            with model_path.open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "hash check"):
                Track2Surrogate.load(model_dir)

            invalid_metadata = {
                "nonfinite_state_scale": {"state_scale": [float("nan")] * 3},
                "nonpositive_feature_scale": {
                    "feature_scale": [0.0] * len(surrogate_module.RESIDUAL_FEATURES)
                },
                "nonfinite_ood_threshold": {"ood_disagreement": float("nan")},
            }
            for label, replacement in invalid_metadata.items():
                with self.subTest(label=label):
                    metadata = json.loads(original_model_bytes)
                    metadata.update(replacement)
                    model_path.write_text(
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        encoding="utf-8",
                    )
                    forged_files = {
                        **original_files,
                        "model.json": _file_hash(model_path),
                    }
                    write_manifest(forged_files)
                    with self.assertRaisesRegex(ValueError, "finite|invalid"):
                        Track2Surrogate.load(model_dir)

    def test_train_cli_refuses_existing_output_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            script = Path(__file__).resolve().parents[1] / "scripts/train_track2_surrogate.py"

            def invoke(output: Path) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--model-y-proof",
                        "--raw-dir",
                        str(root / "missing-raw"),
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

            output = root / "existing-output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_bytes(b"unchanged")
            result = invoke(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite output", result.stderr)
            self.assertEqual(marker.read_bytes(), b"unchanged")
            self.assertEqual(list(output.iterdir()), [marker])

            symlink = root / "broken-output-link"
            target = root / "missing-target"
            symlink.symlink_to(target, target_is_directory=True)
            result = invoke(symlink)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite output", result.stderr)
            self.assertTrue(symlink.is_symlink())
            self.assertFalse(target.exists())

    def test_canonical_long_frame_contract(self) -> None:
        item = _scenarios(1)[0]
        rows = []
        for month, date in enumerate(item.dates):
            for well, well_id in enumerate(item.well_ids):
                rows.append({
                    "scenario_id": item.scenario_id,
                    "source_model": item.source_model,
                    "date": date,
                    "well": well_id,
                    "oil_tpd": item.states[month, well, 0],
                    "liquid_tpd": item.states[month, well, 1],
                    "pressure_bar": item.states[month, well, 2],
                    "control_value": item.actions[month, well, 0],
                    "control_target": "WRAT",
                    "status": item.actions[month, well, 2],
                })
        restored = trajectory_from_frame(pd.DataFrame(rows))
        expected = ScenarioTrajectory(
            scenario_id=item.scenario_id,
            source_model=item.source_model,
            dates=item.dates,
            well_ids=item.well_ids,
            states=item.states,
            actions=item.actions,
            metadata={"contract": "canonical_long_frame_v1"},
        )
        self.assertEqual(restored.content_hash, expected.content_hash)

    def test_model_z_csv_labels_without_provenance_are_not_ready(self) -> None:
        with TemporaryDirectory() as directory:
            trajectories = load_trajectory_dataset(_write_csvs(Path(directory)))
            run = fit_track2_surrogate(
                trajectories, ensemble_size=2, n_estimators=8, horizon=3
            )
        self.assertFalse(run.model_z_ready)

    def test_tampered_dataset_hash_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = _write_csvs(root)
            proof = _write_provenance(root, data)
            with next(data.glob("*.csv")).open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "dataset hash mismatch"):
                load_trajectory_dataset(
                    data,
                    manifest=proof,
                    _summary_run=_summary_replay(
                        (root / "run" / "summary.txt").read_bytes()
                    ),
                )

    def test_verified_opm_hash_chain_marks_model_z_ready(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = _write_csvs(root)
            trajectories = load_trajectory_dataset(
                data,
                manifest=_write_provenance(root, data),
                _summary_run=_summary_replay(
                    (root / "run" / "summary.txt").read_bytes()
                ),
            )
            run = fit_track2_surrogate(
                trajectories, ensemble_size=2, n_estimators=8, horizon=3
            )
        self.assertTrue(run.model_z_ready)

    def test_forged_extraction_chain_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = _write_csvs(root)
            proof = _write_provenance(root, data)
            extraction = root / "run" / "summary-extraction.json"
            value = json.loads(extraction.read_text(encoding="utf-8"))
            value["commands"]["report"][8] = "openporousmedia/opmreleases:latest"
            extraction.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            for manifest_path in proof.glob("*.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["provenance"]["summary_extraction_manifest_sha256"] = _file_hash(
                    extraction
                )
                manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "canonical pinned command"):
                load_trajectory_dataset(data, manifest=proof)

    def test_exact_sidecar_with_forged_report_is_rejected_by_replay(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = _write_csvs(root)
            proof = _write_provenance(root, data)
            summary = root / "run" / "summary.txt"
            replay = _summary_replay(summary.read_bytes())
            summary.write_bytes(b"forged OPM summary\n")
            extraction = root / "run" / "summary-extraction.json"
            value = json.loads(extraction.read_text(encoding="utf-8"))
            value["output_report"].update(
                {"bytes": summary.stat().st_size, "sha256": _file_hash(summary)}
            )
            extraction.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            for manifest_path in proof.glob("*.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["source"]["summary_csv_sha256"] = _file_hash(summary)
                manifest["provenance"]["summary_extraction_manifest_sha256"] = (
                    _file_hash(extraction)
                )
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )

            with self.assertRaisesRegex(ValueError, "deterministic replay"):
                load_trajectory_dataset(
                    data,
                    manifest=proof,
                    _summary_run=replay,
                )

    def test_model_z_identity_requires_archive_digest_and_canonical_label(self) -> None:
        cases = (
            ("model_z_opm", _MODEL_Y_SOURCE_SHA256),
            ("model_y_opm", _MODEL_Z_SOURCE_SHA256),
        )
        for source_model, source_sha256 in cases:
            with self.subTest(source_model=source_model, source_sha256=source_sha256):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    data = _write_csvs(root, source_model=source_model)
                    trajectories = load_trajectory_dataset(
                        data,
                        manifest=_write_provenance(
                            root, data, source_sha256=source_sha256
                        ),
                        _summary_run=_summary_replay(
                            (root / "run" / "summary.txt").read_bytes()
                        ),
                    )
                    run = fit_track2_surrogate(
                        trajectories, ensemble_size=2, n_estimators=8, horizon=3
                    )
                self.assertFalse(run.model_z_ready)


if __name__ == "__main__":
    unittest.main()
