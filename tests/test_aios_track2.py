from __future__ import annotations

from hashlib import sha256
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from timesoil.aios.opm import OPM_EXPORT_VECTORS, OPM_IMAGE, OPM_IMAGE_DIGEST
from timesoil.aios.surrogate import ScenarioTrajectory
from timesoil.aios.track2 import load_trajectory_dataset, trajectory_from_frame


_CONNECTION_VECTORS = {
    "COFR", "CWFR", "COPR", "COPT", "CWPR", "CWPT",
    "COIT", "CWIR", "CWIT",
}


_MODEL_Z_SOURCE_SHA256 = "4af3b60f8c053b858d52882bc514f2cdf434573c3919574e532e620d06c45aaa"
_OTHER_SOURCE_SHA256 = "261591b458084eaaf8c86a601e68d3bdc6e91fed9f0117fdcbe58cfca4eb882e"


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

    def test_dataset_manifest_is_required(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                load_trajectory_dataset(_write_csvs(Path(directory)))

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

    def test_verified_opm_hash_chain_marks_model_z_identity(self) -> None:
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
        self.assertTrue(trajectories.model_z_identity)
        self.assertEqual(
            trajectories.scenario_hashes,
            tuple(item.content_hash for item in trajectories),
        )

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
            ("model_z_opm", _OTHER_SOURCE_SHA256),
            ("other_opm", _MODEL_Z_SOURCE_SHA256),
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
                self.assertFalse(trajectories.model_z_identity)


if __name__ == "__main__":
    unittest.main()
