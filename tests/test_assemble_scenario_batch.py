from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from assemble_scenario_batch import main
from benchmark_bhp_surrogate import START
from timesoil.aios.opm import OPM_EXPORT_VECTORS, OPM_IMAGE, OPM_IMAGE_DIGEST
from timesoil.aios.track2 import (
    CANONICAL_COLUMNS,
    MODEL_Z_SOURCE_SHA256,
    load_trajectory_dataset,
)


_CONNECTION_VECTORS = {
    "COFR", "CWFR", "COPR", "COPT", "CWPR", "CWPT", "COIT", "CWIR", "CWIT",
}
_WELLS = ("P1", "P2")
_SUMMARY_REPORT = b"verified OPM summary\n"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _summary_replay(command, **_kwargs):
    return subprocess.CompletedProcess(command, 0, _SUMMARY_REPORT, b"")


def _frame(scenario_id: str, *, shift: float, history_shift: float) -> pd.DataFrame:
    dates = pd.date_range("2006-09-01", periods=6, freq="MS")
    rows = []
    for month, date in enumerate(dates):
        for index, well in enumerate(_WELLS):
            state_bump = shift if date > START else history_shift
            action_bump = shift if date >= START else 0.0
            rows.append({
                "scenario_id": scenario_id,
                "source_model": "model_z_opm",
                "date": date.date().isoformat(),
                "well": well,
                "oil_tpd": 10.0 + index + month + state_bump,
                "liquid_tpd": 20.0 + index + month + state_bump,
                "pressure_bar": 200.0 - month + state_bump,
                "control_value": 30.0 + index + action_bump,
                "control_target": "ORAT",
                "status": 1.0,
            })
    return pd.DataFrame(rows)[list(CANONICAL_COLUMNS)]


def _artifact(path: Path, run: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(run).as_posix(),
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _run_tree(
    batch: Path,
    scenario_id: str,
    *,
    shift: float = 0.0,
    history_shift: float = 0.0,
    complete: bool = True,
) -> Path:
    run = batch / scenario_id
    (run / "input").mkdir(parents=True)
    (run / "output").mkdir()
    (run / "canonical").mkdir()
    deck = run / "input" / "CASE.DATA"
    overlay = run / "input" / "_TIMESOIL_SUMMARY.INC"
    deck.write_text("RUNSPEC\n", encoding="ascii")
    overlay.write_text("DATE\n/\n", encoding="ascii")
    smspec = run / "output" / "CASE.SMSPEC"
    unsmry = run / "output" / "CASE.UNSMRY"
    smspec.write_bytes(b"smspec")
    unsmry.write_bytes(b"unsmry")
    report = run / "summary-report.txt"
    report.write_bytes(_SUMMARY_REPORT)
    chdd = run / "canonical" / "chdd.csv"
    chdd.write_text("date,well,value\n2007-01-01,P1,1.0\n", encoding="utf-8")
    trajectory = run / "canonical" / "trajectory.csv"
    _frame(scenario_id, shift=shift, history_shift=history_shift).to_csv(
        trajectory, index=False
    )

    raw_records = [
        {"path": path.relative_to(run).as_posix(), "bytes": path.stat().st_size,
         "sha256": _hash(path)}
        for path in (smspec, unsmry)
    ]
    run_manifest = run / "manifest.json"
    run_manifest.write_text(json.dumps({
        "schema": "timesoil.aios.opm-run/v1",
        "status": "success",
        "returncode": 0,
        "image_reference": OPM_IMAGE,
        "image_digest": OPM_IMAGE_DIGEST,
        "source_sha256": MODEL_Z_SOURCE_SHA256,
        "deck": deck.name,
        "deck_sha256": _hash(deck),
        "summary_contract": {"overlay": overlay.name, "overlay_sha256": _hash(overlay)},
        "artifacts": [
            {"path": f"input/{deck.name}", "sha256": _hash(deck)},
            {"path": f"input/{overlay.name}", "sha256": _hash(overlay)},
            *raw_records,
        ],
    }, sort_keys=True), encoding="utf-8")

    available = [
        "TIME", "YEARS",
        *(f"{vector}:P1" for vector in OPM_EXPORT_VECTORS if vector not in _CONNECTION_VECTORS),
        *(f"{vector}:P1:1" for vector in OPM_EXPORT_VECTORS if vector in _CONNECTION_VECTORS),
    ]
    mount = f"type=bind,src={(run / 'output').resolve()},dst=/output,readonly"
    user = f"{os.getuid()}:{os.getgid()}"
    extraction = run / "summary-extraction.json"
    extraction.write_text(json.dumps({
        "schema": "timesoil.aios.opm-summary-extraction/v1",
        "run_manifest": {"path": run_manifest.name, "sha256": _hash(run_manifest)},
        "image": {"reference": OPM_IMAGE, "digest": OPM_IMAGE_DIGEST},
        "raw_summary_artifacts": raw_records,
        "summary_input": "output/CASE.SMSPEC",
        "commands": {
            "list": ["docker", "run", "--rm", "--network=none", "--user", user,
                     "--mount", mount, OPM_IMAGE, "summary", "-l", "/output/CASE.SMSPEC"],
            "report": ["docker", "run", "--rm", "--network=none", "--user", user,
                       "--mount", mount, OPM_IMAGE, "summary", "-r", "/output/CASE.SMSPEC",
                       *available],
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
        "output_report": {"path": report.name, "bytes": report.stat().st_size,
                          "sha256": _hash(report)},
    }, sort_keys=True), encoding="utf-8")

    export_manifest = run / "canonical" / "manifest.json"
    export_manifest.write_text(json.dumps({
        "schema_version": 1,
        "generator": "timesoil.aios.opm_chdd",
        "provenance": {
            "opm_run_manifest": "../manifest.json",
            "opm_run_manifest_sha256": _hash(run_manifest),
            "opm_source_sha256": MODEL_Z_SOURCE_SHA256,
            "summary_extraction_manifest": "../summary-extraction.json",
            "summary_extraction_manifest_sha256": _hash(extraction),
        },
        "source": {
            "summary_csv": "../summary-report.txt",
            "summary_csv_sha256": _hash(report),
            "deck_sha256": _hash(deck),
        },
        "scenario": {"scenario_id": scenario_id, "source_model": "model_z_opm"},
        "outputs": {"track2_csv": {
            "name": trajectory.name,
            "row_count": len(pd.read_csv(trajectory)),
            "sha256": _hash(trajectory),
        }},
    }, sort_keys=True), encoding="utf-8")

    receipt = run / "full-cycle-receipt.json"
    receipt.write_text(json.dumps({
        "schema": "timesoil.aios.track2-full-cycle/v1",
        "complete": complete,
        "run_id": scenario_id,
        "source_sha256": MODEL_Z_SOURCE_SHA256,
        "controls": {"canonical_actions_sha256": sha256(scenario_id.encode()).hexdigest()},
        "artifacts": {
            "opm_run_manifest": _artifact(run_manifest, run),
            "summary_report": _artifact(report, run),
            "summary_extraction_manifest": _artifact(extraction, run),
            "canonical_export_manifest": _artifact(export_manifest, run),
            "canonical_chdd_csv": _artifact(chdd, run),
            "canonical_trajectory_csv": _artifact(trajectory, run),
        },
    }, sort_keys=True), encoding="utf-8")
    return run


def _batch(root: Path, **overrides: dict[str, object]) -> Path:
    batch = root / "cycles"
    batch.mkdir()
    _run_tree(batch, "baseline")
    _run_tree(batch, "candidate-001", shift=3.0, **overrides.get("candidate-001", {}))
    _run_tree(batch, "candidate-002", shift=-2.0, **overrides.get("candidate-002", {}))
    return batch


class AssembleScenarioBatchTests(unittest.TestCase):
    def test_assembled_batch_is_loadable_and_hash_consistent(self) -> None:
        with TemporaryDirectory() as directory:
            batch = _batch(Path(directory))
            self.assertEqual(main([str(batch)]), 0)

            manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "timesoil.aios.track2-scenario-run/v2")
            self.assertEqual(manifest["official_source_sha256"], MODEL_Z_SOURCE_SHA256)
            self.assertEqual(manifest["scenario_count"], 3)
            self.assertEqual(len(manifest["scenarios"]), 3)
            self.assertEqual(manifest["scenarios"][0]["scenario_id"], "baseline")

            # Every sha256 field in the batch manifest matches the file it names.
            for record in manifest["scenarios"]:
                for name in ("dataset", "export_manifest", "run_manifest", "canonical_chdd"):
                    path = (batch / record[name]).resolve()
                    self.assertTrue(path.is_relative_to(batch.resolve()), name)
                    self.assertEqual(_hash(path), record[name + "_sha256"], name)
                run = batch / record["run_directory"]
                self.assertEqual(
                    _hash(run / "summary-report.txt"), record["summary_report_sha256"]
                )
                self.assertEqual(
                    _hash(run / "summary-extraction.json"),
                    record["summary_extraction_sha256"],
                )
                self.assertEqual(
                    _hash(run / "canonical" / "manifest.json"),
                    record["source_export_manifest_sha256"],
                )

            full = load_trajectory_dataset(
                batch / "dataset",
                manifest=batch / "manifests",
                _summary_run=_summary_replay,
            )
            self.assertTrue(full.model_z_identity)
            self.assertEqual(
                {item.scenario_id for item in full},
                {record["scenario_id"] for record in manifest["scenarios"]},
            )

    def test_dry_run_writes_nothing(self) -> None:
        with TemporaryDirectory() as directory:
            batch = _batch(Path(directory))
            self.assertEqual(main([str(batch), "--dry-run"]), 0)
            for name in ("dataset", "manifests", "manifest.json"):
                self.assertFalse((batch / name).exists(), name)

    def test_incomplete_run_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            batch = _batch(Path(directory), **{"candidate-001": {"complete": False}})
            with self.assertRaises(SystemExit):
                main([str(batch)])
            self.assertFalse((batch / "dataset").exists())

    def test_missing_receipt_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            batch = _batch(Path(directory))
            (batch / "candidate-001" / "full-cycle-receipt.json").unlink()
            with self.assertRaises(SystemExit):
                main([str(batch), "--run", "baseline", "--run", "candidate-001"])

    def test_divergent_history_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            batch = _batch(Path(directory), **{"candidate-002": {"history_shift": 5.0}})
            with self.assertRaises(SystemExit):
                main([str(batch)])
            self.assertFalse((batch / "manifest.json").exists())

    def test_missing_baseline_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            batch = _batch(Path(directory))
            with self.assertRaises(SystemExit):
                main([str(batch), "--run", "candidate-001", "--run", "candidate-002"])

    def test_second_assembly_refuses_to_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            batch = _batch(Path(directory))
            self.assertEqual(main([str(batch)]), 0)
            with self.assertRaises(SystemExit):
                main([str(batch)])


if __name__ == "__main__":
    unittest.main()
