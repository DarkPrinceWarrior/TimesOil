from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
import subprocess
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from timesoil.aios.opm import OPM_EXPORT_VECTORS, OPM_IMAGE, OPM_IMAGE_DIGEST
from timesoil.aios.opm_chdd import (
    OpmChddError,
    _check_connection_total,
    _float32_identity_tolerance,
    _read_summary,
    export_opm_chdd,
    read_deck_densities,
)


VECTORS = (
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
CONNECTION_VECTORS = (
    "COFR",
    "CWFR",
    "COPR",
    "COPT",
    "CWPR",
    "CWPT",
    "COIT",
    "CWIR",
    "CWIT",
)


class OpmChddTest(unittest.TestCase):
    def test_float32_summary_identity_bound_is_scale_aware(self) -> None:
        tolerance = _float32_identity_tolerance(130799.710938, 175774.128907, 44974.417969)
        self.assertGreaterEqual(tolerance, 0.023438)
        self.assertLess(tolerance, 0.05)

        production_tolerance = _float32_identity_tolerance(2.095487, 8.186954, 6.091465)
        self.assertGreaterEqual(production_tolerance, 0.000002)
        self.assertLess(production_tolerance, 0.000004)

    def test_connection_reconciliation_uses_printed_rounding_bound(self) -> None:
        _check_connection_total("P1", "2025-01-01", "COPR", 1.000018, 1.0, 36)
        with self.assertRaisesRegex(OpmChddError, "abs_tol"):
            _check_connection_total("P1", "2025-01-01", "COPR", 1.000019, 1.0, 36)

    @staticmethod
    def _summary_replay(summary: Path):
        report = summary.read_bytes()

        def replay(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, report, b"")

        return replay

    def _extraction_chain(
        self, root: Path, summary: Path, *, source_sha256: str = "a" * 64
    ) -> tuple[Path, Path]:
        output = root / "output"
        output.mkdir(exist_ok=True)
        raw_paths = (output / "CASE.SMSPEC", output / "CASE.UNSMRY")
        raw_paths[0].write_bytes(b"smspec")
        raw_paths[1].write_bytes(b"unsmry")
        raw_records = [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for path in raw_paths
        ]
        header = next(
            line for line in summary.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        headers = header.split(",") if "," in header else header.split()
        wells = sorted(
            value.split(":", 1)[1]
            for value in headers
            if value.startswith("WLPR:")
        )
        available = [
            "TIME",
            "YEARS",
            *(f"{vector}:{well}" for well in wells for vector in VECTORS),
            *(
                f"{vector}:{well}:1,1,1"
                for well in wells
                for vector in CONNECTION_VECTORS
            ),
        ]
        run_manifest = root / "opm-run.json"
        run_manifest.write_text(
            json.dumps(
                {
                    "schema": "timesoil.aios.opm-run/v1",
                    "status": "success",
                    "returncode": 0,
                    "image_reference": OPM_IMAGE,
                    "image_digest": OPM_IMAGE_DIGEST,
                    "source_sha256": source_sha256,
                    "summary_contract": {
                        "vectors": {
                            "COFR": {
                                "transform": "signed-net audit only: reconcile COPR-COIR"
                            }
                        }
                    },
                    "artifacts": raw_records,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        extraction = root / "summary-extraction.json"
        extraction.write_text(
            json.dumps(
                {
                    "schema": "timesoil.aios.opm-summary-extraction/v1",
                    "run_manifest": {
                        "path": run_manifest.name,
                        "sha256": sha256(run_manifest.read_bytes()).hexdigest(),
                    },
                    "image": {
                        "reference": OPM_IMAGE,
                        "digest": OPM_IMAGE_DIGEST,
                    },
                    "raw_summary_artifacts": raw_records,
                    "summary_input": "output/CASE.SMSPEC",
                    "commands": {
                        "list": [
                            "docker", "run", "--rm", "--network=none", "--user",
                            f"{os.getuid()}:{os.getgid()}", "--mount",
                            f"type=bind,src={output.resolve()},dst=/output,readonly",
                            OPM_IMAGE, "summary", "-l", "/output/CASE.SMSPEC",
                        ],
                        "report": [
                            "docker", "run", "--rm", "--network=none", "--user",
                            f"{os.getuid()}:{os.getgid()}", "--mount",
                            f"type=bind,src={output.resolve()},dst=/output,readonly",
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
                        "sha256": sha256(summary.read_bytes()).hexdigest(),
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return run_manifest, extraction

    def test_opm_whitespace_header_preserves_ijk_connection_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.txt"
            headers = [
                "TIME",
                *(f"{vector}:P1" for vector in VECTORS),
                *(f"{vector}:P1:1,1,1" for vector in CONNECTION_VECTORS),
            ]
            values = [0.0] * len(headers)
            values[headers.index("WEFF:P1")] = 1.0
            later = list(values)
            later[0] = 31.0
            path.write_text(
                " ".join(headers)
                + "\n"
                + " ".join(map(str, later))
                + "\n"
                + " ".join(map(str, values))
                + "\n",
                encoding="utf-8",
            )

            rows, _ = _read_summary(path, start_date=date(2025, 1, 1))

            self.assertEqual(set(rows[0][2]["P1"]), {"1,1,1"})

    def _summary(
        self,
        root: Path,
        rows: list[dict[str, float | str]],
        wells=("P1",),
        extra_fields: tuple[str, ...] = (),
    ) -> Path:
        path = root / "summary.csv"
        for row in rows:
            for well in wells:
                for vector in VECTORS:
                    row.setdefault(f"{vector}:{well}", 0.0)
                row[f"WWPR:{well}"] = float(row.get(f"WLPR:{well}", 0)) - float(
                    row.get(f"WOPR:{well}", 0)
                )
                row[f"WWPT:{well}"] = float(row.get(f"WLPT:{well}", 0)) - float(
                    row.get(f"WOPT:{well}", 0)
                )
        fields = [
            "DATE",
            "TIME",
            *(f"{vector}:{well}" for well in wells for vector in VECTORS),
            *extra_fields,
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _density(self, root: Path, values: dict[str, dict[str, float]]) -> Path:
        path = root / "density.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        return path

    def _schedule_deck(self, root: Path, summary: Path, unit_system: str) -> None:
        with summary.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        wells = sorted(
            header.split(":", 1)[1]
            for header in (rows[0] if rows else {})
            if isinstance(header, str) and header.startswith("WLPR:")
        )
        def parsed_date(value: str) -> datetime:
            for format_ in ("%Y-%m-%d", "%d-%b-%Y"):
                try:
                    return datetime.strptime(value, format_)
                except ValueError:
                    pass
            raise AssertionError(f"unsupported synthetic date: {value}")

        start = parsed_date(rows[0]["DATE"])
        lines = [
            "RUNSPEC",
            "START",
            f" {start.day} {start.strftime('%b').upper()} {start.year} /",
            unit_system,
            "DIMENS",
            " 1 1 1 /",
            "TABDIMS",
            " 1 /",
            "PROPS",
            "DENSITY",
            " 800 1000 1 /",
            "REGIONS",
            "PVTNUM",
            " 1 /",
            "ACTNUM",
            " 1 /",
            "SCHEDULE",
            "WELSPECS",
            " 'DUMMY' 'G' 1 1 /",
            *(f" '{well}' 'G' 1 1 /" for well in wells),
            "/",
            "COMPDAT",
            " 'DUMMY' 1 1 1 1 OPEN /",
            "/",
        ]
        for row in rows:
            current = parsed_date(row["DATE"])
            lines.extend(
                (
                    "DATES",
                    f" {current.day} {current.strftime('%b').upper()} {current.year} /",
                    "/",
                )
            )
            producers = [well for well in wells if float(row[f"WWIR:{well}"]) == 0]
            injectors = [well for well in wells if float(row[f"WWIR:{well}"]) > 0]
            if producers:
                lines.append("WCONPROD")
                lines.extend(
                    f" '{well}' 'OPEN' 'LRAT' 1* 1* 1* {row[f'WLPR:{well}']} /"
                    for well in producers
                )
                lines.append("/")
            if injectors:
                lines.append("WCONINJE")
                lines.extend(
                    f" '{well}' 'WATER' 'OPEN' 'RATE' {row[f'WWIR:{well}']} /"
                    for well in injectors
                )
                lines.append("/")
        lines.append("END")
        (root / "MODEL.DATA").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _export(
        self,
        root: Path,
        summary: Path,
        density: Path,
        *,
        unit_system: str = "METRIC",
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
        chdd, track2, manifest = root / "chdd.csv", root / "track2.csv", root / "manifest.json"
        run_manifest, extraction = self._extraction_chain(root, summary)
        self._schedule_deck(root, summary, unit_system)
        export_opm_chdd(
            summary,
            chdd,
            track2,
            manifest,
            scenario_id="synthetic",
            source_model="synthetic_opm",
            opm_run_manifest=run_manifest,
            summary_extraction_manifest=extraction,
            density_map=density,
            deck_dir=root,
            unit_system=unit_system,
            _summary_run=self._summary_replay(summary),
        )
        with chdd.open(encoding="utf-8", newline="") as stream:
            chdd_rows = list(csv.DictReader(stream))
        with track2.open(encoding="utf-8", newline="") as stream:
            track2_rows = list(csv.DictReader(stream))
        return chdd_rows, track2_rows, json.loads(manifest.read_text(encoding="utf-8"))

    def test_metric_conversion_and_canonical_track2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for month, time, p1_total, i1_total in (
                ("01-JAN-2025", 0, 300.0, 600.0),
                ("01-FEB-2025", 31, 650.0, 1250.0),
            ):
                rows.append(
                    {
                        "DATE": month,
                        "TIME": time,
                        "WLPR:P1": 30.0,
                        "WLPT:P1": p1_total,
                        "WOPR:P1": 10.0,
                        "WOPT:P1": p1_total / 3,
                        "WWIR:P1": 0.0,
                        "WWIT:P1": 0.0,
                        "WBHP:P1": 120.0,
                        "WBP9:P1": 200.0,
                        "WEFF:P1": 1.0,
                        "WLPR:I1": 0.0,
                        "WLPT:I1": 0.0,
                        "WOPR:I1": 0.0,
                        "WOPT:I1": 0.0,
                        "WWIR:I1": 20.0,
                        "WWIT:I1": i1_total,
                        "WBHP:I1": 250.0,
                        "WBP9:I1": 205.0,
                        "WEFF:I1": 1.0,
                    }
                )
            summary = self._summary(root, rows, wells=("P1", "I1"))
            density = self._density(
                root,
                {
                    "P1": {"oil_kg_m3": 850.0, "water_kg_m3": 1000.0},
                    "I1": {"oil_kg_m3": 850.0, "water_kg_m3": 1000.0},
                },
            )
            chdd, track2, manifest = self._export(root, summary, density)

            producer = next(row for row in chdd if row["well"] == "P1")
            self.assertEqual(producer["DATA"], "2025-01-01")
            self.assertAlmostEqual(float(producer["WOMR"]), 8.5)
            self.assertAlmostEqual(float(producer["WOMT_Diff"]), 85.0)
            self.assertAlmostEqual(float(producer["WLPR"]), 30.0)
            self.assertAlmostEqual(float(producer["WLPT_Diff"]), 285.0)
            producer_state = next(row for row in track2 if row["well"] == "P1")
            self.assertAlmostEqual(float(producer_state["liquid_tpd"]), 28.5)
            self.assertEqual(producer_state["control_target"], "LRAT")
            self.assertEqual(manifest["source"]["ignored_summary_columns"], ["TIME"])
            self.assertEqual(manifest["provenance"]["opm_run_manifest"], "opm-run.json")
            self.assertEqual(manifest["provenance"]["opm_source_sha256"], "a" * 64)
            self.assertEqual(
                tuple(track2[0]),
                (
                    "scenario_id",
                    "source_model",
                    "date",
                    "well",
                    "oil_tpd",
                    "liquid_tpd",
                    "pressure_bar",
                    "control_value",
                    "control_target",
                    "status",
                ),
            )

    def test_field_unit_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "TIME": 0,
                "WLPR:P1": 2.0,
                "WOPR:P1": 1.0,
                "WWIR:P1": 0.0,
                "WWIT:P1": 0.0,
                "WBHP:P1": 100.0,
                "WBP9:P1": 200.0,
                "WEFF:P1": 1.0,
            }
            summary = self._summary(
                root,
                [
                    {**base, "DATE": "2025-01-31", "WLPT:P1": 2.0, "WOPT:P1": 1.0},
                    {**base, "DATE": "2025-02-28", "TIME": 31, "WLPT:P1": 4.0, "WOPT:P1": 2.0},
                ],
            )
            density = self._density(
                root, {"P1": {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}}
            )
            chdd, _, _ = self._export(root, summary, density, unit_system="FIELD")
            self.assertAlmostEqual(float(chdd[0]["WLPR"]), 2 * 0.158987294928)
            self.assertAlmostEqual(float(chdd[0]["WOMR"]), 0.158987294928 * 0.8)
            self.assertAlmostEqual(float(chdd[0]["BHP"]), 100 * 0.0689475729318)

    def test_chdd_liquid_rate_is_volume_but_total_is_mass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for date, factor in (("2025-01-01", 1.0), ("2025-02-01", 2.0)):
                row = {
                    "DATE": date,
                    "TIME": 0,
                    **{f"{vector}:P1": 0 for vector in VECTORS},
                }
                row.update(
                    {
                        "WLPR:P1": 500.0,
                        "WLPT:P1": 1000.0 * factor,
                        "WOPR:P1": 100.0,
                        "WOPT:P1": 200.0 * factor,
                        "WEFF:P1": 1.0,
                    }
                )
                rows.append(row)
            summary = self._summary(root, rows)
            density = self._density(
                root, {"P1": {"oil_kg_m3": 832.0, "water_kg_m3": 1012.8}}
            )

            chdd, track2, _ = self._export(root, summary, density)

            expected_total_tonnes = (200.0 * 832.0 + 800.0 * 1012.8) / 1000
            expected_rate_tpd = (100.0 * 832.0 + 400.0 * 1012.8) / 1000
            self.assertEqual(float(chdd[0]["WLPR"]), 500.0)
            self.assertAlmostEqual(float(chdd[0]["WLPT"]), expected_total_tonnes)
            self.assertAlmostEqual(float(chdd[0]["WLPT_Diff"]), expected_total_tonnes)
            self.assertAlmostEqual(float(track2[0]["liquid_tpd"]), expected_rate_tpd)

    def test_2014_liquid_mass_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            liquid_m3 = 1080.64350568
            oil_m3 = 67.71011726449233
            rows = []
            for date, factor in (("2014-01-01", 1.0), ("2014-02-01", 2.0)):
                row = {
                    "DATE": date,
                    "TIME": 0,
                    **{f"{vector}:P1": 0 for vector in VECTORS},
                }
                row.update(
                    {
                        "WLPR:P1": 500.0,
                        "WLPT:P1": liquid_m3 * factor,
                        "WOPR:P1": 50.0,
                        "WOPT:P1": oil_m3 * factor,
                        "WEFF:P1": 1.0,
                    }
                )
                rows.append(row)
            summary = self._summary(root, rows)
            density = self._density(
                root, {"P1": {"oil_kg_m3": 832.0, "water_kg_m3": 1012.8}}
            )

            chdd, _, _ = self._export(root, summary, density)

            self.assertEqual(float(chdd[0]["WLPR"]), 500.0)
            self.assertAlmostEqual(float(chdd[0]["WLPT"]), 1082.233753351284)
            self.assertAlmostEqual(float(chdd[1]["WLPT_Diff"]), 1082.233753351284)

    def test_missing_vector_and_density_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "DATE": date,
                    "TIME": index,
                    **{f"{vector}:P1": 0 for vector in VECTORS},
                }
                for index, date in enumerate(("2025-01-01", "2025-02-01"))
            ]
            rows[0]["WLPR:P1"] = rows[1]["WLPR:P1"] = 1
            rows[0]["WEFF:P1"] = rows[1]["WEFF:P1"] = 1
            summary = self._summary(root, rows)
            text = summary.read_text(encoding="utf-8").replace(",WBP9:P1", "")
            summary.write_text(text, encoding="utf-8")
            density = self._density(
                root, {"OTHER": {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}}
            )
            with self.assertRaisesRegex(OpmChddError, "misses SUMMARY vectors"):
                self._export(root, summary, density)

            summary = self._summary(root, rows)
            with self.assertRaisesRegex(OpmChddError, "missing explicit density"):
                self._export(root, summary, density)

    def test_negative_cumulative_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for date, total in (("2025-01-01", 10.0), ("2025-02-01", 9.0)):
                row = {"DATE": date, "TIME": 0, **{f"{vector}:P1": 0 for vector in VECTORS}}
                row.update({"WLPR:P1": 1, "WLPT:P1": total, "WEFF:P1": 1})
                rows.append(row)
            summary = self._summary(root, rows)
            density = self._density(
                root, {"P1": {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}}
            )
            with self.assertRaisesRegex(OpmChddError, "negative cumulative difference"):
                self._export(root, summary, density)

    def test_run_manifest_source_identity_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for date in ("2025-01-01", "2025-02-01"):
                row = {
                    "DATE": date,
                    "TIME": 0,
                    **{f"{vector}:P1": 0 for vector in VECTORS},
                }
                row.update({"WLPR:P1": 1, "WEFF:P1": 1})
                rows.append(row)
            summary = self._summary(root, rows)
            density = self._density(
                root, {"P1": {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}}
            )
            run_manifest, extraction = self._extraction_chain(
                root, summary, source_sha256="A" * 64
            )
            with self.assertRaisesRegex(OpmChddError, "lowercase SHA-256"):
                export_opm_chdd(
                    summary,
                    root / "chdd.csv",
                    root / "track2.csv",
                    root / "manifest.json",
                    scenario_id="synthetic",
                    source_model="synthetic_opm",
                    opm_run_manifest=run_manifest,
                    summary_extraction_manifest=extraction,
                    density_map=density,
                    unit_system="METRIC",
                )

    def test_arbitrary_summary_cannot_reuse_an_honest_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "DATE": date,
                    "TIME": 0,
                    **{f"{vector}:P1": 0 for vector in VECTORS},
                    "WLPR:P1": 1,
                    "WLPT:P1": total,
                    "WEFF:P1": 1,
                }
                for date, total in (("2025-01-01", 1), ("2025-02-01", 2))
            ]
            summary = self._summary(root, rows)
            density = self._density(
                root, {"P1": {"oil_kg_m3": 800.0, "water_kg_m3": 1000.0}}
            )
            run_manifest, extraction = self._extraction_chain(root, summary)
            replay = self._summary_replay(summary)
            with summary.open("a", encoding="utf-8") as stream:
                stream.write("forged\n")
            proof = json.loads(extraction.read_text(encoding="utf-8"))
            proof["output_report"].update(
                {
                    "bytes": summary.stat().st_size,
                    "sha256": sha256(summary.read_bytes()).hexdigest(),
                }
            )
            extraction.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(OpmChddError, "deterministic replay"):
                export_opm_chdd(
                    summary,
                    root / "chdd.csv",
                    root / "track2.csv",
                    root / "manifest.json",
                    scenario_id="forged",
                    source_model="model_z_opm",
                    opm_run_manifest=run_manifest,
                    summary_extraction_manifest=extraction,
                    density_map=density,
                    unit_system="METRIC",
                    _summary_run=replay,
                )

    def test_official_whitespace_summary_uses_deck_start_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "MODEL.DATA").write_text(
                """RUNSPEC
START
 1 JAN 2025 /
METRIC
DIMENS
 1 1 1 /
TABDIMS
 1 /
PROPS
DENSITY
 800 1000 1 /
REGIONS
PVTNUM
 1 /
ACTNUM
 1 /
SCHEDULE
COMPDAT
 'P1' 1 1 1 1 OPEN /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 1* 1* 1* 1 /
/
END
""",
                encoding="utf-8",
            )
            summary = root / "summary-report.csv"
            fields = ["TIME", *(f"{vector}:P1" for vector in VECTORS), "YEARS"]
            rows = []
            for time, total in ((31, 31), (59, 59)):
                values = {f"{vector}:P1": 0 for vector in VECTORS}
                values.update(
                    {
                        "WLPR:P1": 1,
                        "WLPT:P1": total,
                        "WWPR:P1": 1,
                        "WWPT:P1": total,
                        "WEFF:P1": 1,
                    }
                )
                rows.append([time, *(values[field] for field in fields[1:-1]), time / 365.25])
            summary.write_text(
                "\n" + " ".join(fields) + "\n"
                + "\n".join(" ".join(map(str, row)) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            run_manifest, extraction = self._extraction_chain(
                root, summary, source_sha256="b" * 64
            )
            export_opm_chdd(
                summary,
                root / "chdd.csv",
                root / "track2.csv",
                root / "manifest.json",
                scenario_id="synthetic",
                source_model="synthetic_opm",
                opm_run_manifest=run_manifest,
                summary_extraction_manifest=extraction,
                deck_dir=root,
                _summary_run=self._summary_replay(summary),
            )
            with (root / "chdd.csv").open(encoding="utf-8", newline="") as stream:
                dates = [row["DATA"] for row in csv.DictReader(stream)]
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(dates, ["2025-02-01", "2025-03-01"])
            self.assertEqual(manifest["source"]["ignored_summary_columns"], ["TIME", "YEARS"])

    def test_deck_density_mapping_exposes_multi_region_well(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "MODEL.DATA").write_text(
                """RUNSPEC
START
 1 JAN 2025 /
METRIC
DIMENS
 2 1 1 /
TABDIMS
 2 /
PROPS
DENSITY
 800 1000 1 /
 900 1100 1 /
REGIONS
PVTNUM
 1 2 /
ACTNUM
 2*1 /
SCHEDULE
COMPDAT
 'P1' 1 1 1 1 OPEN /
 'P2' 1 1 1 1 OPEN /
 'P2' 2 1 1 1 OPEN /
/
END
""",
                encoding="utf-8",
            )
            unit, resolved, ambiguous, digest, connections = read_deck_densities(root)
            self.assertEqual(unit, "METRIC")
            self.assertEqual(resolved["P1"].oil_kg_m3, 800)
            self.assertEqual(ambiguous["P2"], (1, 2))
            self.assertEqual(set(connections["P2"]), {"1,1,1", "2,1,1"})
            self.assertEqual(len(digest), 64)

    def test_requested_controls_role_switch_and_zero_phase_without_density(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "MODEL.DATA").write_text(
                """RUNSPEC
START
 1 DEC 2024 /
METRIC
DIMENS
 1 1 1 /
TABDIMS
 1 /
PROPS
DENSITY
 832 1012.8 1 /
REGIONS
PVTNUM
 1 /
ACTNUM
 1 /
SCHEDULE
WELSPECS
 'P1' 'G' 1 1 /
 '59' 'G' 1 1 /
/
COMPDAT
 'P1' 1 1 1 1 OPEN /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 1* 1* 1* 500 /
/
DATES
 1 JAN 2025 /
/
WCONINJE
 'P1' 'WATER' 'OPEN' 'RATE' 200 /
/
DATES
 1 FEB 2025 /
/
WCONPROD
 '59' 'OPEN' 'LRAT' 1* 1* 1* 700 /
/
END
""",
                encoding="utf-8",
            )
            rows = []
            for month in ("2025-01-01", "2025-02-01"):
                row = {
                    "DATE": month,
                    "TIME": 0,
                    **{
                        f"{vector}:{well}": 0
                        for well in ("P1", "59")
                        for vector in VECTORS
                    },
                }
                row["WEFF:P1"] = 1
                if month.endswith("01-01"):
                    row.update(
                        {
                            "WLPR:P1": 350,
                            "WLPT:P1": 350,
                            "WOPR:P1": 100,
                            "WOPT:P1": 100,
                        }
                    )
                else:
                    row.update(
                        {
                            "WLPT:P1": 350,
                            "WOPT:P1": 100,
                            "WWIR:P1": 200,
                            "WWIT:P1": 200,
                        }
                    )
                rows.append(row)
            summary = self._summary(root, rows, wells=("P1", "59"))
            run_manifest, extraction = self._extraction_chain(root, summary)
            chdd = root / "chdd.csv"
            trajectory = root / "track2.csv"
            manifest = root / "manifest.json"

            export_opm_chdd(
                summary,
                chdd,
                trajectory,
                manifest,
                scenario_id="controls",
                source_model="synthetic_opm",
                opm_run_manifest=run_manifest,
                summary_extraction_manifest=extraction,
                deck_dir=root,
                _summary_run=self._summary_replay(summary),
            )

            with trajectory.open(encoding="utf-8", newline="") as stream:
                track2 = list(csv.DictReader(stream))
            p1 = [row for row in track2 if row["well"] == "P1"]
            shut = [row for row in track2 if row["well"] == "59"]
            self.assertEqual(
                [(row["control_target"], float(row["control_value"])) for row in p1],
                [("WRAT", 200.0), ("WRAT", 200.0)],
            )
            self.assertEqual([row["status"] for row in p1], ["1", "1"])
            self.assertGreater(float(p1[0]["liquid_tpd"]), 0)
            self.assertEqual(float(p1[1]["liquid_tpd"]), 0)
            self.assertEqual(
                [(row["control_target"], float(row["control_value"])) for row in shut],
                [("LRAT", 0.0), ("LRAT", 700.0)],
            )
            self.assertEqual([row["status"] for row in shut], ["0", "1"])
            self.assertTrue(all(float(row["liquid_tpd"]) == 0 for row in shut))
            proof = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("future target/status are not used", proof["conversion"]["control"])
            self.assertEqual(
                proof["conversion"]["mass_method_by_well"]["59"],
                "zero_phase_no_density",
            )

            original_first = shut[0]
            deck = root / "MODEL.DATA"
            deck.write_text(
                deck.read_text(encoding="utf-8").replace("1* 700 /", "1* 900 /"),
                encoding="utf-8",
            )
            changed_track2 = root / "changed-track2.csv"
            export_opm_chdd(
                summary,
                root / "changed-chdd.csv",
                changed_track2,
                root / "changed-manifest.json",
                scenario_id="controls",
                source_model="synthetic_opm",
                opm_run_manifest=run_manifest,
                summary_extraction_manifest=extraction,
                deck_dir=root,
                _summary_run=self._summary_replay(summary),
            )
            with changed_track2.open(encoding="utf-8", newline="") as stream:
                changed = [
                    row for row in csv.DictReader(stream) if row["well"] == "59"
                ]
            self.assertEqual(changed[0], original_first)
            self.assertEqual(float(changed[1]["control_value"]), 900.0)

    def test_connection_vectors_use_cell_density(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "MODEL.DATA").write_text(
                """RUNSPEC
START
 1 JAN 2025 /
METRIC
DIMENS
 1 1 2 /
TABDIMS
 2 /
PROPS
DENSITY
 800 1000 1 /
 900 1100 1 /
REGIONS
PVTNUM
 1 2 /
ACTNUM
 2*1 /
SCHEDULE
WELSPECS
 'P1' 'G' 1 1 /
/
COMPDAT
 'P1' 1* 1* 1 2 OPEN /
/
WCONPROD
 'P1' 'OPEN' 'LRAT' 1* 1* 1* 30 /
/
END
""",
                encoding="utf-8",
            )
            connection_fields = tuple(
                f"{vector}:P1:{completion}"
                for completion in ("1,1,1", "1,1,2")
                for vector in CONNECTION_VECTORS
            )
            rows = []
            for month, factor in (("2025-02-01", 1.0), ("2025-03-01", 1.5)):
                row = {
                    "DATE": month,
                    "TIME": 0,
                    "WLPR:P1": 30.0,
                    "WLPT:P1": 300.0 * factor,
                    "WOPR:P1": 9.0,
                    "WOPT:P1": 90.0 * factor,
                    "WOIR:P1": 0.0,
                    "WOIT:P1": 0.0,
                    "WWPR:P1": 21.0,
                    "WWPT:P1": 210.0 * factor,
                    "WWIR:P1": 0.0,
                    "WWIT:P1": 0.0,
                    "WBHP:P1": 120.0,
                    "WBP9:P1": 200.0,
                    "WEFF:P1": 1.0,
                }
                row.update(
                    {
                        "COFR:P1:1,1,1": -1.0,
                        "CWFR:P1:1,1,1": -2.0,
                        "COPR:P1:1,1,1": -1.0,
                        "COPT:P1:1,1,1": -10.0 * factor,
                        "CWPR:P1:1,1,1": -2.0,
                        "CWPT:P1:1,1,1": -20.0 * factor,
                        "COIT:P1:1,1,1": 0.0,
                        "CWIR:P1:1,1,1": 0.0,
                        "CWIT:P1:1,1,1": 0.0,
                        "COFR:P1:1,1,2": 11.0,
                        "CWFR:P1:1,1,2": 22.0,
                        "COPR:P1:1,1,2": 11.0,
                        "COPT:P1:1,1,2": 110.0 * factor,
                        "CWPR:P1:1,1,2": 22.0,
                        "CWPT:P1:1,1,2": 220.0 * factor,
                        "COIT:P1:1,1,2": 0.0,
                        "CWIR:P1:1,1,2": 0.0,
                        "CWIT:P1:1,1,2": 0.0,
                    }
                )
                rows.append(row)
            summary = self._summary(root, rows, extra_fields=connection_fields)
            run_manifest, extraction = self._extraction_chain(root, summary)
            chdd, track2, manifest = root / "chdd.csv", root / "track2.csv", root / "manifest.json"
            export_opm_chdd(
                summary,
                chdd,
                track2,
                manifest,
                scenario_id="connections",
                source_model="synthetic_opm",
                opm_run_manifest=run_manifest,
                summary_extraction_manifest=extraction,
                deck_dir=root,
                _summary_run=self._summary_replay(summary),
            )
            with chdd.open(encoding="utf-8", newline="") as stream:
                first = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(first["WOMR"]), 9.1)
            self.assertAlmostEqual(float(first["WOMT"]), 91.0)
            self.assertAlmostEqual(float(first["WLPR"]), 30.0)
            self.assertAlmostEqual(float(first["WLPT"]), 313.0)
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["conversion"][
                    "mass_method_by_well"
                ]["P1"],
                "connection_surface_vectors",
            )
            self.assertIn(
                "COIR=COPR-COFR",
                json.loads(manifest.read_text(encoding="utf-8"))["provenance"][
                    "raw_opm_manifest_caveats"
                ][0],
            )


if __name__ == "__main__":
    unittest.main()
