from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from timesoil.aios.contracts import (
    Case,
    ControlAction,
    ControlTarget,
    State,
    WellRole,
    WellState,
    WellStatus,
)
from timesoil.aios.economics import CHDD_FIELDS, EconomicResult
from timesoil.aios.opm import (
    OPM_IMAGE,
    OPM_IMAGE_DIGEST,
    OpmCertificationError,
    OpmFlowRunner,
    OpmGdmBackend,
    OpmRunResult,
    _sha256_file,
    _source_digest,
)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    return _sha256_file(path)


def _signed(path: Path, value: object) -> str:
    digest = _write_json(path, value)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return f"{path.resolve()}#sha256={digest}"


def _source(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "MODEL.DATA").write_text(
        "RUNSPEC\nMETRIC\nSUMMARY\nSCHEDULE\n"
        "WELSPECS\n 'P1' 'G' 1 1 /\n 'I1' 'G' 1 1 /\n/\n"
        "INCLUDE\n 'schedule.inc' /\n"
    )
    (source / "schedule.inc").write_text(
        "DATES\n 1 JAN 2014 /\n/\n"
        "DATES\n 1 FEB 2014 /\n/\n"
        "DATES\n 1 MAR 2014 /\n/\n"
        "DATES\n 1 APR 2014 /\n/\nEND\n"
    )
    return source


def _baseline(root: Path, source_sha: str) -> str:
    run = root / "prior" / "baseline"
    output = run / "output"
    output.mkdir(parents=True)
    primary = output / "BASE.SMSPEC"
    companion = output / "BASE.UNSMRY"
    primary.write_bytes(b"smspec")
    companion.write_bytes(b"unsmry")
    artifacts = [
        {
            "path": path.relative_to(run).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in (primary, companion)
    ]
    manifest = {
        "schema": "timesoil.aios.opm-run/v1",
        "status": "success",
        "returncode": 0,
        "source_sha256": source_sha,
        "image_reference": OPM_IMAGE,
        "image_digest": OPM_IMAGE_DIGEST,
        "artifacts": artifacts,
    }
    return _signed(run / "manifest.json", manifest)


def _summary_report() -> str:
    vectors = (
        "WLPR", "WLPT", "WOPR", "WOPT", "WOIR", "WOIT", "WWPR", "WWPT",
        "WWIR", "WWIT", "WBHP", "WBP9", "WEFF",
    )
    headers = ["DATE", *(f"{vector}:{well}" for well in ("P1", "I1") for vector in vectors)]
    producer = (80, 800, 30, 300, 0, 0, 50, 500, 0, 0, 150, 160, 1)
    injector = (0, 0, 0, 0, 0, 0, 0, 0, 120, 1200, 170, 165, 1)
    values = " ".join(str(value) for value in (*producer, *injector))
    return (
        " ".join(headers)
        + "\n"
        + f"2014-02-01 {values}\n"
        + f"2014-03-01 {values}\n"
    )


class _Economics:
    def calculate(self, records, *, start_year: int, output_dir: Path) -> EconomicResult:
        assert list(records)
        assert start_year == 2014
        output_dir.mkdir()
        manifest = output_dir / "manifest.json"
        _write_json(manifest, {"official": True})
        _write_json(output_dir / "result.json", {"total_chdd_m": 42.5})
        (output_dir / "report.xlsx").write_bytes(b"xlsx")
        return EconomicResult(
            42.5,
            1.2,
            "2014-01-01",
            "2014-03-01",
            {},
            output_dir,
            manifest,
        )


def test_wells_at_uses_deck_start_for_whitespace_time_summary(
    tmp_path: Path,
) -> None:
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "MODEL.DATA").write_text(
        "RUNSPEC\nSTART\n 1 JAN 2025 /\nMETRIC\nEND\n", encoding="utf-8"
    )
    report = tmp_path / "summary-report.txt"
    vectors = (
        "WLPR", "WLPT", "WOPR", "WOPT", "WOIR", "WOIT", "WWPR", "WWPT",
        "WWIR", "WWIT", "WBHP", "WBP9", "WEFF",
    )
    fields = [
        "TIME",
        *(f"{vector}:{well}" for well in ("P1", "I1") for vector in vectors),
        "YEARS",
    ]
    producer = (80, 800, 30, 300, 0, 0, 50, 500, 0, 0, 150, 160, 1)
    injector = (0, 0, 0, 0, 0, 0, 0, 0, 120, 1200, 170, 165, 1)
    report.write_text(
        " ".join(fields)
        + "\n31 "
        + " ".join(map(str, (*producer, *injector)))
        + f" {31 / 365.25}\n59 "
        + " ".join(map(str, (*producer, *injector)))
        + f" {59 / 365.25}\n",
        encoding="utf-8",
    )
    case = Case(
        "model-y",
        date(2025, 2, 1),
        date(2025, 2, 1),
        date(2025, 2, 1),
        ("P1",),
        ("I1",),
    )

    wells = OpmGdmBackend._wells_at(
        report, case, date(2025, 2, 1), deck_dir=deck
    )

    assert [(well.well, well.liquid_rate, well.injection_rate) for well in wells] == [
        ("I1", 0.0, 120.0),
        ("P1", 80.0, 0.0),
    ]


def test_full_replay_carries_authenticated_controls_and_emits_lineage(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    source_sha = _source_digest(source)
    baseline_ref = _baseline(tmp_path, source_sha)
    case = Case(
        "model-y",
        date(2014, 1, 1),
        date(2014, 3, 1),
        date(2014, 1, 1),
        ("P1",),
        ("I1",),
    )
    january = ControlAction(
        date(2014, 1, 1),
        "P1",
        WellRole.PRODUCER,
        WellStatus.OPEN,
        ControlTarget.LIQUID_RATE,
        75,
    )
    state = State(
        case.case_id,
        date(2014, 2, 1),
        "pending",
        (
            WellState("P1", WellRole.PRODUCER, True, 20, 70, 0, 150),
            WellState("I1", WellRole.INJECTOR, True, 0, 0, 100, 170),
        ),
    )
    prior_dir = tmp_path / "prior"
    backend = OpmGdmBackend(
        OpmFlowRunner(),
        source,
        runs_dir=tmp_path / "runs",
        deck="MODEL.DATA",
        schedule_include="schedule.inc",
        economics=_Economics(),
    )
    prior_value = {
        "schema": backend._LINEAGE_SCHEMA,
        "status": "certified",
        "provenance": {"mode": "full-replay", "binary_restart": False},
        "case_id": case.case_id,
        "source_sha256": source_sha,
        "prior_restart_ref": baseline_ref,
        "step_actions": [backend._action_value(january)],
        "accepted_actions": [backend._action_value(january)],
        "next_state": backend._state_value(state),
        "artifacts": [
            {
                "purpose": "opm_run_manifest",
                "path": "baseline/manifest.json",
                "sha256": _sha256_file(prior_dir / "baseline" / "manifest.json"),
            }
        ],
    }
    state = State(
        state.case_id,
        state.month,
        _signed(prior_dir / "lineage.json", prior_value),
        state.wells,
    )
    february = (
        ControlAction(
            date(2014, 2, 1),
            "I1",
            WellRole.INJECTOR,
            WellStatus.OPEN,
            ControlTarget.WATER_INJECTION_RATE,
            120,
        ),
        ControlAction(
            date(2014, 2, 1),
            "P1",
            WellRole.PRODUCER,
            WellStatus.OPEN,
            ControlTarget.LIQUID_RATE,
            80,
        ),
    )
    runner = backend.runner

    def run_prepared(prepared, *, parsing_strictness):
        assert parsing_strictness == "strict"
        primary = prepared.output_dir / "MODEL.SMSPEC"
        companion = prepared.output_dir / "MODEL.UNSMRY"
        primary.write_bytes(b"smspec")
        companion.write_bytes(b"unsmry")
        stdout = prepared.run_dir / "stdout.log"
        stderr = prepared.run_dir / "stderr.log"
        stdout.write_text("ok")
        stderr.write_text("")
        now = datetime.now(timezone.utc)
        manifest, digest = runner._write_manifest(
            prepared,
            ["docker", "flow"],
            (),
            status="success",
            returncode=0,
            started_at=now,
            finished_at=now,
            duration_seconds=0,
            container_name="test",
            timeout_cleanup="not-needed",
        )
        return OpmRunResult(
            prepared.run_dir,
            prepared.output_dir,
            prepared.deck_path,
            prepared.summary_overlay_path,
            stdout,
            stderr,
            manifest,
            digest,
            ("docker", "flow"),
            (),
        )

    def extract(result, report_path):
        report = Path(report_path)
        report.write_text(_summary_report())
        extraction = result.run_dir / "summary-extraction.json"
        _write_json(extraction, {"test": True})
        return report, extraction

    def export(_report, chdd, trajectory, manifest, **_kwargs):
        Path(chdd).parent.mkdir(parents=True)
        with Path(chdd).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CHDD_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    field: (
                        "2014-01-01"
                        if field == "DATA"
                        else "P1"
                        if field == "well"
                        else 0
                    )
                    for field in CHDD_FIELDS
                }
            )
        Path(trajectory).write_text("scenario_id\n")
        _write_json(Path(manifest), {"certified": True})
        return {"certified": True}

    with (
        patch.object(runner, "_run_prepared", side_effect=run_prepared),
        patch.object(runner, "extract_summary_report", side_effect=extract),
        patch("timesoil.aios.opm_chdd.export_opm_chdd", side_effect=export),
    ):
        result = backend.run_from_restart(case, state, february)

    assert result.economics.npv_million_rub == 42.5
    assert result.trajectory.next_state.month == date(2014, 3, 1)
    assert result.trajectory.next_state.restart_ref.endswith(
        f"#sha256={_sha256_file(tmp_path / 'runs' / result.trajectory.run_id / 'lineage.json')}"
    )
    run_dir = tmp_path / "runs" / result.trajectory.run_id
    replay = (run_dir / "input" / "schedule.inc").read_text()
    assert replay.count("-- TIMESOIL AIOS OVERRIDE") == 2
    assert "2014-01-01" in replay and "2014-02-01" in replay
    assert "1 MAR 2014" in replay and "1 APR 2014" not in replay
    proof = json.loads((run_dir / "schedule-overlay.json").read_text())
    assert proof["provenance"]["mode"] == "full-replay"
    assert proof["provenance"]["action_count"] == 3
    assert backend._authenticated_history(case, result.trajectory.next_state) == (
        january,
        *february,
    )

    (run_dir / "lineage.json").write_text("{}\n")
    with pytest.raises(OpmCertificationError, match="hash-mismatched"):
        backend._authenticated_history(case, result.trajectory.next_state)
