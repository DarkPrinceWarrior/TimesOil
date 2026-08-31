from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_track2_scenarios.py"
CANONICAL_FIELDS = [
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
]


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "model-z-baseline-controls.csv"
    fields = ["date", "well", "control_value", "control_target", "status"]
    rows = [
        ("2026-01-01", "P1", 200, "LRAT", "OPEN"),
        ("2026-01-01", "P2", 0, "ORAT", "SHUT"),
        ("2026-01-01", "I1", 150, "WRAT", "OPEN"),
        ("2026-02-01", "P1", 80, "WRAT", "OPEN"),
        ("2026-02-01", "P2", 110, "LRAT", "OPEN"),
        ("2026-02-01", "I1", 150, "WRAT", "OPEN"),
    ]
    with baseline.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)

    schedule = tmp_path / "schedule.inc"
    schedule.write_text(
        """DATES
  1 JAN 2026 /
/
WCONPROD
  'P1' 'OPEN' 'LRAT' 3* 200 /
  'P2' 'SHUT' 'ORAT' 0 /
/
WCONINJE
  'I1' 'WATER' 'OPEN' 'RATE' 150 /
/
DATES
  1 FEB 2026 /
/
WCONPROD
  'P2' 'OPEN' 'LRAT' 3* 110 /
/
WCONINJE
  'P1' 'WATER' 'OPEN' 'RATE' 80 /
  'I1' 'WATER' 'OPEN' 'RATE' 150 /
/
END
""",
        encoding="utf-8",
    )
    return baseline, schedule


def _canonical_baseline(baseline: Path) -> Path:
    canonical = baseline.with_name("canonical-model-z.csv")
    with baseline.open(encoding="utf-8", newline="") as source, canonical.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, CANONICAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in csv.DictReader(source):
            writer.writerow(
                {
                    "scenario_id": "model-z-baseline",
                    "source_model": "model_z_opm",
                    "date": row["date"],
                    "well": row["well"],
                    "oil_tpd": 10,
                    "liquid_tpd": 20,
                    "pressure_bar": 100,
                    "control_value": row["control_value"],
                    "control_target": row["control_target"],
                    "status": 1.0 if row["status"] == "OPEN" else 0.0,
                }
            )
    return canonical


def _run(baseline: Path, schedule: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(baseline), str(schedule), str(output), "--seed", "9"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_creates_deterministic_hashed_role_switching_inputs(tmp_path: Path) -> None:
    baseline, schedule = _inputs(tmp_path)
    first, second = tmp_path / "out-1", tmp_path / "out-2"
    assert _run(baseline, schedule, first).returncode == 0
    assert _run(baseline, schedule, second).returncode == 0

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    assert len(first_files) == 13
    assert first_files[Path("baseline/schedule.inc")] == schedule.read_bytes()

    index = json.loads(first_files[Path("index.json")])
    assert index["scenario_count"] == 4
    assert len({item["actions_sha256"] for item in index["scenarios"]}) == 4
    for item in index["scenarios"]:
        manifest_path = Path(item["manifest"])
        assert sha256(first_files[manifest_path]).hexdigest() == item["manifest_sha256"]
        manifest = json.loads(first_files[manifest_path])
        scenario = manifest_path.parent
        assert manifest["overlay"]["mode"] == (
            "identity" if scenario.name == "baseline" else "full"
        )
        for artifact in manifest["artifacts"].values():
            data = first_files[scenario / artifact["path"]]
            assert sha256(data).hexdigest() == artifact["sha256"]
        controls = first_files[scenario / "wells_schedule.inc"].decode()
        assert controls.count("'P1'") == 2
        assert "'P1' 'OPEN' 'LRAT'" in controls
        assert "'P1' 'WATER' 'OPEN' 'RATE'" in controls
        assert not ((first / scenario / "wells_schedule.inc").stat().st_mode & 0o222)
        assert not ((first / scenario / "schedule.inc").stat().st_mode & 0o222)


def test_canonical_model_z_and_control_csv_publish_identical_bytes(tmp_path: Path) -> None:
    baseline, schedule = _inputs(tmp_path)
    canonical = _canonical_baseline(baseline)
    canonical_before = canonical.read_bytes()
    control_output, canonical_output = tmp_path / "control", tmp_path / "canonical"

    assert _run(baseline, schedule, control_output).returncode == 0
    assert _run(canonical, schedule, canonical_output).returncode == 0
    control_files = {
        path.relative_to(control_output): path.read_bytes()
        for path in control_output.rglob("*")
        if path.is_file()
    }
    canonical_files = {
        path.relative_to(canonical_output): path.read_bytes()
        for path in canonical_output.rglob("*")
        if path.is_file()
    }

    assert canonical_files == control_files
    assert sha256(canonical_files[Path("index.json")]).hexdigest() == sha256(
        control_files[Path("index.json")]
    ).hexdigest()
    index = json.loads(canonical_files[Path("index.json")])
    assert index["inputs"]["baseline_csv"] == {
        "name": "model-z-baseline-controls.csv",
        "sha256": sha256(baseline.read_bytes()).hexdigest(),
    }
    assert canonical.read_bytes() == canonical_before


def test_cli_rejects_overwrite_malformed_input_and_symlink(tmp_path: Path) -> None:
    baseline, schedule = _inputs(tmp_path)
    output = tmp_path / "out"
    assert _run(baseline, schedule, output).returncode == 0
    before = (output / "index.json").read_bytes()
    repeated = _run(baseline, schedule, output)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
    assert (output / "index.json").read_bytes() == before

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("date,well\n2026-01-01,P1\n", encoding="utf-8")
    bad_output = tmp_path / "bad-output"
    bad = _run(malformed, schedule, bad_output)
    assert bad.returncode != 0
    assert not bad_output.exists()

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-output"
    link.symlink_to(target, target_is_directory=True)
    linked = _run(baseline, schedule, link)
    assert linked.returncode != 0
    assert not tuple(target.iterdir())
