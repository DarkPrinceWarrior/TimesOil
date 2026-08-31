from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import cast

import numpy as np
import pandas as pd
import pytest

from timesoil.aios.contracts import (
    ControlAction,
    ControlTarget,
    WellRole,
    WellStatus,
)
from timesoil.aios.surrogate import (
    SCHEMA_VERSION,
    STATE_FEATURES,
    ScenarioTrajectory,
    Track2Surrogate,
)
from timesoil.aios.track2 import (
    MODEL_Z_SOURCE_SHA256,
    search_track2_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "search_track2_schedule.py"
SPEC = importlib.util.spec_from_file_location("search_track2_schedule", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


_EXPECTED_SCENARIO_IDS = (
    "baseline",
    *(f"perturbation-{index:03d}" for index in range(1, 10)),
)


def _scenario_batch(
    *,
    baseline_dataset_sha256: str = "d" * 64,
    baseline_manifest_sha256: str = "e" * 64,
) -> dict[str, object]:
    records = [
        {
            "scenario_id": scenario_id,
            "actions_sha256": MODULE._EXPECTED_SCENARIO_ACTIONS_SHA256[scenario_id],
            "dataset_sha256": (
                baseline_dataset_sha256
                if scenario_id == "baseline"
                else sha256(f"dataset:{scenario_id}".encode()).hexdigest()
            ),
            "export_manifest_sha256": (
                baseline_manifest_sha256
                if scenario_id == "baseline"
                else sha256(f"manifest:{scenario_id}".encode()).hexdigest()
            ),
        }
        for scenario_id in _EXPECTED_SCENARIO_IDS
    ]
    return {
        "schema": MODULE._TRAINING_LINEAGE_SCHEMA,
        "scenario_run_schema": MODULE._SCENARIO_RUN_SCHEMA,
        "batch_manifest_sha256": "b" * 64,
        "official_source_sha256": MODEL_Z_SOURCE_SHA256,
        "scenario_index_sha256": MODULE._MODEL_Z_SCENARIO_INDEX_SHA256,
        "scenario_ids": list(_EXPECTED_SCENARIO_IDS),
        "scenarios": records,
    }


class _FakeSurrogate:
    def __init__(self, *, baseline_ood: bool = False) -> None:
        self.calls: list[np.ndarray] = []
        self.baseline_ood = baseline_ood
        self.conformal_level = 0.9
        self.is_calibrated = True
        calibration = {
            "method": "scenario_loso_max_normalized_residual",
            "nominal_coverage": 0.9,
            "scenario_count": 10,
            "scenario_ids": list(_EXPECTED_SCENARIO_IDS),
            "quantile_rank": 10,
            "independent_validation": False,
        }
        self.training_metadata = {
            "model_z_ready": True,
            "source_models": ["model_z_opm"],
            "dataset_hash": "verified-dataset",
            "scenario_ids": calibration["scenario_ids"],
            "conformal_calibration": calibration,
            "scenario_batch": _scenario_batch(),
        }

    def rollout(self, initial_state: np.ndarray, actions: np.ndarray) -> SimpleNamespace:
        self.calls.append(actions.copy())
        producer = np.where(actions[..., 1] != 2.0, actions[..., 0], 0.0)
        oil = 20.0 + producer * 0.1
        liquid = oil + 10.0
        pressure = np.full_like(oil, 150.0)
        mean = np.stack((oil, liquid, pressure), axis=-1)
        std = np.full_like(mean, 0.1)
        ood = np.zeros(len(actions), dtype=bool)
        if self.baseline_ood and len(self.calls) == 1:
            ood[0] = True
        return SimpleNamespace(
            mean=mean,
            std=std,
            interval_half_width=2.0 * std,
            ood_score=np.where(ood, 2.0, 0.1),
            ood=ood,
        )


def _trajectory() -> ScenarioTrajectory:
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    wells = ("I1", "I2", "P1", "P2")
    states = np.empty((6, 4, 3), dtype=float)
    states[..., 0] = 20.0
    states[..., 1] = 35.0
    states[..., 2] = 150.0
    monthly = np.asarray(
        [
            [60.0, 2.0, 1.0],
            [40.0, 2.0, 1.0],
            [100.0, 1.0, 1.0],
            [80.0, 0.0, 1.0],
        ]
    )
    actions = np.repeat(monthly[None, ...], 6, axis=0)
    return ScenarioTrajectory(
        "baseline",
        "model_z_opm",
        dates,
        wells,
        states,
        actions,
    )


def _monthly_injection(actions: tuple[ControlAction, ...]) -> dict[date, float]:
    totals: dict[date, float] = {}
    for action in actions:
        if action.target.value == "WRAT":
            totals[action.month] = totals.get(action.month, 0.0) + action.value
    return totals


def _schedule_source() -> str:
    blocks = []
    for month in pd.date_range("2020-01-01", periods=6, freq="MS"):
        blocks.extend(
            ("DATES", f" 01 {month.strftime('%b').upper()} {month.year} /", "/")
        )
    return "\n".join(blocks) + "\nEND\n"


def _replay_search_bundle(tmp_path: Path) -> SimpleNamespace:
    source = tmp_path / "Model_Z_final_OPM.zip"
    source.write_bytes(b"official source placeholder")
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    source_schedule = _schedule_source().encode()
    action = ControlAction(
        date(2020, 1, 1),
        "P1",
        WellRole.PRODUCER,
        WellStatus.OPEN,
        ControlTarget.OIL_RATE,
        10.0,
    )
    actions = (action,)
    controls = MODULE.schedule_overlay_module._canonical_schedule(actions).encode()
    overlay = MODULE.apply_schedule_overlay(
        source_schedule.decode(), actions, known_wells={"P1"}
    )
    inputs = {
        "schedule_relative_path": "Model_Z/Model_Z_sch.inc",
        "deck_relative_path": "Model_Z/Model_Z.data",
        "source_schedule_sha256": sha256(source_schedule).hexdigest(),
    }
    action_sha = MODULE._actions_sha256(actions)
    lineage_value = {
        "schema": "timesoil.aios.track2-search-lineage/v1",
        "selected_candidate_id": "selected-1",
        "selected_actions_sha256": action_sha,
        "model_z_source_sha256": MODEL_Z_SOURCE_SHA256,
        "deck_relative_path": "Model_Z/Model_Z.data",
        "input_hashes": inputs,
        "selected_actions": [MODULE._action(action)],
        "certified": False,
    }
    lineage = MODULE._json_bytes(lineage_value)
    search = {
        "schema": "timesoil.aios.track2-surrogate-search/v1",
        "selection_only": True,
        "certified": False,
        "model_z_source_sha256": MODEL_Z_SOURCE_SHA256,
        "baseline_scenario_id": "model-z-baseline",
        "start_date": "2020-01-01",
        "horizon_months": 6,
        "search": {"seed": 7},
        "selected": {
            "candidate_id": "selected-1",
            "actions_sha256": action_sha,
            "wells_schedule_sha256": sha256(controls).hexdigest(),
        },
        "inputs": inputs,
        "artifacts": {
            "wells_schedule": "wells_schedule.inc",
            "wells_schedule_sha256": sha256(controls).hexdigest(),
            "modified_schedule": "Model_Z_sch.inc",
            "modified_schedule_sha256": overlay.sha256,
            "lineage": "lineage.json",
            "lineage_sha256": sha256(lineage).hexdigest(),
        },
        "schedule_transformation": {
            "source_schedule_sha256": overlay.source_sha256,
            "controls_sha256": overlay.controls_sha256,
            "output_schedule_sha256": overlay.sha256,
        },
        "final_replay_argv": MODULE._replay_argv(
            source.resolve(),
            search_dir.resolve(),
            (tmp_path / "replay").absolute(),
            MODULE._safe_relative("Model_Z/Model_Z.data", "deck"),
            MODULE._safe_relative(
                "Model_Z/Model_Z_sch.inc", "schedule-relative-path"
            ),
            3600.0,
            "low",
        ),
    }
    search_bytes = MODULE._json_bytes(search)
    (search_dir / "manifest.json").write_bytes(search_bytes)
    (search_dir / "manifest.sha256").write_text(
        sha256(search_bytes).hexdigest() + "\n", encoding="ascii"
    )
    (search_dir / "wells_schedule.inc").write_bytes(controls)
    (search_dir / "Model_Z_sch.inc").write_text(overlay.text, encoding="utf-8")
    (search_dir / "lineage.json").write_bytes(lineage)
    return SimpleNamespace(
        source=source,
        search_dir=search_dir,
        source_schedule=source_schedule,
        controls=controls,
        overlay=overlay.text.encode(),
    )


def _fake_opm_manifest(
    prepared: SimpleNamespace,
    *,
    include_schedule: bool = True,
    schedule_sha256: str | None = None,
    deck: str | None = None,
    deck_sha256: str | None = None,
) -> tuple[Path, str]:
    output = prepared.run_dir / "output"
    output.mkdir()
    smspec = output / "CASE.SMSPEC"
    unsmry = output / "CASE.UNSMRY"
    smspec.write_bytes(b"smspec")
    unsmry.write_bytes(b"unsmry")
    schedule = prepared.input_dir / "Model_Z/Model_Z_sch.inc"
    artifacts = [
        {
            "path": path.relative_to(prepared.run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in (smspec, unsmry)
    ]
    if include_schedule:
        artifacts.append(
            {
                "path": schedule.relative_to(prepared.run_dir).as_posix(),
                "bytes": schedule.stat().st_size,
                "sha256": schedule_sha256 or sha256(schedule.read_bytes()).hexdigest(),
            }
        )
    manifest = prepared.run_dir / "manifest.json"
    manifest.write_bytes(
        MODULE._json_bytes(
            {
                "schema": "timesoil.aios.opm-run/v1",
                "status": "success",
                "returncode": 0,
                "source_sha256": MODEL_Z_SOURCE_SHA256,
                "image": MODULE.opm_module.OPM_IMAGE_TAG,
                "image_digest": MODULE.opm_module.OPM_IMAGE_DIGEST,
                "image_reference": MODULE.opm_module.OPM_IMAGE,
                "deck": deck
                or prepared.deck_path.relative_to(prepared.input_dir).as_posix(),
                "deck_sha256": deck_sha256
                or sha256(prepared.deck_path.read_bytes()).hexdigest(),
                "artifacts": artifacts,
            }
        )
    )
    digest = sha256(manifest.read_bytes()).hexdigest()
    (prepared.run_dir / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )
    return manifest, digest


def _fake_export(
    report: Path,
    chdd_csv: Path,
    trajectory_csv: Path,
    export_manifest: Path,
    **kwargs: object,
) -> dict[str, object]:
    chdd_csv.parent.mkdir(parents=True)
    chdd_csv.write_text(
        "DATA,well,WLPT,WLPR,WOMT,WOMR,WWIR,WWIT,THP,BHP,WEFF,"
        "WLPT_Diff,WOMT_Diff,WWIT_Diff\n"
        "2007-01-01,P1,1,1,1,1,0,0,100,100,1,1,1,0\n",
        encoding="utf-8",
    )
    trajectory_csv.write_text("trajectory\n", encoding="utf-8")
    opm_manifest = cast(Path, kwargs["opm_run_manifest"])
    extraction = cast(Path, kwargs["summary_extraction_manifest"])
    relative = lambda path: Path(
        MODULE.os.path.relpath(path.absolute(), export_manifest.parent.absolute())
    ).as_posix()
    value: dict[str, object] = {
        "schema_version": 1,
        "generator": "timesoil.aios.opm_chdd",
        "provenance": {
            "opm_run_manifest": relative(opm_manifest),
            "opm_run_manifest_sha256": sha256(opm_manifest.read_bytes()).hexdigest(),
            "summary_extraction_manifest": relative(extraction),
            "summary_extraction_manifest_sha256": sha256(
                extraction.read_bytes()
            ).hexdigest(),
        },
        "source": {
            "summary_csv": relative(report),
            "summary_csv_sha256": sha256(report.read_bytes()).hexdigest(),
            "deck_sha256": "d" * 64,
        },
        "outputs": {
            "chdd_csv": {
                "name": chdd_csv.name,
                "sha256": sha256(chdd_csv.read_bytes()).hexdigest(),
            },
            "track2_csv": {
                "name": trajectory_csv.name,
                "sha256": sha256(trajectory_csv.read_bytes()).hexdigest(),
            },
        },
    }
    export_manifest.write_bytes(MODULE._json_bytes(value))
    return value


class _FakeEconomics:
    def __init__(self, root: Path) -> None:
        self.chdd_dir = root / "calculator"
        self.chdd_dir.mkdir(parents=True)
        self.norms_path = root / "norms.xlsx"
        self.norms_path.write_bytes(b"norms")
        sources = {
            "РАСЧЕТ_ЧДД.py": b"runner\n",
            "chdd_model.py": b'VERSION = "fake-1"\n',
            "excel_io.py": b"excel io\n",
        }
        for name, data in sources.items():
            (self.chdd_dir / name).write_bytes(data)

    def calculate(self, records: object, **kwargs: object) -> SimpleNamespace:
        output_dir = cast(Path, kwargs["output_dir"])
        output_dir.mkdir()
        rows = MODULE.economics_module.normalize_chdd_rows(cast(list[dict], records))
        input_path = output_dir / "input.csv"
        MODULE.economics_module._write_csv(input_path, rows)
        raw_result = {
            "startDate": "2007-01-01",
            "maxDate": "2007-01-01",
            "diagnostics": {},
            "summary": {"totalChddM": 123.0, "profitabilityIndex": 1.0},
        }
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(raw_result), encoding="utf-8")
        (output_dir / "report.xlsx").write_bytes(b"report")
        (output_dir / "norms-effective.xlsx").write_bytes(b"norms")
        manifest_value = {
            "schema_version": 1,
            "adapter": "timesoil.aios.CHDDEconomicsAdapter",
            "start_year": 2007,
            "fields": list(MODULE.economics_module.CHDD_FIELDS),
            "row_count": len(rows),
            "input_sha256": sha256(input_path.read_bytes()).hexdigest(),
            "norms_source_sha256": sha256(self.norms_path.read_bytes()).hexdigest(),
            "norms_sha256": sha256(
                (output_dir / "norms-effective.xlsx").read_bytes()
            ).hexdigest(),
            "calculator_sha256": {
                name: sha256((self.chdd_dir / name).read_bytes()).hexdigest()
                for name in MODULE.economics_module._CALCULATOR_FILES
            },
            "calculator_version": "fake-1",
            "result_sha256": sha256(result_path.read_bytes()).hexdigest(),
            "assumption_overrides": {"chargeInitialPump": False},
            "artifacts": {
                "input": "input.csv",
                "result": "result.json",
                "report": "report.xlsx",
                "effective_norms": "norms-effective.xlsx",
            },
            "summary": raw_result["summary"],
        }
        manifest = output_dir / "manifest.json"
        manifest.write_bytes(MODULE._json_bytes(manifest_value))
        return SimpleNamespace(
            output_dir=output_dir,
            manifest_path=manifest,
            total_chdd_m=123.0,
        )


def test_search_consumes_rollout_and_preserves_bounds_and_injection() -> None:
    model = _FakeSurrogate()
    result = search_track2_schedule(
        cast(Track2Surrogate, model),
        _trajectory(),
        start_index=0,
        candidate_count=8,
        seed=7,
    )

    assert len(model.calls) == 8
    assert result.accepted[0].candidate_id == "baseline"
    assert len({item.actions_sha256 for item in result.accepted}) == 8
    assert len({item.wells_schedule_sha256 for item in result.accepted}) == 8
    assert result.certified is False
    assert result.horizon_months == 6
    baseline_injection = {
        (action.month, action.well): action
        for action in result.accepted[0].actions
        if action.target is ControlTarget.WATER_INJECTION_RATE
    }
    for candidate in result.accepted:
        assert {
            (action.month, action.well): action
            for action in candidate.actions
            if action.target is ControlTarget.WATER_INJECTION_RATE
        } == baseline_injection
        assert max(
            action.value for action in candidate.actions if action.target.value == "LRAT"
        ) <= 500.0

    with pytest.raises(ValueError, match="candidate_count"):
        search_track2_schedule(
            cast(Track2Surrogate, model),
            _trajectory(),
            start_index=0,
            candidate_count=501,
        )
    with pytest.raises(ValueError, match="outside the surrogate domain"):
        search_track2_schedule(
            cast(Track2Surrogate, _FakeSurrogate(baseline_ood=True)),
            _trajectory(),
            start_index=0,
            candidate_count=4,
        )


def test_search_cli_writes_uncertified_lineage_and_replay_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_file = model_dir / "model.json"
    model_file.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "ensemble_size": 2}),
        encoding="utf-8",
    )
    model_files = {model_file.name: sha256(model_file.read_bytes()).hexdigest()}
    for member_index in range(2):
        for target in STATE_FEATURES:
            member = model_dir / f"member_{member_index:02d}_{target}.txt"
            member.write_bytes(f"model snapshot {member_index} {target}".encode())
            model_files[member.name] = sha256(member.read_bytes()).hexdigest()
    model_manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": model_files,
        "artifact_hash": sha256(
            json.dumps(
                model_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    model_manifest_path = model_dir / "manifest.json"
    model_manifest_path.write_text(json.dumps(model_manifest), encoding="utf-8")
    dataset = tmp_path / "scenario.csv"
    dataset.write_text("verified dataset placeholder", encoding="utf-8")
    export_manifest = tmp_path / "scenario.json"
    export_manifest.write_text("{}", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    calibration = {
        "method": "scenario_loso_max_normalized_residual",
        "nominal_coverage": 0.9,
        "scenario_count": 10,
        "scenario_ids": list(_EXPECTED_SCENARIO_IDS),
        "quantile_rank": 10,
        "independent_validation": False,
    }
    metrics_payload = {
        "model_z_ready": True,
        "pipeline_proof_only": False,
        "source_models": ["model_z_opm"],
        "dataset_hash": "verified-dataset",
        "surrogate_artifact_hash": model_manifest["artifact_hash"],
        "surrogate_manifest_sha256": sha256(
            model_manifest_path.read_bytes()
        ).hexdigest(),
        "conformal_calibration": calibration,
        "scenario_batch": _scenario_batch(
            baseline_dataset_sha256=sha256(dataset.read_bytes()).hexdigest(),
            baseline_manifest_sha256=sha256(export_manifest.read_bytes()).hexdigest(),
        ),
    }
    metrics.write_text(json.dumps(metrics_payload), encoding="utf-8")
    source = tmp_path / "Model_Z_final_OPM.zip"
    source.write_bytes(b"official source placeholder")
    schedule = tmp_path / "Model_Z_sch.inc"
    schedule.write_text(_schedule_source(), encoding="utf-8")
    output = tmp_path / "search-output"

    class _Verified(list[ScenarioTrajectory]):
        model_z_identity = True

    fake = _FakeSurrogate()
    fake.training_metadata["scenario_batch"] = metrics_payload["scenario_batch"]
    monkeypatch.setattr(MODULE.Track2Surrogate, "load", lambda _: fake)
    monkeypatch.setattr(
        MODULE, "load_trajectory_dataset", lambda *_args, **_kwargs: _Verified([_trajectory()])
    )
    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)
    monkeypatch.setattr(
        MODULE,
        "_actions_sha256",
        lambda _: MODULE._EXPECTED_SCENARIO_ACTIONS_SHA256["baseline"],
    )
    args = argparse.Namespace(
        model=model_dir,
        dataset=dataset,
        export_manifest=export_manifest,
        metrics=metrics,
        source=source,
        schedule=schedule,
        output=output,
        scenario_id="baseline",
        start_date="2020-01-01",
        candidate_count=4,
        seed=11,
        perturbation_fraction=0.05,
        uncertainty_weight=1.0,
        injection_cost_equivalent=0.01,
        deck="Model_Z/Model_Z.data",
        schedule_relative_path="Model_Z/Model_Z_sch.inc",
        timeout_seconds=3600.0,
        parsing_strictness="low",
    )
    original_manifest_bytes = model_manifest_path.read_bytes()
    extra = model_dir / "dummy.bin"
    extra.write_bytes(b"dummy")
    forged_files = {**model_files, extra.name: sha256(extra.read_bytes()).hexdigest()}
    model_manifest_path.write_text(
        json.dumps(
            {
                **model_manifest,
                "files": forged_files,
                "artifact_hash": sha256(
                    json.dumps(
                        forged_files,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact file set"):
        MODULE._search(args)
    assert not output.exists()
    model_manifest_path.write_bytes(original_manifest_bytes)

    fake.training_metadata["scenario_hashes"] = {}
    with pytest.raises(ValueError, match="absent from surrogate training metadata"):
        MODULE._search(args)
    fake.training_metadata["scenario_hashes"] = {"baseline": "0" * 64}
    with pytest.raises(ValueError, match="hash disagrees"):
        MODULE._search(args)
    fake.training_metadata["scenario_hashes"] = {
        "baseline": _trajectory().content_hash
    }
    manifest_bytes_before_swap = model_manifest_path.read_bytes()

    def swap_manifest(_: Path) -> _FakeSurrogate:
        replacement = model_dir / "replacement-manifest.json"
        replacement.write_bytes(manifest_bytes_before_swap)
        replacement.replace(model_manifest_path)
        return fake

    monkeypatch.setattr(MODULE.Track2Surrogate, "load", swap_manifest)
    with pytest.raises(ValueError, match="changed during load"):
        MODULE._search(args)
    assert not output.exists()
    monkeypatch.setattr(MODULE.Track2Surrogate, "load", lambda _: fake)

    dataset_bytes = dataset.read_bytes()

    def swap_dataset(*_args: object, **_kwargs: object) -> _Verified:
        replacement = tmp_path / "replacement-scenario.csv"
        replacement.write_bytes(dataset_bytes)
        replacement.replace(dataset)
        return _Verified([_trajectory()])

    monkeypatch.setattr(MODULE, "load_trajectory_dataset", swap_dataset)
    with pytest.raises(ValueError, match="changed during load or search"):
        MODULE._search(args)
    assert not output.exists()

    export_bytes = export_manifest.read_bytes()

    def mutate_export(*_args: object, **_kwargs: object) -> _Verified:
        export_manifest.write_bytes(export_bytes + b"\n")
        return _Verified([_trajectory()])

    monkeypatch.setattr(MODULE, "load_trajectory_dataset", mutate_export)
    with pytest.raises(ValueError, match="changed during load or search"):
        MODULE._search(args)
    assert not output.exists()
    export_manifest.write_bytes(export_bytes)
    monkeypatch.setattr(
        MODULE,
        "load_trajectory_dataset",
        lambda *_args, **_kwargs: _Verified([_trajectory()]),
    )

    scenario_batch = metrics_payload["scenario_batch"]
    assert isinstance(scenario_batch, dict)
    metrics_payload["scenario_batch"] = {
        **scenario_batch,
        "scenario_index_sha256": "0" * 64,
    }
    metrics.write_text(json.dumps(metrics_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="scenario lineage disagree"):
        MODULE._search(args)
    assert not output.exists()
    metrics_payload["scenario_batch"] = scenario_batch

    mismatched_batch = _scenario_batch(
        baseline_dataset_sha256="0" * 64,
        baseline_manifest_sha256=sha256(export_manifest.read_bytes()).hexdigest(),
    )
    metrics_payload["scenario_batch"] = mismatched_batch
    fake.training_metadata["scenario_batch"] = mismatched_batch
    metrics.write_text(json.dumps(metrics_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory hash disagrees with scenario batch"):
        MODULE._search(args)
    assert not output.exists()
    metrics_payload["scenario_batch"] = scenario_batch
    fake.training_metadata["scenario_batch"] = scenario_batch

    metrics_payload["surrogate_artifact_hash"] = "b" * 64
    metrics.write_text(json.dumps(metrics_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hashes disagree"):
        MODULE._search(args)
    assert not output.exists()
    metrics_payload["surrogate_artifact_hash"] = model_manifest["artifact_hash"]
    metrics.write_text(json.dumps(metrics_payload), encoding="utf-8")
    manifest_path = MODULE._search(args)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    schedule_bytes = (output / "wells_schedule.inc").read_bytes()
    assert manifest["certified"] is False
    assert manifest["selection_only"] is True
    assert lineage["certified"] is False
    assert manifest["score"]["control_scope"] == "producer_controls_only"
    assert manifest["search"]["gates"][0] == (
        "baseline_per_well_injection_controls_frozen"
    )
    assert manifest["inputs"]["scenario_batch_manifest_sha256"] == "b" * 64
    assert manifest["inputs"]["scenario_index_sha256"] == (
        MODULE._MODEL_Z_SCENARIO_INDEX_SHA256
    )
    assert manifest["inputs"]["scenario_actions_sha256"] == (
        MODULE._EXPECTED_SCENARIO_ACTIONS_SHA256["baseline"]
    )
    assert manifest["inputs"]["deck_relative_path"] == "Model_Z/Model_Z.data"
    assert lineage["deck_relative_path"] == "Model_Z/Model_Z.data"
    assert manifest["artifacts"]["wells_schedule_sha256"] == sha256(
        schedule_bytes
    ).hexdigest()
    assert manifest["schedule_transformation"] == {
        "source_schedule_sha256": sha256(schedule.read_bytes()).hexdigest(),
        "controls_sha256": sha256(schedule_bytes).hexdigest(),
        "output_schedule_sha256": sha256(
            (output / "Model_Z_sch.inc").read_bytes()
        ).hexdigest(),
    }
    assert lineage["selected_actions"]
    assert all(
        set(item) == {"path", "sha256"}
        and Path(item["path"]).is_absolute()
        and len(item["sha256"]) == 64
        for item in manifest["execution_sources"].values()
    )
    assert manifest["final_replay_argv"][4] == "replay"
    assert "successful_opm_run_manifest" in manifest["missing_certification_evidence"]
    assert (output / "Model_Z_sch.inc").read_text(encoding="utf-8") != _schedule_source()


def test_final_replay_records_operational_sunk_assets_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _replay_search_bundle(tmp_path)

    class _Runner:
        def __init__(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 3600.0

        def prepare(self, _source: Path, output: Path, *, deck: str) -> SimpleNamespace:
            input_dir = output / "input"
            deck_path = input_dir / deck
            deck_path.parent.mkdir(parents=True)
            deck_path.write_text("RUNSPEC\n", encoding="utf-8")
            schedule = input_dir / "Model_Z/Model_Z_sch.inc"
            schedule.write_bytes(bundle.source_schedule)
            return SimpleNamespace(
                input_dir=input_dir,
                deck_path=deck_path,
                unit_system="METRIC",
                run_dir=output,
            )

        def _run_prepared(
            self, prepared: SimpleNamespace, *, parsing_strictness: str
        ) -> SimpleNamespace:
            assert parsing_strictness == "low"
            manifest, digest = _fake_opm_manifest(prepared)
            return SimpleNamespace(
                run_dir=prepared.run_dir,
                deck_path=prepared.deck_path,
                manifest_path=manifest,
                manifest_sha256=digest,
            )

        def extract_summary_report(
            self, result: SimpleNamespace, report: Path
        ) -> tuple[Path, Path]:
            report.write_text("summary\n", encoding="utf-8")
            extraction = result.run_dir / "summary-extraction.json"
            extraction.write_text("{}\n", encoding="utf-8")
            return report, extraction

    calls: list[dict[str, object]] = []

    class _Economics(_FakeEconomics):
        def calculate(self, records: object, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return super().calculate(records, **kwargs)

    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)
    monkeypatch.setattr(MODULE, "OpmFlowRunner", _Runner)
    monkeypatch.setattr(MODULE, "export_opm_chdd", _fake_export)
    monkeypatch.setattr(
        MODULE.CHDDEconomicsAdapter,
        "from_env",
        lambda: _Economics(tmp_path / "chdd-source"),
    )
    receipt_path = MODULE._replay(
        argparse.Namespace(
            source=bundle.source,
            search_dir=bundle.search_dir,
            output=tmp_path / "replay",
            deck="Model_Z/Model_Z.data",
            schedule_relative_path="Model_Z/Model_Z_sch.inc",
            timeout_seconds=3600.0,
            parsing_strictness="low",
        )
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert calls == [
        {
            "start_year": 2007,
            "output_dir": tmp_path / "replay/economics-2007",
            "charge_initial_pump": False,
        }
    ]
    assert receipt["economics_profile"] == {
        "name": "operational_sunk_assets",
        "charge_initial_pump": False,
        "semantics": (
            "existing ESPs on 2007-01-01 are sunk assets and are not charged again"
        ),
    }
    assert receipt["schema"] == "timesoil.aios.track2-final-replay/v2"
    assert receipt["schedule_transformation"] == {
        "source_schedule_sha256": sha256(bundle.source_schedule).hexdigest(),
        "controls_sha256": sha256(bundle.controls).hexdigest(),
        "output_schedule_sha256": sha256(bundle.overlay).hexdigest(),
    }
    assert receipt["horizon"] == {
        "start_inclusive": "2020-01-01",
        "end_exclusive": "2020-07-01",
        "months": 6,
    }
    assert receipt["search_seed"] == 7
    assert receipt["run_ids"]["opm"].startswith("opm-")
    assert receipt["run_ids"]["economics"].startswith("economics-")
    assert receipt["scenario"] == {
        "baseline_id": "model-z-baseline",
        "selected_candidate_id": "selected-1",
        "replay_id": "track2-selected-final-replay",
    }
    assert receipt["simulator"] == {
        "image": MODULE.opm_module.OPM_IMAGE_TAG,
        "image_digest": MODULE.opm_module.OPM_IMAGE_DIGEST,
        "image_reference": MODULE.opm_module.OPM_IMAGE,
        "deck": "Model_Z/Model_Z.data",
        "deck_sha256": sha256(b"RUNSPEC\n").hexdigest(),
    }
    required = {
        "search_manifest",
        "lineage",
        "submitted_wells_schedule",
        "replay_overlay",
        "opm_run_manifest",
        "exact_opm_input_schedule",
        "summary_report",
        "summary_extraction",
        "canonical_export_manifest",
        "chdd_csv",
        "trajectory_csv",
        "economics_manifest",
        "economics_input",
        "economics_result",
        "economics_report",
        "economics_effective_norms",
        "economics_norms_source",
        "economics_calculator_main",
        "economics_calculator_model",
        "economics_calculator_excel_io",
    }
    assert set(receipt["artifacts"]) == required
    assert all(
        set(item) == {"path", "sha256"}
        and Path(item["path"]).is_absolute()
        and len(item["sha256"]) == 64
        for item in receipt["artifacts"].values()
    )
    assert receipt["artifacts"]["submitted_wells_schedule"]["sha256"] == sha256(
        bundle.controls
    ).hexdigest()
    assert receipt["artifacts"]["replay_overlay"]["sha256"] == receipt["artifacts"][
        "exact_opm_input_schedule"
    ]["sha256"]
    assert set(receipt["execution_sources"]) == set(MODULE._EXECUTION_SOURCE_PATHS)


@pytest.mark.parametrize(
    "mode",
    [
        "export-empty",
        "export-forged",
        "economics-empty",
        "economics-forged",
        "economics-provenance",
    ],
)
def test_final_replay_rejects_unauthenticated_downstream_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    bundle = _replay_search_bundle(tmp_path)
    stage = mode.split("-", 1)[0]

    class _Runner:
        def __init__(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 3600.0

        def prepare(self, _source: Path, output: Path, *, deck: str) -> SimpleNamespace:
            input_dir = output / "input"
            deck_path = input_dir / deck
            deck_path.parent.mkdir(parents=True)
            deck_path.write_text("RUNSPEC\n", encoding="utf-8")
            (input_dir / "Model_Z/Model_Z_sch.inc").write_bytes(
                bundle.source_schedule
            )
            return SimpleNamespace(
                input_dir=input_dir,
                deck_path=deck_path,
                unit_system="METRIC",
                run_dir=output,
            )

        def _run_prepared(
            self, prepared: SimpleNamespace, *, parsing_strictness: str
        ) -> SimpleNamespace:
            assert parsing_strictness == "low"
            manifest, digest = _fake_opm_manifest(prepared)
            return SimpleNamespace(
                run_dir=prepared.run_dir,
                deck_path=prepared.deck_path,
                manifest_path=manifest,
                manifest_sha256=digest,
            )

        def extract_summary_report(
            self, result: SimpleNamespace, report: Path
        ) -> tuple[Path, Path]:
            report.write_text("summary\n", encoding="utf-8")
            extraction = result.run_dir / "summary-extraction.json"
            extraction.write_text("{}\n", encoding="utf-8")
            return report, extraction

    def empty_export(
        _report: Path,
        chdd_csv: Path,
        trajectory_csv: Path,
        export_manifest: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        chdd_csv.parent.mkdir(parents=True)
        chdd_csv.write_text("DATA\n", encoding="utf-8")
        trajectory_csv.write_text("trajectory\n", encoding="utf-8")
        export_manifest.write_text("{}\n", encoding="utf-8")
        return {}

    def forged_export(*args: object, **kwargs: object) -> dict[str, object]:
        value = _fake_export(*args, **kwargs)
        cast(dict[str, object], value["source"])["summary_csv_sha256"] = "0" * 64
        cast(Path, args[3]).write_bytes(MODULE._json_bytes(value))
        return value

    class _EmptyEconomics(_FakeEconomics):
        def calculate(self, records: object, **kwargs: object) -> SimpleNamespace:
            result = super().calculate(records, **kwargs)
            result.manifest_path.write_text("{}\n", encoding="utf-8")
            return result

    class _ForgedEconomics(_FakeEconomics):
        def calculate(self, records: object, **kwargs: object) -> SimpleNamespace:
            result = super().calculate(records, **kwargs)
            value = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            value["input_sha256"] = "0" * 64
            result.manifest_path.write_bytes(MODULE._json_bytes(value))
            return result

    class _ForgedProvenanceEconomics(_FakeEconomics):
        def calculate(self, records: object, **kwargs: object) -> SimpleNamespace:
            result = super().calculate(records, **kwargs)
            value = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            value["calculator_sha256"]["chdd_model.py"] = "0" * 64
            result.manifest_path.write_bytes(MODULE._json_bytes(value))
            return result

    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)
    monkeypatch.setattr(MODULE, "OpmFlowRunner", _Runner)
    exports = {
        "export-empty": empty_export,
        "export-forged": forged_export,
    }
    monkeypatch.setattr(MODULE, "export_opm_chdd", exports.get(mode, _fake_export))
    economics = {
        "economics-empty": _EmptyEconomics,
        "economics-forged": _ForgedEconomics,
        "economics-provenance": _ForgedProvenanceEconomics,
    }
    monkeypatch.setattr(
        MODULE.CHDDEconomicsAdapter,
        "from_env",
        lambda: economics.get(mode, _FakeEconomics)(tmp_path / "chdd-source"),
    )
    error = (
        "official CHDD input"
        if mode == "economics-forged"
        else "economics provenance"
        if mode == "economics-provenance"
        else f"{stage} manifest"
    )
    with pytest.raises(ValueError, match=error):
        MODULE._replay(
            argparse.Namespace(
                source=bundle.source,
                search_dir=bundle.search_dir,
                output=tmp_path / "replay",
                deck="Model_Z/Model_Z.data",
                schedule_relative_path="Model_Z/Model_Z_sch.inc",
                timeout_seconds=3600.0,
                parsing_strictness="low",
            )
        )
    assert not (tmp_path / "replay/final-replay-receipt.json").exists()


@pytest.mark.parametrize("target", ["lineage", "calculator"])
def test_final_replay_rejects_long_replay_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    bundle = _replay_search_bundle(tmp_path)

    class _Runner:
        def __init__(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 3600.0

        def prepare(self, _source: Path, output: Path, *, deck: str) -> SimpleNamespace:
            input_dir = output / "input"
            deck_path = input_dir / deck
            deck_path.parent.mkdir(parents=True)
            deck_path.write_text("RUNSPEC\n", encoding="utf-8")
            (input_dir / "Model_Z/Model_Z_sch.inc").write_bytes(
                bundle.source_schedule
            )
            return SimpleNamespace(
                input_dir=input_dir,
                deck_path=deck_path,
                unit_system="METRIC",
                run_dir=output,
            )

        def _run_prepared(
            self, prepared: SimpleNamespace, *, parsing_strictness: str
        ) -> SimpleNamespace:
            assert parsing_strictness == "low"
            manifest, digest = _fake_opm_manifest(prepared)
            if target == "lineage":
                lineage = bundle.search_dir / "lineage.json"
                lineage.write_bytes(lineage.read_bytes() + b"\n")
            return SimpleNamespace(
                run_dir=prepared.run_dir,
                deck_path=prepared.deck_path,
                manifest_path=manifest,
                manifest_sha256=digest,
            )

        def extract_summary_report(
            self, result: SimpleNamespace, report: Path
        ) -> tuple[Path, Path]:
            report.write_text("summary\n", encoding="utf-8")
            extraction = result.run_dir / "summary-extraction.json"
            extraction.write_text("{}\n", encoding="utf-8")
            return report, extraction

    adapter = _FakeEconomics(tmp_path / "chdd-source")
    real_artifact = MODULE._artifact
    mutated = False

    def mutate_after_initial_auth(
        path: Path,
        label: str,
        *,
        limit: int = MODULE._MAX_REGULAR_FILE_BYTES,
    ) -> dict[str, str]:
        nonlocal mutated
        artifact = real_artifact(path, label, limit=limit)
        if target == "calculator" and label == "canonical CHDD CSV" and not mutated:
            (adapter.chdd_dir / "chdd_model.py").write_text(
                'VERSION = "mutated"\n', encoding="utf-8"
            )
            mutated = True
        return artifact

    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)
    monkeypatch.setattr(MODULE, "OpmFlowRunner", _Runner)
    monkeypatch.setattr(MODULE, "export_opm_chdd", _fake_export)
    monkeypatch.setattr(MODULE, "_artifact", mutate_after_initial_auth)
    monkeypatch.setattr(
        MODULE.CHDDEconomicsAdapter,
        "from_env",
        lambda: adapter,
    )
    error = "final replay artifacts changed" if target == "lineage" else "source files changed"
    with pytest.raises(ValueError, match=error):
        MODULE._replay(
            argparse.Namespace(
                source=bundle.source,
                search_dir=bundle.search_dir,
                output=tmp_path / "replay",
                deck="Model_Z/Model_Z.data",
                schedule_relative_path="Model_Z/Model_Z_sch.inc",
                timeout_seconds=3600.0,
                parsing_strictness="low",
            )
        )
    assert not (tmp_path / "replay/final-replay-receipt.json").exists()


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("missing-record", "misses exact prepared input schedule"),
        ("tampered-hash", "exact input schedule hash mismatch"),
        ("missing-file", "must be a regular"),
    ],
)
def test_final_replay_rejects_unproven_opm_input_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    error: str,
) -> None:
    bundle = _replay_search_bundle(tmp_path)

    class _Runner:
        def __init__(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 3600.0

        def prepare(self, _source: Path, output: Path, *, deck: str) -> SimpleNamespace:
            input_dir = output / "input"
            deck_path = input_dir / deck
            deck_path.parent.mkdir(parents=True)
            deck_path.write_text("RUNSPEC\n", encoding="utf-8")
            schedule = input_dir / "Model_Z/Model_Z_sch.inc"
            schedule.write_bytes(bundle.source_schedule)
            return SimpleNamespace(
                input_dir=input_dir,
                deck_path=deck_path,
                unit_system="METRIC",
                run_dir=output,
            )

        def _run_prepared(
            self, prepared: SimpleNamespace, *, parsing_strictness: str
        ) -> SimpleNamespace:
            assert parsing_strictness == "low"
            manifest, digest = _fake_opm_manifest(
                prepared,
                include_schedule=mode != "missing-record",
                schedule_sha256="0" * 64 if mode == "tampered-hash" else None,
            )
            if mode == "missing-file":
                (prepared.input_dir / "Model_Z/Model_Z_sch.inc").unlink()
            return SimpleNamespace(
                run_dir=prepared.run_dir,
                deck_path=prepared.deck_path,
                manifest_path=manifest,
                manifest_sha256=digest,
            )

        def extract_summary_report(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail("SUMMARY extraction started before exact OPM input verification")

    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)
    monkeypatch.setattr(MODULE, "OpmFlowRunner", _Runner)
    with pytest.raises(ValueError, match=error):
        MODULE._replay(
            argparse.Namespace(
                source=bundle.source,
                search_dir=bundle.search_dir,
                output=tmp_path / "replay",
                deck="Model_Z/Model_Z.data",
                schedule_relative_path="Model_Z/Model_Z_sch.inc",
                timeout_seconds=3600.0,
                parsing_strictness="low",
            )
        )
    assert not (tmp_path / "replay/final-replay-receipt.json").exists()


@pytest.mark.parametrize("mismatch", ["path", "hash"])
def test_final_replay_rejects_opm_deck_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    bundle = _replay_search_bundle(tmp_path)

    class _Runner:
        def __init__(self, *, timeout_seconds: float) -> None:
            assert timeout_seconds == 3600.0

        def prepare(self, _source: Path, output: Path, *, deck: str) -> SimpleNamespace:
            input_dir = output / "input"
            deck_path = input_dir / deck
            deck_path.parent.mkdir(parents=True)
            deck_path.write_text("RUNSPEC\n", encoding="utf-8")
            schedule = input_dir / "Model_Z/Model_Z_sch.inc"
            schedule.write_bytes(bundle.source_schedule)
            return SimpleNamespace(
                input_dir=input_dir,
                deck_path=deck_path,
                unit_system="METRIC",
                run_dir=output,
            )

        def _run_prepared(
            self, prepared: SimpleNamespace, *, parsing_strictness: str
        ) -> SimpleNamespace:
            assert parsing_strictness == "low"
            manifest, digest = _fake_opm_manifest(
                prepared,
                deck="Model_Z/other.data" if mismatch == "path" else None,
                deck_sha256="0" * 64 if mismatch == "hash" else None,
            )
            return SimpleNamespace(
                run_dir=prepared.run_dir,
                deck_path=prepared.deck_path,
                manifest_path=manifest,
                manifest_sha256=digest,
            )

        def extract_summary_report(self, *_args: object, **_kwargs: object) -> object:
            pytest.fail("SUMMARY extraction started before deck verification")

    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)
    monkeypatch.setattr(MODULE, "OpmFlowRunner", _Runner)
    with pytest.raises(ValueError, match=f"deck {mismatch}"):
        MODULE._replay(
            argparse.Namespace(
                source=bundle.source,
                search_dir=bundle.search_dir,
                output=tmp_path / "replay",
                deck="Model_Z/Model_Z.data",
                schedule_relative_path="Model_Z/Model_Z_sch.inc",
                timeout_seconds=3600.0,
                parsing_strictness="low",
            )
        )


def test_execution_source_snapshot_fails_closed_on_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "governing.py"
    source.write_text("before\n", encoding="utf-8")
    monkeypatch.setattr(MODULE, "_EXECUTION_SOURCE_PATHS", {"governing": source})
    snapshot = MODULE._execution_snapshot()
    source.write_text("after\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="executed Python source changed"):
        MODULE._verify_execution_snapshot(snapshot)


def test_regular_snapshot_reads_opened_inode_and_detects_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "input.csv"
    replacement = tmp_path / "replacement.csv"
    target.write_bytes(b"trusted\n")
    replacement.write_bytes(b"swapped\n")
    real_open = MODULE.os.open
    swapped = False

    def swap_after_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is not None and path == target.name and not swapped:
            replacement.replace(target)
            swapped = True
        return descriptor

    monkeypatch.setattr(MODULE.os, "open", swap_after_open)
    snapshot = MODULE._regular_snapshot(target, "input")

    assert snapshot[0] == b"trusted\n"
    with pytest.raises(ValueError, match="changed during load or search"):
        MODULE._verify_regular_snapshot(target, "input", snapshot)


def test_regular_snapshot_bound_covers_observed_model_z_summary(tmp_path: Path) -> None:
    observed_summary_bytes = 88_598_498

    assert MODULE._MAX_REGULAR_FILE_BYTES == 64 * 1024**2
    assert MODULE._MAX_SUMMARY_REPORT_BYTES == 128 * 1024**2
    assert (
        MODULE._MAX_REGULAR_FILE_BYTES
        < observed_summary_bytes
        <= MODULE._MAX_SUMMARY_REPORT_BYTES
    )
    assert (
        MODULE._regular_snapshot.__kwdefaults__["limit"]
        == MODULE._MAX_REGULAR_FILE_BYTES
    )
    assert (
        MODULE._regular_bytes.__kwdefaults__["limit"]
        == MODULE._MAX_REGULAR_FILE_BYTES
    )

    oversized = tmp_path / "oversized-summary.txt"
    with oversized.open("wb") as stream:
        stream.truncate(MODULE._MAX_SUMMARY_REPORT_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds 134217728 bytes"):
        MODULE._regular_bytes(
            oversized,
            "SUMMARY report",
            limit=MODULE._MAX_SUMMARY_REPORT_BYTES,
        )


def test_regular_snapshot_rejects_torn_read_with_stable_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "input.csv"
    target.write_bytes(b"trusted\n")
    real_read = MODULE.os.read
    read_count = 0

    def torn_first_pass(descriptor: int, size: int) -> bytes:
        nonlocal read_count
        read_count += 1
        if read_count == 1:
            assert size == len(b"trusted\n")
            return b"forged!\n"
        if read_count == 2:
            return b""
        return real_read(descriptor, size)

    monkeypatch.setattr(MODULE.os, "read", torn_first_pass)
    with pytest.raises(ValueError, match="changed while it was read"):
        MODULE._regular_bytes(target, "input")


@pytest.mark.parametrize("mismatch", ["actions", "wells"])
def test_final_replay_rejects_forged_selected_hashes_before_opm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    bundle = _replay_search_bundle(tmp_path)
    search_path = bundle.search_dir / "manifest.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    if mismatch == "actions":
        lineage_path = bundle.search_dir / "lineage.json"
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        lineage["selected_actions_sha256"] = "a" * 64
        search["selected"]["actions_sha256"] = "a" * 64
        lineage_bytes = MODULE._json_bytes(lineage)
        lineage_path.write_bytes(lineage_bytes)
        search["artifacts"]["lineage_sha256"] = sha256(lineage_bytes).hexdigest()
        error = "selected actions canonical hash mismatch"
    else:
        search["selected"]["wells_schedule_sha256"] = "b" * 64
        error = "selected wells schedule disagrees"
    search_bytes = MODULE._json_bytes(search)
    search_path.write_bytes(search_bytes)
    (bundle.search_dir / "manifest.sha256").write_text(
        sha256(search_bytes).hexdigest() + "\n", encoding="ascii"
    )

    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)

    class _NoRunner:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("OPM runner initialized before selected hash verification")

    monkeypatch.setattr(MODULE, "OpmFlowRunner", _NoRunner)
    with pytest.raises(ValueError, match=error):
        MODULE._replay(
            argparse.Namespace(
                source=bundle.source,
                search_dir=bundle.search_dir,
                output=tmp_path / "replay",
                deck="Model_Z/Model_Z.data",
                schedule_relative_path="Model_Z/Model_Z_sch.inc",
                timeout_seconds=3600.0,
                parsing_strictness="low",
            )
        )


def test_final_replay_requires_exact_recorded_arguments_before_opm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _replay_search_bundle(tmp_path)
    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)

    class _NoRunner:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("OPM runner initialized before replay argument verification")

    monkeypatch.setattr(MODULE, "OpmFlowRunner", _NoRunner)
    with pytest.raises(ValueError, match="replay arguments disagree"):
        MODULE._replay(
            argparse.Namespace(
                source=bundle.source,
                search_dir=bundle.search_dir,
                output=tmp_path / "different-replay",
                deck="Model_Z/Model_Z.data",
                schedule_relative_path="Model_Z/Model_Z_sch.inc",
                timeout_seconds=3600.0,
                parsing_strictness="low",
            )
        )


@pytest.mark.parametrize(
    ("artifact_path", "artifact_bytes", "error"),
    [
        ("wells_schedule.inc", None, "must be a regular"),
        ("wells_schedule.inc", b"tampered\n", "hash mismatch"),
        ("../wells_schedule.inc", b"expected\n", "unsafe wells schedule artifact"),
    ],
)
def test_final_replay_rejects_untrusted_wells_schedule_before_opm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_path: str,
    artifact_bytes: bytes | None,
    error: str,
) -> None:
    source = tmp_path / "Model_Z_final_OPM.zip"
    source.write_bytes(b"official source placeholder")
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    expected = b"expected\n"
    search = {
        "schema": "timesoil.aios.track2-surrogate-search/v1",
        "selection_only": True,
        "certified": False,
        "model_z_source_sha256": MODEL_Z_SOURCE_SHA256,
        "inputs": {
            "schedule_relative_path": "Model_Z/Model_Z_sch.inc",
            "deck_relative_path": "Model_Z/Model_Z.data",
        },
        "artifacts": {
            "wells_schedule": artifact_path,
            "wells_schedule_sha256": sha256(expected).hexdigest(),
        },
        "final_replay_argv": MODULE._replay_argv(
            source.resolve(),
            search_dir.resolve(),
            (tmp_path / "replay").absolute(),
            MODULE._safe_relative("Model_Z/Model_Z.data", "deck"),
            MODULE._safe_relative(
                "Model_Z/Model_Z_sch.inc", "schedule-relative-path"
            ),
            3600.0,
            "low",
        ),
    }
    search_bytes = MODULE._json_bytes(search)
    (search_dir / "manifest.json").write_bytes(search_bytes)
    (search_dir / "manifest.sha256").write_text(
        sha256(search_bytes).hexdigest() + "\n", encoding="ascii"
    )
    if artifact_bytes is not None:
        (search_dir / "wells_schedule.inc").write_bytes(artifact_bytes)

    monkeypatch.setattr(MODULE, "_source_digest", lambda _: MODEL_Z_SOURCE_SHA256)

    class _NoRunner:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("OPM runner initialized before wells schedule verification")

    monkeypatch.setattr(MODULE, "OpmFlowRunner", _NoRunner)
    with pytest.raises(ValueError, match=error):
        MODULE._replay(
            argparse.Namespace(
                source=source,
                search_dir=search_dir,
                output=tmp_path / "replay",
                deck="Model_Z/Model_Z.data",
                schedule_relative_path="Model_Z/Model_Z_sch.inc",
                timeout_seconds=3600.0,
                parsing_strictness="low",
            )
        )
