"""Verify a completed lifecycle run, its committed history and a paired full-period baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from run_track1_mpc import _action_payload, _next_month, _state_payload, build_backend, load_config
from timesoil.aios.schedule import ScheduleCompiler


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(root, expected_months, *, run_dir=None, baseline_run=None):
    config = load_config(root / "case.json")
    backend = build_backend(config)
    months = sorted(config.candidates)
    assert len(months) == expected_months
    run = run_dir if run_dir is not None else root / "delivery" / config.run_id
    manifest = json.loads((run / "manifest.json").read_text())
    assert digest(run / "manifest.json") == (run / "manifest.sha256").read_text().split()[0]
    for item in manifest["artifacts"].values():
        assert digest(run / item["path"]) == item["sha256"]
    result = json.loads((run / "result.json").read_text())
    assert result["horizon_protocol"]["planning"] == "full_remaining_period"
    assert result["source_sha256"] == config.source_sha256
    actions = result["schedule"]["actions"]
    wells = set(config.case.producers + config.case.injectors)
    assert Counter(a["month"] for a in actions) == {m.isoformat(): len(wells) for m in months}
    assert len({(a["month"], a["well"]) for a in actions}) == len(actions)
    assert {a["well"] for a in actions} == wells
    assert digest(run / "wells_schedule.inc") == result["schedule"]["sha256"]
    assert (run / "wells_schedule.inc").read_text() == result["schedule"]["text"]
    trajectories = result["evidence"]["trajectories"]
    assert len(trajectories) == expected_months
    verified_files = 0
    lineages = []
    previous_state = _state_payload(config.initial_state)
    for month, trajectory in zip(months, trajectories, strict=True):
        assert trajectory["month"] == month.isoformat()
        assert trajectory["certified"] and trajectory["chdd_complete"] and not trajectory["invariant_violations"]
        ref, expected = trajectory["next_state"]["restart_ref"].split("#sha256=")
        path = Path(ref)
        assert digest(path) == expected
        backend._verify_opm_manifest(path.parent / "manifest.json", baseline=False)
        lineage = json.loads(path.read_text())
        assert lineage["input_state"] == {k: v for k, v in previous_state.items() if k != "restart_ref"}
        assert lineage["prior_restart_ref"] == previous_state["restart_ref"]
        assert lineage["next_state"] == {k: v for k, v in trajectory["next_state"].items() if k != "restart_ref"}
        assert lineage["accepted_actions"] == [a for a in actions if a["month"] <= month.isoformat()]
        assert lineage["step_actions"] == [a for a in actions if a["month"] == month.isoformat()]
        assert not lineage["planning"]["future_states_committed"]
        assert trajectory["next_state"]["month"] == _next_month(month).isoformat()
        assert {w["well"] for w in trajectory["next_state"]["wells"]} == wells
        for item in lineage["artifacts"]:
            artifact = path.parent / item["path"]
            assert artifact.resolve().is_relative_to(path.parent.resolve()) and not artifact.is_symlink()
            assert digest(artifact) == item["sha256"]
            verified_files += 1
        lineages.append(path)
        previous_state = trajectory["next_state"]
    planning = [r for r in result["agent"]["records"] if r["phase"] == "planning"]
    reviews = [r for r in result["agent"]["records"] if r["phase"] == "terminal_month_review"]
    assert len(planning) == len(reviews) == expected_months
    assert "google_forecast" not in result["agent"]
    expected_state = _state_payload(config.initial_state)
    for plan, review, trajectory in zip(planning, reviews, trajectories, strict=True):
        assert plan["agent"]["context"]["surrogate_used"] is False
        assert review["agent"]["context"]["surrogate_used"] is False
        assert plan["agent"]["context"]["state"] == expected_state
        assert all(d["approved"] for d in review["agent"]["decisions"])
        assert review["agent"]["decisions"][-1]["tool_evidence"]
        expected_state = trajectory["next_state"]
    baseline_actions = [_action_payload(a) for a in ScheduleCompiler().validate(
        config.case, (a for m in months for a in config.candidates[m][0])
    )]
    baseline = [baseline_run] if baseline_run is not None else []
    for path in ([] if baseline_run is not None else config.opm_runs_dir.glob("*/lineage.json")):
        item = json.loads(path.read_text())
        if item["input_state"]["month"] == config.case.start.isoformat() and (
            item["step_actions"] + item.get("planning", {}).get("tail_actions", []) == baseline_actions
        ):
            baseline.append(path.parent)
    assert len(baseline) == 1, "a unique complete incumbent baseline is required"
    base, final = baseline[0], lineages[-1].parent
    backend._verify_opm_manifest(base / "manifest.json", baseline=False)
    opm = [json.loads((p / "manifest.json").read_text()) for p in (base, final)]
    for key in ("source_sha256", "deck_sha256", "image_reference"):
        assert opm[0][key] == opm[1][key]
    inputs = [{a["path"]: a["sha256"] for a in m["artifacts"] if a["path"].startswith("input/")
               and a["path"] != "input/" + str(config.schedule_include)} for m in opm]
    assert inputs[0] == inputs[1], "non-schedule physical inputs differ"
    rows = [list(csv.DictReader((p / "canonical/chdd.csv").open())) for p in (base, final)]
    expected_grid = {(_next_month(m).isoformat(), well) for m in months for well in wells}
    for data in rows:
        managed = [r for r in data if config.case.start.isoformat() < r['DATA'] <= _next_month(config.case.end).isoformat()]
        assert len(managed) == len(expected_grid)
        assert {(r['DATA'], r['well']) for r in managed} == expected_grid
        assert max(float(r['WLPR']) for r in managed) <= config.case.max_liquid_rate + 1e-6
    history = [{(r["DATA"], r["well"]): r for r in data if r["DATA"] <= config.case.start.isoformat()} for data in rows]
    assert history[0] == history[1], "pre-control history differs"
    econ_dirs = [base / ("planning-economics" if expected_months > 1 else "economics"), final / "economics"]
    econ = [json.loads((p / "manifest.json").read_text()) for p in econ_dirs]
    for key in ("calculator_sha256", "norms_source_sha256", "norms_sha256", "assumption_overrides", "start_year"):
        assert econ[0][key] == econ[1][key]
    values = [json.loads((p / "result.json").read_text())["summary"]["totalChddM"] for p in econ_dirs]
    assert values[1] == result["evidence"]["step_economics"][-1]["npv_million_rub"]
    return {"schema": "timesoil.track1-lifecycle-audit/v1", "run_dir": str(run),
            "surrogate_used": False,
            "months": expected_months, "wells": len(wells), "actions": len(actions),
            "start_inclusive": config.case.start.isoformat(), "end_exclusive": _next_month(config.case.end).isoformat(),
            "lineage_files_verified": verified_files, "approved_monthly_reviews": len(reviews),
            "future_states_committed": False, "identical_pre_control_history": True,
            "schedule_sha256": digest(run / "wells_schedule.inc"), "manifest_sha256": digest(run / "manifest.json"),
            "baseline_chdd_m": values[0], "candidate_chdd_m": values[1],
            "delta_chdd_m": values[1] - values[0], "uplift_percent": (values[1] / values[0] - 1) * 100,
            "baseline_run": str(base), "final_run": str(final), "agent_seconds": result["agent"]["elapsed_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-months", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--baseline-run", type=Path)
    args = parser.parse_args()
    checked = audit(args.root, args.expected_months, run_dir=args.run_dir, baseline_run=args.baseline_run)
    with args.output.open("x") as stream:
        json.dump(checked, stream, indent=2)
    print(json.dumps(checked))
