"""Exercise a conversion and BHP change through two committed OPM steps and official costs."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from run_track1_mpc import _continuation_tail, _next_month, _propose_controls, build_backend, load_config


def check(config_path, output):
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    config = replace(config, case=replace(config.case, allow_conversion_to_injection=True),
                     opm_runs_dir=output / "opm")
    backend = build_backend(config)
    state = config.initial_state
    months = [state.month, _next_month(state.month)]
    steps, previous = [], None
    for month in months:
        baseline = config.candidates[month][0] if previous is None else tuple(
            replace(a, month=month) for a in previous)
        actions = _propose_controls(config.case, baseline, [
            {"well": "1", "role": "injector", "status": "OPEN", "target": "WRAT", "value": 80, "bhp_limit": 280},
            {"well": "12", "role": "producer", "status": "OPEN", "target": "LRAT", "value": 100, "bhp_limit": 70},
        ])
        step = backend.run_from_restart(config.case, state, actions,
            planning_tail=_continuation_tail(config, state, actions))
        by_well = {w.well: w for w in step.trajectory.next_state.wells}
        assert by_well["1"].role.value == "injector" and by_well["1"].injection_rate > 0
        assert by_well["1"].bhp <= 280 + 1e-5
        assert by_well["12"].bhp >= 70 - 1e-5
        path, _ = backend._parse_restart_ref(step.trajectory.next_state.restart_ref)
        backend._verify_opm_manifest(path.parent / "manifest.json", baseline=False)
        history = backend._authenticated_history(config.case, step.trajectory.next_state)
        assert len(history) == len(actions) * (len(steps) + 1)
        economics = json.loads((path.parent / "economics/result.json").read_text())
        conversions = [e for e in economics["conversionTransitions"] if e["well"] == "1"]
        assert len(conversions) == 1 and conversions[0]["conversionBaseCostM"] == 5
        assert not any(e["well"] == "1" and e["month"] == months[0].strftime("%Y-%m")
                       for e in economics["activityTransitions"])
        steps.append({"month": month.isoformat(), "restart_ref": step.trajectory.next_state.restart_ref,
                      "conversion_count": len(conversions), "conversion_opex_m": 5,
                      "injection_m3d": by_well["1"].injection_rate,
                      "injector_bhp_bar": by_well["1"].bhp, "producer_bhp_bar": by_well["12"].bhp,
                      "cumulative_chdd_m": step.economics.npv_million_rub,
                      "planning_chdd_m": step.planning_economics.npv_million_rub})
        state, previous = step.trajectory.next_state, actions
    result = {"scope": "Model Y control regression: two committed months, each evaluated over the full provided planning horizon; not an optimization result.",
              "source_sha256": config.source_sha256,
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "planning_end_exclusive": _next_month(config.case.end).isoformat(),
              "well_count": len(previous), "steps": steps, "passed": True}
    (output / "verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.config, args.output)))
