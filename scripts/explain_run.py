"""Build the interpretability report for a sealed search directory or a final directory.

Every section is optional: inputs the directory does not carry are reported as
"not computable" instead of failing the run.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from timesoil.aios.economics import opm_management_rows
from timesoil.aios.explain import (
    candidate_lineage, check_evidence, constraint_margins, forecast_vs_physics,
    npv_decomposition, read_chdd_csv, search_trace, selection_summary, sensitivity, write_report,
)


def _first(root: Path, *patterns: str) -> Path | None:
    for pattern in patterns:
        found = sorted(root.glob(pattern))
        if found:
            return found[0]
    return None


def _json(path: Path | None) -> Any:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def as_trace_entry(candidate: dict[str, Any]) -> dict[str, Any]:
    """Map a candidate ledger row onto the search-trace keys the helpers expect."""
    entry = dict(candidate)
    entry.setdefault("npv", candidate.get("forecast_chdd_m"))
    if "feasible" not in entry and "forecast_eligible" in candidate:
        entry["feasible"] = bool(candidate["forecast_eligible"])
    return entry


def _next_month(stamp: date) -> date:
    return date(stamp.year + stamp.month // 12, stamp.month % 12 + 1, 1)


def management_period(official: Any, audit: Any) -> tuple[date, date] | None:
    """The canonical CSV holds OPM report endpoints; economics needs elapsed months."""
    if isinstance(audit, dict) and audit.get("start_inclusive") and audit.get("end_exclusive"):
        return date.fromisoformat(audit["start_inclusive"]), date.fromisoformat(audit["end_exclusive"])
    if isinstance(official, dict) and official.get("startDate") and official.get("maxDate"):
        return (date.fromisoformat(official["startDate"]),
                _next_month(date.fromisoformat(official["maxDate"])))
    return None


def discover(root: Path, overrides: dict[str, Path | None]) -> dict[str, Path | None]:
    """Locate known artifacts, tolerating the `bounded-` prefix used in evidence bundles."""
    found = {
        "seal": _first(root, "selection-before-opm.json", "*selection-before-opm.json"),
        "audit": _first(root, "final-audit.json", "*final-audit.json"),
        "receipt": _first(root, "full-cycle-receipt.json", "*full-cycle-receipt.json"),
        "official_result": _first(root, "official-result.json", "*official-result.json"),
        "candidates": _first(root, "candidates.json", "*candidates.json"),
        "trace": _first(root, "search-trace.json", "*search-trace.json"),
        "agent_plan": _first(root, "first-agent-plan.json", "*agent-plan.json", "agent-00.json"),
        "forecast": _first(root, "forecast-*.npz", "*.npz"),
        "candidate_rows": _first(root, "physical-chdd-input.csv", "canonical/chdd.csv", "*chdd*.csv"),
        "profile": _first(root, "case_constraints.json", "*case-profile.json"),
    }
    found.update({key: value for key, value in overrides.items() if value is not None})
    return found


def build_sections(paths: dict[str, Path | None], *, densities: dict[str, float] | None,
                   block_of_well: dict[str, Any] | None,
                   start_year: int | None = None,
                   period: tuple[date, date] | None = None) -> dict[str, Any]:
    seal = _json(paths.get("seal"))
    official = _json(paths.get("official_result"))
    plan = _json(paths.get("agent_plan"))
    candidates = _json(paths.get("candidates"))
    if candidates is None and isinstance(plan, dict):
        candidates = (plan.get("context") or {}).get("candidates")
    trace = _json(paths.get("trace"))
    if trace is None and isinstance(candidates, list):
        trace = [as_trace_entry(row) for row in candidates]
    profile = _json(paths.get("profile"))
    norms = (official or {}).get("assumptions") if isinstance(official, dict) else None
    if norms is None and isinstance(seal, dict):
        norms = ((seal.get("normative_profile") or {}).get("assumptions"))
    if start_year is None and isinstance(official, dict) and official.get("startDate"):
        start_year = int(str(official["startDate"])[:4])
    audit = _json(paths.get("audit"))
    if period is None:
        period = management_period(official, audit)
    if start_year is None and period is not None:
        start_year = period[0].year

    def rows_of(path: Path) -> list[dict[str, Any]]:
        rows = read_chdd_csv(path)
        return rows if period is None else opm_management_rows(rows, period)

    def require(value: Any, message: str) -> Any:
        if value is None or (isinstance(value, (list, dict)) and not value):
            raise ValueError(message)
        return value

    def decomposition() -> Any:
        base = require(paths.get("baseline_rows"), "baseline chdd rows are absent from this directory")
        cand = require(paths.get("candidate_rows"), "candidate chdd rows are absent from this directory")
        return npv_decomposition(rows_of(base), rows_of(cand), norms,
                                 start_year=start_year, block_of_well=block_of_well)

    def margins() -> Any:
        cand = require(paths.get("candidate_rows"), "candidate chdd rows are absent from this directory")
        summary = _json(paths.get("summary_rows"))
        return constraint_margins(rows_of(cand), require(profile, "no case profile was supplied"),
                                  summary_rows=summary, densities=densities)

    def accuracy() -> Any:
        forecast = require(paths.get("forecast"), "no forecast npz in this directory")
        cand = require(paths.get("candidate_rows"), "candidate chdd rows are absent from this directory")
        return forecast_vs_physics(forecast, rows_of(cand), norms=norms)

    def selection() -> Any:
        return selection_summary(require(seal, "no selection-before-opm.json in this directory"),
                                 require(candidates, "no candidate ledger in this directory"))

    def evidence() -> Any:
        decisions = (plan or {}).get("decisions") if isinstance(plan, dict) else None
        context = (plan or {}).get("context") if isinstance(plan, dict) else None
        return check_evidence(require(decisions, "no agent decisions in this directory"),
                              require(context, "no agent context to verify citations against"))

    return {
        "npv_decomposition": decomposition,
        "constraint_margins": margins,
        "forecast_vs_physics": accuracy,
        "selection": selection,
        "search_trace": lambda: search_trace(require(trace, "no search trace or candidate ledger")),
        "sensitivity": lambda: sensitivity(require(trace, "no search trace or candidate ledger")),
        "candidate_lineage": lambda: candidate_lineage(require(trace, "no search trace or candidate ledger")),
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="sealed search directory or final directory")
    parser.add_argument("output", type=Path, help="directory that receives interpretability/")
    parser.add_argument("--baseline-rows", type=Path, default=None)
    parser.add_argument("--candidate-rows", type=Path, default=None)
    parser.add_argument("--forecast", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--summary-rows", type=Path, default=None)
    parser.add_argument("--blocks", type=Path, default=None, help="blocks.json with well_to_block")
    parser.add_argument("--management-period", nargs=2, metavar=("START", "END_EXCLUSIVE"),
                        default=None, help="first-of-month bounds mapping OPM endpoints to months")
    parser.add_argument("--start-year", type=int, default=None,
                        help="management start year; taken from the run's official result when absent")
    parser.add_argument("--oil-density", type=float, default=None, help="t/m3")
    parser.add_argument("--water-density", type=float, default=None, help="t/m3")
    args = parser.parse_args(argv)

    root = args.run.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"run directory not found: {root}")
    paths = discover(root, {"candidate_rows": args.candidate_rows, "forecast": args.forecast,
                            "profile": args.profile})
    paths["baseline_rows"] = args.baseline_rows
    paths["summary_rows"] = args.summary_rows

    densities = None
    if args.oil_density and args.water_density:
        densities = {"oil": args.oil_density, "water": args.water_density}
    block_of_well = None
    if args.blocks is not None:
        block_of_well = json.loads(args.blocks.read_text(encoding="utf-8")).get("well_to_block")

    period = (tuple(date.fromisoformat(value) for value in args.management_period)
              if args.management_period else None)
    sections = build_sections(paths, densities=densities, block_of_well=block_of_well,
                              start_year=args.start_year, period=period)
    report = write_report(args.output, sections,
                          inputs={name: path for name, path in paths.items() if path is not None},
                          title=f"Interpretability report for {root.name}")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
