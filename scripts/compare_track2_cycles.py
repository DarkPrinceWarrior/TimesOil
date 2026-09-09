"""Check existing full-cycle artifacts before reporting paired experimental CHDD."""

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


def verify(root, entry):
    path = root / entry["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes run directory")
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"artifact changed: {path}")


def load(root):
    receipt_raw = (root / "full-cycle-receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    if not receipt["complete"] or not receipt["execution_source_binding"]["verified"]:
        raise ValueError("incomplete full cycle")
    for entry in receipt["artifacts"].values():
        verify(root, entry)
    opm = json.loads((root / "manifest.json").read_text())
    for entry in opm["artifacts"]:
        verify(root, entry)
    if opm["status"] != "success" or opm["returncode"] != 0:
        raise ValueError("OPM failed")
    economics_dir = root / f"economics-{receipt['economics']['start_year']}"
    econ = json.loads((economics_dir / "manifest.json").read_text())
    result = json.loads((economics_dir / "result.json").read_text())
    if result["summary"]["totalChddM"] != receipt["economics"]["total_chdd_m"]:
        raise ValueError("receipt and calculator disagree")
    rows = list(csv.DictReader((root / "canonical/chdd.csv").open()))
    keys = [(r["DATA"], r["well"]) for r in rows]
    if len(keys) != len(set(keys)) or any(
        not math.isfinite(float(v)) for r in rows for k, v in r.items() if k not in ("DATA", "well")
    ):
        raise ValueError("duplicate or non-finite canonical data")
    return receipt, opm, econ, result, rows, sha256(receipt_raw).hexdigest()


def compare(baseline, candidate, expected_months=None):
    left, right = load(baseline), load(candidate)
    a, b = left[0], right[0]
    for key in ("source_sha256",):
        assert a[key] == b[key], key
    for key in ("months", "well_count", "source_well_count", "source_control_inventory_sha256"):
        assert a["controls"][key] == b["controls"][key], key
    for key in ("image_reference", "source_sha256", "deck_sha256"):
        assert left[1][key] == right[1][key], key
    schedules = {data[0]["artifacts"]["exact_opm_input_schedule"]["path"] for data in (left, right)}
    inputs = [{e["path"]: e["sha256"] for e in data[1]["artifacts"]
               if e["path"].startswith("input/") and e["path"] not in schedules} for data in (left, right)]
    assert inputs[0] == inputs[1], "non-schedule simulator inputs changed"
    for key in ("calculator_sha256", "norms_source_sha256", "norms_sha256", "assumption_overrides", "start_year"):
        assert left[2][key] == right[2][key], key
    assert left[3]["assumptions"] == right[3]["assumptions"]
    for key in ("start_inclusive", "end_exclusive", "months", "historical_cash_flows_included"):
        assert a["economics"]["management_period"][key] == b["economics"]["management_period"][key], key
    period = a["economics"]["management_period"]
    if expected_months is not None and (type(expected_months) is not int or expected_months < 1 or len(period["months"]) != expected_months):
        raise ValueError("paired calculation does not cover the requested economic horizon")
    start, end = period["start_inclusive"], period["end_exclusive"]
    history = [{(r["DATA"], r["well"]): r for r in data[4] if r["DATA"] <= start} for data in (left, right)]
    assert history[0] == history[1], "pre-control physical history changed"
    summaries = []
    for root, data in ((baseline, left), (candidate, right)):
        managed = [r for r in data[4] if start < r["DATA"] <= end]
        assert len(managed) == len(period["months"]) * a["controls"]["well_count"]
        maximum = max(float(r["WLPR"]) for r in managed)
        assert maximum <= 500 + 1e-6, "actual liquid rate exceeds 500 m3/day"
        summaries.append({"run": str(root.resolve()), "receipt_sha256": data[5],
                          "chdd_m": data[3]["summary"]["totalChddM"],
                          "summary": data[3]["summary"], "max_liquid_m3d": maximum,
                          "opm_seconds": data[1]["duration_seconds"],
                          "reported_critic_approved": data[0]["critic_approved"]})
    base, best = (s["chdd_m"] for s in summaries)
    assert base > 0 and math.isfinite(best)
    return {"schema": "timesoil.paired-cycle-audit/v1", "baseline": summaries[0],
            "candidate": summaries[1], "start_inclusive": start, "end_exclusive": end,
            "well_count": a["controls"]["well_count"], "months": len(period["months"]),
            "source_sha256": a["source_sha256"], "artifacts_verified": True,
            "identical_pre_control_history": True, "delta_chdd_m": best - base,
            "uplift_percent": (best / base - 1) * 100,
            "scope": "Training-archive experiment; competition end date and future constraints unconfirmed.",
            "approval": "Numerical audit only; does not validate LLM reasoning or certify surrogate UQ/OOD."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-months", type=int, help="Reject a shorter experimental window, e.g. require all 224 months")
    parser.add_argument("--select-from", type=Path, nargs="*", help="Additional completed candidates; select by full-period CHDD including the baseline")
    parser.add_argument("--agent-review", action="store_true", help="Ask the external agent workflow to audit the verified completed comparison")
    args = parser.parse_args()
    if args.select_from is not None and args.expected_months is None:
        parser.error("--select-from requires an explicit --expected-months economic horizon")
    result = compare(args.baseline, args.candidate, args.expected_months)
    if args.select_from is not None:
        comparisons = [result] + [compare(args.baseline, candidate, args.expected_months) for candidate in args.select_from]
        choices = [result["baseline"]] + [item["candidate"] for item in comparisons]
        selected = max(choices, key=lambda item: item["chdd_m"])
        result = {"schema": "timesoil.full-horizon-selection/v1", "months": result["months"],
                  "baseline": result["baseline"], "selected": selected,
                  "selection_metric": "official_chdd_over_identical_complete_period",
                  "comparisons": comparisons,
                  "claim": "Model-based training experiment; selection runs are not an untouched test set."}
    if args.agent_review:
        import asyncio
        from dataclasses import asdict
        from timesoil.aios.agents import AgentRole, AgentWorkflow, ToolDefinition, ToolRegistry
        from timesoil.aios.llm import ExternalQwenClient, LLMConfig

        tool = ToolDefinition("read_verified_comparison", "Read the completed paired artifact audit and full-period official CHDD.",
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            lambda _arguments, _context: result)

        async def review():
            async with ExternalQwenClient(LLMConfig.from_env()) as client:
                return await AgentWorkflow(client, ToolRegistry((tool,)),
                    role_tools={role: (tool.name,) for role in AgentRole},
                    required_tools={role: (tool.name,) for role in AgentRole},
                ).run({"track": 2, "phase": "completed_paired_numerical_audit",
                    "objective": "Audit the completed full-period OPM and official CHDD comparison. Read the tool. No new simulator run is requested. State actual delta and percentage; distinguish numeric validity from deployment readiness. Proposal provenance is outside this numerical audit: do not assert that a surrogate proposed or selected a candidate without explicit evidence.",
                    "facts": {"paired_opm_and_economics_verified": True,
                              "surrogate_uncertainty_independently_calibrated": False,
                              "autonomous_surrogate_deployment_certified": False,
                              "competition_result_claimed": False}})

        reviewed = asyncio.run(review())
        result = {**result, "agent_review": asdict(reviewed),
                  "agent_review_approved": reviewed.critic_approved}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
