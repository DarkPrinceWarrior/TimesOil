"""Block-agent planning protocol: schemas, concurrency, retry routing and merges."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from timesoil.aios.llm import _cerebras_json_schema
from timesoil.aios.planning import (
    BLOCK_INTENT_SCHEMA,
    CRITIC_VERDICT_SCHEMA,
    FIELD_PLAN_SCHEMA,
    MAX_BRIEF_CHARS,
    PlanningError,
    build_block_briefs,
    build_field_brief,
    merge_intents,
    parse_block_intent,
    parse_field_plan,
    parse_critic_verdict,
    run_critic,
    run_injection,
    run_round0,
    validate_schema,
)

ASSUMPTIONS = {
    "oilPriceRubT": 28_000.0, "deductionsRubT": 19_600.0, "oilOpexRubT": 40.0,
    "liquidOpexRubT": 100.0, "injectionOpexRubM3": 30.0, "fundAnnualRubWell": 1_000_000.0,
    "pumpOperationCostM": 1.8, "stopStartCostM": 1.0, "conversionBaseCostM": 5.0, "waccRate": 10.0,
}
NORMATIVE = {"assumptions": ASSUMPTIONS, "pumps": [{"costM": 0.55}, {"costM": 8.05}]}
PROFILE = {
    "liquid_cap_m3d": 600.0, "injection_cap_m3d": 600.0, "bhp_bounds": (50.66, 303.98),
    "vrr": {"min": 0.85, "max": 1.15, "window_months": 3, "denominator": "liquid_reservoir",
            "lower_bound_status": "diagnostic"},
    "water_balance": {"deficit_m3": 0.0, "carryover": False},
    "pressure": {"field_min_bar": 109.43, "block_min_bar": None, "regions": "FIP_C1"},
    "repairs": (), "selection_margins": {"eps_liquid": 0.03, "eps_injection": 0.03, "phi": 0.95},
}
BLOCKS = {
    "blocks": [
        {"id": "B2", "wells": ["54", "75"], "component": 0, "fip_regions": [1], "centroid_ij": [9.0, 4.0]},
        {"id": "B1", "wells": ["3", "27"], "component": 0, "fip_regions": [1, 2], "centroid_ij": [2.0, 3.0]},
    ],
    "well_to_block": {"3": "B1", "27": "B1", "54": "B2", "75": "B2"},
}
FIELD_STATE = {
    "month": "2007-01-01",
    "wells": [
        {"well": "3", "role": "producer", "status": "OPEN", "target": "LRAT", "value": 123.456, "bhp_limit": 60.1234},
        {"well": "27", "role": "injector", "status": "OPEN", "target": "WRAT", "value": 88.8888, "bhp_limit": 290.0},
        {"well": "54", "role": "producer", "status": "OPEN", "target": "LRAT", "value": 50.0, "bhp_limit": 60.0},
        {"well": "75", "role": "producer", "status": "SHUT", "target": "LRAT", "value": 0.0, "bhp_limit": 60.0},
    ],
    "totals": {"liquid_m3d": 389.123, "oil_tpd": 337.0, "injection_m3d": 547.0},
    "neighbours": {"3": [["27", 0.123456], ["54", 0.02], ["75", 0.01], ["99", 0.001]]},
}
FORECAST = {
    "3": {"liquid_m3d": 100.0, "oil_tpd": 80.0, "water_cut": 0.1234, "wbp9": 118.7654},
    "27": {"injection_m3d": 200.0, "water_cut": 1.0, "wbp9": 121.0},
    "54": {"liquid_m3d": 60.0, "oil_tpd": 20.0, "water_cut": 0.66, "wbp9": 115.0},
    "75": {"liquid_m3d": 0.0, "oil_tpd": 0.0, "water_cut": 0.9, "wbp9": 112.0},
}
CANDIDATES = [
    {"id": i, "npv_m": 1000.0 + i, "margins": {"liquid_m3d": 12.3456}} for i in range(12)
]
FEEDBACK = {"block_contribution": {"B1": 12.3456, "B2": -1.0}, "rejections": [{"well": "54", "reason": "bhp"}]}


def intent_payload(block: str, wells: list[str], **overrides: Any) -> dict[str, Any]:
    payload = {
        "block": block,
        "analysis": f"block {block}",
        "well_scales": [{"well": wells[0], "scale": 1.1}],
        "well_updates": [],
        "shut_wells": [],
        "boundary_requests": [],
        "expected": {"oil_delta_tpd": 1.0, "liquid_delta_m3d": 2.0, "injection_delta_m3d": 0.0},
    }
    payload.update(overrides)
    return payload


def plan_payload(count: int = 1) -> dict[str, Any]:
    policy = {
        "producer_scale": 1.0, "injector_scale": 1.0, "well_scales": [], "shut_wells": [],
        "well_updates": [], "producer_bhp_add": 0.0, "injector_bhp_factor": 1.0,
    }
    genes = {"shut_wells": ["75"], "well_updates": [], "overrides": [{"well": "3", "scale": 1.2}]}
    return {
        "seeds": [{"policy": policy, "genes": genes, "x_hint": [0.5] * 4} for _ in range(count)],
        "bounds_hint": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "next_focus": ["water allocation"],
    }


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.content_sha256 = f"sha-{len(json.dumps(payload, sort_keys=True))}"
        self.finish_reason = "stop"


class FakeClient:
    """Canned structured answers keyed by schema name and block, with a scripted delay."""

    def __init__(self, answers: dict[str, Any], *, delays: dict[str, float] | None = None) -> None:
        self.answers = answers
        self.delays = delays or {}
        self.calls: list[dict[str, Any]] = []
        self.finished: list[str] = []

    async def structured(self, messages, *, schema, schema_name, max_tokens=None, timeout_seconds=None):
        data = json.loads(messages[-1].content.removeprefix("<data>").removesuffix("</data>"))
        key = str(data.get("block", schema_name))
        self.calls.append({"key": key, "schema_name": schema_name, "data": data})
        await asyncio.sleep(self.delays.get(key, 0.0))
        answer = self.answers[key]
        if callable(answer):
            answer = answer(data, len([call for call in self.calls if call["key"] == key]))
        if isinstance(answer, Exception):
            raise answer
        self.finished.append(key)
        return answer, FakeResponse(answer)


def briefs_and_field():
    briefs = build_block_briefs(FIELD_STATE, FORECAST, BLOCKS, PROFILE, NORMATIVE, CANDIDATES, FEEDBACK)
    field = build_field_brief(FIELD_STATE, FORECAST, BLOCKS, PROFILE, NORMATIVE, CANDIDATES, FEEDBACK)
    return briefs, field


def test_briefs_round_to_three_figures_stay_in_scope_and_fit_the_bound():
    briefs, field = briefs_and_field()
    assert [brief.block for brief in briefs] == ["B1", "B2"]
    first = briefs[0].to_dict()
    well = {row["well"]: row for row in first["wells"]}["3"]
    assert well["value"] == 123.0 and well["bhp_limit"] == 60.1 and well["water_cut"] == 0.123
    assert well["neighbours"] == [["27", 0.123], ["54", 0.02], ["75", 0.01]]  # top three only
    assert first["caps"]["remaining_liquid_m3d"] == pytest.approx(211.0)
    assert first["prices"]["liquid_opex_rub_t"] == 100.0 and first["prices"]["conversion_cost_m"] == 5.0
    assert first["prices"]["pump_change_cost_m"] == 1.8 and first["prices"]["pump_capex_m"] == [0.55, 8.05]
    assert len(first["candidates_digest"]) == 8 and first["candidates_digest"][0]["npv_m"] == 1010.0
    assert first["feedback"]["rejections"] == []  # well 54 belongs to B2
    assert briefs[1].to_dict()["feedback"]["rejections"] == [{"reason": "bhp", "well": "54"}]
    assert max(len(json.dumps(brief.to_dict())) for brief in briefs) < MAX_BRIEF_CHARS
    assert field.to_dict()["blocks"][0]["mean_water_cut"] == pytest.approx(0.562, rel=1e-3)


def test_brief_rejects_a_payload_above_the_character_bound():
    wide = {"blocks": [{"id": "B1", "wells": [f"w{index}" for index in range(4000)]}],
            "well_to_block": {f"w{index}": "B1" for index in range(4000)}}
    with pytest.raises(PlanningError, match="above the 40000"):
        build_block_briefs({"wells": [], "totals": {}}, {}, wide, PROFILE, NORMATIVE, [], {})


def test_schemas_reject_unknown_keys_wrong_types_and_foreign_wells():
    for schema in (BLOCK_INTENT_SCHEMA, FIELD_PLAN_SCHEMA, CRITIC_VERDICT_SCHEMA):
        text = json.dumps(schema)
        assert '"additionalProperties": false' in text
        for forbidden in ("pattern", "format", "minItems", "maxItems", "minLength", "maxLength"):
            assert f'"{forbidden}"' not in text  # Cerebras strict mode rejects these keywords
        assert _cerebras_json_schema(schema) == schema  # the wire dialect changes nothing
        assert set(schema["required"]) == set(schema["properties"])  # strict mode requires every key
    wells = frozenset({"3", "27"})
    with pytest.raises(PlanningError, match="unknown keys"):
        parse_block_intent({**intent_payload("B1", ["3"]), "extra": 1}, block="B1", wells=wells)
    with pytest.raises(PlanningError, match="omits required keys"):
        payload = intent_payload("B1", ["3"])
        payload.pop("expected")
        parse_block_intent(payload, block="B1", wells=wells)
    with pytest.raises(PlanningError, match="outside its scope"):
        parse_block_intent(intent_payload("B1", ["54"]), block="B1", wells=wells)
    with pytest.raises(PlanningError, match="shut_wells touches wells outside block"):
        parse_block_intent(intent_payload("B1", ["3"], shut_wells=["75"]), block="B1", wells=wells)
    with pytest.raises(PlanningError, match="must be number"):
        parse_block_intent(intent_payload("B1", ["3"], well_scales=[{"well": "3", "scale": "big"}]),
                           block="B1", wells=wells)
    with pytest.raises(PlanningError, match="above 600"):
        parse_block_intent(intent_payload("B1", ["3"], analysis="x" * 601), block="B1", wells=wells)
    update = {"well": "3", "start": "2007-01-01", "end": "2007-06-01", "role": None, "status": "SHUT",
              "target": None, "value": None, "bhp_limit": None}
    parsed = parse_block_intent(intent_payload("B1", ["3"], well_updates=[update]), block="B1", wells=wells)
    assert parsed.well_updates == ({"well": "3", "start": "2007-01-01", "end": "2007-06-01", "status": "SHUT"},)
    with pytest.raises(PlanningError, match="outside its enum"):
        validate_schema({**update, "status": "HALF"}, BLOCK_INTENT_SCHEMA["properties"]["well_updates"]["items"])


def test_round0_calls_every_block_concurrently_and_orders_the_journal_by_block():
    briefs, field = briefs_and_field()
    client = FakeClient(
        {"B1": intent_payload("B1", ["3"]), "B2": intent_payload("B2", ["54"]), "field_plan": plan_payload()},
        delays={"B1": 0.05},  # B1 answers last, yet must still be first in the journal
    )
    intents, plan, journal = asyncio.run(run_round0(client, briefs, field, seed=20260909))
    assert client.finished == ["B2", "B1", "field_plan"]  # B1 answered last
    assert sorted(call["key"] for call in client.calls[:2]) == ["B1", "B2"]  # both started before either finished
    assert [intent.block for intent in intents] == ["B1", "B2"]
    assert [(entry.role, entry.block) for entry in journal] == [
        ("block_planner", "B1"), ("block_planner", "B2"), ("field_planner", None)
    ]
    assert all(len(entry.request_sha256) == 64 for entry in journal)
    assert all(entry.response_sha256 and not entry.rejections for entry in journal)
    assert len(client.calls) == 3  # one per block plus the global planner
    assert plan.next_focus == ("water allocation",) and len(plan.seeds) == 1


def test_round0_routes_a_validator_error_back_to_one_block_and_retries_it_once():
    briefs, field = briefs_and_field()
    seen: list[dict[str, Any]] = []

    def answer_b1(data, attempt):
        seen.append(data)
        return intent_payload("B1", ["3"], shut_wells=["27"] if attempt == 1 else [])

    def validate(intent):
        if "27" in intent.shut_wells:
            raise ValueError("shutting the only injector: wells ['27']")

    client = FakeClient({"B1": answer_b1, "B2": intent_payload("B2", ["54"]), "field_plan": plan_payload()})
    intents, _, journal = asyncio.run(run_round0(client, briefs, field, seed=1, validate=validate))
    keys = [call["key"] for call in client.calls]
    assert keys.count("B1") == 2 and keys.count("B2") == 1 and keys[-1] == "field_plan"  # only B1 retried
    assert "wells ['27']" in seen[1]["previous_rejection"]
    assert [entry.attempt for entry in journal if entry.block == "B1"] == [0, 1]
    assert journal[0].rejections and not journal[1].rejections
    assert [intent.block for intent in intents] == ["B1", "B2"]


def test_round0_drops_a_block_that_fails_twice_without_stopping_the_round():
    briefs, field = briefs_and_field()
    client = FakeClient(
        {"B1": intent_payload("B1", ["54"]), "B2": intent_payload("B2", ["54"]), "field_plan": plan_payload()}
    )
    intents, plan, journal = asyncio.run(run_round0(client, briefs, field, seed=1))
    assert [intent.block for intent in intents] == ["B2"]
    assert len([entry for entry in journal if entry.block == "B1"]) == 2
    assert all(entry.rejections for entry in journal if entry.block == "B1")
    assert plan.seeds and client.calls[-1]["data"]["intents"][0]["block"] == "B2"


def test_injection_is_capped_at_four_edits_and_the_critic_can_only_veto():
    _, field = briefs_and_field()
    client = FakeClient({"field_plan": plan_payload(count=4), "critic_verdict": {
        "approved": True, "blocking_findings": ["forecast NPV quoted as official"], "evidence": ["seal.json:1"]}})
    edits, journal = asyncio.run(run_injection(client, [{"id": 1, "npv_m": 10.0}], field, seed=1))
    assert len(edits) == 4 and edits[0][0]["shut_wells"] == ["75"] and edits[0][1] == (0.5, 0.5, 0.5, 0.5)
    assert len(journal) == 1 and not journal[0].rejections
    too_many = FakeClient({"field_plan": plan_payload(count=5)})
    assert asyncio.run(run_injection(too_many, [], field, seed=1))[0] == ()
    verdict, entry = asyncio.run(run_critic(client, {"chdd_m": 1.0}, {"wape": 0.1}))
    assert verdict.approved is False and verdict.blocking_findings and entry.role == "critic"
    assert parse_critic_verdict({"approved": True, "blocking_findings": [], "evidence": []}).approved is True


def test_merge_is_deterministic_and_drops_unknown_wells_and_reverse_conversions():
    wells = frozenset({"3", "27"})
    intent_b1 = parse_block_intent(
        intent_payload("B1", ["3"], shut_wells=["27"], well_updates=[
            {"well": "27", "start": "2010-01-01", "end": "2012-01-01", "role": "producer",
             "status": "OPEN", "target": "LRAT", "value": 10.0, "bhp_limit": 60.0},
            {"well": "3", "start": "2010-01-01", "end": "2012-01-01", "role": "injector",
             "status": "OPEN", "target": "WRAT", "value": 20.0, "bhp_limit": 300.0},
        ]),
        block="B1", wells=wells,
    )
    intent_b2 = parse_block_intent(
        intent_payload("B2", ["54"], shut_wells=["75"]), block="B2", wells=frozenset({"54", "75"})
    )
    plan = parse_field_plan(plan_payload())
    roles = {"3": "producer", "27": "injector", "54": "producer", "75": "producer"}
    merged = merge_intents([intent_b2, intent_b1], plan, roles)
    assert merged == merge_intents([intent_b1, intent_b2], plan, roles)
    combined = merged[-1][0]
    assert combined["shut_wells"] == ["27", "75"]
    assert [item["well"] for item in combined["well_updates"]] == ["3"]  # 27 back to producer is dropped
    assert combined["overrides"] == {"3": 1.1, "54": 1.1}
    assert merged[-1][1] == (0.5, 0.5)  # midpoint of the bounds hint
    assert merge_intents([], plan, roles) == [(
        {"shut_wells": ["75"], "well_updates": [], "overrides": {"3": 1.2}}, (0.5, 0.5, 0.5, 0.5))]
    unknown = merge_intents([intent_b1], plan, {"3": "producer"})[-1]
    assert unknown[0]["shut_wells"] == [] and unknown[0]["overrides"] == {"3": 1.1}
