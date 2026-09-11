"""Deterministic feasible-regime scenario bank for Model Z Track 2 training runs.

Emits full-period CycleRequest JSON files whose injection is limited to the
produced water of the same month (organizer rule K5, no carryover). Controls
only: no OPM run, no forecast and no economics happen here. The ``expected``
vectors in the manifest are *baseline arithmetic*, never a physical prediction.
"""

from __future__ import annotations

import argparse
import calendar
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from scipy.stats import qmc

from timesoil.aios.workflow import CycleRequest

SCHEMA = "timesoil.feasible-bank/v1"
# Mirrors workflow._ID: the scenario_id accepted by CycleRequest.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
LRAT_CAP_M3D = 500.0
INJECTION_CAP_M3D = 1500.0
MAX_BHP_BAR = 300.0
CONVERSION_WRAT_M3D = 100.0
CONVERSION_BHP_BAR = 300.0
BLOCK_SCALE_RANGE = (0.5, 1.3)
# Frozen holdout of the existing evaluation: never reachable as a training entry.
HELD_OUT_TEST_IDS = frozenset({"physical-sweep-03", "physical-sweep-05", "physical-sweep-06"})
HELD_OUT_TEST_PREFIXES = ("fresh-uncertainty",)
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
ID_PREFIX = "feasible-"

F1_PHI = (0.7, 0.85, 1.0)
F1_PRODUCER = (0.8, 1.0, 1.1)
F2_PHI = 0.85
F2_PRODUCER = 1.0
# (producer multiplier per segment, phi per segment); segments split at --switch-months.
F4_DESIGNS: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.8, 1.0, 1.1), (0.85, 0.85, 0.85)),
    ((1.1, 1.0, 0.8), (0.85, 0.85, 0.85)),
    ((1.0, 1.0, 1.0), (1.0, 0.85, 0.7)),
    ((1.0, 1.0, 1.0), (0.7, 0.85, 1.0)),
    ((1.1, 0.9, 1.1), (1.0, 0.7, 1.0)),
    ((0.9, 1.1, 0.9), (0.7, 1.0, 0.7)),
)
F5_SAMPLES = 8
F5_PHI = 0.85

Control = dict[str, Any]


class BankError(ValueError):
    """The baseline inputs or the requested bank design are unusable."""


# --------------------------------------------------------------------------- inputs


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def baseline_controls(request: Mapping[str, Any]) -> list[Control]:
    controls = request.get("controls")
    if not isinstance(controls, list) or not controls:
        raise BankError("baseline request has no controls")
    return [dict(action) for action in controls]


def months_of(controls: Sequence[Control]) -> list[str]:
    return sorted({str(action["month"]) for action in controls})


def days_in_month(month: str) -> int:
    parsed = date.fromisoformat(month)
    if parsed.day != 1:
        raise BankError(f"control month must be a first-of-month date: {month}")
    return calendar.monthrange(parsed.year, parsed.month)[1]


def surface_densities(manifest: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """(oil, water) kg/m3 per well from the canonical export manifest; multi-PVT wells use the connection mean."""
    conversion = manifest.get("conversion")
    if not isinstance(conversion, Mapping):
        raise BankError("export manifest has no conversion block")
    result: dict[str, tuple[float, float]] = {}
    for well, values in (conversion.get("density_by_well") or {}).items():
        result[str(well)] = (float(values["oil_kg_m3"]), float(values["water_kg_m3"]))
    for well, values in (conversion.get("connection_density_by_well") or {}).items():
        oil, water = values.get("oil_kg_m3"), values.get("water_kg_m3")
        if not isinstance(oil, list) or not isinstance(water, list) or not oil or not water:
            continue
        # Multi-PVT wells list their distinct connection densities; the mean is used, as in
        # timesfm_economics.export_densities, because these volumes feed the bank's caps
        # (3% margin), never the official calculator, which works in mass.
        result.setdefault(str(well), (sum(map(float, oil)) / len(oil), sum(map(float, water)) / len(water)))
    if not result:
        raise BankError("export manifest carries no usable surface densities")
    if any(not math.isfinite(v) or v <= 0 for pair in result.values() for v in pair):
        raise BankError("surface densities must be finite and positive")
    return result


def canonical_volumes(
    rows: Iterable[Mapping[str, str]], densities: Mapping[str, tuple[float, float]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Per (month, well) produced water and oil in m3 from tonne cumulative deltas."""
    water: dict[str, dict[str, float]] = defaultdict(dict)
    oil: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        month, well = str(row["DATA"]), str(row["well"])
        liquid_t, oil_t = float(row["WLPT_Diff"]), float(row["WOMT_Diff"])
        if not math.isfinite(liquid_t) or not math.isfinite(oil_t) or liquid_t < 0 or oil_t < 0:
            raise BankError(f"canonical cumulative delta is negative or non-finite: {well} {month}")
        water_t = liquid_t - oil_t
        if water_t < -1e-9 * max(1.0, liquid_t):
            raise BankError(f"oil mass exceeds liquid mass in the baseline export: {well} {month}")
        if well in water and month in water[well]:
            raise BankError(f"duplicate canonical row: {well} {month}")
        if (liquid_t or oil_t) and well not in densities:
            raise BankError(f"producing well has no surface density in the export manifest: {well}")
        oil_kg, water_kg = densities.get(well, (1.0, 1.0))
        water[well][month] = max(0.0, water_t) * 1000.0 / water_kg
        oil[well][month] = oil_t * 1000.0 / oil_kg
    if not water:
        raise BankError("baseline canonical export is empty")
    return dict(water), dict(oil)


def field_water_m3d(
    water: Mapping[str, Mapping[str, float]], months: Sequence[str]
) -> dict[str, float]:
    """Average daily produced-water rate of the field per month (K5 budget)."""
    result = {}
    for month in months:
        total = math.fsum(rows[month] for rows in water.values() if month in rows)
        result[month] = total / days_in_month(month)
    return result


# --------------------------------------------------------------------------- regime


def first_open_index(controls: Sequence[Control], months: Sequence[str]) -> dict[str, int | None]:
    """First month each well is OPEN in the baseline: a conservative stand-in for its first source WCON."""
    order = {month: index for index, month in enumerate(months)}
    result: dict[str, int | None] = {}
    for action in controls:
        well = str(action["well"])
        result.setdefault(well, None)
        if action["status"] == "OPEN":
            index = order[str(action["month"])]
            current = result[well]
            result[well] = index if current is None else min(current, index)
    return result


def regime_controls(
    baseline: Sequence[Control],
    *,
    producer_scale: Callable[[str, int], float],
    phi: Callable[[int], float],
    water_m3d: Mapping[str, float],
    shut_from: Mapping[str, int] = {},
    conversions: Sequence[Mapping[str, Any]] = (),
    injection_cap_m3d: float = INJECTION_CAP_M3D,
    injection_basis: str = "water",
    liquid_cap_m3d: float | None = None,
) -> list[Control]:
    """The single code path behind every family: scale, shut one-way, convert, then allocate water.

    ``injection_basis="water"`` allocates ``phi × produced water`` (training-deck families);
    ``"cap"`` allocates ``phi × injection_cap_m3d`` — the test case has an external supply
    capped at the field limit, so the bank must cover regimes near that limit, not near the
    (small) produced-water volume. ``liquid_cap_m3d`` scales every OPEN LRAT producer of a
    month proportionally when their targets exceed the field liquid cap.
    """
    if injection_basis not in ("water", "cap"):
        raise BankError("injection_basis must be 'water' or 'cap'")
    if liquid_cap_m3d is not None and (not math.isfinite(liquid_cap_m3d) or liquid_cap_m3d <= 0):
        raise BankError("liquid_cap_m3d must be a positive number")
    months = months_of(baseline)
    order = {month: index for index, month in enumerate(months)}
    converted = {str(item["well"]): item for item in conversions}
    if len(converted) != len(conversions):
        raise BankError("duplicate conversion well")
    output = [dict(action) for action in baseline]
    weight: dict[str, dict[str, float]] = defaultdict(dict)
    for action in output:
        month, well = str(action["month"]), str(action["well"])
        index = order[month]
        conversion = converted.get(well)
        if conversion is not None and index >= int(conversion["from_index"]):
            action.update(role="injector", status="OPEN", target="WRAT",
                          value=float(conversion["wrat_m3d"]),
                          bhp_limit=float(conversion["bhp_limit_bar"]))
            weight[month][well] = float(conversion["wrat_m3d"])
            continue
        if well in shut_from and index >= shut_from[well]:
            action.update(status="SHUT", value=0.0)
            continue
        if action["status"] != "OPEN":
            continue
        if action["role"] == "injector":
            weight[month][well] = float(action["value"])
            continue
        scale = producer_scale(well, index)
        if not math.isfinite(scale) or scale < 0:
            raise BankError("producer scale must be finite and nonnegative")
        action["value"] = float(action["value"]) * scale
        if action["target"] == "LRAT":
            action["value"] = min(action["value"], LRAT_CAP_M3D)
    by_key = {(str(a["month"]), str(a["well"])): a for a in output}
    for month in months:
        share = weight[month]
        total = math.fsum(share.values())
        if total <= 0:
            continue
        fraction = phi(order[month])
        if not math.isfinite(fraction) or fraction < 0:
            raise BankError("phi must be finite and nonnegative")
        basis = water_m3d[month] if injection_basis == "water" else injection_cap_m3d
        target = min(fraction * basis, injection_cap_m3d)
        for well, value in share.items():
            by_key[month, well]["value"] = target * value / total
    if liquid_cap_m3d is not None:
        for month in months:
            open_lrat = [a for a in output if str(a["month"]) == month and a["role"] == "producer"
                         and a["status"] == "OPEN" and a["target"] == "LRAT"]
            total = math.fsum(float(a["value"]) for a in open_lrat)
            if total > liquid_cap_m3d:
                factor = liquid_cap_m3d / total
                for action in open_lrat:
                    action["value"] = float(action["value"]) * factor
    return output


def check_invariants(
    controls: Sequence[Control],
    baseline: Sequence[Control],
    *,
    max_bhp_bar: float = MAX_BHP_BAR,
) -> None:
    """Every hard control rule the deterministic validator will re-check on A100."""
    months = months_of(baseline)
    order = {month: index for index, month in enumerate(months)}
    reference = {(str(a["month"]), str(a["well"])): a for a in baseline}
    opened = first_open_index(baseline, months)
    roles: dict[str, str] = {}
    if len(controls) != len(baseline):
        raise BankError("scenario changes the control grid")
    for action in sorted(controls, key=lambda a: (str(a["month"]), str(a["well"]))):
        month, well = str(action["month"]), str(action["well"])
        source = reference.get((month, well))
        if source is None:
            raise BankError(f"control outside the baseline grid: {well} {month}")
        value = float(action["value"])
        if not math.isfinite(value) or value < 0:
            raise BankError(f"control value must be finite and nonnegative: {well} {month}")
        if action["status"] == "SHUT" and value != 0:
            raise BankError(f"shut well carries a nonzero target: {well} {month}")
        if action["role"] == "injector" and action["target"] != "WRAT":
            raise BankError(f"injector must target WRAT: {well} {month}")
        if action["role"] == "producer" and action["target"] not in {"ORAT", "LRAT"}:
            raise BankError(f"producer must target ORAT or LRAT: {well} {month}")
        if action["target"] == "LRAT" and value > LRAT_CAP_M3D + 1e-9:
            raise BankError(f"liquid target exceeds {LRAT_CAP_M3D} m3/day: {well} {month}")
        start = opened[well]
        if (start is None or order[month] < start) and (action["status"] != "SHUT" or value != 0):
            raise BankError(f"well operated before its first source control month: {well} {month}")
        if roles.get(well) == "injector" and action["role"] == "producer":
            raise BankError(f"reverse conversion is not permitted: {well} {month}")
        roles[well] = str(action["role"])
        if action["status"] != "OPEN":
            continue
        limit, original = action.get("bhp_limit"), source.get("bhp_limit")
        if action["role"] != source["role"]:
            if limit is None or not 0 < float(limit) <= max_bhp_bar:
                raise BankError(f"conversion requires an explicit injection BHP ceiling: {well} {month}")
        elif original is None:
            if limit is not None:
                raise BankError(f"BHP limit invented without a reference bound: {well} {month}")
        elif limit is None:
            raise BankError(f"BHP limit dropped relative to the baseline: {well} {month}")
        elif action["role"] == "injector":
            if float(limit) > float(original) + 1e-9:
                raise BankError(f"injection BHP ceiling relaxed: {well} {month}")
        elif float(limit) < float(original) - 1e-9:
            raise BankError(f"producer BHP floor relaxed: {well} {month}")


def expected_field(
    controls: Sequence[Control], months: Sequence[str], water_m3d: Mapping[str, float]
) -> dict[str, list[float]]:
    """Baseline-arithmetic field vectors: targets, not a physical forecast."""
    totals: dict[str, dict[str, float]] = {key: defaultdict(float)
                                           for key in ("LRAT", "ORAT", "WRAT")}
    for action in controls:
        if action["status"] == "OPEN":
            totals[str(action["target"])][str(action["month"])] += float(action["value"])
    return {
        "months": list(months),
        "liquid_m3d": [round(totals["LRAT"][month], 6) for month in months],
        "oil_target_m3d": [round(totals["ORAT"][month], 6) for month in months],
        "injection_m3d": [round(totals["WRAT"][month], 6) for month in months],
        "water_m3d": [round(water_m3d[month], 6) for month in months],
    }


# --------------------------------------------------------------------------- split


def split_of(scenario_id: str) -> str:
    """Frozen 70/15/15 assignment by sha256 of the scenario id; existing test cases stay test."""
    if scenario_id in HELD_OUT_TEST_IDS or scenario_id.startswith(HELD_OUT_TEST_PREFIXES):
        return "test"
    draw = int(sha256(scenario_id.encode()).hexdigest()[:16], 16) / 2**64
    if draw < TRAIN_FRACTION:
        return "train"
    return "validation" if draw < TRAIN_FRACTION + VALIDATION_FRACTION else "test"


# --------------------------------------------------------------------------- blocks


def load_blocks(path: Path, producers: Sequence[str]) -> dict[str, list[str]]:
    """Accept scripts/export_blocks.py output: {"blocks": [{"id", "wells"}]} or {well: block}."""
    raw = json.loads(path.read_text())
    groups: dict[str, list[str]] = defaultdict(list)
    if isinstance(raw, Mapping) and isinstance(raw.get("blocks"), list):
        for block in raw["blocks"]:
            for well in block["wells"]:
                groups[str(block["id"])].append(str(well))
    elif isinstance(raw, Mapping) and all(isinstance(value, str) for value in raw.values()):
        for well, block in raw.items():
            groups[str(block)].append(str(well))
    else:
        raise BankError("blocks.json must carry a blocks list or a well -> block mapping")
    known = set(producers)
    result = {block: sorted(w for w in wells if w in known) for block, wells in groups.items()}
    result = {block: wells for block, wells in result.items() if wells}
    covered = [well for wells in result.values() for well in wells]
    if len(covered) != len(set(covered)):
        raise BankError("blocks.json assigns a well to more than one block")
    missing = known - set(covered)
    if missing:
        raise BankError(f"blocks.json misses {len(missing)} producers, e.g. {sorted(missing)[:5]}")
    return dict(sorted(result.items()))


def production_quantile_blocks(
    producers: Sequence[str], liquid_m3: Mapping[str, float], count: int
) -> dict[str, list[str]]:
    """Fallback grouping when no blocks.json is supplied.

    The request carries well names only - no FIPNUM and no WELSPECS coordinates - so
    geology cannot be recovered here. Wells are split into equal-size quantiles of
    baseline cumulative liquid instead. This is a production proxy, NOT a geological
    block map; the manifest records it as such.
    """
    if count < 1:
        raise BankError("block count must be positive")
    ordered = sorted(producers, key=lambda well: (-liquid_m3.get(well, 0.0), well))
    blocks = min(count, len(ordered))
    result: dict[str, list[str]] = {f"q{index}": [] for index in range(blocks)}
    for position, well in enumerate(ordered):
        result[f"q{position * blocks // len(ordered)}"].append(well)
    return {block: sorted(wells) for block, wells in result.items() if wells}


# --------------------------------------------------------------------------- families


def _segment(index: int, switches: Sequence[int]) -> int:
    return sum(1 for switch in switches if index >= switch)


def water_cut_order(
    producers: Sequence[str],
    water: Mapping[str, Mapping[str, float]],
    oil: Mapping[str, Mapping[str, float]],
    months: Sequence[str],
    until: int,
) -> list[str]:
    """Producers ranked by cumulative volumetric water cut over months[:until], ties by name."""
    window = set(months[: max(1, until)])
    ranked = []
    for well in producers:
        produced = math.fsum(v for month, v in water.get(well, {}).items() if month in window)
        produced_oil = math.fsum(v for month, v in oil.get(well, {}).items() if month in window)
        liquid = produced + produced_oil
        ranked.append((-(produced / liquid) if liquid > 0 else 1.0, well))
    return [well for _, well in sorted(ranked)]


def build_scenarios(
    baseline: Sequence[Control],
    *,
    water: Mapping[str, Mapping[str, float]],
    oil: Mapping[str, Mapping[str, float]],
    water_m3d: Mapping[str, float],
    blocks: Mapping[str, Sequence[str]],
    seed: int,
    shut_counts: Sequence[int],
    shut_months: Sequence[int],
    switch_months: Sequence[int],
    conversion_anchor: str,
    conversion_from_month: int,
    injection_cap_m3d: float,
    injection_basis: str = "water",
    liquid_cap_m3d: float | None = None,
) -> list[dict[str, Any]]:
    months = months_of(baseline)
    opened = first_open_index(baseline, months)
    producers = sorted({str(a["well"]) for a in baseline if a["role"] == "producer"})
    if not producers:
        raise BankError("baseline has no producers")
    constant = lambda value: (lambda *_: value)  # noqa: E731 - one-line family scales
    scenarios: list[dict[str, Any]] = []

    def emit(scenario_id: str, family: str, parameters: dict[str, Any], **kwargs: Any) -> None:
        controls = regime_controls(baseline, water_m3d=water_m3d,
                                   injection_cap_m3d=injection_cap_m3d,
                                   injection_basis=injection_basis,
                                   liquid_cap_m3d=liquid_cap_m3d, **kwargs)
        check_invariants(controls, baseline)
        scenarios.append({"id": scenario_id, "family": family, "parameters": parameters,
                          "controls": controls})

    for phi in F1_PHI:
        for multiplier in F1_PRODUCER:
            emit(f"{ID_PREFIX}f1-p{round(phi * 100):03d}-m{round(multiplier * 100):03d}", "F1",
                 {"phi": phi, "producer_multiplier": multiplier},
                 producer_scale=constant(multiplier), phi=constant(phi))

    for start in shut_months:
        if not 0 <= start < len(months):
            raise BankError(f"shut month {start} is outside the management period")
        ranked = water_cut_order(producers, water, oil, months, start)
        for count in shut_counts:
            if count > len(ranked):
                raise BankError(f"cannot shut {count} producers: only {len(ranked)} exist")
            chosen = sorted(ranked[:count])
            emit(f"{ID_PREFIX}f2-s{count:02d}-m{start:03d}", "F2",
                 {"phi": F2_PHI, "producer_multiplier": F2_PRODUCER,
                  "shut_from_month_index": start, "shut_from_month": months[start],
                  "shut_wells": chosen},
                 producer_scale=constant(F2_PRODUCER), phi=constant(F2_PHI),
                 shut_from={well: start for well in chosen})

    eligible = [well for well in water_cut_order(producers, water, oil, months, len(months))
                if well != conversion_anchor and opened[well] is not None]
    if opened.get(conversion_anchor) is None:
        raise BankError(f"conversion anchor {conversion_anchor!r} is never open in the baseline")
    if len(eligible) < 2:
        raise BankError("fewer than two convertible producers besides the anchor")
    first, second = eligible[0], eligible[1]
    designs = ((conversion_anchor,), (first,), (conversion_anchor, first),
               (first, second), (conversion_anchor, first, second))
    for index, wells in enumerate(designs):
        conversions = [{"well": well,
                        "from_index": max(conversion_from_month, opened[well] or 0),
                        "wrat_m3d": CONVERSION_WRAT_M3D,
                        "bhp_limit_bar": CONVERSION_BHP_BAR} for well in wells]
        emit(f"{ID_PREFIX}f3-{index:02d}", "F3",
             {"phi": F2_PHI, "producer_multiplier": F2_PRODUCER,
              "conversions": [{**item, "from_month": months[item["from_index"]]}
                              for item in conversions]},
             producer_scale=constant(F2_PRODUCER), phi=constant(F2_PHI), conversions=conversions)

    switches = sorted(switch_months)
    if any(not 0 < switch < len(months) for switch in switches) or len(set(switches)) != len(switches):
        raise BankError("switch months must be distinct and inside the management period")
    for index, (producer_segments, phi_segments) in enumerate(F4_DESIGNS):
        if len(producer_segments) != len(switches) + 1:
            raise BankError("F4 design does not match the number of segments")
        emit(f"{ID_PREFIX}f4-{index:02d}", "F4",
             {"switch_month_indices": switches,
              "switch_months": [months[switch] for switch in switches],
              "producer_multipliers": list(producer_segments), "phi": list(phi_segments)},
             producer_scale=lambda _well, i, p=producer_segments: p[_segment(i, switches)],
             phi=lambda i, p=phi_segments: p[_segment(i, switches)])

    block_of = {well: block for block, wells in blocks.items() for well in wells}
    names = sorted(blocks)
    sample = qmc.LatinHypercube(d=len(names), seed=seed).random(n=F5_SAMPLES)
    low, high = BLOCK_SCALE_RANGE
    for index, row in enumerate(sample):
        scales = {name: round(low + (high - low) * float(value), 6)
                  for name, value in zip(names, row, strict=True)}
        emit(f"{ID_PREFIX}f5-{index:02d}", "F5",
             {"phi": F5_PHI, "block_scales": scales, "blocks": {n: list(blocks[n]) for n in names}},
             producer_scale=lambda well, _i, s=scales: s[block_of[well]], phi=constant(F5_PHI))
    return scenarios


# --------------------------------------------------------------------------- output


def scenario_request(
    request: Mapping[str, Any], scenario: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    context = {key: value for key, value in dict(request["context"]).items()
               if key not in {"policy", "relative_to_incumbent", "baseline_comparison"}}
    constraints = dict(context.get("constraints") or {})
    if scenario["family"] == "F3":
        constraints["allow_conversion_to_injection"] = True
    if constraints:
        context["constraints"] = constraints
    context.update(
        track=2,
        objective=("Populate the feasible-regime training bank: injection limited to the produced "
                   "water of the same month. No improvement is claimed and no selection is made."),
        facts={"is_baseline": False, "schedule_kind": "feasible_regime_bank",
               "surrogate_used_for_candidate_selection": False,
               "optimization_improvement_claimed": False},
        feasible_bank={"schema": SCHEMA, "family": scenario["family"], "seed": seed,
                       "parameters": scenario["parameters"]},
    )
    return {**{key: value for key, value in request.items() if key != "controls"},
            "context": context, "controls": scenario["controls"],
            "scenario_id": scenario["id"]}


def build_bank(
    request: Mapping[str, Any],
    canonical_rows: Iterable[Mapping[str, str]],
    export_manifest: Mapping[str, Any],
    *,
    seed: int,
    blocks_path: Path | None,
    block_count: int,
    shut_counts: Sequence[int],
    shut_months: Sequence[int],
    switch_months: Sequence[int],
    conversion_anchor: str,
    conversion_from_month: int,
    injection_cap_m3d: float,
    injection_basis: str = "water",
    liquid_cap_m3d: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return the scenarios, the manifest head and the derived baseline vectors."""
    baseline = baseline_controls(request)
    months = months_of(baseline)
    densities = surface_densities(export_manifest)
    water, oil = canonical_volumes(canonical_rows, densities)
    wells = {str(action["well"]) for action in baseline}
    if set(water) - wells:
        raise BankError("canonical export carries wells absent from the baseline request")
    missing = {month for rows in water.values() for month in rows} ^ set(months)
    if missing:
        raise BankError("canonical export months disagree with the baseline control period")
    check_invariants(baseline, baseline)
    water_m3d = field_water_m3d(water, months)
    producers = sorted({str(a["well"]) for a in baseline if a["role"] == "producer"})
    liquid = {well: math.fsum(water.get(well, {}).values()) + math.fsum(oil.get(well, {}).values())
              for well in producers}
    if blocks_path is not None:
        blocks = load_blocks(blocks_path, producers)
        blocks_source = {"path": str(blocks_path), "sha256": _digest(blocks_path),
                         "geological": True}
    else:
        blocks = production_quantile_blocks(producers, liquid, block_count)
        blocks_source = {"path": None, "sha256": None, "geological": False,
                         "fallback": "equal-size quantiles of baseline cumulative liquid; the "
                                     "request carries no FIPNUM and no WELSPECS coordinates"}
    scenarios = build_scenarios(
        baseline, water=water, oil=oil, water_m3d=water_m3d, blocks=blocks, seed=seed,
        shut_counts=shut_counts, shut_months=shut_months, switch_months=switch_months,
        conversion_anchor=conversion_anchor, conversion_from_month=conversion_from_month,
        injection_cap_m3d=injection_cap_m3d, injection_basis=injection_basis,
        liquid_cap_m3d=liquid_cap_m3d)
    identifiers = [scenario["id"] for scenario in scenarios]
    if len(set(identifiers)) != len(identifiers):
        raise BankError("scenario identifiers are not unique")
    if set(identifiers) & HELD_OUT_TEST_IDS or any(
            name.startswith(HELD_OUT_TEST_PREFIXES) for name in identifiers):
        raise BankError("generated scenario collides with a frozen test identifier")
    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "months": len(months),
        "period": [months[0], months[-1]],
        "wells": len(wells),
        "injection_allocation": ("field target = min(phi * baseline produced water of the same "
                                 "month, injection cap); split across OPEN injectors in proportion "
                                 "to their baseline WRAT, converted wells weighted by their "
                                 "explicit WRAT"),
        "injection_cap_m3d": injection_cap_m3d,
        "injection_basis": injection_basis,
        "field_liquid_cap_m3d": liquid_cap_m3d,
        "well_liquid_cap_m3d": LRAT_CAP_M3D,
        "expected_semantics": "baseline arithmetic on control targets; not a physical forecast",
        "split": {"train": TRAIN_FRACTION, "validation": VALIDATION_FRACTION,
                  "rule": "sha256(scenario_id)[:16] / 2**64",
                  "frozen_test_ids": sorted(HELD_OUT_TEST_IDS),
                  "frozen_test_prefixes": list(HELD_OUT_TEST_PREFIXES)},
        "blocks": blocks_source,
        "scenarios": [],
    }
    return scenarios, manifest, {"months": months, "water_m3d": water_m3d}


def physical_entries(specifications: Sequence[str]) -> list[dict[str, Any]]:
    """F6: register already-computed candidates by their canonical export, without a new run."""
    entries = []
    for specification in specifications:
        scenario_id, _, raw = specification.partition("=")
        directory = Path(raw)
        if not _IDENTIFIER.fullmatch(scenario_id) or not raw or not directory.is_dir():
            raise BankError(f"physical entry must be id=canonical_export_dir: {specification!r}")
        files = {name: directory / name for name in ("chdd.csv", "trajectory.csv", "manifest.json")}
        absent = [name for name, path in files.items() if not path.is_file()]
        if absent:
            raise BankError(f"canonical export {directory} misses {absent}")
        entries.append({"id": scenario_id, "family": "F6", "split": split_of(scenario_id),
                        "parameters": {"canonical_export": str(directory.resolve())},
                        "request": None,
                        "canonical_sha256": {name: _digest(path) for name, path in files.items()}})
    return entries


def write_bank(
    output: Path,
    request: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    manifest: dict[str, Any],
    *,
    seed: int,
    water_m3d: Mapping[str, float],
    months: Sequence[str],
    physical: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    for scenario in scenarios:
        payload = scenario_request(request, scenario, seed=seed)
        cycle = CycleRequest.from_mapping(payload)
        path = output / f"request-{scenario['id']}.json"
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        manifest["scenarios"].append({
            "id": scenario["id"], "family": scenario["family"], "split": split_of(scenario["id"]),
            "parameters": scenario["parameters"], "request": path.name,
            "request_file_sha256": _digest(path), "controls_sha256": cycle.controls_sha256,
            "request_sha256": cycle.request_sha256,
            "expected": expected_field(scenario["controls"], months, water_m3d)})
    manifest["scenarios"].extend(physical)
    counts: dict[str, int] = defaultdict(int)
    for entry in manifest["scenarios"]:
        counts[entry["family"]] += 1
        counts[f"split:{entry['split']}"] += 1
    manifest["counts"] = dict(sorted(counts.items()))
    # A hash split is frozen and unbiased in expectation, but at bank size it is lumpy.
    # Report the realized imbalance instead of rebalancing: a moved holdout is a leak.
    thin = sorted(name for name in ("train", "validation", "test")
                  if counts[f"split:{name}"] < 3)
    manifest["split_warning"] = (
        f"under-populated splits at this bank size: {thin}; the frozen hash assignment is "
        "deliberately independent of bank composition and must not be rebalanced" if thin else None)
    manifest["script_sha256"] = _digest(Path(__file__))
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def _integers(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def self_check() -> None:
    month = "2007-01-01"
    baseline = [
        {"month": month, "well": "P", "role": "producer", "status": "OPEN", "target": "LRAT",
         "value": 400.0, "bhp_limit": 60.0},
        {"month": month, "well": "I", "role": "injector", "status": "OPEN", "target": "WRAT",
         "value": 200.0, "bhp_limit": 280.0},
        {"month": month, "well": "S", "role": "producer", "status": "SHUT", "target": "LRAT",
         "value": 0.0},
    ]
    controls = regime_controls(baseline, producer_scale=lambda *_: 2.0, phi=lambda _: 0.85,
                               water_m3d={month: 1000.0})
    assert [action["value"] for action in controls] == [500.0, 850.0, 0.0], controls
    check_invariants(controls, baseline)
    assert split_of("physical-sweep-03") == "test"
    assert split_of("fresh-uncertainty-01") == "test"
    assert split_of("feasible-f1-p085-m100") == split_of("feasible-f1-p085-m100")
    print("Water-limited allocation, LRAT clip and frozen split checks passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, help="baseline full-period CycleRequest JSON")
    parser.add_argument("--canonical", type=Path, help="baseline canonical chdd.csv")
    parser.add_argument("--export-manifest", type=Path, help="canonical export manifest.json")
    parser.add_argument("--output", type=Path, help="new bank directory (must not exist)")
    parser.add_argument("--blocks", type=Path, default=None, help="blocks.json from export_blocks.py")
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--shut-counts", type=_integers, default=(5, 10, 20))
    parser.add_argument("--shut-months", type=_integers, default=(24, 60))
    parser.add_argument("--switch-months", type=_integers, default=(60, 120))
    parser.add_argument("--conversion-anchor", default="104")
    parser.add_argument("--conversion-from-month", type=int, default=0)
    parser.add_argument("--injection-cap-m3d", type=float, default=INJECTION_CAP_M3D)
    parser.add_argument("--injection-basis", choices=("water", "cap"), default="water",
                        help="phi multiplies produced water (training deck) or the external cap (test case)")
    parser.add_argument("--liquid-cap-m3d", type=float, default=None,
                        help="field liquid cap; OPEN LRAT producers are scaled proportionally per month")
    parser.add_argument("--physical", action="append", default=[],
                        metavar="ID=DIR", help="F6: register a computed candidate's canonical export")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if not all((args.request, args.canonical, args.export_manifest, args.output)):
        parser.error("request, canonical, export-manifest and output are required")
    request = json.loads(args.request.read_text())
    export_manifest = json.loads(args.export_manifest.read_text())
    with args.canonical.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    scenarios, manifest, derived = build_bank(
        request, rows, export_manifest, seed=args.seed, blocks_path=args.blocks,
        block_count=args.block_count, shut_counts=args.shut_counts, shut_months=args.shut_months,
        switch_months=args.switch_months, conversion_anchor=args.conversion_anchor,
        conversion_from_month=args.conversion_from_month,
        injection_cap_m3d=args.injection_cap_m3d, injection_basis=args.injection_basis,
        liquid_cap_m3d=args.liquid_cap_m3d)
    manifest["inputs"] = {
        "request": str(args.request.resolve()), "request_sha256": _digest(args.request),
        "canonical": str(args.canonical.resolve()), "canonical_sha256": _digest(args.canonical),
        "export_manifest": str(args.export_manifest.resolve()),
        "export_manifest_sha256": _digest(args.export_manifest)}
    written = write_bank(args.output, request, scenarios, manifest, seed=args.seed,
                         water_m3d=derived["water_m3d"], months=derived["months"],
                         physical=physical_entries(args.physical))
    print(json.dumps({"output": str(args.output.resolve()), "counts": written["counts"]}), flush=True)


if __name__ == "__main__":
    main()
