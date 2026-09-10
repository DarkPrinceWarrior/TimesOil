"""Explicit case limits: surface rates, BHP and monthly water/voidage volumes."""

from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Mapping, Sequence


_LIMITS = {"max_oil_m3d", "max_liquid_m3d", "max_injection_m3d", "min_injection_m3d",
           "max_watercut", "min_bhp_bar", "max_bhp_bar",
           "max_monthly_water_deficit_m3", "min_monthly_voidage_replacement", "max_monthly_voidage_replacement"}
_WATER_LIMITS = {"max_monthly_water_deficit_m3", "min_monthly_voidage_replacement", "max_monthly_voidage_replacement"}


@dataclass(frozen=True, slots=True)
class OperatingConstraint:
    start: date
    end: date
    wells: tuple[str, ...]
    limits: tuple[tuple[str, float], ...] = ()
    unavailable: bool = False

    def __post_init__(self):
        if self.start.day != 1 or self.end.day != 1 or self.end < self.start:
            raise ValueError("operating constraint requires ordered monthly dates")
        if not self.wells or len(self.wells) != len(set(self.wells)):
            raise ValueError("operating constraint requires unique explicit wells")
        if any(not isinstance(w, str) or not w.strip() or w.strip() != w for w in self.wells):
            raise ValueError("invalid operating constraint well")
        if type(self.unavailable) is not bool or not (self.unavailable or self.limits):
            raise ValueError("operating constraint must impose a limit or unavailability")
        keys = [key for key, _ in self.limits]
        if len(keys) != len(set(keys)) or set(keys) - _LIMITS:
            raise ValueError("unknown or duplicate operating limit")
        for key, value in self.limits:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("operating limits must be finite nonnegative numbers")
            if key == "max_watercut" and value > 1:
                raise ValueError("max_watercut must be a fraction in [0, 1]")
        limits = dict(self.limits)
        for low, high in (("min_bhp_bar", "max_bhp_bar"), ("min_injection_m3d", "max_injection_m3d"),
                          ("min_monthly_voidage_replacement", "max_monthly_voidage_replacement")):
            if low in limits and high in limits and limits[low] > limits[high]:
                raise ValueError("minimum operating limit exceeds maximum")

    def to_dict(self):
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "wells": list(self.wells), "limits": dict(self.limits), "unavailable": self.unavailable}


def parse_constraints(raw, *, wells, start, end):
    if not isinstance(raw, list):
        raise ValueError("operating_constraints must be an array")
    result = []
    for item in raw:
        required = {"start", "end", "wells"}
        if not isinstance(item, dict) or not required <= set(item) <= required | {"limits", "unavailable"}:
            raise ValueError("unknown or missing operating constraint fields")
        if not isinstance(item["wells"], list) or not isinstance(item.get("limits", {}), dict):
            raise ValueError("operating constraint wells/limits have invalid types")
        rule = OperatingConstraint(date.fromisoformat(item["start"]), date.fromisoformat(item["end"]),
                                   tuple(item["wells"]), tuple(sorted(item.get("limits", {}).items())),
                                   item.get("unavailable", False))
        if not set(rule.wells) <= set(wells) or not start <= rule.start <= rule.end <= end:
            raise ValueError("operating constraint well or date is outside the case")
        result.append(rule)
    return tuple(result)


def check_controls(rules: Sequence[OperatingConstraint], actions):
    """Reject planned operation of unavailable wells and excessive known rate targets."""
    for rule in rules:
        for month in sorted({a.month for a in actions if rule.start <= a.month <= rule.end}):
            selected = [a for a in actions if a.month == month and a.well in rule.wells]
            if {a.well for a in selected} != set(rule.wells):
                raise ValueError("operating constraint control scope is incomplete")
            if rule.unavailable and any(a.status.value != "SHUT" or a.value != 0 for a in selected):
                raise ValueError(f"unavailable well is operated in {month}")
            for key, cap in rule.limits:
                target = {"max_oil_m3d": "ORAT", "max_liquid_m3d": "LRAT", "max_injection_m3d": "WRAT"}.get(key)
                if target and sum(a.value for a in selected if a.target.value == target) > cap + 1e-6:
                    raise ValueError(f"planned {key} exceeds organizer limit in {month}")


def check_observed(rules: Sequence[OperatingConstraint], month: date, values: Mapping[str, Mapping[str, float]]):
    """Check group rate totals and each well's water cut / BHP at reported endpoints."""
    for rule in rules:
        if not rule.start <= month <= rule.end:
            continue
        if not set(rule.wells) <= values.keys():
            raise ValueError("operating constraint observed scope is incomplete")
        selected = [values[well] for well in rule.wells]
        required = {"WLPR", "WWIR", "WBHP"}
        if any(key in {"max_oil_m3d", "max_watercut"} for key, _ in rule.limits):
            required.add("WOPR")
        if any(not required <= row.keys() or rule.unavailable and not {"WOPR", "WOMR"} & row.keys()
               for row in selected):
            raise ValueError("operating constraint observations miss required vectors")
        if any(not math.isfinite(row[k]) for row in selected
               for k in ("WOPR", "WOMR", "WLPR", "WWIR", "WBHP") if k in row):
            raise ValueError("non-finite operating constraint observations")
        # Oil mass can establish nonzero flow, but cannot replace WOPR for volume or water-cut limits.
        if rule.unavailable and any(abs(row[k]) > 1e-6 for row in selected
                                    for k in ("WOPR", "WOMR", "WLPR", "WWIR") if k in row):
            raise ValueError(f"unavailable well has physical flow in {month}")
        for key, bound in rule.limits:
            if key in _WATER_LIMITS:
                vectors = ("WWPT", "WWIT") if key == "max_monthly_water_deficit_m3" else ("WVPT", "WVIT")
                if any(f"{v}_DELTA" not in row or not math.isfinite(row[f"{v}_DELTA"])
                       or row[f"{v}_DELTA"] < 0 for row in selected for v in vectors):
                    raise ValueError("monthly water balance requires nonnegative cumulative increments")
                produced, injected = (sum(row[f"{v}_DELTA"] for row in selected) for v in vectors)
                if key == "max_monthly_water_deficit_m3":
                    observed, limit = max(0.0, injected - produced), bound
                else:
                    # Cross-multiplication avoids an undefined 0/0 in idle groups.
                    observed, limit = injected, bound * produced
                violates = observed < limit - 1e-6 if key.startswith("min_") else observed > limit + 1e-6
                if violates:
                    raise ValueError(f"observed {key} violates organizer limit in {month}")
                continue
            vector = {"max_oil_m3d": "WOPR", "max_liquid_m3d": "WLPR",
                      "max_injection_m3d": "WWIR", "min_injection_m3d": "WWIR"}.get(key)
            if vector:
                observed = [sum(row[vector] for row in selected)]
            elif key == "max_watercut":
                observed = [max(0.0, 1 - row["WOPR"] / row["WLPR"]) for row in selected if row["WLPR"] > 1e-6]
            else:
                observed = [row["WBHP"] for row in selected if row["WLPR"] > 1e-6 or row["WWIR"] > 1e-6]
            if any((value < bound - 1e-6 if key.startswith("min_") else value > bound + 1e-6) for value in observed):
                raise ValueError(f"observed {key} violates organizer limit in {month}")


def check_summary(rules, report, *, deck_dir, months, unit_system):
    if not rules:
        return
    from .opm_chdd import OpmChddError, _deck_text, _eclipse_date, _read_summary, _single_record

    if unit_system != "METRIC":
        raise ValueError("operating limits currently require a METRIC OPM model")
    try:
        summary, _ = _read_summary(report)
    except OpmChddError as exc:
        if str(exc) != "SUMMARY without DATE requires deck START and TIME":
            raise
        expanded, _ = _deck_text(deck_dir)
        summary, _ = _read_summary(report, start_date=_eclipse_date(_single_record(expanded, "START"), "START"))
    expected = {(month.replace(day=28) + timedelta(days=4)).replace(day=1): month for month in months}
    seen = set()
    previous = None
    for stamp, values, _ in summary:
        if stamp in expected:
            if stamp in seen:
                raise ValueError("duplicate operating constraint report month")
            month = expected[stamp]
            water_rules = [r for r in rules if r.start <= month <= r.end and set(dict(r.limits)) & _WATER_LIMITS]
            if water_rules:
                if previous is None or previous[0] != month:
                    raise ValueError("monthly water balance misses the start-of-month report")
                values = {well: dict(row) for well, row in values.items()}
                for rule in water_rules:
                    vectors = set()
                    for key, _ in rule.limits:
                        if key == "max_monthly_water_deficit_m3":
                            vectors.update(("WWPT", "WWIT"))
                        elif key in _WATER_LIMITS:
                            vectors.update(("WVPT", "WVIT"))
                    for well in rule.wells:
                        for vector in vectors:
                            if vector not in values[well] or vector not in previous[1][well]:
                                raise ValueError(f"monthly water balance misses {vector} for {well}")
                            delta = values[well][vector] - previous[1][well][vector]
                            if not math.isfinite(delta) or delta < -1e-6:
                                raise ValueError("monthly water balance cumulative volume decreased")
                            values[well][f"{vector}_DELTA"] = max(0.0, delta)
            check_observed(rules, month, values)
            seen.add(stamp)
        previous = stamp, values
    if seen != expected.keys():
        raise ValueError("operating constraint report misses management months")
