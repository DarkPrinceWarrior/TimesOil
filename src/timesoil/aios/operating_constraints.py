"""Explicit case limits: surface rates, BHP and monthly water/voidage volumes.

Semantics required by the organizers (DESIGN_TRACK2 §1.1):

* field liquid/injection caps are **monthly averages** -- a cumulative increment
  divided by the days in that month -- never an endpoint rate;
* voidage replacement is the ratio of sums over a moving window of monthly
  increments, so the first months of a horizon use the window available to them;
* BHP is checked on every SUMMARY record, not only on the monthly report dates,
  and the worst remaining margin is reported with its well and timestamp.

A rule carries a ``status``: a ``diagnostic`` rule is evaluated and reported but
never makes a candidate infeasible; only ``hard`` rules raise.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Mapping, Sequence

# Endpoint rate limits (legacy semantics, kept for callers that pass the old shapes).
_RATE_VECTORS = {"max_oil_m3d": "WOPR", "max_liquid_m3d": "WLPR",
                 "max_injection_m3d": "WWIR", "min_injection_m3d": "WWIR"}
# Monthly-average field limits: cumulative increment over the month / days in month.
_MONTHLY_AVERAGE_VECTORS = {"max_monthly_liquid_m3d": "WLPT", "max_monthly_injection_m3d": "WWIT"}
# Replacement ratios: (withdrawal vector, injection vector).
_RATIO_VECTORS = {"min_monthly_voidage_replacement": ("WVPT", "WVIT"),
                  "max_monthly_voidage_replacement": ("WVPT", "WVIT"),
                  "min_window_voidage_replacement": ("WVPT", "WVIT"),
                  "max_window_voidage_replacement": ("WVPT", "WVIT"),
                  "min_window_water_replacement": ("WWPT", "WWIT"),
                  "max_window_water_replacement": ("WWPT", "WWIT")}
_DEFICIT_VECTORS = {"max_monthly_water_deficit_m3": ("WWPT", "WWIT")}
_WINDOW_LIMITS = frozenset(key for key in _RATIO_VECTORS if "_window_" in key)
# Retained name and membership: the three original increment rules.
_WATER_LIMITS = frozenset(_DEFICIT_VECTORS) | {"min_monthly_voidage_replacement",
                                               "max_monthly_voidage_replacement"}
_INCREMENT_LIMITS = frozenset(_RATIO_VECTORS) | frozenset(_DEFICIT_VECTORS) | frozenset(_MONTHLY_AVERAGE_VECTORS)
_BHP_LIMITS = ("min_bhp_bar", "max_bhp_bar")
_LIMITS = (frozenset(_RATE_VECTORS) | frozenset(_INCREMENT_LIMITS)
           | {"max_watercut", "min_bhp_bar", "max_bhp_bar"})
_ORDERED_PAIRS = (("min_bhp_bar", "max_bhp_bar"), ("min_injection_m3d", "max_injection_m3d"),
                  ("min_monthly_voidage_replacement", "max_monthly_voidage_replacement"),
                  ("min_window_voidage_replacement", "max_window_voidage_replacement"),
                  ("min_window_water_replacement", "max_window_water_replacement"))
_STATUSES = ("hard", "diagnostic")


def increment_vectors(limits) -> tuple[str, ...]:
    """Cumulative SUMMARY vectors that must be differenced for these limit keys."""
    vectors: set[str] = set()
    for key in limits:
        if key in _MONTHLY_AVERAGE_VECTORS:
            vectors.add(_MONTHLY_AVERAGE_VECTORS[key])
        elif key in _RATIO_VECTORS:
            vectors.update(_RATIO_VECTORS[key])
        elif key in _DEFICIT_VECTORS:
            vectors.update(_DEFICIT_VECTORS[key])
    return tuple(sorted(vectors))


@dataclass(frozen=True, slots=True)
class Verdict:
    """Structured outcome of one limit; ``margin`` is signed slack, negative means violated."""

    rule: str
    status: str
    ok: bool
    worst_value: float | None
    margin: float | None
    month: date | None
    wells: tuple[str, ...]
    scope: str
    message: str = ""

    def to_dict(self):
        return {"rule": self.rule, "status": self.status, "ok": self.ok,
                "worst_value": self.worst_value, "margin": self.margin,
                "month": None if self.month is None else self.month.isoformat(),
                "wells": list(self.wells), "scope": self.scope, "message": self.message}


def failures(verdicts: Sequence[Verdict]) -> tuple[Verdict, ...]:
    """Hard verdicts that were violated; diagnostic ones never make a candidate infeasible."""
    return tuple(v for v in verdicts if not v.ok and v.status == "hard")


@dataclass(frozen=True, slots=True)
class OperatingConstraint:
    start: date
    end: date
    wells: tuple[str, ...]
    limits: tuple[tuple[str, float], ...] = ()
    unavailable: bool = False
    window_months: int = 1
    status: str = "hard"

    def __post_init__(self):
        if self.start.day != 1 or self.end.day != 1 or self.end < self.start:
            raise ValueError("operating constraint requires ordered monthly dates")
        if not self.wells or len(self.wells) != len(set(self.wells)):
            raise ValueError("operating constraint requires unique explicit wells")
        if any(not isinstance(w, str) or not w.strip() or w.strip() != w for w in self.wells):
            raise ValueError("invalid operating constraint well")
        if type(self.unavailable) is not bool or not (self.unavailable or self.limits):
            raise ValueError("operating constraint must impose a limit or unavailability")
        if self.status not in _STATUSES:
            raise ValueError("operating constraint status must be hard or diagnostic")
        if (isinstance(self.window_months, bool) or not isinstance(self.window_months, int)
                or not 1 <= self.window_months <= 60):
            raise ValueError("operating constraint window_months must be an integer in [1, 60]")
        keys = [key for key, _ in self.limits]
        if len(keys) != len(set(keys)) or set(keys) - _LIMITS:
            raise ValueError("unknown or duplicate operating limit")
        for key, value in self.limits:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("operating limits must be finite nonnegative numbers")
            if key == "max_watercut" and value > 1:
                raise ValueError("max_watercut must be a fraction in [0, 1]")
        limits = dict(self.limits)
        for low, high in _ORDERED_PAIRS:
            if low in limits and high in limits and limits[low] > limits[high]:
                raise ValueError("minimum operating limit exceeds maximum")

    def to_dict(self):
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "wells": list(self.wells), "limits": dict(self.limits), "unavailable": self.unavailable,
                "window_months": self.window_months, "status": self.status}


def parse_constraints(raw, *, wells, start, end):
    if not isinstance(raw, list):
        raise ValueError("operating_constraints must be an array")
    result = []
    for item in raw:
        required = {"start", "end", "wells"}
        optional = {"limits", "unavailable", "window_months", "status"}
        if not isinstance(item, dict) or not required <= set(item) <= required | optional:
            raise ValueError("unknown or missing operating constraint fields")
        if not isinstance(item["wells"], list) or not isinstance(item.get("limits", {}), dict):
            raise ValueError("operating constraint wells/limits have invalid types")
        rule = OperatingConstraint(date.fromisoformat(item["start"]), date.fromisoformat(item["end"]),
                                   tuple(item["wells"]), tuple(sorted(item.get("limits", {}).items())),
                                   item.get("unavailable", False), item.get("window_months", 1),
                                   item.get("status", "hard"))
        if not set(rule.wells) <= set(wells) or not start <= rule.start <= rule.end <= end:
            raise ValueError("operating constraint well or date is outside the case")
        result.append(rule)
    return tuple(result)


def window_months(rules: Sequence[OperatingConstraint]) -> int | None:
    """One window length for every moving-window rule, or None when no rule needs one."""
    lengths = {rule.window_months for rule in rules if set(dict(rule.limits)) & _WINDOW_LIMITS}
    if len(lengths) > 1:
        raise ValueError("moving-window operating rules must share one window length")
    return lengths.pop() if lengths else None


def check_controls(rules: Sequence[OperatingConstraint], actions):
    """Reject planned operation of unavailable wells and excessive known rate targets."""
    for rule in rules:
        if rule.status != "hard":
            continue
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


def own_control_constraints(actions):
    """Express validated schedule controls as limits on the resulting well response."""
    rules = []
    for action in actions:
        shut = action.status.value == 'SHUT'
        limits = []
        if not shut:
            rate = {'ORAT': 'max_oil_m3d', 'LRAT': 'max_liquid_m3d', 'WRAT': 'max_injection_m3d'}[action.target.value]
            limits.append((rate, action.value))
            limits.append(('max_liquid_m3d' if action.role.value == 'injector' else 'max_injection_m3d', 0.))
            if action.bhp_limit is not None:
                limits.append(('max_bhp_bar' if action.role.value == 'injector' else 'min_bhp_bar', action.bhp_limit))
        rules.append(OperatingConstraint(action.month, action.month, (action.well,), tuple(limits), shut))
    return tuple(rules)


def _increment(rows, key, suffix):
    """Sum one cumulative-increment channel over the group, rejecting missing or negative data."""
    totals = []
    for vector in _RATIO_VECTORS.get(key) or _DEFICIT_VECTORS.get(key) or (_MONTHLY_AVERAGE_VECTORS[key],):
        field = f"{vector}_{suffix}"
        if any(field not in row or not math.isfinite(row[field]) or row[field] < 0 for row in rows):
            raise ValueError("monthly cumulative increments must be present and nonnegative")
        totals.append(sum(row[field] for row in rows))
    return totals


def _ratio_verdict(rule, key, bound, rows, month, suffix):
    produced, injected = _increment(rows, key, suffix)
    # Cross-multiplication avoids an undefined 0/0 in idle groups.
    limit = bound * produced
    if key.startswith("min_"):
        ok, margin = injected >= limit - 1e-6, injected - limit
    else:
        ok, margin = injected <= limit + 1e-6, limit - injected
    ratio = injected / produced if produced > 1e-12 else (None if injected <= 1e-12 else math.inf)
    return Verdict(key, rule.status, ok, ratio, margin, month, rule.wells,
                   "window" if suffix == "WINDOW" else "month",
                   "" if ok else f"observed {key} violates organizer limit in {month}")


def evaluate_observed(rules: Sequence[OperatingConstraint], month: date,
                      values: Mapping[str, Mapping[str, float]]) -> tuple[Verdict, ...]:
    """Evaluate every rule active in ``month`` and return one verdict per limit."""
    verdicts: list[Verdict] = []
    days = calendar.monthrange(month.year, month.month)[1]
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
        if rule.unavailable:
            flow = max((abs(row[k]) for row in selected
                        for k in ("WOPR", "WOMR", "WLPR", "WWIR") if k in row), default=0.0)
            ok = flow <= 1e-6
            verdicts.append(Verdict("unavailable", rule.status, ok, flow, -flow, month, rule.wells, "month",
                                    "" if ok else f"unavailable well has physical flow in {month}"))
        for key, bound in rule.limits:
            if key in _RATIO_VECTORS:
                verdicts.append(_ratio_verdict(rule, key, bound, selected, month,
                                               "WINDOW" if key in _WINDOW_LIMITS else "DELTA"))
                continue
            if key in _DEFICIT_VECTORS:
                produced, injected = _increment(selected, key, "DELTA")
                deficit = max(0.0, injected - produced)
                ok = deficit <= bound + 1e-6
                verdicts.append(Verdict(key, rule.status, ok, deficit, bound - deficit, month, rule.wells, "month",
                                        "" if ok else f"observed {key} violates organizer limit in {month}"))
                continue
            if key in _MONTHLY_AVERAGE_VECTORS:
                total, = _increment(selected, key, "DELTA")
                average = total / days
                ok = average <= bound + 1e-6
                verdicts.append(Verdict(key, rule.status, ok, average, bound - average, month, rule.wells,
                                        "monthly_average",
                                        "" if ok else f"observed {key} violates organizer limit in {month}"))
                continue
            vector = _RATE_VECTORS.get(key)
            if vector:
                observed = [(None, sum(row[vector] for row in selected))]
            elif key == "max_watercut":
                observed = [(well, max(0.0, 1 - values[well]["WOPR"] / values[well]["WLPR"]))
                            for well in rule.wells if values[well]["WLPR"] > 1e-6]
            else:
                observed = [(well, values[well]["WBHP"]) for well in rule.wells
                            if values[well]["WLPR"] > 1e-6 or values[well]["WWIR"] > 1e-6]
            if not observed:
                continue
            worst_well, worst = (min(observed, key=lambda item: item[1]) if key.startswith("min_")
                                 else max(observed, key=lambda item: item[1]))
            margin = worst - bound if key.startswith("min_") else bound - worst
            ok = margin >= -1e-6
            verdicts.append(Verdict(key, rule.status, ok, worst, margin, month,
                                    rule.wells if worst_well is None else (worst_well,), "report_month",
                                    "" if ok else f"observed {key} violates organizer limit in {month}"))
    return tuple(verdicts)


def check_observed(rules: Sequence[OperatingConstraint], month: date, values: Mapping[str, Mapping[str, float]]):
    """Check group rate totals and each well's water cut / BHP at reported endpoints."""
    verdicts = evaluate_observed(rules, month, values)
    for verdict in failures(verdicts):
        raise ValueError(verdict.message)
    return verdicts


def _summary_rows(report, *, deck_dir, unit_system):
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
    return summary


def _row_month(stamp: date) -> date:
    """Month a SUMMARY record belongs to; a first-of-month stamp closes the previous month."""
    return (stamp - timedelta(days=1)).replace(day=1) if stamp.day == 1 else stamp.replace(day=1)


def evaluate_summary(rules, report, *, deck_dir, months, unit_system) -> tuple[Verdict, ...]:
    """Verdicts for every rule: monthly limits on report dates, BHP on every record."""
    if not rules:
        return ()
    summary = _summary_rows(report, deck_dir=deck_dir, unit_system=unit_system)
    expected = {(month.replace(day=28) + timedelta(days=4)).replace(day=1): month for month in months}
    window = window_months(rules) or 1
    seen: set[date] = set()
    previous = None
    history: list[dict[str, dict[str, float]]] = []
    verdicts: list[Verdict] = []
    # worst BHP over every SUMMARY record, not only the monthly report dates.
    extremes: dict[tuple[str, float, str], tuple[float, str, date]] = {}
    for stamp, values, _ in summary:
        row_month = _row_month(stamp)
        for rule in rules:
            if not rule.start <= row_month <= rule.end:
                continue
            for key, bound in rule.limits:
                if key not in _BHP_LIMITS:
                    continue
                if not set(rule.wells) <= values.keys():
                    raise ValueError("operating constraint observed scope is incomplete")
                for well in rule.wells:
                    row = values[well]
                    if not {"WLPR", "WWIR", "WBHP"} <= row.keys():
                        raise ValueError("operating constraint observations miss required vectors")
                    if not (row["WLPR"] > 1e-6 or row["WWIR"] > 1e-6):
                        continue
                    value = row["WBHP"]
                    if not math.isfinite(value):
                        raise ValueError("non-finite operating constraint observations")
                    slot = (key, bound, rule.status)
                    current = extremes.get(slot)
                    better = (current is None
                              or (value < current[0] if key.startswith("min_") else value > current[0]))
                    if better:
                        extremes[slot] = value, well, stamp
        if stamp in expected:
            if stamp in seen:
                raise ValueError("duplicate operating constraint report month")
            month = expected[stamp]
            active = [r for r in rules if r.start <= month <= r.end and set(dict(r.limits)) & _INCREMENT_LIMITS]
            if active:
                if previous is None or previous[0] != month:
                    raise ValueError("monthly water balance misses the start-of-month report")
                values = {well: dict(row) for well, row in values.items()}
                deltas: dict[str, dict[str, float]] = {}
                for rule in active:
                    for vector in increment_vectors(dict(rule.limits)):
                        for well in rule.wells:
                            if vector not in values[well] or vector not in previous[1][well]:
                                raise ValueError(f"monthly water balance misses {vector} for {well}")
                            delta = values[well][vector] - previous[1][well][vector]
                            if not math.isfinite(delta) or delta < -1e-6:
                                raise ValueError("monthly water balance cumulative volume decreased")
                            delta = max(0.0, delta)
                            values[well][f"{vector}_DELTA"] = delta
                            deltas.setdefault(well, {})[vector] = delta
                history.append(deltas)
                for well, row in deltas.items():
                    for vector, value in row.items():
                        values[well][f"{vector}_WINDOW"] = sum(
                            past.get(well, {}).get(vector, 0.0) for past in history[-window:])
            verdicts.extend(evaluate_observed(rules, month, values))
            seen.add(stamp)
        previous = stamp, values
    if seen != expected.keys():
        raise ValueError("operating constraint report misses management months")
    for (key, bound, status), (value, well, stamp) in sorted(extremes.items()):
        margin = value - bound if key.startswith("min_") else bound - value
        ok = margin >= -1e-6
        verdicts.append(Verdict(key, status, ok, value, margin, _row_month(stamp), (well,), "all_summary_steps",
                                "" if ok else f"observed {key} violates organizer limit at {stamp} in well {well}"))
    return tuple(verdicts)


def check_summary(rules, report, *, deck_dir, months, unit_system):
    """Raise on the first violated hard rule; return every verdict, diagnostics included."""
    verdicts = evaluate_summary(rules, report, deck_dir=deck_dir, months=months, unit_system=unit_system)
    for verdict in failures(verdicts):
        raise ValueError(verdict.message)
    return verdicts
