"""The case constraint profile is data, not code: one validated JSON file drives every gate.

Fields are exactly those of DESIGN_TRACK2_20260911 §1.4. Unknown keys are rejected,
ranges are checked, and the file hash travels into ``proposal-receipt.json``, into
every candidate record and into the seal, so a run states which rules it obeyed.

``operating_rules`` turns the profile into the ``OperatingConstraint`` rules of the
three gates. Nothing is invented here: an unrepresentable field (water carryover,
a numeric reservoir-pressure threshold) fails closed instead of being ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .operating_constraints import OperatingConstraint

_TOP_KEYS = {"liquid_cap_m3d", "injection_cap_m3d", "bhp_bounds", "vrr", "water_balance",
             "pressure", "repairs", "selection_margins"}
_VRR_KEYS = {"min", "max", "window_months", "denominator", "lower_bound_status"}
_WATER_KEYS = {"deficit_m3", "carryover"}
_PRESSURE_KEYS = {"field_min_bar", "block_min_bar", "regions"}
_MARGIN_KEYS = {"eps_liquid", "eps_injection", "phi"}
_REPAIR_KEYS = {"well", "start", "end"}
_DENOMINATORS = ("liquid_reservoir", "water_surface")
_LOWER_BOUND_STATUSES = ("hard", "diagnostic")
_REGIONS = ("FIP_C1", "FIP_ZONE", "ward6")


class CaseProfileError(ValueError):
    """The profile is not a valid case description; no gate may run without one."""


def _object(value, name, keys):
    if not isinstance(value, dict) or set(value) != keys:
        raise CaseProfileError(f"{name} must be an object with exactly {sorted(keys)}")
    return value


def _number(value, name, *, low=None, high=None, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CaseProfileError(f"{name} must be a finite number")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise CaseProfileError(f"{name} must lie in [{low}, {high}]")
    return value


def _choice(value, name, allowed, *, allow_none=False):
    if value is None and allow_none:
        return None
    if value not in allowed:
        raise CaseProfileError(f"{name} must be one of {list(allowed)}")
    return value


def _month(value, name):
    if not isinstance(value, str):
        raise CaseProfileError(f"{name} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise CaseProfileError(f"{name} is not an ISO date") from error
    if parsed.day != 1:
        raise CaseProfileError(f"{name} must be the first day of a month")
    return parsed


@dataclass(frozen=True, slots=True)
class CaseProfile:
    liquid_cap_m3d: float
    injection_cap_m3d: float
    bhp_bounds: tuple[float, float]
    vrr: Mapping[str, Any]
    water_balance: Mapping[str, Any]
    pressure: Mapping[str, Any]
    repairs: tuple[tuple[str, date, date], ...]
    selection_margins: Mapping[str, float]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"liquid_cap_m3d": self.liquid_cap_m3d, "injection_cap_m3d": self.injection_cap_m3d,
                "bhp_bounds": list(self.bhp_bounds), "vrr": dict(self.vrr),
                "water_balance": dict(self.water_balance), "pressure": dict(self.pressure),
                "repairs": [{"well": well, "start": start.isoformat(), "end": end.isoformat()}
                            for well, start, end in self.repairs],
                "selection_margins": dict(self.selection_margins)}

    def operating_rules(self, *, wells: Sequence[str], start: date, end: date) -> tuple[OperatingConstraint, ...]:
        """Field-wide gate rules for the management period [start, end] over the whole stock."""
        if start.day != 1 or end.day != 1 or end < start:
            raise CaseProfileError("ordered monthly management boundaries required")
        stock = tuple(dict.fromkeys(wells))
        if not stock:
            raise CaseProfileError("the case profile needs the explicit well stock")
        if self.water_balance["carryover"]:
            raise CaseProfileError("water carryover between months is not implemented; set carryover false")
        for key in ("field_min_bar", "block_min_bar"):
            if self.pressure[key] is not None:
                raise CaseProfileError(
                    "numeric reservoir pressure thresholds need FPR/RPR in the canonical export; "
                    "they are not checked here")
        low, high = self.bhp_bounds
        window = int(self.vrr["window_months"])
        ratio = "voidage" if self.vrr["denominator"] == "liquid_reservoir" else "water"
        rules = [
            OperatingConstraint(start, end, stock, (("max_monthly_liquid_m3d", self.liquid_cap_m3d),)),
            OperatingConstraint(start, end, stock, (("max_monthly_injection_m3d", self.injection_cap_m3d),)),
            OperatingConstraint(start, end, stock, (("min_bhp_bar", low), ("max_bhp_bar", high))),
            OperatingConstraint(start, end, stock, ((f"max_window_{ratio}_replacement", self.vrr["max"]),),
                                window_months=window),
            OperatingConstraint(start, end, stock, ((f"min_window_{ratio}_replacement", self.vrr["min"]),),
                                window_months=window, status=self.vrr["lower_bound_status"]),
            OperatingConstraint(start, end, stock,
                                (("max_monthly_water_deficit_m3", self.water_balance["deficit_m3"]),)),
        ]
        for well, first, last in self.repairs:
            if well not in stock:
                raise CaseProfileError(f"repair calendar names an unknown well: {well}")
            if not start <= first <= last <= end:
                continue
            rules.append(OperatingConstraint(first, last, (well,), (), True))
        return tuple(rules)


def parse_case_profile(data: Mapping[str, Any], *, sha256_hex: str) -> CaseProfile:
    """Strict schema: unknown keys, wrong types and out-of-range values are all rejected."""
    if not isinstance(data, dict) or set(data) != _TOP_KEYS:
        raise CaseProfileError(f"case profile must be an object with exactly {sorted(_TOP_KEYS)}")
    bounds = data["bhp_bounds"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise CaseProfileError("bhp_bounds must be a two-element list")
    low = _number(bounds[0], "bhp_bounds[0]", low=0)
    high = _number(bounds[1], "bhp_bounds[1]", low=0)
    if not 0 < low < high:
        raise CaseProfileError("bhp_bounds must satisfy 0 < min < max")

    raw_vrr = _object(data["vrr"], "vrr", _VRR_KEYS)
    vrr_min = _number(raw_vrr["min"], "vrr.min", low=0, high=10)
    vrr_max = _number(raw_vrr["max"], "vrr.max", low=0, high=10)
    if vrr_min > vrr_max:
        raise CaseProfileError("vrr.min exceeds vrr.max")
    window = raw_vrr["window_months"]
    if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 12:
        raise CaseProfileError("vrr.window_months must be an integer in [1, 12]")
    vrr = {"min": vrr_min, "max": vrr_max, "window_months": window,
           "denominator": _choice(raw_vrr["denominator"], "vrr.denominator", _DENOMINATORS),
           "lower_bound_status": _choice(raw_vrr["lower_bound_status"], "vrr.lower_bound_status",
                                         _LOWER_BOUND_STATUSES)}

    raw_water = _object(data["water_balance"], "water_balance", _WATER_KEYS)
    if type(raw_water["carryover"]) is not bool:
        raise CaseProfileError("water_balance.carryover must be a boolean")
    water = {"deficit_m3": _number(raw_water["deficit_m3"], "water_balance.deficit_m3", low=0),
             "carryover": raw_water["carryover"]}

    raw_pressure = _object(data["pressure"], "pressure", _PRESSURE_KEYS)
    pressure = {"field_min_bar": _number(raw_pressure["field_min_bar"], "pressure.field_min_bar",
                                         low=0, allow_none=True),
                "block_min_bar": _number(raw_pressure["block_min_bar"], "pressure.block_min_bar",
                                         low=0, allow_none=True),
                "regions": _choice(raw_pressure["regions"], "pressure.regions", _REGIONS, allow_none=True)}

    raw_margins = _object(data["selection_margins"], "selection_margins", _MARGIN_KEYS)
    margins = {"eps_liquid": _number(raw_margins["eps_liquid"], "selection_margins.eps_liquid", low=0, high=.5),
               "eps_injection": _number(raw_margins["eps_injection"], "selection_margins.eps_injection",
                                        low=0, high=.5),
               "phi": _number(raw_margins["phi"], "selection_margins.phi", low=0, high=1)}
    if margins["phi"] <= 0:
        raise CaseProfileError("selection_margins.phi must be positive")

    if not isinstance(data["repairs"], list):
        raise CaseProfileError("repairs must be an array")
    repairs = []
    for item in data["repairs"]:
        record = _object(item, "repair", _REPAIR_KEYS)
        well = record["well"]
        if not isinstance(well, str) or not well.strip() or well.strip() != well:
            raise CaseProfileError("repair.well must be a well name without padding")
        first, last = _month(record["start"], "repair.start"), _month(record["end"], "repair.end")
        if last < first:
            raise CaseProfileError("repair.end precedes repair.start")
        repairs.append((well, first, last))

    if not isinstance(sha256_hex, str) or len(sha256_hex) != 64 or sha256_hex.strip().lower() != sha256_hex:
        raise CaseProfileError("profile sha256 must be a lowercase hex digest")
    return CaseProfile(
        liquid_cap_m3d=_number(data["liquid_cap_m3d"], "liquid_cap_m3d", low=0),
        injection_cap_m3d=_number(data["injection_cap_m3d"], "injection_cap_m3d", low=0),
        bhp_bounds=(low, high), vrr=vrr, water_balance=water, pressure=pressure,
        repairs=tuple(repairs), selection_margins=margins, sha256=sha256_hex)


def load_case_profile(path: str | Path) -> CaseProfile:
    """Load and hash the profile; the hash is over the file bytes handed to the organizers."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise CaseProfileError(f"cannot read case profile: {source}") from error
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CaseProfileError(f"case profile is not valid JSON: {source}") from error
    return parse_case_profile(data, sha256_hex=sha256(raw).hexdigest())
