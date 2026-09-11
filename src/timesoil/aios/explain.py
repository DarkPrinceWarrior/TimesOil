"""Interpretability layer: numbers from the official calculator, never narrative.

Nothing here is official. Every function is pure over dicts/arrays so it can be
driven from stubs; the only external dependency is the organizers' calculator
module, imported read-only to reproduce its arithmetic exactly.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_CHDD_DIR = _REPO / "docs" / "hackathon" / "chdd" / "CHDD_PYTHON"

# Terms below are exact per well-month sums; `discount` closes FCF -> CHDD.
NPV_TERMS = ("revenue", "deductions", "oil_opex", "liquid_opex", "injection_opex",
             "fund_opex", "pump_capex", "pump_operation", "start_stop", "conversion",
             "property_tax", "other_included", "profit_tax", "discount")


def load_chdd_model(chdd_dir: str | Path | None = None):
    """Import the organizers' `chdd_model` without mutating `sys.path` permanently."""
    directory = Path(chdd_dir or _CHDD_DIR).resolve()
    path = directory / "chdd_model.py"
    if not path.is_file():
        raise FileNotFoundError(f"official calculator not found: {path}")
    name = f"_timesoil_chdd_model_{sha256(str(path).encode()).hexdigest()[:12]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official calculator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- S3


def _assumptions(model, norms: Mapping[str, Any] | None) -> dict[str, Any]:
    econ = dict(model.DEFAULT_ASSUMPTIONS)
    if norms:
        unknown = set(norms) - set(econ)
        if unknown:
            raise ValueError(f"unknown normative codes: {sorted(unknown)}")
        econ.update(norms)
    econ.update(model.METHODOLOGY_LOCKS)
    econ["chargeInitialPump"] = model.to_bool(econ.get("chargeInitialPump", False))
    return econ


def _pump_table(model, pumps: Sequence[Mapping[str, Any]] | None) -> list[dict[str, float]]:
    table = [{key: model.to_number(pump[key]) for key in ("nominal", "min", "max", "costM")}
             for pump in (pumps or model.DEFAULT_PUMPS)]
    if not table:
        raise ValueError("pump table is empty")
    table.sort(key=lambda pump: pump["nominal"])
    return table


def _allocate(weights: list[float], total: float) -> list[float]:
    """Split `total` over rows by nonnegative weight; equal shares when all weights vanish."""
    positive = sum(weights)
    if positive > 0:
        return [total * weight / positive for weight in weights]
    count = len(weights)
    return [total / count if count else 0.0] * count


def decompose_rows(rows: Iterable[Mapping[str, Any]], norms: Mapping[str, Any] | None = None, *,
                   start_year: int | None = None, pumps: Sequence[Mapping[str, Any]] | None = None,
                   chdd_dir: str | Path | None = None) -> dict[str, Any]:
    """Reproduce the calculator per well-month, asserting the sum against `totalChddM`.

    Attribution notes state which terms are not additive per well and how they are split.
    """
    model = load_chdd_model(chdd_dir)
    econ = _assumptions(model, norms)
    table = _pump_table(model, pumps)
    records = [dict(row) for row in rows]

    all_rows, diagnostics = model.normalize_rows(records)
    if not all_rows:
        raise ValueError("no valid rows to decompose")
    economic_rows = [row for row in all_rows
                     if row["WLPT_Diff"] >= 0 and row["WOMT_Diff"] >= 0 and row["WWIT_Diff"] >= 0]
    if not economic_rows:
        raise ValueError("every row carries a negative monthly delta")
    year = int(start_year) if start_year is not None else int(economic_rows[0]["DATA"][:4])
    calculation_start = f"{year}-01-01"
    filtered = [row for row in economic_rows if row["DATA"] >= calculation_start]
    if not filtered:
        raise ValueError("start year is later than the last row")

    pump_history = model._build_pump_history(economic_rows, table, econ, calculation_start, diagnostics)
    activity = model._build_activity_transitions(economic_rows, calculation_start)
    conversion = model._build_conversion_transitions(economic_rows, calculation_start, pump_history, econ)

    months: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        months.setdefault(row["DATA"][:7], []).append(row)

    cells: list[dict[str, Any]] = []
    month_taxable: dict[str, float] = {}
    for month in sorted(months):
        month_year = int(month[:4])
        discount_factor = 1 / ((1 + model.to_number(econ["waccRate"]) / 100) ** max(0, month_year - year))
        group = months[month]
        built: list[dict[str, Any]] = []
        for row in group:
            key = f"{row['well']}|{row['DATA']}"
            oil = max(0.0, row["WOMT_Diff"])
            pump_event = (pump_history["rowPump"].get(row["_row_id"]) or {}).get("pumpEvent")
            terms = {
                "revenue": oil * model.to_number(econ["oilPriceRubT"]) / 1e6,
                "deductions": -oil * model.to_number(econ["deductionsRubT"]) / 1e6,
                "oil_opex": -oil * model.to_number(econ["oilOpexRubT"]) / 1e6,
                "liquid_opex": -row["WLPT_Diff"] * model.to_number(econ["liquidOpexRubT"]) / 1e6,
                "injection_opex": -row["WWIT_Diff"] * model.to_number(econ["injectionOpexRubM3"]) / 1e6,
                "fund_opex": -(model.to_number(econ["fundAnnualRubWell"]) / 12 / 1e6
                               if row["WLPR"] > 0 or row["WWIR"] > 0 else 0.0),
                "pump_capex": -(pump_event["newPump"]["costM"] if pump_event else 0.0),
                "pump_operation": -(model.to_number(econ["pumpOperationCostM"]) if pump_event else 0.0),
                "start_stop": -(model.to_number(econ["stopStartCostM"])
                                if key in activity["byKey"] else 0.0),
                "conversion": -(conversion["byKey"][key]["totalEventCostM"]
                                if key in conversion["byKey"] else 0.0),
            }
            built.append({"well": row["well"], "date": row["DATA"], "month": month, "year": month_year,
                          "discount_factor": discount_factor, "oil_t": oil,
                          "liquid_t": row["WLPT_Diff"], "injection_m3": row["WWIT_Diff"], **terms})

        # Month-level items without a per-well basis: split by revenue share.
        property_tax = -((model.to_number(econ.get("residualStartM", econ["existingAssetResidualM"]))
                          + model.to_number(econ.get("residualEndM", econ["existingAssetResidualM"]))) / 2
                         * model.to_number(econ["propertyTaxRate"]) / 100 / 12)
        depreciation = -model.to_number(econ["annualDepreciationM"]) / 12
        other_included = -model.to_number(econ["otherIncludedEbitdaM"]) / 12
        other_excluded = -model.to_number(econ["otherExcludedEbitdaM"]) / 12
        weights = [max(0.0, cell["revenue"]) for cell in built]
        for name, total in (("property_tax", property_tax), ("_depreciation", depreciation),
                            ("other_included", other_included), ("_other_excluded", other_excluded)):
            for cell, share in zip(built, _allocate(weights, total)):
                cell[name] = share

        for cell in built:
            cell["_taxable"] = (cell["revenue"] + cell["deductions"] + cell["other_included"]
                                + cell["_other_excluded"] + cell["_depreciation"]
                                + sum(cell[name] for name in ("oil_opex", "liquid_opex", "injection_opex",
                                                              "fund_opex", "property_tax", "pump_operation",
                                                              "start_stop", "conversion")))
        month_taxable[month] = sum(cell["_taxable"] for cell in built)
        cells.extend(built)

    by_year: dict[int, list[str]] = {}
    for month in sorted(month_taxable):
        by_year.setdefault(int(month[:4]), []).append(month)
    month_tax: dict[str, float] = {}
    rate = model.to_number(econ["profitTaxRate"]) / 100
    for year_months in by_year.values():
        annual = max(0.0, sum(month_taxable[month] for month in year_months)) * rate
        positive = sum(max(0.0, month_taxable[month]) for month in year_months)
        for month in year_months:
            month_tax[month] = (annual * max(0.0, month_taxable[month]) / positive) if positive > 0 else 0.0

    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["month"], []).append(cell)
    for month, group in grouped.items():
        weights = [max(0.0, cell["_taxable"]) for cell in group]
        for cell, share in zip(group, _allocate(weights, -month_tax[month])):
            cell["profit_tax"] = share

    for cell in cells:
        fcf = sum(cell[name] for name in NPV_TERMS if name != "discount")
        cell["fcf_m"] = fcf
        cell["chdd_m"] = fcf * cell["discount_factor"]
        cell["discount"] = cell["chdd_m"] - fcf
        for name in ("_taxable", "_depreciation", "_other_excluded"):
            cell.pop(name)

    official = model.compute_calculation(records, assumptions=dict(econ), pumps=table,
                                         start_date=calculation_start)
    total = official["summary"]["totalChddM"]
    reconstructed = sum(cell["chdd_m"] for cell in cells)
    scale = max(abs(total), 1.0)
    assert abs(reconstructed - total) / scale < 1e-6, (
        f"per-well decomposition {reconstructed} does not reproduce totalChddM {total}")
    return {"cells": cells, "start_year": year, "total_chdd_m": total,
            "reconstructed_chdd_m": reconstructed,
            "relative_residual": abs(reconstructed - total) / scale,
            "summary": dict(official["summary"])}


def _fold(cells: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any],
          sign: float) -> dict[Any, dict[str, float]]:
    out: dict[Any, dict[str, float]] = {}
    for cell in cells:
        bucket = out.setdefault(key(cell), {name: 0.0 for name in (*NPV_TERMS, "chdd_m")})
        for name in NPV_TERMS:
            bucket[name] += sign * cell[name]
        bucket["chdd_m"] += sign * cell["chdd_m"]
    return out


def _merge(base: dict[Any, dict[str, float]], candidate: dict[Any, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for bucket in sorted(set(base) | set(candidate), key=str):
        zero = {name: 0.0 for name in (*NPV_TERMS, "chdd_m")}
        left, right = base.get(bucket, zero), candidate.get(bucket, zero)
        terms = {name: right[name] + left[name] for name in NPV_TERMS}  # base folded with sign -1
        rows.append({"key": bucket, "delta_chdd_m": right["chdd_m"] + left["chdd_m"], "terms": terms})
    return rows


def npv_decomposition(base_rows: Iterable[Mapping[str, Any]], cand_rows: Iterable[Mapping[str, Any]],
                      norms: Mapping[str, Any] | None = None, *, start_year: int | None = None,
                      pumps: Sequence[Mapping[str, Any]] | None = None,
                      block_of_well: Mapping[str, Any] | None = None,
                      chdd_dir: str | Path | None = None) -> dict[str, Any]:
    """Delta CHDD by calculator term x year x well x block, reconstructed from `chdd_model`."""
    base = decompose_rows(base_rows, norms, start_year=start_year, pumps=pumps, chdd_dir=chdd_dir)
    cand = decompose_rows(cand_rows, norms, start_year=start_year or base["start_year"],
                          pumps=pumps, chdd_dir=chdd_dir)
    delta_total = cand["total_chdd_m"] - base["total_chdd_m"]

    def pair(key: Callable[[Mapping[str, Any]], Any]) -> list[dict[str, Any]]:
        return _merge(_fold(base["cells"], key, -1.0), _fold(cand["cells"], key, +1.0))

    by_term = {name: sum(cell[name] for cell in cand["cells"]) - sum(cell[name] for cell in base["cells"])
               for name in NPV_TERMS}
    result: dict[str, Any] = {
        "official": False,
        "base_chdd_m": base["total_chdd_m"], "candidate_chdd_m": cand["total_chdd_m"],
        "delta_chdd_m": delta_total,
        "uplift_percent": (100 * (cand["total_chdd_m"] / base["total_chdd_m"] - 1)
                           if base["total_chdd_m"] > 0 else None),
        "by_term": by_term,
        "by_year": [{"year": row["key"], **{k: v for k, v in row.items() if k != "key"}}
                    for row in pair(lambda cell: cell["year"])],
        "by_well": sorted((({"well": row["key"], **{k: v for k, v in row.items() if k != "key"}})
                           for row in pair(lambda cell: cell["well"])),
                          key=lambda row: (-abs(row["delta_chdd_m"]), row["well"])),
        "by_block": None,
        "checks": {"base_relative_residual": base["relative_residual"],
                   "candidate_relative_residual": cand["relative_residual"],
                   "term_sum_minus_delta": sum(by_term.values()) - delta_total},
        "attribution_notes": [
            "revenue, deductions, oil/liquid/injection OPEX, well fund, pump CAPEX, pump operation, "
            "start/stop and conversion are exact per well-month sums of the calculator's own arithmetic",
            "profit tax is a yearly quantity: the calculator's monthly tax is split across wells "
            "proportionally to positive per-well taxable profit (equal shares when none is positive)",
            "property tax, depreciation and other EBITDA items have no per-well basis and are split "
            "proportionally to per-well revenue; they are zero under the shipped norms",
            "depreciation is omitted from the term list because it does not enter FCF, only taxable profit",
            "discount = chdd - fcf, so the terms sum to CHDD exactly within each bucket",
        ],
    }
    if block_of_well:
        blocks = pair(lambda cell: str(block_of_well.get(str(cell["well"]), "unassigned")))
        result["by_block"] = [{"block": row["key"], **{k: v for k, v in row.items() if k != "key"}}
                              for row in blocks]
    return result


# --------------------------------------------------------------------------- S3


def _pget(profile: Any, *path: str, default: Any = None) -> Any:
    node: Any = profile
    for key in path:
        if node is None:
            return default
        if isinstance(node, Mapping):
            node = node.get(key, None)
        else:
            node = getattr(node, key, None)
    return default if node is None else node


def _days_in_month(month: str) -> int:
    year, index = int(month[:4]), int(month[5:7])
    following = date(year + index // 12, index % 12 + 1, 1)
    return (following - date(year, index, 1)).days


def constraint_margins(rows: Iterable[Mapping[str, Any]], profile: Any, *,
                       summary_rows: Iterable[Mapping[str, Any]] | None = None,
                       densities: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Per-month field liquid/injection/VRR3 margins, per-well BHP margin, FPR vs floor.

    Liquid and VRR need tonnes -> m3 densities; without them those entries stay null.
    """
    liquid_cap = _pget(profile, "liquid_cap_m3d")
    injection_cap = _pget(profile, "injection_cap_m3d")
    bounds = _pget(profile, "bhp_bounds", default=[None, None])
    bhp_min, bhp_max = (list(bounds) + [None, None])[:2]
    vrr_min = _pget(profile, "vrr", "min")
    vrr_max = _pget(profile, "vrr", "max")
    window = int(_pget(profile, "vrr", "window_months", default=3))
    field_min_bar = _pget(profile, "pressure", "field_min_bar")
    repairs = [dict(item) for item in (_pget(profile, "repairs", default=[]) or [])]

    oil_density = float(densities["oil"]) if densities and "oil" in densities else None
    water_density = float(densities["water"]) if densities and "water" in densities else None
    notes: list[str] = []
    if oil_density is None or water_density is None:
        notes.append("liquid and VRR margins require densities {'oil': .., 'water': ..} in t/m3")

    repair_months: set[tuple[str, str]] = set()
    for item in repairs:
        well, start, end = str(item["well"]), str(item["start"])[:7], str(item["end"])[:7]
        month = start
        while month <= end:
            repair_months.add((well, month))
            year, index = int(month[:4]), int(month[5:7])
            month = f"{year + index // 12:04d}-{index % 12 + 1:02d}"

    monthly: dict[str, dict[str, float]] = {}
    well_bhp: dict[str, dict[str, Any]] = {}
    repair_activity: dict[tuple[str, str], float] = {}
    for raw in rows:
        month = str(raw["DATA"])[:7]
        well = str(raw["well"])
        bucket = monthly.setdefault(month, {"oil_t": 0.0, "liquid_t": 0.0, "injection_m3": 0.0})
        oil = max(0.0, float(raw.get("WOMT_Diff", 0.0) or 0.0))
        bucket["oil_t"] += oil
        bucket["liquid_t"] += float(raw.get("WLPT_Diff", 0.0) or 0.0)
        bucket["injection_m3"] += float(raw.get("WWIT_Diff", 0.0) or 0.0)
        liquid_rate = float(raw.get("WLPR", 0.0) or 0.0)
        injection_rate = float(raw.get("WWIR", 0.0) or 0.0)
        if (well, month) in repair_months:
            repair_activity[(well, month)] = max(repair_activity.get((well, month), 0.0),
                                                 liquid_rate, injection_rate)
            continue
        bhp = float(raw.get("BHP", 0.0) or 0.0)
        if liquid_rate > 0 and bhp_min is not None:
            margin = bhp - float(bhp_min)
        elif injection_rate > 0 and bhp_max is not None:
            margin = float(bhp_max) - bhp
        else:
            continue
        current = well_bhp.get(well)
        if current is None or margin < current["margin_bar"]:
            well_bhp[well] = {"well": well, "margin_bar": margin, "month": month, "bhp_bar": bhp,
                              "role": "producer" if liquid_rate > 0 else "injector"}

    ordered = sorted(monthly)
    produced_volume: dict[str, float | None] = {}
    for month in ordered:
        bucket = monthly[month]
        if oil_density and water_density:
            oil_m3 = bucket["oil_t"] / oil_density
            water_m3 = max(0.0, bucket["liquid_t"] - bucket["oil_t"]) / water_density
            produced_volume[month] = oil_m3 + water_m3
        else:
            produced_volume[month] = None

    months_out: list[dict[str, Any]] = []
    for index, month in enumerate(ordered):
        bucket = monthly[month]
        days = _days_in_month(month)
        liquid_m3d: float | None = None
        if oil_density and water_density:
            liquid_m3d = produced_volume[month] / days
        injection_m3d = bucket["injection_m3"] / days
        vrr3: float | None = None
        if index + 1 >= window and all(produced_volume[m] is not None for m in ordered[index + 1 - window:index + 1]):
            produced = sum(produced_volume[m] for m in ordered[index + 1 - window:index + 1])
            injected = sum(monthly[m]["injection_m3"] for m in ordered[index + 1 - window:index + 1])
            vrr3 = injected / produced if produced > 0 else None
        months_out.append({
            "month": month, "days": days,
            "liquid_m3d": liquid_m3d, "injection_m3d": injection_m3d, "vrr3": vrr3,
            "liquid_margin_m3d": (None if liquid_m3d is None or liquid_cap is None
                                  else float(liquid_cap) - liquid_m3d),
            "injection_margin_m3d": (None if injection_cap is None
                                     else float(injection_cap) - injection_m3d),
            "vrr3_above_min": (None if vrr3 is None or vrr_min is None else vrr3 - float(vrr_min)),
            "vrr3_below_max": (None if vrr3 is None or vrr_max is None else float(vrr_max) - vrr3),
        })

    def worst(key: str) -> dict[str, Any] | None:
        present = [row for row in months_out if row[key] is not None]
        return min(present, key=lambda row: row[key]) if present else None

    summary_list = [dict(row) for row in (summary_rows or [])]
    pressure: dict[str, Any] = {"rows": len(summary_list), "min_fpr_bar": None, "min_fpr_date": None,
                                "floor_bar": field_min_bar, "margin_bar": None}
    fpr = [(str(row.get("DATE") or row.get("DATA") or ""), float(row["FPR"]))
           for row in summary_list if row.get("FPR") is not None]
    if fpr:
        when, value = min(fpr, key=lambda item: item[1])
        pressure.update(min_fpr_bar=value, min_fpr_date=when,
                        margin_bar=None if field_min_bar is None else value - float(field_min_bar))
    elif summary_list:
        notes.append("summary rows carry no FPR column")

    return {
        "official": False, "notes": notes, "window_months": window,
        "months": months_out,
        "worst": {"liquid": worst("liquid_margin_m3d"), "injection": worst("injection_margin_m3d"),
                  "vrr3_above_min": worst("vrr3_above_min"), "vrr3_below_max": worst("vrr3_below_max")},
        "bhp": {"per_well": sorted(well_bhp.values(), key=lambda row: (row["margin_bar"], row["well"])),
                "minimum": min(well_bhp.values(), key=lambda row: (row["margin_bar"], row["well"]))
                if well_bhp else None},
        "pressure": pressure,
        "repairs": [{"well": str(item["well"]), "start": str(item["start"]), "end": str(item["end"]),
                     "months_observed": sum(1 for (well, _) in repair_activity
                                            if well == str(item["well"])),
                     "max_observed_rate": max((rate for (well, _), rate in repair_activity.items()
                                               if well == str(item["well"])), default=0.0),
                     "idle": all(rate <= 0 for (well, _), rate in repair_activity.items()
                                 if well == str(item["well"]))} for item in repairs],
    }


# --------------------------------------------------------------------------- S3


def _load_forecast(forecast_npz: Any) -> dict[str, Any]:
    import numpy as np

    payload = forecast_npz if isinstance(forecast_npz, Mapping) else dict(
        np.load(Path(forecast_npz), allow_pickle=False))
    prediction = np.asarray(payload["prediction"], dtype=float)
    wells = [str(value) for value in payload["well_ids"]]
    targets = [str(value) for value in payload["targets"]]
    stamps = [str(value) for value in payload["timestamps"]]
    if prediction.shape != (len(stamps), len(wells), len(targets)):
        raise ValueError("forecast prediction shape disagrees with its own axes")
    return {"prediction": prediction, "wells": wells, "targets": targets, "timestamps": stamps}


def forecast_vs_physics(forecast_npz: Any, physical_rows: Iterable[Mapping[str, Any]], *,
                        norms: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Per-well and per-target WAPE plus the money each well-term miss is worth."""
    import numpy as np

    forecast = _load_forecast(forecast_npz)
    wells, targets, stamps = forecast["wells"], forecast["targets"], forecast["timestamps"]
    index = {(str(stamp)[:10], str(well)): (row, column)
             for row, stamp in enumerate(stamps) for column, well in enumerate(wells)}
    truth = np.full(forecast["prediction"].shape, np.nan)
    for raw in physical_rows:
        position = index.get((str(raw["DATA"])[:10], str(raw["well"])))
        if position is None:
            continue
        for depth, field in enumerate(targets):
            value = raw.get(field)
            if value is not None and value != "":
                truth[position[0], position[1], depth] = float(value)
    covered = np.isfinite(truth)
    if not covered.any():
        raise ValueError("no physical row matches the forecast grid")

    def wape(pred: Any, true: Any, mask: Any) -> float | None:
        if not mask.any():
            return None
        denominator = float(np.abs(true[mask]).sum())
        if denominator == 0:
            return None
        return float(np.abs(pred[mask] - true[mask]).sum() / denominator)

    prediction = forecast["prediction"]
    by_target = {field: wape(prediction[..., depth], truth[..., depth], covered[..., depth])
                 for depth, field in enumerate(targets)}
    by_well = [{"well": well,
                "targets": {field: wape(prediction[:, column, depth], truth[:, column, depth],
                                        covered[:, column, depth])
                            for depth, field in enumerate(targets)}}
               for column, well in enumerate(wells)]

    econ = dict(norms or {})
    price = float(econ.get("oilPriceRubT", 28000.0))
    deductions = float(econ.get("deductionsRubT", 19600.0))
    oil_opex = float(econ.get("oilOpexRubT", 40.0))
    liquid_opex = float(econ.get("liquidOpexRubT", 100.0))
    injection_opex = float(econ.get("injectionOpexRubM3", 30.0))
    money = {"WOMT_Diff": (price - deductions - oil_opex) / 1e6,
             "WLPT_Diff": -liquid_opex / 1e6, "WWIT_Diff": -injection_opex / 1e6}
    shortfall: list[dict[str, Any]] = []
    for column, well in enumerate(wells):
        terms: dict[str, float] = {}
        for field, rate in money.items():
            if field not in targets:
                continue
            depth = targets.index(field)
            mask = covered[:, column, depth]
            if not mask.any():
                continue
            terms[field] = float((prediction[mask, column, depth] - truth[mask, column, depth]).sum() * rate)
        shortfall.append({"well": well, "terms": terms, "overstatement_m": sum(terms.values())})
    shortfall.sort(key=lambda row: (-abs(row["overstatement_m"]), row["well"]))

    return {"official": False, "months": len(stamps), "wells": len(wells), "targets": list(targets),
            "covered_cells": int(covered.sum()), "total_cells": int(covered.size),
            "wape_by_target": by_target, "wape_by_well": by_well,
            "money_shortfall_by_well": shortfall,
            "total_overstatement_m": sum(row["overstatement_m"] for row in shortfall),
            "notes": ["undiscounted margin terms; profit tax, fund, pump and event costs are excluded",
                      "a positive overstatement means the forecast promised more money than physics gave"]}


# --------------------------------------------------------------------------- S1


def search_trace(records: Sequence[Mapping[str, Any]], *, generation_key: str = "generation") -> dict[str, Any]:
    """Per-generation best/median CHDD, feasible share, sigma and injections."""
    def median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(record.get(generation_key, 0), []).append(record)
    generations = []
    for key in sorted(groups, key=lambda value: (value is None, value)):
        group = groups[key]
        values = [float(row["npv"]) for row in group
                  if row.get("npv") is not None and math.isfinite(float(row["npv"]))]
        feasible = [float(row["npv"]) for row in group if row.get("feasible")]
        sigmas = [float(row["sigma"]) for row in group if row.get("sigma") is not None]
        generations.append({
            "generation": key, "evaluated": len(group),
            "best_npv": max(values) if values else None,
            "best_feasible_npv": max(feasible) if feasible else None,
            "median_npv": median(values),
            "feasible_fraction": sum(1 for row in group if row.get("feasible")) / len(group) if group else None,
            "sigma": sigmas[0] if sigmas else None,
            "injections": sum(1 for row in group if row.get("injected")),
        })
    feasible_all = [row for row in records if row.get("feasible")]
    best = max(feasible_all, key=lambda row: float(row["npv"])) if feasible_all else None
    return {"official": False, "evaluated": len(records), "generations": generations,
            "feasible_total": len(feasible_all),
            "best": None if best is None else {"id": best.get("id"), "npv": float(best["npv"]),
                                               "generation": best.get(generation_key),
                                               "x": [float(v) for v in best.get("x", [])]}}


def sensitivity(records: Sequence[Mapping[str, Any]], *, param_names: Sequence[str] | None = None,
                min_step: float = 0.02, bounds: tuple[float, float] = (0.0, 1.0),
                feasible_only: bool = True) -> dict[str, Any]:
    """Finite differences over the already-evaluated population; no new forecasts.

    For each axis every point is paired with the population member closest to it in
    the other axes, giving d(npv)/d(x_j) without evaluating anything new.
    """
    import numpy as np

    population = [row for row in records
                  if row.get("x") is not None and row.get("npv") is not None
                  and (row.get("feasible") or not feasible_only)]
    if len(population) < 2:
        return {"official": False, "parameters": [], "evaluated": len(population),
                "notes": ["fewer than two evaluated points; no finite difference is possible"]}
    matrix = np.asarray([[float(value) for value in row["x"]] for row in population], dtype=float)
    values = np.asarray([float(row["npv"]) for row in population], dtype=float)
    if matrix.ndim != 2:
        raise ValueError("every trace entry needs an x vector of equal length")
    names = list(param_names) if param_names else [f"x{i}" for i in range(matrix.shape[1])]
    if len(names) != matrix.shape[1]:
        raise ValueError("param_names length disagrees with the x vectors")

    best = int(np.argmax(values))
    parameters = []
    # ponytail: O(n^2) pairing over the evaluated population; fine to a few thousand points.
    for axis, name in enumerate(names):
        others = np.delete(matrix, axis, axis=1)
        slopes: list[float] = []
        for i in range(len(matrix)):
            distance = np.abs(others - others[i]).max(axis=1)
            step = matrix[:, axis] - matrix[i, axis]
            eligible = np.where(np.abs(step) >= min_step)[0]
            if eligible.size == 0:
                continue
            partner = eligible[int(np.argmin(distance[eligible]))]
            slopes.append(float((values[partner] - values[i]) / step[partner]))
        column = matrix[:, axis]
        correlation = None
        if column.std() > 0 and values.std() > 0:
            correlation = float(np.corrcoef(column, values)[0, 1])
        parameters.append({
            "parameter": name, "index": axis, "pairs": len(slopes),
            "median_slope": float(np.median(slopes)) if slopes else None,
            "mean_slope": float(np.mean(slopes)) if slopes else None,
            "correlation": correlation,
            "best_value": float(matrix[best, axis]),
            "at_bounds": bool(abs(matrix[best, axis] - bounds[0]) < 1e-9
                              or abs(matrix[best, axis] - bounds[1]) < 1e-9),
        })
    return {"official": False, "evaluated": len(population), "min_step": min_step,
            "parameters": parameters,
            "winner_on_box_boundary": [row["parameter"] for row in parameters if row["at_bounds"]],
            "notes": ["slopes come from nearest-neighbour pairs inside the evaluated population",
                      "correlation is a fallback ranking when pairing is sparse"]}


def candidate_lineage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flat table: where each candidate came from, what it scored, why it was refused."""
    fields = ("id", "generation", "source", "parent_id", "injected", "npv", "feasible",
              "rejection", "rejected_reason", "policy_sha256", "controls_sha256")
    rows = []
    for record in records:
        row = {name: record[name] for name in fields if name in record}
        margins = record.get("margins")
        if isinstance(margins, Mapping):
            row["margins"] = {str(key): margins[key] for key in sorted(margins, key=str)}
        rows.append(row)
    rows.sort(key=lambda row: (row.get("generation") if row.get("generation") is not None else -1,
                               row.get("id") if row.get("id") is not None else -1))
    reasons: dict[str, int] = {}
    for row in rows:
        reason = row.get("rejection") or row.get("rejected_reason")
        if reason:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {"official": False, "count": len(rows), "candidates": rows, "rejection_counts": reasons}


# --------------------------------------------------------------------------- S2


def selection_summary(seal: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *,
                      selected_controls: Sequence[Mapping[str, Any]] | None = None,
                      baseline_controls: Sequence[Mapping[str, Any]] | None = None,
                      runners_up: int = 5) -> dict[str, Any]:
    """Forecast delta against the baseline forecast and the runners-up, plus control diff."""
    selected_id = seal.get("selected_id")
    forecast = seal.get("forecast_chdd_m")
    baseline = seal.get("baseline_forecast_chdd_m")
    scored = [row for row in candidates
              if isinstance(row.get("forecast_chdd_m"), (int, float))
              and not isinstance(row.get("forecast_chdd_m"), bool)]
    eligible = sorted((row for row in scored if row.get("forecast_eligible")),
                      key=lambda row: (-float(row["forecast_chdd_m"]), row.get("id", 0)))
    if baseline is None and scored:
        baseline = next((float(row["forecast_chdd_m"]) for row in scored if row.get("id") == 0), None)
    others = [row for row in eligible if row.get("id") != selected_id][:runners_up]

    notes: list[str] = []
    diff: list[dict[str, Any]] | None = None
    if selected_controls is None or baseline_controls is None:
        notes.append("control diff needs both selected_controls and baseline_controls")
    else:
        def index(actions: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
            return {(str(a["month"])[:10], str(a["well"])): dict(a) for a in actions}

        left, right = index(baseline_controls), index(selected_controls)
        diff = []
        for key in sorted(set(left) | set(right)):
            before, after = left.get(key), right.get(key)
            changed = {field: [None if before is None else before.get(field),
                               None if after is None else after.get(field)]
                       for field in ("role", "status", "target", "value", "bhp_limit")
                       if (None if before is None else before.get(field))
                       != (None if after is None else after.get(field))}
            if changed:
                diff.append({"month": key[0], "well": key[1], "changed": changed})

    return {
        "official": False,
        "selected_id": selected_id, "selection_metric": seal.get("selection_metric"),
        "forecast_chdd_m": forecast, "baseline_forecast_chdd_m": baseline,
        "delta_vs_baseline_m": None if forecast is None or baseline is None else forecast - baseline,
        "uplift_percent_vs_baseline": seal.get("forecast_uplift_percent"),
        "runners_up": [{"id": row.get("id"), "forecast_chdd_m": float(row["forecast_chdd_m"]),
                        "delta_vs_selected_m": (None if forecast is None
                                                else float(row["forecast_chdd_m"]) - forecast),
                        "policy": row.get("policy")} for row in others],
        "eligible_count": len(eligible), "scored_count": len(scored),
        "seal_contract": {"search_opm_calls": seal.get("search_opm_calls"),
                          "maximum_final_opm_calls": seal.get("maximum_final_opm_calls"),
                          "reselection_after_opm_allowed": seal.get("reselection_after_opm_allowed")},
        "control_diff": diff,
        "control_diff_count": None if diff is None else len(diff),
        "notes": notes,
    }


# --------------------------------------------------------------------- rationale

_CITATION = re.compile(r"^([\w\[\]\.]+):\s*(-?\d[\d\.,]*)")


def _to_float(text: str) -> float | None:
    cleaned = text.rstrip(".,")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(",") == 1:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _resolve(context: Any, path: str) -> tuple[bool, Any]:
    node = context
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            position = int(part[1:-1])
            if not isinstance(node, (list, tuple)) or not -len(node) <= position < len(node):
                return False, None
            node = node[position]
        elif isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return False, None
    return True, node


def check_evidence(decisions: Sequence[Mapping[str, Any]], context: Any, *,
                   tolerance: float = 0.005) -> dict[str, Any]:
    """Verify every `key: value` citation in agent evidence against the real context."""
    results = []
    totals = {"confirmed": 0, "not_found": 0, "differs": 0, "not_numeric": 0}
    for position, decision in enumerate(decisions):
        checks = []
        for item in decision.get("evidence", []) or []:
            text = str(item)
            match = _CITATION.match(text.strip())
            cited = None if match is None else _to_float(match.group(2))
            if match is None or cited is None:
                checks.append({"evidence": text, "status": "not_numeric"})
                totals["not_numeric"] += 1
                continue
            path = match.group(1)
            found, actual = _resolve(context, path)
            if not found:  # agents also cite their own tool output
                found, actual = _resolve(decision, path)
            if not found or isinstance(actual, bool) or not isinstance(actual, (int, float)):
                checks.append({"evidence": text, "path": path, "cited": cited,
                               "status": "not_found"})
                totals["not_found"] += 1
                continue
            actual = float(actual)
            scale = max(abs(actual), abs(cited), 1e-12)
            ok = abs(actual - cited) / scale <= tolerance
            checks.append({"evidence": text, "path": path, "cited": cited, "actual": actual,
                           "relative_error": abs(actual - cited) / scale,
                           "status": "confirmed" if ok else "differs"})
            totals["confirmed" if ok else "differs"] += 1
        results.append({"index": position, "role": decision.get("role"),
                        "approved": decision.get("approved"),
                        "summary_sha256": sha256(str(decision.get("summary", "")).encode()).hexdigest(),
                        "checks": checks})
    return {"official": False, "tolerance": tolerance, "decisions": results, "totals": totals,
            "notes": ["only numeric citations are verifiable; narrative text is not evidence"]}


# ------------------------------------------------------------------------ writer


def sha256_path(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def read_chdd_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a canonical chdd input CSV into calculator rows."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in value}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    for attribute in ("item", "tolist"):
        if hasattr(value, attribute):
            return _plain(getattr(value, attribute)())
    return str(value)


def _headline(name: str, section: Any) -> list[str]:
    if not isinstance(section, Mapping):
        return []
    keys = ("delta_chdd_m", "uplift_percent", "total_chdd_m", "candidate_chdd_m",
            "forecast_chdd_m", "delta_vs_baseline_m", "total_overstatement_m",
            "evaluated", "feasible_total", "control_diff_count", "eligible_count", "count")
    lines = [f"- `{key}`: {section[key]}" for key in keys if section.get(key) is not None]
    totals = section.get("totals")
    if isinstance(totals, Mapping):
        lines.append("- `totals`: " + ", ".join(f"{key}={totals[key]}" for key in sorted(totals)))
    worst = section.get("worst")
    if isinstance(worst, Mapping):
        for key in sorted(worst):
            if isinstance(worst[key], Mapping):
                lines.append(f"- worst `{key}`: {json.dumps(_plain(worst[key]), sort_keys=True, ensure_ascii=False)}")
    return lines


def write_report(directory: str | Path, sections: Mapping[str, Any], *,
                 inputs: Mapping[str, str | Path] | None = None,
                 title: str = "Interpretability report") -> dict[str, Any]:
    """Run each section under try/except and write `interpretability/*.json` + `summary.md`.

    `sections` maps a name to a zero-argument callable or a ready value. A failing
    section is recorded in `report.json`; the writer itself never raises on it.
    """
    root = Path(directory).resolve() / "interpretability"
    root.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {"official": False, "schema": "timesoil.interpretability/v1",
                              "inputs": {}, "sections": {}, "errors": {}}
    for name in sorted(dict(inputs or {})):
        path = Path(dict(inputs or {})[name])
        try:
            header["inputs"][name] = {"path": str(path), "sha256": sha256_path(path)}
        except OSError as exc:
            header["inputs"][name] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    written: dict[str, Any] = {}
    for name in sorted(sections):
        source = sections[name]
        try:
            value = source() if callable(source) else source
            payload = _plain(value)
            path = root / f"{name}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n", encoding="utf-8")
            header["sections"][name] = {"file": path.name, "sha256": sha256_path(path)}
            written[name] = payload
        except BaseException as exc:  # a broken section must never sink the report
            header["errors"][name] = f"{type(exc).__name__}: {exc}"
            header["sections"][name] = {"file": None}

    lines = [f"# {title}", "",
             "Not official. Numbers come from the organizers' calculator and from recorded artifacts;",
             "agent narrative is not evidence.", ""]
    if header["inputs"]:
        lines += ["## Inputs", ""]
        lines += [f"- `{name}`: `{header['inputs'][name].get('sha256', header['inputs'][name].get('error'))}`"
                  for name in sorted(header["inputs"])]
        lines.append("")
    for name in sorted(header["sections"]):
        lines.append(f"## {name}")
        lines.append("")
        if name in header["errors"]:
            lines += [f"Not computable: {header['errors'][name]}", ""]
            continue
        body = _headline(name, written.get(name))
        notes = written.get(name, {}).get("notes") if isinstance(written.get(name), Mapping) else None
        lines += body or ["(no headline figures)"]
        if isinstance(notes, list) and notes:
            lines += [""] + [f"> {note}" for note in notes]
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = root / "report.json"
    report.write_text(json.dumps(header, ensure_ascii=False, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return {"directory": str(root), "report": str(report), "summary": str(root / "summary.md"),
            "errors": dict(header["errors"]), "written": sorted(written)}
