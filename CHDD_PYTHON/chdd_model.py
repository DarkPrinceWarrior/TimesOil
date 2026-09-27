from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Iterable


VERSION = "7.0.2-negative-row-filter"

REQUIRED_COLUMNS = [
    "DATA",
    "well",
    "WLPT",
    "WLPR",
    "WOMT",
    "WOMR",
    "WWIR",
    "WWIT",
    "THP",
    "BHP",
    "WEFF",
    "WLPT_Diff",
    "WOMT_Diff",
    "WWIT_Diff",
]

DEFAULT_PUMPS = [
    {"nominal": 15, "min": 0, "max": 20, "costM": 0.55},
    {"nominal": 20, "min": 15, "max": 30, "costM": 0.65},
    {"nominal": 30, "min": 20, "max": 50, "costM": 0.75},
    {"nominal": 50, "min": 30, "max": 80, "costM": 0.90397},
    {"nominal": 80, "min": 50, "max": 100, "costM": 1.85},
    {"nominal": 100, "min": 80, "max": 125, "costM": 2.30},
    {"nominal": 125, "min": 100, "max": 160, "costM": 2.75},
    {"nominal": 160, "min": 125, "max": 210, "costM": 3.35},
    {"nominal": 210, "min": 160, "max": 250, "costM": 4.10},
    {"nominal": 250, "min": 210, "max": 320, "costM": 4.75},
    {"nominal": 320, "min": 250, "max": 400, "costM": 6.00},
    {"nominal": 400, "min": 320, "max": 500, "costM": 8.05},
]

DEFAULT_ASSUMPTIONS = {
    "oilPriceRubT": 28_000.0,
    "deductionsRubT": 19_600.0,
    "oilOpexRubT": 40.0,
    "liquidOpexRubT": 100.0,
    "injectionOpexRubM3": 30.0,
    "fundAnnualRubWell": 1_000_000.0,
    "pumpOperationCostM": 1.8,
    "profitTaxRate": 25.0,
    "propertyTaxRate": 2.2,
    "waccRate": 10.0,
    "annualDepreciationM": 0.0,
    "existingAssetResidualM": 0.0,
    "residualStartM": 0.0,
    "residualEndM": 0.0,
    "otherIncludedEbitdaM": 0.0,
    "otherExcludedEbitdaM": 0.0,
    "stopStartCostM": 1.0,
    "conversionBaseCostM": 5.0,
    "regimeThreshold": 10.0,
    "downsizeThreshold": 100.0,
    "chargeInitialPump": False,
    "clampNegativeDiffs": False,
    "allowTaxShield": False,
    "capitalizePumpAssets": False,
    "ignoreTechnicalReset": False,
}

METHODOLOGY_LOCKS = {
    "allowTaxShield": False,
    "capitalizePumpAssets": False,
    "clampNegativeDiffs": False,
    "ignoreTechnicalReset": False,
    "downsizeThreshold": 100.0,
    "stopStartCostM": 1.0,
    "conversionBaseCostM": 5.0,
}


def to_number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else 0.0
    text = str(value).strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return number if math.isfinite(number) else 0.0


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "да", "истина"}


def normalize_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    for pattern in ("%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    return None


def natural_key(value: Any) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


def _group_by(items: Iterable[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item[key]].append(item)
    return dict(grouped)


def _month_key(iso_date: str) -> str:
    return iso_date[:7]


def _year_of(iso_date: str) -> int:
    return int(iso_date[:4])


def normalize_rows(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics = {
        "ignoredBlankWells": 0,
        "invalidDates": 0,
        "negativeDiffRows": 0,
        "negativeOilRows": 0,
        "negativeNonOilDiffRows": 0,
        "duplicateWellMonths": 0,
        "outOfRangeRates": 0,
        "warnings": [],
    }
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for row_id, record in enumerate(records):
        well = str(record.get("well", "") if record.get("well") is not None else "").strip()
        if not well:
            diagnostics["ignoredBlankWells"] += 1
            continue
        row_date = normalize_date(record.get("DATA"))
        if not row_date:
            diagnostics["invalidDates"] += 1
            continue
        row: dict[str, Any] = {"DATA": row_date, "well": well, "_row_id": row_id}
        for column in REQUIRED_COLUMNS:
            if column not in {"DATA", "well"}:
                row[column] = to_number(record.get(column))
        if row["WLPT_Diff"] < 0 or row["WOMT_Diff"] < 0 or row["WWIT_Diff"] < 0:
            diagnostics["negativeDiffRows"] += 1
        if row["WOMT_Diff"] < 0:
            diagnostics["negativeOilRows"] += 1
        if row["WLPT_Diff"] < 0 or row["WWIT_Diff"] < 0:
            diagnostics["negativeNonOilDiffRows"] += 1
        duplicate_key = f"{well}|{row_date[:7]}"
        if duplicate_key in seen:
            diagnostics["duplicateWellMonths"] += 1
        seen.add(duplicate_key)
        rows.append(row)

    rows.sort(key=lambda item: (item["DATA"], natural_key(item["well"])))
    return rows, diagnostics


def validate_headers(headers: Iterable[str]) -> list[str]:
    present = set(headers)
    return [column for column in REQUIRED_COLUMNS if column not in present]


def _pick_initial_pump(rate: float, pumps: list[dict[str, float]]) -> dict[str, float]:
    candidates = [pump for pump in pumps if rate >= pump["min"] and rate <= pump["max"]]
    if candidates:
        return min(candidates, key=lambda pump: pump["nominal"])
    if rate > pumps[-1]["max"]:
        return pumps[-1]
    return pumps[0]


def _pick_upsize_pump(rate: float, current: dict[str, float], pumps: list[dict[str, float]]) -> dict[str, float]:
    candidates = [
        pump
        for pump in pumps
        if pump["nominal"] > current["nominal"] and rate >= pump["min"] and rate <= pump["max"]
    ]
    if candidates:
        return min(candidates, key=lambda pump: pump["nominal"])
    higher = [pump for pump in pumps if pump["nominal"] > current["nominal"]]
    return higher[-1] if higher else current


def _pick_downsize_pump(rate: float, current: dict[str, float], pumps: list[dict[str, float]]) -> dict[str, float]:
    candidates = [
        pump
        for pump in pumps
        if pump["nominal"] < current["nominal"] and rate >= pump["min"] and rate <= pump["max"]
    ]
    if candidates:
        return max(candidates, key=lambda pump: pump["nominal"])
    lower = [pump for pump in pumps if pump["nominal"] < current["nominal"]]
    return lower[0] if lower else current


def _build_pump_history(
    rows: list[dict[str, Any]],
    pumps: list[dict[str, float]],
    assumptions: dict[str, Any],
    start_date: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    row_pump: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    state_by_well: dict[str, dict[str, float] | None] = {}

    for well, well_rows in _group_by(rows, "well").items():
        sorted_rows = sorted(well_rows, key=lambda row: row["DATA"])
        first_calculation_date = next((row["DATA"] for row in sorted_rows if row["DATA"] >= start_date), None)
        current_pump: dict[str, float] | None = None
        previous: dict[str, Any] | None = None

        for row in sorted_rows:
            rate = max(0.0, row["WLPR"] or 0.0)
            pump_event: dict[str, Any] | None = None
            if rate > 0:
                if current_pump is None:
                    current_pump = _pick_initial_pump(rate, pumps)
                    if assumptions["chargeInitialPump"] and row["DATA"] >= start_date:
                        pump_event = {
                            "type": "initial_pump",
                            "oldPump": None,
                            "newPump": current_pump,
                            "costM": current_pump["costM"] + assumptions["pumpOperationCostM"],
                        }
                elif rate > current_pump["max"]:
                    next_pump = _pick_upsize_pump(rate, current_pump, pumps)
                    if next_pump["nominal"] != current_pump["nominal"]:
                        pump_event = {
                            "type": "pump_up",
                            "oldPump": current_pump,
                            "newPump": next_pump,
                            "costM": next_pump["costM"] + assumptions["pumpOperationCostM"],
                        }
                        current_pump = next_pump
                elif (current_pump["min"] - rate) > assumptions["downsizeThreshold"]:
                    next_pump = _pick_downsize_pump(rate, current_pump, pumps)
                    if next_pump["nominal"] != current_pump["nominal"]:
                        pump_event = {
                            "type": "pump_down",
                            "oldPump": current_pump,
                            "newPump": next_pump,
                            "costM": next_pump["costM"] + assumptions["pumpOperationCostM"],
                        }
                        current_pump = next_pump
                if rate > pumps[-1]["max"]:
                    diagnostics["outOfRangeRates"] += 1

            if (
                assumptions["chargeInitialPump"]
                and row["DATA"] == first_calculation_date
                and rate > 0
                and current_pump is not None
                and pump_event is None
            ):
                pump_event = {
                    "type": "initial_pump",
                    "oldPump": None,
                    "newPump": current_pump,
                    "costM": current_pump["costM"] + assumptions["pumpOperationCostM"],
                }

            pump_info = copy.deepcopy(current_pump) if current_pump else None
            row_pump[row["_row_id"]] = {"pump": pump_info, "pumpEvent": pump_event}

            if row["DATA"] >= start_date and previous is None and pump_event:
                events.append(
                    {
                        "id": f"{well}|{row['DATA']}|{pump_event['type']}",
                        "date": row["DATA"],
                        "month": _month_key(row["DATA"]),
                        "well": well,
                        "type": pump_event["type"],
                        "oldRate": 0.0,
                        "newRate": rate,
                        "deltaRate": rate,
                        "oldPump": None,
                        "newPump": pump_info["nominal"] if pump_info else None,
                        "pumpCapexM": pump_event["newPump"]["costM"],
                        "operationCostM": assumptions["pumpOperationCostM"],
                        "totalEventCostM": pump_event["costM"],
                    }
                )

            if row["DATA"] >= start_date and previous is not None:
                previous_rate = max(0.0, previous["WLPR"] or 0.0)
                delta = rate - previous_rate
                event_type: str | None = None
                if pump_event:
                    event_type = pump_event["type"]
                elif previous_rate <= 0 and rate > 0:
                    event_type = "start"
                elif previous_rate > 0 and rate <= 0 and not (row["WWIR"] > 0):
                    event_type = "stop"
                elif previous_rate > 0 and rate > 0 and abs(delta) >= assumptions["regimeThreshold"]:
                    event_type = "rate_up" if delta > 0 else "rate_down"

                if event_type:
                    previous_pump = row_pump.get(previous["_row_id"], {}).get("pump")
                    events.append(
                        {
                            "id": f"{well}|{row['DATA']}|{event_type}",
                            "date": row["DATA"],
                            "month": _month_key(row["DATA"]),
                            "well": well,
                            "type": event_type,
                            "oldRate": previous_rate,
                            "newRate": rate,
                            "deltaRate": delta,
                            "oldPump": (
                                pump_event["oldPump"]["nominal"]
                                if pump_event and pump_event["oldPump"]
                                else previous_pump["nominal"] if previous_pump else None
                            ),
                            "newPump": pump_info["nominal"] if pump_info else None,
                            "pumpCapexM": pump_event["newPump"]["costM"] if pump_event else 0.0,
                            "operationCostM": assumptions["pumpOperationCostM"] if pump_event else 0.0,
                            "totalEventCostM": pump_event["costM"] if pump_event else 0.0,
                        }
                    )
            previous = row

        state_by_well[well] = copy.deepcopy(current_pump) if current_pump else None

    return {"rowPump": row_pump, "events": events, "stateByWell": state_by_well}


def _build_activity_transitions(rows: list[dict[str, Any]], start_date: str) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for well, well_rows in _group_by(rows, "well").items():
        previous: dict[str, Any] | None = None
        for row in sorted(well_rows, key=lambda item: item["DATA"]):
            if previous is not None and row["DATA"] >= start_date:
                previous_active = previous["WLPR"] > 0 or previous["WWIR"] > 0
                active = row["WLPR"] > 0 or row["WWIR"] > 0
                if previous_active != active:
                    event = {
                        "well": well,
                        "date": row["DATA"],
                        "month": _month_key(row["DATA"]),
                        "type": "start" if active else "stop",
                        "prevActive": previous_active,
                        "active": active,
                    }
                    transitions.append(event)
                    by_key[f"{well}|{row['DATA']}"] = event
            previous = row
    return {"transitions": transitions, "byKey": by_key}


def _build_conversion_transitions(
    rows: list[dict[str, Any]],
    start_date: str,
    pump_history: dict[str, Any],
    assumptions: dict[str, Any],
) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for well, well_rows in _group_by(rows, "well").items():
        previous: dict[str, Any] | None = None
        for row in sorted(well_rows, key=lambda item: item["DATA"]):
            if previous is not None and row["DATA"] >= start_date:
                was_producer = previous["WLPR"] > 0 or previous["WOMR"] > 0
                became_injector = row["WWIR"] > 0 and row["WLPR"] <= 0 and row["WOMR"] <= 0
                if was_producer and became_injector:
                    previous_pump = pump_history["rowPump"].get(previous["_row_id"], {}).get("pump")
                    base_cost = float(assumptions["conversionBaseCostM"])
                    event = {
                        "id": f"{well}|{row['DATA']}|conversion_to_injection",
                        "well": well,
                        "date": row["DATA"],
                        "month": _month_key(row["DATA"]),
                        "type": "conversion_to_injection",
                        "oldRate": max(0.0, previous["WLPR"] or 0.0),
                        "newRate": 0.0,
                        "newInjectionRate": max(0.0, row["WWIR"] or 0.0),
                        "deltaRate": -max(0.0, previous["WLPR"] or 0.0),
                        "oldPump": previous_pump["nominal"] if previous_pump else None,
                        "newPump": None,
                        "pumpCapexM": 0.0,
                        "conversionPumpCostM": 0.0,
                        "conversionBaseCostM": base_cost,
                        "operationCostM": base_cost,
                        "totalEventCostM": base_cost,
                    }
                    transitions.append(event)
                    by_key[f"{well}|{row['DATA']}"] = event
                    by_month[event["month"]].append(event)
            previous = row
    return {"transitions": transitions, "byKey": by_key, "byMonth": dict(by_month)}


def _build_warnings(diagnostics: dict[str, Any], assumptions: dict[str, Any], max_date: str) -> list[str]:
    warnings: list[str] = []
    if diagnostics.get("excludedNegativeRows"):
        dates = ", ".join(diagnostics.get("excludedNegativeDates", []))
        warnings.append(
            f"Из экономического расчета полностью исключено "
            f"{diagnostics['excludedNegativeRows']} строк с отрицательным WLPT_Diff, WOMT_Diff "
            f"или WWIT_Diff{f' (даты: {dates})' if dates else ''}."
        )
    if diagnostics["ignoredBlankWells"]:
        warnings.append(f"Пропущено строк без номера скважины: {diagnostics['ignoredBlankWells']}.")
    if diagnostics["invalidDates"]:
        warnings.append(f"Пропущено строк с некорректной датой: {diagnostics['invalidDates']}.")
    if diagnostics["duplicateWellMonths"]:
        warnings.append(
            f"Найдены дубли «скважина-месяц»: {diagnostics['duplicateWellMonths']}. Проверьте исходные данные."
        )
    if diagnostics["outOfRangeRates"]:
        warnings.append(
            f"Найдено значений WLPR выше 500 м³/сут: {diagnostics['outOfRangeRates']}. "
            "Такой сценарий не соответствует методике."
        )
    return warnings


def _source_cumulative_by_year(rows: list[dict[str, Any]], years: list[int]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for year in years:
        cutoff = f"{year}-12-31"
        last_by_well: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["DATA"] <= cutoff:
                last_by_well[row["well"]] = row
        result[year] = {
            "sourceCumulativeLiquidKt": sum(row["WLPT"] for row in last_by_well.values()) / 1000,
            "sourceCumulativeOilKt": sum(row["WOMT"] for row in last_by_well.values()) / 1000,
            "sourceCumulativeInjectionKm3": sum(row["WWIT"] for row in last_by_well.values()) / 1000,
        }
    return result


def compute_calculation(
    records: list[dict[str, Any]],
    *,
    headers: Iterable[str] | None = None,
    assumptions: dict[str, Any] | None = None,
    pumps: list[dict[str, Any]] | None = None,
    start_date: Any = None,
    name: str = "Расчет ЧДД",
    source_file: str = "",
) -> dict[str, Any]:
    actual_headers = list(headers or (records[0].keys() if records else []))
    missing = validate_headers(actual_headers)
    if missing:
        raise ValueError(f"В исходных данных отсутствуют обязательные столбцы: {', '.join(missing)}")

    econ = copy.deepcopy(DEFAULT_ASSUMPTIONS)
    if assumptions:
        econ.update(assumptions)
    for key, locked_value in METHODOLOGY_LOCKS.items():
        econ[key] = locked_value
    econ["chargeInitialPump"] = to_bool(econ.get("chargeInitialPump", False))

    pump_table = [
        {
            "nominal": to_number(pump["nominal"]),
            "min": to_number(pump["min"]),
            "max": to_number(pump["max"]),
            "costM": to_number(pump["costM"]),
        }
        for pump in (pumps or DEFAULT_PUMPS)
    ]
    pump_table.sort(key=lambda pump: pump["nominal"])
    if not pump_table:
        raise ValueError("Таблица ЭЦН пуста.")

    all_rows, diagnostics = normalize_rows(records)
    if not all_rows:
        raise ValueError("В исходных данных нет корректных строк для расчета.")
    source_max_date = all_rows[-1]["DATA"]
    economic_rows = [
        row
        for row in all_rows
        if row["WLPT_Diff"] >= 0 and row["WOMT_Diff"] >= 0 and row["WWIT_Diff"] >= 0
    ]
    excluded_negative_rows = [
        row
        for row in all_rows
        if row["WLPT_Diff"] < 0 or row["WOMT_Diff"] < 0 or row["WWIT_Diff"] < 0
    ]
    diagnostics["excludedNegativeRows"] = len(excluded_negative_rows)
    diagnostics["excludedNegativeDates"] = sorted({row["DATA"] for row in excluded_negative_rows})
    diagnostics["sourceMaxDate"] = source_max_date
    if not economic_rows:
        raise ValueError(
            "После исключения строк с отрицательными месячными значениями "
            "не осталось данных для экономического расчета."
        )
    min_date = economic_rows[0]["DATA"]
    max_date = economic_rows[-1]["DATA"]
    requested_start = normalize_date(start_date) or min_date
    start_year = _year_of(requested_start)
    calculation_start_date = f"{start_year}-01-01"
    filtered_rows = [row for row in economic_rows if row["DATA"] >= calculation_start_date]
    if not filtered_rows:
        raise ValueError("Стартовый год находится позже последней даты в исходных данных.")

    excessive_rates = [row for row in filtered_rows if row["WLPR"] > 500]
    if excessive_rates:
        examples = ", ".join(
            f"{row['well']} ({row['DATA']}: {row['WLPR']:g})" for row in excessive_rates[:5]
        )
        raise ValueError(
            "Методика не допускает WLPR свыше 500 м³/сут. "
            f"Исправьте {len(excessive_rates)} значений" + (f": {examples}." if examples else ".")
        )

    pump_history = _build_pump_history(economic_rows, pump_table, econ, calculation_start_date, diagnostics)
    activity_history = _build_activity_transitions(economic_rows, calculation_start_date)
    conversion_history = _build_conversion_transitions(
        economic_rows, calculation_start_date, pump_history, econ
    )
    decision_events = pump_history["events"] + conversion_history["transitions"]

    month_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered_rows:
        month_rows[_month_key(row["DATA"])].append(row)

    monthly: list[dict[str, Any]] = []
    for month in sorted(month_rows):
        year = int(month[:4])
        discount_factor = 1 / ((1 + to_number(econ["waccRate"]) / 100) ** max(0, year - start_year))
        item: dict[str, Any] = {
            "month": month,
            "year": year,
            "discountFactor": discount_factor,
            "oilT": 0.0,
            "liquidT": 0.0,
            "injectionM3": 0.0,
            "activeWells": 0,
            "avgTHP": 0.0,
            "avgBHP": 0.0,
            "avgWLPR": 0.0,
            "avgWOMR": 0.0,
            "avgWWIR": 0.0,
            "thpCount": 0,
            "bhpCount": 0,
            "wlprCount": 0,
            "womrCount": 0,
            "wwirCount": 0,
            "pumpCapexM": 0.0,
            "pumpOperationM": 0.0,
            "pumpChanges": 0,
            "pumpInstallCounts": {},
            "startStopCount": 0,
            "conversionCount": 0,
            "conversionOpexM": 0.0,
        }

        for row in month_rows[month]:
            # Negative monthly oil deltas are technical corrections, not production.
            item["oilT"] += max(0.0, row["WOMT_Diff"])
            item["liquidT"] += row["WLPT_Diff"]
            item["injectionM3"] += row["WWIT_Diff"]
            if row["WLPR"] > 0 or row["WWIR"] > 0:
                item["activeWells"] += 1
            for source_key, avg_key, count_key in (
                ("THP", "avgTHP", "thpCount"),
                ("BHP", "avgBHP", "bhpCount"),
                ("WLPR", "avgWLPR", "wlprCount"),
                ("WOMR", "avgWOMR", "womrCount"),
                ("WWIR", "avgWWIR", "wwirCount"),
            ):
                if row[source_key] > 0:
                    item[avg_key] += row[source_key]
                    item[count_key] += 1

            if f"{row['well']}|{row['DATA']}" in activity_history["byKey"]:
                item["startStopCount"] += 1
            row_pump = pump_history["rowPump"].get(row["_row_id"])
            if row_pump and row_pump["pumpEvent"] and row["DATA"] >= calculation_start_date:
                pump_event = row_pump["pumpEvent"]
                nominal = pump_event["newPump"]["nominal"]
                item["pumpChanges"] += 1
                item["pumpCapexM"] += pump_event["newPump"]["costM"]
                item["pumpOperationM"] += to_number(econ["pumpOperationCostM"])
                item["pumpInstallCounts"][str(int(nominal) if nominal.is_integer() else nominal)] = (
                    item["pumpInstallCounts"].get(str(int(nominal) if nominal.is_integer() else nominal), 0) + 1
                )

        for avg_key, count_key in (
            ("avgTHP", "thpCount"),
            ("avgBHP", "bhpCount"),
            ("avgWLPR", "wlprCount"),
            ("avgWOMR", "womrCount"),
            ("avgWWIR", "wwirCount"),
        ):
            if item[count_key]:
                item[avg_key] /= item[count_key]

        month_conversions = conversion_history["byMonth"].get(month, [])
        item["conversionCount"] = len(month_conversions)
        item["conversionOpexM"] = sum(event["totalEventCostM"] for event in month_conversions)

        item["baseDepreciationM"] = to_number(econ["annualDepreciationM"]) / 12
        item["depreciationM"] = item["baseDepreciationM"]
        item["residualStartM"] = to_number(econ.get("residualStartM", econ["existingAssetResidualM"]))
        item["residualEndM"] = to_number(econ.get("residualEndM", econ["existingAssetResidualM"]))
        item["propertyTaxM"] = (
            (item["residualStartM"] + item["residualEndM"])
            / 2
            * to_number(econ["propertyTaxRate"])
            / 100
            / 12
        )
        item["oilPriceRubT"] = to_number(econ["oilPriceRubT"])
        item["mainRevenueM"] = item["oilT"] * item["oilPriceRubT"] / 1_000_000
        item["revenueM"] = item["mainRevenueM"]
        item["capexM"] = item["pumpCapexM"]
        item["oilOpexM"] = item["oilT"] * to_number(econ["oilOpexRubT"]) / 1_000_000
        item["liquidOpexM"] = item["liquidT"] * to_number(econ["liquidOpexRubT"]) / 1_000_000
        item["injectionOpexM"] = (
            item["injectionM3"] * to_number(econ["injectionOpexRubM3"]) / 1_000_000
        )
        item["fundOpexM"] = (
            item["activeWells"] * to_number(econ["fundAnnualRubWell"]) / 12 / 1_000_000
        )
        item["gtmM"] = item["pumpOperationM"]
        item["startStopCostM"] = item["startStopCount"] * to_number(econ["stopStartCostM"])
        item["mainOpexM"] = sum(
            item[key]
            for key in (
                "oilOpexM",
                "liquidOpexM",
                "injectionOpexM",
                "fundOpexM",
                "propertyTaxM",
                "gtmM",
                "startStopCostM",
                "conversionOpexM",
            )
        )
        item["opexM"] = item["mainOpexM"]
        item["deductionsM"] = item["oilT"] * to_number(econ["deductionsRubT"]) / 1_000_000
        item["otherIncludedEbitdaM"] = to_number(econ["otherIncludedEbitdaM"]) / 12
        item["otherExcludedEbitdaM"] = to_number(econ["otherExcludedEbitdaM"]) / 12
        item["taxableProfitM"] = (
            item["revenueM"]
            - item["opexM"]
            - item["depreciationM"]
            - item["deductionsM"]
            - item["otherIncludedEbitdaM"]
            - item["otherExcludedEbitdaM"]
        )
        item["profitTaxM"] = max(0.0, item["taxableProfitM"]) * to_number(econ["profitTaxRate"]) / 100
        item["netProfitM"] = item["taxableProfitM"] - item["profitTaxM"]
        item["ebitdaM"] = (
            item["revenueM"]
            - item["opexM"]
            - item["deductionsM"]
            - item["otherIncludedEbitdaM"]
        )
        item["fcfM"] = item["ebitdaM"] - item["profitTaxM"] - item["capexM"]
        item["chddM"] = item["fcfM"] * item["discountFactor"]
        other_income_expense = item["otherIncludedEbitdaM"] + item["otherExcludedEbitdaM"]
        item["discountedInflowM"] = (
            item["revenueM"] + max(0.0, -other_income_expense)
        ) * item["discountFactor"]
        item["discountedOutflowM"] = (
            item["capexM"]
            + item["opexM"]
            + item["deductionsM"]
            + item["profitTaxM"]
            + max(0.0, other_income_expense)
        ) * item["discountFactor"]
        monthly.append(item)

    months_by_year = _group_by(monthly, "year")
    for year_months in months_by_year.values():
        annual_taxable = sum(item["taxableProfitM"] for item in year_months)
        annual_tax = max(0.0, annual_taxable) * to_number(econ["profitTaxRate"]) / 100
        positive_base = sum(max(0.0, item["taxableProfitM"]) for item in year_months)
        for item in year_months:
            item["profitTaxM"] = (
                annual_tax * max(0.0, item["taxableProfitM"]) / positive_base if positive_base > 0 else 0.0
            )
            item["netProfitM"] = item["taxableProfitM"] - item["profitTaxM"]
            item["fcfM"] = item["ebitdaM"] - item["profitTaxM"] - item["capexM"]
            item["chddM"] = item["fcfM"] * item["discountFactor"]
            other_income_expense = item["otherIncludedEbitdaM"] + item["otherExcludedEbitdaM"]
            item["discountedInflowM"] = (
                item["revenueM"] + max(0.0, -other_income_expense)
            ) * item["discountFactor"]
            item["discountedOutflowM"] = (
                item["capexM"]
                + item["opexM"]
                + item["deductionsM"]
                + item["profitTaxM"]
                + max(0.0, other_income_expense)
            ) * item["discountFactor"]

    cumulative_chdd = 0.0
    for item in monthly:
        cumulative_chdd += item["chddM"]
        item["cumulativeChddM"] = cumulative_chdd

    annual: list[dict[str, Any]] = []
    cumulative_chdd = 0.0
    cumulative_inflow = 0.0
    cumulative_outflow = 0.0

    def sum_key(items: list[dict[str, Any]], key: str) -> float:
        return sum(to_number(item.get(key)) for item in items)

    def weighted_average(items: list[dict[str, Any]], value_key: str, count_key: str) -> float:
        denominator = sum_key(items, count_key)
        if denominator == 0:
            return 0.0
        return sum(item[value_key] * item[count_key] for item in items) / denominator

    activity_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in activity_history["transitions"]:
        activity_by_year[_year_of(event["date"])].append(event)
    paid_pump_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in pump_history["events"]:
        if event["pumpCapexM"] > 0:
            paid_pump_by_year[_year_of(event["date"])].append(event)

    for year in sorted(months_by_year):
        year_months = months_by_year[year]
        install_counts: dict[str, int] = defaultdict(int)
        for item in year_months:
            for nominal, count in item["pumpInstallCounts"].items():
                install_counts[nominal] += int(count)
        annual_item = {
            "year": year,
            "months": len(year_months),
            "oilPriceRubT": to_number(econ["oilPriceRubT"]),
            "discountFactor": year_months[0]["discountFactor"],
            "oilKt": sum_key(year_months, "oilT") / 1000,
            "liquidKt": sum_key(year_months, "liquidT") / 1000,
            "injectionKm3": sum_key(year_months, "injectionM3") / 1000,
            "averageActiveWells": sum_key(year_months, "activeWells") / len(year_months),
            "activeWellMonths": sum_key(year_months, "activeWells"),
            "averageTHP": weighted_average(year_months, "avgTHP", "thpCount"),
            "averageBHP": weighted_average(year_months, "avgBHP", "bhpCount"),
            "averageWLPR": weighted_average(year_months, "avgWLPR", "wlprCount"),
            "averageWOMR": weighted_average(year_months, "avgWOMR", "womrCount"),
            "averageWWIR": weighted_average(year_months, "avgWWIR", "wwirCount"),
            "mainRevenueM": sum_key(year_months, "mainRevenueM"),
            "revenueM": sum_key(year_months, "revenueM"),
            "pumpCapexM": sum_key(year_months, "pumpCapexM"),
            "capexM": sum_key(year_months, "capexM"),
            "oilOpexM": sum_key(year_months, "oilOpexM"),
            "liquidOpexM": sum_key(year_months, "liquidOpexM"),
            "injectionOpexM": sum_key(year_months, "injectionOpexM"),
            "fundOpexM": sum_key(year_months, "fundOpexM"),
            "pumpOperationM": sum_key(year_months, "pumpOperationM"),
            "gtmM": sum_key(year_months, "gtmM"),
            "startStopCount": sum_key(year_months, "startStopCount"),
            "startStopCostM": sum_key(year_months, "startStopCostM"),
            "conversionCount": sum_key(year_months, "conversionCount"),
            "conversionOpexM": sum_key(year_months, "conversionOpexM"),
            "propertyTaxM": sum_key(year_months, "propertyTaxM"),
            "mainOpexM": sum_key(year_months, "mainOpexM"),
            "opexM": sum_key(year_months, "opexM"),
            "baseDepreciationM": sum_key(year_months, "baseDepreciationM"),
            "depreciationM": sum_key(year_months, "depreciationM"),
            "residualStartM": to_number(econ.get("residualStartM", econ["existingAssetResidualM"])),
            "residualEndM": to_number(econ.get("residualEndM", econ["existingAssetResidualM"])),
            "deductionsM": sum_key(year_months, "deductionsM"),
            "otherIncludedEbitdaM": sum_key(year_months, "otherIncludedEbitdaM"),
            "otherExcludedEbitdaM": sum_key(year_months, "otherExcludedEbitdaM"),
            "taxableProfitM": sum_key(year_months, "taxableProfitM"),
            "profitTaxM": sum_key(year_months, "profitTaxM"),
            "netProfitM": sum_key(year_months, "netProfitM"),
            "ebitdaM": sum_key(year_months, "ebitdaM"),
            "fcfM": sum_key(year_months, "fcfM"),
            "chddM": sum_key(year_months, "chddM"),
            "discountedInflowM": sum_key(year_months, "discountedInflowM"),
            "discountedOutflowM": sum_key(year_months, "discountedOutflowM"),
            "pumpChanges": sum_key(year_months, "pumpChanges"),
            "pumpInstallCounts": dict(install_counts),
            "stopCount": sum(event["type"] == "stop" for event in activity_by_year[year]),
            "startCount": sum(event["type"] == "start" for event in activity_by_year[year]),
            "pumpUps": sum(event["type"] == "pump_up" for event in paid_pump_by_year[year]),
            "pumpDowns": sum(event["type"] == "pump_down" for event in paid_pump_by_year[year]),
            "initialPumps": sum(event["type"] == "initial_pump" for event in paid_pump_by_year[year]),
        }
        cumulative_chdd += annual_item["chddM"]
        cumulative_inflow += annual_item["discountedInflowM"]
        cumulative_outflow += annual_item["discountedOutflowM"]
        annual_item["cumulativeChddM"] = cumulative_chdd
        annual_item["profitabilityIndex"] = (
            cumulative_inflow / cumulative_outflow if cumulative_outflow != 0 else 0.0
        )
        annual.append(annual_item)

    source_cumulative = _source_cumulative_by_year(economic_rows, [item["year"] for item in annual])
    for item in annual:
        item.update(source_cumulative[item["year"]])

    diagnostics["technicalResetDates"] = []
    diagnostics["technicalResetRows"] = 0
    diagnostics["warnings"] = _build_warnings(diagnostics, econ, max_date)

    total_inflow = sum_key(monthly, "discountedInflowM")
    total_outflow = sum_key(monthly, "discountedOutflowM")
    summary = {
        "totalOilKt": sum_key(monthly, "oilT") / 1000,
        "totalLiquidKt": sum_key(monthly, "liquidT") / 1000,
        "totalInjectionK": sum_key(monthly, "injectionM3") / 1000,
        "totalRevenueM": sum_key(monthly, "revenueM"),
        "totalOpexM": sum_key(monthly, "opexM"),
        "totalCapexM": sum_key(monthly, "capexM"),
        "totalFcfM": sum_key(monthly, "fcfM"),
        "totalChddM": sum_key(monthly, "chddM"),
        "profitabilityIndex": total_inflow / total_outflow if total_outflow != 0 else 0.0,
        "pumpChanges": sum_key(monthly, "pumpChanges"),
        "startStopCount": len(activity_history["transitions"]),
        "conversionCount": len(conversion_history["transitions"]),
        "totalConversionOpexM": sum_key(monthly, "conversionOpexM"),
        "years": len(annual),
    }

    clean_monthly = []
    for item in monthly:
        clean_monthly.append({key: value for key, value in item.items() if not key.endswith("Count") or key in {
            "startStopCount", "conversionCount"
        }})

    return {
        "version": VERSION,
        "name": name,
        "sourceFile": source_file,
        "minDate": min_date,
        "maxDate": max_date,
        "startDate": calculation_start_date,
        "assumptions": econ,
        "pumps": pump_table,
        "diagnostics": diagnostics,
        "summary": summary,
        "annual": annual,
        "fieldMonthly": clean_monthly,
        "events": sorted(decision_events, key=lambda event: (event["date"], natural_key(event["well"]))),
        "activityTransitions": activity_history["transitions"],
        "conversionTransitions": conversion_history["transitions"],
    }
