"""Case intake for track 2 (Model Z): read the organizer archive, derive the incumbent
request for the management period, print the A100 command sequence.

The archive layout is not fixed by the organizers, so nothing here is assumed silently:
every layout expectation is a named check whose failure names the member, the well or the
schedule line that broke it. One ``inspect`` call is meant to say within a minute what a
new case differs in.

Subcommands
-----------
``inspect``        describe the archive: deck, schedule, wells, dates, cut state, regions.
``build-request``  emit the incumbent ``CycleRequest`` JSON plus ``manifest.json``.
``plan``           print the A100 command sequence for the case.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Callable, Mapping, Sequence
import zipfile

from timesoil.aios.case_profile import CaseProfile, load_case_profile
from timesoil.aios.contracts import ControlTarget, WellRole, WellStatus
from timesoil.aios.schedule_overlay import (
    ScheduleOverlayError,
    _code,
    _control_templates,
    _ControlTemplate,
    _date_blocks,
    _record_tokens,
    _terminal_line,
    _word,
)
from timesoil.aios.workflow import (
    CycleError,
    CycleRequest,
    _source_control_inventory,
    _validate_source_well_scope,
)

_SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE")
_UNIT_SYSTEMS = ("METRIC", "FIELD", "LAB", "PVT-M")
_KEYWORD = re.compile(r"^[A-Z][A-Z0-9_-]{0,7}$")
_REGION_KEYWORD = re.compile(r"^(FIPNUM|FIP_[A-Z0-9_]{1,4})$")
_REPEAT = re.compile(r"^(\d+)\*(.*)$")
_MONTH_NAMES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_OPEN_STATUS = {"OPEN"}
_SHUT_STATUS = {"SHUT", "STOP"}


class IntakeError(ValueError):
    """The case archive does not match a checked intake assumption."""


# --------------------------------------------------------------------------- archive


class CaseArchive:
    """A ZIP or directory case, read member by member with every path checked."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if self.path.is_dir():
            files = sorted(item for item in self.path.rglob("*") if item.is_file())
            self.members = tuple(item.relative_to(self.path).as_posix() for item in files)
            self.kind = "directory"
            self._read: Callable[[str], bytes] = lambda name: (self.path / name).read_bytes()
            self.sha256 = _tree_digest(self.members, self._read)
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            with zipfile.ZipFile(self.path) as archive:
                self.members = tuple(sorted(
                    item.filename for item in archive.infolist() if not item.is_dir()))
            self.kind = "zip"
            self.sha256 = sha256(self.path.read_bytes()).hexdigest()
            self._read = self._read_zip
        else:
            raise IntakeError(f"case source must be a directory or a ZIP archive: {self.path}")
        if not self.members:
            raise IntakeError(f"case source holds no files: {self.path}")
        for name in self.members:
            posix = PurePosixPath(name)
            if posix.is_absolute() or ".." in posix.parts or "\\" in name:
                raise IntakeError(f"unsafe archive member path: {name!r}")

    def _read_zip(self, name: str) -> bytes:
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(name)

    def raw(self, name: str) -> bytes:
        if name not in self.members:
            raise IntakeError(f"archive member is missing: {name!r}")
        return self._read(name)

    def text(self, name: str) -> str:
        try:
            return self.raw(name).decode("utf-8")
        except UnicodeError as error:
            raise IntakeError(
                f"archive member {name!r} is not UTF-8; the cycle requires UTF-8 decks"
            ) from error

    def digest(self, name: str) -> str:
        return sha256(self.raw(name)).hexdigest()


def _tree_digest(members: Sequence[str], read: Callable[[str], bytes]) -> str:
    digest = sha256()
    for name in members:
        digest.update(name.encode())
        digest.update(sha256(read(name)).digest())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CaseLayout:
    """Everything the cycle request needs to address the case, each field checked."""

    deck_member: str
    schedule_member: str
    unit_system: str
    includes: tuple[tuple[str, str], ...]  # (section, member)


def _deck_lines(text: str) -> list[str]:
    return [_code(line) for line in text.splitlines()]


def read_layout(archive: CaseArchive) -> CaseLayout:
    decks = [name for name in archive.members if name.upper().endswith(".DATA")]
    if len(decks) != 1:
        raise IntakeError(
            "expected exactly one OPM .DATA deck in the case, found: "
            + (", ".join(decks) or "none")
        )
    deck_member = decks[0]
    deck_dir = PurePosixPath(deck_member).parent
    section = "RUNSPEC"
    unit_system: str | None = None
    includes: list[tuple[str, str]] = []
    pending = False
    for code in _deck_lines(archive.text(deck_member)):
        if not code:
            continue
        upper = code.upper()
        if pending:
            tokens = code.rstrip("/").strip().split()
            if not tokens:
                raise IntakeError(f"INCLUDE without a file name in deck {deck_member!r}")
            resolved = (deck_dir / _word(tokens[0])).as_posix()
            if resolved not in archive.members:
                raise IntakeError(
                    f"deck {deck_member!r} includes a missing member: {resolved!r}")
            includes.append((section, resolved))
            pending = False
            continue
        if upper in _SECTIONS:
            section = upper
        elif upper in _UNIT_SYSTEMS:
            if unit_system is not None and unit_system != upper:
                raise IntakeError("deck declares more than one unit system")
            unit_system = upper
        elif upper == "INCLUDE":
            pending = True
    if pending:
        raise IntakeError(f"deck {deck_member!r} ends with an unterminated INCLUDE")
    if unit_system is None:
        raise IntakeError("deck declares no unit system; METRIC is required")
    if unit_system != "METRIC":
        raise IntakeError(f"deck unit system is {unit_system!r}; the case pipeline assumes METRIC")
    schedule = [member for section_name, member in includes if section_name == "SCHEDULE"]
    if len(schedule) != 1:
        raise IntakeError(
            "expected exactly one SCHEDULE include, found: " + (", ".join(schedule) or "none")
        )
    return CaseLayout(deck_member, schedule[0], unit_system, tuple(includes))


# -------------------------------------------------------------------------- schedule


@dataclass(frozen=True, slots=True)
class Regime:
    """One well's simulator-facing control as the source schedule states it."""

    role: WellRole
    status: WellStatus
    target: ControlTarget
    value: float
    bhp_limit: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "status": self.status.value,
                "target": self.target.value, "value": self.value, "bhp_limit": self.bhp_limit}


def _field(template: _ControlTemplate, index: int, label: str) -> float | None:
    if index >= len(template.fields):
        return None
    raw = template.fields[index]
    if raw == "1*":
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise IntakeError(
            f"well {template.well!r} has a non-numeric {label} on schedule line "
            f"{template.line + 1}: {raw!r}"
        ) from error
    return value


def template_regime(lines: Sequence[str], template: _ControlTemplate) -> Regime:
    """Decode one WCONPROD/WCONINJE record; an unsupported mode is an error, not a guess."""
    code = _code(lines[template.line])
    tokens = _record_tokens(code[:-1], template.line + 1)
    if template.role is WellRole.PRODUCER:
        status, mode = _word(tokens[1]).upper(), _word(tokens[2]).upper()
        if mode != "LRAT":
            raise IntakeError(
                f"producer {template.well!r} runs control mode {mode!r} on schedule line "
                f"{template.line + 1}; the intake continues LRAT producers only"
            )
        value, bhp = _field(template, 3, "LRAT"), _field(template, 5, "BHP")
        target = ControlTarget.LIQUID_RATE
    else:
        if (template.fluid or "") != "WATER":
            raise IntakeError(
                f"injector {template.well!r} injects {template.fluid!r} on schedule line "
                f"{template.line + 1}; only WATER injection is supported"
            )
        status, mode = _word(tokens[2]).upper(), _word(tokens[3]).upper()
        if mode != "RATE":
            raise IntakeError(
                f"injector {template.well!r} runs control mode {mode!r} on schedule line "
                f"{template.line + 1}; the intake continues RATE injectors only"
            )
        value, bhp = _field(template, 0, "RATE"), _field(template, 2, "BHP")
        target = ControlTarget.WATER_INJECTION_RATE
    if status in _OPEN_STATUS:
        if value is None:
            raise IntakeError(
                f"open well {template.well!r} has no rate target on schedule line "
                f"{template.line + 1}"
            )
        return Regime(template.role, WellStatus.OPEN, target, float(value), bhp)
    if status not in _SHUT_STATUS:
        raise IntakeError(
            f"well {template.well!r} has unsupported status {status!r} on schedule line "
            f"{template.line + 1}"
        )
    return Regime(template.role, WellStatus.SHUT, target, 0.0, bhp)


class Schedule:
    """Report dates and effective WCON records of one case schedule include."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = text.splitlines()
        try:
            self.blocks = _date_blocks(text)
            self.templates = _control_templates(text)
        except ScheduleOverlayError as error:
            raise IntakeError(f"case schedule cannot be parsed: {error}") from error
        if not self.templates:
            raise IntakeError("case schedule declares no WCONPROD/WCONINJE control")
        self._index = {block.month: position for position, block in enumerate(self.blocks)}
        self._terminal = _terminal_line(text.splitlines(keepends=True), self.blocks[-1])
        self.wells = tuple(sorted({template.well for template in self.templates}))
        self.first_template: dict[str, _ControlTemplate] = {}
        for template in self.templates:
            self.first_template.setdefault(template.well, template)

    @property
    def months(self) -> tuple[date, ...]:
        return tuple(block.month for block in self.blocks)

    def boundary(self, month: date) -> int:
        """The first source line that belongs to the month after ``month``."""
        try:
            position = self._index[month]
        except KeyError:
            raise IntakeError(
                f"report date {month.isoformat()} is absent from the case schedule"
            ) from None
        if position + 1 < len(self.blocks):
            return self.blocks[position + 1].keyword_line
        return self._terminal

    def snapshots(
        self, months: Sequence[date]
    ) -> dict[date, tuple[dict[str, _ControlTemplate], dict[tuple[str, WellRole], _ControlTemplate]]]:
        """Effective template per well and per (well, role) at each month, in one pass."""
        ordered = list(months)
        if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
            raise IntakeError("snapshot months must be strictly increasing")
        latest: dict[str, _ControlTemplate] = {}
        by_role: dict[tuple[str, WellRole], _ControlTemplate] = {}
        position = 0
        result = {}
        for month in ordered:
            limit = self.boundary(month)
            while position < len(self.templates) and self.templates[position].line < limit:
                template = self.templates[position]
                latest[template.well] = template
                by_role[(template.well, template.role)] = template
                position += 1
            result[month] = (dict(latest), dict(by_role))
        return result

    def keywords_after(self, month: date) -> tuple[str, ...]:
        found = {
            _code(line).upper()
            for line in self.lines[self.boundary(month):]
            if _KEYWORD.fullmatch(_code(line).upper())
        }
        return tuple(sorted(found))

    def welspecs_wells(self) -> tuple[str, ...]:
        wells: list[str] = []
        inside = False
        for line in self.lines:
            code = _code(line)
            if not code:
                continue
            if not inside:
                inside = code.upper() == "WELSPECS"
                continue
            if code == "/":
                inside = False
                continue
            tokens = _record_tokens(code.rstrip("/").strip(), 0)
            if tokens:
                wells.append(_word(tokens[0]))
        return tuple(sorted(set(wells)))


# --------------------------------------------------------------------------- regions


def region_counts(archive: CaseArchive, layout: CaseLayout) -> dict[str, int]:
    """Distinct region values per FIPNUM/FIP_* array; only REGIONS members are scanned."""
    counts: dict[str, int] = {}
    for section, member in layout.includes:
        if section != "REGIONS":
            continue
        keyword: str | None = None
        values: set[str] = set()
        for line in archive.text(member).splitlines():
            code = _code(line)
            if not code:
                continue
            if keyword is None:
                if _REGION_KEYWORD.fullmatch(code.upper()):
                    keyword, values = code.upper(), set()
                continue
            for token in code.split():
                if token == "/":
                    counts[keyword], keyword = len(values), None
                    break
                repeat = _REPEAT.fullmatch(token)
                values.add(repeat.group(2) if repeat else token.rstrip("/"))
        if keyword is not None:
            counts[keyword] = len(values)
    return counts


# -------------------------------------------------------------------------- incumbent


def month_range(start: date, end: date) -> tuple[date, ...]:
    """Monthly control steps in ``[start, end)``; ``end`` is the terminal report date."""
    if start.day != 1 or end.day != 1:
        raise IntakeError("management boundaries must be the first day of a month")
    if end <= start:
        raise IntakeError("management end must follow management start")
    months: list[date] = []
    year, month = start.year, start.month
    while date(year, month, 1) < end:
        months.append(date(year, month, 1))
        year, month = (year + month // 12, month % 12 + 1)
    return tuple(months)


def _repair_months(profile: CaseProfile, wells: Sequence[str]) -> dict[str, list[tuple[date, date]]]:
    known = set(wells)
    calendar: dict[str, list[tuple[date, date]]] = {}
    for well, first, last in profile.repairs:
        if well not in known:
            raise IntakeError(
                f"the case profile schedules a repair on unknown well {well!r}; "
                f"the schedule controls {len(known)} wells"
            )
        calendar.setdefault(well, []).append((first, last))
    return calendar


def incumbent_controls(
    schedule: Schedule,
    profile: CaseProfile,
    *,
    cut_month: date,
    months: Sequence[date],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Continue the cut regime over the management period under the case caps.

    Role per month follows the source inventory, because ``_validate_source_well_scope``
    compares each action against the source role of that very month. Where the source
    itself moves a well to a role that was never observed at the cut, that role's first
    source regime is used; where a well has no source control yet, it stays SHUT.
    """
    if cut_month >= months[0]:
        raise IntakeError("the cut month must precede the management period")
    snapshots = schedule.snapshots((cut_month, *months))
    cut_latest, cut_by_role = snapshots[cut_month]
    calendar = _repair_months(profile, schedule.wells)
    low, high = profile.bhp_bounds
    rows: list[dict[str, Any]] = []
    repaired: list[dict[str, str]] = []
    late: dict[str, str] = {}
    followed: dict[str, str] = {}
    for month in months:
        latest, by_role = snapshots[month]
        for well in schedule.wells:
            template = latest.get(well)
            if template is None:
                role = schedule.first_template[well].role
                late.setdefault(well, month.isoformat())
                rows.append({"month": month, "well": well, "role": role,
                             "status": WellStatus.SHUT, "value": 0.0, "bhp_limit": None,
                             "target": (ControlTarget.LIQUID_RATE if role is WellRole.PRODUCER
                                        else ControlTarget.WATER_INJECTION_RATE)})
                continue
            role = template.role
            source = cut_by_role.get((well, role))
            if source is None:
                source = by_role[(well, role)]
                if well in cut_latest:
                    followed.setdefault(well, f"{role.value} from {month.isoformat()}")
            regime = template_regime(schedule.lines, source)
            bhp = (max(regime.bhp_limit or low, low) if role is WellRole.PRODUCER
                   else min(regime.bhp_limit or high, high))
            shut = regime.status is WellStatus.SHUT or any(
                first <= month < last for first, last in calendar.get(well, ()))
            if shut and regime.status is WellStatus.OPEN:
                repaired.append({"well": well, "month": month.isoformat()})
            rows.append({"month": month, "well": well, "role": role,
                         "status": WellStatus.SHUT if shut else WellStatus.OPEN,
                         "target": regime.target, "value": 0.0 if shut else regime.value,
                         "bhp_limit": bhp})

    unscaled = _monthly_totals(rows)
    factors = {
        "producer": _factor(max(row["liquid_m3d"] for row in unscaled), profile.liquid_cap_m3d),
        "injector": _factor(max(row["injection_m3d"] for row in unscaled), profile.injection_cap_m3d),
    }
    for row in rows:
        if row["status"] is WellStatus.OPEN:
            factor = factors["injector" if row["role"] is WellRole.INJECTOR else "producer"]
            row["value"] = min(row["value"] * factor, 500.0) if (
                row["target"] is ControlTarget.LIQUID_RATE) else row["value"] * factor
    scaled = _monthly_totals(rows)
    worst_liquid = max(row["liquid_m3d"] for row in scaled)
    worst_injection = max(row["injection_m3d"] for row in scaled)
    tolerance = 1e-9
    if (worst_liquid > profile.liquid_cap_m3d + tolerance
            or worst_injection > profile.injection_cap_m3d + tolerance):
        raise IntakeError(
            f"scaled field targets still exceed the case caps: liquid {worst_liquid:.6f} of "
            f"{profile.liquid_cap_m3d}, injection {worst_injection:.6f} of {profile.injection_cap_m3d}"
        )
    controls = [
        {"month": row["month"].isoformat(), "well": row["well"], "role": row["role"].value,
         "status": row["status"].value, "target": row["target"].value, "value": float(row["value"]),
         **({"bhp_limit": float(row["bhp_limit"])} if row["bhp_limit"] is not None else {})}
        for row in sorted(rows, key=lambda item: (item["month"], item["well"]))
    ]
    report = {
        "months": len(months), "first_month": months[0].isoformat(),
        "last_month": months[-1].isoformat(), "wells": len(schedule.wells),
        "scale_factors": factors,
        "field_targets_before_scaling": {"max_liquid_m3d": max(r["liquid_m3d"] for r in unscaled),
                                         "max_injection_m3d": max(r["injection_m3d"] for r in unscaled)},
        "field_targets_after_scaling": {"max_liquid_m3d": worst_liquid,
                                        "max_injection_m3d": worst_injection},
        "repair_shut_well_months": len(repaired),
        "repair_shut_wells": sorted({item["well"] for item in repaired}),
        "wells_shut_before_first_source_control": late,
        "wells_following_a_source_role_change_after_the_cut": followed,
        "bhp_bounds_bar": {"producer_min": low, "injector_max": high},
    }
    return controls, report


def _factor(observed: float, cap: float) -> float:
    if observed <= cap or observed <= 0:
        return 1.0
    return cap / observed


def _monthly_totals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    liquid: Counter[date] = Counter()
    injection: Counter[date] = Counter()
    for row in rows:
        if row["status"] is not WellStatus.OPEN:
            continue
        bucket = injection if row["role"] is WellRole.INJECTOR else liquid
        bucket[row["month"]] += float(row["value"])
    months = sorted(set(liquid) | set(injection) | {row["month"] for row in rows})
    return [{"month": month, "liquid_m3d": liquid.get(month, 0.0),
             "injection_m3d": injection.get(month, 0.0)} for month in months]


# ------------------------------------------------------------------ schedule extension


def _date_record(month: date) -> str:
    return f"DATES\n {month.day:02d} {_MONTH_NAMES[month.month - 1]} {month.year} /\n/\n"


def extended_schedule(schedule: Schedule, months: Sequence[date], terminal: date) -> str:
    """Append the missing monthly report dates; interior gaps are refused, never filled."""
    have = set(schedule.months)
    wanted = [*months, terminal]
    missing = [month for month in wanted if month not in have]
    if not missing:
        raise IntakeError("the case schedule already covers the management period")
    last = schedule.months[-1]
    if missing[0] <= last:
        raise IntakeError(
            f"the case schedule is missing report date {missing[0].isoformat()} before its own "
            f"last date {last.isoformat()}; an interior gap cannot be repaired automatically"
        )
    body = "".join(_date_record(month) + "\n" for month in missing)
    return (schedule.text.rstrip("\n") + "\n\n"
            "-- timesoil case intake: monthly report dates appended to cover the management\n"
            "-- period; no control record was added, changed or removed.\n\n" + body)


def write_extended_archive(archive: CaseArchive, member: str, text: str, destination: Path) -> str:
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as output:
        for name in archive.members:
            output.writestr(name, text.encode() if name == member else archive.raw(name))
    return sha256(destination.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- commands


def inspect_case(archive: CaseArchive, cut: date) -> dict[str, Any]:
    layout = read_layout(archive)
    schedule = Schedule(archive.text(layout.schedule_member))
    cut_month = date(cut.year, cut.month, 1)
    welspecs = schedule.welspecs_wells()
    report: dict[str, Any] = {
        "archive": {"path": str(archive.path), "kind": archive.kind, "sha256": archive.sha256,
                    "members": len(archive.members)},
        "deck": {"member": layout.deck_member, "sha256": archive.digest(layout.deck_member),
                 "unit_system": layout.unit_system,
                 "includes": [{"section": section, "member": member}
                              for section, member in layout.includes]},
        "schedule": {"member": layout.schedule_member,
                     "sha256": archive.digest(layout.schedule_member),
                     "dates": len(schedule.blocks),
                     "first_date": schedule.months[0].isoformat(),
                     "last_date": schedule.months[-1].isoformat(),
                     "monthly_grid": _is_monthly(schedule.months),
                     "control_records": len(schedule.templates)},
        "wells": {"welspecs": len(welspecs), "controlled": len(schedule.wells),
                  "welspecs_without_control": sorted(set(welspecs) - set(schedule.wells)),
                  "controlled_without_welspecs": sorted(set(schedule.wells) - set(welspecs)),
                  "names": list(schedule.wells)},
        "cut": {"date": cut.isoformat(), "month": cut_month.isoformat(),
                "last_date_not_after_cut": schedule.months[-1] <= cut_month},
        "regions": region_counts(archive, layout),
    }
    if cut_month not in set(schedule.months):
        report["cut"]["present_in_schedule"] = False
        report["cut"]["error"] = "the cut month has no report date in the case schedule"
        return report
    report["cut"]["present_in_schedule"] = True
    latest = schedule.snapshots((cut_month,))[cut_month][0]
    regimes: dict[str, Any] = {}
    problems: list[str] = []
    for well, template in sorted(latest.items()):
        try:
            regimes[well] = template_regime(schedule.lines, template).to_dict()
        except IntakeError as error:
            problems.append(str(error))
    roles = Counter(item["role"] for item in regimes.values())
    active = Counter(item["role"] for item in regimes.values() if item["status"] == "OPEN")
    report["cut"].update(
        wells_with_control=len(latest),
        wells_without_control=sorted(set(schedule.wells) - set(latest)),
        roles={role: roles.get(role, 0) for role in ("producer", "injector")},
        active_roles={role: active.get(role, 0) for role in ("producer", "injector")},
        field_liquid_target_m3d=sum(item["value"] for item in regimes.values()
                                    if item["role"] == "producer" and item["status"] == "OPEN"),
        field_injection_target_m3d=sum(item["value"] for item in regimes.values()
                                       if item["role"] == "injector" and item["status"] == "OPEN"),
        regimes=regimes,
        unreadable_regimes=problems,
    )
    report["keywords_after_cut"] = list(schedule.keywords_after(cut_month))
    return report


def _is_monthly(months: Sequence[date]) -> bool:
    return all(
        month.day == 1 and (month.year * 12 + month.month) == (prior.year * 12 + prior.month + 1)
        for prior, month in zip(months, months[1:], strict=False)
    )


def build_request(
    archive: CaseArchive,
    profile: CaseProfile,
    *,
    cut: date,
    start: date,
    end: date,
    scenario_id: str,
    parsing_strictness: str,
    output: Path,
    extend_schedule: bool,
) -> dict[str, Any]:
    """Emit the incumbent request; every assumption below is a recorded, failing check."""
    checks: list[dict[str, Any]] = []

    def note(name: str, detail: Any) -> None:
        checks.append({"check": name, "detail": detail})

    layout = read_layout(archive)
    note("single_data_deck_and_schedule_include",
         {"deck": layout.deck_member, "schedule": layout.schedule_member})
    note("unit_system_is_metric", layout.unit_system)
    schedule = Schedule(archive.text(layout.schedule_member))
    cut_month = date(cut.year, cut.month, 1)
    months = month_range(start, end)
    note("management_period", {"first": months[0].isoformat(), "last": months[-1].isoformat(),
                               "terminal_report_date": end.isoformat(), "months": len(months)})
    if cut_month not in set(schedule.months):
        raise IntakeError(
            f"the cut month {cut_month.isoformat()} has no report date in "
            f"{layout.schedule_member!r}; the schedule runs "
            f"{schedule.months[0].isoformat()}..{schedule.months[-1].isoformat()}"
        )
    note("cut_month_present", cut_month.isoformat())

    output.mkdir(parents=True, exist_ok=False)
    source_member = layout.schedule_member
    source_path = archive.path
    source_sha = archive.sha256
    missing = [month for month in (*months, end) if month not in set(schedule.months)]
    if missing:
        if not extend_schedule:
            raise IntakeError(
                f"{len(missing)} management report dates are absent from "
                f"{layout.schedule_member!r} (first {missing[0].isoformat()}, last "
                f"{missing[-1].isoformat()}); the overlay needs one DATES block per control "
                "month. Re-run with --extend-schedule to emit an archive with the missing "
                "report dates appended and nothing else changed."
            )
        text = extended_schedule(schedule, months, end)
        source_path = output / "case-extended.zip"
        source_sha = write_extended_archive(archive, source_member, text, source_path)
        schedule = Schedule(text)
        note("schedule_extended", {"appended_report_dates": len(missing),
                                   "first": missing[0].isoformat(), "last": missing[-1].isoformat(),
                                   "archive": str(source_path), "sha256": source_sha})
    note("schedule_covers_management_period",
         {"dates": len(schedule.blocks), "last": schedule.months[-1].isoformat()})

    controls, report = incumbent_controls(schedule, profile, cut_month=cut_month, months=months)
    note("repair_calendar_applied", {"repairs": len(profile.repairs),
                                     "shut_well_months": report["repair_shut_well_months"],
                                     "wells": report["repair_shut_wells"]})
    note("field_targets_within_case_caps",
         {"caps": {"liquid_m3d": profile.liquid_cap_m3d, "injection_m3d": profile.injection_cap_m3d},
          "before": report["field_targets_before_scaling"],
          "after": report["field_targets_after_scaling"],
          "factors": report["scale_factors"]})
    note("no_production_before_first_source_control",
         report["wells_shut_before_first_source_control"])

    # The profile's rules travel inside the request so that every candidate inherits them
    # and the final full-cycle checks the OPM SUMMARY against the same limits (gate G3).
    well_ids = sorted({str(action["well"]) for action in controls}, key=lambda w: (len(w), w))
    months_sorted = sorted({str(action["month"]) for action in controls})
    operating_constraints = [rule.to_dict() for rule in profile.operating_rules(
        wells=well_ids, start=date.fromisoformat(months_sorted[0]),
        end=date.fromisoformat(months_sorted[-1]))]
    request = {
        "context": {
            "track": 2,
            "objective": ("Baseline incumbent for the official track 2 case: the source regime at "
                          "the cut continued over the management period under the case caps. "
                          "Paired baseline for every candidate; no gain is claimed here."),
            "constraints": {"allow_conversion_to_injection": True},
            "operating_constraints": operating_constraints,
            "facts": {
                "is_baseline": True,
                "schedule_kind": "case_incumbent_continuation",
                "case_profile_sha256": profile.sha256,
                "case_archive_sha256": source_sha,
                "original_case_archive_sha256": archive.sha256,
                "cut_month": cut_month.isoformat(),
                "producer_scale": report["scale_factors"]["producer"],
                "injector_scale": report["scale_factors"]["injector"],
                "surrogate_used_for_candidate_selection": False,
                "optimization_improvement_claimed": False,
            },
        },
        "controls": controls,
        "source": str(source_path),
        "deck": layout.deck_member,
        "schedule_relative_path": source_member,
        "scenario_id": scenario_id,
        "source_model": "model_z_opm",
        "start_year": start.year,
        "parsing_strictness": parsing_strictness,
        "charge_initial_pump": False,
        "horizon_months": len(months),
    }
    checked = CycleRequest.from_mapping(request)
    inventory = _source_control_inventory(schedule.text, sorted(months))
    _validate_source_well_scope(checked.controls, inventory, allow_conversion_to_injection=True)
    note("request_validates", {"controls": len(controls),
                               "controls_sha256": checked.controls_sha256,
                               "request_sha256": checked.request_sha256})

    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n")
    inspection = inspect_case(archive, cut)
    (output / "inspect.json").write_text(json.dumps(inspection, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "generated_by": "scripts/intake_case_z.py build-request",
        "archive": inspection["archive"],
        "deck_sha256": inspection["deck"]["sha256"],
        "schedule_sha256": inspection["schedule"]["sha256"],
        "case_profile": {"sha256": profile.sha256, "liquid_cap_m3d": profile.liquid_cap_m3d,
                         "injection_cap_m3d": profile.injection_cap_m3d,
                         "bhp_bounds": list(profile.bhp_bounds), "repairs": len(profile.repairs)},
        "request": {"path": str(request_path), "sha256": sha256(request_path.read_bytes()).hexdigest(),
                    "controls_sha256": checked.controls_sha256,
                    "request_sha256": checked.request_sha256,
                    "source": str(source_path), "source_sha256": source_sha},
        "incumbent": report,
        "checks": checks,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


_PLAN = """\
# Track 2 case run on A100 ({host}). Every command from /root/projects/TimesOil.
# PYTHONPATH=src:scripts; project venv {venv}; GPU {gpu}; results root R={results}.

# 0. Deliver the code and pin the case archive (no scp of source).
git -C /root/projects/TimesOil pull --ff-only
cp {archive} {pinned} && sha256sum {pinned}

# 1. Intake: read the archive, then derive the incumbent request (no OPM yet, < 1 min).
{venv} scripts/intake_case_z.py inspect {pinned} --cut {cut} \\
    --output {intake}/inspect.json
{venv} scripts/intake_case_z.py build-request {pinned} \\
    --cut {cut} --start {start} --end {end} --profile {profile} --output {intake}

# 2. Baseline full cycle of the incumbent over the whole management period (~4 min, 16 MPI).
OPM_MPI_PROCESSES=16 OPM_THREADS_PER_PROCESS=1 OMP_NUM_THREADS=1 OPM_CPU_AFFINITY=14-29 \\
PYTHONPATH=src:scripts {venv} -m timesoil.aios.cli full-cycle {intake}/request.json \\
    --runs-dir $R/case-z/cycles --run-id baseline --timeout 7200

# 3. Canonical export check: the paired baseline every later claim is measured against.
test -s $R/case-z/cycles/baseline/canonical/chdd.csv
test -s $R/case-z/cycles/baseline/canonical/trajectory.csv
sha256sum $R/case-z/cycles/baseline/canonical/manifest.json | tee {intake}/baseline-canonical.sha256
{venv} -c "import json,pathlib;m=json.loads(pathlib.Path('$R/case-z/cycles/baseline/canonical/manifest.json').read_text());print(json.dumps(m,indent=2)[:2000])"

# 4. Blocks and connectivity for the search (a few minutes, no OPM).
{venv} scripts/export_opm_connectivity.py {pinned} $R/case-z/connectivity.json
{venv} scripts/export_blocks.py {pinned} $R/case-z/connectivity.json \\
    $R/case-z/blocks.json

# 5. Feasible candidate bank (~40 runs on 2 workers, ~80 min).
{venv} scripts/build_feasible_bank.py --request {intake}/request.json \\
    --canonical $R/case-z/cycles/baseline/canonical/chdd.csv \\
    --export-manifest $R/case-z/cycles/baseline/canonical/manifest.json \\
    --blocks $R/case-z/blocks.json --output $R/case-z/bank

# 6. TimesFM head fine-tune on the case history (45-90 min, GPU {gpu}) and evaluation (5 min).
CUDA_VISIBLE_DEVICES={gpu} {gpu_venv} scripts/finetune_timesfm_head.py ...  # see docs/RUNBOOK.md
CUDA_VISIBLE_DEVICES={gpu} {gpu_venv} scripts/evaluate_timesfm_scenarios.py ...

# 7. Search: zero OPM calls, sealed before the final run (~15-30 min).
CUDA_VISIBLE_DEVICES={gpu} {gpu_venv} scripts/propose_track2_policies.py \\
    $R/case-z/cycles/baseline {intake}/request.json $R/case-z/search --rounds 3 \\
    --economic-selection --case-profile {profile} \\
    --connectivity $R/case-z/connectivity.json --head <full-model.pt> --head-sha256 <sha>
SEAL=$(sha256sum $R/case-z/search/selection-before-opm.json | cut -d' ' -f1); echo $SEAL

# 8. Exactly one final OPM verification; no re-selection after this (~8 min).
{venv} scripts/track2_final_selection.py $R/case-z/search \\
    $R/case-z/cycles/baseline $R/case-z/final --seal-sha256 $SEAL
"""


def plan_text(args: argparse.Namespace) -> str:
    return _PLAN.format(
        host="a100-remote", venv="/root/projects/TimesOil/.venv/bin/python",
        gpu_venv="/tmp/timesoil-kt3-20260908/venv/bin/python", gpu="5",
        results=args.results_root, archive=args.archive, pinned=args.pinned,
        cut=args.cut.isoformat(),
        start=args.start.isoformat(), end=args.end.isoformat(), profile=args.profile,
        intake=args.intake,
    )


# ------------------------------------------------------------------------------- cli


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an ISO date: {value!r}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="describe the case archive as JSON")
    inspect.add_argument("archive", type=Path)
    inspect.add_argument("--cut", type=_date, default=date(2006, 12, 31))
    inspect.add_argument("--output", type=Path, default=None, help="write JSON here as well")

    build = commands.add_parser("build-request", help="emit the incumbent CycleRequest JSON")
    build.add_argument("archive", type=Path)
    build.add_argument("--cut", type=_date, default=date(2006, 12, 31))
    build.add_argument("--start", type=_date, default=date(2007, 1, 1))
    build.add_argument("--end", type=_date, default=date(2025, 9, 1),
                       help="terminal report date; control months are [start, end)")
    build.add_argument("--profile", type=Path, required=True, help="case_constraints.json")
    build.add_argument("--output", type=Path, required=True, help="new directory; never overwritten")
    build.add_argument("--scenario-id", default="case-z-incumbent")
    build.add_argument("--parsing-strictness", default="low", choices=("strict", "low"))
    build.add_argument("--extend-schedule", action="store_true",
                       help="append the missing monthly report dates to a copy of the archive")

    plan = commands.add_parser("plan", help="print the A100 command sequence for the case")
    plan.add_argument("--archive", default="~/case_z.zip", help="the delivered archive")
    plan.add_argument("--pinned", default="/tmp/timesoil-kt2/case_z.zip", help="immutable copy on A100")
    plan.add_argument("--profile", default="config/case_constraints.json")
    plan.add_argument("--intake", default="/root/projects/TimesOil/results/case-z-intake")
    plan.add_argument("--results-root", default="/root/projects/TimesOil/results/case-20260911")
    plan.add_argument("--cut", type=_date, default=date(2006, 12, 31))
    plan.add_argument("--start", type=_date, default=date(2007, 1, 1))
    plan.add_argument("--end", type=_date, default=date(2025, 9, 1))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        print(plan_text(args))
        return 0
    try:
        archive = CaseArchive(args.archive)
        if args.command == "inspect":
            report = inspect_case(archive, args.cut)
        else:
            report = build_request(
                archive, load_case_profile(args.profile), cut=args.cut, start=args.start,
                end=args.end, scenario_id=args.scenario_id,
                parsing_strictness=args.parsing_strictness, output=args.output,
                extend_schedule=args.extend_schedule)
    except (IntakeError, CycleError, ScheduleOverlayError) as error:
        print(f"case intake failed: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.command == "inspect" and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
