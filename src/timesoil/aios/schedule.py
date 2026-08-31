"""Deterministic compiler for the supported Model Y schedule subset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import re
from typing import Iterable

from .contracts import (
    Case,
    ControlAction,
    ControlTarget,
    WellRole,
    WellStatus,
)


_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_DATE_LINE = re.compile(r"\s*(\d{1,2})\s+([A-Z]{3})\s+(\d{4})\s*/\s*")
_PROD_LINE = re.compile(
    r"\s*'([^']+)'\s+'(OPEN|SHUT)'\s+'(ORAT|LRAT)'\s+(.+?)\s*/\s*"
)
_INJ_LINE = re.compile(
    r"\s*'([^']+)'\s+'WATER'\s+'(OPEN|SHUT)'\s+'RATE'\s+([0-9.eE+-]+)\s*/\s*"
)


class ScheduleError(ValueError):
    """Schedule is invalid or cannot be losslessly round-tripped."""


@dataclass(frozen=True, slots=True)
class ScheduleArtifact:
    text: str
    sha256: str
    actions: tuple[ControlAction, ...]


def _action_key(action: ControlAction) -> tuple[date, str, str, str, str, float]:
    return (
        action.month,
        action.role.value,
        action.well,
        action.status.value,
        action.target.value,
        action.value,
    )


def _normalized(actions: Iterable[ControlAction]) -> tuple[ControlAction, ...]:
    return tuple(sorted(actions, key=_action_key))


class ScheduleCompiler:
    """Compile and parse only the explicit DATES/WCONPROD/WCONINJE subset."""

    def validate(self, case: Case, actions: Iterable[ControlAction]) -> tuple[ControlAction, ...]:
        ordered = _normalized(actions)
        seen: set[tuple[date, str]] = set()
        for action in ordered:
            if not case.start <= action.month <= case.end:
                raise ScheduleError(f"action {action.well} is outside case horizon")
            if case.role_of(action.well) is not action.role:
                raise ScheduleError(f"wrong role for well {action.well}")
            key = (action.month, action.well)
            if key in seen:
                raise ScheduleError(f"duplicate monthly control for well {action.well}")
            seen.add(key)
            if (
                action.role is WellRole.PRODUCER
                and action.target is ControlTarget.LIQUID_RATE
                and action.value > case.max_liquid_rate
            ):
                raise ScheduleError(
                    f"liquid target for {action.well} exceeds {case.max_liquid_rate:g}"
                )
        return ordered

    def compile(self, case: Case, actions: Iterable[ControlAction]) -> ScheduleArtifact:
        ordered = self.validate(case, actions)
        lines = ["-- TIMESOIL AIOS GENERATED; DO NOT EDIT"]
        months = sorted({action.month for action in ordered})
        for month in months:
            monthly = tuple(action for action in ordered if action.month == month)
            lines.extend(("", "DATES", f"  1 {_MONTHS[month.month - 1]} {month.year} /", "/"))
            producers = [a for a in monthly if a.role is WellRole.PRODUCER]
            injectors = [a for a in monthly if a.role is WellRole.INJECTOR]
            if producers:
                lines.extend(("", "WCONPROD"))
                lines.extend(self._producer_line(action) for action in producers)
                lines.append("/")
            if injectors:
                lines.extend(("", "WCONINJE"))
                lines.extend(self._injector_line(action) for action in injectors)
                lines.append("/")
        text = "\n".join(lines) + "\n"
        parsed = self.parse(case, text)
        if parsed != ordered:
            raise ScheduleError("generated schedule failed parser round-trip")
        return ScheduleArtifact(text, sha256(text.encode()).hexdigest(), ordered)

    def parse(self, case: Case, text: str) -> tuple[ControlAction, ...]:
        actions: list[ControlAction] = []
        current_month: date | None = None
        block: str | None = None
        lines = iter(enumerate(text.splitlines(), start=1))
        for line_number, raw in lines:
            line = raw.strip()
            if not line or line.startswith("--"):
                continue
            if line == "DATES":
                date_number, date_raw = next(lines, (0, ""))
                match = _DATE_LINE.fullmatch(date_raw)
                if not match:
                    raise ScheduleError(f"invalid DATES entry on line {date_number}")
                day, month_name, year = match.groups()
                if month_name not in _MONTHS:
                    raise ScheduleError(f"invalid month on line {date_number}")
                current_month = date(int(year), _MONTHS.index(month_name) + 1, int(day))
                end_number, end_raw = next(lines, (0, ""))
                if end_raw.strip() != "/":
                    raise ScheduleError(f"unterminated DATES block on line {end_number}")
                block = None
                continue
            if line in {"WCONPROD", "WCONINJE"}:
                if current_month is None:
                    raise ScheduleError(f"{line} before DATES on line {line_number}")
                block = line
                continue
            if line == "/":
                if block is None:
                    raise ScheduleError(f"unexpected block terminator on line {line_number}")
                block = None
                continue
            if current_month is None:
                raise ScheduleError(f"control before DATES on line {line_number}")
            if block == "WCONPROD":
                actions.append(self._parse_producer(current_month, line, line_number))
            elif block == "WCONINJE":
                actions.append(self._parse_injector(current_month, line, line_number))
            else:
                raise ScheduleError(f"unsupported schedule syntax on line {line_number}")
        if block is not None:
            raise ScheduleError(f"unterminated {block} block")
        return self.validate(case, actions)

    @staticmethod
    def _producer_line(action: ControlAction) -> str:
        target = action.target.value
        value = f"{action.value:.6f}"
        controls = value if action.target is ControlTarget.OIL_RATE else f"3* {value}"
        return f"  '{action.well}' '{action.status.value}' '{target}' {controls} /"

    @staticmethod
    def _injector_line(action: ControlAction) -> str:
        return (
            f"  '{action.well}' 'WATER' '{action.status.value}' "
            f"'RATE' {action.value:.6f} /"
        )

    @staticmethod
    def _parse_producer(month: date, line: str, line_number: int) -> ControlAction:
        match = _PROD_LINE.fullmatch(line)
        if not match:
            raise ScheduleError(f"invalid WCONPROD entry on line {line_number}")
        well, status, target, controls = match.groups()
        parts = controls.split()
        if target == ControlTarget.OIL_RATE.value and len(parts) == 1:
            value = parts[0]
        elif target == ControlTarget.LIQUID_RATE.value and len(parts) == 2 and parts[0] == "3*":
            value = parts[1]
        else:
            raise ScheduleError(f"unsupported producer controls on line {line_number}")
        try:
            return ControlAction(
                month, well, WellRole.PRODUCER, WellStatus(status), ControlTarget(target), float(value)
            )
        except ValueError as exc:
            raise ScheduleError(f"invalid WCONPROD value on line {line_number}") from exc

    @staticmethod
    def _parse_injector(month: date, line: str, line_number: int) -> ControlAction:
        match = _INJ_LINE.fullmatch(line)
        if not match:
            raise ScheduleError(f"invalid WCONINJE entry on line {line_number}")
        well, status, value = match.groups()
        try:
            return ControlAction(
                month,
                well,
                WellRole.INJECTOR,
                WellStatus(status),
                ControlTarget.WATER_INJECTION_RATE,
                float(value),
            )
        except ValueError as exc:
            raise ScheduleError(f"invalid WCONINJE value on line {line_number}") from exc
