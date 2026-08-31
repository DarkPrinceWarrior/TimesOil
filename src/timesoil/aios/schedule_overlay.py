"""Fail-closed control overlays for existing Eclipse schedule includes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import re
from typing import Literal

from .contracts import ControlAction, ControlTarget, WellRole
from .schedule import ScheduleArtifact, ScheduleCompiler


_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_MONTH_NUMBER = {name: index for index, name in enumerate(_MONTHS, start=1)}
_DATES = re.compile(r"DATES", re.IGNORECASE)
_DATE_RECORD = re.compile(r"(\d{1,2})\s+'?([A-Z]{3})'?\s+(\d{4})\s*/", re.IGNORECASE)
_WELL = re.compile(r"[A-Za-z0-9_.-]+")
_TOKEN = re.compile(r"'(?:''|[^'])*'|[^\s'/]+")
_REPEAT = re.compile(r"([1-9]\d*)\*(.*)")
_MARKER = "-- TIMESOIL AIOS OVERRIDE"


class ScheduleOverlayError(ValueError):
    """Source schedule or requested overlay is ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class ScheduleOverlayArtifact:
    """Rendered include plus its complete deterministic provenance."""

    text: str
    sha256: str
    source_sha256: str
    controls_sha256: str
    mode: Literal["full", "one_month"]
    action_count: int
    action_months: tuple[date, ...]
    truncated_after: date | None

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "controls_sha256": self.controls_sha256,
            "output_sha256": self.sha256,
            "mode": self.mode,
            "action_count": self.action_count,
            "action_months": [month.isoformat() for month in self.action_months],
            "truncated_after": (
                self.truncated_after.isoformat() if self.truncated_after else None
            ),
        }


@dataclass(frozen=True, slots=True)
class _DateBlock:
    month: date
    keyword_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class _ControlTemplate:
    line: int
    well: str
    role: WellRole
    fields: tuple[str, ...]
    fluid: str | None = None


def apply_schedule_overlay(
    source: str,
    controls: ScheduleArtifact | Iterable[ControlAction],
    *,
    known_wells: Iterable[str],
    replay_month: date | None = None,
) -> ScheduleOverlayArtifact:
    """Append controls to matching report-date blocks without mutating ``source``.

    ``replay_month`` selects one-month replay: only that month's controls are
    accepted and output stops immediately after the following ``DATES`` block.
    Without it, every requested month is overlaid and the full source is kept.
    """

    _validate_source(source)
    wells = _validate_wells(known_wells)
    actions, controls_hash = _validated_controls(controls, wells)
    blocks = _date_blocks(source)
    templates = _control_templates(source)
    by_month = {block.month: block for block in blocks}
    months = tuple(sorted({action.month for action in actions}))
    missing = [month for month in months if month not in by_month]
    if missing:
        raise ScheduleOverlayError(
            "control date absent from source schedule: "
            + ", ".join(month.isoformat() for month in missing)
        )

    mode: Literal["full", "one_month"] = "one_month" if replay_month else "full"
    truncated_after: date | None = None
    cutoff: int | None = None
    if replay_month is not None:
        if replay_month.day != 1 or months != (replay_month,):
            raise ScheduleOverlayError(
                "one-month replay requires controls only for replay_month"
            )
        index = next(
            i for i, block in enumerate(blocks) if block.month == replay_month
        )
        if index + 1 == len(blocks):
            raise ScheduleOverlayError("one-month replay requires a following date")
        following = blocks[index + 1]
        truncated_after = following.month
        cutoff = following.end_line

    lines = source.splitlines(keepends=True)
    terminal = _terminal_line(lines, blocks[-1])
    insertions: dict[int, str] = {}
    for month in months:
        index = next(i for i, block in enumerate(blocks) if block.month == month)
        boundary = blocks[index + 1].keyword_line if index + 1 < len(blocks) else terminal
        monthly = tuple(action for action in actions if action.month == month)
        effective = {
            (template.well, template.role): template
            for template in templates
            if template.line < boundary
            and (template.role is WellRole.PRODUCER or template.fluid == "WATER")
        }
        insertions[boundary] = _render_override(
            month, monthly, controls_hash, effective
        )

    output: list[str] = []
    limit = len(lines) if cutoff is None else cutoff
    for index in range(limit + 1):
        overlay = insertions.get(index)
        if overlay is not None:
            if output and not output[-1].endswith(("\n", "\r")):
                output.append("\n")
            output.append(overlay)
        if index < limit:
            output.append(lines[index])
    text = "".join(output)
    digest = sha256(text.encode()).hexdigest()
    return ScheduleOverlayArtifact(
        text=text,
        sha256=digest,
        source_sha256=sha256(source.encode()).hexdigest(),
        controls_sha256=controls_hash,
        mode=mode,
        action_count=len(actions),
        action_months=months,
        truncated_after=truncated_after,
    )


def _validate_source(source: str) -> None:
    if not source.strip():
        raise ScheduleOverlayError("source schedule is empty")
    if _MARKER in source:
        raise ScheduleOverlayError("source already contains a TimesOil override")
    if any(ord(char) < 32 and char not in "\n\t\r" for char in source):
        raise ScheduleOverlayError("source contains unsafe control characters")


def _validate_wells(values: Iterable[str]) -> frozenset[str]:
    wells = tuple(values)
    if not wells or any(not isinstance(well, str) or not _WELL.fullmatch(well) for well in wells):
        raise ScheduleOverlayError("known_wells contains an unsafe or empty name")
    if len(set(wells)) != len(wells):
        raise ScheduleOverlayError("known_wells contains duplicates")
    return frozenset(wells)


def _validated_controls(
    controls: ScheduleArtifact | Iterable[ControlAction], wells: frozenset[str]
) -> tuple[tuple[ControlAction, ...], str]:
    if isinstance(controls, ScheduleArtifact):
        artifact = controls
        actions = artifact.actions
    else:
        artifact = None
        try:
            actions = tuple(controls)
        except TypeError as exc:
            raise ScheduleOverlayError("controls must be iterable") from exc
    if not actions or any(not isinstance(action, ControlAction) for action in actions):
        raise ScheduleOverlayError("controls must contain ControlAction values")

    ordered = tuple(sorted(actions, key=_action_key))
    if len({(action.month, action.well) for action in ordered}) != len(ordered):
        raise ScheduleOverlayError("duplicate monthly control for a well")
    unknown = sorted({action.well for action in ordered} - wells)
    if unknown:
        raise ScheduleOverlayError(f"unknown controlled wells: {unknown}")

    canonical = _canonical_schedule(ordered)
    digest = sha256(canonical.encode()).hexdigest()
    if artifact is not None and (
        artifact.actions != ordered
        or artifact.text != canonical
        or artifact.sha256 != digest
    ):
        raise ScheduleOverlayError("ScheduleArtifact failed canonical hash validation")
    return ordered, digest


def _action_key(action: ControlAction) -> tuple[date, str, str, str, str, float]:
    return (
        action.month,
        action.role.value,
        action.well,
        action.status.value,
        action.target.value,
        action.value,
    )


def _canonical_schedule(actions: tuple[ControlAction, ...]) -> str:
    lines = ["-- TIMESOIL AIOS GENERATED; DO NOT EDIT"]
    for month in sorted({action.month for action in actions}):
        monthly = tuple(action for action in actions if action.month == month)
        lines.extend(("", "DATES", f"  1 {_MONTHS[month.month - 1]} {month.year} /", "/"))
        producers = [action for action in monthly if action.role is WellRole.PRODUCER]
        injectors = [action for action in monthly if action.role is WellRole.INJECTOR]
        if producers:
            lines.extend(("", "WCONPROD"))
            lines.extend(ScheduleCompiler._producer_line(action) for action in producers)
            lines.append("/")
        if injectors:
            lines.extend(("", "WCONINJE"))
            lines.extend(ScheduleCompiler._injector_line(action) for action in injectors)
            lines.append("/")
    return "\n".join(lines) + "\n"


def _date_blocks(source: str) -> tuple[_DateBlock, ...]:
    lines = source.splitlines(keepends=True)
    blocks: list[_DateBlock] = []
    index = 0
    while index < len(lines):
        if not _DATES.fullmatch(_code(lines[index])):
            index += 1
            continue
        keyword_line = index
        index += 1
        records: list[date] = []
        while index < len(lines) and _code(lines[index]) != "/":
            code = _code(lines[index])
            if code:
                match = _DATE_RECORD.fullmatch(code)
                if not match:
                    raise ScheduleOverlayError(
                        f"invalid DATES syntax on line {index + 1}"
                    )
                day, month_name, year = match.groups()
                try:
                    records.append(
                        date(int(year), _MONTH_NUMBER[month_name.upper()], int(day))
                    )
                except (KeyError, ValueError) as exc:
                    raise ScheduleOverlayError(
                        f"invalid report date on line {index + 1}"
                    ) from exc
            index += 1
        if index == len(lines):
            raise ScheduleOverlayError(f"unterminated DATES block on line {keyword_line + 1}")
        if len(records) != 1:
            raise ScheduleOverlayError(
                "each DATES keyword must contain exactly one report date"
            )
        blocks.append(_DateBlock(records[0], keyword_line, index + 1))
        index += 1

    if not blocks:
        raise ScheduleOverlayError("source has no DATES blocks")
    dates = [block.month for block in blocks]
    if len(set(dates)) != len(dates):
        raise ScheduleOverlayError("source contains duplicate report dates")
    if dates != sorted(dates):
        raise ScheduleOverlayError("source report dates are not strictly increasing")
    return tuple(blocks)


def _terminal_line(lines: list[str], final_block: _DateBlock) -> int:
    ends = [
        index
        for index in range(final_block.end_line, len(lines))
        if _code(lines[index]).upper() == "END"
    ]
    if len(ends) > 1:
        raise ScheduleOverlayError("source contains duplicate END keywords")
    return ends[0] if ends else len(lines)


def _control_templates(source: str) -> tuple[_ControlTemplate, ...]:
    templates: list[_ControlTemplate] = []
    block: WellRole | None = None
    for index, raw in enumerate(source.splitlines()):
        code = _code(raw)
        upper = code.upper()
        if block is None:
            if upper == "WCONPROD":
                block = WellRole.PRODUCER
            elif upper == "WCONINJE":
                block = WellRole.INJECTOR
            elif upper.startswith(("WCONPROD", "WCONINJE")):
                raise ScheduleOverlayError(
                    f"unsupported WCON keyword syntax on line {index + 1}"
                )
            continue
        if not code:
            continue
        if code == "/":
            block = None
            continue
        if not code.endswith("/"):
            raise ScheduleOverlayError(f"unterminated WCON entry on line {index + 1}")

        tokens = _record_tokens(code[:-1], index + 1)
        minimum = 3 if block is WellRole.PRODUCER else 4
        if len(tokens) < minimum:
            raise ScheduleOverlayError(f"incomplete WCON entry on line {index + 1}")
        well = _word(tokens[0])
        if not _WELL.fullmatch(well):
            raise ScheduleOverlayError(f"unsupported well name on line {index + 1}")
        offset = 3 if block is WellRole.PRODUCER else 4
        fields = _expanded_fields(tokens[offset:], index + 1)
        fluid = None if block is WellRole.PRODUCER else _word(tokens[1]).upper()
        templates.append(_ControlTemplate(index, well, block, fields, fluid))
    if block is not None:
        raise ScheduleOverlayError(f"unterminated WCON{block.value} block")
    return tuple(templates)


def _record_tokens(record: str, line: int) -> tuple[str, ...]:
    tokens: list[str] = []
    end = 0
    for match in _TOKEN.finditer(record):
        gap = record[end : match.start()]
        if (end and not gap) or (gap and not gap.isspace()):
            raise ScheduleOverlayError(f"invalid WCON syntax on line {line}")
        tokens.append(match.group())
        end = match.end()
    if record[end:].strip():
        raise ScheduleOverlayError(f"invalid WCON syntax on line {line}")
    return tuple(tokens)


def _expanded_fields(tokens: tuple[str, ...], line: int) -> tuple[str, ...]:
    fields: list[str] = []
    for token in tokens:
        repeat = _REPEAT.fullmatch(token)
        if repeat is None:
            if "*" in token:
                raise ScheduleOverlayError(f"invalid WCON repeat on line {line}")
            fields.append(token)
        else:
            count, value = repeat.groups()
            fields.extend([value or "1*"] * int(count))
        if len(fields) > 32:
            raise ScheduleOverlayError(f"unsupported WCON record width on line {line}")
    return tuple(fields)


def _word(token: str) -> str:
    return token[1:-1].replace("''", "'") if token.startswith("'") else token


def _render_override(
    month: date,
    actions: tuple[ControlAction, ...],
    controls_hash: str,
    templates: dict[tuple[str, WellRole], _ControlTemplate],
) -> str:
    lines = ["", f"{_MARKER} {month.isoformat()} {controls_hash}"]
    producers = [action for action in actions if action.role is WellRole.PRODUCER]
    injectors = [action for action in actions if action.role is WellRole.INJECTOR]
    if producers:
        lines.append("WCONPROD")
        lines.extend(
            _render_action(action, templates.get((action.well, action.role)))
            for action in producers
        )
        lines.append("/")
    if injectors:
        lines.append("WCONINJE")
        lines.extend(
            _render_action(action, templates.get((action.well, action.role)))
            for action in injectors
        )
        lines.append("/")
    lines.append("")
    return "\n".join(lines)


def _render_action(action: ControlAction, template: _ControlTemplate | None) -> str:
    if template is None or (
        action.role is WellRole.INJECTOR and template.fluid != "WATER"
    ):
        renderer = (
            ScheduleCompiler._producer_line
            if action.role is WellRole.PRODUCER
            else ScheduleCompiler._injector_line
        )
        return renderer(action)

    fields = list(template.fields)
    primary = (
        3
        if action.role is WellRole.PRODUCER
        and action.target is ControlTarget.LIQUID_RATE
        else 0
    )
    fields.extend(["1*"] * (primary + 1 - len(fields)))
    fields[primary] = f"{action.value:.6f}"
    if action.role is WellRole.PRODUCER:
        head = f"  '{action.well}' '{action.status.value}' '{action.target.value}'"
    else:
        head = f"  '{action.well}' 'WATER' '{action.status.value}' 'RATE'"
    return f"{head} {' '.join(fields)} /"


def _code(raw: str) -> str:
    """Return code before an Eclipse ``--`` comment, respecting quotes."""

    quoted = False
    index = 0
    while index < len(raw):
        if raw[index] == "'":
            if quoted and index + 1 < len(raw) and raw[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif not quoted and raw[index : index + 2] == "--":
            return raw[:index].strip()
        index += 1
    if quoted:
        raise ScheduleOverlayError("source contains an unterminated quote")
    return raw.strip()
