"""Deterministic OPM SUMMARY export for official CHDD and Track 2 inputs."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .economics import CHDD_FIELDS
from .opm import OpmSummaryError, verify_summary_extraction
from .track2 import CANONICAL_COLUMNS as TRACK2_FIELDS


REQUIRED_VECTORS = (
    "WLPR",
    "WLPT",
    "WOPR",
    "WOPT",
    "WOIR",
    "WOIT",
    "WWPR",
    "WWPT",
    "WWIR",
    "WWIT",
    "WBHP",
    "WBP9",
    "WEFF",
)
CONNECTION_VECTORS = (
    "COFR",
    "CWFR",
    "COPR",
    "COPT",
    "CWPR",
    "CWPT",
    "COIT",
    "CWIR",
    "CWIT",
)
_STB_TO_M3 = 0.158987294928
_PSI_TO_BAR = 0.0689475729318
_LB_FT3_TO_KG_M3 = 16.01846337396
_KEYWORD = re.compile(r"(?mi)^\s*{keyword}\b")
_INCLUDE = re.compile(
    r"(?is)^\s*INCLUDE\s+(['\"])(.+?)\1\s*/\s*$"
)
_MONTHS = {
    name: index
    for index, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        1,
    )
}


class OpmChddError(ValueError):
    """SUMMARY or deck cannot be converted without an unsupported assumption."""


@dataclass(frozen=True, slots=True)
class SurfaceDensity:
    oil_kg_m3: float
    water_kg_m3: float

    def __post_init__(self) -> None:
        for name, value in (
            ("oil_kg_m3", self.oil_kg_m3),
            ("water_kg_m3", self.water_kg_m3),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise OpmChddError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class ScheduledControl:
    target: str
    value: float
    status: int


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _strip_comment(line: str) -> str:
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if character in "'\"":
            quote = None if quote == character else character if quote is None else quote
        elif quote is None and line[index : index + 2] == "--":
            return line[:index]
        index += 1
    return line


def _slash_outside_quotes(value: str) -> bool:
    quote: str | None = None
    for character in value:
        if character in "'\"":
            quote = None if quote == character else character if quote is None else quote
        elif character == "/" and quote is None:
            return True
    return False


def _deck_text(root: Path) -> tuple[str, tuple[Path, ...]]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise OpmChddError(f"deck directory is not a regular directory: {root}")
    decks = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.upper() == ".DATA"
    )
    if len(decks) != 1:
        raise OpmChddError(f"expected exactly one .DATA deck, found {len(decks)}")

    used: set[Path] = set()

    def expand(path: Path, stack: tuple[Path, ...]) -> str:
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise OpmChddError(f"INCLUDE escapes deck directory: {path}") from None
        if path in stack:
            raise OpmChddError(f"cyclic INCLUDE: {path.relative_to(root)}")
        if not path.is_file() or path.is_symlink():
            raise OpmChddError(f"INCLUDE is not a regular file: {path}")
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeError as exc:
            raise OpmChddError(f"deck file is not UTF-8: {path}") from exc
        used.add(path)
        output: list[str] = []
        index = 0
        while index < len(lines):
            line = _strip_comment(lines[index])
            if not re.match(r"(?i)^\s*INCLUDE\b", line):
                output.append(line)
                index += 1
                continue
            include = line
            while not _slash_outside_quotes(include):
                index += 1
                if index >= len(lines):
                    raise OpmChddError(f"unterminated INCLUDE in {path}")
                include += "\n" + _strip_comment(lines[index])
            match = _INCLUDE.fullmatch(include)
            if match is None:
                raise OpmChddError(f"unsupported INCLUDE syntax in {path}: {include!r}")
            relative = Path(match.group(2))
            if relative.is_absolute() or ".." in relative.parts:
                raise OpmChddError(f"unsafe INCLUDE path: {match.group(2)!r}")
            output.append(expand(path.parent / relative, (*stack, path)))
            index += 1
        return "\n".join(output)

    return expand(decks[0], ()), tuple(sorted(used))


def _keyword_offsets(text: str, keyword: str) -> list[re.Match[str]]:
    return list(re.finditer(_KEYWORD.pattern.format(keyword=re.escape(keyword)), text, _KEYWORD.flags))


def _tokens(value: str) -> Iterator[str]:
    lexer = shlex.shlex(value, posix=True, punctuation_chars="/")
    lexer.whitespace_split = True
    lexer.commenters = ""
    yield from lexer


def _records_after(text: str, match: re.Match[str], count: int | None) -> list[list[str]]:
    records: list[list[str]] = []
    current: list[str] = []
    for token in _tokens(text[match.end() :]):
        if token != "/":
            current.append(token)
            continue
        if not current:
            if count is None:
                return records
            raise OpmChddError(f"{match.group(0).strip()} ended before {count} records")
        records.append(current)
        current = []
        if count is not None and len(records) == count:
            return records
    raise OpmChddError(f"unterminated {match.group(0).strip()} keyword")


def _single_record(text: str, keyword: str) -> list[str]:
    matches = _keyword_offsets(text, keyword)
    if len(matches) != 1:
        raise OpmChddError(f"deck must contain exactly one {keyword}, found {len(matches)}")
    return _records_after(text, matches[0], 1)[0]


def _number(value: str, context: str) -> float:
    try:
        result = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise OpmChddError(f"invalid number for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise OpmChddError(f"non-finite number for {context}: {value!r}")
    return result


def _integer(value: str, context: str) -> int:
    result = _number(value, context)
    if not result.is_integer():
        raise OpmChddError(f"non-integer value for {context}: {value!r}")
    return int(result)


def _eclipse_date(record: Sequence[str], context: str) -> date:
    if len(record) < 3:
        raise OpmChddError(f"{context} must contain day month year")
    month = _MONTHS.get(record[1].upper())
    if month is None:
        raise OpmChddError(f"unsupported {context} month: {record[1]!r}")
    try:
        return date(
            _integer(record[2], f"{context} year"),
            month,
            _integer(record[0], f"{context} day"),
        )
    except ValueError as exc:
        raise OpmChddError(f"invalid {context}: {record[:3]!r}") from exc


def _expanded_integers(tokens: Iterable[str], context: str) -> list[int]:
    values: list[int] = []
    for token in tokens:
        if "*" not in token:
            values.append(_integer(token, context))
            continue
        count, repeated = token.split("*", 1)
        if not count or not repeated:
            raise OpmChddError(f"defaulted repeat is unsupported in {context}: {token!r}")
        repetitions = _integer(count, f"{context} repeat count")
        if repetitions < 1:
            raise OpmChddError(f"invalid repeat count in {context}: {token!r}")
        values.extend([_integer(repeated, context)] * repetitions)
    return values


def _expanded_record(tokens: Sequence[str], context: str) -> list[str]:
    values: list[str] = []
    for token in tokens:
        match = re.fullmatch(r"([0-9]+)\*(.*)", token)
        if match is None:
            values.append(token)
            continue
        count = _integer(match.group(1), f"{context} repeat count")
        if count < 1:
            raise OpmChddError(f"invalid repeat count in {context}: {token!r}")
        values.extend([match.group(2) or "*"] * count)
    return values


def _read_deck_densities_and_start(
    deck_dir: str | Path,
) -> tuple[
    str,
    dict[str, SurfaceDensity],
    dict[str, tuple[int, ...]],
    str,
    dict[str, dict[str, SurfaceDensity]],
    date | None,
    str,
]:
    """Map simple IJK COMPDAT wells to DENSITY through active PVTNUM cells.

    Wells completed in zero or multiple PVT regions remain in ``ambiguous``;
    callers must provide an explicit density instead of averaging regions.
    """

    text, files = _deck_text(Path(deck_dir))
    unit_systems = [
        unit
        for unit in ("METRIC", "FIELD")
        if re.search(rf"(?mi)^\s*{unit}\s*$", text)
    ]
    if len(unit_systems) != 1:
        raise OpmChddError("deck must declare exactly one supported unit system: METRIC or FIELD")
    unit_system = unit_systems[0]

    start_matches = _keyword_offsets(text, "START")
    if len(start_matches) > 1:
        raise OpmChddError(f"deck must contain at most one START, found {len(start_matches)}")
    start_date: date | None = None
    if start_matches:
        start_record = _records_after(text, start_matches[0], 1)[0]
        start_date = _eclipse_date(start_record, "START")

    dimens = _single_record(text, "DIMENS")
    if len(dimens) < 3:
        raise OpmChddError("DIMENS must contain NX NY NZ")
    nx, ny, nz = (_integer(value, "DIMENS") for value in dimens[:3])
    if min(nx, ny, nz) < 1:
        raise OpmChddError("DIMENS values must be positive")

    tabdims = _single_record(text, "TABDIMS")
    density_count = _integer(tabdims[0], "TABDIMS NTPVT")
    density_matches = _keyword_offsets(text, "DENSITY")
    if len(density_matches) != 1 or density_count < 1:
        raise OpmChddError("deck must contain one DENSITY table and positive TABDIMS NTPVT")
    factor = 1.0 if unit_system == "METRIC" else _LB_FT3_TO_KG_M3
    density_records = _records_after(text, density_matches[0], density_count)
    densities: list[SurfaceDensity] = []
    for index, record in enumerate(density_records, 1):
        if len(record) != 3 or any("*" in value for value in record):
            raise OpmChddError(f"DENSITY region {index} must explicitly contain oil water gas")
        densities.append(
            SurfaceDensity(
                _number(record[0], f"DENSITY region {index} oil") * factor,
                _number(record[1], f"DENSITY region {index} water") * factor,
            )
        )

    pvtnum = _expanded_integers(_single_record(text, "PVTNUM"), "PVTNUM")
    expected = nx * ny * nz
    if len(pvtnum) != expected:
        raise OpmChddError(f"PVTNUM contains {len(pvtnum)} cells, expected {expected}")
    actnum = _expanded_integers(_single_record(text, "ACTNUM"), "ACTNUM")
    if len(actnum) != expected or any(value not in {0, 1} for value in actnum):
        raise OpmChddError(f"ACTNUM must contain exactly {expected} zero/one cells")
    coordinates_by_well: dict[str, tuple[int, int]] = {}
    for match in _keyword_offsets(text, "WELSPECS"):
        for record in _records_after(text, match, None):
            if len(record) < 4 or any("*" in value for value in record[2:4]):
                raise OpmChddError(f"WELSPECS must explicitly contain well/group/I/J: {record!r}")
            well = record[0].strip()
            coordinates = (
                _integer(record[2], f"WELSPECS {well} I"),
                _integer(record[3], f"WELSPECS {well} J"),
            )
            previous = coordinates_by_well.setdefault(well, coordinates)
            if previous != coordinates:
                raise OpmChddError(f"WELSPECS changes I/J for well {well!r}")

    regions_by_well: dict[str, set[int]] = {}
    connections_by_well: dict[str, dict[str, SurfaceDensity]] = {}
    compdat_matches = _keyword_offsets(text, "COMPDAT")
    if not compdat_matches:
        raise OpmChddError("deck contains no COMPDAT records")
    for match in compdat_matches:
        for record in _records_after(text, match, None):
            if len(record) < 5:
                raise OpmChddError(f"short COMPDAT record: {record!r}")
            well = record[0].strip()
            if not well:
                raise OpmChddError("COMPDAT contains a blank well")
            coordinates = coordinates_by_well.get(well)
            ij: list[int] = []
            for offset, value in enumerate(record[1:3]):
                if value == "1*":
                    if coordinates is None:
                        raise OpmChddError(
                            f"COMPDAT defaults I/J without WELSPECS for well {well!r}"
                        )
                    ij.append(coordinates[offset])
                elif "*" in value:
                    raise OpmChddError(f"unsupported COMPDAT I/J default for well {well!r}")
                else:
                    ij.append(_integer(value, f"COMPDAT {well}"))
            if any("*" in value for value in record[3:5]):
                raise OpmChddError(f"COMPDAT defaults K1/K2 for well {well!r}")
            i, j = ij
            k1, k2 = (_integer(value, f"COMPDAT {well}") for value in record[3:5])
            if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k1 <= k2 <= nz):
                raise OpmChddError(f"COMPDAT cell is outside DIMENS for well {well!r}")
            well_regions = regions_by_well.setdefault(well, set())
            well_connections = connections_by_well.setdefault(well, {})
            for k in range(k1, k2 + 1):
                cell = (k - 1) * nx * ny + (j - 1) * nx + i - 1
                if actnum[cell]:
                    region = pvtnum[cell]
                    if not 1 <= region <= len(densities):
                        raise OpmChddError(
                            f"active COMPDAT cell has invalid PVTNUM for well {well!r}: "
                            f"{i},{j},{k}"
                        )
                    well_regions.add(region)
                    well_connections[f"{i},{j},{k}"] = densities[region - 1]

    resolved: dict[str, SurfaceDensity] = {}
    ambiguous: dict[str, tuple[int, ...]] = {}
    for well, regions in regions_by_well.items():
        ordered = tuple(sorted(regions))
        if len(ordered) == 1 and 1 <= ordered[0] <= len(densities):
            resolved[well] = densities[ordered[0] - 1]
        else:
            ambiguous[well] = ordered
    entries = [
        (path.relative_to(Path(deck_dir).resolve()).as_posix(), path.stat().st_size, _sha256_file(path))
        for path in files
    ]
    connection_densities = {
        well: dict(sorted(values.items()))
        for well, values in connections_by_well.items()
    }
    return (
        unit_system,
        resolved,
        ambiguous,
        _sha256_bytes(_canonical_json(entries)),
        connection_densities,
        start_date,
        text,
    )


def read_deck_densities(
    deck_dir: str | Path,
) -> tuple[
    str,
    dict[str, SurfaceDensity],
    dict[str, tuple[int, ...]],
    str,
    dict[str, dict[str, SurfaceDensity]],
]:
    """Return strict well/connection surface densities and deck digest."""

    unit, resolved, ambiguous, digest, connections, _, _ = (
        _read_deck_densities_and_start(deck_dir)
    )
    return unit, resolved, ambiguous, digest, connections


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise OpmChddError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _opm_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpmChddError(f"cannot read OPM run manifest: {path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "timesoil.aios.opm-run/v1"
    ):
        raise OpmChddError("OPM run manifest has an unsupported schema")
    source_sha256 = manifest.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise OpmChddError("OPM run manifest source_sha256 must be lowercase SHA-256")
    return manifest, source_sha256


def load_density_mapping(path: str | Path) -> dict[str, SurfaceDensity]:
    """Read ``{well: {oil_kg_m3, water_kg_m3}}`` JSON without defaults."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpmChddError(f"cannot read density mapping: {source}") from exc
    if not isinstance(raw, dict) or not raw:
        raise OpmChddError("density mapping must be a non-empty JSON object")
    result: dict[str, SurfaceDensity] = {}
    for raw_well, values in raw.items():
        well = str(raw_well).strip()
        if not well or well != raw_well or not isinstance(values, dict):
            raise OpmChddError(f"invalid density mapping entry: {raw_well!r}")
        expected = {"oil_kg_m3", "water_kg_m3"}
        if set(values) != expected:
            raise OpmChddError(f"density for {well!r} must contain exactly {sorted(expected)}")
        result[well] = SurfaceDensity(
            _mapping_number(values["oil_kg_m3"], f"density {well} oil"),
            _mapping_number(values["water_kg_m3"], f"density {well} water"),
        )
    return result


def _mapping_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpmChddError(f"{context} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise OpmChddError(f"{context} must be finite")
    return result


def _date(value: str, row: int) -> date:
    normalized = value.strip()
    formats = ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%d.%m.%Y")
    for format_ in formats:
        try:
            parsed = datetime.strptime(normalized, format_).date()
            return parsed
        except ValueError:
            pass
    raise OpmChddError(f"summary row {row} has an unsupported DATE: {value!r}")


def _read_summary(
    path: Path,
    *,
    start_date: date | None = None,
) -> tuple[
    list[
        tuple[
            date,
            dict[str, dict[str, float]],
            dict[str, dict[str, dict[str, float]]],
        ]
    ],
    list[str],
]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise OpmChddError(f"cannot read SUMMARY table: {path}") from exc
    with stream:
        first_line = next((line for line in stream if line.strip()), None)
        if first_line is None:
            raise OpmChddError("SUMMARY table is empty")
        stripped_header = first_line.strip()
        if "," in stripped_header and not any(
            character.isspace() for character in stripped_header
        ):
            reader: Iterable[list[str]] = csv.reader(chain((first_line,), stream))
            raw_headers = next(iter(reader))
        else:
            raw_headers = first_line.split()
            reader = (line.split() for line in stream if line.strip())
        if not raw_headers:
            raise OpmChddError("SUMMARY table has no headers")
        columns: list[tuple[str, str | None, str | None]] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        ignored: list[str] = []
        for raw_header in raw_headers:
            header = raw_header.strip()
            if header.upper() in {"DATE", "TIME", "YEARS"} and ":" not in header:
                column = (header.upper(), None, None)
                if column[0] != "DATE":
                    ignored.append(column[0])
            else:
                parts = [part.strip() for part in header.split(":")]
                if len(parts) == 2 and parts[0].upper() in REQUIRED_VECTORS and parts[1]:
                    column = (parts[0].upper(), parts[1], None)
                elif (
                    len(parts) == 3
                    and parts[0].upper() in CONNECTION_VECTORS
                    and parts[1]
                    and re.fullmatch(r"[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*", parts[2])
                ):
                    column = (parts[0].upper(), parts[1], parts[2])
                else:
                    raise OpmChddError(f"unsupported SUMMARY column: {raw_header!r}")
            if column in seen:
                raise OpmChddError(f"duplicate SUMMARY column: {raw_header!r}")
            seen.add(column)
            columns.append(column)
        if ("DATE", None, None) not in seen and (
            start_date is None or ("TIME", None, None) not in seen
        ):
            raise OpmChddError("SUMMARY without DATE requires deck START and TIME")

        wells = sorted(
            {
                well
                for vector, well, completion in columns
                if vector in REQUIRED_VECTORS and well is not None and completion is None
            }
        )
        if not wells:
            raise OpmChddError("SUMMARY CSV contains no well vectors")
        for well in wells:
            missing = [
                vector for vector in REQUIRED_VECTORS if (vector, well, None) not in seen
            ]
            if missing:
                raise OpmChddError(f"well {well!r} misses SUMMARY vectors: {missing}")
        connection_keys = {
            (well, completion)
            for vector, well, completion in columns
            if vector in CONNECTION_VECTORS and well is not None and completion is not None
        }
        for well, completion in sorted(connection_keys):
            if well not in wells:
                raise OpmChddError(f"connection vectors reference unknown well {well!r}")
            missing = [
                vector
                for vector in CONNECTION_VECTORS
                if (vector, well, completion) not in seen
            ]
            if missing:
                raise OpmChddError(
                    f"connection {well}:{completion} misses SUMMARY vectors: {missing}"
                )

        rows: list[
            tuple[
                date,
                dict[str, dict[str, float]],
                dict[str, dict[str, dict[str, float]]],
            ]
        ] = []
        for row_number, values in enumerate(reader, 2):
            if len(values) != len(columns):
                raise OpmChddError(
                    f"summary row {row_number} has {len(values)} cells, expected {len(columns)}"
                )
            if not any(value.strip() for value in values):
                raise OpmChddError(f"summary row {row_number} is blank")
            row_date: date | None = None
            time_days: float | None = None
            by_well = {well: {} for well in wells}
            by_connection: dict[str, dict[str, dict[str, float]]] = {}
            for (vector, well, completion), value in zip(columns, values, strict=True):
                if vector == "DATE":
                    row_date = _date(value, row_number)
                elif vector == "TIME":
                    time_days = _number(value, f"summary row {row_number} TIME")
                    if time_days < 0:
                        raise OpmChddError(f"summary row {row_number} TIME is negative")
                elif vector == "YEARS":
                    years = _number(value, f"summary row {row_number} YEARS")
                    if years < 0:
                        raise OpmChddError(f"summary row {row_number} YEARS is negative")
                else:
                    assert well is not None
                    suffix = f":{completion}" if completion is not None else ""
                    number = _number(
                        value, f"summary row {row_number} {vector}:{well}{suffix}"
                    )
                    # OPM connection vectors are signed: crossflow can make both
                    # instantaneous and cumulative completion values negative.
                    # Well vectors remain non-negative and are reconciled with the
                    # signed completion sums below.
                    if number < 0 and completion is None:
                        raise OpmChddError(
                            f"summary row {row_number} {vector}:{well}{suffix} is negative"
                        )
                    if completion is None:
                        by_well[well][vector] = number
                    else:
                        by_connection.setdefault(well, {}).setdefault(completion, {})[
                            vector
                        ] = number
            if row_date is None:
                assert start_date is not None and time_days is not None
                if not time_days.is_integer():
                    raise OpmChddError(
                        f"summary row {row_number} TIME must be whole days without DATE"
                    )
                row_date = start_date + timedelta(days=int(time_days))
            rows.append((row_date, by_well, by_connection))

    rows.sort(key=lambda item: item[0])
    if len(rows) < 2:
        raise OpmChddError("Track 2 trajectory requires at least two monthly report rows")
    periods = [item[0].year * 12 + item[0].month for item in rows]
    if any(right - left != 1 for left, right in zip(periods, periods[1:])):
        raise OpmChddError("SUMMARY dates must be unique consecutive months")
    for row_number, (_, by_well, _) in enumerate(rows, 1):
        for well, values in by_well.items():
            if values["WOPR"] > values["WLPR"] or values["WOPT"] > values["WLPT"]:
                raise OpmChddError(f"well {well!r} row {row_number}: oil exceeds liquid")
            if not math.isclose(
                values["WWPR"],
                values["WLPR"] - values["WOPR"],
                rel_tol=0,
                abs_tol=_float32_identity_tolerance(
                    values["WWPR"], values["WLPR"], values["WOPR"]
                ),
            ):
                raise OpmChddError(
                    f"well {well!r} row {row_number}: WWPR disagrees with WLPR-WOPR"
                )
            if not math.isclose(
                values["WWPT"],
                values["WLPT"] - values["WOPT"],
                rel_tol=0,
                abs_tol=_float32_identity_tolerance(
                    values["WWPT"], values["WLPT"], values["WOPT"]
                ),
            ):
                raise OpmChddError(
                    f"well {well!r} row {row_number}: WWPT disagrees with WLPT-WOPT"
                )
            if values["WEFF"] > 1:
                raise OpmChddError(f"well {well!r} row {row_number}: WEFF exceeds 1")
            if values["WEFF"] == 0 and (values["WLPR"] > 0 or values["WWIR"] > 0):
                raise OpmChddError(f"well {well!r} row {row_number}: positive rate with WEFF=0")
            if values["WLPR"] > 0 and values["WWIR"] > 0:
                raise OpmChddError(f"well {well!r} row {row_number}: produces and injects")
    return rows, ignored


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0 else value


def _printed_sum_tolerance(printed_terms: int) -> float:
    """Maximum six-decimal text rounding error for a sum/difference identity."""

    return printed_terms * 0.5e-6 + 1e-12


def _float32_identity_tolerance(*terms: float) -> float:
    """Bound independent float32 SUMMARY values plus six-decimal report text."""

    return (
        _printed_sum_tolerance(len(terms))
        + 2**-22 * max(1.0, *(abs(value) for value in terms))
    )


def _check_connection_total(
    well: str,
    month: str,
    vector: str,
    actual: float,
    expected: float,
    printed_terms: int,
) -> None:
    tolerance = _printed_sum_tolerance(printed_terms)
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise OpmChddError(
            f"connection {vector} identity fails for {well!r} {month}: "
            f"{actual} != {expected} (abs_tol={tolerance})"
        )


def _scheduled_control(record: Sequence[str], keyword: str) -> tuple[str, ScheduledControl]:
    values = _expanded_record(record, keyword)
    if keyword == "WCONPROD":
        if len(values) < 3:
            raise OpmChddError(f"short {keyword} record: {record!r}")
        well, status, mode = values[:3]
        target_index = {"ORAT": 3, "LRAT": 6}.get(mode.upper())
        target = mode.upper()
    else:
        if len(values) < 4:
            raise OpmChddError(f"short {keyword} record: {record!r}")
        well, status, mode = values[0], values[2], values[3]
        target_index = 4 if mode.upper() in {"RATE", "WRAT"} else None
        target = "WRAT"
    if not well or any(character in well for character in "*?"):
        raise OpmChddError(f"{keyword} requires an explicit well name: {well!r}")
    normalized_status = status.upper()
    if normalized_status not in {"OPEN", "SHUT", "STOP"}:
        raise OpmChddError(f"unsupported {keyword} status for {well!r}: {status!r}")
    if target_index is None:
        raise OpmChddError(f"unsupported {keyword} control mode for {well!r}: {mode!r}")
    token = values[target_index] if target_index < len(values) else "*"
    if token == "*":
        if normalized_status == "OPEN":
            raise OpmChddError(
                f"active {keyword} control value is defaulted for {well!r}"
            )
        control_value = 0.0
    else:
        control_value = _number(token, f"{keyword} {well} {target}")
        if control_value < 0:
            raise OpmChddError(f"negative {keyword} control value for {well!r}")
    return well, ScheduledControl(target, control_value, int(normalized_status == "OPEN"))


def _scheduled_controls(
    deck_text: str,
    start_date: date | None,
    summary: list[
        tuple[
            date,
            dict[str, dict[str, float]],
            dict[str, dict[str, dict[str, float]]],
        ]
    ],
    wells: Sequence[str],
) -> dict[str, list[ScheduledControl]]:
    if start_date is None:
        raise OpmChddError("deck START is required for requested-control chronology")
    schedules = _keyword_offsets(deck_text, "SCHEDULE")
    if len(schedules) != 1:
        raise OpmChddError(f"deck must contain exactly one SCHEDULE, found {len(schedules)}")
    schedule = deck_text[schedules[0].end() :]
    current_date = start_date
    events: dict[str, list[tuple[date, ScheduledControl]]] = {
        well: [] for well in wells
    }
    commands = re.finditer(
        r"(?mi)^\s*(DATES|TSTEP|WCONPROD|WCONINJE)\b", schedule
    )
    for match in commands:
        keyword = match.group(1).upper()
        if keyword == "DATES":
            for record in _records_after(schedule, match, None):
                next_date = _eclipse_date(record, "DATES")
                if next_date < current_date:
                    raise OpmChddError("SCHEDULE DATES move backwards")
                current_date = next_date
            continue
        if keyword == "TSTEP":
            for token in _expanded_record(
                _records_after(schedule, match, 1)[0], "TSTEP"
            ):
                days = _number(token, "TSTEP")
                if days <= 0 or not days.is_integer():
                    raise OpmChddError("TSTEP must contain positive whole days")
                current_date += timedelta(days=int(days))
            continue
        for record in _records_after(schedule, match, None):
            well, control = _scheduled_control(record, keyword)
            if well in events:
                events[well].append((current_date, control))

    result: dict[str, list[ScheduledControl]] = {}
    for well in wells:
        scheduled = events[well]
        selected: list[ScheduledControl] = []
        for report_date, by_well, _ in summary:
            effective = [control for event_date, control in scheduled if event_date <= report_date]
            if not effective:
                if not scheduled:
                    raise OpmChddError(f"well {well!r} has no requested control")
                realized = by_well[well]
                if any(
                    realized[vector] != 0
                    for vector in ("WLPR", "WLPT", "WOPR", "WOPT", "WWIR", "WWIT")
                ):
                    raise OpmChddError(
                        f"well {well!r} has production/injection before its first "
                        f"requested control on {report_date}"
                    )
                # Only role/type is known before first WCON. Never leak its future
                # numeric target or status into an earlier Track 2 action.
                control = ScheduledControl(scheduled[0][1].target, 0.0, 0)
            else:
                control = effective[-1]
            selected.append(control)
        # action[t] drives state[t+1]. A WCON record following DATES at report
        # date t therefore must be checked against the next report, never the
        # already-realized same-date state.
        for (_, by_well, _), control in zip(
            summary[1:], selected[:-1], strict=True
        ):
            realized = by_well[well]
            producer = control.target in {"ORAT", "LRAT"}
            if realized["WLPR"] > 0 and (not producer or not control.status):
                raise OpmChddError(f"well {well!r} produces against prior action")
            if realized["WWIR"] > 0 and (producer or not control.status):
                raise OpmChddError(f"well {well!r} injects against prior action")
        result[well] = selected
    return result


def export_opm_chdd(
    summary_csv: str | Path,
    chdd_output: str | Path,
    trajectory_output: str | Path,
    manifest_output: str | Path,
    *,
    scenario_id: str,
    source_model: str,
    opm_run_manifest: str | Path,
    summary_extraction_manifest: str | Path,
    deck_dir: str | Path | None = None,
    density_map: str | Path | None = None,
    unit_system: str | None = None,
    _summary_run: Any = None,
) -> dict[str, Any]:
    """Validate, convert and write both canonical CSV contracts plus manifest."""

    if not scenario_id.strip() or not source_model.strip():
        raise OpmChddError("scenario_id and source_model are required")
    run_manifest = Path(opm_run_manifest).resolve()
    if not run_manifest.is_file() or run_manifest.is_symlink():
        raise OpmChddError(f"OPM run manifest is not a regular file: {run_manifest}")
    extraction_manifest = Path(summary_extraction_manifest).resolve()
    try:
        verify_summary_extraction(
            summary_csv,
            extraction_manifest,
            run_manifest,
            _summary_run=_summary_run,
        )
    except OpmSummaryError as exc:
        raise OpmChddError(f"unverified OPM summary extraction: {exc}") from exc
    raw_opm_manifest, opm_source_sha256 = _opm_manifest(run_manifest)
    raw_cofr_transform = (
        raw_opm_manifest.get("summary_contract", {})
        .get("vectors", {})
        .get("COFR", {})
        .get("transform")
    )
    raw_manifest_caveats = []
    if isinstance(raw_cofr_transform, str) and "COPR-COIR" in raw_cofr_transform:
        raw_manifest_caveats.append(
            "raw COFR transform text predates the executable vector contract: COIR was "
            "not requested; certification derives COIR=COPR-COFR per connection and "
            "does not assert a cross-level connection-to-well volume identity"
        )
    if deck_dir is None:
        raise OpmChddError("deck_dir is required for requested-control provenance")
    requested_unit = unit_system.upper() if unit_system else None
    if requested_unit is not None and requested_unit not in {"METRIC", "FIELD"}:
        raise OpmChddError("unit_system must be METRIC or FIELD")

    densities: dict[str, SurfaceDensity] = {}
    density_provenance: dict[str, str] = {}
    deck_connections: dict[str, dict[str, SurfaceDensity]] = {}
    deck_digest: str | None = None
    deck_start: date | None = None
    deck_text: str | None = None
    ambiguous: dict[str, tuple[int, ...]] = {}
    if deck_dir is not None:
        (
            deck_unit,
            deck_densities,
            ambiguous,
            deck_digest,
            deck_connections,
            deck_start,
            deck_text,
        ) = _read_deck_densities_and_start(deck_dir)
        if requested_unit is not None and requested_unit != deck_unit:
            raise OpmChddError(
                f"explicit unit_system {requested_unit} disagrees with deck {deck_unit}"
            )
        requested_unit = deck_unit
        densities.update(deck_densities)
        density_provenance.update(
            {well: "deck:DENSITY/PVTNUM/ACTNUM/WELSPECS/COMPDAT" for well in deck_densities}
        )

    source = Path(summary_csv)
    summary, ignored_columns = _read_summary(source, start_date=deck_start)
    wells = sorted(summary[0][1])

    mapping_digest: str | None = None
    if density_map is not None:
        mapping_path = Path(density_map)
        explicit = load_density_mapping(mapping_path)
        densities.update(explicit)
        density_provenance.update({well: "explicit_density_mapping" for well in explicit})
        mapping_digest = _sha256_file(mapping_path)
    if requested_unit is None:
        raise OpmChddError("unit_system is required when no deck is provided")

    summary_connections = summary[0][2]
    connection_wells: set[str] = set()
    for well, connection_values in summary_connections.items():
        if deck_dir is None:
            raise OpmChddError("connection SUMMARY vectors require deck_dir")
        actual = set(connection_values)
        expected = set(deck_connections.get(well, {}))
        if actual != expected:
            raise OpmChddError(
                f"connection numbers disagree with deck for {well!r}: "
                f"summary={sorted(actual)}, deck={sorted(expected)}"
            )
        connection_wells.add(well)

    phase_vectors = (
        "WLPR",
        "WLPT",
        "WOPR",
        "WOPT",
        "WOIR",
        "WOIT",
        "WWPR",
        "WWPT",
        "WWIR",
        "WWIT",
    )
    zero_phase_wells = {
        well
        for well in wells
        if all(
            by_well[well][vector] == 0
            for _, by_well, _ in summary
            for vector in phase_vectors
        )
    }
    missing_density = [
        well
        for well in wells
        if well not in connection_wells
        and well not in densities
        and well not in zero_phase_wells
    ]
    if missing_density:
        details = {
            well: list(ambiguous[well])
            for well in missing_density
            if well in ambiguous
        }
        suffix = f"; ambiguous PVT regions: {details}" if details else ""
        raise OpmChddError(f"missing explicit density for wells: {missing_density}{suffix}")

    volume_factor = 1.0 if requested_unit == "METRIC" else _STB_TO_M3
    pressure_factor = 1.0 if requested_unit == "METRIC" else _PSI_TO_BAR
    assert deck_text is not None
    controls = _scheduled_controls(deck_text, deck_start, summary, wells)
    chdd_rows: list[dict[str, str | float]] = []
    trajectory_rows: list[dict[str, str | float | int]] = []
    previous = {well: {"WLPT": 0.0, "WOMT": 0.0, "WWIT": 0.0} for well in wells}

    for date_index, (month, by_well, by_connection) in enumerate(summary):
        iso_date = month.replace(day=1).isoformat()
        for well in wells:
            raw = by_well[well]
            wlpr = raw["WLPR"] * volume_factor
            wlpt = raw["WLPT"] * volume_factor
            wopr = raw["WOPR"] * volume_factor
            wopt = raw["WOPT"] * volume_factor
            wwir = raw["WWIR"] * volume_factor
            wwit = raw["WWIT"] * volume_factor
            if well in connection_wells:
                connection_values = by_connection[well]
                for number, values in connection_values.items():
                    _check_connection_total(
                        well, iso_date, f"CWFR:{number}",
                        values["CWFR"], values["CWPR"] - values["CWIR"], 3,
                    )
            if well in densities:
                density = densities[well]
                womr = wopr * density.oil_kg_m3 / 1000
                womt = wopt * density.oil_kg_m3 / 1000
                liquid_tpd = (
                    wopr * density.oil_kg_m3
                    + (wlpr - wopr) * density.water_kg_m3
                ) / 1000
                liquid_total_tonnes = (
                    wopt * density.oil_kg_m3
                    + (wlpt - wopt) * density.water_kg_m3
                ) / 1000
            elif well in connection_wells:
                connection_values = by_connection[well]
                connection_density = deck_connections[well]
                womr = sum(
                    connection_values[number]["COPR"]
                    * volume_factor
                    * density.oil_kg_m3
                    / 1000
                    for number, density in connection_density.items()
                )
                womt = sum(
                    connection_values[number]["COPT"]
                    * volume_factor
                    * density.oil_kg_m3
                    / 1000
                    for number, density in connection_density.items()
                )
                liquid_tpd = womr + sum(
                    connection_values[number]["CWPR"]
                    * volume_factor
                    * density.water_kg_m3
                    / 1000
                    for number, density in connection_density.items()
                )
                liquid_total_tonnes = womt + sum(
                    connection_values[number]["CWPT"]
                    * volume_factor
                    * density.water_kg_m3
                    / 1000
                    for number, density in connection_density.items()
                )
                negative_mass = {
                    name: value
                    for name, value in {
                        "WOMR": womr,
                        "WOMT": womt,
                        "liquid_tpd": liquid_tpd,
                        "WLPT": liquid_total_tonnes,
                    }.items()
                    if value < -1e-9
                }
                if negative_mass:
                    raise OpmChddError(
                        f"negative density-weighted production mass for {well!r} "
                        f"{iso_date}: {negative_mass}"
                    )
            else:
                womr = womt = liquid_tpd = liquid_total_tonnes = 0.0
            cumulative = {
                "WLPT": liquid_total_tonnes,
                "WOMT": womt,
                "WWIT": wwit,
            }
            diffs = {key: value - previous[well][key] for key, value in cumulative.items()}
            negative = {key: value for key, value in diffs.items() if value < 0}
            if negative:
                raise OpmChddError(
                    f"negative cumulative difference for {well!r} {iso_date}: {negative}"
                )
            previous[well] = cumulative
            chdd_rows.append(
                {
                    "DATA": iso_date,
                    "well": well,
                    "WLPT": _clean_zero(liquid_total_tonnes),
                    "WLPR": _clean_zero(wlpr),
                    "WOMT": _clean_zero(womt),
                    "WOMR": _clean_zero(womr),
                    "WWIR": _clean_zero(wwir),
                    "WWIT": _clean_zero(wwit),
                    "THP": _clean_zero(raw["WBP9"] * pressure_factor),
                    "BHP": _clean_zero(raw["WBHP"] * pressure_factor),
                    "WEFF": raw["WEFF"],
                    "WLPT_Diff": _clean_zero(diffs["WLPT"]),
                    "WOMT_Diff": _clean_zero(diffs["WOMT"]),
                    "WWIT_Diff": _clean_zero(diffs["WWIT"]),
                }
            )
            control = controls[well][date_index]
            trajectory_rows.append(
                {
                    "scenario_id": scenario_id,
                    "source_model": source_model,
                    "date": iso_date,
                    "well": well,
                    "oil_tpd": _clean_zero(womr),
                    "liquid_tpd": _clean_zero(liquid_tpd),
                    "pressure_bar": _clean_zero(raw["WBP9"] * pressure_factor),
                    "control_value": _clean_zero(control.value * volume_factor),
                    "control_target": control.target,
                    "status": control.status,
                }
            )

    chdd_bytes = _csv_bytes(CHDD_FIELDS, chdd_rows)
    trajectory_bytes = _csv_bytes(TRACK2_FIELDS, trajectory_rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generator": "timesoil.aios.opm_chdd",
        "provenance": {
            "opm_run_manifest": Path(
                os.path.relpath(run_manifest, Path(manifest_output).resolve().parent)
            ).as_posix(),
            "opm_run_manifest_sha256": _sha256_file(run_manifest),
            "opm_source_sha256": opm_source_sha256,
            "raw_opm_manifest_caveats": raw_manifest_caveats,
            "summary_extraction_manifest": Path(
                os.path.relpath(
                    extraction_manifest, Path(manifest_output).resolve().parent
                )
            ).as_posix(),
            "summary_extraction_manifest_sha256": _sha256_file(
                extraction_manifest
            ),
        },
        "contracts": {
            "chdd_fields": list(CHDD_FIELDS),
            "track2_fields": list(TRACK2_FIELDS),
        },
        "source": {
            "summary_csv": Path(
                os.path.relpath(source.resolve(), Path(manifest_output).resolve().parent)
            ).as_posix(),
            "summary_csv_sha256": _sha256_file(source),
            "deck_sha256": deck_digest,
            "density_mapping_sha256": mapping_digest,
            "unit_system": requested_unit,
            "ignored_summary_columns": ignored_columns,
            "connection_vectors": list(CONNECTION_VECTORS) if connection_wells else [],
        },
        "scenario": {"scenario_id": scenario_id, "source_model": source_model},
        "conversion": {
            "dates": "OPM report date normalized to YYYY-MM-01",
            "volume": f"{requested_unit} surface volume converted to m3",
            "pressure": f"{requested_unit} pressure converted to bar",
            "WOMR": "single-PVT well: WOPR m3/day * its unambiguous surface oil density; multi-PVT well: sum(signed COPR producer-mode connection m3/day * cell oil density) / 1000",
            "WOMT": "single-PVT well: WOPT m3 * its unambiguous surface oil density; multi-PVT well: sum(COPT connection m3 * cell oil density) / 1000",
            "WLPR": "surface liquid volume m3/day; retained for the official 500 m3/day limit",
            "WLPT": "cumulative liquid mass t: single-PVT WOPT/WWPT times unambiguous phase densities; multi-PVT connection COPT/CWPT times cell phase densities",
            "liquid_tpd": "Track 2 liquid mass t/day: single-PVT WOPR/WWPR times unambiguous phase densities; multi-PVT signed COPR/CWPR producer-mode connection volumes times cell phase densities",
            "connection_validation": "SUMMARY must contain every active-deck I,J,K completion and every required vector exactly once; each connection enforces CWFR=CWPR-CWIR with rel_tol=0 and the six-decimal text bound",
            "connection_source_semantics": "pinned opm-common release/2026.04/final Summary.cpp: COPR/CWPR are signed producer-mode crate values, COPT/CWPT their duration integrals, COIT/CWIR/CWIT injector-mode values, COFR=COPR-COIR, and CWFR=CWPR-CWIR; well rate and connection crate are distinct evaluator quantities, so their cross-level surface-volume sums are not asserted",
            "connection_sign": "OPM completion vectors retain signed crossflow; COIR is derived as COPR-COFR because pinned OPM rejects COIR in the SUMMARY deck",
            "well_phase_identity_tolerance": "max(three-term six-decimal print bound, 2^-22 * max(1, abs(WWP*), abs(WLP*), abs(WOP*))) for independently stored float32 SUMMARY quantities",
            "diffs": "first cumulative minus zero; later current minus previous month",
            "THP": "WBP9 nine-point block pressure; not wellhead pressure",
            "BHP": "WBHP bottom-hole pressure",
            "control": "requested WCONPROD ORAT/LRAT or WCONINJE RATE from expanded SCHEDULE, carried forward by DATES/TSTEP; action at report date t includes WCON following DATES t and drives state t+1; before first WCON only its static role/type is retained with status=0,value=0 (future target/status are not used)",
            "mass_method_by_well": {
                well: (
                    "well_surface_density"
                    if well in densities
                    else "connection_surface_vectors"
                    if well in connection_wells
                    else "zero_phase_no_density"
                )
                for well in wells
            },
            "density_by_well": {
                well: {
                    "oil_kg_m3": densities[well].oil_kg_m3,
                    "water_kg_m3": densities[well].water_kg_m3,
                    "provenance": density_provenance[well],
                }
                for well in wells
                if well in densities
            },
            "connection_density_by_well": {
                well: {
                    "connection_count": len(deck_connections[well]),
                    "oil_kg_m3": sorted(
                        {density.oil_kg_m3 for density in deck_connections[well].values()}
                    ),
                    "water_kg_m3": sorted(
                        {density.water_kg_m3 for density in deck_connections[well].values()}
                    ),
                    "provenance": "SMSPEC NUMS I,J,K; ACTNUM/PVTNUM/DENSITY",
                }
                for well in sorted(connection_wells)
            },
        },
        "outputs": {
            "chdd_csv": {
                "name": Path(chdd_output).name,
                "row_count": len(chdd_rows),
                "sha256": _sha256_bytes(chdd_bytes),
            },
            "track2_csv": {
                "name": Path(trajectory_output).name,
                "row_count": len(trajectory_rows),
                "sha256": _sha256_bytes(trajectory_bytes),
            },
        },
    }
    manifest_bytes = _canonical_json(manifest)
    targets = (
        (Path(chdd_output), chdd_bytes),
        (Path(trajectory_output), trajectory_bytes),
        (Path(manifest_output), manifest_bytes),
    )
    resolved_targets = [path.resolve() for path, _ in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise OpmChddError("output paths must be distinct")
    existing = [str(path) for path, _ in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    for path, _ in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in targets:
        with path.open("xb") as stream:
            stream.write(content)
    return manifest
