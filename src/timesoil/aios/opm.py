"""Fail-closed OPM Flow 2026.04 execution boundary."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from math import isclose
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import time
from typing import TYPE_CHECKING, Any, Iterable, Iterator
from zipfile import BadZipFile, ZipFile, ZipInfo

from .contracts import (
    Case,
    ControlAction,
    ControlTarget,
    Economics,
    State,
    Trajectory,
    WellRole,
    WellState,
    WellStatus,
)

if TYPE_CHECKING:
    from .track1 import GdmResult


OPM_IMAGE_TAG = "openporousmedia/opmreleases:2026.04_amd64"
OPM_IMAGE_DIGEST = "sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9"
OPM_IMAGE = f"{OPM_IMAGE_TAG}@{OPM_IMAGE_DIGEST}"

_LOW_PARSING_WARNING = (
    "LOW_PARSING: --parsing-strictness=low explicitly enabled; unsupported "
    "keywords, if any, are recorded in stdout.log and stderr.log"
)
_MAX_ZIP_MEMBERS = 100_000
_MAX_ZIP_BYTES = 64 * 1024**3
_INCLUDE = re.compile(r"(?is)^\s*INCLUDE\s+(['\"])(.+?)\1\s*/\s*$")
_CONNECTION_VECTORS = (
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
_SUMMARY_VECTORS = (
    "DATE",
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
    *_CONNECTION_VECTORS,
)
OPM_EXPORT_VECTORS = _SUMMARY_VECTORS[1:]
_SUMMARY_ARTIFACT_SUFFIXES = {
    ".SMSPEC",
    ".FSMSPEC",
    ".ESMRY",
    ".UNSMRY",
    ".FUNSMRY",
}
_SUMMARY_PRIMARY_SUFFIXES = {".SMSPEC", ".FSMSPEC", ".ESMRY"}
_SUMMARY_EXTRACTION_SCHEMA = "timesoil.aios.opm-summary-extraction/v1"
_SUMMARY_REPLAY_TIMEOUT_SECONDS = 900.0
_SUMMARY_QUANTITIES = {
    "DATE": "calendar date",
    "WLPR": "well surface liquid production rate",
    "WLPT": "well cumulative surface liquid production",
    "WOPR": "well surface oil production rate",
    "WOPT": "well cumulative surface oil production",
    "WWIR": "well surface water injection rate",
    "WWIT": "well cumulative surface water injection",
    "WBHP": "well bottom-hole pressure",
    "WBP9": "well nine-point averaged block pressure",
    "WEFF": "well efficiency factor",
    "WOIR": "well surface oil injection rate",
    "WOIT": "well cumulative surface oil injection",
    "WWPR": "well surface water production rate",
    "WWPT": "well cumulative surface water production",
    "COFR": "connection signed net surface oil flow rate",
    "COPT": "connection cumulative surface oil production",
    "CWFR": "connection signed net surface water flow rate",
    "COPR": "connection surface oil production rate",
    "CWPR": "connection surface water production rate",
    "CWPT": "connection cumulative surface water production",
    "COIT": "connection cumulative surface oil injection",
    "CWIR": "connection surface water injection rate",
    "CWIT": "connection cumulative surface water injection",
}
_CHDD_MAPPING = {
    "DATE": ("DATA", "direct date normalization"),
    "WLPR": ("WLPR", "direct after unit verification"),
    "WLPT": (
        "WLPT",
        "requires oil/water surface volumes and densities for liquid mass",
    ),
    "WOPR": ("WOMR", "requires proven oil volume-to-mass conversion"),
    "WOPT": ("WOMT", "requires proven oil volume-to-mass conversion"),
    "WOIR": ("", "validates signed connection oil flow; not a CHDD output"),
    "WOIT": ("", "validates signed connection oil total; not a CHDD output"),
    "WWPR": ("WLPR", "validates WLPR-WOPR and production-only connections"),
    "WWPT": ("WLPT", "validates WLPT-WOPT and production-only connections"),
    "WWIR": ("WWIR", "direct after unit verification"),
    "WWIT": ("WWIT", "direct after unit verification"),
    "WBHP": ("BHP", "direct after pressure-unit verification"),
    "WBP9": ("THP", "candidate reservoir-pressure mapping; not certified"),
    "WEFF": ("WEFF", "direct dimensionless well efficiency factor"),
    "COFR": (
        "",
        "signed-net audit only: derive COIR=COPR-COFR and reconcile with WOIR",
    ),
    "COPT": (
        "WOMT",
        "aggregate connections per well, then convert oil volume to mass",
    ),
    "CWFR": (
        "",
        "signed-net audit only: reconcile CWPR-CWIR with WWPR-WWIR",
    ),
    "CWPT": (
        "WLPT",
        "aggregate with COPT per well, then convert phase volumes to liquid mass",
    ),
    "COPR": ("WOMR", "signed completion production terms converted with cell oil density"),
    "CWPR": ("WLPR", "signed completion production terms converted with cell water density"),
    "COIT": ("", "connection cumulative oil-injection reconciliation only"),
    "CWIR": ("WWIR", "connection water-injection reconciliation only"),
    "CWIT": ("WWIT", "connection cumulative water-injection reconciliation only"),
}
_UNIT_SYSTEMS = {
    "METRIC": {"rate": "SM3/DAY", "total": "SM3", "pressure": "BARSA"},
    "FIELD": {"rate": "STB/DAY", "total": "STB", "pressure": "PSIA"},
}


class OpmError(RuntimeError):
    """OPM preparation or execution failed."""


class OpmTimeoutError(OpmError):
    """OPM Flow exceeded its configured wall-clock timeout."""


class OpmSummaryError(OpmError):
    """The OPM summary utility failed or returned invalid output."""


class OpmCertificationError(OpmError):
    """A baseline run cannot satisfy the Track 1 certification contract."""


@dataclass(frozen=True, slots=True)
class DeckTransformation:
    name: str
    start_line: int
    end_line: int
    removed_sha256: str
    deck_sha256_before: str
    deck_sha256_after: str


@dataclass(frozen=True, slots=True)
class PreparedCase:
    source: Path
    source_sha256: str
    run_dir: Path
    input_dir: Path
    output_dir: Path
    deck_path: Path
    summary_overlay_path: Path
    connection_wells: tuple[str, ...]
    unit_system: str
    transformations: tuple[DeckTransformation, ...]


@dataclass(frozen=True, slots=True)
class OpmRunResult:
    run_dir: Path
    output_dir: Path
    deck_path: Path
    summary_overlay_path: Path
    stdout_path: Path
    stderr_path: Path
    manifest_path: Path
    manifest_sha256: str
    command: tuple[str, ...]
    warnings: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise OpmError(f"symbolic links are not allowed in a case: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise OpmError(f"special files are not allowed in a case: {path}")
    return files


def _source_digest(source: Path) -> str:
    if source.is_file() and not source.is_symlink():
        return _sha256_file(source)
    if source.is_dir() and not source.is_symlink():
        entries = [
            (path.relative_to(source).as_posix(), path.stat().st_size, _sha256_file(path))
            for path in _regular_files(source)
        ]
        return sha256(
            json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    raise OpmError(f"case source is not a regular file or directory: {source}")


def _lower_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OpmSummaryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OpmSummaryError(f"duplicate JSON key in provenance: {key!r}")
        result[key] = value
    return result


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpmSummaryError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise OpmSummaryError(f"{label} must be a JSON object")
    return value


def _linked_run_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise OpmSummaryError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise OpmSummaryError(f"{label} path is unsafe")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise OpmSummaryError(f"{label} escapes the OPM run directory") from None
    if not path.is_file() or path.is_symlink():
        raise OpmSummaryError(f"{label} is not a regular file: {path}")
    return path


def _validated_summary_artifacts(
    run_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, str | int]]]:
    run_manifest = _json_object(run_manifest_path, "OPM run manifest")
    if (
        run_manifest.get("schema") != "timesoil.aios.opm-run/v1"
        or run_manifest.get("status") != "success"
        or run_manifest.get("returncode") != 0
        or isinstance(run_manifest.get("returncode"), bool)
        or run_manifest.get("image_reference") != OPM_IMAGE
        or run_manifest.get("image_digest") != OPM_IMAGE_DIGEST
    ):
        raise OpmSummaryError("OPM run is not a successful pinned-image run")
    _lower_digest(run_manifest.get("source_sha256"), "OPM source")
    artifacts = run_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise OpmSummaryError("OPM run artifact list is missing")

    records: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise OpmSummaryError("OPM run artifact entry is invalid")
        relative = item["path"]
        if relative in seen:
            raise OpmSummaryError(f"duplicate OPM run artifact: {relative}")
        seen.add(relative)
        if Path(relative).suffix.upper() not in _SUMMARY_ARTIFACT_SUFFIXES:
            continue
        digest = _lower_digest(item.get("sha256"), f"OPM artifact {relative}")
        size = item.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise OpmSummaryError(f"OPM artifact {relative} has invalid byte count")
        path = _linked_run_file(run_manifest_path.parent, relative, "OPM summary artifact")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise OpmSummaryError(f"OPM summary artifact hash mismatch: {relative}")
        records.append({"path": relative, "bytes": size, "sha256": digest})
    records.sort(key=lambda item: str(item["path"]))
    primaries = [
        item
        for item in records
        if Path(str(item["path"])).suffix.upper() in _SUMMARY_PRIMARY_SUFFIXES
    ]
    if not primaries:
        raise OpmSummaryError("OPM run contains no raw SMSPEC, FSMSPEC, or ESMRY artifact")
    paths = {str(item["path"]) for item in records}
    for item in primaries:
        path = Path(str(item["path"]))
        companion = {
            ".SMSPEC": path.with_suffix(".UNSMRY").as_posix(),
            ".FSMSPEC": path.with_suffix(".FUNSMRY").as_posix(),
        }.get(path.suffix.upper())
        if companion is not None and companion not in paths:
            raise OpmSummaryError(f"OPM summary artifact misses companion: {companion}")
    return run_manifest, records


def _canonical_summary_selection(available: Iterable[str]) -> tuple[str, ...]:
    vectors = tuple(available)
    if not vectors or any(
        not vector or any(character.isspace() for character in vector)
        for vector in vectors
    ):
        raise OpmSummaryError("summary -l returned invalid vectors")
    temporal = {"DATE", "TIME", "YEARS"}
    required: set[str] = set(OPM_EXPORT_VECTORS)
    selected = tuple(
        vector
        for vector in vectors
        if vector.upper() in temporal or vector.split(":", 1)[0].upper() in required
    )
    if len(set(selected)) != len(selected):
        raise OpmSummaryError("summary -l returned duplicate canonical vectors")
    selected_prefixes = {vector.split(":", 1)[0].upper() for vector in selected}
    if not ({"DATE", "TIME"} & selected_prefixes):
        raise OpmSummaryError("summary vector list contains neither DATE nor TIME")
    missing = sorted(required - selected_prefixes)
    if missing:
        raise OpmSummaryError(f"summary vector list misses canonical vectors: {missing}")
    return selected


def verify_summary_extraction(
    summary_report: str | Path,
    extraction_manifest: str | Path,
    opm_run_manifest: str | Path,
    *,
    _summary_run: Any = None,
    _docker_executable: str = "docker",
) -> dict[str, Any]:
    """Verify and replay a derivation from raw OPM summary artifacts."""

    report_path = Path(summary_report).resolve()
    extraction_path = Path(extraction_manifest).resolve()
    run_manifest_path = Path(opm_run_manifest).resolve()
    for path, label in (
        (report_path, "summary report"),
        (extraction_path, "summary extraction manifest"),
        (run_manifest_path, "OPM run manifest"),
    ):
        if not path.is_file() or path.is_symlink():
            raise OpmSummaryError(f"{label} is not a regular file: {path}")

    run_manifest, raw_records = _validated_summary_artifacts(run_manifest_path)
    extraction = _json_object(extraction_path, "summary extraction manifest")
    if extraction.get("schema") != _SUMMARY_EXTRACTION_SCHEMA:
        raise OpmSummaryError("summary extraction manifest has an unsupported schema")
    run_ref = extraction.get("run_manifest")
    image = extraction.get("image")
    output = extraction.get("output_report")
    if not all(isinstance(item, dict) for item in (run_ref, image, output)):
        raise OpmSummaryError("summary extraction manifest is incomplete")
    assert isinstance(run_ref, dict) and isinstance(image, dict) and isinstance(output, dict)
    linked_run = _linked_run_file(
        extraction_path.parent, run_ref.get("path"), "linked OPM run manifest"
    )
    if linked_run != run_manifest_path or _sha256_file(linked_run) != _lower_digest(
        run_ref.get("sha256"), "linked OPM run manifest"
    ):
        raise OpmSummaryError("summary extraction links a different OPM run manifest")
    if image != {"reference": OPM_IMAGE, "digest": OPM_IMAGE_DIGEST}:
        raise OpmSummaryError("summary extraction did not use the pinned OPM image")
    if extraction.get("raw_summary_artifacts") != raw_records:
        raise OpmSummaryError("summary extraction raw artifact chain is incomplete")
    selection = extraction.get("vector_selection")
    commands = extraction.get("commands")
    if not isinstance(selection, dict) or not isinstance(commands, dict):
        raise OpmSummaryError("summary extraction vector selection is missing")
    available = selection.get("available")
    selected = selection.get("selected")
    if (
        selection.get("mode") != "filtered-summary-list"
        or selection.get("required") != list(OPM_EXPORT_VECTORS)
        or not isinstance(available, list)
        or any(not isinstance(item, str) for item in available)
        or not isinstance(selected, list)
        or any(not isinstance(item, str) for item in selected)
        or selection.get("available_sha256")
        != sha256(
            json.dumps(available, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        or tuple(selected) != _canonical_summary_selection(available)
    ):
        raise OpmSummaryError("summary extraction vector selection is not canonical")
    if extraction.get("report_steps_only") is not True or extraction.get("shell") is not False:
        raise OpmSummaryError("summary extraction execution contract is unsafe")

    summary_input = extraction.get("summary_input")
    raw_paths = {str(item["path"]) for item in raw_records}
    if not isinstance(summary_input, str) or summary_input not in raw_paths:
        raise OpmSummaryError("summary extraction input is not a raw run artifact")
    input_path = Path(summary_input)
    if input_path.suffix.upper() not in _SUMMARY_PRIMARY_SUFFIXES:
        raise OpmSummaryError("summary extraction input is not SMSPEC, FSMSPEC, or ESMRY")
    try:
        container_input = "/output/" + input_path.relative_to("output").as_posix()
    except ValueError:
        raise OpmSummaryError("summary extraction input is outside run output") from None

    expected_mount = (
        f"type=bind,src={(run_manifest_path.parent / 'output').resolve()},"
        "dst=/output,readonly"
    )
    for name, option, vectors in (
        ("list", "-l", []),
        ("report", "-r", selected),
    ):
        command = commands.get(name)
        expected_tail = [OPM_IMAGE, "summary", option, container_input, *vectors]
        if (
            not isinstance(command, list)
            or any(not isinstance(item, str) for item in command)
            or len(command) != 8 + len(expected_tail)
            or command[0] != _docker_executable
            or command[1:5] != ["run", "--rm", "--network=none", "--user"]
            or command[5] != f"{os.getuid()}:{os.getgid()}"
            or command[6:8] != ["--mount", expected_mount]
            or command[8:] != expected_tail
        ):
            raise OpmSummaryError(
                f"summary extraction {name} command is not the canonical pinned command"
            )

    linked_report = _linked_run_file(
        extraction_path.parent, output.get("path"), "linked summary report"
    )
    size = output.get("bytes")
    if (
        linked_report != report_path
        or isinstance(size, bool)
        or not isinstance(size, int)
        or linked_report.stat().st_size != size
        or _sha256_file(linked_report)
        != _lower_digest(output.get("sha256"), "summary report")
    ):
        raise OpmSummaryError("summary report does not match its extraction manifest")

    replay = subprocess.run if _summary_run is None else _summary_run
    try:
        completed = replay(
            commands["report"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_SUMMARY_REPLAY_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OpmSummaryError("pinned OPM summary replay timed out") from exc
    except OSError as exc:
        raise OpmSummaryError("cannot execute pinned OPM summary replay") from exc
    if not isinstance(completed, subprocess.CompletedProcess):
        raise OpmSummaryError("pinned OPM summary replay returned an invalid result")
    if completed.returncode != 0:
        stderr = completed.stderr if isinstance(completed.stderr, bytes) else b""
        detail = stderr.decode("utf-8", "replace").strip()[-2000:]
        raise OpmSummaryError(
            f"pinned OPM summary replay failed with exit code {completed.returncode}: {detail}"
        )
    if not isinstance(completed.stdout, bytes) or completed.stdout != report_path.read_bytes():
        raise OpmSummaryError("summary report differs from pinned OPM deterministic replay")
    return run_manifest


def _safe_zip_members(archive: ZipFile, destination: Path) -> list[tuple[ZipInfo, Path]]:
    infos = archive.infolist()
    if len(infos) > _MAX_ZIP_MEMBERS:
        raise OpmError("ZIP contains too many members")
    if sum(info.file_size for info in infos) > _MAX_ZIP_BYTES:
        raise OpmError("ZIP expands beyond the configured safety limit")

    root = destination.resolve()
    selected: list[tuple[ZipInfo, Path]] = []
    seen: set[Path] = set()
    for info in infos:
        name = info.filename
        parts = PurePosixPath(name).parts
        mode = info.external_attr >> 16
        if (
            not name
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or stat.S_ISLNK(mode)
            or info.flag_bits & 0x1
        ):
            raise OpmError(f"unsafe ZIP member: {name!r}")
        target = (destination / Path(*parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise OpmError(f"ZIP member escapes destination: {name!r}") from None
        if target in seen:
            raise OpmError(f"duplicate ZIP member: {name!r}")
        seen.add(target)
        selected.append((info, target))
    return selected


def _extract_zip(source: Path, destination: Path) -> None:
    try:
        with ZipFile(source) as archive:
            members = _safe_zip_members(archive, destination)
            for info, target in members:
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    except (BadZipFile, OSError) as exc:
        raise OpmError(f"cannot prepare ZIP case {source}") from exc


def _copy_case_tree(source: Path, destination: Path) -> None:
    destination.mkdir()
    for path in _regular_files(source):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _relative_deck(input_dir: Path, deck: str | Path | None) -> Path:
    if deck is not None:
        relative = Path(deck)
        if relative.is_absolute() or ".." in relative.parts:
            raise OpmError(f"unsafe deck path: {deck!r}")
        candidate = (input_dir / relative).resolve()
        try:
            candidate.relative_to(input_dir.resolve())
        except ValueError:
            raise OpmError(f"deck escapes prepared input: {deck!r}") from None
        choices = [candidate]
    else:
        choices = [path for path in _regular_files(input_dir) if path.suffix.upper() == ".DATA"]
    if len(choices) != 1 or not choices[0].is_file():
        names = ", ".join(path.relative_to(input_dir).as_posix() for path in choices)
        raise OpmError(f"expected exactly one OPM .DATA deck, found: {names or 'none'}")
    if choices[0].suffix.upper() != ".DATA":
        raise OpmError("OPM deck must use the .DATA extension")
    return choices[0]


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _container_name(run_dir: Path) -> str:
    digest = sha256(os.fsencode(run_dir)).hexdigest()[:20]
    return f"timesoil-aios-{digest}"


def _unit_system(deck: bytes) -> str:
    found = [
        system
        for system in ("METRIC", "FIELD", "LAB", "PVT-M")
        if re.search(rb"(?mi)^\s*" + re.escape(system.encode()) + rb"\s*(?:--.*)?$", deck)
    ]
    return found[0] if len(found) == 1 else "UNKNOWN"


def _summary_mapping(unit_system: str) -> dict[str, dict[str, str]]:
    units = _UNIT_SYSTEMS.get(unit_system)
    mapping: dict[str, dict[str, str]] = {}
    for vector in _SUMMARY_VECTORS:
        if vector == "DATE":
            unit = "calendar date"
        elif vector == "WEFF":
            unit = "dimensionless"
        elif vector.endswith("R"):
            unit = units["rate"] if units else "SMSPEC-declared rate unit (unverified)"
        elif vector.endswith("T"):
            unit = units["total"] if units else "SMSPEC-declared total unit (unverified)"
        else:
            unit = units["pressure"] if units else "SMSPEC-declared pressure unit (unverified)"
        chdd_field, transform = _CHDD_MAPPING[vector]
        mapping[vector] = {
            "quantity": _SUMMARY_QUANTITIES[vector],
            "unit": unit,
            "chdd_field": chdd_field,
            "transform": transform,
        }
    return mapping


def _strip_comment(line: str) -> str:
    quote: str | None = None
    for index, character in enumerate(line):
        if character in "'\"":
            quote = None if quote == character else character if quote is None else quote
        elif quote is None and line[index : index + 2] == "--":
            return line[:index]
    return line


def _slash_outside_quotes(value: str) -> bool:
    quote: str | None = None
    for character in value:
        if character in "'\"":
            quote = None if quote == character else character if quote is None else quote
        elif character == "/" and quote is None:
            return True
    return False


def _expanded_deck_text(deck_path: Path, input_dir: Path) -> str:
    root = input_dir.resolve()

    def expand(path: Path, stack: tuple[Path, ...]) -> str:
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise OpmError(f"INCLUDE escapes prepared input: {path}") from None
        if path in stack:
            raise OpmError(f"cyclic INCLUDE: {path.relative_to(root)}")
        if not path.is_file() or path.is_symlink():
            raise OpmError(f"INCLUDE is not a regular file: {path}")
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeError as exc:
            raise OpmError(f"deck file is not UTF-8: {path}") from exc
        except OSError as exc:
            raise OpmError(f"cannot read deck file: {path}") from exc

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
                    raise OpmError(f"unterminated INCLUDE in {path}")
                include += "\n" + _strip_comment(lines[index])
            match = _INCLUDE.fullmatch(include)
            if match is None:
                raise OpmError(f"unsupported INCLUDE syntax in {path}: {include!r}")
            include_name = match.group(2)
            relative = Path(include_name)
            if (
                not include_name
                or "\\" in include_name
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise OpmError(f"unsafe INCLUDE path: {include_name!r}")
            output.append(expand(path.parent / relative, (*stack, path)))
            index += 1
        return "\n".join(output)

    return expand(deck_path, ())


def _tokens(value: str) -> Iterator[str]:
    lexer = shlex.shlex(value, posix=True, punctuation_chars="/")
    lexer.whitespace_split = True
    lexer.commenters = ""
    yield from lexer


def _connection_wells(deck_path: Path, input_dir: Path) -> tuple[str, ...]:
    text = _expanded_deck_text(deck_path, input_dir)
    names: set[str] = set()
    for match in re.finditer(r"(?mi)^\s*WELSPECS\b", text):
        record: list[str] = []
        terminated = False
        try:
            for token in _tokens(text[match.end() :]):
                if token != "/":
                    record.append(token)
                    continue
                if not record:
                    terminated = True
                    break
                name = record[0]
                if "'" in name or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in name
                ):
                    raise OpmError(f"unsafe WELSPECS well name: {name!r}")
                names.add(name)
                record = []
        except ValueError as exc:
            raise OpmError("invalid WELSPECS quoting") from exc
        if not terminated:
            raise OpmError("unterminated WELSPECS keyword")
    if not names:
        raise OpmError("deck must define at least one WELSPECS well for connection vectors")
    return tuple(sorted(names))


def build_summary_overlay(connection_wells: Iterable[str] = ()) -> str:
    """Return all-well and explicit per-well OPM SUMMARY requests."""
    wells = tuple(sorted(set(connection_wells)))
    if not wells:
        raise OpmError("connection_wells must contain at least one WELSPECS well")
    if any(
        not well
        or "'" in well
        or any(ord(character) < 32 or ord(character) == 127 for character in well)
        for well in wells
    ):
        raise OpmError("connection_wells contains an unsafe well name")
    lines = ["-- TIMESOIL AIOS SUMMARY OVERLAY; GENERATED IN RUN SNAPSHOT", "DATE"]
    for vector in _SUMMARY_VECTORS[1 : -len(_CONNECTION_VECTORS)]:
        lines.extend((vector, "/"))
    for vector in _CONNECTION_VECTORS:
        lines.append(vector)
        lines.extend(f" '{well}' /" for well in wells)
        lines.append("/")
    return "\n".join(lines) + "\n"


def _install_summary_overlay(
    deck_path: Path, input_dir: Path
) -> tuple[Path, tuple[str, ...], str]:
    deck = deck_path.read_bytes()
    matches = list(re.finditer(rb"(?mi)^\s*SUMMARY\s*(?:--.*)?(?:\r?\n|$)", deck))
    if len(matches) != 1:
        raise OpmError("deck must contain exactly one explicit SUMMARY section")
    overlay_path = deck_path.with_name("_TIMESOIL_SUMMARY.INC")
    if overlay_path.exists():
        raise OpmError(f"summary overlay path already exists: {overlay_path.name}")
    newline = b"\r\n" if b"\r\n" in deck else b"\n"
    include = b"INCLUDE" + newline + b" '_TIMESOIL_SUMMARY.INC' /" + newline
    end = matches[0].end()
    connection_wells = _connection_wells(deck_path, input_dir)
    deck_path.write_bytes(deck[:end] + include + deck[end:])
    overlay_path.write_text(build_summary_overlay(connection_wells), encoding="ascii")
    return overlay_path, connection_wells, _unit_system(deck)


def _sanitize_model_y(deck_path: Path) -> DeckTransformation:
    """Remove the one confirmed tNavigator-disabled block from a run-copy deck."""
    deck = deck_path.read_bytes()
    lines = deck.splitlines(keepends=True)
    marker = re.compile(
        rb"^[ \t]*#if[ \t]+0[ \t]*//[ \t]*tNavigator[ \t]+keyword[ \t]*(?:\r?\n)?$"
    )
    starts = [index for index, line in enumerate(lines) if marker.fullmatch(line)]
    if len(starts) != 1:
        raise OpmError(
            "Model Y normalization requires exactly one '#if 0 // tNavigator keyword' block"
        )
    start = starts[0]
    depth = 0
    end: int | None = None
    directive = re.compile(rb"^[ \t]*#(if|ifdef|ifndef|elif|else|endif)\b")
    for index in range(start, len(lines)):
        match = directive.match(lines[index])
        if match is None:
            continue
        kind = match.group(1)
        if kind in {b"if", b"ifdef", b"ifndef"}:
            depth += 1
        elif kind in {b"elif", b"else"}:
            raise OpmError("Model Y tNavigator block contains an active #else/#elif branch")
        else:
            depth -= 1
            if depth == 0:
                end = index
                break
            if depth < 0:
                break
    if end is None or depth != 0:
        raise OpmError("Model Y tNavigator block is not correctly balanced")

    removed = b"".join(lines[start : end + 1])
    sanitized = b"".join((*lines[:start], *lines[end + 1 :]))
    before = sha256(deck).hexdigest()
    after = sha256(sanitized).hexdigest()
    deck_path.write_bytes(sanitized)
    return DeckTransformation(
        name="remove_model_y_tnavigator_disabled_block",
        start_line=start + 1,
        end_line=end + 1,
        removed_sha256=sha256(removed).hexdigest(),
        deck_sha256_before=before,
        deck_sha256_after=after,
    )


class OpmFlowRunner:
    """Run an immutable case snapshot with digest-pinned OPM Flow."""

    def __init__(self, *, timeout_seconds: float = 3600.0, docker_executable: str = "docker"):
        if not 0 < timeout_seconds <= 7 * 24 * 3600:
            raise ValueError("OPM timeout must be in (0, 604800] seconds")
        if not docker_executable or "\x00" in docker_executable:
            raise ValueError("docker_executable must be non-empty and contain no NUL")
        self.timeout_seconds = timeout_seconds
        self.docker_executable = docker_executable

    def get_provenance(self) -> str:
        return f"OPM Flow 2026.04; image={OPM_IMAGE}"

    def prepare(
        self,
        source: str | Path,
        run_dir: str | Path,
        *,
        deck: str | Path | None = None,
        normalize_model_y: bool = False,
    ) -> PreparedCase:
        original = Path(source).resolve()
        destination = Path(run_dir).resolve()
        if destination.exists():
            raise FileExistsError(f"OPM run directory already exists: {destination}")
        if original.is_dir():
            try:
                destination.relative_to(original)
            except ValueError:
                pass
            else:
                raise OpmError("run directory cannot be inside the source case")

        source_sha = _source_digest(original)
        input_dir = destination / "input"
        output_dir = destination / "output"
        destination.mkdir(parents=True)
        try:
            if original.is_dir():
                _copy_case_tree(original, input_dir)
            elif original.suffix.lower() == ".zip":
                input_dir.mkdir()
                _extract_zip(original, input_dir)
            else:
                input_dir.mkdir()
                if original.suffix.upper() != ".DATA":
                    raise OpmError("case source must be a directory, ZIP, or .DATA deck")
                shutil.copy2(original, input_dir / original.name)
            output_dir.mkdir()
            deck_path = _relative_deck(input_dir, deck)
            transformations = (
                (_sanitize_model_y(deck_path),) if normalize_model_y else ()
            )
            overlay_path, connection_wells, unit_system = _install_summary_overlay(
                deck_path, input_dir
            )
            if _source_digest(original) != source_sha:
                raise OpmError("case source changed while its immutable snapshot was prepared")
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        return PreparedCase(
            original,
            source_sha,
            destination,
            input_dir,
            output_dir,
            deck_path,
            overlay_path,
            connection_wells,
            unit_system,
            transformations,
        )

    def run(
        self,
        source: str | Path,
        run_dir: str | Path,
        *,
        deck: str | Path | None = None,
        parsing_strictness: str = "strict",
        normalize_model_y: bool = False,
    ) -> OpmRunResult:
        if parsing_strictness not in {"strict", "low"}:
            raise ValueError("parsing_strictness must be 'strict' or explicitly 'low'")
        prepared = self.prepare(
            source, run_dir, deck=deck, normalize_model_y=normalize_model_y
        )
        return self._run_prepared(prepared, parsing_strictness=parsing_strictness)

    def _run_prepared(
        self,
        prepared: PreparedCase,
        *,
        parsing_strictness: str = "strict",
    ) -> OpmRunResult:
        """Execute a prepared snapshot after a caller-owned, recorded transform."""

        if parsing_strictness not in {"strict", "low"}:
            raise ValueError("parsing_strictness must be 'strict' or explicitly 'low'")
        relative_deck = prepared.deck_path.relative_to(prepared.input_dir).as_posix()
        warnings = (_LOW_PARSING_WARNING,) if parsing_strictness == "low" else ()
        container_name = _container_name(prepared.run_dir)
        command = [
            self.docker_executable,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network=none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={prepared.input_dir},dst=/case,readonly",
            "--mount",
            f"type=bind,src={prepared.output_dir},dst=/output",
            OPM_IMAGE,
            "flow",
            "--output-dir=/output",
        ]
        if parsing_strictness == "low":
            command.append("--parsing-strictness=low")
        command.append(f"/case/{relative_deck}")

        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        status = "failed"
        returncode: int | None = None
        stdout = stderr = ""
        timeout_cleanup = "not-needed"
        failure: Exception | None = None
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
            returncode = completed.returncode
            status = "success" if returncode == 0 else "failed"
            if returncode:
                failure = OpmError(f"OPM Flow failed with exit code {returncode}")
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _text(exc.stdout), _text(exc.stderr)
            status = "timeout"
            timeout_cleanup = self._cleanup_container(container_name)
            failure = OpmTimeoutError(
                f"OPM Flow exceeded {self.timeout_seconds:g} seconds"
            )
        except OSError as exc:
            stderr = str(exc)
            failure = OpmError(f"cannot execute {self.docker_executable!r}")

        stdout_path = prepared.run_dir / "stdout.log"
        stderr_path = prepared.run_dir / "stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        if _source_digest(prepared.source) != prepared.source_sha256:
            status = "source-changed"
            failure = OpmError("case source changed during OPM execution")

        finished_at = datetime.now(timezone.utc)
        manifest_path, manifest_sha = self._write_manifest(
            prepared,
            command,
            warnings,
            status=status,
            returncode=returncode,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - started,
            container_name=container_name,
            timeout_cleanup=timeout_cleanup,
        )
        if failure is not None:
            raise failure
        return OpmRunResult(
            prepared.run_dir,
            prepared.output_dir,
            prepared.deck_path,
            prepared.summary_overlay_path,
            stdout_path,
            stderr_path,
            manifest_path,
            manifest_sha,
            tuple(command),
            warnings,
        )

    def _cleanup_container(self, container_name: str) -> str:
        try:
            completed = subprocess.run(
                [
                    self.docker_executable,
                    "container",
                    "rm",
                    "--force",
                    container_name,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30.0,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return "cleanup-timeout"
        except OSError:
            return "cleanup-execution-error"
        return (
            "removed"
            if completed.returncode == 0
            else f"cleanup-exit-{completed.returncode}"
        )

    def list_summary_vectors(
        self, result: OpmRunResult, summary_file: str | Path | None = None
    ) -> tuple[str, ...]:
        stdout = self._run_summary(result, ("-l",), (), summary_file)
        vectors = tuple(stdout.split())
        if not vectors:
            raise OpmSummaryError("summary -l returned no vectors")
        return vectors

    def extract_summary(
        self,
        result: OpmRunResult,
        vectors: Iterable[str],
        *,
        summary_file: str | Path | None = None,
        report_steps_only: bool = True,
    ) -> str:
        selected = tuple(vectors)
        if not selected or any(
            not vector.strip() or any(c.isspace() for c in vector)
            for vector in selected
        ):
            raise ValueError("at least one whitespace-free summary vector is required")
        options = ("-r",) if report_steps_only else ()
        return self._run_summary(result, options, selected, summary_file)

    def extract_summary_report(
        self,
        result: OpmRunResult,
        report_path: str | Path,
        *,
        extraction_manifest_path: str | Path | None = None,
        summary_file: str | Path | None = None,
    ) -> tuple[Path, Path]:
        """Persist the canonical report and its raw-artifact derivation chain."""

        run_dir = result.run_dir.resolve()
        report = Path(report_path).resolve()
        extraction = (
            Path(extraction_manifest_path).resolve()
            if extraction_manifest_path is not None
            else run_dir / "summary-extraction.json"
        )
        if report.parent != run_dir or extraction.parent != run_dir or report == extraction:
            raise OpmSummaryError(
                "summary report and extraction manifest must be distinct run-root files"
            )
        if report.exists() or extraction.exists():
            raise FileExistsError("refusing to overwrite summary extraction outputs")
        if result.manifest_path.resolve().parent != run_dir:
            raise OpmSummaryError("OPM result manifest is outside its run directory")
        if _sha256_file(result.manifest_path) != result.manifest_sha256:
            raise OpmSummaryError("OPM result manifest hash changed before summary extraction")

        _, raw_records = _validated_summary_artifacts(result.manifest_path)
        listing, summary_path, list_command = self._run_summary_details(
            result, ("-l",), (), summary_file
        )
        available = tuple(listing.split())
        selected = _canonical_summary_selection(available)
        stdout, report_summary_path, report_command = self._run_summary_details(
            result, ("-r",), selected, summary_file
        )
        if report_summary_path != summary_path:
            raise OpmSummaryError("summary input changed between list and report extraction")
        summary_relative = summary_path.relative_to(run_dir).as_posix()
        if summary_relative not in {str(item["path"]) for item in raw_records}:
            raise OpmSummaryError("selected summary input is absent from the OPM run manifest")
        report_bytes = stdout.encode("utf-8")
        extraction_value = {
            "schema": _SUMMARY_EXTRACTION_SCHEMA,
            "generator": "OpmFlowRunner.extract_summary_report",
            "run_manifest": {
                "path": result.manifest_path.relative_to(run_dir).as_posix(),
                "sha256": result.manifest_sha256,
            },
            "image": {"reference": OPM_IMAGE, "digest": OPM_IMAGE_DIGEST},
            "raw_summary_artifacts": raw_records,
            "summary_input": summary_relative,
            "commands": {"list": list_command, "report": report_command},
            "shell": False,
            "report_steps_only": True,
            "vector_selection": {
                "mode": "filtered-summary-list",
                "available": list(available),
                "available_sha256": sha256(
                    json.dumps(
                        available, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "selected": list(selected),
                "required": list(OPM_EXPORT_VECTORS),
            },
            "output_report": {
                "path": report.relative_to(run_dir).as_posix(),
                "bytes": len(report_bytes),
                "sha256": sha256(report_bytes).hexdigest(),
            },
        }
        extraction_bytes = (
            json.dumps(extraction_value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
        created_report = False
        try:
            with report.open("xb") as stream:
                stream.write(report_bytes)
            created_report = True
            with extraction.open("xb") as stream:
                stream.write(extraction_bytes)
            verify_summary_extraction(
                report,
                extraction,
                result.manifest_path,
                _docker_executable=self.docker_executable,
            )
        except Exception:
            if created_report:
                report.unlink(missing_ok=True)
            extraction.unlink(missing_ok=True)
            raise
        return report, extraction

    def _run_summary(
        self,
        result: OpmRunResult,
        options: tuple[str, ...],
        vectors: tuple[str, ...],
        summary_file: str | Path | None,
    ) -> str:
        stdout, _, _ = self._run_summary_details(result, options, vectors, summary_file)
        return stdout

    def _run_summary_details(
        self,
        result: OpmRunResult,
        options: tuple[str, ...],
        vectors: tuple[str, ...],
        summary_file: str | Path | None,
    ) -> tuple[str, Path, list[str]]:
        summary_path = self._summary_path(result.output_dir, summary_file)
        container_path = f"/output/{summary_path.relative_to(result.output_dir).as_posix()}"
        command = [
            self.docker_executable,
            "run",
            "--rm",
            "--network=none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={result.output_dir},dst=/output,readonly",
            OPM_IMAGE,
            "summary",
            *options,
            container_path,
            *vectors,
        ]
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OpmSummaryError("OPM summary extraction timed out") from exc
        except OSError as exc:
            raise OpmSummaryError("cannot execute OPM summary utility") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2000:]
            raise OpmSummaryError(
                f"OPM summary failed with exit code {completed.returncode}: {detail}"
            )
        if not completed.stdout.strip():
            raise OpmSummaryError("OPM summary returned empty output")
        return completed.stdout, summary_path, command

    @staticmethod
    def _summary_path(output_dir: Path, summary_file: str | Path | None) -> Path:
        if summary_file is not None:
            path = Path(summary_file)
            candidate = path.resolve() if path.is_absolute() else (output_dir / path).resolve()
            try:
                candidate.relative_to(output_dir.resolve())
            except ValueError:
                raise OpmSummaryError("summary file escapes the OPM output directory") from None
            choices = [candidate]
        else:
            choices = sorted(output_dir.rglob("*.SMSPEC"))
            if not choices:
                choices = sorted(output_dir.rglob("*.FSMSPEC"))
            if not choices:
                choices = sorted(output_dir.rglob("*.ESMRY"))
        if len(choices) != 1 or not choices[0].is_file():
            raise OpmSummaryError("expected exactly one SMSPEC, FSMSPEC, or ESMRY file")
        if choices[0].is_symlink():
            raise OpmSummaryError("summary file cannot be a symbolic link")
        return choices[0]

    def _write_manifest(
        self,
        prepared: PreparedCase,
        command: list[str],
        warnings: tuple[str, ...],
        *,
        status: str,
        returncode: int | None,
        started_at: datetime,
        finished_at: datetime,
        duration_seconds: float,
        container_name: str,
        timeout_cleanup: str,
    ) -> tuple[Path, str]:
        excluded = {"manifest.json", "manifest.sha256"}
        artifacts = [
            {
                "path": path.relative_to(prepared.run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in _regular_files(prepared.run_dir)
            if path.relative_to(prepared.run_dir).as_posix() not in excluded
        ]
        manifest = {
            "schema": "timesoil.aios.opm-run/v1",
            "status": status,
            "returncode": returncode,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "container": {
                "name": container_name,
                "timeout_cleanup": timeout_cleanup,
            },
            "source": str(prepared.source),
            "source_sha256": prepared.source_sha256,
            "deck_transformations": [
                {
                    "name": item.name,
                    "opt_in": True,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "removed_sha256": item.removed_sha256,
                    "deck_sha256_before": item.deck_sha256_before,
                    "deck_sha256_after": item.deck_sha256_after,
                }
                for item in prepared.transformations
            ],
            "deck": prepared.deck_path.relative_to(prepared.input_dir).as_posix(),
            "deck_sha256": _sha256_file(prepared.deck_path),
            "summary_contract": {
                "overlay": prepared.summary_overlay_path.relative_to(
                    prepared.input_dir
                ).as_posix(),
                "overlay_sha256": _sha256_file(prepared.summary_overlay_path),
                "unit_system": prepared.unit_system,
                "vectors": _summary_mapping(prepared.unit_system),
                "all_wells": True,
                "well_selector": (
                    "empty '/' record requests all wells for well vectors; connection "
                    "vectors enumerate sorted WELSPECS names explicitly"
                ),
                "connection_wells": list(prepared.connection_wells),
                "postprocessor_certified": False,
                "unsupported_direct_vectors": {
                    "WOMT": "requires a proven volume-to-mass conversion and density source",
                    "WLPT_Diff": "requires a proven cumulative-delta postprocessor",
                    "WOMT_Diff": "requires proven mass conversion before cumulative delta",
                    "WWIT_Diff": "requires a proven cumulative-delta postprocessor",
                },
            },
            "image": OPM_IMAGE_TAG,
            "image_digest": OPM_IMAGE_DIGEST,
            "image_reference": OPM_IMAGE,
            "command": command,
            "warnings": list(warnings),
            "artifacts": artifacts,
            "provenance": self.get_provenance(),
        }
        manifest_path = prepared.run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        digest = _sha256_file(manifest_path)
        (prepared.run_dir / "manifest.sha256").write_text(
            f"{digest}  manifest.json\n", encoding="ascii"
        )
        return manifest_path, digest


class OpmGdmBackend:
    """Certified Track 1 backend using deterministic full replay with OPM Flow."""

    _LINEAGE_SCHEMA = "timesoil.aios.track1-opm-lineage/v1"

    def __init__(
        self,
        runner: OpmFlowRunner,
        source: str | Path,
        *,
        runs_dir: str | Path | None = None,
        deck: str | Path | None = None,
        schedule_include: str | Path | None = None,
        normalize_model_y: bool = False,
        parsing_strictness: str = "strict",
        economics: Any = None,
        density_map: str | Path | None = None,
        source_model: str | None = None,
    ):
        self.runner = runner
        self.source = Path(source).resolve()
        self.runs_dir = Path(runs_dir).absolute() if runs_dir is not None else None
        self.deck = deck
        self.schedule_include = schedule_include
        self.normalize_model_y = normalize_model_y
        self.parsing_strictness = parsing_strictness
        self.economics = economics
        self.density_map = (
            Path(density_map).resolve() if density_map is not None else None
        )
        self.source_model = source_model

    def validate_case(self, case: Case) -> None:
        if not case.case_id.strip():
            raise OpmCertificationError("case_id is required")
        if not case.producers or not case.injectors:
            raise OpmCertificationError("OPM Track 1 requires producers and injectors")
        _source_digest(self.source)
        if self.runs_dir is None or self.schedule_include is None:
            raise OpmCertificationError(
                "OPM Track 1 configuration requires runs_dir and schedule_include"
            )
        if self.parsing_strictness not in {"strict", "low"}:
            raise OpmCertificationError(
                "parsing_strictness must be 'strict' or explicitly 'low'"
            )
        relative = Path(self.schedule_include)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in str(self.schedule_include)
        ):
            raise OpmCertificationError("schedule_include must be a safe relative path")
        if self.density_map is not None and (
            not self.density_map.is_file() or self.density_map.is_symlink()
        ):
            raise OpmCertificationError(
                f"density map is not a regular file: {self.density_map}"
            )

    def get_provenance(self) -> str:
        return f"{self.runner.get_provenance()}; restart=authenticated-full-replay/v1"

    def run_from_restart(
        self, case: Case, state: State, actions: tuple[ControlAction, ...]
    ) -> GdmResult:
        from .economics import CHDDEconomicsAdapter
        from .opm_chdd import export_opm_chdd
        from .schedule import ScheduleCompiler
        from .track1 import GdmResult

        self.validate_case(case)
        if state.case_id != case.case_id:
            raise OpmCertificationError("restart state belongs to another case")
        if set(well.well for well in state.wells) != set(
            (*case.producers, *case.injectors)
        ):
            raise OpmCertificationError("restart state must contain every case well once")

        compiler = ScheduleCompiler()
        ordered = compiler.validate(case, actions)
        if not ordered or ordered != actions or any(
            action.month != state.month for action in ordered
        ):
            raise OpmCertificationError(
                "candidate must be non-empty, canonical, and belong to restart month"
            )
        history = self._authenticated_history(case, state)
        if any(action.month >= state.month for action in history):
            raise OpmCertificationError("restart lineage contains future controls")
        accepted = compiler.validate(case, (*history, *ordered))

        source_sha = _source_digest(self.source)
        payload = {
            "case_id": case.case_id,
            "restart_ref": state.restart_ref,
            "source_sha256": source_sha,
            "actions": [self._action_value(action) for action in ordered],
        }
        run_id = "opm-full-replay-" + sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        run_root = self._output_root()
        run_dir = run_root / run_id
        prepared = self.runner.prepare(
            self.source,
            run_dir,
            deck=self.deck,
            normalize_model_y=self.normalize_model_y,
        )

        schedule_path = self._prepared_schedule(prepared)
        source_schedule = schedule_path.read_bytes().decode("utf-8")
        overlay_text, overlay_provenance = self._full_replay_schedule(
            source_schedule,
            accepted,
            known_wells=(*case.producers, *case.injectors),
            replay_month=state.month,
        )
        schedule_path.write_bytes(overlay_text.encode("utf-8"))
        overlay_manifest = prepared.run_dir / "schedule-overlay.json"
        self._write_new_json(
            overlay_manifest,
            {
                "schema": "timesoil.aios.schedule-full-replay/v1",
                "generator": "OpmGdmBackend",
                "schedule": schedule_path.relative_to(prepared.input_dir).as_posix(),
                "provenance": overlay_provenance,
            },
        )

        result = self.runner._run_prepared(
            prepared, parsing_strictness=self.parsing_strictness
        )
        report, extraction = self.runner.extract_summary_report(
            result, result.run_dir / "summary-report.txt"
        )
        canonical_dir = result.run_dir / "canonical"
        chdd_csv = canonical_dir / "chdd.csv"
        trajectory_csv = canonical_dir / "trajectory.csv"
        export_manifest = canonical_dir / "manifest.json"
        export_opm_chdd(
            report,
            chdd_csv,
            trajectory_csv,
            export_manifest,
            scenario_id=run_id,
            source_model=self.source_model or case.case_id,
            opm_run_manifest=result.manifest_path,
            summary_extraction_manifest=extraction,
            deck_dir=prepared.deck_path.parent,
            density_map=self.density_map,
            unit_system=prepared.unit_system,
        )

        next_month = self._next_month(state.month)
        next_wells = self._wells_at(
            report, case, next_month, deck_dir=prepared.deck_path.parent
        )
        violations = tuple(
            f"{well.well}: actual liquid rate {well.liquid_rate:g} exceeds "
            f"{case.max_liquid_rate:g}"
            for well in next_wells
            if well.role is WellRole.PRODUCER
            and well.liquid_rate > case.max_liquid_rate + 1e-6
        )
        if violations:
            raise OpmCertificationError("; ".join(violations))

        with chdd_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        economic_records = [
            record
            for record in records
            if date.fromisoformat(str(record["DATA"])) >= case.economics_start
        ]
        adapter = self.economics or CHDDEconomicsAdapter.from_env()
        economic_result = adapter.calculate(
            economic_records,
            start_year=case.economics_start.year,
            output_dir=result.run_dir / "economics",
        )
        if economic_result.start_date != case.economics_start.isoformat():
            raise OpmCertificationError(
                "official CHDD start date differs from case economics_start"
            )

        provisional_state = State(
            case.case_id,
            next_month,
            "pending-authenticated-lineage",
            next_wells,
        )
        lineage_path = result.run_dir / "lineage.json"
        lineage_value = {
            "schema": self._LINEAGE_SCHEMA,
            "status": "certified",
            "provenance": {
                "mode": "full-replay",
                "simulator": self.runner.get_provenance(),
                "binary_restart": False,
            },
            "run_id": run_id,
            "case_id": case.case_id,
            "source_sha256": source_sha,
            "prior_restart_ref": state.restart_ref,
            "input_state": self._state_value(state),
            "step_actions": [self._action_value(action) for action in ordered],
            "accepted_actions": [self._action_value(action) for action in accepted],
            "schedule_overlay": overlay_provenance,
            "next_state": self._state_value(provisional_state),
            "economics": {
                "start_date": economic_result.start_date,
                "total_chdd_m": economic_result.total_chdd_m,
                "profitability_index": economic_result.profitability_index,
            },
            "artifacts": self._lineage_artifacts(
                result,
                overlay_manifest,
                report,
                extraction,
                chdd_csv,
                trajectory_csv,
                export_manifest,
                economic_result.output_dir,
            ),
        }
        self._write_new_json(lineage_path, lineage_value)
        lineage_sha = _sha256_file(lineage_path)
        lineage_sidecar = lineage_path.with_suffix(".sha256")
        with lineage_sidecar.open("x", encoding="ascii") as stream:
            stream.write(f"{lineage_sha}  {lineage_path.name}\n")
        restart_ref = self._restart_ref(lineage_path, lineage_sha)
        next_state = State(case.case_id, next_month, restart_ref, next_wells)
        trajectory = Trajectory(
            run_id=run_id,
            case_id=case.case_id,
            month=state.month,
            actions=ordered,
            next_state=next_state,
            simulator=self.get_provenance(),
            certified=True,
            chdd_complete=True,
        )
        economics = Economics(
            run_id=run_id,
            start_date=case.economics_start,
            npv_million_rub=economic_result.total_chdd_m,
            complete=True,
        )
        return GdmResult(trajectory, economics)

    def _output_root(self) -> Path:
        assert self.runs_dir is not None
        current = self.runs_dir
        for path in (current, *current.parents):
            if path.exists() and path.is_symlink():
                raise OpmCertificationError(f"output path contains a symlink: {path}")
        current.mkdir(parents=True, exist_ok=True)
        if not current.is_dir():
            raise OpmCertificationError(f"OPM runs_dir is not a directory: {current}")
        return current.resolve()

    def _prepared_schedule(self, prepared: PreparedCase) -> Path:
        assert self.schedule_include is not None
        path = (prepared.input_dir / Path(self.schedule_include)).resolve()
        try:
            path.relative_to(prepared.input_dir.resolve())
        except ValueError:
            raise OpmCertificationError("schedule include escapes prepared case") from None
        if not path.is_file() or path.is_symlink():
            raise OpmCertificationError(
                f"schedule include is not a regular file: {path}"
            )
        return path

    @staticmethod
    def _full_replay_schedule(
        source: str,
        actions: tuple[ControlAction, ...],
        *,
        known_wells: tuple[str, ...],
        replay_month: date,
    ) -> tuple[str, dict[str, Any]]:
        from .schedule_overlay import _date_blocks, apply_schedule_overlay

        full = apply_schedule_overlay(source, actions, known_wells=known_wells)
        blocks = _date_blocks(full.text)
        matches = [index for index, block in enumerate(blocks) if block.month == replay_month]
        if len(matches) != 1 or matches[0] + 1 >= len(blocks):
            raise OpmCertificationError(
                "full replay month must have exactly one following report date"
            )
        following = blocks[matches[0] + 1]
        text = "".join(full.text.splitlines(keepends=True)[: following.end_line])
        return text, {
            "mode": "full-replay",
            "source_sha256": full.source_sha256,
            "controls_sha256": full.controls_sha256,
            "output_sha256": sha256(text.encode()).hexdigest(),
            "action_count": len(actions),
            "action_months": [month.isoformat() for month in full.action_months],
            "truncated_after": following.month.isoformat(),
        }

    def _authenticated_history(
        self, case: Case, state: State
    ) -> tuple[ControlAction, ...]:
        try:
            path, _ = self._parse_restart_ref(state.restart_ref)
            value = _json_object(path, "restart lineage")
            if value.get("schema") == "timesoil.aios.opm-run/v1":
                self._verify_opm_manifest(path, baseline=True)
                extraction = path.parent / "summary-extraction.json"
                proof = _json_object(extraction, "baseline summary extraction")
                output = proof.get("output_report")
                report = _linked_run_file(
                    path.parent,
                    output.get("path") if isinstance(output, dict) else None,
                    "baseline summary report",
                )
                verify_summary_extraction(
                    report,
                    extraction,
                    path,
                    _docker_executable=self.runner.docker_executable,
                )
                baseline_deck = _linked_run_file(
                    path.parent,
                    f"input/{value.get('deck')}",
                    "baseline deck",
                )
                expected = State(
                    case.case_id,
                    state.month,
                    state.restart_ref,
                    self._wells_at(
                        report, case, state.month, deck_dir=baseline_deck.parent
                    ),
                )
                self._assert_same_state(state, expected)
                return ()
            if value.get("schema") != self._LINEAGE_SCHEMA:
                raise OpmCertificationError("restart_ref has an unsupported manifest schema")
            history = self._history_from_lineage(case, path, value, set())
            if self._state_value(state) != value.get("next_state"):
                raise OpmCertificationError("restart state differs from authenticated lineage")
            return history
        except OpmCertificationError:
            raise
        except Exception as exc:
            raise OpmCertificationError(
                f"restart lineage verification failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _history_from_lineage(
        self,
        case: Case,
        path: Path,
        value: dict[str, Any],
        seen: set[Path],
    ) -> tuple[ControlAction, ...]:
        resolved = path.resolve()
        if resolved in seen:
            raise OpmCertificationError("restart lineage contains a cycle")
        seen.add(resolved)
        if (
            value.get("schema") != self._LINEAGE_SCHEMA
            or value.get("status") != "certified"
            or value.get("case_id") != case.case_id
            or value.get("source_sha256") != _source_digest(self.source)
            or not isinstance(value.get("provenance"), dict)
            or value["provenance"].get("mode") != "full-replay"
            or value["provenance"].get("binary_restart") is not False
        ):
            raise OpmCertificationError("restart lineage identity or provenance is invalid")
        self._verify_lineage_artifacts(path.parent, value.get("artifacts"))

        prior = value.get("prior_restart_ref")
        if not isinstance(prior, str):
            raise OpmCertificationError("restart lineage misses prior_restart_ref")
        prior_path, _ = self._parse_restart_ref(prior)
        prior_value = _json_object(prior_path, "prior restart manifest")
        if prior_value.get("schema") == self._LINEAGE_SCHEMA:
            previous = self._history_from_lineage(
                case, prior_path, prior_value, seen
            )
        elif prior_value.get("schema") == "timesoil.aios.opm-run/v1":
            self._verify_opm_manifest(prior_path, baseline=True)
            previous = ()
        else:
            raise OpmCertificationError("restart chain terminates in an unsupported manifest")

        step = self._actions_value(value.get("step_actions"))
        accepted = self._actions_value(value.get("accepted_actions"))
        if accepted != (*previous, *step):
            raise OpmCertificationError("restart lineage action history is not contiguous")
        return accepted

    def _verify_opm_manifest(self, path: Path, *, baseline: bool) -> None:
        manifest, _ = _validated_summary_artifacts(path)
        if manifest.get("source_sha256") != _source_digest(self.source):
            raise OpmCertificationError("OPM restart source differs from configured case")
        artifacts = manifest.get("artifacts")
        assert isinstance(artifacts, list)
        paths: set[str] = set()
        for item in artifacts:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise OpmCertificationError("OPM artifact entry is invalid")
            relative = item["path"]
            if relative in paths:
                raise OpmCertificationError(f"duplicate OPM artifact: {relative}")
            paths.add(relative)
            artifact = _linked_run_file(path.parent, relative, "OPM artifact")
            digest = _lower_digest(item.get("sha256"), f"OPM artifact {relative}")
            size = item.get("bytes")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or artifact.stat().st_size != size
                or _sha256_file(artifact) != digest
            ):
                raise OpmCertificationError(f"OPM artifact hash mismatch: {relative}")
        if baseline and "schedule-overlay.json" in paths:
            raise OpmCertificationError("initial restart must be an unmodified baseline run")

    def _verify_lineage_artifacts(self, root: Path, raw: Any) -> None:
        if not isinstance(raw, list) or not raw:
            raise OpmCertificationError("restart lineage artifact list is missing")
        seen: set[str] = set()
        opm_manifests: list[Path] = []
        for item in raw:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("purpose"), str)
            ):
                raise OpmCertificationError("restart lineage artifact entry is invalid")
            relative = item["path"]
            if relative in seen:
                raise OpmCertificationError(f"duplicate lineage artifact: {relative}")
            seen.add(relative)
            artifact = _linked_run_file(root, relative, "lineage artifact")
            digest = _lower_digest(item.get("sha256"), f"lineage artifact {relative}")
            if _sha256_file(artifact) != digest:
                raise OpmCertificationError(f"lineage artifact hash mismatch: {relative}")
            if item["purpose"] == "opm_run_manifest":
                opm_manifests.append(artifact)
        if len(opm_manifests) != 1:
            raise OpmCertificationError("lineage must reference exactly one OPM run manifest")
        self._verify_opm_manifest(opm_manifests[0], baseline=False)

    def _parse_restart_ref(self, value: str) -> tuple[Path, str]:
        marker = "#sha256="
        if value.count(marker) != 1:
            raise OpmCertificationError(
                "restart_ref must be '<manifest path>#sha256=<digest>'"
            )
        raw_path, digest = value.split(marker, 1)
        if not raw_path or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise OpmCertificationError("restart_ref path or SHA-256 is invalid")
        path = Path(raw_path).resolve()
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != digest:
            raise OpmCertificationError("restart_ref manifest is missing or hash-mismatched")
        sidecar = path.with_suffix(".sha256")
        expected = f"{digest}  {path.name}\n"
        if (
            not sidecar.is_file()
            or sidecar.is_symlink()
            or sidecar.read_text(encoding="ascii") != expected
        ):
            raise OpmCertificationError("restart_ref SHA-256 sidecar is missing or invalid")
        return path, digest

    @staticmethod
    def _restart_ref(path: Path, digest: str) -> str:
        return f"{path.resolve()}#sha256={digest}"

    @staticmethod
    def _next_month(month: date) -> date:
        return date(month.year + (month.month == 12), month.month % 12 + 1, 1)

    @staticmethod
    def _action_value(action: ControlAction) -> dict[str, Any]:
        return {
            "month": action.month.isoformat(),
            "well": action.well,
            "role": action.role.value,
            "status": action.status.value,
            "target": action.target.value,
            "value": action.value,
        }

    @classmethod
    def _actions_value(cls, raw: Any) -> tuple[ControlAction, ...]:
        if not isinstance(raw, list):
            raise OpmCertificationError("lineage actions must be a JSON array")
        actions: list[ControlAction] = []
        expected = {"month", "well", "role", "status", "target", "value"}
        for item in raw:
            if not isinstance(item, dict) or set(item) != expected:
                raise OpmCertificationError("lineage action has invalid fields")
            value = item["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OpmCertificationError("lineage action value must be numeric")
            actions.append(
                ControlAction(
                    date.fromisoformat(str(item["month"])),
                    str(item["well"]),
                    WellRole(str(item["role"])),
                    WellStatus(str(item["status"])),
                    ControlTarget(str(item["target"])),
                    float(value),
                )
            )
        return tuple(actions)

    @staticmethod
    def _state_value(state: State) -> dict[str, Any]:
        return {
            "case_id": state.case_id,
            "month": state.month.isoformat(),
            "wells": [
                {
                    "well": well.well,
                    "role": well.role.value,
                    "active": well.active,
                    "oil_rate": well.oil_rate,
                    "liquid_rate": well.liquid_rate,
                    "injection_rate": well.injection_rate,
                    "bhp": well.bhp,
                }
                for well in sorted(state.wells, key=lambda item: item.well)
            ],
        }

    @staticmethod
    def _wells_at(
        report: Path, case: Case, month: date, *, deck_dir: Path
    ) -> tuple[WellState, ...]:
        from .opm_chdd import (
            OpmChddError,
            _deck_text,
            _eclipse_date,
            _read_summary,
            _single_record,
        )

        try:
            summary, _ = _read_summary(report)
        except OpmChddError as exc:
            if str(exc) != "SUMMARY without DATE requires deck START and TIME":
                raise
            expanded_deck, _ = _deck_text(deck_dir)
            deck_start = _eclipse_date(_single_record(expanded_deck, "START"), "START")
            summary, _ = _read_summary(report, start_date=deck_start)
        matches = [by_well for current, by_well, _ in summary if current == month]
        if len(matches) != 1:
            raise OpmCertificationError(
                f"OPM summary must contain exactly one row for {month.isoformat()}"
            )
        values = matches[0]
        expected = set((*case.producers, *case.injectors))
        if set(values) != expected:
            raise OpmCertificationError("OPM summary well set differs from case")

        result: list[WellState] = []
        for name in sorted(expected):
            raw = values[name]
            rates = [raw[vector] for vector in ("WOPR", "WLPR", "WWIR")]
            if any(value < -1e-6 for value in rates):
                raise OpmCertificationError(f"OPM returned a negative well rate for {name}")
            oil, liquid, injection = (max(0.0, value) for value in rates)
            if oil > liquid + 1e-6:
                raise OpmCertificationError(f"OPM oil rate exceeds liquid rate for {name}")
            result.append(
                WellState(
                    name,
                    case.role_of(name),
                    raw["WEFF"] > 0,
                    oil,
                    max(oil, liquid),
                    injection,
                    raw["WBHP"],
                )
            )
        return tuple(result)

    @staticmethod
    def _assert_same_state(actual: State, expected: State) -> None:
        actual_by_well = {well.well: well for well in actual.wells}
        expected_by_well = {well.well: well for well in expected.wells}
        if (
            actual.case_id != expected.case_id
            or actual.month != expected.month
            or actual_by_well.keys() != expected_by_well.keys()
        ):
            raise OpmCertificationError("restart state identity differs from baseline summary")
        for name, wanted in expected_by_well.items():
            found = actual_by_well[name]
            if found.role is not wanted.role or found.active is not wanted.active:
                raise OpmCertificationError(f"restart state flags differ for {name}")
            pairs = (
                (found.oil_rate, wanted.oil_rate),
                (found.liquid_rate, wanted.liquid_rate),
                (found.injection_rate, wanted.injection_rate),
            )
            if any(not isclose(left, right, rel_tol=1e-6, abs_tol=1e-6) for left, right in pairs):
                raise OpmCertificationError(f"restart state rates differ for {name}")
            if found.bhp is None or wanted.bhp is None or not isclose(
                found.bhp, wanted.bhp, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise OpmCertificationError(f"restart state BHP differs for {name}")

    @staticmethod
    def _lineage_artifacts(
        result: OpmRunResult,
        overlay_manifest: Path,
        report: Path,
        extraction: Path,
        chdd_csv: Path,
        trajectory_csv: Path,
        export_manifest: Path,
        economics_dir: Path,
    ) -> list[dict[str, str]]:
        paths = [
            ("schedule_overlay", overlay_manifest),
            ("opm_run_manifest", result.manifest_path),
            ("opm_run_manifest_hash", result.manifest_path.with_suffix(".sha256")),
            ("summary_report", report),
            ("summary_extraction", extraction),
            ("canonical_chdd", chdd_csv),
            ("canonical_trajectory", trajectory_csv),
            ("canonical_export_manifest", export_manifest),
            *(("official_chdd_artifact", path) for path in _regular_files(economics_dir)),
        ]
        return [
            {
                "purpose": purpose,
                "path": path.relative_to(result.run_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
            for purpose, path in paths
        ]

    @staticmethod
    def _write_new_json(path: Path, value: dict[str, Any]) -> None:
        encoded = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        with path.open("xb") as stream:
            stream.write(encoded)
