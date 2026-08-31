"""Fail-closed OPM Flow 2026.04 execution boundary."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import time
from typing import Any, Iterable, Iterator
from zipfile import BadZipFile, ZipFile, ZipInfo

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
        )

    def run(
        self,
        source: str | Path,
        run_dir: str | Path,
        *,
        deck: str | Path | None = None,
        parsing_strictness: str = "strict",
    ) -> OpmRunResult:
        if parsing_strictness not in {"strict", "low"}:
            raise ValueError("parsing_strictness must be 'strict' or explicitly 'low'")
        prepared = self.prepare(source, run_dir, deck=deck)
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
