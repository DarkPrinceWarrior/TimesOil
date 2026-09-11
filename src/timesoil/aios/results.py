"""Read-only view over a sealed Track 2 run directory.

Pure functions: every entry point takes a runs root plus a run id and reads
whatever the run actually produced. A missing artifact yields ``None`` or an
empty container -- never an exception -- because the operator UI must render a
partially finished run. Nothing is cached and nothing is written.
"""

from __future__ import annotations

import calendar
import csv
import json
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException


RUNS_ROOT_ENV = "TIMESOIL_RESULTS_ROOT"
DEFAULT_RUNS_ROOT = "results/audit-20260909"
RUN_GLOB = "case-z-*"
CONTROLS_DIFF_LIMIT = 500

# ponytail: stock oil density when the run carries no canonical manifest value.
# Water volume is the only consumer; swap for the deck DENSITY table if the UI
# ever needs water cut to match the simulator exactly.
_DEFAULT_OIL_DENSITY_T_M3 = 0.85


class ResultsError(ValueError):
    """The requested run id is not a usable directory name."""


def runs_root(repo_root: Path | None = None, env: dict[str, str] | None = None) -> Path:
    import os

    environ = os.environ if env is None else env
    configured = environ.get(RUNS_ROOT_ENV, "").strip()
    if configured:
        return Path(configured)
    base = repo_root or Path(__file__).resolve().parents[3]
    return base / DEFAULT_RUNS_ROOT


def validate_run_id(run_id: str) -> str:
    """Reject anything that is not a single plain directory name."""

    if (
        not run_id
        or len(run_id) > 128
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or "\x00" in run_id
        or run_id != Path(run_id).name
    ):
        raise ResultsError("run_id must be a plain directory name")
    return run_id


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _first(root: Path, pattern: str) -> Path | None:
    try:
        return min((p for p in root.rglob(pattern) if p.is_file()), key=lambda p: str(p))
    except (OSError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _month_start(raw: str) -> str | None:
    text = raw.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        return parsed.replace(day=1).isoformat()
    return None


def _days_in_month(month: str) -> int:
    stamp = date.fromisoformat(month)
    return calendar.monthrange(stamp.year, stamp.month)[1]


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Every run directory under ``root``, newest modification first."""

    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    runs: list[dict[str, Any]] = []
    for path in entries:
        audit = path / "final" / "audit.json"
        if not (path.match(RUN_GLOB) or audit.is_file()):
            continue
        runs.append(
            {
                "run_id": path.name,
                "path": str(path),
                "final": audit.is_file(),
                "updated_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                )
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )
    runs.sort(key=lambda row: row["updated_utc"], reverse=True)
    return runs


def _load_profile(run_dir: Path, repo_root: Path) -> dict[str, Any] | None:
    candidates = [
        run_dir / "case-profile.json",
        run_dir / "profile.json",
        run_dir / "final" / "profile.json",
        repo_root / "config" / "case_z_test.json",
    ]
    raw: Any = None
    for path in candidates:
        raw = _read_json(path)
        if isinstance(raw, dict):
            break
        raw = None
    if raw is None:
        return None
    profile = raw.get("profile") if isinstance(raw.get("profile"), dict) else raw
    vrr = profile.get("vrr") if isinstance(profile.get("vrr"), dict) else {}
    bounds = profile.get("bhp_bounds")
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        bounds = [profile.get("bhp_min"), profile.get("bhp_max")]
    return {
        "liquid_cap_m3d": _number(profile.get("liquid_cap_m3d")),
        "injection_cap_m3d": _number(profile.get("injection_cap_m3d")),
        "vrr": {
            "min": _number(vrr.get("min")),
            "max": _number(vrr.get("max")),
            "window_months": vrr.get("window_months"),
        },
        "bhp_bounds": [_number(bounds[0]), _number(bounds[1])],
    }


def _load_blocks(run_dir: Path) -> dict[str, dict[str, Any]]:
    """well -> {block, x, y}; coordinates fall back to completion centroids."""

    raw = _read_json(run_dir / "blocks.json")
    if raw is None:
        found = _first(run_dir, "blocks.json")
        raw = _read_json(found) if found else None
    if isinstance(raw, dict) and isinstance(raw.get("wells"), (dict, list)):
        raw = raw["wells"]
    rows: list[tuple[str, Any]]
    if isinstance(raw, dict):
        rows = list(raw.items())
    elif isinstance(raw, list):
        rows = [(str(item.get("well", "")), item) for item in raw if isinstance(item, dict)]
    else:
        return {}
    blocks: dict[str, dict[str, Any]] = {}
    for well, entry in rows:
        if not well:
            continue
        if not isinstance(entry, dict):
            blocks[well] = {"block": entry if isinstance(entry, int) else None, "x": None, "y": None}
            continue
        x, y = _number(entry.get("x")), _number(entry.get("y"))
        if x is None or y is None:
            centroid = entry.get("centroid") or entry.get("completion_centroid")
            if isinstance(centroid, dict):
                x, y = _number(centroid.get("x") or centroid.get("i")), _number(
                    centroid.get("y") or centroid.get("j")
                )
            elif isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                x, y = _number(centroid[0]), _number(centroid[1])
        if x is None or y is None:
            x, y = _number(entry.get("i")), _number(entry.get("j"))
        block = entry.get("block")
        blocks[well] = {
            "block": block if isinstance(block, int) and not isinstance(block, bool) else None,
            "x": x,
            "y": y,
        }
    return blocks


def _read_chdd(path: Path) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    """Return ordered months and well -> month -> monthly volumes from chdd.csv."""

    months: list[str] = []
    seen: set[str] = set()
    wells: dict[str, dict[str, dict[str, float]]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                month = _month_start(str(row.get("DATA", "")))
                well = str(row.get("well", "")).strip()
                if month is None or not well:
                    continue
                if month not in seen:
                    seen.add(month)
                    months.append(month)
                oil_t = _number(row.get("WOMT_Diff")) or 0.0
                liquid = _number(row.get("WLPT_Diff")) or 0.0
                injection = _number(row.get("WWIT_Diff")) or 0.0
                wells.setdefault(well, {})[month] = {
                    "oil_t": oil_t,
                    "liquid_m3": liquid,
                    "water_m3": max(liquid - oil_t / _DEFAULT_OIL_DENSITY_T_M3, 0.0),
                    "injection_m3": injection,
                    "bhp_bar": _number(row.get("BHP")),
                    "weff": _number(row.get("WEFF")),
                    "rate": abs(_number(row.get("WLPR")) or 0.0)
                    + abs(_number(row.get("WWIR")) or 0.0),
                }
    except (OSError, csv.Error):
        return [], {}
    months.sort()
    return months, wells


def _npv(run_dir: Path, audit: Any) -> dict[str, Any]:
    npv: dict[str, Any] = {
        "final_m": None,
        "incumbent_m": None,
        "gain_pct": None,
        "oil_total_t": None,
        "oil_incumbent_t": None,
        "source": None,
    }
    blocks: list[Any] = []
    if isinstance(audit, dict):
        blocks.append(audit)
        for key in ("npv", "economics", "final"):
            if isinstance(audit.get(key), dict):
                blocks.append(audit[key])
    economics = _first(run_dir / "final", "result.json") if (run_dir / "final").is_dir() else None
    if economics is not None:
        payload = _read_json(economics)
        if isinstance(payload, dict):
            blocks.append(payload)
            npv["source"] = str(economics.relative_to(run_dir).as_posix())
    keys = {
        "final_m": ("final_m", "final_chdd_m", "total_chdd_m", "candidate_chdd_m"),
        "incumbent_m": ("incumbent_m", "incumbent_chdd_m", "baseline_chdd_m"),
        "gain_pct": ("gain_pct", "npv_gain_pct", "gain_percent"),
        "oil_total_t": ("oil_total_t", "oil_t", "final_oil_t"),
        "oil_incumbent_t": ("oil_incumbent_t", "incumbent_oil_t", "baseline_oil_t"),
    }
    for field, names in keys.items():
        for block in blocks:
            for name in names:
                value = _number(block.get(name)) if isinstance(block, dict) else None
                if value is not None:
                    npv[field] = value
                    break
            if npv[field] is not None:
                break
    if npv["gain_pct"] is None and npv["final_m"] is not None and npv["incumbent_m"]:
        npv["gain_pct"] = (npv["final_m"] / npv["incumbent_m"] - 1.0) * 100.0
    if npv["source"] is None and isinstance(audit, dict):
        npv["source"] = "final/audit.json"
    return npv


def _violations(
    field: dict[str, list[Any]], months: list[str], constraints: dict[str, Any]
) -> list[dict[str, Any]]:
    rules = (
        ("liquid_cap", "max_liquid_m3d", "liquid_m3d", "gt"),
        ("injection_cap", "max_injection_m3d", "injection_m3d", "gt"),
        ("vrr_min", "vrr_min", "vrr", "lt"),
        ("vrr_max", "vrr_max", "vrr", "gt"),
    )
    violations: list[dict[str, Any]] = []
    for rule, limit_key, series_key, sense in rules:
        limit = constraints.get(limit_key)
        if limit is None:
            continue
        for month, value in zip(months, field.get(series_key, [])):
            if value is None:
                continue
            if (sense == "gt" and value > limit) or (sense == "lt" and value < limit):
                violations.append(
                    {
                        "rule": rule,
                        "month": month,
                        "well": None,
                        "detail": f"{series_key}={value:.3f} vs limit {limit:.3f}",
                    }
                )
    return violations


def _controls_diff(run_dir: Path) -> list[dict[str, Any]]:
    # ponytail: the sealed run already ships the incumbent/final comparison;
    # recomputing it from two schedules would duplicate planning.py.
    raw = _read_json(run_dir / "final" / "controls-diff.json")
    if raw is None:
        raw = _read_json(run_dir / "final" / "controls_diff.json")
    if isinstance(raw, dict):
        raw = raw.get("rows") or raw.get("controls_diff")
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw[:CONTROLS_DIFF_LIMIT]:
        if not isinstance(item, dict) or item.get("field") not in {"value", "status"}:
            continue
        rows.append(
            {
                "month": str(item.get("month", "")),
                "well": str(item.get("well", "")),
                "field": item["field"],
                "incumbent": item.get("incumbent"),
                "final": item.get("final"),
            }
        )
    return rows


def _sources(run_dir: Path, paths: list[Path]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in paths:
        try:
            digests[path.relative_to(run_dir).as_posix()] = sha256(
                path.read_bytes()
            ).hexdigest()
        except (OSError, ValueError):
            continue
    return digests


def load_run(root: Path, run_id: str) -> dict[str, Any] | None:
    """Full contract payload for one run, or ``None`` when it does not exist."""

    run_dir = root / validate_run_id(run_id)
    if not run_dir.is_dir():
        return None
    repo_root = Path(__file__).resolve().parents[3]
    final_dir = run_dir / "final"

    audit_path = final_dir / "audit.json"
    audit = _read_json(audit_path)
    official = isinstance(audit, dict) and audit.get("accepted") is not False

    chdd_path = _first(final_dir, "chdd.csv") if final_dir.is_dir() else None
    months, well_months = _read_chdd(chdd_path) if chdd_path else ([], {})

    blocks = _load_blocks(run_dir)
    profile = _load_profile(run_dir, repo_root)

    field = {
        key: [0.0] * len(months)
        for key in ("oil_t", "liquid_m3", "water_m3", "injection_m3")
    }
    well_series: dict[str, dict[str, Any]] = {}
    wells: list[dict[str, Any]] = []
    for well in sorted(well_months):
        by_month = well_months[well]
        series = {
            key: [float(by_month.get(month, {}).get(key, 0.0)) for month in months]
            for key in ("oil_t", "liquid_m3", "water_m3", "injection_m3")
        }
        series["bhp_bar"] = [by_month.get(month, {}).get("bhp_bar") for month in months]
        well_series[well] = series
        for key in field:
            for index, value in enumerate(series[key]):
                field[key][index] += value
        totals = {key: sum(series[key]) for key in field}
        last = by_month.get(months[-1], {}) if months else {}
        shut = bool(months) and (last.get("weff") == 0.0 or last.get("rate", 0.0) == 0.0)
        placement = blocks.get(well, {})
        wells.append(
            {
                "well": well,
                "role": "injector" if totals["injection_m3"] > totals["liquid_m3"] else "producer",
                "block": placement.get("block"),
                "x": placement.get("x"),
                "y": placement.get("y"),
                "status_last": "SHUT" if shut else "OPEN",
                "totals": totals,
            }
        )

    days = [_days_in_month(month) for month in months]
    field["liquid_m3d"] = [volume / day for volume, day in zip(field["liquid_m3"], days)]
    field["injection_m3d"] = [
        volume / day for volume, day in zip(field["injection_m3"], days)
    ]
    field["vrr"] = [
        (injected / liquid) if liquid > 0 else None
        for injected, liquid in zip(field["injection_m3"], field["liquid_m3"])
    ]

    vrr_profile = (profile or {}).get("vrr") or {}
    constraints = {
        "max_liquid_m3d": (profile or {}).get("liquid_cap_m3d"),
        "max_injection_m3d": (profile or {}).get("injection_cap_m3d"),
        "vrr_min": vrr_profile.get("min"),
        "vrr_max": vrr_profile.get("max"),
    }
    constraints["violations"] = _violations(field, months, constraints)

    report = _read_json(run_dir / "explain" / "interpretability" / "report.json")
    summary_md = _read_text(run_dir / "explain" / "summary.md")

    commit = None
    if isinstance(audit, dict):
        for key in ("commit", "git_commit", "repo_commit"):
            if isinstance(audit.get(key), str):
                commit = audit[key]
                break

    sources = [path for path in (audit_path, chdd_path) if path is not None]
    return {
        "run_id": run_dir.name,
        "official": bool(official),
        "commit": commit,
        "profile": profile,
        "months": months,
        "npv": _npv(run_dir, audit),
        "wells": wells,
        "series": {"field": field, "wells": well_series},
        "constraints": constraints,
        "controls_diff": _controls_diff(run_dir),
        "interpretability": {
            "summary_md": summary_md,
            "report": report if isinstance(report, dict) else None,
        },
        "sources": _sources(run_dir, sources),
    }


router = APIRouter(prefix="/v1/results", tags=["results"])


@router.get("")
def list_results() -> dict[str, Any]:
    return {"runs": list_runs(runs_root())}


@router.get("/{run_id}")
def get_result(run_id: str) -> dict[str, Any]:
    try:
        payload = load_run(runs_root(), run_id)
    except ResultsError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    if payload is None:
        raise HTTPException(status_code=404, detail="run not found")
    return payload
