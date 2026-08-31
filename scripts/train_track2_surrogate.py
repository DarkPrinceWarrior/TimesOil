#!/usr/bin/env python3
"""Train and persist the Track 2 stateful surrogate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from timesoil.aios import surrogate as _surrogate_module
from timesoil.aios import track2 as _track2_module
from timesoil.aios.track2 import (
    CANONICAL_COLUMNS,
    fit_track2_surrogate,
    load_model_y_pipeline_proof,
    load_trajectory_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
_MAX_SOURCE_BYTES = 4 * 1024**2


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink path component is forbidden: {current}")


def _read_regular(path: Path, label: str) -> bytes:
    def read_once(descriptor: int, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024**2))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining or os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)

    absolute = Path(os.path.abspath(path))
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        directory_fds.append(os.open(absolute.anchor, directory_flags))
        for part in absolute.parts[1:-1]:
            directory_fds.append(
                os.open(part, directory_flags, dir_fd=directory_fds[-1])
            )
        file_fd = os.open(
            absolute.name or ".", os.O_RDONLY | no_follow, dir_fd=directory_fds[-1]
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if before.st_size > _MAX_SOURCE_BYTES:
            raise ValueError(f"{label} exceeds {_MAX_SOURCE_BYTES} bytes")
        first = read_once(file_fd, before.st_size)
        middle = os.fstat(file_fd)
        os.lseek(file_fd, 0, os.SEEK_SET)
        second = read_once(file_fd, before.st_size)
        after = os.fstat(file_fd)
        identity = lambda item: (
            item.st_mode,
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if not (
            identity(before) == identity(middle) == identity(after)
            and first == second
        ):
            raise ValueError(f"{label} changed while it was read")
        return first
    finally:
        for descriptor in ([file_fd] if file_fd is not None else []) + list(
            reversed(directory_fds)
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _snapshot_executed_sources() -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for module_file in (__file__, _track2_module.__file__, _surrogate_module.__file__):
        if not isinstance(module_file, str) or not module_file:
            raise ValueError("executed source module has no __file__")
        path = Path(module_file).absolute()
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(_read_regular(path, "executed source")).hexdigest(),
            }
        )
    return sources


def _verify_executed_sources(expected: list[dict[str, str]]) -> None:
    if _snapshot_executed_sources() != expected:
        raise RuntimeError("executed source changed during surrogate training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, help="canonical Model Z OPM CSV/Parquet or directory")
    source.add_argument(
        "--model-y-proof",
        action="store_true",
        help="use available Model Y scenarios strictly as a pipeline proof (default)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="OPM/CHDD export manifest file, or directory of manifests",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw_data")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "track2_surrogate")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conformal-level", type=float, default=0.9)
    args = parser.parse_args()
    if args.dataset and not args.manifest:
        parser.error("--manifest is required with --dataset")
    if args.manifest and not args.dataset:
        parser.error("--manifest requires --dataset")
    return args


def main() -> int:
    args = parse_args()
    executed_sources = _snapshot_executed_sources()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory {output}")
    if args.dataset:
        trajectories = load_trajectory_dataset(args.dataset, manifest=args.manifest)
    else:
        print("MODEL Y PIPELINE PROOF: these metrics are not Model Z surrogate metrics.")
        trajectories = load_model_y_pipeline_proof(args.raw_dir)
    run = fit_track2_surrogate(
        trajectories,
        test_fraction=args.test_fraction,
        ensemble_size=args.ensemble_size,
        n_estimators=args.n_estimators,
        horizon=args.horizon,
        seed=args.seed,
        conformal_level=args.conformal_level if args.dataset else None,
    )
    _verify_executed_sources(executed_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    report = run.report()
    report["canonical_columns"] = CANONICAL_COLUMNS
    report["executed_sources"] = executed_sources
    manifest_path = output / "model" / "manifest.json"
    manifest = run.model.save(manifest_path.parent)
    report["surrogate_artifact_hash"] = manifest["artifact_hash"]
    report["surrogate_manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    _verify_executed_sources(executed_sources)
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "model_artifact_hash": manifest["artifact_hash"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
