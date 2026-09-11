"""Derive geometry-first well blocks of a deck and write a hashed ``blocks.json``."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from timesoil.aios.blocks import BlocksError, build_blocks, read_deck_geometry


def deck_root(path: Path, workspace: Path) -> Path:
    """Accept an extracted deck directory or a deck archive; return the directory holding the deck."""

    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip" or not path.is_file():
        raise BlocksError(f"deck must be a directory or a .zip archive: {path}")
    with ZipFile(path) as archive:
        archive.extractall(workspace)
    decks = sorted(
        item.parent
        for item in workspace.rglob("*")
        if item.is_file() and not item.is_symlink() and item.suffix.upper() == ".DATA"
    )
    if len(decks) != 1:
        raise BlocksError(f"archive must contain exactly one .DATA deck, found {len(decks)}")
    return decks[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", type=Path, help="deck directory or deck .zip archive")
    parser.add_argument("connectivity", type=Path, help="connectivity.json from export_opm_connectivity.py")
    parser.add_argument("output", type=Path, help="blocks.json to create; an existing file is never overwritten")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--min-component-wells", type=int, default=3)
    parser.add_argument(
        "--no-zcorn",
        action="store_true",
        help="ignore ZCORN vertical overlap and keep every face-adjacent pair connected",
    )
    args = parser.parse_args()
    raw = args.connectivity.read_bytes()
    connectivity = json.loads(raw)
    with TemporaryDirectory() as workspace:
        geometry = read_deck_geometry(deck_root(args.deck, Path(workspace)), not args.no_zcorn)
        payload = build_blocks(
            geometry,
            connectivity,
            args.blocks,
            args.min_component_wells,
            sha256(raw).hexdigest(),
            {
                "tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "zcorn_overlap": not args.no_zcorn,
            },
        )
    with open(args.output, "x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "deck_sha256": payload["deck_sha256"],
                "connectivity_sha256": payload["connectivity_sha256"],
                "source_sha256": payload["source_sha256"],
                "grid": payload["grid"],
                "stats": payload["stats"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
