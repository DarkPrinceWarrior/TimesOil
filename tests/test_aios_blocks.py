"""Geometry-derived blocks: components, well membership, deterministic partition."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from timesoil.aios.blocks import (
    BlocksError,
    build_blocks,
    cell_components,
    partition_wells,
    read_deck_geometry,
    well_membership,
)

DIMENS = (9, 2, 1)
WELL_CELLS: dict[str, tuple[tuple[int, int], ...]] = {
    "w1": ((1, 1),),
    "w2": ((2, 2),),
    "w3": ((4, 1),),
    "w4": ((3, 2), (4, 2), (6, 2)),
    "w5": ((6, 1),),
    "w6": ((7, 2),),
    "w7": ((9, 1),),
    "w8": ((9, 2),),
}
REAL_DECK = Path("docs/hackathon/models/model_z_final_opm/Model_Z_final_OPM.zip")


def _zcorn(offset_from_i: int | None) -> list[float]:
    """Flat corner depths; columns from ``offset_from_i`` (1-based) drop below the rest."""

    nx, ny, nz = DIMENS
    values = np.empty((nz, 2, ny, 2, nx, 2))
    values[:, 0] = 1000.0
    values[:, 1] = 1010.0
    if offset_from_i is not None:
        values[:, :, :, :, offset_from_i - 1 :, :] += 100.0
    return values.ravel().tolist()


def _deck(path: Path, active: list[int], offset_from_i: int | None = None) -> Path:
    nx, ny, nz = DIMENS
    wells = "\n".join(f" '{well}' 'G' {cells[0][0]} {cells[0][1]} 1* OIL /" for well, cells in WELL_CELLS.items())
    connections = "\n".join(
        f" '{well}' {i} {j} 1 1 OPEN /" for well, cells in WELL_CELLS.items() for i, j in cells
    )
    region = " ".join("1" if index % nx < 4 else "2" for index in range(nx * ny * nz))
    text = f"""RUNSPEC
METRIC
DIMENS
 {nx} {ny} {nz} /
TABDIMS
 1 1 /
GRID
ACTNUM
 {" ".join(str(value) for value in active)} /
ZCORN
 {" ".join(f"{value:.2f}" for value in _zcorn(offset_from_i))}
/
PROPS
DENSITY
 860 1010 0.9 /
REGIONS
PVTNUM
 {nx * ny * nz}*1 /
FIP_C1
 {region} /
SCHEDULE
WELSPECS
{wells}
/
COMPDAT
{connections}
/
"""
    deck = path / "synthetic.DATA"
    deck.write_text(text, encoding="utf-8")
    return path


def _split_active() -> list[int]:
    """Column i=5 inactive, so the lateral grid falls into two bodies."""

    nx, ny, nz = DIMENS
    return [0 if index % nx == 4 else 1 for index in range(nx * ny * nz)]


def _connectivity(well_ids: list[str], geometry) -> dict[str, object]:
    centroids = np.array([well_membership(geometry, well)["centroid_ij"] for well in well_ids])
    distance = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    weights = np.where(distance > 0, 1.0 / np.maximum(distance, 1e-9), 0.0)
    return {"well_ids": well_ids, "weights": weights.tolist()}


def test_two_bodies_and_multi_component_well(tmp_path: Path) -> None:
    geometry = read_deck_geometry(_deck(tmp_path, _split_active()))
    assert geometry.adjacency["components"] == 2
    assert sorted(geometry.component_sizes.values()) == [8, 8]
    left = geometry.labels[geometry.well_cells["w1"][0]]
    right = geometry.labels[geometry.well_cells["w5"][0]]
    assert left != right
    crossing = well_membership(geometry, "w4")
    assert crossing["components"] == {int(left): 2, int(right): 1}
    assert crossing["component"] == int(left)
    assert crossing["completions"] == 2 + 1
    assert well_membership(geometry, "w1")["regions"]["FIP_C1"] == {1: 1}
    assert well_membership(geometry, "w7")["regions"]["FIP_C1"] == {2: 1}


def test_zcorn_offset_cuts_a_face_adjacent_grid(tmp_path: Path) -> None:
    active = [1] * (DIMENS[0] * DIMENS[1] * DIMENS[2])
    geometry = read_deck_geometry(_deck(tmp_path, active, offset_from_i=5))
    assert geometry.adjacency["i_faces_without_overlap"] == DIMENS[1]
    assert geometry.adjacency["components"] == 2
    plain = read_deck_geometry(_deck(tmp_path, active, offset_from_i=5), zcorn_overlap=False)
    assert plain.adjacency["components"] == 1
    assert "i_faces_without_overlap" not in plain.adjacency


def test_partition_keeps_bodies_apart_and_ignores_well_order(tmp_path: Path) -> None:
    geometry = read_deck_geometry(_deck(tmp_path, _split_active()))
    well_ids = sorted(WELL_CELLS)
    payload = build_blocks(geometry, _connectivity(well_ids, geometry), blocks=3)
    assignment = payload["well_to_block"]
    assert sorted(payload["stats"]["block_well_counts"]) == [2, 2, 4]
    bodies = {
        entry["id"]: {payload["wells"][well]["component"] for well in entry["wells"]}
        for entry in payload["blocks"]
    }
    assert all(len(value) == 1 for value in bodies.values())
    assert assignment["w5"] == assignment["w6"] == assignment["w7"] == assignment["w8"]
    assert assignment["w1"] != assignment["w3"]
    shuffled = list(reversed(well_ids))
    connectivity = _connectivity(well_ids, geometry)
    order = [well_ids.index(well) for well in shuffled]
    weights = np.asarray(connectivity["weights"])[np.ix_(order, order)]
    reordered = build_blocks(
        geometry, {"well_ids": shuffled, "weights": weights.tolist()}, blocks=3
    )
    assert reordered["well_to_block"] == assignment
    assert reordered["blocks"] == payload["blocks"]


def test_small_component_merges_by_strongest_weight() -> None:
    well_ids = ["a", "b", "c", "d", "e"]
    components = [0, 0, 0, 0, 7]
    centroids = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0], [10.5, 0.5]])
    weights = np.zeros((5, 5))
    weights[4, 2] = weights[2, 4] = 9.0
    weights[4, 0] = weights[0, 4] = 1.0
    block, detail = partition_wells(well_ids, components, centroids, weights, blocks=2)
    assert detail["merged_wells"] == {"e": 0}
    assert block[4] == block[2] == block[3]
    assert block[0] == block[1] != block[2]
    with pytest.raises(BlocksError):
        partition_wells(well_ids, components, centroids, weights, blocks=1, min_component_wells=1)


def test_inactive_grid_and_shape_guards() -> None:
    with pytest.raises(BlocksError):
        cell_components(np.zeros(8, dtype=bool), (2, 2, 2))
    with pytest.raises(BlocksError):
        cell_components(np.ones(7, dtype=bool), (2, 2, 2))


@pytest.mark.skipif(not REAL_DECK.is_file(), reason="Model Z deck archive is not present")
def test_model_z_blocks(tmp_path: Path) -> None:
    with ZipFile(REAL_DECK) as archive:
        archive.extractall(tmp_path)
    geometry = read_deck_geometry(tmp_path)
    assert geometry.dimens == (91, 102, 59)
    bodies = sorted(geometry.component_sizes.values(), reverse=True)[:3]
    assert bodies == [143621, 52953, 41369]
    connectivity = json.loads(
        Path("deliverables/control_coverage_20260909/geology-extended-model-z.json").read_text()
    )
    payload = build_blocks(geometry, connectivity, blocks=6)
    assert len(payload["well_to_block"]) == 103
    assert sorted(payload["stats"]["block_well_counts"]) == [3, 11, 12, 14, 30, 33]
    assert payload["stats"]["weight_inside_blocks"] > 0.98
    assert payload["pressure_control"]["region_keyword"] == "FIP_C1"
