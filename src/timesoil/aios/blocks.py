"""Geometry-derived blocks of Model Z.

The deck declares no ``FAULTS``/``MULTFLT``, yet the grid is not one body: active
cells split into face-connected components and ``ZCORN`` offsets remove contact
between face-adjacent columns. Blocks are therefore derived from geometry first
(connected component of the majority of a well's completions), and only inside a
component from Ward clustering of completion coordinates. Connectivity weights
are a prior used for merging under-sized components and for reporting, never a
substitute for the deck geometry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .opm_chdd import (
    _expanded_integers,
    _expanded_record,
    _keyword_offsets,
    _read_deck_densities_and_start,
    _records_after,
    _single_record,
    _tokens,
)

SCHEMA = "timesoil.blocks/1"
REGION_KEYWORDS: tuple[str, ...] = ("FIPNUM", "FIP_C1", "FIP_ZONE", "EQLNUM", "SATNUM", "PVTNUM")
"""Region cubes parsed when present; the first one present also drives K6 pressure control."""
PRESSURE_REGION_ORDER: tuple[str, ...] = ("FIPNUM", "FIP_C1", "FIP_ZONE", "EQLNUM")
_NEIGHBOURS = 5


class BlocksError(ValueError):
    """Raised when the deck, the connectivity prior or the parameters do not admit blocks."""


@dataclass(frozen=True)
class DeckGeometry:
    """Active-cell bodies and well completions of one deck, in global cell order."""

    dimens: tuple[int, int, int]
    active: np.ndarray
    labels: np.ndarray
    regions: dict[str, np.ndarray]
    well_cells: dict[str, np.ndarray]
    welspecs_ij: dict[str, tuple[int, int]]
    deck_sha256: str
    adjacency: dict[str, Any]

    @property
    def component_sizes(self) -> dict[int, int]:
        counts = Counter(int(value) for value in self.labels[self.active])
        return dict(sorted(counts.items()))


def _significant(value: float, digits: int = 6) -> float:
    return float(f"{float(value):.{digits}g}")


def read_zcorn(text: str, dimens: tuple[int, int, int]) -> np.ndarray | None:
    """Return corner depths shaped ``(nz, 2, ny, 2, nx, 2)`` or ``None`` when absent."""

    matches = _keyword_offsets(text, "ZCORN")
    if not matches:
        return None
    if len(matches) != 1:
        raise BlocksError(f"deck must contain at most one ZCORN, found {len(matches)}")
    end = text.find("/", matches[0].end())
    if end < 0:
        raise BlocksError("unterminated ZCORN keyword")
    chunk = text[matches[0].end() : end]
    nx, ny, nz = dimens
    expected = 8 * nx * ny * nz
    if "*" in chunk:
        values = np.array(
            _expanded_record(list(_tokens(chunk)), "ZCORN"), dtype=float
        )  # repeat counts are rare here and never worth a second parser
    else:
        values = np.fromstring(chunk, sep=" ")
    if values.size != expected:
        raise BlocksError(f"ZCORN contains {values.size} corners, expected {expected}")
    if not np.isfinite(values).all():
        raise BlocksError("non-finite ZCORN corner depth")
    return values.reshape(nz, 2, ny, 2, nx, 2)


def _face_overlap(zcorn: np.ndarray, axis: int) -> np.ndarray:
    """Positive vertical overlap of the shared face, per lateral cell pair."""

    if axis == 2:
        top = np.maximum(zcorn[:, 0, :, :, :-1, 1], zcorn[:, 0, :, :, 1:, 0])
        bottom = np.minimum(zcorn[:, 1, :, :, :-1, 1], zcorn[:, 1, :, :, 1:, 0])
        return (bottom - top > 0).any(axis=2)
    top = np.maximum(zcorn[:, 0, :-1, 1, :, :], zcorn[:, 0, 1:, 0, :, :])
    bottom = np.minimum(zcorn[:, 1, :-1, 1, :, :], zcorn[:, 1, 1:, 0, :, :])
    return (bottom - top > 0).any(axis=-1)


def cell_components(
    active: np.ndarray, dimens: tuple[int, int, int], zcorn: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Label face-connected bodies of active cells; ``ZCORN`` removes non-overlapping faces.

    Returns global-order labels (``-1`` on inactive cells) and an adjacency report.
    """

    nx, ny, nz = dimens
    if active.shape != (nx * ny * nz,) or active.dtype != bool:
        raise BlocksError("ACTNUM must be a boolean array in global cell order")
    if not active.any():
        raise BlocksError("deck has no active cells")
    grid = active.reshape(nz, ny, nx)
    index = np.full(grid.shape, -1, dtype=np.int64)
    count = int(grid.sum())
    index[grid] = np.arange(count)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    report: dict[str, Any] = {"zcorn_overlap": zcorn is not None}
    for axis, name in ((2, "i"), (1, "j"), (0, "k")):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        touching = grid[tuple(lower)] & grid[tuple(upper)]
        report[f"{name}_faces"] = int(touching.sum())
        if zcorn is not None and axis != 0:
            overlap = _face_overlap(zcorn, axis)
            if overlap.shape != touching.shape:
                raise BlocksError("ZCORN shape does not match DIMENS")
            report[f"{name}_faces_without_overlap"] = int((touching & ~overlap).sum())
            touching &= overlap
        rows.append(index[tuple(lower)][touching])
        columns.append(index[tuple(upper)][touching])
    row, column = np.concatenate(rows), np.concatenate(columns)
    graph = coo_matrix((np.ones(row.size), (row, column)), shape=(count, count))
    total, local = connected_components(graph, directed=False)
    labels = np.full(active.size, -1, dtype=np.int64)
    labels[active] = local
    report["components"] = int(total)
    return labels, report


def read_deck_geometry(deck_dir: str | Path, zcorn_overlap: bool = True) -> DeckGeometry:
    """Parse DIMENS, ACTNUM, ZCORN, region cubes, WELSPECS and COMPDAT of one deck."""

    _, _, _, deck_sha256, connections, _, text = _read_deck_densities_and_start(deck_dir)
    dimens = tuple(int(float(value)) for value in _single_record(text, "DIMENS")[:3])
    if len(dimens) != 3 or min(dimens) < 1:
        raise BlocksError("DIMENS must contain positive NX NY NZ")
    nx, ny, nz = dimens
    active = np.array(_expanded_integers(_single_record(text, "ACTNUM"), "ACTNUM"), dtype=bool)
    zcorn = read_zcorn(text, dimens) if zcorn_overlap else None
    labels, adjacency = cell_components(active, dimens, zcorn)
    regions: dict[str, np.ndarray] = {}
    for keyword in REGION_KEYWORDS:
        if not _keyword_offsets(text, keyword):
            continue
        values = np.array(_expanded_integers(_single_record(text, keyword), keyword), dtype=np.int64)
        if values.size != active.size:
            raise BlocksError(f"{keyword} contains {values.size} cells, expected {active.size}")
        regions[keyword] = values
    welspecs_ij: dict[str, tuple[int, int]] = {}
    for match in _keyword_offsets(text, "WELSPECS"):
        for record in _records_after(text, match, None):
            welspecs_ij[record[0].strip()] = (int(float(record[2])), int(float(record[3])))
    well_cells: dict[str, np.ndarray] = {}
    for well, cells in sorted(connections.items()):
        indices = []
        for key in cells:
            i, j, k = (int(value) for value in key.split(","))
            indices.append((k - 1) * nx * ny + (j - 1) * nx + i - 1)
        cell_index = np.unique(np.asarray(indices, dtype=np.int64))
        if not cell_index.size or not active[cell_index].all():
            raise BlocksError(f"well {well!r} has no active completion cells")
        well_cells[well] = cell_index
    return DeckGeometry(dimens, active, labels, regions, well_cells, welspecs_ij, deck_sha256, adjacency)


def well_membership(geometry: DeckGeometry, well: str) -> dict[str, Any]:
    """Component and region membership of one well, with completion counts."""

    nx, ny, _ = geometry.dimens
    cells = geometry.well_cells[well]
    lateral = np.column_stack((cells % nx + 1, cells // nx % ny + 1))
    components = Counter(int(value) for value in geometry.labels[cells])
    regions = {
        keyword: dict(sorted(Counter(int(value) for value in values[cells]).items()))
        for keyword, values in geometry.regions.items()
    }
    return {
        "completions": int(cells.size),
        "component": _dominant(components),
        "components": dict(sorted(components.items())),
        "centroid_ij": [_significant(value) for value in lateral.mean(axis=0)],
        "welspecs_ij": list(geometry.welspecs_ij.get(well, ())),
        "regions": regions,
    }


def _dominant(counts: Mapping[int, int]) -> int:
    """Most completed key; ties resolved by the smaller key, so the result is order-free."""

    if not counts:
        raise BlocksError("empty membership counter")
    return min(counts, key=lambda key: (-counts[key], key))


def _allocate(sizes: Sequence[int], blocks: int) -> list[int]:
    """One block per group, the rest by D'Hondt largest average; ties to the first group."""

    if blocks < len(sizes):
        raise BlocksError(
            f"{blocks} blocks cannot cover {len(sizes)} geometric groups; "
            "raise --blocks or --min-component-wells"
        )
    allocation = [1] * len(sizes)
    for _ in range(blocks - len(sizes)):
        best = max(range(len(sizes)), key=lambda index: (sizes[index] / (allocation[index] + 1), -index))
        allocation[best] += 1
    return allocation


def partition_wells(
    well_ids: Sequence[str],
    components: Sequence[int],
    centroids: np.ndarray,
    weights: np.ndarray,
    blocks: int = 6,
    min_component_wells: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Geometry-first partition: components never share a block unless merged for size.

    Wells of a component holding fewer than ``min_component_wells`` wells join the
    component with the strongest summed connectivity weight; inside every remaining
    component Ward clustering of ``centroids`` splits the allocated blocks. The result
    does not depend on the order of ``well_ids``.
    """

    count = len(well_ids)
    if count != len(set(well_ids)) or not count:
        raise BlocksError("partition requires unique ordered well IDs")
    if len(components) != count or centroids.shape != (count, 2) or weights.shape != (count, count):
        raise BlocksError("partition inputs must share the well order")
    if blocks < 1 or min_component_wells < 1:
        raise BlocksError("blocks and min_component_wells must be positive")
    if blocks > count:
        raise BlocksError(f"{blocks} blocks exceed {count} wells")
    order = sorted(range(count), key=lambda index: well_ids[index])
    component = np.asarray(components, dtype=np.int64)
    population = Counter(int(value) for value in component)
    large = sorted(key for key, value in population.items() if value >= min_component_wells)
    if not large:
        raise BlocksError(
            f"no component holds {min_component_wells} wells; lower --min-component-wells"
        )
    group = component.copy()
    merged: dict[str, int] = {}
    for index in order:
        if population[int(component[index])] >= min_component_wells:
            continue
        host = max(
            large,
            key=lambda key: (float(weights[index][component == key].sum()), -key),
        )
        group[index] = host
        merged[well_ids[index]] = host
    sizes = [int((group == key).sum()) for key in large]
    allocation = _allocate(sizes, blocks)
    cluster = np.full(count, -1, dtype=np.int64)
    label = 0
    for key, share in zip(large, allocation, strict=True):
        members = [index for index in order if group[index] == key]
        if share == 1 or len(members) <= share:
            local = np.zeros(len(members), dtype=np.int64) if share == 1 else np.arange(len(members))
        else:
            local = fcluster(linkage(centroids[members], "ward"), share, "maxclust") - 1
        for offset, index in enumerate(members):
            cluster[index] = label + int(local[offset])
        label += share
    if (cluster < 0).any():
        raise BlocksError("partition left a well unassigned")
    keys = {
        int(value): (
            int(_dominant(Counter(int(item) for item in group[cluster == value]))),
            *(_significant(item) for item in centroids[cluster == value].mean(axis=0)),
        )
        for value in np.unique(cluster)
    }
    ranking = {value: rank for rank, value in enumerate(sorted(keys, key=lambda value: keys[value]))}
    block = np.array([ranking[int(value)] for value in cluster], dtype=np.int64)
    return block, {"group": group.tolist(), "merged_wells": dict(sorted(merged.items()))}


def _top_neighbours(weights: np.ndarray, index: int) -> list[int]:
    row = weights[index].copy()
    row[index] = 0.0
    candidates = np.argsort(-row, kind="stable")[:_NEIGHBOURS]
    return [int(value) for value in candidates if row[value] > 0]


def build_blocks(
    geometry: DeckGeometry,
    connectivity: Mapping[str, Any],
    blocks: int = 6,
    min_component_wells: int = 3,
    connectivity_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the deterministic ``blocks.json`` payload for K blocks."""

    well_ids = [str(value) for value in connectivity["well_ids"]]
    weights = np.asarray(connectivity["weights"], dtype=float)
    if sorted(well_ids) != sorted(geometry.well_cells):
        raise BlocksError("connectivity well IDs differ from the deck COMPDAT wells")
    if weights.shape != (len(well_ids), len(well_ids)) or not np.isfinite(weights).all() or (weights < 0).any():
        raise BlocksError("connectivity weights must be a finite non-negative square matrix")
    membership = {well: well_membership(geometry, well) for well in sorted(well_ids)}
    order = sorted(range(len(well_ids)), key=lambda index: well_ids[index])
    ordered = [well_ids[index] for index in order]
    permuted = weights[np.ix_(order, order)]
    centroids = np.array([membership[well]["centroid_ij"] for well in ordered], dtype=float)
    components = [membership[well]["component"] for well in ordered]
    block, detail = partition_wells(
        ordered, components, centroids, permuted, blocks, min_component_wells
    )
    assignment = {well: int(value) for well, value in zip(ordered, block, strict=True)}
    for well in ordered:
        membership[well]["block"] = assignment[well]
    pressure_keyword = next((name for name in PRESSURE_REGION_ORDER if name in geometry.regions), None)
    block_entries: list[dict[str, Any]] = []
    for value in range(blocks):
        members = [well for well in ordered if assignment[well] == value]
        if not members:
            raise BlocksError(f"block {value} is empty")
        regions: dict[str, dict[str, int]] = {}
        for keyword in geometry.regions:
            counter: Counter[int] = Counter()
            for well in members:
                counter.update(membership[well]["regions"][keyword])
            regions[keyword] = {str(key): int(counter[key]) for key in sorted(counter)}
        centroid = centroids[[ordered.index(well) for well in members]].mean(axis=0)
        block_entries.append(
            {
                "id": value,
                "wells": members,
                "component": _dominant(Counter(membership[well]["component"] for well in members)),
                "fip_regions": regions,
                "centroid_ij": [_significant(item) for item in centroid],
            }
        )
    same = block[:, None] == block[None, :]
    np.fill_diagonal(same, False)
    total = float(permuted.sum())
    component_array = np.asarray(components)
    boundary: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    inside_top = 0.0
    for index, well in enumerate(ordered):
        neighbours = _top_neighbours(permuted, index)
        if neighbours:
            inside_top += float(np.mean([block[other] == block[index] for other in neighbours]))
        for other in neighbours:
            if block[other] == block[index]:
                continue
            pair = tuple(sorted((well, ordered[other])))
            if pair in seen:
                continue
            seen.add(pair)
            boundary.append(
                {
                    "wells": list(pair),
                    "blocks": sorted((assignment[pair[0]], assignment[pair[1]])),
                    "weight": _significant(permuted[index, other]),
                }
            )
    boundary.sort(key=lambda entry: (-entry["weight"], entry["wells"]))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "deck_sha256": geometry.deck_sha256,
        "connectivity_sha256": connectivity_sha256,
        "method": (
            "connected components of active cells by face adjacency (ZCORN vertical overlap "
            "required on lateral faces), dominant component per well by completion count, "
            "components under the size floor merged by strongest summed connectivity weight, "
            "Ward clustering of completion-mean (i,j) inside each remaining component"
        ),
        "parameters": {
            "blocks": blocks,
            "min_component_wells": min_component_wells,
            "linkage": "ward",
            "coordinates": "completion-mean i,j",
            "neighbours": _NEIGHBOURS,
            **dict(sorted((extra or {}).items())),
        },
        "grid": {
            "dimens": list(geometry.dimens),
            "active_cells": int(geometry.active.sum()),
            **{key: geometry.adjacency[key] for key in sorted(geometry.adjacency)},
        },
        "components": [
            {
                "id": key,
                "cells": size,
                "k_range": _k_range(geometry, key),
                "wells": [well for well in ordered if membership[well]["component"] == key],
            }
            for key, size in geometry.component_sizes.items()
            if size
        ],
        "blocks": block_entries,
        "well_to_block": dict(sorted(assignment.items())),
        "wells": {well: membership[well] for well in sorted(ordered)},
        "merged_wells": detail["merged_wells"],
        "boundary_pairs": boundary,
        "stats": {
            "weight_inside_blocks": _significant(float(permuted[same].sum()) / total) if total else 0.0,
            "top_neighbours_inside_blocks": _significant(inside_top / len(ordered)),
            "cross_component_weight_share": _significant(
                float(permuted[component_array[:, None] != component_array[None, :]].sum()) / total
            )
            if total
            else 0.0,
            "block_well_counts": [len(entry["wells"]) for entry in block_entries],
            "wells_in_several_components": sum(
                1 for well in ordered if len(membership[well]["components"]) > 1
            ),
        },
    }
    if pressure_keyword is not None:
        dominant = {
            well: _dominant(membership[well]["regions"][pressure_keyword]) for well in ordered
        }
        payload["pressure_control"] = {
            "region_keyword": pressure_keyword,
            "regions_available": sorted(geometry.regions),
            "well_to_region": {well: dominant[well] for well in sorted(dominant)},
            "region_to_wells": {
                str(value): sorted(well for well in ordered if dominant[well] == value)
                for value in sorted(set(dominant.values()))
            },
            "block_to_regions": {
                str(entry["id"]): {
                    str(value): sum(1 for well in entry["wells"] if dominant[well] == value)
                    for value in sorted({dominant[well] for well in entry["wells"]})
                }
                for entry in block_entries
            },
        }
    return payload


def _k_range(geometry: DeckGeometry, component: int) -> list[int]:
    nx, ny, _ = geometry.dimens
    cells = np.flatnonzero(geometry.labels == component)
    layers = cells // (nx * ny) + 1
    return [int(layers.min()), int(layers.max())]
