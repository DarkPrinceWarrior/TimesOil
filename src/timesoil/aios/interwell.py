"""Model Z adaptation of Track 1 allocation.py hydro_weights (eccbb39b).

Geological weights are a prior, not OPM transmissibilities. No future production
or pressure is used. Roles and allocation normalization follow each action.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

INTERWELL_FEATURES = ("allocated_injection_m3d", "neighbor_pressure_bar", "perm_md", "poro", "completed_net_thickness_m")


@dataclass(frozen=True)
class WellConnectivity:
    well_ids: tuple[str, ...]
    weights: np.ndarray
    static: np.ndarray
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.well_ids)
        if not count or len(set(self.well_ids)) != count:
            raise ValueError("connectivity requires unique ordered well IDs")
        for name, shape in (("weights", (count, count)), ("static", (count, 3))):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"invalid connectivity {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(np.diag(self.weights)) or not np.allclose(self.weights, self.weights.T):
            raise ValueError("geological weights must be symmetric with zero diagonal")
        if (self.static[:, 1] > 1).any():
            raise ValueError("porosity must be between zero and one")

    def features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Accept one field or flattened complete fields; never mix months."""
        state, action = np.asarray(state, float), np.asarray(action, float)
        count = len(self.well_ids)
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 3 or len(state) % count:
            raise ValueError("interwell inputs require complete ordered fields")
        fields = action.reshape(-1, count, 3)
        producers = ((fields[..., 1] != 2) & (fields[..., 2] > 0.5)).astype(float)
        injection = np.where((fields[..., 1] == 2) & (fields[..., 2] > 0.5), fields[..., 0], 0)
        denominator = producers @ self.weights
        allocated = np.divide(injection, denominator, out=np.zeros_like(injection), where=denominator > 0) @ self.weights.T
        allocated *= producers
        pressure = state.reshape(-1, count, 3)[..., 2]
        valid = pressure > 0
        pressure_weight = valid @ self.weights.T
        neighbor = np.divide((pressure * valid) @ self.weights.T, pressure_weight,
                             out=pressure.copy(), where=pressure_weight > 0)
        static = np.broadcast_to(self.static, (*fields.shape[:2], 3))
        return np.concatenate((allocated[..., None], neighbor[..., None], static), axis=2).reshape(-1, 5)

    def as_dict(self) -> dict[str, Any]:
        return dict(well_ids=list(self.well_ids), weights=self.weights.tolist(),
                    static=self.static.tolist(), provenance=self.provenance)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WellConnectivity":
        return cls(tuple(value["well_ids"]), value["weights"], value["static"], value["provenance"])

    @classmethod
    def from_source(cls, source: Path | str, well_ids: tuple[str, ...]) -> "WellConnectivity":
        from .opm import OpmFlowRunner
        from .opm_chdd import _expanded_record, _keyword_offsets, _read_deck_densities_and_start, _single_record

        with TemporaryDirectory(prefix="timesoil-geology-") as temporary:
            prepared = OpmFlowRunner().prepare(source, Path(temporary) / "case")
            unit, _, _, digest, connections, _, text = _read_deck_densities_and_start(prepared.input_dir)
            if unit != "METRIC":
                raise ValueError("Model Z geometry requires a METRIC deck")
            # Unsupported edits cannot silently invalidate the geological prior.
            for keyword in ("FAULTS", "MULTFLT", "MULTX", "MULTY", "EQUALS", "COPY", "MULTIPLY", "ADD"):
                if _keyword_offsets(text, keyword):
                    raise ValueError(f"geological prior does not yet support {keyword}")

            def array(keyword: str, shape: tuple[int, ...]) -> np.ndarray:
                values = np.asarray(_expanded_record(_single_record(text, keyword), keyword), float)
                if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
                    raise ValueError(f"invalid {keyword} grid array")
                return values.reshape(shape)

            nx, ny, nz = map(int, _single_record(text, "DIMENS"))
            shape = (nz, ny, nx)
            active = array("ACTNUM", shape) > 0
            px, py, pz = (array(name, shape) for name in ("PERMX", "PERMY", "PERMZ"))
            poro, ntg = array("PORO", shape), array("NTG", shape)
            multz = array("MULTZ", shape) if _keyword_offsets(text, "MULTZ") else np.ones(shape)
            coord = array("COORD", (ny + 1, nx + 1, 2, 3))
            zcorn = array("ZCORN", (nz, 2, ny, 2, nx, 2))
            labels = _components(active, px, py, pz, multz)
            positions, static, blocks, widths = [], [], [], []
            for well in well_ids:
                cells = [tuple(int(x) - 1 for x in key.split(",")) for key in connections.get(well, {})]
                if not cells:
                    raise ValueError(f"no active Model Z completions for {well}")
                centers, net, permeability, porosity, well_blocks = [], [], [], [], set()
                for i, j, k in cells:
                    depths = zcorn[k, :, j, :, i, :]
                    pillars = coord[j:j+2, i:i+2]
                    dz = pillars[..., 1, 2] - pillars[..., 0, 2]
                    if (np.abs(dz) < 1e-9).any():
                        raise ValueError("degenerate COORD pillar")
                    fraction = (depths - pillars[..., 0, 2]) / dz
                    xy = pillars[None, ..., 0, :2] + fraction[..., None] * (pillars[..., 1, :2] - pillars[..., 0, :2])
                    centers.append(np.r_[xy.mean(axis=(0, 1, 2)), depths.mean()])
                    thickness = float(np.mean(depths[1] - depths[0]))
                    if thickness < 0 or not 0 <= ntg[k, j, i] <= 1:
                        raise ValueError("invalid completion thickness or NTG")
                    net.append(thickness * ntg[k, j, i])
                    permeability.append(np.sqrt(max(px[k, j, i], 0) * max(py[k, j, i], 0)))
                    porosity.append(poro[k, j, i])
                    well_blocks.add(int(labels[k, j, i]))
                    widths.extend(np.linalg.norm(np.diff(xy.mean(axis=0), axis=axis), axis=-1).ravel() for axis in (0, 1))
                if sum(net) <= 0 or -1 in well_blocks:
                    raise ValueError(f"nonconducting completions for {well}")
                positions.append(np.average(centers, axis=0, weights=net))
                static.append([np.average(permeability, weights=net), np.average(porosity, weights=net), sum(net)])
                blocks.append(well_blocks)
            positions, static = np.asarray(positions), np.asarray(static)
            distance2 = np.sum((positions[:, None, :2] - positions[None, :, :2]) ** 2, axis=2)
            positive_widths = np.concatenate(widths)
            positive_widths = positive_widths[positive_widths > 0]
            if not len(positive_widths):
                raise ValueError("grid cell width is unavailable")
            floor_m = float(np.median(positive_widths))
            perm = static[:, 0]
            denominator = perm[:, None] + perm[None, :]
            harmonic = np.divide(2 * perm[:, None] * perm[None, :], denominator,
                                 out=np.zeros_like(denominator), where=denominator > 0)
            weights = harmonic / np.maximum(distance2, floor_m ** 2)
            weights *= np.array([[bool(left & right) for right in blocks] for left in blocks])
            np.fill_diagonal(weights, 0)
            return cls(well_ids, weights, static, {
                "source_sha256": prepared.source_sha256, "expanded_deck_sha256": digest,
                "method": "Track1 harmonic permeability / squared horizontal distance; monthly role normalization",
                "coordinates_m": positions.tolist(), "distance_floor_m": floor_m,
                "completion_components": [sorted(value) for value in blocks],
                "limitations": "Geological prior only: face adjacency with ACTNUM, directional permeability and MULTZ; PINCH/NNC and OPM transmissibilities are not reproduced.",
            })


def _components(active: np.ndarray, px: np.ndarray, py: np.ndarray,
                pz: np.ndarray, multz: np.ndarray) -> np.ndarray:
    """Connected active cells; directional zero permeability blocks a face."""
    labels = np.full(active.shape, -1, dtype=int)
    nz, ny, nx = active.shape
    component = 0
    for origin in zip(*np.nonzero(active)):
        if labels[origin] >= 0:
            continue
        labels[origin] = component
        queue = deque([origin])
        while queue:
            k, j, i = queue.popleft()
            for dk, dj, di, perm in ((0, 0, 1, px), (0, 0, -1, px), (0, 1, 0, py), (0, -1, 0, py), (1, 0, 0, pz), (-1, 0, 0, pz)):
                neighbor = k + dk, j + dj, i + di
                z, y, x = neighbor
                if not (0 <= z < nz and 0 <= y < ny and 0 <= x < nx):
                    continue
                if not active[neighbor] or labels[neighbor] >= 0 or min(perm[k, j, i], perm[neighbor]) <= 0:
                    continue
                if dk and multz[min(k, z), j, i] <= 0:
                    continue
                labels[neighbor] = component
                queue.append(neighbor)
        component += 1
    return labels
