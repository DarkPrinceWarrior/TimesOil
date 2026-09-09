"""Validated full-field geological connectivity imported from OPM exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
        if (state.ndim != 2 or state.shape[1] != 3 or action.ndim != 2
                or action.shape[0] != len(state) or action.shape[1] not in (3, 4) or len(state) % count):
            raise ValueError("interwell inputs require complete ordered fields")
        fields = action[..., :3].reshape(-1, count, 3)
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

