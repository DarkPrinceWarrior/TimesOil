"""Canonical Track 2 scenario trajectory and its feature contract.

One ``ScenarioTrajectory`` is an indivisible simulator scenario: monthly states
and the controls that drive them. ``content_hash`` is the identity used for
grouped train/validation/test splitting and leakage checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd


STATE_FEATURES = ("oil_tpd", "liquid_tpd", "pressure_bar")
ACTION_FEATURES = ("control_value", "control_target_code", "status")
BHP_ACTION_FEATURES = (*ACTION_FEATURES, "bhp_limit")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


@dataclass(frozen=True)
class ScenarioTrajectory:
    """One indivisible simulator trajectory used for grouped splitting.

    ``states`` has shape ``[month, well, 3]`` and ``actions`` has shape
    ``[month, well, 3 or 4]``. Optional fourth feature is BHP bound in bar;
    zero means unspecified, never a measured pressure. Target codes are
    ORAT=0, LRAT=1 and WRAT=2. The
    action at index ``t`` drives ``states[t + 1]``.
    ``source_model`` must explicitly identify simulator provenance.
    """

    scenario_id: str
    source_model: str
    dates: pd.DatetimeIndex
    well_ids: tuple[str, ...]
    states: np.ndarray
    actions: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dates = pd.DatetimeIndex(self.dates)
        states = np.asarray(self.states, dtype=float)
        actions = np.asarray(self.actions, dtype=float)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)

        if not self.scenario_id or not self.source_model:
            raise ValueError("scenario_id and source_model are required")
        if len(dates) < 2 or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(f"scenario {self.scenario_id!r}: dates must be unique and increasing")
        periods = dates.to_period("M").astype(int)
        if not np.all(np.diff(periods) == 1):
            raise ValueError(f"scenario {self.scenario_id!r}: monthly trajectory has gaps")
        expected_states = (len(dates), len(self.well_ids), len(STATE_FEATURES))
        expected_actions = (len(dates), len(self.well_ids), 3)
        if states.shape != expected_states or actions.shape not in (
            expected_actions, (*expected_actions[:2], 4)
        ):
            raise ValueError(
                f"scenario {self.scenario_id!r}: expected states {expected_states} and "
                f"actions {expected_actions}, got {states.shape} and {actions.shape}"
            )
        if len(set(self.well_ids)) != len(self.well_ids):
            raise ValueError(f"scenario {self.scenario_id!r}: duplicate wells")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"scenario {self.scenario_id!r}: non-finite values")
        oil, liquid = states[..., 0], states[..., 1]
        if np.any(oil < 0) or np.any(liquid < oil) or np.any(states[..., 2] < 0):
            raise ValueError(f"scenario {self.scenario_id!r}: physical state invariants violated")
        if (
            np.any(actions[..., 0] < 0)
            or not np.isin(actions[..., 1], (0.0, 1.0, 2.0)).all()
            or not np.isin(actions[..., 2], (0.0, 1.0)).all()
            or (actions.shape[-1] == 4 and np.any(actions[..., 3] < 0))
        ):
            raise ValueError(f"scenario {self.scenario_id!r}: invalid control target/value/status")

    @property
    def content_hash(self) -> str:
        digest = sha256()
        digest.update(_canonical_json({
            "scenario_id": self.scenario_id,
            "source_model": self.source_model,
            "dates": [value.isoformat() for value in self.dates],
            "well_ids": self.well_ids,
            "metadata": self.metadata,
        }))
        digest.update(np.asarray(self.states, dtype="<f8").tobytes())
        digest.update(np.asarray(self.actions, dtype="<f8").tobytes())
        return digest.hexdigest()
