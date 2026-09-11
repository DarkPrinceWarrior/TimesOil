"""Typed, dependency-free contracts shared by AIOS control loops."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from math import isfinite
import re


_WELL_NAME = re.compile(r"[A-Za-z0-9_.-]+")


class ContractError(ValueError):
    """A typed AIOS contract is internally inconsistent."""


class WellRole(StrEnum):
    PRODUCER = "producer"
    INJECTOR = "injector"


class WellStatus(StrEnum):
    OPEN = "OPEN"
    SHUT = "SHUT"


class ControlTarget(StrEnum):
    OIL_RATE = "ORAT"
    LIQUID_RATE = "LRAT"
    WATER_INJECTION_RATE = "WRAT"


def _validate_month(value: date, field: str) -> None:
    if value.day != 1:
        raise ContractError(f"{field} must be the first day of a month")


def _validate_well_name(value: str) -> None:
    if not _WELL_NAME.fullmatch(value):
        raise ContractError(f"unsafe or empty well name: {value!r}")


@dataclass(frozen=True, slots=True)
class ControlAction:
    """One simulator-facing well control for one monthly step."""

    month: date
    well: str
    role: WellRole
    status: WellStatus
    target: ControlTarget
    value: float
    bhp_limit: float | None = None

    def __post_init__(self) -> None:
        _validate_month(self.month, "action month")
        _validate_well_name(self.well)
        if not isfinite(self.value) or self.value < 0:
            raise ContractError("control value must be finite and non-negative")
        if self.bhp_limit is not None and (
            isinstance(self.bhp_limit, bool) or not isfinite(self.bhp_limit) or self.bhp_limit <= 0
        ):
            raise ContractError("bhp_limit must be finite and positive")
        if self.status is WellStatus.SHUT and self.value != 0:
            raise ContractError("a shut well must have a zero target")
        producer_targets = {ControlTarget.OIL_RATE, ControlTarget.LIQUID_RATE}
        if self.role is WellRole.PRODUCER and self.target not in producer_targets:
            raise ContractError("producer control must target ORAT or LRAT")
        if (
            self.role is WellRole.INJECTOR
            and self.target is not ControlTarget.WATER_INJECTION_RATE
        ):
            raise ContractError("injector control must target water injection rate")

    def to_dict(self) -> dict[str, object]:
        return {"month": self.month.isoformat(), "well": self.well, "role": self.role.value,
                "status": self.status.value, "target": self.target.value, "value": self.value,
                **({"bhp_limit": self.bhp_limit} if self.bhp_limit is not None else {})}
