"""Canonical WCONPROD/WCONINJE record rendering for the supported subset."""

from __future__ import annotations

from .contracts import ControlAction, ControlTarget


def producer_line(action: ControlAction) -> str:
    target = action.target.value
    value = f"{action.value:.6f}"
    controls = value if action.target is ControlTarget.OIL_RATE else f"3* {value}"
    if action.bhp_limit is not None:
        controls += f" {'4*' if action.target is ControlTarget.OIL_RATE else '1*'} {action.bhp_limit:.6f}"
    return f"  '{action.well}' '{action.status.value}' '{target}' {controls} /"


def injector_line(action: ControlAction) -> str:
    pressure = "" if action.bhp_limit is None else f" 1* {action.bhp_limit:.6f}"
    return (
        f"  '{action.well}' 'WATER' '{action.status.value}' "
        f"'RATE' {action.value:.6f}{pressure} /"
    )
