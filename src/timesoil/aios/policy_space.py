"""Box parameterisation of Track 2 field policies (design 20260911, section 3.2).

The searcher works on ``x in [0, 1]^d``; this module is the only place that maps
that box onto the ``policy`` schema consumed by
``scripts/propose_track2_policies.py:policy_controls``. It is deterministic and
free of randomness, so the same ``x`` always yields byte-identical policies.

Two deliberate semantics, forced by the policy schema:

* ``well_scales`` carries a single multiplier per well, so the first-half field
  multiplier is expressed relatively (the month profile of the source schedule is
  preserved) while the second half is expressed as an absolute rate through
  ``well_updates``. Absolute rates need a representative per-well baseline, which
  the caller injects as ``baseline_rates``.
* Injection targets are exact with respect to ``baseline_rates``, not with respect
  to the month-by-month source values. Residual cap violations on the forecast are
  the job of ``repair_policy`` (design section 1.3), not of the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_ROLES = ('producer', 'injector')
_BOX_TOLERANCE = 1e-9
_LRAT_CAP_M3D = 500.0
_SCALE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Group:
    """One contiguous block of search coordinates mapped onto a physical range."""

    name: str
    size: int
    low: float
    high: float
    unit: str


class PolicySpace:
    """Decode ``x in [0, 1]^d`` plus discrete LLM genes into a ``policy`` mapping."""

    def __init__(self, months: Sequence[str], well_roles: Mapping[str, str],
                 blocks: Mapping[str, str] | None = None,
                 caps: Mapping[str, float] | None = None,
                 baseline_rates: Mapping[str, float] | None = None,
                 water_cut: Mapping[str, float] | None = None) -> None:
        months = tuple(str(month) for month in months)
        if not months or len(set(months)) != len(months) or list(months) != sorted(months):
            raise ValueError('months must be unique and sorted')
        if not well_roles:
            raise ValueError('well_roles must not be empty')
        roles = {str(well): str(role) for well, role in well_roles.items()}
        unknown_roles = sorted({role for role in roles.values()} - set(_ROLES))
        if unknown_roles:
            raise ValueError(f'unknown well roles: {unknown_roles}')
        block_of = {well: 'field' for well in roles}
        if blocks is not None:
            unknown = sorted(set(blocks) - set(roles))
            if unknown:
                raise ValueError(f'blocks reference unknown wells: {unknown}')
            block_of.update({str(well): str(block) for well, block in blocks.items()})
        caps = dict(caps or {})
        liquid_cap = float(caps.get('liquid_cap_m3d', 0.0))
        injection_cap = float(caps.get('injection_cap_m3d', 0.0))
        if not np.isfinite(liquid_cap) or not np.isfinite(injection_cap) or min(liquid_cap, injection_cap) <= 0:
            raise ValueError('caps must carry positive finite liquid_cap_m3d and injection_cap_m3d')
        rates = {str(well): float(rate) for well, rate in (baseline_rates or {}).items()}
        unknown = sorted(set(rates) - set(roles))
        if unknown:
            raise ValueError(f'baseline_rates reference unknown wells: {unknown}')
        if any(not np.isfinite(rate) or rate < 0 for rate in rates.values()):
            raise ValueError('baseline_rates must be finite and nonnegative')
        cuts = {str(well): float(cut) for well, cut in (water_cut or {}).items()}
        unknown = sorted(set(cuts) - set(roles))
        if unknown:
            raise ValueError(f'water_cut references unknown wells: {unknown}')

        self.months = months
        self.well_roles = roles
        self.block_of = block_of
        self.block_ids = tuple(sorted(set(block_of.values())))
        self.liquid_cap_m3d = liquid_cap
        self.injection_cap_m3d = injection_cap
        self.baseline_rates = rates
        self.water_cut = cuts
        self.half_index = len(months) // 2
        blocks_n = len(self.block_ids)
        self.groups = (
            Group('field_producer', 2, 0.6, 1.2, 'multiplier'),
            Group('field_injection_share', 2, 0.5, 1.0, 'share of injection_cap'),
            Group('block_producer', blocks_n, 0.5, 1.3, 'multiplier'),
            Group('block_injection_share', blocks_n, 0.0, 1.0, 'raw share'),
            Group('watercut_shut', 1, 0.90, 0.995, 'fraction'),
            Group('producer_bhp_add', 1, 0.0, 30.0, 'bar'),
        )
        self.dim = sum(group.size for group in self.groups)

    # ------------------------------------------------------------------ box

    def _slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        start = 0
        for group in self.groups:
            out[group.name] = slice(start, start + group.size)
            start += group.size
        return out

    def unpack(self, x: Sequence[float] | np.ndarray) -> dict[str, np.ndarray]:
        """Map a box vector onto physical parameter arrays, one entry per group."""
        vector = np.asarray(x, dtype=float).reshape(-1)
        if vector.size != self.dim:
            raise ValueError(f'x must have {self.dim} coordinates, got {vector.size}')
        if not np.all(np.isfinite(vector)):
            raise ValueError('x must be finite')
        if np.any(vector < -_BOX_TOLERANCE) or np.any(vector > 1.0 + _BOX_TOLERANCE):
            raise ValueError('x must lie in the unit box')
        vector = np.clip(vector, 0.0, 1.0)
        slices = self._slices()
        return {group.name: group.low + (group.high - group.low) * vector[slices[group.name]]
                for group in self.groups}

    def block_injection_shares(self, raw: np.ndarray, eligible: Sequence[str]) -> dict[str, float]:
        """Normalise raw block shares to sum to one over blocks that can take water."""
        index = {block: position for position, block in enumerate(self.block_ids)}
        weights = {block: max(float(raw[index[block]]), 0.0) for block in eligible}
        total = sum(weights.values())
        if not eligible:
            return {}
        if total <= 0.0:
            return {block: 1.0 / len(eligible) for block in eligible}
        return {block: weight / total for block, weight in weights.items()}

    # ----------------------------------------------------------------- genes

    def validate_genes(self, genes: Mapping[str, Any] | None) -> dict[str, Any]:
        """Reject unknown wells, unknown keys and reverse conversions before decoding."""
        if genes is None:
            return {'shut_wells': [], 'well_updates': [], 'overrides': {}}
        if not isinstance(genes, Mapping):
            raise ValueError('genes must be a mapping')
        allowed = {'shut_wells', 'well_updates', 'overrides'}
        extra = sorted(set(genes) - allowed)
        if extra:
            raise ValueError(f'unknown gene keys: {extra}')
        shut = [str(well) for well in genes.get('shut_wells', [])]
        unknown = sorted(set(shut) - set(self.well_roles))
        if unknown:
            raise ValueError(f'genes shut unknown wells: {unknown}')
        overrides_raw = genes.get('overrides', {}) or {}
        if not isinstance(overrides_raw, Mapping):
            raise ValueError('genes overrides must be a mapping')
        overrides: dict[str, float] = {}
        for well, scale in overrides_raw.items():
            well = str(well)
            if well not in self.well_roles:
                raise ValueError(f'genes override unknown well: {well}')
            value = float(scale)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'genes override for {well} must be finite and positive')
            overrides[well] = value
        updates: list[dict[str, Any]] = []
        months = set(self.months)
        for update in genes.get('well_updates', []) or []:
            if not isinstance(update, Mapping) or not {'well', 'start', 'end'} <= set(update):
                raise ValueError('gene well update needs well, start and end')
            well, start, end = str(update['well']), str(update['start']), str(update['end'])
            if well not in self.well_roles:
                raise ValueError(f'gene well update on unknown well: {well}')
            if start not in months or end not in months or start > end:
                raise ValueError(f'gene well update on {well} is outside the month grid')
            role = update.get('role')
            if role is not None:
                if str(role) not in _ROLES:
                    raise ValueError(f'gene well update on {well} carries an unknown role')
                if str(role) == 'producer' and self.well_roles[well] == 'injector':
                    raise ValueError(f'reverse conversion requested for {well}')
            item = {key: value for key, value in update.items()}
            item.update(well=well, start=start, end=end)
            updates.append(item)
        updates.sort(key=lambda item: (item['well'], item['start'], item['end']))
        return {'shut_wells': sorted(set(shut)), 'well_updates': updates, 'overrides': overrides}

    # ---------------------------------------------------------------- decode

    def decode(self, x: Sequence[float] | np.ndarray,
               genes: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return a ``policy`` mapping in the exact ``policy_controls`` schema."""
        params = self.unpack(x)
        checked = self.validate_genes(genes)
        threshold = float(params['watercut_shut'][0])
        shut = set(checked['shut_wells'])
        shut.update(well for well, cut in self.water_cut.items()
                    if self.well_roles[well] == 'producer' and cut >= threshold)

        producer_scales = self._producer_scales(params, shut, checked['overrides'])
        injector_scales = self._injector_scales(params, shut, checked['overrides'])
        scales = {**producer_scales, **injector_scales}

        gene_wells = {item['well'] for item in checked['well_updates']}
        second_half = self._second_half_updates(scales, shut | gene_wells)
        well_updates = [*second_half, *checked['well_updates']]
        well_updates.sort(key=lambda item: (item['well'], item['start'], item['end']))

        return {
            'producer_scale': 1.0,
            'injector_scale': 1.0,
            'well_scales': [{'well': well, 'scale': round(pair[0], 12)}
                            for well, pair in sorted(scales.items()) if well not in shut],
            'shut_wells': sorted(shut),
            'producer_bhp_add': round(float(params['producer_bhp_add'][0]), 12),
            'well_updates': well_updates,
        }

    def _producer_scales(self, params: dict[str, np.ndarray], shut: set[str],
                         overrides: Mapping[str, float]) -> dict[str, tuple[float, float]]:
        index = {block: position for position, block in enumerate(self.block_ids)}
        field = params['field_producer']
        block_mult = params['block_producer']
        out: dict[str, tuple[float, float]] = {}
        for well, role in self.well_roles.items():
            if role != 'producer' or well in shut:
                continue
            multiplier = float(block_mult[index[self.block_of[well]]]) * overrides.get(well, 1.0)
            out[well] = (float(field[0]) * multiplier, float(field[1]) * multiplier)
        return out

    def _injector_scales(self, params: dict[str, np.ndarray], shut: set[str],
                         overrides: Mapping[str, float]) -> dict[str, tuple[float, float]]:
        """Split the injection budget over blocks, then over wells by baseline WRAT."""
        capacity: dict[str, float] = {}
        members: dict[str, list[str]] = {}
        for well, role in self.well_roles.items():
            if role != 'injector' or well in shut:
                continue
            rate = self.baseline_rates.get(well, 0.0)
            if rate <= 0:
                continue
            block = self.block_of[well]
            capacity[block] = capacity.get(block, 0.0) + rate
            members.setdefault(block, []).append(well)
        eligible = sorted(capacity)
        shares = self.block_injection_shares(params['block_injection_share'], eligible)
        field = params['field_injection_share']
        targets = (float(field[0]) * self.injection_cap_m3d, float(field[1]) * self.injection_cap_m3d)
        out: dict[str, tuple[float, float]] = {}
        for block in eligible:
            block_share = shares[block]
            base = capacity[block]
            for well in members[block]:
                factor = overrides.get(well, 1.0)
                out[well] = (targets[0] * block_share / base * factor,
                             targets[1] * block_share / base * factor)
        return out

    def _second_half_updates(self, scales: Mapping[str, tuple[float, float]],
                             skip: set[str]) -> list[dict[str, Any]]:
        """Express the second-half multiplier as absolute rates, the only interval form available."""
        if self.half_index >= len(self.months):
            return []
        start, end = self.months[self.half_index], self.months[-1]
        out: list[dict[str, Any]] = []
        for well, (first, second) in sorted(scales.items()):
            if well in skip or abs(second - first) <= _SCALE_TOLERANCE:
                continue
            base = self.baseline_rates.get(well)
            if base is None:
                raise ValueError(f'second-half multiplier for {well} needs a baseline rate')
            value = base * second
            if self.well_roles.get(well) == 'producer':
                value = min(value, _LRAT_CAP_M3D)  # the same clip policy_controls applies to first-half scales
            out.append({'well': well, 'start': start, 'end': end,
                        'value': round(value, 12)})
        return out

    # ----------------------------------------------------------- warm starts

    def encode_seed(self, policy_like: Mapping[str, Any]) -> np.ndarray:
        """Invert named parameters into the box; unknown parameters land on the midpoint."""
        allowed = {group.name for group in self.groups}
        extra = sorted(set(policy_like) - allowed)
        if extra:
            raise ValueError(f'encode_seed accepts only {sorted(allowed)}, got extra {extra}')
        x = np.full(self.dim, 0.5)
        slices = self._slices()
        index = {block: position for position, block in enumerate(self.block_ids)}
        for group in self.groups:
            if group.name not in policy_like:
                continue
            raw = policy_like[group.name]
            if isinstance(raw, Mapping):
                values = np.full(group.size, np.nan)
                for block, value in raw.items():
                    if str(block) not in index:
                        raise ValueError(f'encode_seed got unknown block {block}')
                    values[index[str(block)]] = float(value)
                if group.name == 'block_injection_share':
                    finite = values[np.isfinite(values)]
                    peak = float(finite.max()) if finite.size and float(finite.max()) > 0 else 1.0
                    values = values / peak
            else:
                values = np.asarray(raw, dtype=float).reshape(-1)
                if values.size == 1:
                    values = np.full(group.size, float(values[0]))
                if values.size != group.size:
                    raise ValueError(f'encode_seed got {values.size} values for {group.name}')
            span = group.high - group.low
            scaled = (values - group.low) / span if span > 0 else np.zeros(group.size)
            known = np.isfinite(scaled)
            x[slices[group.name]] = np.where(known, np.clip(scaled, 0.0, 1.0), 0.5)
        return x

    # ------------------------------------------------------- interpretability

    def describe(self, x: Sequence[float] | np.ndarray) -> list[dict[str, Any]]:
        """Return one row per search coordinate plus the derived injection targets."""
        params = self.unpack(x)
        rows: list[dict[str, Any]] = []
        position = 0
        for group in self.groups:
            labels = self._labels(group)
            for offset, label in enumerate(labels):
                rows.append({'index': position + offset, 'group': group.name, 'name': label,
                             'value': round(float(params[group.name][offset]), 6),
                             'unit': group.unit, 'low': group.low, 'high': group.high})
            position += group.size
        shares = self.block_injection_shares(params['block_injection_share'], self.block_ids)
        for half, share in enumerate(params['field_injection_share']):
            rows.append({'index': None, 'group': 'derived', 'name': f'injection_target_half{half + 1}',
                         'value': round(float(share) * self.injection_cap_m3d, 6), 'unit': 'm3/d',
                         'low': 0.0, 'high': self.injection_cap_m3d})
        for block in self.block_ids:
            rows.append({'index': None, 'group': 'derived', 'name': f'injection_share_{block}',
                         'value': round(shares.get(block, 0.0), 6), 'unit': 'normalised share',
                         'low': 0.0, 'high': 1.0})
        rows.append({'index': None, 'group': 'derived', 'name': 'liquid_cap_m3d',
                     'value': self.liquid_cap_m3d, 'unit': 'm3/d', 'low': 0.0,
                     'high': self.liquid_cap_m3d})
        return rows

    def _labels(self, group: Group) -> tuple[str, ...]:
        if group.size == 1:
            return (group.name,)
        if group.name in ('block_producer', 'block_injection_share'):
            return tuple(f'{group.name}[{block}]' for block in self.block_ids)
        return tuple(f'{group.name}[half{half + 1}]' for half in range(group.size))


def block_map(well_ids: Iterable[str], assignment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Fill a well to block map, defaulting every unassigned well to a single block."""
    assignment = assignment or {}
    return {str(well): str(assignment.get(str(well), 'field')) for well in well_ids}
