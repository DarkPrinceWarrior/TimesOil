"""PolicySpace decodes the unit box into the exact policy_controls schema."""

from __future__ import annotations

import numpy as np
import pytest

from timesoil.aios.policy_space import PolicySpace, block_map
from propose_track2_policies import policy_controls

MONTHS = [f'2007-{month:02d}-01' for month in range(1, 13)]
ROLES = {'P1': 'producer', 'P2': 'producer', 'P3': 'producer',
         'I1': 'injector', 'I2': 'injector', 'I3': 'injector'}
BLOCKS = {'P1': 'A', 'P2': 'A', 'I1': 'A', 'I2': 'A', 'P3': 'B', 'I3': 'B'}
RATES = {'P1': 50.0, 'P2': 60.0, 'P3': 70.0, 'I1': 100.0, 'I2': 200.0, 'I3': 300.0}
CAPS = {'liquid_cap_m3d': 600.0, 'injection_cap_m3d': 600.0}


def space(water_cut: dict[str, float] | None = None) -> PolicySpace:
    return PolicySpace(MONTHS, ROLES, BLOCKS, CAPS, RATES,
                       water_cut if water_cut is not None else {'P1': 0.99, 'P2': 0.5, 'P3': 0.0})


def source_controls() -> list[dict[str, object]]:
    return [{'month': month, 'well': well, 'role': role, 'status': 'OPEN',
             'target': 'WRAT' if role == 'injector' else 'LRAT',
             'value': RATES[well], 'bhp_limit': 120.0 if role == 'injector' else 60.0}
            for month in MONTHS for well, role in sorted(ROLES.items())]


def scale_of(policy: dict[str, object], well: str) -> float:
    return next(item['scale'] for item in policy['well_scales'] if item['well'] == well)


def test_dimension_and_default_block() -> None:
    assert space().dim == 10
    flat = PolicySpace(MONTHS, ROLES, block_map(ROLES), CAPS, RATES)
    assert flat.block_ids == ('field',) and flat.dim == 8


def test_injection_targets_sum_to_share_times_cap() -> None:
    model = space(water_cut={})
    policy = model.decode(np.full(model.dim, 0.5))
    injected = sum(RATES[item['well']] * item['scale'] for item in policy['well_scales']
                   if ROLES[item['well']] == 'injector')
    assert injected == pytest.approx(0.75 * CAPS['injection_cap_m3d'])
    # Equal raw genes normalise to equal block shares: A and B hold 300 m3/d of baseline each.
    assert scale_of(policy, 'I1') == pytest.approx(scale_of(policy, 'I3'))
    assert policy['producer_bhp_add'] == pytest.approx(15.0)
    assert scale_of(policy, 'P1') == pytest.approx(0.9 * 0.9)


def test_block_shares_renormalise_over_blocks_that_still_hold_injectors() -> None:
    model = space(water_cut={})
    policy = model.decode(np.full(model.dim, 0.5), {'shut_wells': ['I3']})
    injected = sum(RATES[item['well']] * item['scale'] for item in policy['well_scales']
                   if ROLES[item['well']] == 'injector')
    assert injected == pytest.approx(0.75 * CAPS['injection_cap_m3d'])
    assert 'I3' in policy['shut_wells'] and scale_of(policy, 'I1') == pytest.approx(1.5)


def test_water_cut_threshold_shuts_only_producers_above_it() -> None:
    model = space()
    policy = model.decode(np.full(model.dim, 0.5))
    assert policy['shut_wells'] == ['P1']
    high = model.encode_seed({'watercut_shut': 0.995})
    assert model.decode(high)['shut_wells'] == []


def test_second_half_updates_carry_absolute_rates() -> None:
    model = space(water_cut={})
    x = model.encode_seed({'field_producer': [0.8, 1.1], 'field_injection_share': [0.6, 0.9]})
    policy = model.decode(x)
    updates = {item['well']: item for item in policy['well_updates']}
    assert set(updates) == set(ROLES)
    assert all(item['start'] == MONTHS[6] and item['end'] == MONTHS[-1] for item in updates.values())
    assert updates['P2']['value'] == pytest.approx(60.0 * 1.1 * 0.9)
    second = sum(updates[well]['value'] for well, role in ROLES.items() if role == 'injector')
    assert second == pytest.approx(0.9 * CAPS['injection_cap_m3d'])
    first = sum(RATES[item['well']] * item['scale'] for item in policy['well_scales']
                if ROLES[item['well']] == 'injector')
    assert first == pytest.approx(0.6 * CAPS['injection_cap_m3d'])


def test_decoded_policy_is_accepted_by_policy_controls() -> None:
    model = space()
    x = model.encode_seed({'field_producer': [0.8, 1.1], 'field_injection_share': [0.6, 0.9]})
    actions = policy_controls(source_controls(), model.decode(x))
    by_month = {month: {a['well']: a for a in actions if a['month'] == month} for month in MONTHS}
    assert by_month[MONTHS[0]]['P1']['status'] == 'SHUT'
    assert by_month[MONTHS[0]]['P2']['value'] == pytest.approx(60.0 * 0.8 * 0.9)
    assert by_month[MONTHS[6]]['P2']['value'] == pytest.approx(60.0 * 1.1 * 0.9)
    late = sum(by_month[MONTHS[6]][well]['value'] for well, role in ROLES.items() if role == 'injector')
    assert late == pytest.approx(0.9 * CAPS['injection_cap_m3d'])


def test_gene_updates_win_over_the_generated_second_half_update() -> None:
    model = space(water_cut={})
    x = model.encode_seed({'field_producer': [0.8, 1.1]})
    genes = {'well_updates': [{'well': 'P2', 'start': MONTHS[2], 'end': MONTHS[4], 'value': 11.0}]}
    policy = model.decode(x, genes)
    wells = [item['well'] for item in policy['well_updates']]
    assert wells.count('P2') == 1 and policy['well_updates'][wells.index('P2')]['value'] == 11.0
    policy_controls(source_controls(), policy)


@pytest.mark.parametrize('genes, message', [
    ({'nonsense': []}, 'unknown gene keys'),
    ({'shut_wells': ['P9']}, 'shut unknown wells'),
    ({'overrides': {'P9': 1.0}}, 'override unknown well'),
    ({'overrides': {'P1': 0.0}}, 'finite and positive'),
    ({'well_updates': [{'well': 'I1', 'start': MONTHS[0], 'end': MONTHS[1], 'role': 'producer'}]},
     'reverse conversion'),
    ({'well_updates': [{'well': 'P1', 'start': '1999-01-01', 'end': MONTHS[1]}]}, 'month grid'),
    ({'well_updates': [{'well': 'P1', 'start': MONTHS[3], 'end': MONTHS[1]}]}, 'month grid'),
    ({'well_updates': [{'well': 'P1', 'start': MONTHS[0]}]}, 'needs well, start and end'),
])
def test_genes_are_validated_against_well_roles(genes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        space().decode(np.full(10, 0.5), genes)


def test_forward_conversion_gene_is_accepted() -> None:
    genes = {'well_updates': [{'well': 'P3', 'start': MONTHS[1], 'end': MONTHS[-1],
                               'role': 'injector', 'target': 'WRAT', 'value': 40.0,
                               'bhp_limit': 120.0}]}
    policy = space().decode(np.full(10, 0.5), genes)
    assert policy_controls(source_controls(), policy)


def test_box_guard_and_determinism() -> None:
    model = space()
    with pytest.raises(ValueError, match='unit box'):
        model.decode(np.full(model.dim, 1.4))
    with pytest.raises(ValueError, match='10 coordinates'):
        model.decode(np.full(3, 0.5))
    x = np.linspace(0.05, 0.95, model.dim)
    assert model.decode(x) == model.decode(x)


def test_encode_seed_round_trips_named_parameters() -> None:
    model = space()
    seed = {'field_producer': [0.7, 1.15], 'field_injection_share': [0.55, 0.95],
            'block_producer': {'A': 0.6, 'B': 1.25}, 'block_injection_share': {'A': 0.25, 'B': 0.75},
            'watercut_shut': 0.97, 'producer_bhp_add': 12.0}
    params = model.unpack(model.encode_seed(seed))
    assert params['field_producer'] == pytest.approx([0.7, 1.15])
    assert params['block_producer'] == pytest.approx([0.6, 1.25])
    assert params['producer_bhp_add'][0] == pytest.approx(12.0)
    shares = model.block_injection_shares(params['block_injection_share'], model.block_ids)
    assert shares == pytest.approx({'A': 0.25, 'B': 0.75})
    assert model.encode_seed({})[0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match='extra'):
        model.encode_seed({'producer_scale': 1.0})


def test_describe_covers_every_coordinate() -> None:
    model = space()
    rows = model.describe(np.full(model.dim, 0.5))
    assert [row['index'] for row in rows if row['index'] is not None] == list(range(model.dim))
    derived = {row['name']: row['value'] for row in rows if row['group'] == 'derived'}
    assert derived['injection_target_half1'] == pytest.approx(450.0)
    assert derived['injection_share_A'] == pytest.approx(0.5)
