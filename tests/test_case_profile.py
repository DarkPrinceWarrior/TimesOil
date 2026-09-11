"""The case profile is the only source of the gate limits, and it fails closed."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import pytest

from timesoil.aios.case_profile import CaseProfileError, load_case_profile, parse_case_profile

CONFIG = Path(__file__).resolve().parents[1] / 'config'
DIGEST = '0' * 64


def profile_data(name='case_constraints.example.json'):
    return json.loads((CONFIG / name).read_text())


def test_shipped_profiles_differ_only_in_the_vrr_lower_bound_status():
    development = load_case_profile(CONFIG / 'case_constraints.example.json')
    test_case = load_case_profile(CONFIG / 'case_constraints.test.example.json')
    assert development.vrr['lower_bound_status'] == 'diagnostic'
    assert test_case.vrr['lower_bound_status'] == 'hard'
    assert development.liquid_cap_m3d == test_case.liquid_cap_m3d == 1500
    assert development.bhp_bounds == (50., 300.) and development.vrr['window_months'] == 3
    assert development.water_balance == {'deficit_m3': 0., 'carryover': False}
    assert development.selection_margins == {'eps_liquid': .03, 'eps_injection': .03, 'phi': .95}
    assert development.sha256 == sha256((CONFIG / 'case_constraints.example.json').read_bytes()).hexdigest()
    assert development.sha256 != test_case.sha256
    assert parse_case_profile(development.to_dict(), sha256_hex=DIGEST).to_dict() == development.to_dict()


def test_strict_schema_rejects_unknown_keys_ranges_and_enums():
    base = profile_data()
    for change in ({'extra': 1}, {'liquid_cap_m3d': 'x'}, {'liquid_cap_m3d': -1},
                   {'bhp_bounds': [50]}, {'bhp_bounds': [300, 50]}, {'bhp_bounds': [0, 300]},
                   {'vrr': {**base['vrr'], 'denominator': 'barrels'}},
                   {'vrr': {**base['vrr'], 'lower_bound_status': 'soft'}},
                   {'vrr': {**base['vrr'], 'window_months': 0}},
                   {'vrr': {**base['vrr'], 'window_months': 3.0}},
                   {'vrr': {**base['vrr'], 'min': 2, 'max': 1}},
                   {'vrr': {k: v for k, v in base['vrr'].items() if k != 'min'}},
                   {'water_balance': {'deficit_m3': 0, 'carryover': 'no'}},
                   {'water_balance': {'deficit_m3': -1, 'carryover': False}},
                   {'pressure': {**base['pressure'], 'regions': 'blocks'}},
                   {'selection_margins': {**base['selection_margins'], 'phi': 0}},
                   {'selection_margins': {**base['selection_margins'], 'eps_liquid': .9}},
                   {'repairs': [{'well': 'P', 'start': '2007-01-15', 'end': '2007-02-01'}]},
                   {'repairs': [{'well': 'P', 'start': '2007-03-01', 'end': '2007-02-01'}]},
                   {'repairs': [{'well': 'P', 'start': '2007-01-01'}]}):
        with pytest.raises(CaseProfileError):
            parse_case_profile({**base, **change}, sha256_hex=DIGEST)
    missing = {k: v for k, v in base.items() if k != 'repairs'}
    with pytest.raises(CaseProfileError):
        parse_case_profile(missing, sha256_hex=DIGEST)
    with pytest.raises(CaseProfileError):
        parse_case_profile(base, sha256_hex='ABC')


def test_profile_builds_the_three_gate_rules_and_refuses_what_it_cannot_check(tmp_path):
    data = deepcopy(profile_data())
    data['repairs'] = [{'well': 'P', 'start': '2007-02-01', 'end': '2007-03-01'},
                       {'well': 'I', 'start': '2030-01-01', 'end': '2030-02-01'}]
    start, end = date(2007, 1, 1), date(2007, 4, 1)
    profile = parse_case_profile(data, sha256_hex=DIGEST)
    rules = profile.operating_rules(wells=['P', 'I'], start=start, end=end)
    limits = {key: (value, rule.status, rule.window_months)
              for rule in rules for key, value in rule.limits}
    assert limits['max_monthly_liquid_m3d'] == (1500., 'hard', 1)
    assert limits['max_monthly_injection_m3d'] == (1500., 'hard', 1)
    assert limits['min_bhp_bar'] == (50., 'hard', 1) and limits['max_bhp_bar'] == (300., 'hard', 1)
    assert limits['max_window_voidage_replacement'] == (1.15, 'hard', 3)
    assert limits['min_window_voidage_replacement'] == (.85, 'diagnostic', 3)
    assert limits['max_monthly_water_deficit_m3'] == (0., 'hard', 1)
    # Only the repair inside the management period becomes an unavailability rule.
    outages = [rule for rule in rules if rule.unavailable]
    assert [(r.wells, r.start, r.end) for r in outages] == [(('P',), date(2007, 2, 1), date(2007, 3, 1))]
    assert all(set(r.wells) == {'P', 'I'} for r in rules if not r.unavailable)

    surface = parse_case_profile({**data, 'vrr': {**data['vrr'], 'denominator': 'water_surface'}},
                                 sha256_hex=DIGEST).operating_rules(wells=['P', 'I'], start=start, end=end)
    assert any(key == 'max_window_water_replacement' for rule in surface for key, _ in rule.limits)

    for change, match in (({'water_balance': {'deficit_m3': 0, 'carryover': True}}, 'carryover'),
                          ({'pressure': {**data['pressure'], 'field_min_bar': 90}}, 'FPR/RPR')):
        broken = parse_case_profile({**data, **change}, sha256_hex=DIGEST)
        with pytest.raises(CaseProfileError, match=match):
            broken.operating_rules(wells=['P', 'I'], start=start, end=end)
    with pytest.raises(CaseProfileError, match='unknown well'):
        profile.operating_rules(wells=['I', 'X'], start=start, end=end)
    with pytest.raises(CaseProfileError, match='ordered monthly'):
        profile.operating_rules(wells=['P', 'I'], start=end, end=start)
    with pytest.raises(CaseProfileError, match='well stock'):
        profile.operating_rules(wells=[], start=start, end=end)

    path = tmp_path / 'broken.json'
    path.write_text('{')
    with pytest.raises(CaseProfileError, match='valid JSON'):
        load_case_profile(path)
    with pytest.raises(CaseProfileError, match='cannot read'):
        load_case_profile(tmp_path / 'absent.json')


def test_null_water_deficit_disables_the_water_gate(tmp_path):
    """The official case has an external water supply: no produced-water rule at all."""
    from timesoil.aios.case_profile import load_case_profile
    import json, datetime
    data = profile_data()
    data['water_balance'] = {'deficit_m3': None, 'carryover': False}
    path = tmp_path / 'p.json'; path.write_text(json.dumps(data))
    profile = load_case_profile(path)
    rules = profile.operating_rules(wells=('1', '2'), start=datetime.date(2007, 1, 1), end=datetime.date(2007, 3, 1))
    kinds = {key for rule in rules for key, _ in rule.limits}
    assert 'max_monthly_water_deficit_m3' not in kinds
    assert 'max_monthly_liquid_m3d' in kinds and 'min_bhp_bar' in kinds
