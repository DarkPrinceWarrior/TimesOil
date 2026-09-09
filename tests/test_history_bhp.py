from datetime import date
from timesoil.aios.opm_chdd import _scheduled_controls


def test_historical_pressure_is_distinct_from_requested_bhp():
    source = """SCHEDULE
WCONHIST
'1' 'OPEN' 'ORAT' 79 44 0 3* 99 /
/
WELTARG
'1' 'BHP' 20 /
/
DATES
1 JUN 2007 /
/
WCONPROD
'1' 'OPEN' 'LRAT' 3* 100 1* 50 /
/
"""
    summary = [(date(2007, month, 1), {"1": {"WLPR": 0, "WWIR": 0}}, {}) for month in (5, 6)]
    controls = _scheduled_controls(source, date(2007, 5, 1), summary, ("1",))["1"]
    assert [(c.target, c.value, c.bhp_limit) for c in controls] == [("ORAT", 79, 20), ("LRAT", 100, 50)]
    observed_only = source.replace("WELTARG\n'1' 'BHP' 20 /\n/\n", "")
    assert _scheduled_controls(observed_only, date(2007, 5, 1), summary, ("1",))["1"][0].bhp_limit == 0
