from datetime import date

import pytest

from timesoil.aios.contracts import ControlAction, ControlTarget, WellRole, WellStatus
from timesoil.aios.workflow import CycleError, _physical_control_evidence


def test_actual_rate_limit_covers_every_controlled_month_and_well():
    controls = tuple(ControlAction(date(2007, month, 1), well, WellRole.PRODUCER,
                                  WellStatus.OPEN, ControlTarget.LIQUID_RATE, 100)
                     for month in (1, 2) for well in ('A', 'B'))
    rows = [dict(DATA=f'2007-{month:02d}-01', well=well, WLPR='100')
            for month in (2, 3) for well in ('A', 'B')]
    # Historical values are outside the management-period rate gate.
    history = [dict(DATA='2007-01-01', well='A', WLPR='600')]
    assert _physical_control_evidence(history + rows, controls)['observations_checked'] == 4
    for invalid in (rows[:-1], rows + rows[:1], rows[:-1] + [{**rows[-1], 'well': 'C'}],
                    rows[:-1] + [{**rows[-1], 'WLPR': '500.01'}],
                    rows[:-1] + [{**rows[-1], 'WLPR': 'nan'}]):
        with pytest.raises(CycleError):
            _physical_control_evidence(history + invalid, controls)
