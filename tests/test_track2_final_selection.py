from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import track2_final_selection as selection
from timesfm_economics import ECONOMIC_TARGETS
from timesoil.aios.workflow import CycleRequest


def test_forecast_selection_seal_tamper_and_one_final_attempt(tmp_path, monkeypatch):
    root = tmp_path / 'search'; root.mkdir()
    source = tmp_path / 'model.zip'; source.write_bytes(b'source')
    profile = {'source_sha256': {'Нормативы_ЧДД.xlsx': 'norms', 'calculator.py': 'calculator'}}
    def write(name, value):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
        return selection.digest(path)

    receipt = dict(economic_selection=True, search_opm_calls=0, reference_manifest_sha256=None,
        reference_correction_sha256=None, final_chdd_computed=False, head_sha256='head',
        head_report_sha256='training', horizon_months=6, normative_profile=profile, agent_proposal_ids=[1])
    candidates = []
    for i, score in enumerate([10., 20., 100.]):
        request = dict(context={'track': 2}, source=str(source), deck='CASE.DATA',
            schedule_relative_path='schedule.inc', scenario_id=f'candidate-{i}', source_model='model_z_opm',
            start_year=2007, charge_initial_pump=False, controls=[dict(month=f'2007-{m:02d}-01',
                well=str(w), role='producer', status='OPEN', target='LRAT', value=10. + i)
                for m in range(1, 7) for w in range(103)])
        checked = CycleRequest.from_mapping(request)
        write(f'request-{i:02d}.json', request)
        forecast_hash = write(f'forecast-{i:02d}.npz', ['opaque forecast artifact'])
        directory = f'economics-{i:02d}'
        input_hash = write(f'{directory}/input.csv', ['opaque input artifact'])
        result_hash = write(f'{directory}/result.json', {'summary': {'totalChddM': score}})
        manifest_hash = write(f'{directory}/manifest.json', dict(management_period={'total_chdd_m':score},
            norms_source_sha256='norms', calculator_sha256={'calculator.py':'calculator'},
            assumption_overrides={'chargeInitialPump':False}, artifacts={'input':'input.csv','result':'result.json'},
            input_sha256=input_hash, result_sha256=result_hash))
        candidates.append(dict(id=i, forecast_chdd_m=score, economic_targets=list(ECONOMIC_TARGETS),
            is_official_chdd=False, forecast_eligible=i != 2, trained_head_sha256='head',
            training_report_sha256='training', controls_sha256=checked.controls_sha256,
            schedule_overlay_sha256=f'overlay-{i}', forecast_sha256=forecast_hash,
            forecast_economics_directory=directory, forecast_economics_manifest_sha256=manifest_hash))
    write('proposal-receipt.json', receipt); write('candidates.json', candidates)
    write('agent-00.json', {'decisions':[{'approved':True}] * 3})
    for bad in ({'forecast_chdd_m':float('nan')}, {'forecast_chdd_m':True}, {'economic_targets':['oil']},
                {'forecast_chdd_m':None}, {'is_official_chdd':True}):
        rows = deepcopy(candidates);rows[0].update(bad)
        with pytest.raises(ValueError):
            selection.best_forecast(rows)
    assert selection.best_forecast(candidates)['id'] == 1  # Infeasible larger score cannot win.
    seal = selection.seal_forecast_selection(root)
    assert seal['selected_id'] == 1
    original = (root / 'request-01.json').read_bytes()
    (root / 'request-01.json').write_bytes(original + b' ')
    with pytest.raises(ValueError, match='changed'):
        selection.verify_selection(root, seal['sha256'])
    (root / 'request-01.json').write_bytes(original)
    _, _, path = selection.verify_selection(root, seal['sha256'])
    assert path.name == 'request-01.json'
    monkeypatch.setattr(selection.CHDDEconomicsAdapter, 'from_env', lambda: SimpleNamespace(
        normative_profile=lambda **kwargs: profile))
    calls = []
    def fail(command, **kwargs):
        assert command[command.index('full-cycle') + 1] == str(path)
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(selection.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='second OPM'):
        selection.run_final_verification(root, seal['sha256'], tmp_path / 'baseline', tmp_path / 'final')
    with pytest.raises(FileExistsError):
        selection.run_final_verification(root, seal['sha256'], tmp_path / 'baseline', tmp_path / 'another')
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        selection.seal_forecast_selection(root)
