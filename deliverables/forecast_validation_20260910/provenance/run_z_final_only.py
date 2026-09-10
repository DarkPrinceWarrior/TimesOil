"""Original schedule → Google forecast CHDD search → sealed graph → one final OPM."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

from timesoil.aios.track2 import load_trajectory_dataset
from timesoil.aios.workflow import CycleRequest

R = Path('/root/projects/TimesOil/results/audit-20260909')
OUT = R / 'timesfm-final-only-z-20260910'
TRAINING = R / 'timesfm-economic-targets-z-20260910'
GPU_PYTHON = '/tmp/timesoil-kt3-20260908/venv/bin/python'
ROOT_PYTHON = '/root/projects/TimesOil/.venv/bin/python'


def run():
    protocol = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'waiting_for_session': 'timesoil-economic-targets-z-20260910',
        'search': 'Original source schedule, deterministic grid and three Qwen proposals; rank forecast CHDD.',
        'historical_physical_candidate_bank_used_for_selection': False,
        'new_search_opm_calls': 0, 'maximum_final_opm_calls': 1,
        'independently_certified_forecast_accuracy': False}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    while not (TRAINING / 'exit').exists():
        live = subprocess.run(['tmux', 'list-panes', '-t', protocol['waiting_for_session'],
                              '-F', '#{pane_dead}'], capture_output=True, text=True)
        if live.returncode or live.stdout.strip() != '0':
            raise RuntimeError('training process ended without its completion receipt')
        time.sleep(30)
    if (TRAINING / 'exit').read_text().strip() != '0':
        raise RuntimeError('economic target training failed; no search or OPM started')
    model = TRAINING / 'training'
    report = json.loads((model / 'report.json').read_text())
    checkpoint = sha256((model / 'full-model.pt').read_bytes()).hexdigest()
    if report.get('complete') is not True or checkpoint != report['checkpoint_sha256']:
        raise ValueError('completed frozen training checkpoint required')
    baseline = R / 'timesfm-bhp-policy-20260909/baseline-view'
    canonical = baseline / 'canonical'
    if sha256((canonical / 'manifest.json').read_bytes()).hexdigest() != 'b9ee48df3c4023e526597b5a54b47bdaf13dcf20de0c08dbd187a0f4fdc1d52a':
        raise ValueError('original baseline export changed')
    data = load_trajectory_dataset(canonical / 'trajectory.csv', manifest=canonical / 'manifest.json')
    if len(data) != 1 or not data.model_z_identity:
        raise ValueError('authenticated original Model Z schedule required')
    trajectory = data[0]
    origin = int(trajectory.dates.get_loc('2007-01-01'))
    controls = []
    for index in range(origin, origin + 224):
        for well, action in zip(trajectory.well_ids, trajectory.actions[index], strict=True):
            controls.append({'month': trajectory.dates[index].date().isoformat(), 'well': well,
                'role': 'injector' if action[1] == 2 else 'producer',
                'status': 'OPEN' if action[2] else 'SHUT',
                'target': ('ORAT', 'LRAT', 'WRAT')[int(action[1])], 'value': float(action[0]),
                **({'bhp_limit': float(action[3])} if action[3] > 0 else {})})
    request = {'context': {'track': 2,
        'objective': 'Optimize the complete original schedule with forecast CHDD before final OPM.',
        'constraints': {'allow_conversion_to_injection': True},
        'facts': {'original_schedule_start': True, 'physical_candidate_bank_used': False}},
        'controls': controls, 'source': '/tmp/timesoil-kt2/model_z/Model_Z_final_OPM.zip',
        'deck': 'Model_Z/Model_Z.data', 'schedule_relative_path': 'Model_Z/Model_Z_sch.inc',
        'scenario_id': 'forecast-economic-search', 'source_model': 'model_z_opm', 'start_year': 2007,
        'parsing_strictness': 'low', 'charge_initial_pump': False, 'horizon_months': 224}
    checked = CycleRequest.from_mapping(request)
    (OUT / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'CUDA_VISIBLE_DEVICES': '5',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'OPM_MPI_PROCESSES': '16', 'OPM_THREADS_PER_PROCESS': '1', 'OPM_CPU_AFFINITY': '30-45',
        'LLM_BASE_URL': 'https://litellm.tatneft.guru/v1', 'LLM_MODEL': 'qwen3.8-27b',
        'LLM_TIMEOUT_SECONDS': '600', 'LLM_MAX_OUTPUT_TOKENS': '8192',
        'LLM_API_KEY': Path('/dev/shm/timesoil-tatneft-20260909-key').read_text().strip()}
    protocol.update(checkpoint_sha256=checkpoint, training_report_sha256=sha256((model / 'report.json').read_bytes()).hexdigest(),
        original_controls_sha256=checked.controls_sha256, started_utc=datetime.now(timezone.utc).isoformat())
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    def execute(name, command):
        with (OUT / f'{name}.log').open('x') as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (OUT / f'{name}.exit').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'{name} failed; retained log, no automatic retry')
    execute('search', [GPU_PYTHON, 'scripts/propose_track2_policies.py', str(baseline),
        str(OUT / 'request.json'), str(OUT / 'search'), '--rounds', '3', '--economic-selection',
        '--head', str(model / 'full-model.pt'), '--head-sha256', checkpoint,
        '--head-report', str(model / 'report.json'),
        '--connectivity', str(R / 'static-head-geology-20260909/model-z/connectivity.json')])
    seal_hash = sha256((OUT / 'search/selection-before-opm.json').read_bytes()).hexdigest()
    protocol['selection_seal_sha256'] = seal_hash
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    execute('final', [ROOT_PYTHON, 'scripts/track2_final_selection.py', str(OUT / 'search'),
        str(R / 'timesfm-bhp-policy-20260909/cycles/baseline'), str(OUT / 'final'), '--seal-sha256', seal_hash])


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    try:
        run()
    except BaseException:
        (OUT / 'exit').write_text('1\n')
        raise
    (OUT / 'exit').write_text('0\n')
