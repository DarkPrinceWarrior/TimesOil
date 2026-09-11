"""Repeat full-field physical MPC with organizer-permitted new conversions enabled."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


def run():
    root = Path('/root/projects/TimesOil/results/audit-20260909')
    original = root / 'physical-only-monthly-y-20260910/case.json'
    original_hash = sha256(original.read_bytes()).hexdigest()
    assert original_hash == '2fa90146a479017efdd40bac65aa82b3936edb10bdc8c072841040e55fcdb4a4'
    config = json.loads(original.read_text())
    assert 'allow_conversion_to_injection' not in config['case']
    config['case']['allow_conversion_to_injection'] = True
    output = root / 'conversion-monthly-y-20260910'
    output.mkdir(exist_ok=False)
    config['opm']['runs_dir'] = str(output / 'opm')
    case = output / 'case.json'
    case.write_text(json.dumps(config, indent=2) + '\n')
    # Only permission and the isolated output location may change.
    check = json.loads(case.read_text())
    assert check['case'].pop('allow_conversion_to_injection') is True
    check['opm']['runs_dir'] = json.loads(original.read_text())['opm']['runs_dir']
    assert check == json.loads(original.read_text())
    python = '/tmp/timesoil-kt3-20260908/venv/bin/python'
    command = [python, 'scripts/run_track1_mpc.py', str(case), '--runs-dir', str(output / 'runs'),
               '--agent', '--full-field', '--lifecycle']
    protocol = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        original_case_sha256=original_hash, case_sha256=sha256(case.read_bytes()).hexdigest(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), command=command,
        surrogate_used=False, new_conversion_permission=True,
        scope='49 wells, 23 monthly decisions, full remaining OPM and official CHDD; assess new conversions',
        permission_source='Organizer conversion economic methodology; 5 million RUB once per conversion',
        conversion_selected_or_profitable=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'OPM_MPI_PROCESSES': '16', 'OPM_THREADS_PER_PROCESS': '1', 'OPM_CPU_AFFINITY': '48-63',
        'LLM_BASE_URL': 'https://litellm.tatneft.guru/v1', 'LLM_MODEL': 'qwen3.8-27b',
        'LLM_TIMEOUT_SECONDS': '600', 'LLM_MAX_OUTPUT_TOKENS': '8192',
        'LLM_API_KEY': Path('/dev/shm/timesoil-tatneft-20260909-key').read_text().strip()}
    with (output / 'controller.log').open('x') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (output / 'controller.exit').write_text(str(result.returncode) + '\n')
    result.check_returncode()
    from run_track1_mpc import load_config
    from audit_track1_lifecycle import audit
    checked = audit(output, 23, run_dir=output / 'runs' / load_config(case).run_id,
        baseline_run=root / 'physical-sweep-y-20260909/opm/opm-full-replay-40f2f804257afe12efd408b7')
    (output / 'full-audit.json').write_text(json.dumps(checked, indent=2) + '\n')
    (output / 'exit').write_text('0\n')


if __name__ == '__main__':
    try:
        run()
    except Exception:
        output = Path('/root/projects/TimesOil/results/audit-20260909/conversion-monthly-y-20260910')
        if output.is_dir():
            (output / 'exit').write_text('1\n')
        raise
