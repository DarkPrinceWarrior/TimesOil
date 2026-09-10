"""Resume the sealed Qwen proposal after the pre-OPM Python environment failure."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from run_z_transition_policy import CYCLE_PYTHON, OUT, R, proposal_id

assert (OUT / 'proposal.exit').read_text().strip() == '0'
assert (OUT / 'candidate.exit').read_text().strip() == '1'
assert not (OUT / 'cycles/candidate').exists()
receipt_path = OUT / 'proposals/proposal-receipt.json'
receipt = json.loads(receipt_path.read_text())
original = json.loads((OUT / 'protocol.json').read_text())
index = proposal_id(receipt, original['checkpoint_sha256'])
request = OUT / f'proposals/request-{index:02d}.json'
protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    reason='Scientific Python lacked FastAPI; no OPM run was created. Reuse the sealed successful forecast and Qwen proposal.',
    cycle_python=CYCLE_PYTHON, proposal_receipt_sha256=sha256(receipt_path.read_bytes()).hexdigest(),
    request_sha256=sha256(request.read_bytes()).hexdigest(), checkpoint_sha256=original['checkpoint_sha256'])
with (OUT / 'recovery-protocol.json').open('x') as stream:
    json.dump(protocol, stream, indent=2)
env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'OPM_MPI_PROCESSES': '16', 'OPM_THREADS_PER_PROCESS': '1', 'OPM_CPU_AFFINITY': '30-45',
    'LLM_BASE_URL': 'https://litellm.tatneft.guru/v1', 'LLM_MODEL': 'qwen3.8-27b',
    'LLM_TIMEOUT_SECONDS': '600', 'LLM_MAX_OUTPUT_TOKENS': '8192',
    'LLM_API_KEY': Path('/dev/shm/timesoil-tatneft-20260909-key').read_text().strip()}


def execute(name, command):
    with (OUT / f'{name}.log').open('x') as log:
        code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (OUT / f'{name}.exit').write_text(str(code) + '\n')
    if code:
        raise RuntimeError(f'{name} failed with {code}; retained logs and initial failure receipts')


execute('candidate-recovery', [CYCLE_PYTHON, '-m', 'timesoil.aios.cli', 'full-cycle', str(request),
    '--runs-dir', str(OUT / 'cycles'), '--run-id', 'candidate', '--timeout', '7200'])
assert sha256(request.read_bytes()).hexdigest() == protocol['request_sha256']
execute('selection', [CYCLE_PYTHON, 'scripts/compare_track2_cycles.py',
    str(R / 'timesfm-bhp-policy-20260909/cycles/baseline'), str(OUT / 'cycles/candidate'),
    str(OUT / 'selection.json'), '--expected-months', '224', '--select-from',
    str(R / 'physical-sweep-z-20260909/cycles/candidate-03'),
    str(R / 'timesfm-early-identity-z-policy-20260910/cycles/candidate'), '--agent-review'])
selection = json.loads((OUT / 'selection.json').read_text())
assert selection['agent_review_approved'] and len(selection['agent_review']['decisions']) == 4
assert all(d['approved'] for d in selection['agent_review']['decisions'])
(OUT / 'recovery.exit').write_text('0\n')
(OUT / 'exit').write_text('0\n')
