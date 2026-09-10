"""Resume only numerical selection/review; never rerun completed forecast or physics."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

R = Path('/root/projects/TimesOil/results/audit-20260909')
OUT = R / 'timesfm-guarded-z-policy-20260910'
assert (OUT / 'candidate.exit').read_text().strip() == '0'
assert (OUT / 'selection.exit').read_text().strip() == '1'
receipt = OUT / 'cycles/candidate/full-cycle-receipt.json'
receipt_hash = sha256(receipt.read_bytes()).hexdigest()
assert json.loads(receipt.read_text())['complete']
assert (OUT / 'numeric-selection.json').is_file()
protocol = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                candidate_receipt_sha256=receipt_hash, physical_rerun=False,
                reason='Only canonical passive water SUMMARY outputs differ; original source/physics/economic guards retained.')
with (OUT / 'selection-recovery-protocol.json').open('x') as stream:
    json.dump(protocol, stream, indent=2)
env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'LLM_BASE_URL': 'https://litellm.tatneft.guru/v1',
       'LLM_MODEL': 'qwen3.8-27b', 'LLM_TIMEOUT_SECONDS': '600', 'LLM_MAX_OUTPUT_TOKENS': '8192',
       'LLM_API_KEY': Path('/dev/shm/timesoil-tatneft-20260909-key').read_text().strip()}
command = ['/root/projects/TimesOil/.venv/bin/python', 'scripts/compare_track2_cycles.py',
    str(R / 'timesfm-bhp-policy-20260909/cycles/baseline'), str(OUT / 'cycles/candidate'),
    str(OUT / 'selection-recovered.json'), '--expected-months', '224', '--select-from',
    str(R / 'physical-sweep-z-20260909/cycles/candidate-03'),
    str(R / 'timesfm-early-identity-z-policy-20260910/cycles/candidate'), '--agent-review']
with (OUT / 'selection-recovery.log').open('x') as log:
    code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
try:
    assert code == 0
    result = json.loads((OUT / 'selection-recovered.json').read_text())
    assert result['agent_review_approved'] and len(result['agent_review']['decisions']) == 4
    assert all(d['approved'] for d in result['agent_review']['decisions'])
    assert sha256(receipt.read_bytes()).hexdigest() == receipt_hash
except Exception:
    (OUT / 'selection-recovery.exit').write_text('1\n')
    raise
(OUT / 'selection-recovery.exit').write_text('0\n')
