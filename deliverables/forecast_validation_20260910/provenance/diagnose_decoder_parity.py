"""Repeat the failed decoder comparison before any optimizer step on A100."""
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys

r = Path('/root/projects/TimesOil/results/audit-20260909')
out = r / 'decoder-parity-z-20260910'
out.mkdir(exist_ok=False)
source = Path('scripts/finetune_timesfm_head.py').resolve()
stop_line = next(i for i, line in enumerate(source.read_text().splitlines(), 1)
                 if line.strip().startswith('wrapped = model.decode('))
protocol = json.loads((r / 'timesfm-economic-regimes-z-20260910/launch-protocol.json').read_text())
args = protocol['command'][1:]
args[args.index('--output') + 1] = str(out / 'setup')


class Complete(Exception):
    pass


def trace(frame, event, arg):
    if event != 'line' or frame.f_code.co_filename != str(source) or frame.f_lineno != stop_line:
        return trace
    sys.settrace(None)
    import torch
    state = frame.f_locals
    model, target, cov = (state[k] for k in ('model', 'target', 'cov'))
    scale = state['scale'][..., None]
    count, horizon = state['targets_count'], state['training_horizon']
    originals = [target.clone(), cov.clone()]
    report = {'source_sha256': sha256(source.read_bytes()).hexdigest(),
        'checkpoint_sha256': protocol['command'][protocol['command'].index('--initial-head-sha256') + 1],
        'torch': torch.__version__, 'matmul_precision': torch.get_float32_matmul_precision(),
        'tf32': torch.backends.cuda.matmul.allow_tf32,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic': torch.are_deterministic_algorithms_enabled(), 'comparisons': []}
    previous = None
    for name, wrapped, inference in [('wrapped-1', True, False), ('wrapped-2', True, False),
            ('unwrapped', False, False), ('inference', True, True), ('wrapped-3', True, False)]:
        torch.manual_seed(20260909)
        with torch.inference_mode() if inference else torch.no_grad():
            value = (model.decode(target, horizon=horizon, past_future_covariates=cov) if wrapped
                     else state['decode'](model, target, horizon=horizon, past_future_covariates=cov))[:, :count]
            value = (value / scale).cpu()
        if previous is not None:
            difference = (value - previous).abs()
            report['comparisons'].append({'call': name, 'maximum_scaled_difference_from_previous': float(difference.max()),
                'elements_above_original_tolerance': int((difference > .001).sum())})
        previous = value
    report['inputs_unchanged'] = all(torch.equal(a, b) for a, b in zip(originals, (target, cov)))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise Complete


sys.argv = args
sys.settrace(trace)
try:
    runpy.run_path(str(source), run_name='__main__')
except Complete:
    pass
else:
    raise RuntimeError('diagnostic breakpoint was not reached')
finally:
    sys.settrace(None)
