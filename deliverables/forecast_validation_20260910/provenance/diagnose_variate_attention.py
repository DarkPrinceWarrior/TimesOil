"""Isolate the first variate-attention block using authenticated saved decoder inputs."""
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from timesfm3 import ModelConfig, TimesFM3Forecaster
from benchmark_timesfm3 import MODEL_REVISION
from timesfm_geology import load_frozen_model
from timesoil.aios.interwell import WellConnectivity

r = Path('/root/projects/TimesOil/results/audit-20260909')
out = r / 'attention-softmax-parity-z-20260910'
out.mkdir(exist_ok=False)
paths = {
    r / 'decoder-parity-components-z-20260910/decoder-inputs.pt': '6ee35fcd634bb28f6695329b11bd24f0172c00e20be4a984e030f3caf032f655',
    r / 'timesfm-economic-targets-z-20260910/training/full-model.pt': '9b954acbe64f0c5ff8dc213315a46b9324a948294ac19bc3b087d55fb50844ae',
    r / 'static-head-geology-20260909/model-z/connectivity.json': 'cff65939ad943dd1df28460306fc433663a22707bd66a087732077135f7992c0'}
for path, expected in paths.items():
    assert sha256(path.read_bytes()).hexdigest() == expected
assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(20260909)
data_path, weight_path, graph_path = paths
data = torch.load(data_path, map_location='cuda', weights_only=True)
weights = torch.load(weight_path, map_location='cuda', weights_only=True)
graph = WellConnectivity.from_dict(json.loads(graph_path.read_text()))
forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
    revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
model = load_frozen_model(forecaster.model, graph, weights).eval()
attention = model.transformer_stack.layers[0].layer.var_attn
captured = {}


class Captured(Exception):
    pass


def capture(module, args, kwargs):
    captured.update(args=args, kwargs=kwargs)
    raise Captured


handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
try:
    model.decode(data['target'], horizon=224, past_future_covariates=data['covariates'])
except Captured:
    pass
finally:
    handle.remove()
assert captured
report = {'native_use_sdpa': attention.use_sdpa, 'rescale_logits': attention.rescale_logits,
          'inputs': {str(path): digest for path, digest in paths.items()}, 'comparisons': []}
projection_inputs = []
handle = attention.out_proj.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0].detach().clone()))
with torch.no_grad():
    for mode in (False, True):
        attention.use_sdpa = mode
        projection_inputs.clear()
        first = attention(*captured['args'], **captured['kwargs'])[0]
        second = attention(*captured['args'], **captured['kwargs'])[0]
        row = {'use_sdpa': mode, 'output_difference': float((first - second).abs().max()),
               'projection_input_difference': float((projection_inputs[0] - projection_inputs[1]).abs().max())}
        source = projection_inputs[0]
        handle.remove()
        a, b = attention.out_proj(source), attention.out_proj(source)
        row['same_projection_input_output_difference'] = float((a - b).abs().max())
        handle = attention.out_proj.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0].detach().clone()))
        report['comparisons'].append(row)
handle.remove()
report['operations'] = []
def repeated(operation, name):
    def call(*args, **kwargs):
        first, second = operation(*args, **kwargs), operation(*args, **kwargs)
        report['operations'].append({'operation': name, 'input_shapes': [list(x.shape) for x in args if isinstance(x, torch.Tensor)],
            'difference': float((first - second).abs().max())})
        if name == 'softmax':
            x, dim = args[0], kwargs['dim']
            def explicit():
                exponent = (x - x.amax(dim=dim, keepdim=True)).exp()
                return exponent / exponent.sum(dim=dim, keepdim=True)
            for label, alternate in [('log_softmax_exp', lambda: torch.nn.functional.log_softmax(x, dim=dim).exp()),
                                     ('explicit_exp_sum', explicit)]:
                a, b = alternate(), alternate()
                report['operations'].append({'operation': label, 'difference': float((a - b).abs().max()),
                    'difference_from_native': float((a - first).abs().max())})
        return first
    return call
attention.use_sdpa = False
with torch.no_grad(), patch.object(torch, 'matmul', repeated(torch.matmul, 'matmul')), \
        patch.object(torch.nn.functional, 'softmax', repeated(torch.nn.functional.softmax, 'softmax')):
    attention(*captured['args'], **captured['kwargs'])
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
