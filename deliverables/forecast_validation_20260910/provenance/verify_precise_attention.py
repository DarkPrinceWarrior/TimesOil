"""Verify precise attention metadata, complete decoder parity and backward on authenticated inputs."""
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
out = r / 'precise-attention-check-z-20260910'
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
torch.cuda.set_per_process_memory_fraction(.50)
data = torch.load(data_path, map_location='cuda', weights_only=True)
weights = torch.load(weight_path, map_location='cuda', weights_only=True)
graph = WellConnectivity.from_dict(json.loads(graph_path.read_text()))
forecaster = TimesFM3Forecaster(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
    revision=MODEL_REVISION, per_core_batch_size=1, device='cuda'))
weights['precise_variate_softmax'] = True
model = load_frozen_model(forecaster.model, graph, weights).eval()

from functools import partial
from torch.utils.checkpoint import checkpoint
from timesfm3.torch import cpm_revin_refine
from timesfm_geology import self_check
self_check()
model.requires_grad_(True)
for layer in model.transformer_stack.layers:
    layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
count = len(model.output_head.features)
scale = torch.tensor(model.cold_start_scale, device='cuda').repeat(len(graph.well_ids))[None, :, None, None]
decode = type(model).decode.__wrapped__
with torch.no_grad():
    first = model.decode(data['target'], horizon=224, past_future_covariates=data['covariates'])[:, :count] / scale
    second = decode(model, data['target'], horizon=224, past_future_covariates=data['covariates'])[:, :count] / scale
    error = float((first - second).abs().max())
    torch.testing.assert_close(first, second, rtol=0, atol=.001)
del first, second
refine = torch.no_grad()(cpm_revin_refine.cpm_iterative_revin_refine)
with patch.object(cpm_revin_refine, 'cpm_iterative_revin_refine', refine):
    prediction = decode(model, data['target'], horizon=224, past_future_covariates=data['covariates'])[:, :count] / scale
    loss = prediction.square().mean()
    loss.backward()
gradients = [value.grad for value in model.parameters() if value.grad is not None]
assert gradients and all(torch.isfinite(value).all() for value in gradients)
report = {'complete': True, 'source_inputs': {str(path): value for path, value in paths.items()},
    'precise_variate_softmax_restored_from_metadata': model.precise_variate_softmax,
    'decoder_parity_max_scaled': error, 'unchanged_parity_tolerance': .001,
    'gradient_tensors': len(gradients), 'all_gradients_finite': True,
    'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
    'optimizer_steps': 0, 'opm_calls': 0}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
