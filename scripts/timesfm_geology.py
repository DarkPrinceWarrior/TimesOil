"""Static conditioning of the native Google output head, after temporal RevIN."""

import numpy as np
import torch
from torch import nn


def geological_inputs(trajectory, origin, context, horizon, connectivity):
    from benchmark_timesfm3 import forecast_inputs
    if tuple(trajectory.well_ids) != tuple(connectivity.well_ids):
        raise ValueError('geological well order differs from trajectory')
    target, cov = forecast_inputs(trajectory.states, trajectory.actions, origin, context, horizon)
    count, _, length = target.shape
    controls = trajectory.actions[origin - length:origin + horizon]
    allocated = connectivity.features(np.zeros((len(controls) * count, 3)),
        controls.reshape(-1, controls.shape[-1]))[:, 0].reshape(len(controls), count).T
    return target.reshape(count * 3, length), np.concatenate(
        [cov[:, :-1].reshape(-1, length + horizon), allocated], axis=0)


class StaticConditionedHead(nn.Module):
    def __init__(self, head, connectivity):
        super().__init__()
        self.head = head
        raw = np.asarray(connectivity.provenance.get('static_features', connectivity.static), dtype=float)
        if raw.ndim != 2 or len(raw) != len(connectivity.well_ids) or not np.isfinite(raw).all():
            raise ValueError('invalid static geological features')
        transformed = np.sign(raw) * np.log1p(np.abs(raw))
        mean, scale = transformed.mean(axis=0), transformed.std(axis=0)
        features = (transformed - mean) / np.maximum(scale, 1e-6)
        # Each target gets its own well's geology and an explicit oil/liquid/pressure identity.
        features = np.column_stack([np.repeat(features, 3, axis=0), np.tile(np.eye(3), (len(raw), 1))])
        self.register_buffer('features', torch.tensor(features, device=head.weight.device, dtype=head.weight.dtype))
        self.conditioner = nn.Linear(features.shape[1], head.out_features, bias=False,
                                     device=head.weight.device, dtype=head.weight.dtype)
        nn.init.zeros_(self.conditioner.weight)
        self.disabled = False

    def forward(self, values):
        output = self.head(values)
        count = len(self.features)
        if output.ndim != 4 or output.shape[1] < count:
            raise ValueError('static head requires joint targets in verified well order')
        if not self.disabled:
            bias = self.conditioner(self.features)[None, :, None, :]
            output = torch.cat([output[:, :count] + bias, output[:, count:]], dim=1)
        return output


def self_check():
    from types import SimpleNamespace
    connection = SimpleNamespace(well_ids=('a', 'b'), static=[[10, .1, 1], [20, .2, 3]], provenance={})
    base = nn.Linear(4, 9)
    head = StaticConditionedHead(base, connection)
    values = torch.ones((1, 8, 2, 4))
    torch.testing.assert_close(head(values), base(values), rtol=0, atol=0)
    with torch.no_grad():
        head.conditioner.weight[:, 0] = 1
    changed = head(values)
    assert not torch.equal(changed[:, 0], changed[:, 3])
    torch.testing.assert_close(changed[:, 6:], base(values)[:, 6:], rtol=0, atol=0)
    head.disabled = True
    torch.testing.assert_close(head(values), base(values), rtol=0, atol=0)
    from timesoil.aios.interwell import WellConnectivity
    connection = WellConnectivity(('a', 'b'), [[0, 1], [1, 0]], [[10, .1, 1], [20, .2, 3]], {})
    states = np.ones((12, 2, 3))
    actions = np.ones((12, 2, 4)); actions[:, 1, 1] = 2
    trajectory = SimpleNamespace(well_ids=('a', 'b'), states=states, actions=actions)
    target, cov = geological_inputs(trajectory, 5, 4, 3, connection)
    states[6:] = 1e9
    other_target, other_cov = geological_inputs(trajectory, 5, 4, 3, connection)
    np.testing.assert_array_equal(target, other_target)
    np.testing.assert_array_equal(cov, other_cov)
    assert cov.shape == (12, 7)
    print('Static conditioning parity, well identity, ablation and covariate isolation passed', flush=True)


if __name__ == '__main__':
    self_check()
