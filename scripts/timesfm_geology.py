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


class StaticConditionedLayer(nn.Module):
    """Identify each well and target/control channel before native variate attention."""

    def __init__(self, layer, head):
        super().__init__()
        self.layer = layer
        geology = head.features[::3, :-3]
        kinds = torch.eye(9, device=geology.device, dtype=geology.dtype)
        features = torch.cat([torch.cat([geology.repeat_interleave(len(group), dim=0),
            kinds[group].repeat(len(geology), 1)], dim=1) for group in
            (list(range(3)), list(range(3, 8)), [8])])
        self.register_buffer('features', features)
        self.conditioner = nn.Linear(features.shape[1], head.head.in_features, bias=False,
            device=geology.device, dtype=geology.dtype)
        nn.init.zeros_(self.conditioner.weight)

    def forward(self, values, *args, **kwargs):
        if values.ndim != 4 or values.shape[1] != len(self.features):
            raise ValueError('static attention requires joint targets, five own controls and allocated injection')
        return self.layer(values + self.conditioner(self.features)[None, :, None], *args, **kwargs)


def load_selected_layer(layer, head, selected):
    if selected.get('static_last_layer', False):
        layer = StaticConditionedLayer(layer, head)
        torch.testing.assert_close(selected['last_layer']['features'], layer.features, rtol=0, atol=0)
    if 'last_layer' in selected:
        layer.load_state_dict(selected['last_layer'])
    return layer


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
    layer = StaticConditionedLayer(nn.Identity(), head)
    embeddings = torch.ones((1, 18, 2, 4))
    torch.testing.assert_close(layer(embeddings), embeddings, rtol=0, atol=0)
    # Targets: 3 rows/well; own controls: 5 rows/well; then one allocated-injection row/well.
    torch.testing.assert_close(layer.features[[0, 6, 16], :-9], head.features[[0, 0, 0], :-3])
    torch.testing.assert_close(layer.features[[3, 11, 17], :-9], head.features[[3, 3, 3], :-3])
    assert layer.features[:, -9:].argmax(dim=1).tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 8]
    layer(embeddings).square().sum().backward()
    assert torch.count_nonzero(layer.conditioner.weight.grad)
    from copy import deepcopy
    with torch.no_grad():
        layer.conditioner.weight.fill_(.01)
    bundle = {'static_last_layer': True, 'last_layer': deepcopy(layer.state_dict())}
    restored = load_selected_layer(nn.Identity(), head, bundle)
    torch.testing.assert_close(restored(embeddings), layer(embeddings), rtol=0, atol=0)
    assert isinstance(load_selected_layer(nn.Identity(), head, {'last_layer': {}}), nn.Identity)
    bundle['last_layer']['features'][0, 0] += 1
    try:
        load_selected_layer(nn.Identity(), head, bundle)
    except AssertionError:
        pass
    else:
        raise AssertionError('mismatched checkpoint geology accepted')
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
