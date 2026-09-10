"""Static conditioning of the native Google output head, after temporal RevIN."""

import numpy as np
import torch
from torch import nn


def geological_inputs(trajectory, origin, context, horizon, connectivity):
    from benchmark_timesfm3 import forecast_inputs
    if tuple(trajectory.well_ids) != tuple(connectivity.well_ids):
        raise ValueError('geological well order differs from trajectory')
    target, cov = forecast_inputs(trajectory.states, trajectory.actions, origin, context, horizon,
                                 target_count=trajectory.states.shape[-1])
    count, target_count, length = target.shape
    controls = trajectory.actions[origin - length:origin + horizon]
    allocated = connectivity.features(np.zeros((len(controls) * count, 3)),
        controls.reshape(-1, controls.shape[-1]))[:, 0].reshape(len(controls), count).T
    return target.reshape(count * target_count, length), np.concatenate(
        [cov[:, :-1].reshape(-1, length + horizon), allocated], axis=0)


class StaticConditionedHead(nn.Module):
    def __init__(self, head, connectivity, *, target_count=3):
        super().__init__()
        if target_count not in (3, 9):
            raise ValueError('three physical or nine economic targets required')
        self.target_count = target_count
        self.head = head
        raw = np.asarray(connectivity.provenance.get('static_features', connectivity.static), dtype=float)
        if raw.ndim != 2 or len(raw) != len(connectivity.well_ids) or not np.isfinite(raw).all():
            raise ValueError('invalid static geological features')
        transformed = np.sign(raw) * np.log1p(np.abs(raw))
        mean, scale = transformed.mean(axis=0), transformed.std(axis=0)
        features = (transformed - mean) / np.maximum(scale, 1e-6)
        # Each target gets its own well's geology and an explicit target identity.
        features = np.column_stack([np.repeat(features, target_count, axis=0),
                                    np.tile(np.eye(target_count), (len(raw), 1))])
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
        count = head.target_count
        geology = head.features[::count, :-count]
        kinds = torch.eye(count + 6, device=geology.device, dtype=geology.dtype)
        features = torch.cat([torch.cat([geology.repeat_interleave(len(group), dim=0),
            kinds[group].repeat(len(geology), 1)], dim=1) for group in
            (list(range(count)), list(range(count, count + 5)), [count + 5])])
        self.register_buffer('features', features)
        self.conditioner = nn.Linear(features.shape[1], head.head.in_features, bias=False,
            device=geology.device, dtype=geology.dtype)
        nn.init.zeros_(self.conditioner.weight)

    def forward(self, values, *args, **kwargs):
        if values.ndim != 4 or values.shape[1] != len(self.features):
            raise ValueError('static attention requires joint targets, five own controls and allocated injection')
        if not getattr(self, 'disabled', False):
            values = values + self.conditioner(self.features)[None, :, None]
        return self.layer(values, *args, **kwargs)


def load_selected_layer(layer, head, selected):
    if selected.get('static_last_layer', False):
        layer = StaticConditionedLayer(layer, head)
        torch.testing.assert_close(selected['last_layer']['features'], layer.features, rtol=0, atol=0)
    if 'last_layer' in selected:
        layer.load_state_dict(selected['last_layer'])
    return layer


def enable_cold_start_normalization(model, scales):
    """Give constant target histories a train-only scale for native output RevIN."""
    scales = np.asarray(scales, dtype=float)
    target_count = model.output_head.target_count
    if scales.shape != (target_count,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('cold-start scales must match finite positive target training statistics')
    if hasattr(model, 'cold_start_scale'):
        np.testing.assert_array_equal(model.cold_start_scale, scales)
        return
    count = len(model.output_head.features)
    if count % target_count:
        raise ValueError('cold-start normalization requires complete joint targets')
    original = model._preprocess

    def preprocess(*args, **kwargs):
        result = original(*args, **kwargs)
        mean, sigma = result[3]
        fallback = torch.as_tensor(np.tile(scales, count // target_count), device=sigma.device, dtype=sigma.dtype)[None, :, None]
        # Constant inputs normalize to zero under either scale; preserve covariate statistics.
        adjusted = torch.cat([torch.where(sigma[:, :count] == 0, fallback, sigma[:, :count]), sigma[:, count:]], dim=1)
        return (*result[:3], (mean, adjusted), result[4])

    model._preprocess = preprocess
    model.cold_start_scale = scales.tolist()


def load_frozen_model(model, connectivity, selected, *, reference_sha256=None):
    if selected.get('reference_manifest_sha256') not in (None, reference_sha256):
        raise ValueError('trained response weights require the matching physical reference')
    economic = selected.get('economic_targets')
    if economic is not None:
        from timesfm_economics import ECONOMIC_TARGETS
        if tuple(economic) != ECONOMIC_TARGETS:
            raise ValueError('checkpoint economic target order differs')
    model.output_head = StaticConditionedHead(model.output_head, connectivity,
                                              target_count=9 if economic is not None else 3)
    full = selected.get('full_model')
    head_weights = ({k.removeprefix('output_head.'): v for k, v in full.items() if k.startswith('output_head.')}
                    if full is not None else selected.get('output_head', selected))
    torch.testing.assert_close(head_weights['features'], model.output_head.features, rtol=0, atol=0)
    model.output_head.load_state_dict(head_weights)
    layer_selected = selected
    if full is not None:
        prefix = f'transformer_stack.layers.{len(model.transformer_stack.layers) - 1}.'
        layer_selected = {'static_last_layer': selected.get('static_last_layer', False),
            'last_layer': {k.removeprefix(prefix): v for k, v in full.items() if k.startswith(prefix)}}
    model.transformer_stack.layers[-1] = load_selected_layer(
        model.transformer_stack.layers[-1], model.output_head, layer_selected)
    if selected.get('static_first_layer', False):
        if full is None:
            raise ValueError('first-layer conditioning requires complete model weights')
        model.transformer_stack.layers[0] = StaticConditionedLayer(model.transformer_stack.layers[0], model.output_head)
        torch.testing.assert_close(full['transformer_stack.layers.0.features'],
                                   model.transformer_stack.layers[0].features, rtol=0, atol=0)
    if full is not None:
        model.load_state_dict(full)
    if 'cold_start_scale' in selected:
        enable_cold_start_normalization(model, selected['cold_start_scale'])
    return model


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
    restored.disabled = True
    torch.testing.assert_close(restored(embeddings), embeddings, rtol=0, atol=0)
    assert isinstance(load_selected_layer(nn.Identity(), head, {'last_layer': {}}), nn.Identity)
    bundle['last_layer']['features'][0, 0] += 1
    try:
        load_selected_layer(nn.Identity(), head, bundle)
    except AssertionError:
        pass
    else:
        raise AssertionError('mismatched checkpoint geology accepted')
    native = nn.Module()
    native.output_head = nn.Linear(4, 9)
    native.transformer_stack = nn.Module()
    native.transformer_stack.layers = nn.ModuleList([nn.Linear(4, 4), nn.Identity()])
    adapted = load_frozen_model(deepcopy(native), connection, head.state_dict())
    adapted.transformer_stack.layers[-1] = StaticConditionedLayer(adapted.transformer_stack.layers[-1], adapted.output_head)
    with torch.no_grad():
        adapted.transformer_stack.layers[0].weight.fill_(.125)
    full = {'full_model': adapted.state_dict(), 'static_last_layer': True}
    sealed = {**full, 'reference_manifest_sha256': 'a' * 64}
    try:
        load_frozen_model(deepcopy(native), connection, sealed)
    except ValueError:
        pass
    else:
        raise AssertionError('response checkpoint loaded without its physical reference')
    load_frozen_model(deepcopy(native), connection, sealed, reference_sha256='a' * 64)
    restored = load_frozen_model(deepcopy(native), connection, full)
    for key, value in adapted.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert not torch.equal(native.transformer_stack.layers[0].weight, restored.transformer_stack.layers[0].weight)
    early = load_frozen_model(deepcopy(native), connection, full)
    first = early.transformer_stack.layers[0]
    early.transformer_stack.layers[0] = StaticConditionedLayer(first, early.output_head)
    torch.testing.assert_close(early.transformer_stack.layers[0](embeddings), first(embeddings), rtol=0, atol=0)
    with torch.no_grad():
        early.transformer_stack.layers[0].conditioner.weight[:, 0] = .1
    early_output = early.transformer_stack.layers[0](embeddings)
    assert not torch.equal(early_output[:, 0], early_output[:, 3])
    early_bundle = {'full_model': deepcopy(early.state_dict()), 'static_first_layer': True, 'static_last_layer': True}
    restored_early = load_frozen_model(deepcopy(native), connection, early_bundle)
    torch.testing.assert_close(restored_early.transformer_stack.layers[0](embeddings), early_output, rtol=0, atol=0)
    early_bundle['full_model']['transformer_stack.layers.0.features'][0, 0] += 1
    try:
        load_frozen_model(deepcopy(native), connection, early_bundle)
    except AssertionError:
        pass
    else:
        raise AssertionError('mismatched first-layer geological identity accepted')
    from timesfm3.torch import util
    sigma = torch.tensor([[[0.], [2.], [0.], [0.], [3.], [0.], [0.], [2.]]])
    mean = torch.zeros_like(sigma)
    original_stats = (None, None, None, (mean, sigma), torch.ones_like(sigma))
    native._preprocess = lambda *args, **kwargs: original_stats
    adapted = load_frozen_model(deepcopy(native), connection,
        {**full, 'cold_start_scale': [10., 20., 250.]})
    adjusted = adapted._preprocess()[3][1]
    torch.testing.assert_close(adjusted.flatten(), torch.tensor([10., 2., 250., 10., 3., 250., 0., 2.]))
    logits = torch.ones((1, 8, 1, 2), requires_grad=True)
    fixed = util.revin(logits, mean, adjusted, reverse=True)
    fixed.sum().backward()
    assert logits.grad[0, 2, 0, 0] == 250
    unchanged = util.revin(logits, mean, sigma, reverse=True)
    torch.testing.assert_close(fixed[:, [1, 4, 6, 7]], unchanged[:, [1, 4, 6, 7]])
    assert not torch.count_nonzero(unchanged[:, 2])
    torch.testing.assert_close(native._preprocess()[3][1], sigma, rtol=0, atol=0)
    enable_cold_start_normalization(adapted, [10., 20., 250.])
    for invalid in ([0., 20., 250.], [10., float('nan'), 250.], [10., 20.]):
        try:
            enable_cold_start_normalization(deepcopy(native), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid cold-start normalization accepted')
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
    economic_states = np.ones((12, 2, 9))
    economic_trajectory = SimpleNamespace(well_ids=('a', 'b'), states=economic_states, actions=actions)
    target, cov = geological_inputs(economic_trajectory, 5, 4, 3, connection)
    economic_states[6:] = np.nan
    other_target, other_cov = geological_inputs(economic_trajectory, 5, 4, 3, connection)
    np.testing.assert_array_equal(target, other_target)
    np.testing.assert_array_equal(cov, other_cov)
    assert target.shape == (18, 4) and cov.shape == (12, 7)
    economic_native = nn.Module()
    economic_native.output_head = nn.Linear(4, 6)
    economic_native.transformer_stack = nn.Module()
    economic_native.transformer_stack.layers = nn.ModuleList([nn.Linear(4, 4)])
    economic_model = deepcopy(economic_native)
    economic_model.output_head = StaticConditionedHead(economic_model.output_head, connection, target_count=9)
    economic_model.transformer_stack.layers[0] = StaticConditionedLayer(
        economic_model.transformer_stack.layers[0], economic_model.output_head)
    economic_layer = economic_model.transformer_stack.layers[0]
    assert economic_layer.features.shape[0] == 30
    assert economic_layer.features[:, -15:].argmax(dim=1).tolist() == list(range(9)) * 2 + list(range(9, 14)) * 2 + [14, 14]
    economic_layer(torch.ones(1, 30, 2, 4)).sum().backward()
    assert torch.count_nonzero(economic_layer.conditioner.weight.grad)
    from timesfm_economics import ECONOMIC_TARGETS
    bundle = {'full_model': economic_model.state_dict(), 'static_last_layer': True,
              'economic_targets': list(ECONOMIC_TARGETS)}
    restored = load_frozen_model(deepcopy(economic_native), connection, bundle)
    for key, value in economic_model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    bundle['economic_targets'] = list(reversed(ECONOMIC_TARGETS))
    try:
        load_frozen_model(deepcopy(economic_native), connection, bundle)
    except ValueError:
        pass
    else:
        raise AssertionError('reordered economic checkpoint targets accepted')
    mean = torch.zeros(1, 30, 1)
    restored._preprocess = lambda: (None, None, None, (mean, mean.clone()), torch.ones_like(mean))
    enable_cold_start_normalization(restored, np.arange(1, 10))
    torch.testing.assert_close(restored._preprocess()[3][1].flatten(),
                               torch.tensor(list(range(1, 10)) * 2 + [0.] * 12))
    print('Static conditioning parity, well identity, ablation and covariate isolation passed', flush=True)


if __name__ == '__main__':
    self_check()
