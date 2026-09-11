"""Static conditioning of the native Google output head, after temporal RevIN."""

import numpy as np
import torch
from torch import nn


MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
TARGET_COUNT = 9  # The nine canonical economic outputs are the only trained shape.


def forecast_inputs(states, actions, origin, context_length, horizon):
    """Align action[t] with state[t+1]; retain BHP when present and exclude future targets."""
    states, actions = np.asarray(states), np.asarray(actions)
    if (states.ndim != 3 or states.shape[-1] != TARGET_COUNT or actions.ndim != 3
            or actions.shape[:2] != states.shape[:2] or actions.shape[-1] not in (3, 4)):
        raise ValueError("forecast requires aligned nine-target states and three/four-feature actions")
    if context_length < 1 or horizon < 1 or origin < 1 or origin + horizon >= len(states):
        raise ValueError("invalid forecast window")
    start = max(1, origin + 1 - context_length)
    targets = states[start : origin + 1].transpose(1, 2, 0).astype(np.float32)
    controls = actions[start - 1 : origin + horizon]
    if (not np.isfinite(targets).all() or not np.isfinite(controls).all()
            or (controls[..., 0] < 0).any()
            or not np.isin(controls[..., 1], (0, 1, 2)).all()
            or not np.isin(controls[..., 2], (0, 1)).all()
            or (controls[..., 3:] < 0).any()):
        raise ValueError("invalid observed states or planned forecast controls")
    rates = np.stack(
        [np.where(controls[..., 1] == code, controls[..., 0], 0.0)
         * controls[..., 2] for code in range(3)], axis=-1
    )
    field_injection = np.broadcast_to(rates[..., 2].sum(axis=1)[:, None], rates.shape[:2])
    covariates = np.concatenate(
        [rates, controls[..., 2:], field_injection[..., None]], axis=-1
    ).transpose(1, 2, 0).astype(np.float32)
    return targets, covariates


def enable_precise_variate_softmax(model):
    """Avoid observed CUDA FP32 softmax repeatability drift on full-field variates."""
    from unittest.mock import patch
    from torch.nn import functional as functional

    if getattr(model, 'precise_variate_softmax', False):
        return
    layers = [module for name, module in model.named_modules() if name.rsplit('.', 1)[-1] == 'var_attn']
    if not layers or any(not hasattr(layer, 'use_sdpa') for layer in layers):
        raise ValueError('native variate attention layers required')
    def softmax(values, dim=None, _stacklevel=3, dtype=None):
        return torch.softmax(values, dim=dim, dtype=torch.float64).to(dtype or values.dtype)
    for layer in layers:
        layer.use_sdpa = False
        original = layer.forward
        def forward(*args, _original=original, **kwargs):
            # ponytail: process-local patch in single-threaded inference/training; use native precision control when available.
            with patch.object(functional, 'softmax', softmax):
                return _original(*args, **kwargs)
        layer.forward = forward
    model.precise_variate_softmax = True


def geological_inputs(trajectory, origin, context, horizon, connectivity):
    if tuple(trajectory.well_ids) != tuple(connectivity.well_ids):
        raise ValueError('geological well order differs from trajectory')
    target, cov = forecast_inputs(trajectory.states, trajectory.actions, origin, context, horizon)
    count, target_count, length = target.shape
    controls = trajectory.actions[origin - length:origin + horizon]
    allocated = connectivity.features(np.zeros((len(controls) * count, 3)),
        controls.reshape(-1, controls.shape[-1]))[:, 0].reshape(len(controls), count).T
    return target.reshape(count * target_count, length), np.concatenate(
        [cov[:, :-1].reshape(-1, length + horizon), allocated], axis=0)


class StaticConditionedHead(nn.Module):
    def __init__(self, head, connectivity):
        super().__init__()
        target_count = self.target_count = TARGET_COUNT
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
    target_count = getattr(model.output_head, 'target_count', None)
    if target_count != TARGET_COUNT:
        raise ValueError('cold-start normalization requires a conditioned nine-target head')
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
    from timesfm_economics import ECONOMIC_TARGETS
    if tuple(selected.get('economic_targets') or ()) != ECONOMIC_TARGETS:
        raise ValueError('checkpoint economic target order differs')
    model.output_head = StaticConditionedHead(model.output_head, connectivity)
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
    if type(selected.get('precise_variate_softmax', False)) is not bool:
        raise ValueError('invalid variate softmax precision metadata')
    if selected.get('precise_variate_softmax', False):
        enable_precise_variate_softmax(model)
    return model


def self_check():
    from copy import deepcopy
    from types import SimpleNamespace
    class Attention(nn.Module):
        use_sdpa = True
        def forward(self, values):
            return torch.nn.functional.softmax(values, dim=-1)
    probe = nn.Module(); probe.var_attn = Attention()
    original_softmax = torch.nn.functional.softmax
    enable_precise_variate_softmax(probe)
    enable_precise_variate_softmax(probe)
    values = torch.tensor([[2., -3., 5.]], requires_grad=True)
    expected = torch.softmax(values.double(), dim=-1).float()
    actual = probe.var_attn(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual[..., 0].sum(), values)[0],
                               torch.autograd.grad(expected[..., 0].sum(), values)[0], rtol=0, atol=0)
    assert not probe.var_attn.use_sdpa and torch.nn.functional.softmax is original_softmax
    # Control lag, future-target leakage and the planned-control cube guard.
    states = np.arange(40 * 2 * 9, dtype=float).reshape(40, 2, 9)
    actions = np.zeros((40, 2, 3))
    actions[..., 0] = np.arange(40)[:, None] + 1
    actions[:, 0, 1], actions[:, 1, 1] = 1, 2
    actions[..., 2] = 1
    target, cov = forecast_inputs(states, actions, 20, 12, 6)
    assert target.shape == (2, 9, 12) and cov.shape == (2, 5, 18)
    np.testing.assert_array_equal(target[0], states[9:21, 0].T)
    np.testing.assert_array_equal(cov[0, 1, :12], actions[8:20, 0, 0])
    np.testing.assert_array_equal(cov[0, 1, 12:], actions[20:26, 0, 0])
    changed = states.copy()
    changed[21:] = -12345
    np.testing.assert_array_equal(forecast_inputs(changed, actions, 20, 12, 6)[0], target)
    other = actions.copy()
    other[20:26, 1, 0] *= 2
    _, other_cov = forecast_inputs(states, other, 20, 12, 6)
    np.testing.assert_array_equal(other_cov[0, :4], cov[0, :4])
    np.testing.assert_array_equal(other_cov[0, 4, 12:], 2 * cov[0, 4, 12:])
    bhp_actions = np.concatenate([actions, np.full((*actions.shape[:2], 1), 70.0)], axis=-1)
    bhp_actions[20:26, 0, 3] = 90.0
    bhp_target, bhp_cov = forecast_inputs(states, bhp_actions, 20, 12, 6)
    assert bhp_cov.shape == (2, 6, 18)
    np.testing.assert_array_equal(bhp_target, target)
    np.testing.assert_array_equal(bhp_cov[:, :4], cov[:, :-1])
    np.testing.assert_array_equal(bhp_cov[:, -1], cov[:, -1])
    np.testing.assert_array_equal(bhp_cov[0, 4, :12], 70.0)
    np.testing.assert_array_equal(bhp_cov[0, 4, 12:], 90.0)
    for channel, invalid in ((3, np.nan), (0, -1.), (1, 3.), (2, 2.), (3, -1.)):
        rejected = bhp_actions.copy()
        rejected[20, 0, channel] = invalid
        try:
            forecast_inputs(states, rejected, 20, 12, 6)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid planned control accepted')
    try:
        forecast_inputs(states[..., :3], actions, 20, 12, 6)
    except ValueError:
        pass
    else:
        raise AssertionError('three-target states accepted')
    connection = SimpleNamespace(well_ids=('a', 'b'), static=[[10, .1, 1], [20, .2, 3]], provenance={})
    base = nn.Linear(4, 9)
    head = StaticConditionedHead(base, connection)
    values = torch.ones((1, 20, 2, 4))
    torch.testing.assert_close(head(values), base(values), rtol=0, atol=0)
    with torch.no_grad():
        head.conditioner.weight[:, 0] = 1
    changed = head(values)
    assert not torch.equal(changed[:, 0], changed[:, 9])
    torch.testing.assert_close(changed[:, 18:], base(values)[:, 18:], rtol=0, atol=0)
    head.disabled = True
    torch.testing.assert_close(head(values), base(values), rtol=0, atol=0)
    head.disabled = False
    layer = StaticConditionedLayer(nn.Identity(), head)
    embeddings = torch.ones((1, 30, 2, 4))
    torch.testing.assert_close(layer(embeddings), embeddings, rtol=0, atol=0)
    # Targets: 9 rows/well; own controls: 5 rows/well; then one allocated-injection row/well.
    torch.testing.assert_close(layer.features[[0, 18, 28], :-15], head.features[[0, 0, 0], :-9])
    torch.testing.assert_close(layer.features[[9, 23, 29], :-15], head.features[[9, 9, 9], :-9])
    assert (layer.features[:, -15:].argmax(dim=1).tolist()
            == list(range(9)) * 2 + list(range(9, 14)) * 2 + [14, 14])
    layer(embeddings).square().sum().backward()
    assert torch.count_nonzero(layer.conditioner.weight.grad)
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
    from timesfm_economics import ECONOMIC_TARGETS
    native = nn.Module()
    native.output_head = nn.Linear(4, 6)
    native.transformer_stack = nn.Module()
    native.transformer_stack.layers = nn.ModuleList([nn.Linear(4, 4), nn.Identity()])
    adapted = deepcopy(native)
    adapted.output_head = StaticConditionedHead(adapted.output_head, connection)
    adapted.transformer_stack.layers[-1] = StaticConditionedLayer(
        adapted.transformer_stack.layers[-1], adapted.output_head)
    with torch.no_grad():
        adapted.transformer_stack.layers[0].weight.fill_(.125)
    full = {'full_model': adapted.state_dict(), 'static_last_layer': True,
            'economic_targets': list(ECONOMIC_TARGETS)}
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
    for invalid in ({'economic_targets': list(reversed(ECONOMIC_TARGETS))},
                    {'economic_targets': None}):
        try:
            load_frozen_model(deepcopy(native), connection, {**full, **invalid})
        except ValueError:
            pass
        else:
            raise AssertionError('checkpoint without the canonical nine targets accepted')
    early = load_frozen_model(deepcopy(native), connection, full)
    first = early.transformer_stack.layers[0]
    early.transformer_stack.layers[0] = StaticConditionedLayer(first, early.output_head)
    torch.testing.assert_close(early.transformer_stack.layers[0](embeddings), first(embeddings), rtol=0, atol=0)
    with torch.no_grad():
        early.transformer_stack.layers[0].conditioner.weight[:, 0] = .1
    early_output = early.transformer_stack.layers[0](embeddings)
    assert not torch.equal(early_output[:, 0], early_output[:, 9])
    early_bundle = {**full, 'full_model': deepcopy(early.state_dict()), 'static_first_layer': True}
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
    sigma = torch.zeros(1, 20, 1)
    sigma[0, 1], sigma[0, 4], sigma[0, 18] = 2., 3., 5.
    mean = torch.zeros_like(sigma)
    original_stats = (None, None, None, (mean, sigma), torch.ones_like(sigma))
    native._preprocess = lambda *args, **kwargs: original_stats
    scales = [10., 20., 250., 30., 40., 50., 60., 70., 80.]
    adapted = load_frozen_model(deepcopy(native), connection, {**full, 'cold_start_scale': scales})
    adjusted = adapted._preprocess()[3][1]
    expected = torch.tensor(scales * 2 + [5., 0.]).reshape(1, 20, 1)
    expected[0, 1], expected[0, 4] = 2., 3.
    torch.testing.assert_close(adjusted, expected)
    logits = torch.ones((1, 20, 1, 2), requires_grad=True)
    fixed = util.revin(logits, mean, adjusted, reverse=True)
    fixed.sum().backward()
    assert logits.grad[0, 2, 0, 0] == 250
    unchanged = util.revin(logits, mean, sigma, reverse=True)
    torch.testing.assert_close(fixed[:, [1, 4, 18, 19]], unchanged[:, [1, 4, 18, 19]])
    assert not torch.count_nonzero(unchanged[:, 2])
    torch.testing.assert_close(native._preprocess()[3][1], sigma, rtol=0, atol=0)
    enable_cold_start_normalization(adapted, scales)
    for invalid in ([0.] + scales[1:], [10., float('nan')] + scales[2:], scales[:8]):
        try:
            enable_cold_start_normalization(
                load_frozen_model(deepcopy(native), connection, full), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid cold-start normalization accepted')
    from timesoil.aios.interwell import WellConnectivity
    connection = WellConnectivity(('a', 'b'), [[0, 1], [1, 0]], [[10, .1, 1], [20, .2, 3]], {})
    economic_states = np.ones((12, 2, 9))
    actions = np.ones((12, 2, 4)); actions[:, 1, 1] = 2
    trajectory = SimpleNamespace(well_ids=('a', 'b'), states=economic_states, actions=actions)
    target, cov = geological_inputs(trajectory, 5, 4, 3, connection)
    economic_states[6:] = np.nan
    other_target, other_cov = geological_inputs(trajectory, 5, 4, 3, connection)
    np.testing.assert_array_equal(target, other_target)
    np.testing.assert_array_equal(cov, other_cov)
    assert target.shape == (18, 4) and cov.shape == (12, 7)
    print('Static conditioning parity, well identity, ablation and covariate isolation passed', flush=True)


if __name__ == '__main__':
    self_check()
