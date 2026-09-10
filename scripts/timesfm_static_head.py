"""Static geological conditioning compatible with the verified Google head checkpoint."""

import numpy as np
import torch
from torch import nn


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


def load_frozen_model(model, connectivity, selected, *, reference_sha256=None):
    if selected.get('reference_manifest_sha256') not in (None, reference_sha256):
        raise ValueError('trained response weights require the matching physical reference')
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
    if full is not None:
        model.load_state_dict(full)
    return model

