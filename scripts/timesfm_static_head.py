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

