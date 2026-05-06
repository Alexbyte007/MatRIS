from __future__ import annotations

import torch
from torch import nn

from .layers import TritonW8A8StaticLinear


def _to_json_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return value


def collect_quant_stats(model: nn.Module) -> list[dict]:
    stats = []
    for _, module in model.named_modules():
        if isinstance(module, TritonW8A8StaticLinear) and module.last_stats:
            stats.append(
                {key: _to_json_value(value) for key, value in module.last_stats.items()}
            )
    return stats
