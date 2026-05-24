from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import Tensor, nn

from .functions import (
    Dimwise_softmax,
    aggregate,
    fused_line_attention_or_none,
    segment_softmax_weighted_sum_sorted_or_none,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def use_p71_line_attn_eval_only_vjp(profile_prefix: str) -> bool:
    if os.environ.get("MATRIS_P71_LINE_ATTN_EVAL_ONLY_VJP", "0") != "1":
        return False
    if not profile_prefix.startswith("interaction_block.") or not profile_prefix.endswith(".attn_line"):
        return False
    try:
        block_idx = int(profile_prefix.split(".")[1])
    except (IndexError, ValueError):
        return False
    min_block = _env_int("MATRIS_P71_MIN_BLOCK", 0)
    max_block = _env_int("MATRIS_P71_MAX_BLOCK", 9)
    return min_block <= block_idx <= max_block


@contextmanager
def _temporarily_disable_param_grad(module: nn.Module) -> Iterator[None]:
    params = list(module.parameters())
    old_requires_grad = [bool(param.requires_grad) for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, requires_grad in zip(params, old_requires_grad):
            param.requires_grad_(requires_grad)


def _line_attention_plain_forward(
    layer: nn.Module,
    node_feat: Tensor,
    edge_feat: Tensor,
    graph: Dict,
) -> tuple[Tensor, Tensor]:
    source_index = graph["source_index"]
    target_index = graph["target_index"]

    source_node_feat = torch.index_select(node_feat, 0, source_index)
    target_node_feat = torch.index_select(node_feat, 0, target_index)
    edge_update_input = torch.cat([edge_feat, target_node_feat, source_node_feat], dim=1)
    edge_update = layer.edge_nonlinear_update(edge_update_input)

    source_logits = layer.source_weight_linear(edge_feat)
    target_logits = layer.target_weight_linear(edge_feat)

    fused_attention = fused_line_attention_or_none(
        source_logits,
        target_logits,
        edge_update,
        source_index,
        target_index,
        len(node_feat),
        enable_hint=True,
        atom_graph=False,
    )
    if fused_attention is None:
        source_alpha = Dimwise_softmax(
            source_logits,
            source_index,
            profile_name="p71.attn_line.source_softmax",
        )
        target_alpha = Dimwise_softmax(
            target_logits,
            target_index,
            profile_name="p71.attn_line.target_softmax",
            bin_count=graph.get("target_bincount"),
        )
        source_out = aggregate(
            data=source_alpha * edge_update,
            segment=source_index,
            bin_count=graph["source_bincount"],
            average=False,
            num_segment=len(node_feat),
            profile_name="p71.attn_line.source_sum",
        )
        target_out = segment_softmax_weighted_sum_sorted_or_none(
            target_logits,
            edge_update,
            target_index,
            len(node_feat),
            enable_hint=graph.get("target_bincount") is not None,
        )
        if target_out is None:
            target_out = aggregate(
                data=target_alpha * edge_update,
                segment=target_index,
                bin_count=graph["target_bincount"],
                average=False,
                num_segment=len(node_feat),
                profile_name="p71.attn_line.target_sum",
            )
    else:
        source_out, target_out = fused_attention

    node_update_input = torch.cat([node_feat, target_out, source_out], dim=1)
    node_update = layer.node_nonlinear_update(node_update_input)

    return (
        node_update + layer.node_res_weight * node_feat,
        edge_update + layer.edge_res_weight * edge_feat,
    )


class _P71LineAttentionEvalOnlyVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer: nn.Module, graph: Dict, node_feat: Tensor, edge_feat: Tensor):
        ctx.layer = layer
        ctx.graph = graph
        ctx.save_for_backward(node_feat.detach(), edge_feat.detach())
        with torch.no_grad(), _temporarily_disable_param_grad(layer):
            out_node, out_edge = _line_attention_plain_forward(layer, node_feat.detach(), edge_feat.detach(), graph)
        return out_node.detach(), out_edge.detach()

    @staticmethod
    def backward(ctx, grad_node_out: Tensor | None, grad_edge_out: Tensor | None):
        node_saved, edge_saved = ctx.saved_tensors
        node = node_saved.detach().requires_grad_(True)
        edge = edge_saved.detach().requires_grad_(True)
        with torch.enable_grad(), _temporarily_disable_param_grad(ctx.layer):
            out_node, out_edge = _line_attention_plain_forward(ctx.layer, node, edge, ctx.graph)
            outputs: list[Tensor] = []
            grad_outputs: list[Tensor] = []
            if grad_node_out is not None:
                outputs.append(out_node)
                grad_outputs.append(grad_node_out)
            if grad_edge_out is not None:
                outputs.append(out_edge)
                grad_outputs.append(grad_edge_out)
            if outputs:
                grad_node, grad_edge = torch.autograd.grad(
                    outputs,
                    (node, edge),
                    grad_outputs=grad_outputs,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            else:
                grad_node = grad_edge = None
        return None, None, grad_node, grad_edge


def p71_line_attention_eval_only_vjp(
    layer: nn.Module,
    graph: Dict,
    node_feat: Tensor,
    edge_feat: Tensor,
) -> tuple[Tensor, Tensor]:
    return _P71LineAttentionEvalOnlyVJP.apply(layer, graph, node_feat, edge_feat)

