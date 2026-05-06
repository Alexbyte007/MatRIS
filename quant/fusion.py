from __future__ import annotations

import fnmatch

from torch import nn

from matris.model.functions import FusedInputGatedMLP, GatedMLP


_GATED_MLP_FUSION_TARGETS: dict[str, list[str]] = {
    "line_attn_edge_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
    ],
    "line_refine_edge_gated_mlp_fused_fp32": [
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
    ],
    "line_attn_node_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update",
    ],
    "line_edge_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
    ],
    "line_node_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update",
    ],
    "line_all_candidate_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update",
    ],
    "line_edge_gated_mlp_second_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
    ],
    "line_edge_gated_mlp_tail_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
    ],
    "line_edge_gated_mlp_second_tail_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
    ],
    "line_all_candidate_gated_mlp_second_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update",
    ],
    "line_all_candidate_gated_mlp_tail_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update",
    ],
    "line_all_candidate_gated_mlp_second_tail_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update",
    ],
    "atom_attn_edge_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
    ],
    "atom_refine_edge_gated_mlp_fused_fp32": [
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
    ],
    "atom_edge_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
    ],
    "atom_edge_gated_mlp_second_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
    ],
    "atom_edge_gated_mlp_tail_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
    ],
    "atom_edge_gated_mlp_second_tail_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
    ],
    "atom_attn_node_gated_mlp_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
    ],
    "atom_attn_node_gated_mlp_second_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
    ],
    "atom_attn_node_gated_mlp_tail_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
    ],
    "atom_attn_node_gated_mlp_second_tail_fused_fp32": [
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
    ],
    "all_passed_gated_mlp_second_fused_fp32": [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update",
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
    ],
}


def get_gated_mlp_fusion_targets(fusion_mode: str | None) -> list[str]:
    if fusion_mode in (None, "", "none"):
        return []
    if fusion_mode not in _GATED_MLP_FUSION_TARGETS:
        known_modes = ", ".join(["none", *sorted(_GATED_MLP_FUSION_TARGETS)])
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}. Known modes: {known_modes}")
    return _GATED_MLP_FUSION_TARGETS[fusion_mode]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _parent_and_child(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    modules = dict(model.named_modules())
    parent_name, child_name = module_name.rsplit(".", 1)
    return modules[parent_name], child_name


def apply_gated_mlp_fusion(model: nn.Module, fusion_mode: str | None) -> list[str]:
    targets = get_gated_mlp_fusion_targets(fusion_mode)
    if not targets:
        print("[fusion] mode=none replaced 0 GatedMLP modules")
        return []
    fuse_second = fusion_mode is not None and (
        fusion_mode.endswith("_second_fused_fp32")
        or fusion_mode.endswith("_second_tail_fused_fp32")
    )
    fuse_tail = fusion_mode is not None and (
        fusion_mode.endswith("_tail_fused_fp32")
        or fusion_mode.endswith("_second_tail_fused_fp32")
    )

    replaced: list[str] = []
    skipped: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not _matches_any(module_name, targets):
            continue
        if isinstance(module, FusedInputGatedMLP):
            skipped.append(module_name)
            continue
        if not isinstance(module, GatedMLP) or not FusedInputGatedMLP.can_fuse(module):
            skipped.append(module_name)
            continue

        parent, child_name = _parent_and_child(model, module_name)
        fused_module = FusedInputGatedMLP.from_gated_mlp(
            module,
            module_name,
            fuse_second=fuse_second,
            fuse_tail=fuse_tail,
        )
        if fuse_second and fused_module.core_second is None:
            skipped.append(module_name)
            continue
        setattr(parent, child_name, fused_module)
        replaced.append(module_name)

    print(f"[fusion] mode={fusion_mode} replaced {len(replaced)} GatedMLP modules")
    for name in replaced:
        print(f"[fusion] replaced: {name}")
    for name in skipped:
        print(f"[fusion] skipped: {name}")
    return replaced
