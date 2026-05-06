from __future__ import annotations

import copy

import torch
from torch import nn

from matris.model.functions import FusedGatedMLPTail, FusedInputGatedMLP, GatedMLP
from quant.config import get_quant_config
from quant.fusion import apply_gated_mlp_fusion
from quant.injector import apply_quant_config


class TinyLineAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_nonlinear_update = GatedMLP(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dropout=0.0,
            norm_type="layer",
        )
        self.node_nonlinear_update = GatedMLP(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dropout=0.0,
            norm_type="layer",
        )


class TinyLineRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_nonlinear_update = GatedMLP(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dropout=0.0,
            norm_type="layer",
        )


class TinyAtomAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_nonlinear_update = GatedMLP(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dropout=0.0,
            norm_type="layer",
        )
        self.node_nonlinear_update = GatedMLP(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dropout=0.0,
            norm_type="layer",
        )


class TinyAtomRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_nonlinear_update = GatedMLP(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            dropout=0.0,
            norm_type="layer",
        )


class TinyInteractionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_block_line_graph = TinyLineAttention()
        self.refine_block_line_graph = TinyLineRefinement()
        self.attn_block_atom_graph = TinyAtomAttention()
        self.refine_block_atom_graph = TinyAtomRefinement()


class TinyMatRISForFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.interaction_block = nn.ModuleList([TinyInteractionBlock(), TinyInteractionBlock()])


def _assert_forward_backward_close(original: nn.Module, fused: nn.Module) -> None:
    original.eval()
    fused.eval()
    x_original = torch.randn(5, 4, requires_grad=True)
    x_fused = x_original.detach().clone().requires_grad_(True)

    y_original = original(x_original)
    y_fused = fused(x_fused)
    torch.testing.assert_close(y_fused, y_original, rtol=1e-6, atol=1e-6)

    grad = torch.randn_like(y_original)
    y_original.backward(grad)
    y_fused.backward(grad)
    torch.testing.assert_close(x_fused.grad, x_original.grad, rtol=1e-6, atol=1e-6)


def test_fused_input_gated_mlp_matches_fp32_gated_mlp_forward_and_backward() -> None:
    torch.manual_seed(7)
    original = GatedMLP(input_dim=4, hidden_dim=4, output_dim=4, dropout=0.0, norm_type="layer")
    fused = FusedInputGatedMLP.from_gated_mlp(copy.deepcopy(original), "gated")

    _assert_forward_backward_close(original, fused)


def test_fused_second_gated_mlp_matches_fp32_gated_mlp_forward_and_backward() -> None:
    torch.manual_seed(17)
    original = GatedMLP(input_dim=4, hidden_dim=4, output_dim=4, dropout=0.0, norm_type="layer")
    fused = FusedInputGatedMLP.from_gated_mlp(
        copy.deepcopy(original),
        "gated",
        fuse_second=True,
    )

    assert fused.core_second is not None
    assert fused.gate_second is not None
    assert fused.fused_second is not None
    _assert_forward_backward_close(original, fused)


def test_fused_tail_gated_mlp_matches_fp32_gated_mlp_forward_and_backward() -> None:
    torch.manual_seed(23)
    original = GatedMLP(input_dim=4, hidden_dim=4, output_dim=4, dropout=0.0, norm_type="layer")
    fused = FusedInputGatedMLP.from_gated_mlp(
        copy.deepcopy(original),
        "gated_tail",
        fuse_tail=True,
    )

    assert isinstance(fused.fused_tail, FusedGatedMLPTail)
    assert fused.core_norm is None
    assert fused.gate_norm is None
    _assert_forward_backward_close(original, fused)


def test_fused_second_tail_gated_mlp_matches_fp32_gated_mlp_forward_and_backward() -> None:
    torch.manual_seed(29)
    original = GatedMLP(input_dim=4, hidden_dim=4, output_dim=4, dropout=0.0, norm_type="layer")
    fused = FusedInputGatedMLP.from_gated_mlp(
        copy.deepcopy(original),
        "gated_second_tail",
        fuse_second=True,
        fuse_tail=True,
    )

    assert fused.core_second is not None
    assert fused.gate_second is not None
    assert fused.fused_second is not None
    assert isinstance(fused.fused_tail, FusedGatedMLPTail)
    _assert_forward_backward_close(original, fused)


def test_fused_second_gated_mlp_declines_dropout_tail() -> None:
    original = GatedMLP(input_dim=4, hidden_dim=4, output_dim=4, dropout=0.1, norm_type="layer")
    fused = FusedInputGatedMLP.from_gated_mlp(
        copy.deepcopy(original),
        "gated_dropout",
        fuse_second=True,
    )

    assert fused.core_second is None
    assert fused.gate_second is None
    assert fused.fused_second is None


def test_fused_input_gated_mlp_matches_fake_quant_first_layers() -> None:
    torch.manual_seed(11)
    original = TinyMatRISForFusion()
    unfused = copy.deepcopy(original)
    fused_model = copy.deepcopy(original)

    apply_quant_config(unfused, get_quant_config("line_edge_core_gate_mlp_w8a32"))
    apply_quant_config(fused_model, get_quant_config("line_edge_core_gate_mlp_w8a32"))
    fused = apply_gated_mlp_fusion(fused_model, "line_attn_edge_gated_mlp_fused_fp32")

    assert fused == [
        "interaction_block.0.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.1.attn_block_line_graph.edge_nonlinear_update",
    ]
    _assert_forward_backward_close(
        unfused.interaction_block[0].attn_block_line_graph.edge_nonlinear_update,
        fused_model.interaction_block[0].attn_block_line_graph.edge_nonlinear_update,
    )


def test_fused_second_gated_mlp_matches_fake_quant_line_edge_layers() -> None:
    torch.manual_seed(19)
    original = TinyMatRISForFusion()
    unfused = copy.deepcopy(original)
    fused_model = copy.deepcopy(original)

    apply_quant_config(unfused, get_quant_config("line_edge_core_gate_mlp_w8a32"))
    apply_quant_config(fused_model, get_quant_config("line_edge_core_gate_mlp_w8a32"))
    fused = apply_gated_mlp_fusion(fused_model, "line_edge_gated_mlp_second_fused_fp32")

    assert fused == [
        "interaction_block.0.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.0.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.1.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.1.refine_block_line_graph.edge_nonlinear_update",
    ]
    fused_module = fused_model.interaction_block[0].attn_block_line_graph.edge_nonlinear_update
    assert isinstance(fused_module, FusedInputGatedMLP)
    assert fused_module.core_second is not None
    assert fused_module.gate_second is not None
    assert fused_module.fused_second is None
    _assert_forward_backward_close(
        unfused.interaction_block[0].attn_block_line_graph.edge_nonlinear_update,
        fused_module,
    )


def test_fused_second_tail_gated_mlp_matches_fake_quant_line_edge_layers() -> None:
    torch.manual_seed(31)
    original = TinyMatRISForFusion()
    unfused = copy.deepcopy(original)
    fused_model = copy.deepcopy(original)

    apply_quant_config(unfused, get_quant_config("line_edge_core_gate_mlp_w8a32"))
    apply_quant_config(fused_model, get_quant_config("line_edge_core_gate_mlp_w8a32"))
    fused = apply_gated_mlp_fusion(fused_model, "line_edge_gated_mlp_second_tail_fused_fp32")

    assert fused == [
        "interaction_block.0.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.0.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.1.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.1.refine_block_line_graph.edge_nonlinear_update",
    ]
    fused_module = fused_model.interaction_block[0].attn_block_line_graph.edge_nonlinear_update
    assert isinstance(fused_module, FusedInputGatedMLP)
    assert fused_module.core_second is not None
    assert fused_module.gate_second is not None
    assert fused_module.fused_second is None
    assert isinstance(fused_module.fused_tail, FusedGatedMLPTail)
    _assert_forward_backward_close(
        unfused.interaction_block[0].attn_block_line_graph.edge_nonlinear_update,
        fused_module,
    )


def test_p1_fusion_modes_replace_only_stable_line_gated_mlp_candidates() -> None:
    cases = {
        "line_attn_edge_gated_mlp_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
        ],
        "line_refine_edge_gated_mlp_fused_fp32": [
            "refine_block_line_graph.edge_nonlinear_update",
        ],
        "line_attn_node_gated_mlp_fused_fp32": [
            "attn_block_line_graph.node_nonlinear_update",
        ],
        "line_edge_gated_mlp_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
        ],
        "line_all_candidate_gated_mlp_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
            "attn_block_line_graph.node_nonlinear_update",
        ],
    }

    for mode, required_fragments in cases.items():
        model = TinyMatRISForFusion()
        replaced = apply_gated_mlp_fusion(model, mode)

        for fragment in required_fragments:
            assert sum(fragment in name for name in replaced) == 2
            assert isinstance(
                dict(model.named_modules())[f"interaction_block.0.{fragment}"],
                FusedInputGatedMLP,
            )
        assert all("attn_block_line_graph" in name or "refine_block_line_graph" in name for name in replaced)


def test_p1c_maybe_fusion_modes_replace_only_atom_edge_candidates() -> None:
    cases = {
        "atom_attn_edge_gated_mlp_fused_fp32": [
            "attn_block_atom_graph.edge_nonlinear_update",
        ],
        "atom_refine_edge_gated_mlp_fused_fp32": [
            "refine_block_atom_graph.edge_nonlinear_update",
        ],
        "atom_edge_gated_mlp_fused_fp32": [
            "attn_block_atom_graph.edge_nonlinear_update",
            "refine_block_atom_graph.edge_nonlinear_update",
        ],
    }

    for mode, required_fragments in cases.items():
        model = TinyMatRISForFusion()
        replaced = apply_gated_mlp_fusion(model, mode)

        for fragment in required_fragments:
            assert sum(fragment in name for name in replaced) == 2
            assert isinstance(
                dict(model.named_modules())[f"interaction_block.0.{fragment}"],
                FusedInputGatedMLP,
            )
        assert all("edge_nonlinear_update" in name for name in replaced)
        assert all("atom_graph" in name for name in replaced)
        assert all("node_nonlinear_update" not in name for name in replaced)


def test_p1d_defer_fusion_mode_replaces_only_atom_node_candidate() -> None:
    model = TinyMatRISForFusion()
    replaced = apply_gated_mlp_fusion(model, "atom_attn_node_gated_mlp_fused_fp32")

    assert replaced == [
        "interaction_block.0.attn_block_atom_graph.node_nonlinear_update",
        "interaction_block.1.attn_block_atom_graph.node_nonlinear_update",
    ]
    assert isinstance(
        dict(model.named_modules())["interaction_block.0.attn_block_atom_graph.node_nonlinear_update"],
        FusedInputGatedMLP,
    )


def test_p1e_second_fusion_modes_replace_only_declared_candidates() -> None:
    cases = {
        "line_edge_gated_mlp_second_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
        ],
        "line_all_candidate_gated_mlp_second_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
            "attn_block_line_graph.node_nonlinear_update",
        ],
        "atom_edge_gated_mlp_second_fused_fp32": [
            "attn_block_atom_graph.edge_nonlinear_update",
            "refine_block_atom_graph.edge_nonlinear_update",
        ],
        "atom_attn_node_gated_mlp_second_fused_fp32": [
            "attn_block_atom_graph.node_nonlinear_update",
        ],
    }

    for mode, required_fragments in cases.items():
        model = TinyMatRISForFusion()
        replaced = apply_gated_mlp_fusion(model, mode)

        for fragment in required_fragments:
            assert sum(fragment in name for name in replaced) == 2
            module = dict(model.named_modules())[f"interaction_block.0.{fragment}"]
            assert isinstance(module, FusedInputGatedMLP)
            assert module.core_second is not None
            assert module.gate_second is not None
            assert module.fused_second is not None


def test_p1e_retest_candidates_match_expected_second_fusion_scope_with_fake_quant() -> None:
    cases = {
        "line_all_candidate_gated_mlp_second_fused_fp32": [
            "interaction_block.0.attn_block_line_graph.edge_nonlinear_update",
            "interaction_block.0.refine_block_line_graph.edge_nonlinear_update",
            "interaction_block.0.attn_block_line_graph.node_nonlinear_update",
            "interaction_block.1.attn_block_line_graph.edge_nonlinear_update",
            "interaction_block.1.refine_block_line_graph.edge_nonlinear_update",
            "interaction_block.1.attn_block_line_graph.node_nonlinear_update",
        ],
        "atom_attn_node_gated_mlp_second_fused_fp32": [
            "interaction_block.0.attn_block_atom_graph.node_nonlinear_update",
            "interaction_block.1.attn_block_atom_graph.node_nonlinear_update",
        ],
    }

    for mode, expected in cases.items():
        model = TinyMatRISForFusion()
        apply_quant_config(model, get_quant_config("p0_line_gate_paper_stable_w8a32"))
        replaced = apply_gated_mlp_fusion(model, mode)

        assert replaced == expected
        for name in expected:
            module = dict(model.named_modules())[name]
            assert isinstance(module, FusedInputGatedMLP)
            assert module.core_second is not None
            assert module.gate_second is not None
            assert module.fused_second is None


def test_p1f_tail_fusion_modes_replace_only_declared_candidates() -> None:
    cases = {
        "line_edge_gated_mlp_tail_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
        ],
        "line_all_candidate_gated_mlp_tail_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
            "attn_block_line_graph.node_nonlinear_update",
        ],
        "atom_edge_gated_mlp_tail_fused_fp32": [
            "attn_block_atom_graph.edge_nonlinear_update",
            "refine_block_atom_graph.edge_nonlinear_update",
        ],
        "atom_attn_node_gated_mlp_tail_fused_fp32": [
            "attn_block_atom_graph.node_nonlinear_update",
        ],
    }

    for mode, required_fragments in cases.items():
        model = TinyMatRISForFusion()
        replaced = apply_gated_mlp_fusion(model, mode)

        for fragment in required_fragments:
            assert sum(fragment in name for name in replaced) == 2
            module = dict(model.named_modules())[f"interaction_block.0.{fragment}"]
            assert isinstance(module, FusedInputGatedMLP)
            assert isinstance(module.fused_tail, FusedGatedMLPTail)
            assert module.core_second is None
            assert module.gate_second is None


def test_p1f_second_tail_fusion_modes_replace_only_declared_candidates() -> None:
    cases = {
        "line_edge_gated_mlp_second_tail_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
        ],
        "line_all_candidate_gated_mlp_second_tail_fused_fp32": [
            "attn_block_line_graph.edge_nonlinear_update",
            "refine_block_line_graph.edge_nonlinear_update",
            "attn_block_line_graph.node_nonlinear_update",
        ],
        "atom_edge_gated_mlp_second_tail_fused_fp32": [
            "attn_block_atom_graph.edge_nonlinear_update",
            "refine_block_atom_graph.edge_nonlinear_update",
        ],
        "atom_attn_node_gated_mlp_second_tail_fused_fp32": [
            "attn_block_atom_graph.node_nonlinear_update",
        ],
    }

    for mode, required_fragments in cases.items():
        model = TinyMatRISForFusion()
        replaced = apply_gated_mlp_fusion(model, mode)

        for fragment in required_fragments:
            assert sum(fragment in name for name in replaced) == 2
            module = dict(model.named_modules())[f"interaction_block.0.{fragment}"]
            assert isinstance(module, FusedInputGatedMLP)
            assert module.core_second is not None
            assert module.gate_second is not None
            assert module.fused_second is not None
            assert isinstance(module.fused_tail, FusedGatedMLPTail)


def test_p1g_all_passed_second_fusion_replaces_line_all_and_atom_attn_node_only() -> None:
    model = TinyMatRISForFusion()
    replaced = apply_gated_mlp_fusion(model, "all_passed_gated_mlp_second_fused_fp32")

    expected = [
        "interaction_block.0.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.0.attn_block_line_graph.node_nonlinear_update",
        "interaction_block.0.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.0.attn_block_atom_graph.node_nonlinear_update",
        "interaction_block.1.attn_block_line_graph.edge_nonlinear_update",
        "interaction_block.1.attn_block_line_graph.node_nonlinear_update",
        "interaction_block.1.refine_block_line_graph.edge_nonlinear_update",
        "interaction_block.1.attn_block_atom_graph.node_nonlinear_update",
    ]
    assert replaced == expected
    assert all("refine_block_atom_graph" not in name for name in replaced)
    assert all("edge_nonlinear_update" not in name or "atom_graph" not in name for name in replaced)

    for name in expected:
        module = dict(model.named_modules())[name]
        assert isinstance(module, FusedInputGatedMLP)
        assert module.core_second is not None
        assert module.gate_second is not None
        assert module.fused_second is not None
        assert module.fused_tail is None


def test_p1g_all_passed_second_fusion_matches_fake_quant_scope() -> None:
    model = TinyMatRISForFusion()
    apply_quant_config(model, get_quant_config("p0_line_gate_paper_stable_w8a32"))
    replaced = apply_gated_mlp_fusion(model, "all_passed_gated_mlp_second_fused_fp32")

    assert len(replaced) == 8
    for name in replaced:
        module = dict(model.named_modules())[name]
        assert isinstance(module, FusedInputGatedMLP)
        assert module.core_second is not None
        assert module.gate_second is not None
        assert module.fused_second is None
        assert module.fused_tail is None
