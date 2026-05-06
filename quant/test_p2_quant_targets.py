from __future__ import annotations

import torch
from torch import nn

from quant.config import get_quant_config
from quant.injector import apply_quant_config
from quant.layers import (
    BackwardFakeQuantLinear,
    CachedLowPrecisionLinear,
    FakeQuantLinear,
    ForwardOnlyFakeQuantLinear,
    LowPrecisionLinear,
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 4),
            nn.SiLU(),
            nn.Linear(4, 4),
        )


class TinyGatedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp_core = TinyMLP()
        self.mlp_gate = TinyMLP()


def _nonlinear_update(gated: bool) -> nn.Module:
    if gated:
        return TinyGatedMLP()
    return nn.Sequential(TinyMLP(), nn.LayerNorm(4))


class TinyAttention(nn.Module):
    def __init__(self, gated: bool) -> None:
        super().__init__()
        self.source_weight_linear = nn.Linear(4, 4, bias=False)
        self.target_weight_linear = nn.Linear(4, 4, bias=False)
        self.node_nonlinear_update = _nonlinear_update(gated)
        self.edge_nonlinear_update = _nonlinear_update(gated)


class TinyRefinement(nn.Module):
    def __init__(self, gated: bool) -> None:
        super().__init__()
        self.edge_nonlinear_update = _nonlinear_update(gated)
        self.node_FFN = TinyMLP()
        self.edge_FFN = TinyMLP()
        self.learnable_envelope = nn.Linear(4, 4, bias=False)


class TinyInteractionBlock(nn.Module):
    def __init__(self, gated: bool) -> None:
        super().__init__()
        self.attn_block_line_graph = TinyAttention(gated)
        self.refine_block_line_graph = TinyRefinement(gated)
        self.attn_block_atom_graph = TinyAttention(gated)
        self.refine_block_atom_graph = TinyRefinement(gated)


class TinyEnergyHead(nn.Module):
    def __init__(self, gated: bool) -> None:
        super().__init__()
        if gated:
            self.energy_head = nn.Sequential(TinyGatedMLP(), nn.Linear(4, 1))
        else:
            self.energy_head = TinyMLP()


class TinyMatRIS(nn.Module):
    def __init__(self, *, gated_readout: bool = True) -> None:
        super().__init__()
        self.graph_converter = nn.Linear(4, 4)
        self.edge_embedding = nn.Linear(4, 4)
        self.three_body_embedding = nn.Linear(4, 4)
        self.interaction_block = nn.ModuleList(
            [
                TinyInteractionBlock(gated=True),
                TinyInteractionBlock(gated=False),
            ]
        )
        self.energy_head = TinyEnergyHead(gated=gated_readout)


def _module(model: nn.Module, name: str) -> nn.Module:
    return dict(model.named_modules())[name]


def test_forward_only_fake_quant_uses_quantized_forward_and_fp32_input_backward() -> None:
    original = nn.Linear(3, 2, bias=False)
    original.weight.data = torch.tensor(
        [
            [0.13, -0.27, 0.41],
            [0.92, -0.56, 0.08],
        ],
        dtype=torch.float32,
    )
    layer = ForwardOnlyFakeQuantLinear(original, "linear", weight_bits=2)
    x = torch.tensor([[0.7, -1.1, 0.3]], dtype=torch.float32, requires_grad=True)

    y = layer(x)
    y.sum().backward()

    dq_weight, _ = layer._fake_quant_weight()
    assert torch.allclose(y, torch.nn.functional.linear(x.detach(), dq_weight), atol=1e-6)
    assert torch.allclose(x.grad, original.weight.detach().sum(dim=0, keepdim=True), atol=1e-6)


def test_backward_fake_quant_uses_fp32_forward_and_quantized_input_backward() -> None:
    original = nn.Linear(3, 2, bias=False)
    original.weight.data = torch.tensor(
        [
            [0.13, -0.27, 0.41],
            [0.92, -0.56, 0.08],
        ],
        dtype=torch.float32,
    )
    layer = BackwardFakeQuantLinear(original, "linear", weight_bits=2)
    x = torch.tensor([[0.7, -1.1, 0.3]], dtype=torch.float32, requires_grad=True)

    y = layer(x)
    y.sum().backward()

    dq_weight, _ = layer._fake_quant_weight()
    assert torch.allclose(y, original(x.detach()), atol=1e-6)
    assert torch.allclose(x.grad, dq_weight.detach().sum(dim=0, keepdim=True), atol=1e-6)


def test_p2_w8a32_replaces_line_graph_nonlinear_and_readout_core_only() -> None:
    model = TinyMatRIS(gated_readout=True)

    replaced = apply_quant_config(model, get_quant_config("p2_w8a32"))

    assert set(replaced) == {
        "interaction_block.0.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "interaction_block.0.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.2",
        "interaction_block.0.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "interaction_block.0.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.2",
        "interaction_block.0.attn_block_line_graph.node_nonlinear_update.mlp_core.layers.0",
        "interaction_block.0.attn_block_line_graph.node_nonlinear_update.mlp_core.layers.2",
        "interaction_block.1.attn_block_line_graph.edge_nonlinear_update.0.layers.0",
        "interaction_block.1.attn_block_line_graph.edge_nonlinear_update.0.layers.2",
        "interaction_block.1.refine_block_line_graph.edge_nonlinear_update.0.layers.0",
        "interaction_block.1.refine_block_line_graph.edge_nonlinear_update.0.layers.2",
        "interaction_block.1.attn_block_line_graph.node_nonlinear_update.0.layers.0",
        "interaction_block.1.attn_block_line_graph.node_nonlinear_update.0.layers.2",
        "energy_head.energy_head.0.mlp_core.layers.0",
    }
    assert all(isinstance(_module(model, name), FakeQuantLinear) for name in replaced)


def test_p2_w8a32_keeps_first_batch_fp32_sensitive_objects() -> None:
    replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config("p2_w8a32"))

    forbidden_fragments = (
        "graph_converter",
        "edge_embedding",
        "three_body_embedding",
        "learnable_envelope",
        "attn_block_atom_graph",
        "refine_block_atom_graph",
        "source_weight_linear",
        "target_weight_linear",
        "node_FFN",
        "edge_FFN",
        "mlp_gate",
        "energy_head.energy_head.1",
    )
    for name in replaced:
        assert not any(fragment in name for fragment in forbidden_fragments)


def test_line_graph_p2_modes_cover_mlp_and_gatedmlp_variants() -> None:
    model = TinyMatRIS(gated_readout=True)

    edge_replaced = apply_quant_config(model, get_quant_config("line_edge_mlp_w8a32"))
    assert {
        "interaction_block.0.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "interaction_block.0.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "interaction_block.1.attn_block_line_graph.edge_nonlinear_update.0.layers.0",
        "interaction_block.1.refine_block_line_graph.edge_nonlinear_update.0.layers.0",
    }.issubset(set(edge_replaced))

    model = TinyMatRIS(gated_readout=True)
    node_replaced = apply_quant_config(model, get_quant_config("line_node_mlp_w8a32"))
    assert {
        "interaction_block.0.attn_block_line_graph.node_nonlinear_update.mlp_core.layers.0",
        "interaction_block.1.attn_block_line_graph.node_nonlinear_update.0.layers.0",
    }.issubset(set(node_replaced))


def test_linear_only_bf16_fp16_modes_replace_expected_targets_and_return_fp32() -> None:
    expected_by_mode = {
        "line_graph_linear_bf16": {
            "interaction_block.0.attn_block_line_graph.source_weight_linear",
            "interaction_block.0.attn_block_line_graph.target_weight_linear",
            "interaction_block.0.refine_block_line_graph.node_FFN.layers.0",
            "interaction_block.0.refine_block_line_graph.node_FFN.layers.2",
            "interaction_block.0.refine_block_line_graph.edge_FFN.layers.0",
            "interaction_block.0.refine_block_line_graph.edge_FFN.layers.2",
            "interaction_block.1.attn_block_line_graph.source_weight_linear",
            "interaction_block.1.attn_block_line_graph.target_weight_linear",
            "interaction_block.1.refine_block_line_graph.node_FFN.layers.0",
            "interaction_block.1.refine_block_line_graph.node_FFN.layers.2",
            "interaction_block.1.refine_block_line_graph.edge_FFN.layers.0",
            "interaction_block.1.refine_block_line_graph.edge_FFN.layers.2",
        },
        "line_edge_gate_linear_bf16": {
            "interaction_block.0.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
            "interaction_block.0.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.2",
            "interaction_block.0.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
            "interaction_block.0.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.2",
        },
        "line_node_gate_linear_bf16": {
            "interaction_block.0.attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.0",
            "interaction_block.0.attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.2",
        },
    }
    expected_by_mode["line_graph_linear_fp16"] = expected_by_mode["line_graph_linear_bf16"]
    expected_by_mode["line_edge_gate_linear_fp16"] = expected_by_mode["line_edge_gate_linear_bf16"]
    expected_by_mode["line_node_gate_linear_fp16"] = expected_by_mode["line_node_gate_linear_bf16"]

    for mode, expected in expected_by_mode.items():
        model = TinyMatRIS(gated_readout=True)
        replaced = apply_quant_config(model, get_quant_config(mode))
        assert set(replaced) == expected

        layer = _module(model, replaced[0])
        assert isinstance(layer, LowPrecisionLinear)
        assert layer.weight.dtype == torch.float32

        out = layer(torch.randn(2, 4, dtype=torch.float32))
        assert out.dtype == torch.float32
        assert layer.last_stats["compute_dtype"] in ("torch.bfloat16", "torch.float16")


def test_p0_line_gate_paper_stable_linear_bf16_fp16_keeps_sensitive_objects_fp32() -> None:
    for mode in ("p0_line_gate_paper_stable_linear_bf16", "p0_line_gate_paper_stable_linear_fp16"):
        model = TinyMatRIS(gated_readout=True)
        replaced = apply_quant_config(model, get_quant_config(mode))

        assert replaced
        assert all(isinstance(_module(model, name), LowPrecisionLinear) for name in replaced)
        assert any("attn_block_line_graph.edge_nonlinear_update.mlp_gate" in name for name in replaced)
        assert any("attn_block_line_graph.node_nonlinear_update.mlp_gate" in name for name in replaced)

        forbidden_fragments = (
            "graph_converter",
            "edge_embedding",
            "three_body_embedding",
            "learnable_envelope",
            "energy_head",
            "attn_block_atom_graph.edge_nonlinear_update",
            "refine_block_atom_graph.edge_nonlinear_update",
            "attn_block_atom_graph.node_nonlinear_update",
        )
        for name in replaced:
            assert not any(fragment in name for fragment in forbidden_fragments)


def test_cached_linear_only_bf16_fp16_modes_cache_low_precision_weight_and_return_fp32() -> None:
    pairs = (
        ("line_graph_linear_bf16", "line_graph_linear_cached_bf16", torch.bfloat16),
        ("line_edge_gate_linear_bf16", "line_edge_gate_linear_cached_bf16", torch.bfloat16),
        ("line_node_gate_linear_bf16", "line_node_gate_linear_cached_bf16", torch.bfloat16),
        ("p0_line_gate_paper_stable_linear_bf16", "p0_line_gate_paper_stable_linear_cached_bf16", torch.bfloat16),
        ("line_graph_linear_fp16", "line_graph_linear_cached_fp16", torch.float16),
        ("line_edge_gate_linear_fp16", "line_edge_gate_linear_cached_fp16", torch.float16),
        ("line_node_gate_linear_fp16", "line_node_gate_linear_cached_fp16", torch.float16),
        ("p0_line_gate_paper_stable_linear_fp16", "p0_line_gate_paper_stable_linear_cached_fp16", torch.float16),
    )

    for uncached_mode, cached_mode, expected_dtype in pairs:
        uncached_replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config(uncached_mode))
        model = TinyMatRIS(gated_readout=True)
        cached_replaced = apply_quant_config(model, get_quant_config(cached_mode))

        assert cached_replaced == uncached_replaced
        layer = _module(model, cached_replaced[0])
        assert isinstance(layer, CachedLowPrecisionLinear)
        assert layer.weight_low.dtype == expected_dtype

        out = layer(torch.randn(2, 4, dtype=torch.float32))
        assert out.dtype == torch.float32
        assert layer.last_stats["backend"] == "linear_low_precision_cached"
        assert layer.last_stats["cached_weight_dtype"] == str(expected_dtype)


def test_readout_pre_final_w8a32_keeps_final_energy_layer_and_gate_fp32() -> None:
    gated_model = TinyMatRIS(gated_readout=True)
    gated_replaced = apply_quant_config(gated_model, get_quant_config("readout_pre_final_w8a32"))

    assert gated_replaced == ["energy_head.energy_head.0.mlp_core.layers.0"]
    assert isinstance(_module(gated_model, gated_replaced[0]), FakeQuantLinear)
    assert not isinstance(_module(gated_model, "energy_head.energy_head.0.mlp_gate.layers.0"), FakeQuantLinear)
    assert not isinstance(_module(gated_model, "energy_head.energy_head.1"), FakeQuantLinear)

    mlp_model = TinyMatRIS(gated_readout=False)
    mlp_replaced = apply_quant_config(mlp_model, get_quant_config("readout_pre_final_w8a32"))

    assert mlp_replaced == ["energy_head.energy_head.layers.0"]
    assert isinstance(_module(mlp_model, mlp_replaced[0]), FakeQuantLinear)
    assert not isinstance(_module(mlp_model, "energy_head.energy_head.layers.2"), FakeQuantLinear)


def test_p0_line_stable_w8a32_combines_p0_and_line_passed_targets_only() -> None:
    replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config("p0_line_stable_w8a32"))

    required_fragments = (
        "refine_block_atom_graph.node_FFN.layers.0",
        "refine_block_atom_graph.edge_FFN.layers.0",
        "attn_block_atom_graph.source_weight_linear",
        "attn_block_atom_graph.target_weight_linear",
        "attn_block_line_graph.source_weight_linear",
        "attn_block_line_graph.target_weight_linear",
        "refine_block_line_graph.node_FFN.layers.0",
        "refine_block_line_graph.edge_FFN.layers.0",
        "attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "attn_block_line_graph.node_nonlinear_update.mlp_core.layers.0",
    )
    for fragment in required_fragments:
        assert any(fragment in name for name in replaced)

    forbidden_fragments = (
        "energy_head",
        "edge_embedding",
        "three_body_embedding",
        "learnable_envelope",
        "attn_block_atom_graph.edge_nonlinear_update",
        "refine_block_atom_graph.edge_nonlinear_update",
        "attn_block_atom_graph.node_nonlinear_update",
        "mlp_gate",
    )
    for name in replaced:
        assert not any(fragment in name for fragment in forbidden_fragments)


def test_p0_line_gate_stable_w8a32_adds_line_gates_to_stable_base_only() -> None:
    base_replaced = set(
        apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config("p0_line_stable_w8a32"))
    )
    replaced = set(
        apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config("p0_line_gate_stable_w8a32"))
    )

    required_gate_fragments = (
        "attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        "refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        "attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.0",
    )
    for fragment in required_gate_fragments:
        assert any(fragment in name for name in replaced)

    added = replaced - base_replaced
    assert added
    assert all("attn_block_line_graph" in name or "refine_block_line_graph" in name for name in added)
    assert all("mlp_gate.layers" in name for name in added)

    forbidden_fragments = (
        "energy_head",
        "edge_embedding",
        "three_body_embedding",
        "learnable_envelope",
        "attn_block_atom_graph.edge_nonlinear_update.mlp_gate",
        "refine_block_atom_graph.edge_nonlinear_update.mlp_gate",
        "attn_block_atom_graph.node_nonlinear_update",
        "refine_block_atom_graph.edge_nonlinear_update",
    )
    for name in replaced:
        assert not any(fragment in name for fragment in forbidden_fragments)


def test_p0_line_gate_stable_w8a32_torchao_uses_same_targets_as_fake_stable_mode() -> None:
    fake_config = get_quant_config("p0_line_gate_stable_w8a32")
    torchao_config = get_quant_config("p0_line_gate_stable_w8a32_torchao")

    assert torchao_config["targets"] == fake_config["targets"]
    assert torchao_config["fake_quant"] is False
    assert torchao_config["kernel"] == "torchao_int8_weight_only"
    assert torchao_config["weight_packing"] == "torchao_int8_weight_only"
    assert torchao_config["backward"] == "torchao_autograd"


def test_torchao_real_quant_screen_modes_reuse_fake_targets() -> None:
    fake_to_torchao = {
        "atom_p0_w8a32": "atom_p0_w8a32_torchao",
        "line_graph_w8a32": "line_graph_w8a32_torchao",
        "line_edge_mlp_w8a32": "line_edge_mlp_w8a32_torchao",
        "line_node_mlp_w8a32": "line_node_mlp_w8a32_torchao",
        "line_edge_gate_mlp_w8a32": "line_edge_gate_mlp_w8a32_torchao",
        "line_node_gate_mlp_w8a32": "line_node_gate_mlp_w8a32_torchao",
    }

    for fake_mode, torchao_mode in fake_to_torchao.items():
        fake_config = get_quant_config(fake_mode)
        torchao_config = get_quant_config(torchao_mode)

        assert torchao_config["targets"] == fake_config["targets"]
        assert torchao_config["fake_quant"] is False
        assert torchao_config["kernel"] == "torchao_int8_weight_only"
        assert torchao_config["weight_packing"] == "torchao_int8_weight_only"
        assert torchao_config["backward"] == "torchao_autograd"


def test_forward_only_and_backward_only_screen_modes_reuse_fake_targets() -> None:
    cases = {
        "line_graph_w8a32": (
            "fwd_only_w8a32_line_graph",
            "bwd_w8a32_line_graph",
        ),
        "line_edge_mlp_w8a32": (
            "fwd_only_w8a32_line_edge_mlp",
            "bwd_w8a32_line_edge_mlp",
        ),
        "line_edge_gate_mlp_w8a32": (
            "fwd_only_w8a32_line_edge_gate_mlp",
            "bwd_w8a32_line_edge_gate_mlp",
        ),
        "line_node_gate_mlp_w8a32": (
            "fwd_only_w8a32_line_node_gate_mlp",
            "bwd_w8a32_line_node_gate_mlp",
        ),
        "atom_p0_w8a32": (
            "fwd_only_w8a32_atom_p0",
            "bwd_w8a32_atom_p0",
        ),
        "p0_line_gate_paper_stable_w8a32": (
            "fwd_only_w8a32_p0_line_gate_paper_stable",
            "bwd_w8a32_p0_line_gate_paper_stable",
        ),
        "all_single_pass_w8a32": (
            "fwd_only_w8a32_all_single_pass",
            "bwd_w8a32_all_single_pass",
        ),
    }

    for fake_mode, scoped_modes in cases.items():
        fake_config = get_quant_config(fake_mode)
        for scoped_mode in scoped_modes:
            scoped_config = get_quant_config(scoped_mode)
            expected_scope = "forward_only" if scoped_mode.startswith("fwd_only") else "backward_only"

            assert scoped_config["targets"] == fake_config["targets"]
            assert scoped_config["fake_quant"] is True
            assert scoped_config["fake_quant_scope"] == expected_scope


def test_forward_only_and_backward_only_inject_expected_layer_types() -> None:
    fwd_model = TinyMatRIS(gated_readout=True)
    fwd_replaced = apply_quant_config(fwd_model, get_quant_config("fwd_only_w8a32_line_graph"))
    assert fwd_replaced
    assert all(isinstance(_module(fwd_model, name), ForwardOnlyFakeQuantLinear) for name in fwd_replaced)

    bwd_model = TinyMatRIS(gated_readout=True)
    bwd_replaced = apply_quant_config(bwd_model, get_quant_config("bwd_w8a32_line_graph"))
    assert bwd_replaced
    assert all(isinstance(_module(bwd_model, name), BackwardFakeQuantLinear) for name in bwd_replaced)


def test_p0_line_gate_atom_attn_edge_w8a32_adds_atom_attn_edge_core_only() -> None:
    base_replaced = set(
        apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config("p0_line_gate_stable_w8a32"))
    )
    replaced = set(
        apply_quant_config(
            TinyMatRIS(gated_readout=True),
            get_quant_config("p0_line_gate_atom_attn_edge_w8a32"),
        )
    )

    required_fragments = (
        "attn_block_atom_graph.edge_nonlinear_update.mlp_core.layers.0",
        "attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        "refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        "attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.0",
    )
    for fragment in required_fragments:
        assert any(fragment in name for name in replaced)

    added = replaced - base_replaced
    assert added
    assert all("attn_block_atom_graph.edge_nonlinear_update.mlp_core.layers" in name for name in added)

    forbidden_fragments = (
        "energy_head",
        "edge_embedding",
        "three_body_embedding",
        "learnable_envelope",
        "attn_block_atom_graph.edge_nonlinear_update.mlp_gate",
        "refine_block_atom_graph.edge_nonlinear_update",
        "attn_block_atom_graph.node_nonlinear_update",
    )
    for name in replaced:
        assert not any(fragment in name for fragment in forbidden_fragments)


def test_p0_line_gate_paper_stable_w8a32_aliases_atom_attn_edge_static_candidate() -> None:
    old_candidate = get_quant_config("p0_line_gate_atom_attn_edge_w8a32")
    paper_stable = get_quant_config("p0_line_gate_paper_stable_w8a32")

    assert paper_stable["targets"] == old_candidate["targets"]
    assert paper_stable["fake_quant"] is True
    assert paper_stable["mode"] == "p0_line_gate_paper_stable_w8a32"
    assert "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core" in paper_stable["targets"]


def test_atom_nonlinear_split_modes_target_one_atom_core_branch_each() -> None:
    cases = {
        "atom_attn_edge_mlp_w8a32": {
            "required": "attn_block_atom_graph.edge_nonlinear_update.mlp_core.layers.0",
            "forbidden": (
                "refine_block_atom_graph.edge_nonlinear_update",
                "attn_block_atom_graph.node_nonlinear_update",
                "attn_block_line_graph",
                "mlp_gate",
            ),
        },
        "atom_refine_edge_mlp_w8a32": {
            "required": "refine_block_atom_graph.edge_nonlinear_update.mlp_core.layers.0",
            "forbidden": (
                "attn_block_atom_graph.edge_nonlinear_update",
                "attn_block_atom_graph.node_nonlinear_update",
                "attn_block_line_graph",
                "mlp_gate",
            ),
        },
        "atom_attn_node_mlp_w8a32": {
            "required": "attn_block_atom_graph.node_nonlinear_update.mlp_core.layers.0",
            "forbidden": (
                "attn_block_atom_graph.edge_nonlinear_update",
                "refine_block_atom_graph.edge_nonlinear_update",
                "attn_block_line_graph",
                "mlp_gate",
            ),
        },
    }

    for mode, expected in cases.items():
        replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config(mode))

        assert any(expected["required"] in name for name in replaced)
        for name in replaced:
            assert not any(fragment in name for fragment in expected["forbidden"])


def test_atom_node_mlp_w8a32_is_alias_for_atom_attn_node_split() -> None:
    old_name_replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config("atom_node_mlp_w8a32"))
    split_name_replaced = apply_quant_config(
        TinyMatRIS(gated_readout=True),
        get_quant_config("atom_attn_node_mlp_w8a32"),
    )

    assert old_name_replaced == split_name_replaced


def test_gate_branch_modes_replace_only_gatedmlp_gate_linears() -> None:
    cases = {
        "line_edge_gate_mlp_w8a32": {
            "required": (
                "attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
                "refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
            ),
            "forbidden": (
                "attn_block_line_graph.edge_nonlinear_update.mlp_core",
                "refine_block_line_graph.edge_nonlinear_update.mlp_core",
                "attn_block_line_graph.node_nonlinear_update",
                "attn_block_atom_graph",
                "refine_block_atom_graph",
            ),
        },
        "line_node_gate_mlp_w8a32": {
            "required": (
                "attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.0",
            ),
            "forbidden": (
                "attn_block_line_graph.node_nonlinear_update.mlp_core",
                "attn_block_line_graph.edge_nonlinear_update",
                "refine_block_line_graph",
                "attn_block_atom_graph",
                "refine_block_atom_graph",
            ),
        },
        "atom_edge_gate_mlp_w8a32": {
            "required": (
                "attn_block_atom_graph.edge_nonlinear_update.mlp_gate.layers.0",
                "refine_block_atom_graph.edge_nonlinear_update.mlp_gate.layers.0",
            ),
            "forbidden": (
                "attn_block_atom_graph.edge_nonlinear_update.mlp_core",
                "refine_block_atom_graph.edge_nonlinear_update.mlp_core",
                "attn_block_atom_graph.node_nonlinear_update",
                "attn_block_line_graph",
                "refine_block_line_graph",
            ),
        },
        "atom_node_gate_mlp_w8a32": {
            "required": (
                "attn_block_atom_graph.node_nonlinear_update.mlp_gate.layers.0",
            ),
            "forbidden": (
                "attn_block_atom_graph.node_nonlinear_update.mlp_core",
                "attn_block_atom_graph.edge_nonlinear_update",
                "refine_block_atom_graph",
                "attn_block_line_graph",
                "refine_block_line_graph",
            ),
        },
    }

    for mode, expected in cases.items():
        replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config(mode))

        for fragment in expected["required"]:
            assert any(fragment in name for name in replaced)
        for name in replaced:
            assert "mlp_gate.layers" in name
            assert not any(fragment in name for fragment in expected["forbidden"])


def test_line_core_gate_modes_replace_line_core_and_gate_only() -> None:
    cases = {
        "line_edge_core_gate_mlp_w8a32": {
            "required": (
                "attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
                "attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
                "refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
                "refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
            ),
            "forbidden": (
                "attn_block_line_graph.node_nonlinear_update",
                "attn_block_atom_graph",
                "refine_block_atom_graph",
                "energy_head",
            ),
        },
        "line_node_core_gate_mlp_w8a32": {
            "required": (
                "attn_block_line_graph.node_nonlinear_update.mlp_core.layers.0",
                "attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.0",
            ),
            "forbidden": (
                "attn_block_line_graph.edge_nonlinear_update",
                "refine_block_line_graph",
                "attn_block_atom_graph",
                "refine_block_atom_graph",
                "energy_head",
            ),
        },
        "line_nonlinear_core_gate_mlp_w8a32": {
            "required": (
                "attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
                "attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
                "refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
                "refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
                "attn_block_line_graph.node_nonlinear_update.mlp_core.layers.0",
                "attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.0",
            ),
            "forbidden": (
                "attn_block_atom_graph",
                "refine_block_atom_graph",
                "energy_head",
                "node_FFN",
                "edge_FFN",
                "source_weight_linear",
                "target_weight_linear",
            ),
        },
    }

    for mode, expected in cases.items():
        replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config(mode))

        for fragment in expected["required"]:
            assert any(fragment in name for name in replaced)
        assert all(("mlp_core.layers" in name or "mlp_gate.layers" in name) for name in replaced)
        for name in replaced:
            assert not any(fragment in name for fragment in expected["forbidden"])


def test_all_single_pass_w8a32_combines_only_single_pass_targets() -> None:
    replaced = apply_quant_config(TinyMatRIS(gated_readout=True), get_quant_config("all_single_pass_w8a32"))

    required_fragments = (
        "refine_block_atom_graph.node_FFN.layers.0",
        "refine_block_atom_graph.edge_FFN.layers.0",
        "attn_block_atom_graph.source_weight_linear",
        "attn_block_atom_graph.target_weight_linear",
        "attn_block_atom_graph.edge_nonlinear_update.mlp_core.layers.0",
        "attn_block_atom_graph.node_nonlinear_update.mlp_gate.layers.0",
        "attn_block_line_graph.source_weight_linear",
        "attn_block_line_graph.target_weight_linear",
        "refine_block_line_graph.node_FFN.layers.0",
        "refine_block_line_graph.edge_FFN.layers.0",
        "attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        "refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        "attn_block_line_graph.node_nonlinear_update.mlp_core.layers.0",
        "attn_block_line_graph.node_nonlinear_update.mlp_gate.layers.0",
    )
    for fragment in required_fragments:
        assert any(fragment in name for name in replaced)

    forbidden_fragments = (
        "energy_head",
        "edge_embedding",
        "three_body_embedding",
        "learnable_envelope",
        "refine_block_atom_graph.edge_nonlinear_update",
        "attn_block_atom_graph.node_nonlinear_update.mlp_core",
        "attn_block_atom_graph.edge_nonlinear_update.mlp_gate",
    )
    for name in replaced:
        assert not any(fragment in name for fragment in forbidden_fragments)
