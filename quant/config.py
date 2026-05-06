from __future__ import annotations


def get_quant_config(quant_mode: str | None) -> dict | None:
    if quant_mode in (None, "", "none"):
        return None

    base = {
        "weight_bits": 8,
        "activation_bits": None,
        "activation_dtype": "fp32",
        "accumulation_dtype": "fp32",
        "output_dtype": "fp32",
        "scale_granularity": "per_channel",
        "fake_quant": True,
        "fake_quant_scope": "forward_backward",
    }
    a8_base = {
        **base,
        "activation_bits": 8,
        "activation_dtype": "int8_fake_dynamic_per_tensor",
        "activation_scale_granularity": "dynamic_per_tensor",
        "accumulation_dtype": "fp32",
        "output_dtype": "fp32",
        "fake_quant": True,
    }
    smooth_a8_base = {
        **a8_base,
        "activation_dtype": "int8_fake_static_per_tensor_smooth",
        "activation_scale_granularity": "static_per_tensor",
        "smooth_quant": True,
        "smooth_alpha": 0.5,
        "smooth_scale_clamp": 1.0e4,
    }
    forward_only_base = {
        **base,
        "fake_quant_scope": "forward_only",
    }
    backward_only_base = {
        **base,
        "fake_quant_scope": "backward_only",
    }
    kernel_base = {
        **base,
        "fake_quant": False,
        "kernel": "w8a32_packed",
        "weight_packing": "int8_per_channel",
        "use_cuda_kernel": True,
        "backward": "torch_fallback_v1",
    }
    triton_w8a32_base = {
        **base,
        "fake_quant": False,
        "kernel": "triton_w8a32",
        "weight_packing": "int8_per_channel",
        "backward": "torch_dequantized_grad_x",
    }
    triton_w8a8_static_base = {
        **base,
        "activation_bits": 8,
        "activation_dtype": "int8_static_per_tensor",
        "activation_scale_granularity": "static_per_tensor",
        "fake_quant": False,
        "kernel": "triton_w8a8_static",
        "weight_packing": "int8_per_channel",
        "backward": "torch_dequantized_grad_x",
    }
    triton_smooth_w8a8_static_base = {
        **triton_w8a8_static_base,
        "activation_dtype": "int8_static_per_tensor_smooth",
        "smooth_quant": True,
        "smooth_alpha": 0.5,
        "smooth_scale_clamp": 1.0e4,
    }
    torch_fp8_static_base = {
        **base,
        "weight_bits": 8,
        "activation_bits": 8,
        "activation_dtype": "fp8_e4m3fn_static_per_tensor",
        "activation_scale_granularity": "static_per_tensor",
        "fake_quant": False,
        "kernel": "torch_scaled_mm_fp8_static",
        "weight_packing": "fp8_e4m3fn_per_tensor",
        "fp8_dtype": "e4m3fn",
        "backward": "torch_dequantized_grad_x",
    }
    te_fp8_base = {
        **base,
        "weight_bits": 8,
        "activation_bits": 8,
        "activation_dtype": "te_fp8_hybrid",
        "activation_scale_granularity": "te_delayed_scaling",
        "fake_quant": False,
        "kernel": "transformer_engine_fp8",
        "weight_packing": "transformer_engine_fp8",
        "fp8_format": "HYBRID",
        "amax_history_len": 16,
        "amax_compute_algo": "max",
        "backward": "transformer_engine_autograd",
    }
    torchao_weight_only_base = {
        **base,
        "fake_quant": False,
        "kernel": "torchao_int8_weight_only",
        "weight_packing": "torchao_int8_weight_only",
        "set_inductor_config": False,
        "backward": "torchao_autograd",
    }
    linear_bf16_base = {
        **base,
        "fake_quant": False,
        "kernel": "linear_low_precision",
        "compute_dtype": "bf16",
        "weight_packing": None,
        "backward": "torch_autograd",
    }
    linear_fp16_base = {
        **linear_bf16_base,
        "compute_dtype": "fp16",
    }
    cached_linear_bf16_base = {
        **linear_bf16_base,
        "kernel": "linear_low_precision_cached",
    }
    cached_linear_fp16_base = {
        **cached_linear_bf16_base,
        "compute_dtype": "fp16",
    }
    w8abf16_reference_base = {
        **base,
        "fake_quant": False,
        "kernel": "w8a_low_precision_reference",
        "weight_packing": "int8_per_channel",
        "activation_dtype": "bf16",
        "compute_dtype": "bf16",
        "backward": "torch_autograd_low_precision_grad_x",
    }
    w8a16_reference_base = {
        **w8abf16_reference_base,
        "activation_dtype": "fp16",
        "compute_dtype": "fp16",
    }
    w8abf16_cuda_v0_base = {
        **w8abf16_reference_base,
        "mixed_backend": "cuda_v0",
    }
    w8a16_cuda_v0_base = {
        **w8a16_reference_base,
        "mixed_backend": "cuda_v0",
    }
    w8abf16_cuda_v1_base = {
        **w8abf16_reference_base,
        "mixed_backend": "cuda_v1",
    }
    w8a16_cuda_v1_base = {
        **w8a16_reference_base,
        "mixed_backend": "cuda_v1",
    }
    w8abf16_wmma_v0_base = {
        **w8abf16_reference_base,
        "mixed_backend": "wmma_v0",
    }
    w8a16_wmma_v0_base = {
        **w8a16_reference_base,
        "mixed_backend": "wmma_v0",
    }
    w8abf16_triton_v0_base = {
        **w8abf16_reference_base,
        "mixed_backend": "triton_v0",
    }
    w8a16_triton_v0_base = {
        **w8a16_reference_base,
        "mixed_backend": "triton_v0",
    }
    w8abf16_cutlass_mixed_v0_base = {
        **w8abf16_reference_base,
        "mixed_backend": "cutlass_mixed_v0",
    }
    w8a16_cutlass_mixed_v0_base = {
        **w8a16_reference_base,
        "mixed_backend": "cutlass_mixed_v0",
    }
    w8abf16_cached_dequant_lowp_v0_base = {
        **w8abf16_reference_base,
        "mixed_backend": "cached_dequant_lowp_v0",
    }
    w8a16_cached_dequant_lowp_v0_base = {
        **w8a16_reference_base,
        "mixed_backend": "cached_dequant_lowp_v0",
    }

    p0_atom_targets = [
        "interaction_block.*.refine_block_atom_graph.node_FFN",
        "interaction_block.*.refine_block_atom_graph.edge_FFN",
        "interaction_block.*.attn_block_atom_graph.source_weight_linear",
        "interaction_block.*.attn_block_atom_graph.target_weight_linear",
    ]
    atom_attention_linear_targets = [
        "interaction_block.*.attn_block_atom_graph.source_weight_linear",
        "interaction_block.*.attn_block_atom_graph.target_weight_linear",
    ]
    line_graph_targets = [
        "interaction_block.*.attn_block_line_graph.source_weight_linear",
        "interaction_block.*.attn_block_line_graph.target_weight_linear",
        "interaction_block.*.refine_block_line_graph.node_FFN",
        "interaction_block.*.refine_block_line_graph.edge_FFN",
    ]
    atom_nonlinear_core_targets = [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core",
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_core",
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update.mlp_core",
    ]
    atom_attn_edge_nonlinear_core_targets = [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core",
    ]
    atom_refine_edge_nonlinear_core_targets = [
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_core",
    ]
    atom_attn_node_nonlinear_core_targets = [
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update.mlp_core",
    ]
    atom_edge_nonlinear_gate_targets = [
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_gate",
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_gate",
    ]
    atom_node_nonlinear_gate_targets = [
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update.mlp_gate",
    ]
    line_edge_nonlinear_core_targets = [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core",
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.0",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.0",
    ]
    line_node_nonlinear_core_targets = [
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update.mlp_core",
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update.0",
    ]
    line_edge_nonlinear_gate_targets = [
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate",
    ]
    line_node_nonlinear_gate_targets = [
        "interaction_block.*.attn_block_line_graph.node_nonlinear_update.mlp_gate",
    ]
    line_edge_nonlinear_core_gate_targets = [
        *line_edge_nonlinear_core_targets,
        *line_edge_nonlinear_gate_targets,
    ]
    refine_line_edge_first_targets = [
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
    ]
    refine_line_edge_second_targets = [
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
    ]
    attn_line_edge_gate_second_blocks_7_9_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
        for idx in range(7, 10)
    ]
    attn_line_edge_core_second_blocks_7_9_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"
        for idx in range(7, 10)
    ]
    attn_line_edge_gate_second_blocks_8_9_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
        for idx in range(8, 10)
    ]
    attn_line_edge_core_second_blocks_8_9_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"
        for idx in range(8, 10)
    ]
    attn_line_edge_core_second_blocks_0_3_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"
        for idx in range(0, 4)
    ]
    attn_line_edge_gate_second_blocks_0_3_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
        for idx in range(0, 4)
    ]
    attn_line_edge_core_second_blocks_4_6_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"
        for idx in range(4, 7)
    ]
    attn_line_edge_gate_second_blocks_4_6_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
        for idx in range(4, 7)
    ]
    attn_line_edge_core_second_blocks_0_6_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"
        for idx in range(0, 7)
    ]
    attn_line_edge_gate_second_blocks_0_6_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
        for idx in range(0, 7)
    ]
    attn_line_edge_gate_second_block_7_targets = [
        "interaction_block.7.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
    ]
    attn_line_edge_core_second_block_7_targets = [
        "interaction_block.7.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
    ]
    line_node_nonlinear_core_gate_targets = [
        *line_node_nonlinear_core_targets,
        *line_node_nonlinear_gate_targets,
    ]
    line_nonlinear_core_gate_targets = [
        *line_edge_nonlinear_core_gate_targets,
        *line_node_nonlinear_core_gate_targets,
    ]
    angle_priority_targets = [
        *line_graph_targets,
        *line_edge_nonlinear_core_targets,
        *line_edge_nonlinear_gate_targets,
    ]
    readout_pre_final_targets = [
        "energy_head.energy_head.layers.0",
        "energy_head.energy_head.0.mlp_core.layers.0",
    ]
    p2_targets = [
        *line_edge_nonlinear_core_targets,
        *line_node_nonlinear_core_targets,
        *readout_pre_final_targets,
    ]
    p0_line_stable_targets = [
        *p0_atom_targets,
        *line_graph_targets,
        *line_edge_nonlinear_core_targets,
        *line_node_nonlinear_core_targets,
    ]
    p0_line_gate_stable_targets = [
        *p0_line_stable_targets,
        *line_edge_nonlinear_gate_targets,
        *line_node_nonlinear_gate_targets,
    ]
    p0_line_gate_atom_attn_edge_targets = [
        *p0_line_gate_stable_targets,
        *atom_attn_edge_nonlinear_core_targets,
    ]
    all_single_pass_targets = [
        *p0_atom_targets,
        *line_graph_targets,
        *line_nonlinear_core_gate_targets,
        *atom_attn_edge_nonlinear_core_targets,
        *atom_node_nonlinear_gate_targets,
    ]
    p1_targets = [
        *line_graph_targets,
        *atom_nonlinear_core_targets,
    ]

    configs = {
        "node_ffn_w8a32": {
            **base,
            "mode": "node_ffn_w8a32",
            "targets": [
                "interaction_block.*.refine_block_atom_graph.node_FFN",
            ],
        },
        "edge_ffn_w8a32": {
            **base,
            "mode": "edge_ffn_w8a32",
            "targets": [
                "interaction_block.*.refine_block_atom_graph.edge_FFN",
            ],
        },
        "attn_atom_linear_w8a32": {
            **base,
            "mode": "attn_atom_linear_w8a32",
            "targets": atom_attention_linear_targets,
        },
        "attn_linear_w8a32": {
            **base,
            "mode": "attn_linear_w8a32",
            "targets": atom_attention_linear_targets,
        },
        "atom_p0_w8a32": {
            **base,
            "mode": "atom_p0_w8a32",
            "targets": p0_atom_targets,
        },
        "fwd_only_w8a32_atom_p0": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_atom_p0",
            "targets": p0_atom_targets,
        },
        "bwd_w8a32_atom_p0": {
            **backward_only_base,
            "mode": "bwd_w8a32_atom_p0",
            "targets": p0_atom_targets,
        },
        "atom_p0_w8a32_kernel": {
            **kernel_base,
            "mode": "atom_p0_w8a32_kernel",
            "targets": p0_atom_targets,
        },
        "atom_p0_w8a32_torchao": {
            **torchao_weight_only_base,
            "mode": "atom_p0_w8a32_torchao",
            "targets": p0_atom_targets,
        },
        "atom_edge_mlp_w8a32": {
            **base,
            "mode": "atom_edge_mlp_w8a32",
            "targets": [
                "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core",
                "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update.mlp_core",
            ],
        },
        "atom_attn_edge_mlp_w8a32": {
            **base,
            "mode": "atom_attn_edge_mlp_w8a32",
            "targets": atom_attn_edge_nonlinear_core_targets,
        },
        "atom_refine_edge_mlp_w8a32": {
            **base,
            "mode": "atom_refine_edge_mlp_w8a32",
            "targets": atom_refine_edge_nonlinear_core_targets,
        },
        "atom_node_mlp_w8a32": {
            **base,
            "mode": "atom_node_mlp_w8a32",
            "targets": atom_attn_node_nonlinear_core_targets,
        },
        "atom_attn_node_mlp_w8a32": {
            **base,
            "mode": "atom_attn_node_mlp_w8a32",
            "targets": atom_attn_node_nonlinear_core_targets,
        },
        "atom_nonlinear_mlp_w8a32": {
            **base,
            "mode": "atom_nonlinear_mlp_w8a32",
            "targets": atom_nonlinear_core_targets,
        },
        "line_graph_w8a32": {
            **base,
            "mode": "line_graph_w8a32",
            "targets": line_graph_targets,
        },
        "fwd_only_w8a32_line_graph": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_line_graph",
            "targets": line_graph_targets,
        },
        "bwd_w8a32_line_graph": {
            **backward_only_base,
            "mode": "bwd_w8a32_line_graph",
            "targets": line_graph_targets,
        },
        "line_graph_w8a32_torchao": {
            **torchao_weight_only_base,
            "mode": "line_graph_w8a32_torchao",
            "targets": line_graph_targets,
        },
        "line_graph_linear_bf16": {
            **linear_bf16_base,
            "mode": "line_graph_linear_bf16",
            "targets": line_graph_targets,
        },
        "line_graph_linear_fp16": {
            **linear_fp16_base,
            "mode": "line_graph_linear_fp16",
            "targets": line_graph_targets,
        },
        "line_graph_linear_cached_bf16": {
            **cached_linear_bf16_base,
            "mode": "line_graph_linear_cached_bf16",
            "targets": line_graph_targets,
        },
        "line_graph_linear_cached_fp16": {
            **cached_linear_fp16_base,
            "mode": "line_graph_linear_cached_fp16",
            "targets": line_graph_targets,
        },
        "line_node_ffn_w8a32": {
            **base,
            "mode": "line_node_ffn_w8a32",
            "targets": [
                "interaction_block.*.refine_block_line_graph.node_FFN",
            ],
        },
        "line_edge_ffn_w8a32": {
            **base,
            "mode": "line_edge_ffn_w8a32",
            "targets": [
                "interaction_block.*.refine_block_line_graph.edge_FFN",
            ],
        },
        "attn_line_linear_w8a32": {
            **base,
            "mode": "attn_line_linear_w8a32",
            "targets": [
                "interaction_block.*.attn_block_line_graph.source_weight_linear",
                "interaction_block.*.attn_block_line_graph.target_weight_linear",
            ],
        },
        "line_edge_mlp_w8a32": {
            **base,
            "mode": "line_edge_mlp_w8a32",
            "targets": line_edge_nonlinear_core_targets,
        },
        "fwd_only_w8a32_line_edge_mlp": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_line_edge_mlp",
            "targets": line_edge_nonlinear_core_targets,
        },
        "bwd_w8a32_line_edge_mlp": {
            **backward_only_base,
            "mode": "bwd_w8a32_line_edge_mlp",
            "targets": line_edge_nonlinear_core_targets,
        },
        "line_edge_mlp_w8a32_torchao": {
            **torchao_weight_only_base,
            "mode": "line_edge_mlp_w8a32_torchao",
            "targets": line_edge_nonlinear_core_targets,
        },
        "line_edge_mlp_triton_w8a32": {
            **triton_w8a32_base,
            "mode": "line_edge_mlp_triton_w8a32",
            "targets": line_edge_nonlinear_core_targets,
        },
        "line_node_mlp_w8a32": {
            **base,
            "mode": "line_node_mlp_w8a32",
            "targets": line_node_nonlinear_core_targets,
        },
        "fwd_only_w8a32_line_node_mlp": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_line_node_mlp",
            "targets": line_node_nonlinear_core_targets,
        },
        "bwd_w8a32_line_node_mlp": {
            **backward_only_base,
            "mode": "bwd_w8a32_line_node_mlp",
            "targets": line_node_nonlinear_core_targets,
        },
        "line_node_mlp_w8a32_torchao": {
            **torchao_weight_only_base,
            "mode": "line_node_mlp_w8a32_torchao",
            "targets": line_node_nonlinear_core_targets,
        },
        "line_edge_gate_mlp_w8a32": {
            **base,
            "mode": "line_edge_gate_mlp_w8a32",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "fwd_only_w8a32_line_edge_gate_mlp": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_line_edge_gate_mlp",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "bwd_w8a32_line_edge_gate_mlp": {
            **backward_only_base,
            "mode": "bwd_w8a32_line_edge_gate_mlp",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "line_edge_gate_mlp_w8a32_torchao": {
            **torchao_weight_only_base,
            "mode": "line_edge_gate_mlp_w8a32_torchao",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "line_edge_gate_mlp_triton_w8a32": {
            **triton_w8a32_base,
            "mode": "line_edge_gate_mlp_triton_w8a32",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "line_edge_gate_linear_bf16": {
            **linear_bf16_base,
            "mode": "line_edge_gate_linear_bf16",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "line_edge_gate_linear_fp16": {
            **linear_fp16_base,
            "mode": "line_edge_gate_linear_fp16",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "line_edge_gate_linear_cached_bf16": {
            **cached_linear_bf16_base,
            "mode": "line_edge_gate_linear_cached_bf16",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "line_edge_gate_linear_cached_fp16": {
            **cached_linear_fp16_base,
            "mode": "line_edge_gate_linear_cached_fp16",
            "targets": line_edge_nonlinear_gate_targets,
        },
        "line_node_gate_mlp_w8a32": {
            **base,
            "mode": "line_node_gate_mlp_w8a32",
            "targets": line_node_nonlinear_gate_targets,
        },
        "fwd_only_w8a32_line_node_gate_mlp": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_line_node_gate_mlp",
            "targets": line_node_nonlinear_gate_targets,
        },
        "bwd_w8a32_line_node_gate_mlp": {
            **backward_only_base,
            "mode": "bwd_w8a32_line_node_gate_mlp",
            "targets": line_node_nonlinear_gate_targets,
        },
        "line_node_gate_mlp_w8a32_torchao": {
            **torchao_weight_only_base,
            "mode": "line_node_gate_mlp_w8a32_torchao",
            "targets": line_node_nonlinear_gate_targets,
        },
        "line_node_gate_linear_bf16": {
            **linear_bf16_base,
            "mode": "line_node_gate_linear_bf16",
            "targets": line_node_nonlinear_gate_targets,
        },
        "line_node_gate_linear_fp16": {
            **linear_fp16_base,
            "mode": "line_node_gate_linear_fp16",
            "targets": line_node_nonlinear_gate_targets,
        },
        "line_node_gate_linear_cached_bf16": {
            **cached_linear_bf16_base,
            "mode": "line_node_gate_linear_cached_bf16",
            "targets": line_node_nonlinear_gate_targets,
        },
        "line_node_gate_linear_cached_fp16": {
            **cached_linear_fp16_base,
            "mode": "line_node_gate_linear_cached_fp16",
            "targets": line_node_nonlinear_gate_targets,
        },
        "line_edge_core_gate_mlp_w8a32": {
            **base,
            "mode": "line_edge_core_gate_mlp_w8a32",
            "targets": line_edge_nonlinear_core_gate_targets,
        },
        "line_edge_core_gate_mlp_triton_w8a32": {
            **triton_w8a32_base,
            "mode": "line_edge_core_gate_mlp_triton_w8a32",
            "targets": line_edge_nonlinear_core_gate_targets,
        },
        "triton_w8a8_attn_line_edge_second_static": {
            **triton_w8a8_static_base,
            "mode": "triton_w8a8_attn_line_edge_second_static",
            "targets": [
                "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
            ],
        },
        "triton_w8a8_refine_line_edge_second_static": {
            **triton_w8a8_static_base,
            "mode": "triton_w8a8_refine_line_edge_second_static",
            "targets": [
                "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
            ],
        },
        "triton_w8a8_line_edge_second_static": {
            **triton_w8a8_static_base,
            "mode": "triton_w8a8_line_edge_second_static",
            "targets": [
                "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
                "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
            ],
        },
        "mixed_refine_w8a8_attn_w8a32_line_edge_second": {
            **base,
            "mode": "mixed_refine_w8a8_attn_w8a32_line_edge_second",
            "target_specs": [
                {
                    **triton_w8a8_static_base,
                    "targets": [
                        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
                    ],
                },
                {
                    **triton_w8a32_base,
                    "targets": [
                        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
                    ],
                },
            ],
        },
        "p8d_refine_second_w8a8_refine_first_w8a32": {
            **base,
            "mode": "p8d_refine_second_w8a8_refine_first_w8a32",
            "target_specs": [
                {
                    **triton_w8a8_static_base,
                    "targets": refine_line_edge_second_targets,
                },
                {
                    **triton_w8a32_base,
                    "targets": refine_line_edge_first_targets,
                },
            ],
        },
        "p8d_refine_first_second_w8a8": {
            **triton_w8a8_static_base,
            "mode": "p8d_refine_first_second_w8a8",
            "targets": [
                *refine_line_edge_first_targets,
                *refine_line_edge_second_targets,
            ],
        },
        "p8d_refine_w8a8_attn_line_gate_second_blocks_7_9_w8a8": {
            **triton_w8a8_static_base,
            "mode": "p8d_refine_w8a8_attn_line_gate_second_blocks_7_9_w8a8",
            "targets": [
                *refine_line_edge_second_targets,
                *attn_line_edge_gate_second_blocks_7_9_targets,
            ],
        },
        "p8d_refine_w8a8_attn_line_core_second_blocks_7_9_w8a8": {
            **triton_w8a8_static_base,
            "mode": "p8d_refine_w8a8_attn_line_core_second_blocks_7_9_w8a8",
            "targets": [
                *refine_line_edge_second_targets,
                *attn_line_edge_core_second_blocks_7_9_targets,
            ],
        },
        "p8d_refine_w8a8_attn_line_core_gate_second_blocks_7_9_w8a8": {
            **triton_w8a8_static_base,
            "mode": "p8d_refine_w8a8_attn_line_core_gate_second_blocks_7_9_w8a8",
            "targets": [
                *refine_line_edge_second_targets,
                *attn_line_edge_core_second_blocks_7_9_targets,
                *attn_line_edge_gate_second_blocks_7_9_targets,
            ],
        },
        "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8": {
            **triton_w8a8_static_base,
            "mode": "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8",
            "targets": [
                *refine_line_edge_second_targets,
                *attn_line_edge_core_second_blocks_8_9_targets,
                *attn_line_edge_gate_second_blocks_8_9_targets,
            ],
        },
        "p8e_blocks_8_9_core_gate_plus_block7_core_w8a8": {
            **triton_w8a8_static_base,
            "mode": "p8e_blocks_8_9_core_gate_plus_block7_core_w8a8",
            "targets": [
                *refine_line_edge_second_targets,
                *attn_line_edge_core_second_blocks_8_9_targets,
                *attn_line_edge_gate_second_blocks_8_9_targets,
                *attn_line_edge_core_second_block_7_targets,
            ],
        },
        "p8e_blocks_8_9_core_gate_plus_block7_gate_w8a8": {
            **triton_w8a8_static_base,
            "mode": "p8e_blocks_8_9_core_gate_plus_block7_gate_w8a8",
            "targets": [
                *refine_line_edge_second_targets,
                *attn_line_edge_core_second_blocks_8_9_targets,
                *attn_line_edge_gate_second_blocks_8_9_targets,
                *attn_line_edge_gate_second_block_7_targets,
            ],
        },
        "triton_smooth_w8a8_line_edge_second_static": {
            **triton_smooth_w8a8_static_base,
            "mode": "triton_smooth_w8a8_line_edge_second_static",
            "targets": [
                "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
                "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
            ],
        },
        "line_edge_core_gate_linear_cached_bf16": {
            **cached_linear_bf16_base,
            "mode": "line_edge_core_gate_linear_cached_bf16",
            "targets": line_edge_nonlinear_core_gate_targets,
        },
        "line_edge_core_gate_linear_cached_fp16": {
            **cached_linear_fp16_base,
            "mode": "line_edge_core_gate_linear_cached_fp16",
            "targets": line_edge_nonlinear_core_gate_targets,
        },
        "line_node_core_gate_mlp_w8a32": {
            **base,
            "mode": "line_node_core_gate_mlp_w8a32",
            "targets": line_node_nonlinear_core_gate_targets,
        },
        "line_nonlinear_core_gate_mlp_w8a32": {
            **base,
            "mode": "line_nonlinear_core_gate_mlp_w8a32",
            "targets": line_nonlinear_core_gate_targets,
        },
        "fwd_only_w8a32_angle_priority": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_angle_priority",
            "targets": angle_priority_targets,
        },
        "bwd_w8a32_angle_priority": {
            **backward_only_base,
            "mode": "bwd_w8a32_angle_priority",
            "targets": angle_priority_targets,
        },
        "atom_edge_gate_mlp_w8a32": {
            **base,
            "mode": "atom_edge_gate_mlp_w8a32",
            "targets": atom_edge_nonlinear_gate_targets,
        },
        "atom_node_gate_mlp_w8a32": {
            **base,
            "mode": "atom_node_gate_mlp_w8a32",
            "targets": atom_node_nonlinear_gate_targets,
        },
        "p1_w8a32": {
            **base,
            "mode": "p1_w8a32",
            "targets": p1_targets,
        },
        "readout_pre_final_w8a32": {
            **base,
            "mode": "readout_pre_final_w8a32",
            "targets": readout_pre_final_targets,
        },
        "p2_w8a32": {
            **base,
            "mode": "p2_w8a32",
            "targets": p2_targets,
        },
        "p0_line_stable_w8a32": {
            **base,
            "mode": "p0_line_stable_w8a32",
            "targets": p0_line_stable_targets,
        },
        "p0_line_gate_stable_w8a32": {
            **base,
            "mode": "p0_line_gate_stable_w8a32",
            "targets": p0_line_gate_stable_targets,
        },
        "p0_line_gate_stable_w8a32_torchao": {
            **torchao_weight_only_base,
            "mode": "p0_line_gate_stable_w8a32_torchao",
            "targets": p0_line_gate_stable_targets,
        },
        "p0_line_gate_atom_attn_edge_w8a32": {
            **base,
            "mode": "p0_line_gate_atom_attn_edge_w8a32",
            "targets": p0_line_gate_atom_attn_edge_targets,
        },
        "p0_line_gate_paper_stable_w8a32": {
            **base,
            "mode": "p0_line_gate_paper_stable_w8a32",
            "targets": p0_line_gate_atom_attn_edge_targets,
        },
        "p0_line_gate_paper_stable_linear_bf16": {
            **linear_bf16_base,
            "mode": "p0_line_gate_paper_stable_linear_bf16",
            "targets": p0_line_gate_stable_targets,
        },
        "p0_line_gate_paper_stable_linear_fp16": {
            **linear_fp16_base,
            "mode": "p0_line_gate_paper_stable_linear_fp16",
            "targets": p0_line_gate_stable_targets,
        },
        "p0_line_gate_paper_stable_linear_cached_bf16": {
            **cached_linear_bf16_base,
            "mode": "p0_line_gate_paper_stable_linear_cached_bf16",
            "targets": p0_line_gate_stable_targets,
        },
        "p0_line_gate_paper_stable_linear_cached_fp16": {
            **cached_linear_fp16_base,
            "mode": "p0_line_gate_paper_stable_linear_cached_fp16",
            "targets": p0_line_gate_stable_targets,
        },
        "fwd_only_w8a32_p0_line_gate_paper_stable": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_p0_line_gate_paper_stable",
            "targets": p0_line_gate_atom_attn_edge_targets,
        },
        "bwd_w8a32_p0_line_gate_paper_stable": {
            **backward_only_base,
            "mode": "bwd_w8a32_p0_line_gate_paper_stable",
            "targets": p0_line_gate_atom_attn_edge_targets,
        },
        "all_single_pass_w8a32": {
            **base,
            "mode": "all_single_pass_w8a32",
            "targets": all_single_pass_targets,
        },
        "fwd_only_w8a32_all_single_pass": {
            **forward_only_base,
            "mode": "fwd_only_w8a32_all_single_pass",
            "targets": all_single_pass_targets,
        },
        "bwd_w8a32_all_single_pass": {
            **backward_only_base,
            "mode": "bwd_w8a32_all_single_pass",
            "targets": all_single_pass_targets,
        },
    }

    a8_screen_targets = {
        # Single candidate groups, evaluated independently first.
        "a8_refine_atom_node_ffn": [
            "interaction_block.*.refine_block_atom_graph.node_FFN",
        ],
        "a8_refine_atom_edge_ffn": [
            "interaction_block.*.refine_block_atom_graph.edge_FFN",
        ],
        "a8_attn_atom_source_weight_linear": [
            "interaction_block.*.attn_block_atom_graph.source_weight_linear",
        ],
        "a8_attn_atom_target_weight_linear": [
            "interaction_block.*.attn_block_atom_graph.target_weight_linear",
        ],
        "a8_attn_atom_edge_core": [
            "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core",
        ],
        "a8_attn_line_source_weight_linear": [
            "interaction_block.*.attn_block_line_graph.source_weight_linear",
        ],
        "a8_attn_line_target_weight_linear": [
            "interaction_block.*.attn_block_line_graph.target_weight_linear",
        ],
        "a8_refine_line_node_ffn": [
            "interaction_block.*.refine_block_line_graph.node_FFN",
        ],
        "a8_refine_line_edge_ffn": [
            "interaction_block.*.refine_block_line_graph.edge_FFN",
        ],
        "a8_attn_line_edge_core": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.0",
        ],
        "a8_attn_line_edge_gate": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate",
        ],
        "a8_refine_line_edge_core": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.0",
        ],
        "a8_refine_line_edge_gate": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate",
        ],
        "a8_attn_line_node_core": [
            "interaction_block.*.attn_block_line_graph.node_nonlinear_update.mlp_core",
            "interaction_block.*.attn_block_line_graph.node_nonlinear_update.0",
        ],
        "a8_attn_line_node_gate": [
            "interaction_block.*.attn_block_line_graph.node_nonlinear_update.mlp_gate",
        ],
        # Related small modules, evaluated after single groups.
        "a8_atom_p0": p0_atom_targets,
        "a8_line_graph": line_graph_targets,
        "a8_attn_line_edge_core_gate": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.0",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate",
        ],
        "a8_refine_line_edge_core_gate": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.0",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate",
        ],
        "a8_attn_line_edge_core_gate_first": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "a8_attn_line_edge_core_gate_second": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
        "a8_refine_line_edge_core_gate_first": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "a8_refine_line_edge_core_gate_second": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
        "a8_line_edge_core_gate_first": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "a8_line_edge_core_gate_second": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
        "a8_attn_line_node_core_gate": [
            "interaction_block.*.attn_block_line_graph.node_nonlinear_update.mlp_core",
            "interaction_block.*.attn_block_line_graph.node_nonlinear_update.0",
            "interaction_block.*.attn_block_line_graph.node_nonlinear_update.mlp_gate",
        ],
        "a8_line_edge_core_gate": line_edge_nonlinear_core_gate_targets,
        "a8_line_node_core_gate": line_node_nonlinear_core_gate_targets,
        "a8_line_nonlinear_core_gate": line_nonlinear_core_gate_targets,
        # Full candidate sets, evaluated last.
        "a8_p0_line_gate_paper_stable": p0_line_gate_atom_attn_edge_targets,
        "a8_all_single_pass": all_single_pass_targets,
    }
    configs.update(
        {
            mode: {
                **a8_base,
                "mode": mode,
                "targets": targets,
            }
            for mode, targets in a8_screen_targets.items()
        }
    )
    for mode, targets in a8_screen_targets.items():
        configs[f"{mode}_static_a8"] = {
            **a8_base,
            "mode": f"{mode}_static_a8",
            "targets": targets,
            "activation_dtype": "int8_fake_static_per_tensor",
            "activation_scale_granularity": "static_per_tensor",
        }
        configs[f"{mode}_smooth_a8"] = {
            **smooth_a8_base,
            "mode": f"{mode}_smooth_a8",
            "targets": targets,
        }
        configs[f"{mode}_smooth_a8_alpha075"] = {
            **smooth_a8_base,
            "mode": f"{mode}_smooth_a8_alpha075",
            "targets": targets,
            "smooth_alpha": 0.75,
        }
        configs[f"{mode}_tile_m32_a8"] = {
            **a8_base,
            "mode": f"{mode}_tile_m32_a8",
            "targets": targets,
            "activation_dtype": "int8_fake_dynamic_per_m_block",
            "activation_scale_granularity": "dynamic_per_m_block",
            "activation_block_m": 32,
        }

    configs["triton_w8a8_attn_atom_edge_core_smooth_static"] = {
        **triton_smooth_w8a8_static_base,
        "mode": "triton_w8a8_attn_atom_edge_core_smooth_static",
        "targets": atom_attn_edge_nonlinear_core_targets,
    }
    configs["triton_w8a8_attn_atom_edge_core_smooth_static_alpha075"] = {
        **triton_smooth_w8a8_static_base,
        "mode": "triton_w8a8_attn_atom_edge_core_smooth_static_alpha075",
        "targets": atom_attn_edge_nonlinear_core_targets,
        "smooth_alpha": 0.75,
    }

    current_p8e_w8a8_targets = [
        *refine_line_edge_second_targets,
        *attn_line_edge_core_second_blocks_8_9_targets,
        *attn_line_edge_gate_second_blocks_8_9_targets,
    ]
    for suffix, extra_targets in {
        "0_3": [
            *attn_line_edge_core_second_blocks_0_3_targets,
            *attn_line_edge_gate_second_blocks_0_3_targets,
        ],
        "4_6": [
            *attn_line_edge_core_second_blocks_4_6_targets,
            *attn_line_edge_gate_second_blocks_4_6_targets,
        ],
        "0_6": [
            *attn_line_edge_core_second_blocks_0_6_targets,
            *attn_line_edge_gate_second_blocks_0_6_targets,
        ],
    }.items():
        mode = f"p8f_attn_line_edge_second_blocks_{suffix}_core_gate_w8a8"
        configs[mode] = {
            **triton_w8a8_static_base,
            "mode": mode,
            "targets": [
                *current_p8e_w8a8_targets,
                *extra_targets,
            ],
        }
    for idx in range(7):
        sweep_targets_by_branch = {
            "core": [
                f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            ],
            "gate": [
                f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
            ],
            "core_gate": [
                f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
                f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
            ],
        }
        for branch, extra_targets in sweep_targets_by_branch.items():
            mode = f"sweep_attn_line_edge_second_b{idx}_{branch}_w8a8"
            configs[mode] = {
                **triton_w8a8_static_base,
                "mode": mode,
                "targets": [
                    *current_p8e_w8a8_targets,
                    *extra_targets,
                ],
            }

    # FP32 hotspot probes layered on top of the stable p8e runtime.  These are
    # intentionally separate from the older standalone A8 sensitivity modes:
    # they answer whether a currently-FP32 hotspot can be added to p8e without
    # changing the default p8e coverage.
    fp32_hotspot_probe_targets = {
        "line_attn_edge_first_core_gate": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "line_refine_edge_first_core_gate": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "line_attn_weight_source_target": [
            "interaction_block.*.attn_block_line_graph.source_weight_linear",
            "interaction_block.*.attn_block_line_graph.target_weight_linear",
        ],
        "atom_attn_weight_source_target": [
            "interaction_block.*.attn_block_atom_graph.source_weight_linear",
            "interaction_block.*.attn_block_atom_graph.target_weight_linear",
        ],
        "atom_attn_node_update": [
            "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
        ],
        "atom_refine_node_update": [
            "interaction_block.*.refine_block_atom_graph.node_FFN",
        ],
        "atom_attn_edge_update": [
            "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
        ],
        "atom_refine_edge_update": [
            "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
        ],
    }
    for suffix, extra_targets in fp32_hotspot_probe_targets.items():
        mode = f"hotspot_p8e_{suffix}_w8a8"
        configs[mode] = {
            **triton_w8a8_static_base,
            "mode": mode,
            "targets": [
                *current_p8e_w8a8_targets,
                *extra_targets,
            ],
        }
        for precision_name, precision_base in {
            "w8abf16_cached_dequant_lowp_v0": w8abf16_cached_dequant_lowp_v0_base,
            "w8a16_cached_dequant_lowp_v0": w8a16_cached_dequant_lowp_v0_base,
            "bf16": cached_linear_bf16_base,
            "fp16": cached_linear_fp16_base,
        }.items():
            mixed_mode = f"hotspot_p8e_{suffix}_{precision_name}"
            configs[mixed_mode] = {
                **base,
                "mode": mixed_mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **precision_base,
                        "targets": extra_targets,
                    },
                ],
            }

    for branch, extra_targets in {
        "gate": attn_line_edge_gate_second_block_7_targets,
        "core": attn_line_edge_core_second_block_7_targets,
        "core_gate": [
            *attn_line_edge_core_second_block_7_targets,
            *attn_line_edge_gate_second_block_7_targets,
        ],
    }.items():
        mode = f"sweep_attn_line_edge_second_b7_{branch}_w8a32"
        configs[mode] = {
            **base,
            "mode": mode,
            "target_specs": [
                {
                    **triton_w8a8_static_base,
                    "targets": current_p8e_w8a8_targets,
                },
                {
                    **triton_w8a32_base,
                    "targets": extra_targets,
                },
            ],
        }

    mixed_sensitive_extra_targets = {
        "b7_gate": attn_line_edge_gate_second_block_7_targets,
        "b7_core": attn_line_edge_core_second_block_7_targets,
        "b7_core_gate": [
            *attn_line_edge_core_second_block_7_targets,
            *attn_line_edge_gate_second_block_7_targets,
        ],
        "b4_core_gate": [
            "interaction_block.4.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.4.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
        "b5_core_gate": [
            "interaction_block.5.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.5.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
        "b6_core_gate": [
            "interaction_block.6.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.6.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
        "b4_6_core_gate": [
            *attn_line_edge_core_second_blocks_4_6_targets,
            *attn_line_edge_gate_second_blocks_4_6_targets,
        ],
    }
    for suffix, extra_targets in mixed_sensitive_extra_targets.items():
        mode = f"mixed_sensitive_{suffix}_w8a32"
        configs.setdefault(
            mode,
            {
                **base,
                "mode": mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **triton_w8a32_base,
                        "targets": extra_targets,
                    },
                ],
            },
        )
        for precision_name, precision_base in {
            "bf16": cached_linear_bf16_base,
            "fp16": cached_linear_fp16_base,
        }.items():
            mode = f"mixed_sensitive_{suffix}_{precision_name}"
            configs[mode] = {
                **base,
                "mode": mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **precision_base,
                        "targets": extra_targets,
                    },
                ],
            }

        for precision_name, precision_base in {
            "w8abf16_probe": w8abf16_reference_base,
            "w8a16_probe": w8a16_reference_base,
            "w8abf16_cuda_v0": w8abf16_cuda_v0_base,
            "w8a16_cuda_v0": w8a16_cuda_v0_base,
            "w8abf16_cuda_v1": w8abf16_cuda_v1_base,
            "w8a16_cuda_v1": w8a16_cuda_v1_base,
            "w8abf16_wmma_v0": w8abf16_wmma_v0_base,
            "w8a16_wmma_v0": w8a16_wmma_v0_base,
            "w8abf16_triton_v0": w8abf16_triton_v0_base,
            "w8a16_triton_v0": w8a16_triton_v0_base,
            "w8abf16_cutlass_mixed_v0": w8abf16_cutlass_mixed_v0_base,
            "w8a16_cutlass_mixed_v0": w8a16_cutlass_mixed_v0_base,
            "w8abf16_cached_dequant_lowp_v0": w8abf16_cached_dequant_lowp_v0_base,
            "w8a16_cached_dequant_lowp_v0": w8a16_cached_dequant_lowp_v0_base,
        }.items():
            mode = f"mixed_sensitive_{suffix}_{precision_name}"
            configs[mode] = {
                **base,
                "mode": mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **precision_base,
                        "targets": extra_targets,
                    },
                ],
            }

    # W16A16/BF16 coverage + speed probes.  These modes keep the stable p8e
    # W8A8 scope intact, then try cached FP16/BF16 Linear only on layers that
    # are still FP32 in the p8e runtime.
    w16_probe_precisions = {
        "bf16": cached_linear_bf16_base,
        "fp16": cached_linear_fp16_base,
    }
    w16_sensitive_probe_targets = {
        "sensitive_b7_core_gate": mixed_sensitive_extra_targets["b7_core_gate"],
        "sensitive_b6_core_gate": mixed_sensitive_extra_targets["b6_core_gate"],
        "sensitive_b4_6_core_gate": mixed_sensitive_extra_targets["b4_6_core_gate"],
    }
    for suffix, extra_targets in w16_sensitive_probe_targets.items():
        for precision_name, precision_base in w16_probe_precisions.items():
            mode = f"{precision_name}_{suffix}"
            configs[mode] = {
                **base,
                "mode": mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **precision_base,
                        "targets": extra_targets,
                    },
                ],
            }

    line_edge_hotspot_probe_targets = [
        # First projections feeding line edge GatedMLP.
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        # Attention source/target weight linears.
        "interaction_block.*.attn_block_line_graph.source_weight_linear",
        "interaction_block.*.attn_block_line_graph.target_weight_linear",
        # Attn line edge second projections that remain FP32 in p8e.
        *attn_line_edge_core_second_blocks_0_6_targets,
        *attn_line_edge_gate_second_blocks_0_6_targets,
        *attn_line_edge_core_second_block_7_targets,
        *attn_line_edge_gate_second_block_7_targets,
    ]
    atom_hotspot_probe_targets = [
        "interaction_block.*.attn_block_atom_graph.source_weight_linear",
        "interaction_block.*.attn_block_atom_graph.target_weight_linear",
        "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
        "interaction_block.*.refine_block_atom_graph.node_nonlinear_update",
        "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
        "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
    ]
    angle_probe_targets = [
        "three_body_embedding.angle_embedding",
        "three_body_embedding.swish_layer.linear",
    ]
    for suffix, extra_targets in {
        "line_edge_hotspot_probe": line_edge_hotspot_probe_targets,
        "atom_hotspot_probe": atom_hotspot_probe_targets,
        "angle_probe": angle_probe_targets,
    }.items():
        for precision_name, precision_base in w16_probe_precisions.items():
            mode = f"{precision_name}_{suffix}"
            configs[mode] = {
                **base,
                "mode": mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **precision_base,
                        "targets": extra_targets,
                    },
                ],
            }

    # Fine-grained W16A16/BF16 hotspot sweep after broad probes.  These modes
    # are intentionally small so a broad-group precision failure does not hide
    # usable low-precision subgroups.
    w16_fine_hotspot_targets = {
        "line_attn_edge_first_core": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        ],
        "line_attn_edge_first_gate": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "line_attn_edge_first_core_gate": [
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "line_refine_edge_first_core": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
        ],
        "line_refine_edge_first_gate": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "line_refine_edge_first_core_gate": [
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.0",
            "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.0",
        ],
        "line_attn_weight_source_target": [
            "interaction_block.*.attn_block_line_graph.source_weight_linear",
            "interaction_block.*.attn_block_line_graph.target_weight_linear",
        ],
        "atom_attn_node_update": [
            "interaction_block.*.attn_block_atom_graph.node_nonlinear_update",
        ],
        "atom_refine_node_update": [
            "interaction_block.*.refine_block_atom_graph.node_nonlinear_update",
        ],
        "atom_attn_edge_update": [
            "interaction_block.*.attn_block_atom_graph.edge_nonlinear_update",
        ],
        "atom_refine_edge_update": [
            "interaction_block.*.refine_block_atom_graph.edge_nonlinear_update",
        ],
    }
    for suffix, extra_targets in w16_fine_hotspot_targets.items():
        for precision_name, precision_base in w16_probe_precisions.items():
            mode = f"{precision_name}_{suffix}"
            configs[mode] = {
                **base,
                "mode": mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **precision_base,
                        "targets": extra_targets,
                    },
                ],
            }

    w16_formal_pass_atom_node_targets = [
        *w16_fine_hotspot_targets["atom_attn_node_update"],
        *w16_fine_hotspot_targets["atom_refine_node_update"],
    ]
    w16_formal_pass_line_second_targets = {
        "b7_atom_node": [
            *mixed_sensitive_extra_targets["b7_core_gate"],
            *w16_formal_pass_atom_node_targets,
        ],
        "b6_atom_node": [
            *mixed_sensitive_extra_targets["b6_core_gate"],
            *w16_formal_pass_atom_node_targets,
        ],
        "b4_6_atom_node": [
            *mixed_sensitive_extra_targets["b4_6_core_gate"],
            *w16_formal_pass_atom_node_targets,
        ],
        "b6_b7_atom_node": [
            *mixed_sensitive_extra_targets["b6_core_gate"],
            *mixed_sensitive_extra_targets["b7_core_gate"],
            *w16_formal_pass_atom_node_targets,
        ],
        "b4_7_atom_node": [
            *mixed_sensitive_extra_targets["b4_6_core_gate"],
            *mixed_sensitive_extra_targets["b7_core_gate"],
            *w16_formal_pass_atom_node_targets,
        ],
    }
    for suffix, extra_targets in w16_formal_pass_line_second_targets.items():
        for precision_name, precision_base in w16_probe_precisions.items():
            mode = f"{precision_name}_combo_{suffix}"
            configs[mode] = {
                **base,
                "mode": mode,
                "target_specs": [
                    {
                        **triton_w8a8_static_base,
                        "targets": current_p8e_w8a8_targets,
                    },
                    {
                        **precision_base,
                        "targets": extra_targets,
                    },
                ],
            }

    fp8_p8e_targets = [
        *current_p8e_w8a8_targets,
    ]
    configs["fp8_p8e_equivalent"] = {
        **torch_fp8_static_base,
        "mode": "fp8_p8e_equivalent",
        "targets": fp8_p8e_targets,
    }
    for suffix, extra_targets in {
        "b7_gate": attn_line_edge_gate_second_block_7_targets,
        "b7_core": attn_line_edge_core_second_block_7_targets,
        "b7_core_gate": [
            *attn_line_edge_core_second_block_7_targets,
            *attn_line_edge_gate_second_block_7_targets,
        ],
        "b6_core_gate": [
            "interaction_block.6.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.6.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
    }.items():
        mode = f"fp8_p8e_plus_{suffix}"
        configs[mode] = {
            **torch_fp8_static_base,
            "mode": mode,
            "targets": [
                *fp8_p8e_targets,
                *extra_targets,
            ],
        }

    te_fp8_p8e_targets = [
        *current_p8e_w8a8_targets,
    ]
    configs["te_fp8_p8e_equivalent"] = {
        **te_fp8_base,
        "mode": "te_fp8_p8e_equivalent",
        "targets": te_fp8_p8e_targets,
    }
    for suffix, extra_targets in {
        "b7_gate": attn_line_edge_gate_second_block_7_targets,
        "b7_core": attn_line_edge_core_second_block_7_targets,
        "b7_core_gate": [
            *attn_line_edge_core_second_block_7_targets,
            *attn_line_edge_gate_second_block_7_targets,
        ],
        "b6_core_gate": [
            "interaction_block.6.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
            "interaction_block.6.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
        ],
    }.items():
        mode = f"te_fp8_p8e_plus_{suffix}"
        configs[mode] = {
            **te_fp8_base,
            "mode": mode,
            "targets": [
                *te_fp8_p8e_targets,
                *extra_targets,
            ],
        }

    if quant_mode not in configs:
        known_modes = ", ".join(["none", *sorted(configs)])
        raise ValueError(f"Unknown quant_mode: {quant_mode}. Known modes: {known_modes}")

    return configs[quant_mode]
