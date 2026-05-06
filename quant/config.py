from __future__ import annotations


P8E_STABLE_MODE = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"


def get_quant_config(quant_mode: str | None) -> dict | None:
    if quant_mode in (None, "", "none"):
        return None

    refine_line_edge_second_targets = [
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_core.layers.3",
        "interaction_block.*.refine_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3",
    ]
    attn_line_edge_core_second_blocks_8_9_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_core.layers.3"
        for idx in range(8, 10)
    ]
    attn_line_edge_gate_second_blocks_8_9_targets = [
        f"interaction_block.{idx}.attn_block_line_graph.edge_nonlinear_update.mlp_gate.layers.3"
        for idx in range(8, 10)
    ]

    configs = {
        P8E_STABLE_MODE: {
            "mode": P8E_STABLE_MODE,
            "weight_bits": 8,
            "activation_bits": 8,
            "activation_dtype": "int8_static_per_tensor",
            "activation_scale_granularity": "static_per_tensor",
            "accumulation_dtype": "fp32",
            "output_dtype": "fp32",
            "scale_granularity": "per_channel",
            "fake_quant": False,
            "kernel": "triton_w8a8_static",
            "weight_packing": "int8_per_channel",
            "backward": "torch_dequantized_grad_x",
            "targets": [
                *refine_line_edge_second_targets,
                *attn_line_edge_core_second_blocks_8_9_targets,
                *attn_line_edge_gate_second_blocks_8_9_targets,
            ],
        },
    }

    if quant_mode not in configs:
        known_modes = ", ".join(sorted(configs))
        raise ValueError(f"Unknown quant_mode: {quant_mode}. Known stable modes: {known_modes}")

    return configs[quant_mode]
