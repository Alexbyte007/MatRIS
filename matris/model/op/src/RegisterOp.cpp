#include "Opdefine.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("fuse_silu_bwd", &fused_SiLU_Bwd, "fuse_silu_bwd");
    m.def("fuse_silu_grad_bwd", &fused_SiLU_Grad_Bwd, "fuse_silu_grad_bwd"); 
    m.def("quant_linear_w8a32", &quant_linear_w8a32, "quant_linear_w8a32");
    m.def("quant_linear_w8a_lowp_forward", &quant_linear_w8a_lowp_forward, "quant_linear_w8a_lowp_forward");
    m.def("quant_linear_w8a_lowp_t_forward", &quant_linear_w8a_lowp_t_forward, "quant_linear_w8a_lowp_t_forward");
    m.def("quant_linear_w8a_lowp_wmma_t_forward", &quant_linear_w8a_lowp_wmma_t_forward, "quant_linear_w8a_lowp_wmma_t_forward");
    m.def("quant_linear_w8a_lowp_cutlass_mixed_forward", &quant_linear_w8a_lowp_cutlass_mixed_forward, "quant_linear_w8a_lowp_cutlass_mixed_forward");
    m.def("quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward", &quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward, "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward");
    m.def("quant_linear_w8a_lowp_grad_input", &quant_linear_w8a_lowp_grad_input, "quant_linear_w8a_lowp_grad_input");
    m.def("quant_linear_w8a_lowp_dual_cached_forward", &quant_linear_w8a_lowp_dual_cached_forward, "quant_linear_w8a_lowp_dual_cached_forward");
    m.def("quant_linear_w8a_lowp_dual_cached_tail_forward", &quant_linear_w8a_lowp_dual_cached_tail_forward, "quant_linear_w8a_lowp_dual_cached_tail_forward");
    m.def("quant_linear_w8a_lowp_dual_cached_tail_forward_aux", &quant_linear_w8a_lowp_dual_cached_tail_forward_aux, "quant_linear_w8a_lowp_dual_cached_tail_forward_aux");
    m.def("quant_linear_w8a8_static_wmma", &quant_linear_w8a8_static_wmma, "quant_linear_w8a8_static_wmma");
    m.def("quant_linear_w8a8_static_grad_input", &quant_linear_w8a8_static_grad_input, "quant_linear_w8a8_static_grad_input");
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128");
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux");
    m.def("quant_linear_w8a8_static_cutlass", &quant_linear_w8a8_static_cutlass, "quant_linear_w8a8_static_cutlass");
    m.def("quant_linear_w8a8_static_cutlass_dual", &quant_linear_w8a8_static_cutlass_dual, "quant_linear_w8a8_static_cutlass_dual");
    m.def("quant_linear_w8a8_static_cutlass_dual_gated_tail", &quant_linear_w8a8_static_cutlass_dual_gated_tail, "quant_linear_w8a8_static_cutlass_dual_gated_tail");
    m.def("quant_linear_w8a8_static_cutlass_grouped_dual", &quant_linear_w8a8_static_cutlass_grouped_dual, "quant_linear_w8a8_static_cutlass_grouped_dual");
    m.def("quant_linear_w8a8_static_wmma_dual", &quant_linear_w8a8_static_wmma_dual, "quant_linear_w8a8_static_wmma_dual");
    m.def("target_attention_sum_forward", &target_attention_sum_forward, "target_attention_sum_forward");
    m.def("target_attention_sum_backward", &target_attention_sum_backward, "target_attention_sum_backward");
    m.def("directed2undirected_average_forward", &directed2undirected_average_forward, "directed2undirected_average_forward");
    m.def("directed2undirected_average_backward", &directed2undirected_average_backward, "directed2undirected_average_backward");
    m.def("line_refine_smooth_agg_forward", &line_refine_smooth_agg_forward, "line_refine_smooth_agg_forward");
    m.def("line_refine_smooth_agg_backward", &line_refine_smooth_agg_backward, "line_refine_smooth_agg_backward");
    m.def("graph_feature_construction_forward", &graph_feature_construction_forward, "graph_feature_construction_forward");
    m.def("graph_feature_construction_backward", &graph_feature_construction_backward, "graph_feature_construction_backward");
    m.def("edge_vectors_forward", &edge_vectors_forward, "edge_vectors_forward");
    m.def("edge_vectors_backward", &edge_vectors_backward, "edge_vectors_backward");
    m.def("fused_line_attention_forward", &fused_line_attention_forward, "fused_line_attention_forward");
    m.def("fused_line_attention_backward", &fused_line_attention_backward, "fused_line_attention_backward");
}

TORCH_LIBRARY(matris_op, m)
{
    m.def("fuse_silu_bwd", &fused_SiLU_Bwd);
    m.def("fuse_silu_grad_bwd", &fused_SiLU_Grad_Bwd);  
    m.def("quant_linear_w8a32", &quant_linear_w8a32);
    m.def("quant_linear_w8a_lowp_forward", &quant_linear_w8a_lowp_forward);
    m.def("quant_linear_w8a_lowp_t_forward", &quant_linear_w8a_lowp_t_forward);
    m.def("quant_linear_w8a_lowp_wmma_t_forward", &quant_linear_w8a_lowp_wmma_t_forward);
    m.def("quant_linear_w8a_lowp_cutlass_mixed_forward", &quant_linear_w8a_lowp_cutlass_mixed_forward);
    m.def("quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward", &quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward);
    m.def("quant_linear_w8a_lowp_grad_input", &quant_linear_w8a_lowp_grad_input);
    m.def("quant_linear_w8a_lowp_dual_cached_forward", &quant_linear_w8a_lowp_dual_cached_forward);
    m.def("quant_linear_w8a_lowp_dual_cached_tail_forward", &quant_linear_w8a_lowp_dual_cached_tail_forward);
    m.def("quant_linear_w8a_lowp_dual_cached_tail_forward_aux", &quant_linear_w8a_lowp_dual_cached_tail_forward_aux);
    m.def("quant_linear_w8a8_static_wmma", &quant_linear_w8a8_static_wmma);
    m.def("quant_linear_w8a8_static_grad_input", &quant_linear_w8a8_static_grad_input);
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128);
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux);
    m.def("quant_linear_w8a8_static_cutlass", &quant_linear_w8a8_static_cutlass);
    m.def("quant_linear_w8a8_static_cutlass_dual", &quant_linear_w8a8_static_cutlass_dual);
    m.def("quant_linear_w8a8_static_cutlass_dual_gated_tail", &quant_linear_w8a8_static_cutlass_dual_gated_tail);
    m.def("quant_linear_w8a8_static_cutlass_grouped_dual", &quant_linear_w8a8_static_cutlass_grouped_dual);
    m.def("quant_linear_w8a8_static_wmma_dual", &quant_linear_w8a8_static_wmma_dual);
    m.def("target_attention_sum_forward", &target_attention_sum_forward);
    m.def("target_attention_sum_backward", &target_attention_sum_backward);
    m.def("directed2undirected_average_forward", &directed2undirected_average_forward);
    m.def("directed2undirected_average_backward", &directed2undirected_average_backward);
    m.def("line_refine_smooth_agg_forward", &line_refine_smooth_agg_forward);
    m.def("line_refine_smooth_agg_backward", &line_refine_smooth_agg_backward);
    m.def("graph_feature_construction_forward", &graph_feature_construction_forward);
    m.def("graph_feature_construction_backward", &graph_feature_construction_backward);
    m.def("edge_vectors_forward", &edge_vectors_forward);
    m.def("edge_vectors_backward", &edge_vectors_backward);
    m.def("fused_line_attention_forward", &fused_line_attention_forward);
    m.def("fused_line_attention_backward", &fused_line_attention_backward);
}
