#ifndef OP_SRC_OPDECLARE_H_
#define OP_SRC_OPDECLARE_H_

#include <torch/extension.h>


torch::Tensor fused_SiLU_Bwd(const torch::Tensor &dgrad, const torch::Tensor &input);

std::vector<torch::Tensor> fused_SiLU_Grad_Bwd(const torch::Tensor &grad_grad_input, const torch::Tensor &grad_output,
                                                const torch::Tensor &input);

torch::Tensor quant_linear_w8a32(const torch::Tensor &input,
                                 const torch::Tensor &q_weight,
                                 const torch::Tensor &scale,
                                 const torch::Tensor &bias,
                                 bool has_bias);

torch::Tensor quant_linear_w8a_lowp_forward(const torch::Tensor &input,
                                            const torch::Tensor &q_weight,
                                            const torch::Tensor &scale,
                                            const torch::Tensor &bias,
                                            bool has_bias);

torch::Tensor quant_linear_w8a_lowp_t_forward(const torch::Tensor &input,
                                              const torch::Tensor &q_weight_t,
                                              const torch::Tensor &scale,
                                              const torch::Tensor &bias,
                                              bool has_bias);

torch::Tensor quant_linear_w8a_lowp_wmma_t_forward(const torch::Tensor &input,
                                                   const torch::Tensor &q_weight_t,
                                                   const torch::Tensor &scale,
                                                   const torch::Tensor &bias,
                                                   bool has_bias);

torch::Tensor quant_linear_w8a_lowp_cutlass_mixed_forward(const torch::Tensor &input,
                                                          const torch::Tensor &q_weight,
                                                          const torch::Tensor &scale,
                                                          const torch::Tensor &bias,
                                                          bool has_bias);

torch::Tensor quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward(const torch::Tensor &input,
                                                                   const torch::Tensor &q_weight,
                                                                   const torch::Tensor &scale,
                                                                   const torch::Tensor &bias,
                                                                   bool has_bias);

torch::Tensor quant_linear_w8a_lowp_grad_input(const torch::Tensor &grad_output,
                                               const torch::Tensor &q_weight,
                                               const torch::Tensor &scale);

std::vector<torch::Tensor> quant_linear_w8a_lowp_dual_cached_forward(const torch::Tensor &core_input,
                                                                     const torch::Tensor &gate_input,
                                                                     const torch::Tensor &core_weight,
                                                                     const torch::Tensor &gate_weight,
                                                                     const torch::Tensor &core_bias,
                                                                     const torch::Tensor &gate_bias,
                                                                     bool core_has_bias,
                                                                     bool gate_has_bias);

torch::Tensor quant_linear_w8a_lowp_dual_cached_tail_forward(const torch::Tensor &core_input,
                                                             const torch::Tensor &gate_input,
                                                             const torch::Tensor &core_weight,
                                                             const torch::Tensor &gate_weight,
                                                             const torch::Tensor &core_bias,
                                                             const torch::Tensor &gate_bias,
                                                             bool core_has_bias,
                                                             bool gate_has_bias,
                                                             const torch::Tensor &core_norm_weight,
                                                             const torch::Tensor &core_norm_bias,
                                                             const torch::Tensor &gate_norm_weight,
                                                             const torch::Tensor &gate_norm_bias,
                                                             double eps);

std::vector<torch::Tensor> quant_linear_w8a_lowp_dual_cached_tail_forward_aux(const torch::Tensor &core_input,
                                                                              const torch::Tensor &gate_input,
                                                                              const torch::Tensor &core_weight,
                                                                              const torch::Tensor &gate_weight,
                                                                              const torch::Tensor &core_bias,
                                                                              const torch::Tensor &gate_bias,
                                                                              bool core_has_bias,
                                                                              bool gate_has_bias,
                                                                              const torch::Tensor &core_norm_weight,
                                                                              const torch::Tensor &core_norm_bias,
                                                                              const torch::Tensor &gate_norm_weight,
                                                                              const torch::Tensor &gate_norm_bias,
                                                                              double eps);

torch::Tensor quant_linear_w8a8_static_wmma(const torch::Tensor &input,
                                            const torch::Tensor &q_weight,
                                            const torch::Tensor &weight_scale,
                                            const torch::Tensor &activation_scale,
                                            const torch::Tensor &bias,
                                            bool has_bias);

torch::Tensor quant_linear_w8a8_static_grad_input(const torch::Tensor &grad_output,
                                                  const torch::Tensor &q_weight,
                                                  const torch::Tensor &weight_scale);

torch::Tensor quant_linear_w8a8_static_cutlass(const torch::Tensor &input,
                                               const torch::Tensor &q_weight,
                                               const torch::Tensor &weight_scale,
                                               const torch::Tensor &activation_scale,
                                               const torch::Tensor &bias,
                                               bool has_bias);

torch::Tensor quant_linear_w8a8_static_wmma_dual_gated_tail_n128(const torch::Tensor &core_input,
                                                                 const torch::Tensor &gate_input,
                                                                 const torch::Tensor &core_q_weight,
                                                                 const torch::Tensor &gate_q_weight,
                                                                 const torch::Tensor &core_weight_scale,
                                                                 const torch::Tensor &gate_weight_scale,
                                                                 const torch::Tensor &core_activation_scale,
                                                                 const torch::Tensor &gate_activation_scale,
                                                                 const torch::Tensor &core_bias,
                                                                 const torch::Tensor &gate_bias,
                                                                 bool core_has_bias,
                                                                 bool gate_has_bias,
                                                                 const torch::Tensor &core_norm_weight,
                                                                 const torch::Tensor &core_norm_bias,
                                                                 const torch::Tensor &gate_norm_weight,
                                                                 const torch::Tensor &gate_norm_bias,
                                                                 double eps);

std::vector<torch::Tensor> quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux(const torch::Tensor &core_input,
                                                                                  const torch::Tensor &gate_input,
                                                                                  const torch::Tensor &core_q_weight,
                                                                                  const torch::Tensor &gate_q_weight,
                                                                                  const torch::Tensor &core_weight_scale,
                                                                                  const torch::Tensor &gate_weight_scale,
                                                                                  const torch::Tensor &core_activation_scale,
                                                                                  const torch::Tensor &gate_activation_scale,
                                                                                  const torch::Tensor &core_bias,
                                                                                  const torch::Tensor &gate_bias,
                                                                                  bool core_has_bias,
                                                                                  bool gate_has_bias,
                                                                                  const torch::Tensor &core_norm_weight,
                                                                                  const torch::Tensor &core_norm_bias,
                                                                                  const torch::Tensor &gate_norm_weight,
                                                                                  const torch::Tensor &gate_norm_bias,
                                                                                  double eps);

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_dual(const torch::Tensor &core_input,
                                                                 const torch::Tensor &gate_input,
                                                                 const torch::Tensor &core_q_weight,
                                                                 const torch::Tensor &gate_q_weight,
                                                                 const torch::Tensor &core_weight_scale,
                                                                 const torch::Tensor &gate_weight_scale,
                                                                 const torch::Tensor &core_activation_scale,
                                                                 const torch::Tensor &gate_activation_scale,
                                                                 const torch::Tensor &core_bias,
                                                                 const torch::Tensor &gate_bias,
                                                                 bool core_has_bias,
                                                                 bool gate_has_bias);

torch::Tensor quant_linear_w8a8_static_cutlass_dual_gated_tail(const torch::Tensor &core_input,
                                                               const torch::Tensor &gate_input,
                                                               const torch::Tensor &core_q_weight,
                                                               const torch::Tensor &gate_q_weight,
                                                               const torch::Tensor &core_weight_scale,
                                                               const torch::Tensor &gate_weight_scale,
                                                               const torch::Tensor &core_activation_scale,
                                                               const torch::Tensor &gate_activation_scale,
                                                               const torch::Tensor &core_bias,
                                                               const torch::Tensor &gate_bias,
                                                               bool core_has_bias,
                                                               bool gate_has_bias,
                                                               const torch::Tensor &core_norm_weight,
                                                               const torch::Tensor &core_norm_bias,
                                                               const torch::Tensor &gate_norm_weight,
                                                               const torch::Tensor &gate_norm_bias,
                                                               double eps);

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_grouped_dual(const torch::Tensor &core_input,
                                                                         const torch::Tensor &gate_input,
                                                                         const torch::Tensor &core_q_weight,
                                                                         const torch::Tensor &gate_q_weight,
                                                                         const torch::Tensor &core_weight_scale,
                                                                         const torch::Tensor &gate_weight_scale,
                                                                         const torch::Tensor &core_activation_scale,
                                                                         const torch::Tensor &gate_activation_scale,
                                                                         const torch::Tensor &core_bias,
                                                                         const torch::Tensor &gate_bias,
                                                                         bool core_has_bias,
                                                                         bool gate_has_bias);

std::vector<torch::Tensor> quant_linear_w8a8_static_wmma_dual(const torch::Tensor &core_input,
                                                              const torch::Tensor &gate_input,
                                                              const torch::Tensor &core_q_weight,
                                                              const torch::Tensor &gate_q_weight,
                                                              const torch::Tensor &core_weight_scale,
                                                              const torch::Tensor &gate_weight_scale,
                                                              const torch::Tensor &core_activation_scale,
                                                              const torch::Tensor &gate_activation_scale,
                                                              const torch::Tensor &core_bias,
                                                              const torch::Tensor &gate_bias,
                                                              bool core_has_bias,
                                                              bool gate_has_bias);

std::vector<torch::Tensor> target_attention_sum_forward(const torch::Tensor &logits,
                                                        const torch::Tensor &values,
                                                        const torch::Tensor &lengths);

std::vector<torch::Tensor> target_attention_sum_backward(const torch::Tensor &grad_out,
                                                         const torch::Tensor &values,
                                                         const torch::Tensor &out,
                                                         const torch::Tensor &alpha,
                                                         const torch::Tensor &lengths);

torch::Tensor directed2undirected_average_forward(const torch::Tensor &input,
                                                  const torch::Tensor &segment,
                                                  int64_t num_segment);

torch::Tensor directed2undirected_average_backward(const torch::Tensor &grad_out,
                                                   const torch::Tensor &segment,
                                                   int64_t rows);

torch::Tensor line_refine_smooth_agg_forward(const torch::Tensor &x,
                                             const torch::Tensor &smooth,
                                             const torch::Tensor &target,
                                             const torch::Tensor &bin_count,
                                             int64_t num_segment);

std::vector<torch::Tensor> line_refine_smooth_agg_backward(const torch::Tensor &grad_out,
                                                           const torch::Tensor &x,
                                                           const torch::Tensor &smooth,
                                                           const torch::Tensor &target);

torch::Tensor graph_feature_construction_forward(const torch::Tensor &edge_feat,
                                                 const torch::Tensor &node_feat,
                                                 const torch::Tensor &target,
                                                 const torch::Tensor &source);

std::vector<torch::Tensor> graph_feature_construction_backward(const torch::Tensor &grad_out,
                                                               const torch::Tensor &target,
                                                               const torch::Tensor &source,
                                                               int64_t num_nodes);

torch::Tensor edge_vectors_forward(const torch::Tensor &coords,
                                   const torch::Tensor &lattice,
                                   const torch::Tensor &image,
                                   const torch::Tensor &target,
                                   const torch::Tensor &source);

std::vector<torch::Tensor> edge_vectors_backward(const torch::Tensor &grad_edge,
                                                 const torch::Tensor &image,
                                                 const torch::Tensor &target,
                                                 const torch::Tensor &source,
                                                 int64_t num_coords,
                                                 int64_t lattice_rows);

std::vector<torch::Tensor> fused_line_attention_forward(const torch::Tensor &source_logits,
                                                        const torch::Tensor &target_logits,
                                                        const torch::Tensor &values,
                                                        const torch::Tensor &source_index,
                                                        const torch::Tensor &target_index,
                                                        int64_t num_segments);

std::vector<torch::Tensor> fused_line_attention_backward(const torch::Tensor &grad_source_out,
                                                         const torch::Tensor &grad_target_out,
                                                         const torch::Tensor &values,
                                                         const torch::Tensor &source_out,
                                                         const torch::Tensor &target_out,
                                                         const torch::Tensor &source_alpha,
                                                         const torch::Tensor &target_alpha,
                                                         const torch::Tensor &source_index,
                                                         const torch::Tensor &target_index);

#endif  // OP_SRC_OPDECLARE_H_
