#include "../Opdefine.h"

#include <ATen/ops/linear.h>
#include <torch/extension.h>

#include <optional>
#include <vector>

namespace {

constexpr int64_t kDim = 128;

void check_float_cuda_2d(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2D");
}

int64_t numel_from_shape(const std::vector<int64_t> &shape) {
  int64_t numel = 1;
  for (const int64_t dim : shape) {
    TORCH_CHECK(dim >= 0, "w8a8 saved-pre backward driver: negative shape dim");
    numel *= dim;
  }
  return numel;
}

void check_group4(const std::vector<torch::Tensor> &tensors, const char *name) {
  TORCH_CHECK(tensors.size() == 4,
              "w8a8 saved-pre group4 backward driver: ", name, " must have 4 tensors");
}

double eps_at(const std::vector<double> &eps_values, size_t idx) {
  TORCH_CHECK(eps_values.size() == 1 || eps_values.size() == 4,
              "w8a8 saved-pre group4 backward driver: eps_values must have size 1 or 4");
  return eps_values.size() == 1 ? eps_values[0] : eps_values[idx];
}

}  // namespace

std::vector<torch::Tensor> w8a8_saved_pre_backward_driver(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps,
    std::vector<int64_t> core_shape,
    std::vector<int64_t> gate_shape,
    bool use_tail_bwd_v2,
    bool use_cublas_pair) {
  TORCH_CHECK(grad_out.is_cuda(), "w8a8 saved-pre backward driver: grad_out must be CUDA");
  check_float_cuda_2d(core_pre, "core_pre");
  check_float_cuda_2d(gate_pre, "gate_pre");
  check_float_cuda_2d(core_dq_weight_t, "core_dq_weight_t");
  check_float_cuda_2d(gate_dq_weight_t, "gate_dq_weight_t");
  TORCH_CHECK(core_pre.sizes() == gate_pre.sizes(),
              "w8a8 saved-pre backward driver: core/gate pre shapes mismatch");
  TORCH_CHECK(core_pre.size(1) == kDim,
              "w8a8 saved-pre backward driver: hidden dim must be 128");
  TORCH_CHECK(core_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 saved-pre backward driver: dq weights must be [128, 128]");
  TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                  gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
              "w8a8 saved-pre backward driver: norm tensors must be CUDA");
  TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "w8a8 saved-pre backward driver: norm tensors must be float32");
  TORCH_CHECK(core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "w8a8 saved-pre backward driver: norm tensors must be [128]");
  TORCH_CHECK(!core_shape.empty() && !gate_shape.empty(),
              "w8a8 saved-pre backward driver: output shapes must be non-empty");
  TORCH_CHECK(core_shape.back() == kDim && gate_shape.back() == kDim,
              "w8a8 saved-pre backward driver: output shapes must end in 128");
  TORCH_CHECK(numel_from_shape(core_shape) == core_pre.numel() &&
                  numel_from_shape(gate_shape) == gate_pre.numel(),
              "w8a8 saved-pre backward driver: output shapes do not match pre tensors");

  auto grad_out_fp32 = grad_out.scalar_type() == torch::kFloat32 ? grad_out : grad_out.to(torch::kFloat32);
  auto grad_out_2d = grad_out_fp32.reshape({-1, kDim}).contiguous();
  TORCH_CHECK(grad_out_2d.sizes() == core_pre.sizes(),
              "w8a8 saved-pre backward driver: grad_out shape mismatch");

  std::vector<torch::Tensor> grad_pre;
  if (use_tail_bwd_v2) {
    grad_pre = input_grad_only_gated_tail_backward_n128_v2(
        grad_out_2d,
        core_pre,
        gate_pre,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps);
  } else {
    grad_pre = input_grad_only_gated_tail_backward(
        grad_out_2d,
        core_pre,
        gate_pre,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps);
  }
  TORCH_CHECK(grad_pre.size() == 2, "w8a8 saved-pre backward driver: tail backward returned wrong arity");

  torch::Tensor grad_core_2d;
  torch::Tensor grad_gate_2d;
  if (use_cublas_pair) {
    auto grad_input = w8a8_dual_input_grad_matmul_n128_cublas_pair(
        grad_pre[0],
        grad_pre[1],
        core_dq_weight_t,
        gate_dq_weight_t);
    TORCH_CHECK(grad_input.size() == 2,
                "w8a8 saved-pre backward driver: cuBLAS pair returned wrong arity");
    grad_core_2d = grad_input[0];
    grad_gate_2d = grad_input[1];
  } else {
    grad_core_2d = at::linear(grad_pre[0], core_dq_weight_t, std::nullopt);
    grad_gate_2d = at::linear(grad_pre[1], gate_dq_weight_t, std::nullopt);
  }
  return {grad_core_2d.reshape(core_shape), grad_gate_2d.reshape(gate_shape)};
}

std::vector<torch::Tensor> w8a8_saved_pre_backward_group4_driver(
    const std::vector<torch::Tensor> &grad_outs,
    const std::vector<torch::Tensor> &core_pres,
    const std::vector<torch::Tensor> &gate_pres,
    const std::vector<torch::Tensor> &core_dq_weight_ts,
    const std::vector<torch::Tensor> &gate_dq_weight_ts,
    const std::vector<torch::Tensor> &core_norm_weights,
    const std::vector<torch::Tensor> &core_norm_biases,
    const std::vector<torch::Tensor> &gate_norm_weights,
    const std::vector<torch::Tensor> &gate_norm_biases,
    const std::vector<double> &eps_values,
    bool use_tail_bwd_v2,
    bool use_cublas_pair) {
  check_group4(grad_outs, "grad_outs");
  check_group4(core_pres, "core_pres");
  check_group4(gate_pres, "gate_pres");
  check_group4(core_dq_weight_ts, "core_dq_weight_ts");
  check_group4(gate_dq_weight_ts, "gate_dq_weight_ts");
  check_group4(core_norm_weights, "core_norm_weights");
  check_group4(core_norm_biases, "core_norm_biases");
  check_group4(gate_norm_weights, "gate_norm_weights");
  check_group4(gate_norm_biases, "gate_norm_biases");

  std::vector<torch::Tensor> outputs;
  outputs.reserve(8);
  for (size_t i = 0; i < 4; ++i) {
    TORCH_CHECK(grad_outs[i].is_cuda(),
                "w8a8 saved-pre group4 backward driver: grad_out must be CUDA");
    check_float_cuda_2d(core_pres[i], "core_pre");
    check_float_cuda_2d(gate_pres[i], "gate_pre");
    check_float_cuda_2d(core_dq_weight_ts[i], "core_dq_weight_t");
    check_float_cuda_2d(gate_dq_weight_ts[i], "gate_dq_weight_t");
    TORCH_CHECK(core_pres[i].sizes() == gate_pres[i].sizes(),
                "w8a8 saved-pre group4 backward driver: core/gate pre shapes mismatch");
    TORCH_CHECK(core_pres[i].size(1) == kDim,
                "w8a8 saved-pre group4 backward driver: hidden dim must be 128");
    TORCH_CHECK(core_dq_weight_ts[i].sizes() == torch::IntArrayRef({kDim, kDim}) &&
                    gate_dq_weight_ts[i].sizes() == torch::IntArrayRef({kDim, kDim}),
                "w8a8 saved-pre group4 backward driver: dq weights must be [128, 128]");
    TORCH_CHECK(core_norm_weights[i].is_cuda() && core_norm_biases[i].is_cuda() &&
                    gate_norm_weights[i].is_cuda() && gate_norm_biases[i].is_cuda(),
                "w8a8 saved-pre group4 backward driver: norm tensors must be CUDA");
    TORCH_CHECK(core_norm_weights[i].scalar_type() == torch::kFloat32 &&
                    core_norm_biases[i].scalar_type() == torch::kFloat32 &&
                    gate_norm_weights[i].scalar_type() == torch::kFloat32 &&
                    gate_norm_biases[i].scalar_type() == torch::kFloat32,
                "w8a8 saved-pre group4 backward driver: norm tensors must be float32");
    TORCH_CHECK(core_norm_weights[i].numel() == kDim && core_norm_biases[i].numel() == kDim &&
                    gate_norm_weights[i].numel() == kDim && gate_norm_biases[i].numel() == kDim,
                "w8a8 saved-pre group4 backward driver: norm tensors must be [128]");

    auto grad_out_fp32 =
        grad_outs[i].scalar_type() == torch::kFloat32 ? grad_outs[i] : grad_outs[i].to(torch::kFloat32);
    auto grad_out_2d = grad_out_fp32.reshape({-1, kDim}).contiguous();
    TORCH_CHECK(grad_out_2d.sizes() == core_pres[i].sizes(),
                "w8a8 saved-pre group4 backward driver: grad_out shape mismatch");

    std::vector<torch::Tensor> grad_pre;
    if (use_tail_bwd_v2) {
      grad_pre = input_grad_only_gated_tail_backward_n128_v2(
          grad_out_2d,
          core_pres[i],
          gate_pres[i],
          core_norm_weights[i],
          core_norm_biases[i],
          gate_norm_weights[i],
          gate_norm_biases[i],
          eps_at(eps_values, i));
    } else {
      grad_pre = input_grad_only_gated_tail_backward(
          grad_out_2d,
          core_pres[i],
          gate_pres[i],
          core_norm_weights[i],
          core_norm_biases[i],
          gate_norm_weights[i],
          gate_norm_biases[i],
          eps_at(eps_values, i));
    }
    TORCH_CHECK(grad_pre.size() == 2,
                "w8a8 saved-pre group4 backward driver: tail backward returned wrong arity");

    if (use_cublas_pair) {
      auto grad_input = w8a8_dual_input_grad_matmul_n128_cublas_pair(
          grad_pre[0],
          grad_pre[1],
          core_dq_weight_ts[i],
          gate_dq_weight_ts[i]);
      TORCH_CHECK(grad_input.size() == 2,
                  "w8a8 saved-pre group4 backward driver: cuBLAS pair returned wrong arity");
      outputs.push_back(grad_input[0]);
      outputs.push_back(grad_input[1]);
    } else {
      outputs.push_back(at::linear(grad_pre[0], core_dq_weight_ts[i], std::nullopt));
      outputs.push_back(at::linear(grad_pre[1], gate_dq_weight_ts[i], std::nullopt));
    }
  }
  return outputs;
}
