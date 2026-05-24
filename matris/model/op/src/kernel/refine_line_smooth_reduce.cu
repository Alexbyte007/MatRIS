#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kDim = 128;

__global__ void refine_line_smooth_reduce_forward_kernel(
    const float* __restrict__ nonlinear,
    const float* __restrict__ base_envelope,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ out,
    int64_t rows) {
  const int64_t row = blockIdx.x;
  const int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];
  const float bi = base_envelope[source * kDim + dim];
  const float bj = base_envelope[target * kDim + dim];
  const float value = nonlinear[row * kDim + dim] * bi * bj;
  atomicAdd(&out[target * kDim + dim], value);
}

__global__ void refine_line_smooth_reduce_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ nonlinear,
    const float* __restrict__ base_envelope,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_nonlinear,
    float* __restrict__ grad_base,
    int64_t rows) {
  const int64_t row = blockIdx.x;
  const int dim = threadIdx.x;
  if (row >= rows || dim >= kDim) {
    return;
  }
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];
  const float grad = grad_out[target * kDim + dim];
  const float nonlinear_v = nonlinear[row * kDim + dim];
  const float bi = base_envelope[source * kDim + dim];
  const float bj = base_envelope[target * kDim + dim];
  grad_nonlinear[row * kDim + dim] = grad * bi * bj;
  atomicAdd(&grad_base[source * kDim + dim], grad * nonlinear_v * bj);
  atomicAdd(&grad_base[target * kDim + dim], grad * nonlinear_v * bi);
}

}  // namespace

torch::Tensor refine_line_smooth_reduce_forward(
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t num_nodes) {
  TORCH_CHECK(nonlinear.is_cuda() && base_envelope.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "refine_line_smooth_reduce_forward: all tensors must be CUDA");
  TORCH_CHECK(nonlinear.scalar_type() == torch::kFloat32 && base_envelope.scalar_type() == torch::kFloat32,
              "refine_line_smooth_reduce_forward: feature tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "refine_line_smooth_reduce_forward: indices must be int64");
  TORCH_CHECK(nonlinear.dim() == 2 && nonlinear.size(1) == kDim,
              "refine_line_smooth_reduce_forward: nonlinear must be [rows, 128]");
  TORCH_CHECK(base_envelope.dim() == 2 && base_envelope.size(1) == kDim,
              "refine_line_smooth_reduce_forward: base_envelope must be [nodes, 128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == nonlinear.size(0) && target_index.size(0) == nonlinear.size(0),
              "refine_line_smooth_reduce_forward: index sizes must match rows");
  TORCH_CHECK(num_nodes >= 0, "refine_line_smooth_reduce_forward: num_nodes must be non-negative");

  auto nonlinear_c = nonlinear.contiguous();
  auto base_c = base_envelope.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto out = nonlinear_c.new_zeros({num_nodes, kDim});
  const int64_t rows = nonlinear_c.size(0);
  if (rows == 0) {
    return out;
  }
  refine_line_smooth_reduce_forward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      nonlinear_c.data_ptr<float>(),
      base_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> refine_line_smooth_reduce_backward(
    const torch::Tensor &grad_out,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index) {
  TORCH_CHECK(grad_out.is_cuda() && nonlinear.is_cuda() && base_envelope.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "refine_line_smooth_reduce_backward: all tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && nonlinear.scalar_type() == torch::kFloat32 &&
                  base_envelope.scalar_type() == torch::kFloat32,
              "refine_line_smooth_reduce_backward: feature tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "refine_line_smooth_reduce_backward: indices must be int64");
  TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(1) == kDim,
              "refine_line_smooth_reduce_backward: grad_out must be [nodes, 128]");
  TORCH_CHECK(nonlinear.dim() == 2 && nonlinear.size(1) == kDim,
              "refine_line_smooth_reduce_backward: nonlinear must be [rows, 128]");
  TORCH_CHECK(base_envelope.dim() == 2 && base_envelope.size(1) == kDim &&
                  base_envelope.size(0) == grad_out.size(0),
              "refine_line_smooth_reduce_backward: base_envelope must match grad_out");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == nonlinear.size(0) && target_index.size(0) == nonlinear.size(0),
              "refine_line_smooth_reduce_backward: index sizes must match rows");

  auto grad_out_c = grad_out.contiguous();
  auto nonlinear_c = nonlinear.contiguous();
  auto base_c = base_envelope.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto grad_nonlinear = nonlinear_c.new_empty(nonlinear_c.sizes());
  auto grad_base = base_c.new_zeros(base_c.sizes());
  const int64_t rows = nonlinear_c.size(0);
  if (rows == 0) {
    return {grad_nonlinear.zero_(), grad_base};
  }
  refine_line_smooth_reduce_backward_kernel<<<static_cast<unsigned int>(rows), kDim>>>(
      grad_out_c.data_ptr<float>(),
      nonlinear_c.data_ptr<float>(),
      base_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_nonlinear.data_ptr<float>(),
      grad_base.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_nonlinear, grad_base};
}
