#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kDim = 128;

__device__ __forceinline__ float sigmoidf_stable_p29(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void two_linear_silu_input_grad_backward_n128_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight2,
    const float* __restrict__ hidden,
    const float* __restrict__ weight1,
    float* __restrict__ grad_x,
    int64_t rows) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows || tid >= kDim) {
    return;
  }

  __shared__ float grad_hidden[kDim];
  const int64_t base = static_cast<int64_t>(row) * kDim;

  float grad_act = 0.0f;
  for (int o = 0; o < kDim; ++o) {
    grad_act += grad_out[base + o] * weight2[o * kDim + tid];
  }

  const float h = hidden[base + tid];
  const float sig = sigmoidf_stable_p29(h);
  const float silu_grad = sig * (1.0f + h * (1.0f - sig));
  grad_hidden[tid] = grad_act * silu_grad;
  __syncthreads();

  float gx = 0.0f;
  for (int j = 0; j < kDim; ++j) {
    gx += grad_hidden[j] * weight1[j * kDim + tid];
  }
  grad_x[base + tid] = gx;
}

}  // namespace

torch::Tensor two_linear_silu_input_grad_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &weight2,
    const torch::Tensor &hidden,
    const torch::Tensor &weight1) {
  TORCH_CHECK(grad_out.is_cuda() && weight2.is_cuda() && hidden.is_cuda() && weight1.is_cuda(),
              "two_linear_silu_input_grad_backward_n128: all tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                  weight2.scalar_type() == torch::kFloat32 &&
                  hidden.scalar_type() == torch::kFloat32 &&
                  weight1.scalar_type() == torch::kFloat32,
              "two_linear_silu_input_grad_backward_n128: all tensors must be float32");
  TORCH_CHECK(grad_out.is_contiguous() && weight2.is_contiguous() &&
                  hidden.is_contiguous() && weight1.is_contiguous(),
              "two_linear_silu_input_grad_backward_n128: all tensors must be contiguous");
  TORCH_CHECK(grad_out.dim() == 2 && hidden.dim() == 2 && weight2.dim() == 2 && weight1.dim() == 2,
              "two_linear_silu_input_grad_backward_n128: expected 2D tensors");
  TORCH_CHECK(grad_out.sizes() == hidden.sizes(),
              "two_linear_silu_input_grad_backward_n128: grad_out and hidden shapes must match");
  TORCH_CHECK(grad_out.size(1) == kDim &&
                  weight2.size(0) == kDim && weight2.size(1) == kDim &&
                  weight1.size(0) == kDim && weight1.size(1) == kDim,
              "two_linear_silu_input_grad_backward_n128: specialized for 128x128 MLPs");

  auto grad_x = torch::empty_like(hidden);
  const int64_t rows = hidden.size(0);
  if (rows == 0) {
    return grad_x;
  }

  two_linear_silu_input_grad_backward_n128_kernel<<<rows, kDim>>>(
      grad_out.data_ptr<float>(),
      weight2.data_ptr<float>(),
      hidden.data_ptr<float>(),
      weight1.data_ptr<float>(),
      grad_x.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_x;
}
