#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "type_shim.h"

namespace {

constexpr int kFeatureDim = 128;
constexpr int kBlockThreads = 256;

__global__ void directed2undirected_average_forward_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ segment,
    float* __restrict__ output,
    int rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * kFeatureDim;
  if (idx >= total) {
    return;
  }
  int row = idx / kFeatureDim;
  int dim = idx - row * kFeatureDim;
  int64_t out_row = segment[row];
  atomicAdd(output + out_row * kFeatureDim + dim, input[idx] * 0.5f);
}

__global__ void directed2undirected_average_backward_kernel(
    const float* __restrict__ grad_out,
    const int64_t* __restrict__ segment,
    float* __restrict__ grad_input,
    int rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * kFeatureDim;
  if (idx >= total) {
    return;
  }
  int row = idx / kFeatureDim;
  int dim = idx - row * kFeatureDim;
  int64_t out_row = segment[row];
  grad_input[idx] = grad_out[out_row * kFeatureDim + dim] * 0.5f;
}

}  // namespace

torch::Tensor directed2undirected_average_forward(const torch::Tensor &input,
                                                  const torch::Tensor &segment,
                                                  int64_t num_segment) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(segment.is_cuda(), "segment must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(segment.scalar_type() == torch::kInt64, "segment must be int64");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kFeatureDim, "input must have shape [N, 128]");
  TORCH_CHECK(segment.dim() == 1 && segment.size(0) == input.size(0), "segment must have shape [N]");

  auto output = torch::zeros({num_segment, kFeatureDim}, input.options());
  int rows = static_cast<int>(input.size(0));
  int total = rows * kFeatureDim;
  int blocks = (total + kBlockThreads - 1) / kBlockThreads;
  directed2undirected_average_forward_kernel<<<blocks, kBlockThreads>>>(
      input.data_ptr<float>(),
      segment.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor directed2undirected_average_backward(const torch::Tensor &grad_out,
                                                   const torch::Tensor &segment,
                                                   int64_t rows) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(segment.is_cuda(), "segment must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be float32");
  TORCH_CHECK(segment.scalar_type() == torch::kInt64, "segment must be int64");
  TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(1) == kFeatureDim, "grad_out must have shape [N, 128]");
  TORCH_CHECK(segment.dim() == 1 && segment.size(0) == rows, "segment must have shape [rows]");

  auto grad_input = torch::empty({rows, kFeatureDim}, grad_out.options());
  int total = static_cast<int>(rows) * kFeatureDim;
  int blocks = (total + kBlockThreads - 1) / kBlockThreads;
  directed2undirected_average_backward_kernel<<<blocks, kBlockThreads>>>(
      grad_out.data_ptr<float>(),
      segment.data_ptr<int64_t>(),
      grad_input.data_ptr<float>(),
      static_cast<int>(rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_input;
}
