#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <limits>

namespace {

constexpr int kDim = 128;
constexpr int kThreads = 256;

__device__ float atomic_max_float(float* addr, float value) {
  int* addr_as_i = reinterpret_cast<int*>(addr);
  int old = *addr_as_i;
  while (__int_as_float(old) < value) {
    int assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
  return __int_as_float(old);
}

__global__ void fused_line_attention_max_kernel(
    const float* __restrict__ source_logits,
    const float* __restrict__ target_logits,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ source_max,
    float* __restrict__ target_max,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  atomic_max_float(source_max + s * kDim + dim, source_logits[idx]);
  atomic_max_float(target_max + t * kDim + dim, target_logits[idx]);
}

__global__ void fused_line_attention_exp_sum_kernel(
    const float* __restrict__ source_logits,
    const float* __restrict__ target_logits,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ source_max,
    const float* __restrict__ target_max,
    float* __restrict__ source_alpha,
    float* __restrict__ target_alpha,
    float* __restrict__ source_sum,
    float* __restrict__ target_sum,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float se = expf(source_logits[idx] - source_max[s * kDim + dim]);
  float te = expf(target_logits[idx] - target_max[t * kDim + dim]);
  source_alpha[idx] = se;
  target_alpha[idx] = te;
  atomicAdd(source_sum + s * kDim + dim, se);
  atomicAdd(target_sum + t * kDim + dim, te);
}

__global__ void fused_line_attention_norm_out_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ source_sum,
    const float* __restrict__ target_sum,
    float* __restrict__ source_alpha,
    float* __restrict__ target_alpha,
    float* __restrict__ source_out,
    float* __restrict__ target_out,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float sa = source_alpha[idx] / source_sum[s * kDim + dim];
  float ta = target_alpha[idx] / target_sum[t * kDim + dim];
  source_alpha[idx] = sa;
  target_alpha[idx] = ta;
  float v = values[idx];
  atomicAdd(source_out + s * kDim + dim, sa * v);
  atomicAdd(target_out + t * kDim + dim, ta * v);
}

__global__ void fused_line_attention_backward_kernel(
    const float* __restrict__ grad_source_out,
    const float* __restrict__ grad_target_out,
    const float* __restrict__ values,
    const float* __restrict__ source_out,
    const float* __restrict__ target_out,
    const float* __restrict__ source_alpha,
    const float* __restrict__ target_alpha,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_source_logits,
    float* __restrict__ grad_target_logits,
    float* __restrict__ grad_values,
    int64_t rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * kDim;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / kDim;
  int dim = idx - row * kDim;
  int64_t s = source_index[row];
  int64_t t = target_index[row];
  float v = values[idx];
  float sa = source_alpha[idx];
  float ta = target_alpha[idx];
  float gs = grad_source_out[s * kDim + dim];
  float gt = grad_target_out[t * kDim + dim];
  float os = source_out[s * kDim + dim];
  float ot = target_out[t * kDim + dim];
  grad_source_logits[idx] = sa * gs * (v - os);
  grad_target_logits[idx] = ta * gt * (v - ot);
  grad_values[idx] = sa * gs + ta * gt;
}

}  // namespace

std::vector<torch::Tensor> fused_line_attention_forward(const torch::Tensor &source_logits,
                                                        const torch::Tensor &target_logits,
                                                        const torch::Tensor &values,
                                                        const torch::Tensor &source_index,
                                                        const torch::Tensor &target_index,
                                                        int64_t num_segments) {
  TORCH_CHECK(source_logits.is_cuda() && target_logits.is_cuda() && values.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda(),
              "fused_line_attention_forward: tensors must be CUDA");
  TORCH_CHECK(source_logits.scalar_type() == torch::kFloat32 &&
                  target_logits.scalar_type() == torch::kFloat32 &&
                  values.scalar_type() == torch::kFloat32,
              "fused_line_attention_forward: float tensors must be float32");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "fused_line_attention_forward: indices must be int64");
  TORCH_CHECK(source_logits.dim() == 2 && source_logits.size(1) == kDim,
              "fused_line_attention_forward: source_logits must be [E, 128]");
  TORCH_CHECK(source_logits.sizes() == target_logits.sizes() && source_logits.sizes() == values.sizes(),
              "fused_line_attention_forward: tensor shape mismatch");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == source_logits.size(0) &&
                  target_index.size(0) == source_logits.size(0),
              "fused_line_attention_forward: index shape mismatch");

  auto source_logits_c = source_logits.contiguous();
  auto target_logits_c = target_logits.contiguous();
  auto values_c = values.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  auto opts = source_logits.options();
  auto source_max = torch::full({num_segments, kDim}, -std::numeric_limits<float>::infinity(), opts);
  auto target_max = torch::full({num_segments, kDim}, -std::numeric_limits<float>::infinity(), opts);
  auto source_sum = torch::zeros({num_segments, kDim}, opts);
  auto target_sum = torch::zeros({num_segments, kDim}, opts);
  auto source_out = torch::zeros({num_segments, kDim}, opts);
  auto target_out = torch::zeros({num_segments, kDim}, opts);
  auto source_alpha = torch::empty_like(source_logits_c);
  auto target_alpha = torch::empty_like(target_logits_c);

  int64_t rows = source_logits_c.size(0);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_max_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      rows);
  fused_line_attention_exp_sum_kernel<<<blocks, kThreads>>>(
      source_logits_c.data_ptr<float>(),
      target_logits_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_max.data_ptr<float>(),
      target_max.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      rows);
  fused_line_attention_norm_out_kernel<<<blocks, kThreads>>>(
      values_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      source_sum.data_ptr<float>(),
      target_sum.data_ptr<float>(),
      source_alpha.data_ptr<float>(),
      target_alpha.data_ptr<float>(),
      source_out.data_ptr<float>(),
      target_out.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {source_out, target_out, source_alpha, target_alpha};
}

std::vector<torch::Tensor> fused_line_attention_backward(const torch::Tensor &grad_source_out,
                                                         const torch::Tensor &grad_target_out,
                                                         const torch::Tensor &values,
                                                         const torch::Tensor &source_out,
                                                         const torch::Tensor &target_out,
                                                         const torch::Tensor &source_alpha,
                                                         const torch::Tensor &target_alpha,
                                                         const torch::Tensor &source_index,
                                                         const torch::Tensor &target_index) {
  TORCH_CHECK(grad_source_out.is_cuda() && grad_target_out.is_cuda() && values.is_cuda() &&
                  source_out.is_cuda() && target_out.is_cuda() && source_alpha.is_cuda() &&
                  target_alpha.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "fused_line_attention_backward: tensors must be CUDA");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 && grad_source_out.scalar_type() == torch::kFloat32 &&
                  grad_target_out.scalar_type() == torch::kFloat32,
              "fused_line_attention_backward: float tensors must be float32");
  auto grad_source_logits = torch::empty_like(values);
  auto grad_target_logits = torch::empty_like(values);
  auto grad_values = torch::empty_like(values);
  auto grad_source_out_c = grad_source_out.contiguous();
  auto grad_target_out_c = grad_target_out.contiguous();
  auto values_c = values.contiguous();
  auto source_out_c = source_out.contiguous();
  auto target_out_c = target_out.contiguous();
  auto source_alpha_c = source_alpha.contiguous();
  auto target_alpha_c = target_alpha.contiguous();
  auto source_index_c = source_index.contiguous();
  auto target_index_c = target_index.contiguous();
  int64_t rows = values_c.size(0);
  int blocks = static_cast<int>((rows * kDim + kThreads - 1) / kThreads);
  fused_line_attention_backward_kernel<<<blocks, kThreads>>>(
      grad_source_out_c.data_ptr<float>(),
      grad_target_out_c.data_ptr<float>(),
      values_c.data_ptr<float>(),
      source_out_c.data_ptr<float>(),
      target_out_c.data_ptr<float>(),
      source_alpha_c.data_ptr<float>(),
      target_alpha_c.data_ptr<float>(),
      source_index_c.data_ptr<int64_t>(),
      target_index_c.data_ptr<int64_t>(),
      grad_source_logits.data_ptr<float>(),
      grad_target_logits.data_ptr<float>(),
      grad_values.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_source_logits, grad_target_logits, grad_values};
}
