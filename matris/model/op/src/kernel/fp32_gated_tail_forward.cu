#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float sigmoidf_stable_p77(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void fp32_gated_tail_forward_kernel(
    const float* __restrict__ core,
    const float* __restrict__ gate,
    const float* __restrict__ core_weight,
    const float* __restrict__ core_bias,
    const float* __restrict__ gate_weight,
    const float* __restrict__ gate_bias,
    float* __restrict__ out,
    int rows,
    int dim,
    int64_t core_stride0,
    int64_t core_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_sum[kThreads];
  __shared__ float s_gate_sum[kThreads];
  __shared__ float s_core_sumsq[kThreads];
  __shared__ float s_gate_sumsq[kThreads];

  int out_base = row * dim;
  int64_t core_base = static_cast<int64_t>(row) * core_stride0;
  int64_t gate_base = static_cast<int64_t>(row) * gate_stride0;
  float core_v = 0.0f;
  float gate_v = 0.0f;
  if (tid < dim) {
    core_v = core[core_base + static_cast<int64_t>(tid) * core_stride1];
    gate_v = gate[gate_base + static_cast<int64_t>(tid) * gate_stride1];
  }

  s_core_sum[tid] = tid < dim ? core_v : 0.0f;
  s_gate_sum[tid] = tid < dim ? gate_v : 0.0f;
  s_core_sumsq[tid] = tid < dim ? core_v * core_v : 0.0f;
  s_gate_sumsq[tid] = tid < dim ? gate_v * gate_v : 0.0f;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_core_sum[tid] += s_core_sum[tid + stride];
      s_gate_sum[tid] += s_gate_sum[tid + stride];
      s_core_sumsq[tid] += s_core_sumsq[tid + stride];
      s_gate_sumsq[tid] += s_gate_sumsq[tid + stride];
    }
    __syncthreads();
  }
  float inv_dim = 1.0f / static_cast<float>(dim);
  float core_mean = s_core_sum[0] * inv_dim;
  float gate_mean = s_gate_sum[0] * inv_dim;
  float core_var = fmaxf(s_core_sumsq[0] * inv_dim - core_mean * core_mean, 0.0f);
  float gate_var = fmaxf(s_gate_sumsq[0] * inv_dim - gate_mean * gate_mean, 0.0f);
  float core_centered = tid < dim ? core_v - core_mean : 0.0f;
  float gate_centered = tid < dim ? gate_v - gate_mean : 0.0f;
  float core_rstd = rsqrtf(core_var + eps);
  float gate_rstd = rsqrtf(gate_var + eps);

  if (tid < dim) {
    float core_ln = core_centered * core_rstd * core_weight[tid] + core_bias[tid];
    float gate_ln = gate_centered * gate_rstd * gate_weight[tid] + gate_bias[tid];
    float core_sig = sigmoidf_stable_p77(core_ln);
    float gate_act = sigmoidf_stable_p77(gate_ln);
    out[out_base + tid] = core_ln * core_sig * gate_act;
  }
}

}  // namespace

torch::Tensor fp32_gated_tail_forward_n128(const torch::Tensor &core,
                                           const torch::Tensor &gate,
                                           const torch::Tensor &core_norm_weight,
                                           const torch::Tensor &core_norm_bias,
                                           const torch::Tensor &gate_norm_weight,
                                           const torch::Tensor &gate_norm_bias,
                                           double eps) {
  TORCH_CHECK(core.is_cuda() && gate.is_cuda(), "fp32_gated_tail_forward_n128: tensors must be CUDA");
  TORCH_CHECK(core.scalar_type() == torch::kFloat32 && gate.scalar_type() == torch::kFloat32,
              "fp32_gated_tail_forward_n128: core/gate must be float32");
  TORCH_CHECK(core.dim() == 2 && gate.dim() == 2 && core.sizes() == gate.sizes(),
              "fp32_gated_tail_forward_n128: core/gate must have matching [rows, dim] shapes");
  int64_t rows64 = core.size(0);
  int64_t dim64 = core.size(1);
  TORCH_CHECK(dim64 == 128 || dim64 == 256,
              "fp32_gated_tail_forward_n128: only dim 128/256 is supported");
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "fp32_gated_tail_forward_n128: too many rows");

  auto check_vec = [dim64](const torch::Tensor &t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() == dim64, name, " must have dim elements");
  };
  check_vec(core_norm_weight, "core_norm_weight");
  check_vec(core_norm_bias, "core_norm_bias");
  check_vec(gate_norm_weight, "gate_norm_weight");
  check_vec(gate_norm_bias, "gate_norm_bias");

  auto out = torch::empty({rows64, dim64}, core.options());
  int rows = static_cast<int>(rows64);
  int dim = static_cast<int>(dim64);
  fp32_gated_tail_forward_kernel<<<rows, kThreads>>>(
      core.data_ptr<float>(),
      gate.data_ptr<float>(),
      core_norm_weight.data_ptr<float>(),
      core_norm_bias.data_ptr<float>(),
      gate_norm_weight.data_ptr<float>(),
      gate_norm_bias.data_ptr<float>(),
      out.data_ptr<float>(),
      rows,
      dim,
      core.stride(0),
      core.stride(1),
      gate.stride(0),
      gate.stride(1),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
