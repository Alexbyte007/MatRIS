#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <array>
#include <vector>

namespace {

constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kThreadsDq = 128;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;
constexpr int kTileM32 = 32;
constexpr int kTileN8 = 8;

__device__ __forceinline__ float sigmoidf_stable_p30(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core_pre,
    const float* __restrict__ gate_pre,
    const int8_t* __restrict__ core_q_weight,
    const int8_t* __restrict__ gate_q_weight,
    const float* __restrict__ core_weight_scale,
    const float* __restrict__ gate_weight_scale,
    const float* __restrict__ core_norm_weight,
    const float* __restrict__ core_norm_bias,
    const float* __restrict__ gate_norm_weight,
    const float* __restrict__ gate_norm_bias,
    float* __restrict__ grad_core_input,
    float* __restrict__ grad_gate_input,
    int64_t rows,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_grad_pre[kDim];
  __shared__ float s_gate_grad_pre[kDim];
  __shared__ float s_reduce_core[kThreads];
  __shared__ float s_reduce_gate[kThreads];

  const int64_t base = static_cast<int64_t>(row) * kDim;
  float core_v = 0.0f;
  float gate_v = 0.0f;
  if (tid < kDim) {
    core_v = core_pre[base + tid];
    gate_v = gate_pre[base + tid];
  }
  __syncthreads();

  s_reduce_core[tid] = tid < kDim ? core_v : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? gate_v : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_mean = s_reduce_core[0] * (1.0f / static_cast<float>(kDim));
  const float gate_mean = s_reduce_gate[0] * (1.0f / static_cast<float>(kDim));
  __syncthreads();

  const float core_centered = tid < kDim ? core_v - core_mean : 0.0f;
  const float gate_centered = tid < kDim ? gate_v - gate_mean : 0.0f;
  s_reduce_core[tid] = tid < kDim ? core_centered * core_centered : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? gate_centered * gate_centered : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_rstd = rsqrtf(s_reduce_core[0] * (1.0f / static_cast<float>(kDim)) + eps);
  const float gate_rstd = rsqrtf(s_reduce_gate[0] * (1.0f / static_cast<float>(kDim)) + eps);
  __syncthreads();

  const float core_xhat = core_centered * core_rstd;
  const float gate_xhat = gate_centered * gate_rstd;
  float grad_core_norm = 0.0f;
  float grad_gate_norm = 0.0f;
  if (tid < kDim) {
    const float core_ln = core_xhat * core_norm_weight[tid] + core_norm_bias[tid];
    const float gate_ln = gate_xhat * gate_norm_weight[tid] + gate_norm_bias[tid];
    const float core_sig = sigmoidf_stable_p30(core_ln);
    const float core_act = core_ln * core_sig;
    const float gate_act = sigmoidf_stable_p30(gate_ln);
    const float grad = grad_out[base + tid];
    const float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    const float grad_core_ln = grad * gate_act * core_silu_grad;
    const float grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
    grad_core_norm = grad_core_ln * core_norm_weight[tid];
    grad_gate_norm = grad_gate_ln * gate_norm_weight[tid];
  }

  s_reduce_core[tid] = tid < kDim ? grad_core_norm : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_sum_grad = s_reduce_core[0];
  const float gate_sum_grad = s_reduce_gate[0];
  __syncthreads();

  s_reduce_core[tid] = tid < kDim ? grad_core_norm * core_xhat : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm * gate_xhat : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_sum_grad_xhat = s_reduce_core[0];
  const float gate_sum_grad_xhat = s_reduce_gate[0];

  if (tid < kDim) {
    const float inv_dim = 1.0f / static_cast<float>(kDim);
    s_core_grad_pre[tid] =
        (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    s_gate_grad_pre[tid] =
        (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
  }
  __syncthreads();

  if (tid < kDim) {
    float core_grad = 0.0f;
    float gate_grad = 0.0f;
    for (int j = 0; j < kDim; ++j) {
      core_grad += s_core_grad_pre[j] *
                   static_cast<float>(core_q_weight[j * kDim + tid]) * core_weight_scale[j];
      gate_grad += s_gate_grad_pre[j] *
                   static_cast<float>(gate_q_weight[j * kDim + tid]) * gate_weight_scale[j];
    }
    grad_core_input[base + tid] = core_grad;
    grad_gate_input[base + tid] = gate_grad;
  }
}

__global__ void w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core_pre,
    const float* __restrict__ gate_pre,
    const float* __restrict__ core_dq_weight_t,
    const float* __restrict__ gate_dq_weight_t,
    const float* __restrict__ core_norm_weight,
    const float* __restrict__ core_norm_bias,
    const float* __restrict__ gate_norm_weight,
    const float* __restrict__ gate_norm_bias,
    float* __restrict__ grad_core_input,
    float* __restrict__ grad_gate_input,
    int64_t rows,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_grad_pre[kDim];
  __shared__ float s_gate_grad_pre[kDim];
  __shared__ float s_reduce_core[kThreads];
  __shared__ float s_reduce_gate[kThreads];

  const int64_t base = static_cast<int64_t>(row) * kDim;
  float core_v = 0.0f;
  float gate_v = 0.0f;
  if (tid < kDim) {
    core_v = core_pre[base + tid];
    gate_v = gate_pre[base + tid];
  }
  __syncthreads();

  s_reduce_core[tid] = tid < kDim ? core_v : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? gate_v : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_mean = s_reduce_core[0] * (1.0f / static_cast<float>(kDim));
  const float gate_mean = s_reduce_gate[0] * (1.0f / static_cast<float>(kDim));
  __syncthreads();

  const float core_centered = tid < kDim ? core_v - core_mean : 0.0f;
  const float gate_centered = tid < kDim ? gate_v - gate_mean : 0.0f;
  s_reduce_core[tid] = tid < kDim ? core_centered * core_centered : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? gate_centered * gate_centered : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_rstd = rsqrtf(s_reduce_core[0] * (1.0f / static_cast<float>(kDim)) + eps);
  const float gate_rstd = rsqrtf(s_reduce_gate[0] * (1.0f / static_cast<float>(kDim)) + eps);
  __syncthreads();

  const float core_xhat = core_centered * core_rstd;
  const float gate_xhat = gate_centered * gate_rstd;
  float grad_core_norm = 0.0f;
  float grad_gate_norm = 0.0f;
  if (tid < kDim) {
    const float core_ln = core_xhat * core_norm_weight[tid] + core_norm_bias[tid];
    const float gate_ln = gate_xhat * gate_norm_weight[tid] + gate_norm_bias[tid];
    const float core_sig = sigmoidf_stable_p30(core_ln);
    const float core_act = core_ln * core_sig;
    const float gate_act = sigmoidf_stable_p30(gate_ln);
    const float grad = grad_out[base + tid];
    const float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    const float grad_core_ln = grad * gate_act * core_silu_grad;
    const float grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
    grad_core_norm = grad_core_ln * core_norm_weight[tid];
    grad_gate_norm = grad_gate_ln * gate_norm_weight[tid];
  }

  s_reduce_core[tid] = tid < kDim ? grad_core_norm : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_sum_grad = s_reduce_core[0];
  const float gate_sum_grad = s_reduce_gate[0];
  __syncthreads();

  s_reduce_core[tid] = tid < kDim ? grad_core_norm * core_xhat : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm * gate_xhat : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_sum_grad_xhat = s_reduce_core[0];
  const float gate_sum_grad_xhat = s_reduce_gate[0];

  if (tid < kDim) {
    const float inv_dim = 1.0f / static_cast<float>(kDim);
    s_core_grad_pre[tid] =
        (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    s_gate_grad_pre[tid] =
        (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
  }
  __syncthreads();

  if (tid < kDim) {
    float core_grad = 0.0f;
    float gate_grad = 0.0f;
    const int weight_base = tid * kDim;
    for (int j = 0; j < kDim; ++j) {
      core_grad += s_core_grad_pre[j] * core_dq_weight_t[weight_base + j];
      gate_grad += s_gate_grad_pre[j] * gate_dq_weight_t[weight_base + j];
    }
    grad_core_input[base + tid] = core_grad;
    grad_gate_input[base + tid] = gate_grad;
  }
}

__global__ void w8a8_dual_input_grad_matmul_n128_tiled_kernel(
    const float* __restrict__ grad_core_pre,
    const float* __restrict__ grad_gate_pre,
    const int8_t* __restrict__ core_q_weight,
    const int8_t* __restrict__ gate_q_weight,
    const float* __restrict__ core_weight_scale,
    const float* __restrict__ gate_weight_scale,
    float* __restrict__ grad_core_input,
    float* __restrict__ grad_gate_input,
    int64_t rows) {
  __shared__ float s_a[kTileM][kTileK];
  __shared__ float s_b[kTileK][kTileN + 1];

  const int branch = blockIdx.z;
  const int row = blockIdx.y * kTileM + threadIdx.y;
  const int col = blockIdx.x * kTileN + threadIdx.x;

  const float* grad_pre = branch == 0 ? grad_core_pre : grad_gate_pre;
  const int8_t* q_weight = branch == 0 ? core_q_weight : gate_q_weight;
  const float* weight_scale = branch == 0 ? core_weight_scale : gate_weight_scale;
  float* grad_input = branch == 0 ? grad_core_input : grad_gate_input;

  float acc = 0.0f;
  for (int k0 = 0; k0 < kDim; k0 += kTileK) {
    const int k_a = k0 + threadIdx.x;
    const int k_b = k0 + threadIdx.y;
    s_a[threadIdx.y][threadIdx.x] = (row < rows) ? grad_pre[static_cast<int64_t>(row) * kDim + k_a] : 0.0f;
    s_b[threadIdx.y][threadIdx.x] =
        (col < kDim) ? static_cast<float>(q_weight[k_b * kDim + col]) * weight_scale[k_b] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      acc += s_a[threadIdx.y][kk] * s_b[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < rows && col < kDim) {
    grad_input[static_cast<int64_t>(row) * kDim + col] = acc;
  }
}

__global__ void w8a8_dual_input_grad_matmul_n128_tiled_m32n8_kernel(
    const float* __restrict__ grad_core_pre,
    const float* __restrict__ grad_gate_pre,
    const int8_t* __restrict__ core_q_weight,
    const int8_t* __restrict__ gate_q_weight,
    const float* __restrict__ core_weight_scale,
    const float* __restrict__ gate_weight_scale,
    float* __restrict__ grad_core_input,
    float* __restrict__ grad_gate_input,
    int64_t rows) {
  __shared__ float s_a[kTileM32][kTileK];
  __shared__ float s_b[kTileK][kTileN8 + 1];

  const int branch = blockIdx.z;
  const int row = blockIdx.y * kTileM32 + threadIdx.y;
  const int col = blockIdx.x * kTileN8 + threadIdx.x;

  const float* grad_pre = branch == 0 ? grad_core_pre : grad_gate_pre;
  const int8_t* q_weight = branch == 0 ? core_q_weight : gate_q_weight;
  const float* weight_scale = branch == 0 ? core_weight_scale : gate_weight_scale;
  float* grad_input = branch == 0 ? grad_core_input : grad_gate_input;

  float acc = 0.0f;
  for (int k0 = 0; k0 < kDim; k0 += kTileK) {
    const int k_a0 = k0 + threadIdx.x;
    const int k_a1 = k0 + threadIdx.x + kTileN8;
    if (row < rows) {
      s_a[threadIdx.y][threadIdx.x] = grad_pre[static_cast<int64_t>(row) * kDim + k_a0];
      s_a[threadIdx.y][threadIdx.x + kTileN8] = grad_pre[static_cast<int64_t>(row) * kDim + k_a1];
    } else {
      s_a[threadIdx.y][threadIdx.x] = 0.0f;
      s_a[threadIdx.y][threadIdx.x + kTileN8] = 0.0f;
    }
    if (threadIdx.y < kTileK && col < kDim) {
      s_b[threadIdx.y][threadIdx.x] =
          static_cast<float>(q_weight[(k0 + threadIdx.y) * kDim + col]) * weight_scale[k0 + threadIdx.y];
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      acc += s_a[threadIdx.y][kk] * s_b[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < rows && col < kDim) {
    grad_input[static_cast<int64_t>(row) * kDim + col] = acc;
  }
}

}  // namespace

std::vector<torch::Tensor> w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda() && core_pre.is_cuda() && gate_pre.is_cuda(),
              "w8a8 saved-pre bwd: tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() &&
                  core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(),
              "w8a8 saved-pre bwd: weights/scales must be CUDA");
  TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                  gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
              "w8a8 saved-pre bwd: norm tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                  core_pre.scalar_type() == torch::kFloat32 &&
                  gate_pre.scalar_type() == torch::kFloat32,
              "w8a8 saved-pre bwd: activations must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "w8a8 saved-pre bwd: q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "w8a8 saved-pre bwd: scales/norm tensors must be float32");
  TORCH_CHECK(core_pre.dim() == 2 && gate_pre.sizes() == core_pre.sizes(),
              "w8a8 saved-pre bwd: pre-tail tensors must be matching 2D tensors");
  TORCH_CHECK(grad_out.sizes() == core_pre.sizes(), "w8a8 saved-pre bwd: grad_out shape mismatch");
  TORCH_CHECK(core_pre.size(1) == kDim, "w8a8 saved-pre bwd: input dim must be 128");
  TORCH_CHECK(core_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 saved-pre bwd: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim,
              "w8a8 saved-pre bwd: weight scales must be [128]");
  TORCH_CHECK(core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "w8a8 saved-pre bwd: norm tensors must be [128]");

  auto grad_out_c = grad_out.contiguous();
  auto core_pre_c = core_pre.contiguous();
  auto gate_pre_c = gate_pre.contiguous();
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();

  auto grad_core = torch::empty_like(core_pre_c);
  auto grad_gate = torch::empty_like(gate_pre_c);
  const int64_t rows = core_pre_c.size(0);
  if (rows == 0) {
    return {grad_core, grad_gate};
  }

  w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128_kernel<<<rows, kThreads>>>(
      grad_out_c.data_ptr<float>(),
      core_pre_c.data_ptr<float>(),
      gate_pre_c.data_ptr<float>(),
      core_q_weight_c.data_ptr<int8_t>(),
      gate_q_weight_c.data_ptr<int8_t>(),
      core_weight_scale_c.data_ptr<float>(),
      gate_weight_scale_c.data_ptr<float>(),
      core_norm_weight_c.data_ptr<float>(),
      core_norm_bias_c.data_ptr<float>(),
      gate_norm_weight_c.data_ptr<float>(),
      gate_norm_bias_c.data_ptr<float>(),
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps) {
  TORCH_CHECK(grad_out.is_cuda() && core_pre.is_cuda() && gate_pre.is_cuda() &&
                  core_dq_weight_t.is_cuda() && gate_dq_weight_t.is_cuda(),
              "w8a8 saved-pre dq bwd: tensors must be CUDA");
  TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                  gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
              "w8a8 saved-pre dq bwd: norm tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                  core_pre.scalar_type() == torch::kFloat32 &&
                  gate_pre.scalar_type() == torch::kFloat32 &&
                  core_dq_weight_t.scalar_type() == torch::kFloat32 &&
                  gate_dq_weight_t.scalar_type() == torch::kFloat32,
              "w8a8 saved-pre dq bwd: activations/weights must be float32");
  TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "w8a8 saved-pre dq bwd: norm tensors must be float32");
  TORCH_CHECK(core_pre.dim() == 2 && gate_pre.sizes() == core_pre.sizes(),
              "w8a8 saved-pre dq bwd: pre-tail tensors must be matching 2D tensors");
  TORCH_CHECK(grad_out.sizes() == core_pre.sizes(), "w8a8 saved-pre dq bwd: grad_out shape mismatch");
  TORCH_CHECK(core_pre.size(1) == kDim, "w8a8 saved-pre dq bwd: input dim must be 128");
  TORCH_CHECK(core_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 saved-pre dq bwd: dq weights must be [128, 128]");
  TORCH_CHECK(core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "w8a8 saved-pre dq bwd: norm tensors must be [128]");

  auto grad_out_c = grad_out.contiguous();
  auto core_pre_c = core_pre.contiguous();
  auto gate_pre_c = gate_pre.contiguous();
  auto core_dq_weight_t_c = core_dq_weight_t.contiguous();
  auto gate_dq_weight_t_c = gate_dq_weight_t.contiguous();
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();

  auto grad_core = torch::empty_like(core_pre_c);
  auto grad_gate = torch::empty_like(gate_pre_c);
  const int64_t rows = core_pre_c.size(0);
  if (rows == 0) {
    return {grad_core, grad_gate};
  }

  w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128_kernel<<<rows, kThreadsDq>>>(
      grad_out_c.data_ptr<float>(),
      core_pre_c.data_ptr<float>(),
      gate_pre_c.data_ptr<float>(),
      core_dq_weight_t_c.data_ptr<float>(),
      gate_dq_weight_t_c.data_ptr<float>(),
      core_norm_weight_c.data_ptr<float>(),
      core_norm_bias_c.data_ptr<float>(),
      gate_norm_weight_c.data_ptr<float>(),
      gate_norm_bias_c.data_ptr<float>(),
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_tiled(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale) {
  TORCH_CHECK(grad_core_pre.is_cuda() && grad_gate_pre.is_cuda(),
              "w8a8 tiled input-grad matmul: grad_pre tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() &&
                  core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(),
              "w8a8 tiled input-grad matmul: weights/scales must be CUDA");
  TORCH_CHECK(grad_core_pre.scalar_type() == torch::kFloat32 &&
                  grad_gate_pre.scalar_type() == torch::kFloat32,
              "w8a8 tiled input-grad matmul: grad_pre tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "w8a8 tiled input-grad matmul: q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32,
              "w8a8 tiled input-grad matmul: scales must be float32");
  TORCH_CHECK(grad_core_pre.dim() == 2 && grad_gate_pre.sizes() == grad_core_pre.sizes(),
              "w8a8 tiled input-grad matmul: grad_pre tensors must be matching 2D tensors");
  TORCH_CHECK(grad_core_pre.size(1) == kDim, "w8a8 tiled input-grad matmul: hidden dim must be 128");
  TORCH_CHECK(core_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 tiled input-grad matmul: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim,
              "w8a8 tiled input-grad matmul: scales must be [128]");

  auto grad_core_pre_c = grad_core_pre.contiguous();
  auto grad_gate_pre_c = grad_gate_pre.contiguous();
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();

  auto grad_core = torch::empty_like(grad_core_pre_c);
  auto grad_gate = torch::empty_like(grad_gate_pre_c);
  const int64_t rows = grad_core_pre_c.size(0);
  if (rows == 0) {
    return {grad_core, grad_gate};
  }

  const dim3 block(kTileN, kTileM);
  const dim3 grid((kDim + kTileN - 1) / kTileN, (rows + kTileM - 1) / kTileM, 2);
  w8a8_dual_input_grad_matmul_n128_tiled_kernel<<<grid, block>>>(
      grad_core_pre_c.data_ptr<float>(),
      grad_gate_pre_c.data_ptr<float>(),
      core_q_weight_c.data_ptr<int8_t>(),
      gate_q_weight_c.data_ptr<int8_t>(),
      core_weight_scale_c.data_ptr<float>(),
      gate_weight_scale_c.data_ptr<float>(),
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_tiled_m32n8(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale) {
  TORCH_CHECK(grad_core_pre.is_cuda() && grad_gate_pre.is_cuda(),
              "w8a8 tiled input-grad matmul m32n8: grad_pre tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() &&
                  core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(),
              "w8a8 tiled input-grad matmul m32n8: weights/scales must be CUDA");
  TORCH_CHECK(grad_core_pre.scalar_type() == torch::kFloat32 &&
                  grad_gate_pre.scalar_type() == torch::kFloat32,
              "w8a8 tiled input-grad matmul m32n8: grad_pre tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "w8a8 tiled input-grad matmul m32n8: q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32,
              "w8a8 tiled input-grad matmul m32n8: scales must be float32");
  TORCH_CHECK(grad_core_pre.dim() == 2 && grad_gate_pre.sizes() == grad_core_pre.sizes(),
              "w8a8 tiled input-grad matmul m32n8: grad_pre tensors must be matching 2D tensors");
  TORCH_CHECK(grad_core_pre.size(1) == kDim, "w8a8 tiled input-grad matmul m32n8: hidden dim must be 128");
  TORCH_CHECK(core_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 tiled input-grad matmul m32n8: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim,
              "w8a8 tiled input-grad matmul m32n8: scales must be [128]");

  auto grad_core_pre_c = grad_core_pre.contiguous();
  auto grad_gate_pre_c = grad_gate_pre.contiguous();
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();

  auto grad_core = torch::empty_like(grad_core_pre_c);
  auto grad_gate = torch::empty_like(grad_gate_pre_c);
  const int64_t rows = grad_core_pre_c.size(0);
  if (rows == 0) {
    return {grad_core, grad_gate};
  }

  const dim3 block(kTileN8, kTileM32);
  const dim3 grid((kDim + kTileN8 - 1) / kTileN8, (rows + kTileM32 - 1) / kTileM32, 2);
  w8a8_dual_input_grad_matmul_n128_tiled_m32n8_kernel<<<grid, block>>>(
      grad_core_pre_c.data_ptr<float>(),
      grad_gate_pre_c.data_ptr<float>(),
      core_q_weight_c.data_ptr<int8_t>(),
      gate_q_weight_c.data_ptr<int8_t>(),
      core_weight_scale_c.data_ptr<float>(),
      gate_weight_scale_c.data_ptr<float>(),
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_cublas_grouped(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t) {
  TORCH_CHECK(grad_core_pre.is_cuda() && grad_gate_pre.is_cuda() &&
                  core_dq_weight_t.is_cuda() && gate_dq_weight_t.is_cuda(),
              "w8a8 cublas grouped input-grad matmul: tensors must be CUDA");
  TORCH_CHECK(grad_core_pre.scalar_type() == torch::kFloat32 &&
                  grad_gate_pre.scalar_type() == torch::kFloat32 &&
                  core_dq_weight_t.scalar_type() == torch::kFloat32 &&
                  gate_dq_weight_t.scalar_type() == torch::kFloat32,
              "w8a8 cublas grouped input-grad matmul: tensors must be float32");
  TORCH_CHECK(grad_core_pre.dim() == 2 && grad_gate_pre.sizes() == grad_core_pre.sizes(),
              "w8a8 cublas grouped input-grad matmul: grad_pre tensors must be matching 2D tensors");
  TORCH_CHECK(grad_core_pre.size(1) == kDim, "w8a8 cublas grouped input-grad matmul: hidden dim must be 128");
  TORCH_CHECK(core_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 cublas grouped input-grad matmul: weights must be [128, 128]");

  auto grad_core_pre_c = grad_core_pre.contiguous();
  auto grad_gate_pre_c = grad_gate_pre.contiguous();
  auto core_weight_c = core_dq_weight_t.contiguous();
  auto gate_weight_c = gate_dq_weight_t.contiguous();
  auto grad_core = torch::empty_like(grad_core_pre_c);
  auto grad_gate = torch::empty_like(grad_gate_pre_c);

  const int rows64 = static_cast<int>(grad_core_pre_c.size(0));
  if (rows64 == 0) {
    return {grad_core, grad_gate};
  }
  TORCH_CHECK(static_cast<int64_t>(rows64) == grad_core_pre_c.size(0),
              "w8a8 cublas grouped input-grad matmul: too many rows for cublasSgemmGroupedBatched");

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const cublasOperation_t transa_array[1] = {CUBLAS_OP_T};
  const cublasOperation_t transb_array[1] = {CUBLAS_OP_N};
  const int m_array[1] = {kDim};
  const int n_array[1] = {rows64};
  const int k_array[1] = {kDim};
  const float alpha_array[1] = {1.0f};
  const float beta_array[1] = {0.0f};
  const int lda_array[1] = {kDim};
  const int ldb_array[1] = {kDim};
  const int ldc_array[1] = {kDim};
  const int group_size[1] = {2};
  const std::array<const float*, 2> h_Aarray = {
      core_weight_c.data_ptr<float>(),
      gate_weight_c.data_ptr<float>(),
  };
  const std::array<const float*, 2> h_Barray = {
      grad_core_pre_c.data_ptr<float>(),
      grad_gate_pre_c.data_ptr<float>(),
  };
  const std::array<float*, 2> h_Carray = {
      grad_core.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
  };
  auto ptr_options = torch::TensorOptions().dtype(torch::kInt64).device(grad_core_pre_c.device());
  auto d_Aarray_storage = torch::empty({2}, ptr_options);
  auto d_Barray_storage = torch::empty({2}, ptr_options);
  auto d_Carray_storage = torch::empty({2}, ptr_options);
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Aarray_storage.data_ptr<int64_t>(),
      h_Aarray.data(),
      sizeof(const float*) * h_Aarray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Barray_storage.data_ptr<int64_t>(),
      h_Barray.data(),
      sizeof(const float*) * h_Barray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Carray_storage.data_ptr<int64_t>(),
      h_Carray.data(),
      sizeof(float*) * h_Carray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));
  const float* const* Aarray = reinterpret_cast<const float* const*>(d_Aarray_storage.data_ptr<int64_t>());
  const float* const* Barray = reinterpret_cast<const float* const*>(d_Barray_storage.data_ptr<int64_t>());
  float* const* Carray = reinterpret_cast<float* const*>(d_Carray_storage.data_ptr<int64_t>());

  cublasStatus_t status = cublasSgemmGroupedBatched(
      handle,
      transa_array,
      transb_array,
      m_array,
      n_array,
      k_array,
      alpha_array,
      Aarray,
      lda_array,
      Barray,
      ldb_array,
      beta_array,
      Carray,
      ldc_array,
      1,
      group_size);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "w8a8 cublas grouped input-grad matmul: cublasSgemmGroupedBatched failed");
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_cublas_pair(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t) {
  TORCH_CHECK(grad_core_pre.is_cuda() && grad_gate_pre.is_cuda() &&
                  core_dq_weight_t.is_cuda() && gate_dq_weight_t.is_cuda(),
              "w8a8 cublas pair input-grad matmul: tensors must be CUDA");
  TORCH_CHECK(grad_core_pre.scalar_type() == torch::kFloat32 &&
                  grad_gate_pre.scalar_type() == torch::kFloat32 &&
                  core_dq_weight_t.scalar_type() == torch::kFloat32 &&
                  gate_dq_weight_t.scalar_type() == torch::kFloat32,
              "w8a8 cublas pair input-grad matmul: tensors must be float32");
  TORCH_CHECK(grad_core_pre.dim() == 2 && grad_gate_pre.sizes() == grad_core_pre.sizes(),
              "w8a8 cublas pair input-grad matmul: grad_pre tensors must be matching 2D tensors");
  TORCH_CHECK(grad_core_pre.size(1) == kDim, "w8a8 cublas pair input-grad matmul: hidden dim must be 128");
  TORCH_CHECK(core_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_dq_weight_t.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 cublas pair input-grad matmul: weights must be [128, 128]");

  auto grad_core_pre_c = grad_core_pre.contiguous();
  auto grad_gate_pre_c = grad_gate_pre.contiguous();
  auto core_weight_c = core_dq_weight_t.contiguous();
  auto gate_weight_c = gate_dq_weight_t.contiguous();
  auto grad_core = torch::empty_like(grad_core_pre_c);
  auto grad_gate = torch::empty_like(grad_gate_pre_c);

  const int rows = static_cast<int>(grad_core_pre_c.size(0));
  if (rows == 0) {
    return {grad_core, grad_gate};
  }
  TORCH_CHECK(static_cast<int64_t>(rows) == grad_core_pre_c.size(0),
              "w8a8 cublas pair input-grad matmul: too many rows for cublasSgemm");

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasStatus_t status = cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      kDim,
      rows,
      kDim,
      &alpha,
      core_weight_c.data_ptr<float>(),
      kDim,
      grad_core_pre_c.data_ptr<float>(),
      kDim,
      &beta,
      grad_core.data_ptr<float>(),
      kDim);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "w8a8 cublas pair input-grad matmul: core cublasSgemm failed");

  status = cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      kDim,
      rows,
      kDim,
      &alpha,
      gate_weight_c.data_ptr<float>(),
      kDim,
      grad_gate_pre_c.data_ptr<float>(),
      kDim,
      &beta,
      grad_gate.data_ptr<float>(),
      kDim);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "w8a8 cublas pair input-grad matmul: gate cublasSgemm failed");
  return {grad_core, grad_gate};
}

std::vector<torch::Tensor> w8a8_group8_input_grad_matmul_n128_cublas_grouped(
    const std::vector<torch::Tensor> &grad_pres,
    const std::vector<torch::Tensor> &dq_weight_ts) {
  constexpr int kGroup = 8;
  TORCH_CHECK(grad_pres.size() == kGroup && dq_weight_ts.size() == kGroup,
              "w8a8 group8 input-grad matmul: expected 8 grad tensors and 8 weight tensors");

  std::vector<torch::Tensor> grad_pres_c;
  std::vector<torch::Tensor> weights_c;
  std::vector<torch::Tensor> outputs;
  grad_pres_c.reserve(kGroup);
  weights_c.reserve(kGroup);
  outputs.reserve(kGroup);

  int rows = -1;
  for (int i = 0; i < kGroup; ++i) {
    TORCH_CHECK(grad_pres[i].is_cuda() && dq_weight_ts[i].is_cuda(),
                "w8a8 group8 input-grad matmul: tensors must be CUDA");
    TORCH_CHECK(grad_pres[i].scalar_type() == torch::kFloat32 &&
                    dq_weight_ts[i].scalar_type() == torch::kFloat32,
                "w8a8 group8 input-grad matmul: tensors must be float32");
    TORCH_CHECK(grad_pres[i].dim() == 2 && grad_pres[i].size(1) == kDim,
                "w8a8 group8 input-grad matmul: grad_pre tensors must be [rows, 128]");
    TORCH_CHECK(dq_weight_ts[i].sizes() == torch::IntArrayRef({kDim, kDim}),
                "w8a8 group8 input-grad matmul: dq weights must be [128, 128]");
    const int tensor_rows = static_cast<int>(grad_pres[i].size(0));
    TORCH_CHECK(static_cast<int64_t>(tensor_rows) == grad_pres[i].size(0),
                "w8a8 group8 input-grad matmul: too many rows");
    if (i == 0) {
      rows = tensor_rows;
    } else {
      TORCH_CHECK(rows == tensor_rows,
                  "w8a8 group8 input-grad matmul: P97B sanity path expects same rows");
    }
    grad_pres_c.push_back(grad_pres[i].contiguous());
    weights_c.push_back(dq_weight_ts[i].contiguous());
    outputs.push_back(torch::empty_like(grad_pres_c.back()));
  }

  if (rows == 0) {
    return outputs;
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const cublasOperation_t transa_array[1] = {CUBLAS_OP_T};
  const cublasOperation_t transb_array[1] = {CUBLAS_OP_N};
  const int m_array[1] = {kDim};
  const int n_array[1] = {rows};
  const int k_array[1] = {kDim};
  const float alpha_array[1] = {1.0f};
  const float beta_array[1] = {0.0f};
  const int lda_array[1] = {kDim};
  const int ldb_array[1] = {kDim};
  const int ldc_array[1] = {kDim};
  const int group_size[1] = {kGroup};

  std::array<const float*, kGroup> h_Aarray{};
  std::array<const float*, kGroup> h_Barray{};
  std::array<float*, kGroup> h_Carray{};
  for (int i = 0; i < kGroup; ++i) {
    h_Aarray[i] = weights_c[i].data_ptr<float>();
    h_Barray[i] = grad_pres_c[i].data_ptr<float>();
    h_Carray[i] = outputs[i].data_ptr<float>();
  }

  auto ptr_options = torch::TensorOptions().dtype(torch::kInt64).device(grad_pres_c[0].device());
  auto d_Aarray_storage = torch::empty({kGroup}, ptr_options);
  auto d_Barray_storage = torch::empty({kGroup}, ptr_options);
  auto d_Carray_storage = torch::empty({kGroup}, ptr_options);
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Aarray_storage.data_ptr<int64_t>(),
      h_Aarray.data(),
      sizeof(const float*) * h_Aarray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Barray_storage.data_ptr<int64_t>(),
      h_Barray.data(),
      sizeof(const float*) * h_Barray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_Carray_storage.data_ptr<int64_t>(),
      h_Carray.data(),
      sizeof(float*) * h_Carray.size(),
      cudaMemcpyHostToDevice,
      stream.stream()));

  const float* const* Aarray = reinterpret_cast<const float* const*>(d_Aarray_storage.data_ptr<int64_t>());
  const float* const* Barray = reinterpret_cast<const float* const*>(d_Barray_storage.data_ptr<int64_t>());
  float* const* Carray = reinterpret_cast<float* const*>(d_Carray_storage.data_ptr<int64_t>());

  cublasStatus_t status = cublasSgemmGroupedBatched(
      handle,
      transa_array,
      transb_array,
      m_array,
      n_array,
      k_array,
      alpha_array,
      Aarray,
      lda_array,
      Barray,
      ldb_array,
      beta_array,
      Carray,
      ldc_array,
      1,
      group_size);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "w8a8 group8 input-grad matmul: cublasSgemmGroupedBatched failed");
  return outputs;
}
