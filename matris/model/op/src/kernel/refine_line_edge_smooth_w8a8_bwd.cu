#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kDim = 128;
constexpr int kOutDim = 256;
constexpr int kInDim = 512;
constexpr int kThreads = 256;
constexpr int kTailThreads = 128;

__device__ __forceinline__ float sigmoidf_stable_p64(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_grad_p64(float x) {
  const float sig = sigmoidf_stable_p64(x);
  return sig * (1.0f + x * (1.0f - sig));
}

__global__ void refine_line_edge_smooth_w8a8_backward_n128_kernel(
    const float* __restrict__ grad_refine_node,
    const float* __restrict__ grad_nonlinear_direct,
    const float* __restrict__ nonlinear,
    const float* __restrict__ base_envelope,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ first_weight,
    const float* __restrict__ core_raw,
    const float* __restrict__ gate_raw,
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
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    float* __restrict__ grad_base,
    int64_t rows,
    bool has_direct_grad,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_grad[kDim];
  __shared__ float s_gate_grad[kDim];
  __shared__ float s_reduce_core[kThreads];
  __shared__ float s_reduce_gate[kThreads];

  const int64_t base = static_cast<int64_t>(row) * kDim;
  const int64_t atom = atom_index[row];
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];

  float grad_nonlinear_value = 0.0f;
  if (tid < kDim) {
    const float grad_refine = grad_refine_node[target * kDim + tid];
    const float nonlinear_v = nonlinear[base + tid];
    const float bi = base_envelope[source * kDim + tid];
    const float bj = base_envelope[target * kDim + tid];
    grad_nonlinear_value = grad_refine * bi * bj;
    if (has_direct_grad) {
      grad_nonlinear_value += grad_nonlinear_direct[base + tid];
    }
    atomicAdd(&grad_base[source * kDim + tid], grad_refine * nonlinear_v * bj);
    atomicAdd(&grad_base[target * kDim + tid], grad_refine * nonlinear_v * bi);
  }

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
    const float core_sig = sigmoidf_stable_p64(core_ln);
    const float core_act = core_ln * core_sig;
    const float gate_act = sigmoidf_stable_p64(gate_ln);
    const float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    const float grad_core_ln = grad_nonlinear_value * gate_act * core_silu_grad;
    const float grad_gate_ln = grad_nonlinear_value * core_act * gate_act * (1.0f - gate_act);
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
  __syncthreads();

  if (tid < kDim) {
    const float inv_dim = 1.0f / static_cast<float>(kDim);
    const float grad_core_pre =
        (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    const float grad_gate_pre =
        (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
    s_core_grad[tid] = grad_core_pre;
    s_gate_grad[tid] = grad_gate_pre;
  }
  __syncthreads();

  if (tid < kDim) {
    float grad_core_input = 0.0f;
    float grad_gate_input = 0.0f;
    for (int j = 0; j < kDim; ++j) {
      grad_core_input += s_core_grad[j] *
                         static_cast<float>(core_q_weight[j * kDim + tid]) *
                         core_weight_scale[j];
      grad_gate_input += s_gate_grad[j] *
                         static_cast<float>(gate_q_weight[j * kDim + tid]) *
                         gate_weight_scale[j];
    }

    s_core_grad[tid] = grad_core_input * silu_grad_p64(core_raw[base + tid]);
    s_gate_grad[tid] = grad_gate_input * silu_grad_p64(gate_raw[base + tid]);
  }
  __syncthreads();

  if (tid < kDim) {
    float grad_edge_value = 0.0f;
    float grad_atom_value = 0.0f;
    float grad_target_value = 0.0f;
    float grad_source_value = 0.0f;
    for (int j = 0; j < kDim; ++j) {
      const float core_g = s_core_grad[j];
      const float gate_g = s_gate_grad[j];
      const int64_t core_weight_base = static_cast<int64_t>(j) * kInDim;
      const int64_t gate_weight_base = static_cast<int64_t>(kDim + j) * kInDim;
      grad_edge_value += core_g * first_weight[core_weight_base + tid] +
                         gate_g * first_weight[gate_weight_base + tid];
      grad_atom_value += core_g * first_weight[core_weight_base + kDim + tid] +
                         gate_g * first_weight[gate_weight_base + kDim + tid];
      grad_target_value += core_g * first_weight[core_weight_base + 2 * kDim + tid] +
                           gate_g * first_weight[gate_weight_base + 2 * kDim + tid];
      grad_source_value += core_g * first_weight[core_weight_base + 3 * kDim + tid] +
                           gate_g * first_weight[gate_weight_base + 3 * kDim + tid];
    }
    grad_edge[base + tid] = grad_edge_value;
    atomicAdd(&grad_atom[atom * kDim + tid], grad_atom_value);
    atomicAdd(&grad_node[target * kDim + tid], grad_target_value);
    atomicAdd(&grad_node[source * kDim + tid], grad_source_value);
  }
}

__global__ void refine_line_smooth_w8a8_tail_input_grad_backward_n128_kernel(
    const float* __restrict__ grad_refine_node,
    const float* __restrict__ grad_nonlinear_direct,
    const float* __restrict__ nonlinear,
    const float* __restrict__ base_envelope,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
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
    float* __restrict__ grad_base,
    int64_t rows,
    bool has_direct_grad,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_grad_pre[kDim];
  __shared__ float s_gate_grad_pre[kDim];
  __shared__ float s_reduce_core[kTailThreads];
  __shared__ float s_reduce_gate[kTailThreads];

  const int64_t base = static_cast<int64_t>(row) * kDim;
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];

  float grad_nonlinear_value = 0.0f;
  {
    const float grad_refine = grad_refine_node[target * kDim + tid];
    const float nonlinear_v = nonlinear[base + tid];
    const float bi = base_envelope[source * kDim + tid];
    const float bj = base_envelope[target * kDim + tid];
    grad_nonlinear_value = grad_refine * bi * bj;
    if (has_direct_grad) {
      grad_nonlinear_value += grad_nonlinear_direct[base + tid];
    }
    atomicAdd(&grad_base[source * kDim + tid], grad_refine * nonlinear_v * bj);
    atomicAdd(&grad_base[target * kDim + tid], grad_refine * nonlinear_v * bi);
  }

  const float core_v = core_pre[base + tid];
  const float gate_v = gate_pre[base + tid];
  __syncthreads();

  s_reduce_core[tid] = core_v;
  s_reduce_gate[tid] = gate_v;
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

  const float core_centered = core_v - core_mean;
  const float gate_centered = gate_v - gate_mean;
  s_reduce_core[tid] = core_centered * core_centered;
  s_reduce_gate[tid] = gate_centered * gate_centered;
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
  const float core_ln = core_xhat * core_norm_weight[tid] + core_norm_bias[tid];
  const float gate_ln = gate_xhat * gate_norm_weight[tid] + gate_norm_bias[tid];
  const float core_sig = sigmoidf_stable_p64(core_ln);
  const float core_act = core_ln * core_sig;
  const float gate_act = sigmoidf_stable_p64(gate_ln);
  const float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
  const float grad_core_ln = grad_nonlinear_value * gate_act * core_silu_grad;
  const float grad_gate_ln = grad_nonlinear_value * core_act * gate_act * (1.0f - gate_act);
  const float grad_core_norm = grad_core_ln * core_norm_weight[tid];
  const float grad_gate_norm = grad_gate_ln * gate_norm_weight[tid];

  s_reduce_core[tid] = grad_core_norm;
  s_reduce_gate[tid] = grad_gate_norm;
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

  s_reduce_core[tid] = grad_core_norm * core_xhat;
  s_reduce_gate[tid] = grad_gate_norm * gate_xhat;
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

  const float inv_dim = 1.0f / static_cast<float>(kDim);
  s_core_grad_pre[tid] =
      (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
      core_rstd * inv_dim;
  s_gate_grad_pre[tid] =
      (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
      gate_rstd * inv_dim;
  __syncthreads();

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

__global__ void refine_line_smooth_w8a8_tail_actgrad_backward_n128_kernel(
    const float* __restrict__ grad_refine_node,
    const float* __restrict__ grad_nonlinear_direct,
    const float* __restrict__ nonlinear,
    const float* __restrict__ base_envelope,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
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
    float* __restrict__ grad_act,
    float* __restrict__ grad_base,
    int64_t rows,
    bool has_direct_grad,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_grad_pre[kDim];
  __shared__ float s_gate_grad_pre[kDim];
  __shared__ float s_reduce_core[kTailThreads];
  __shared__ float s_reduce_gate[kTailThreads];

  const int64_t base = static_cast<int64_t>(row) * kDim;
  const int64_t source = source_index[row];
  const int64_t target = target_index[row];

  const float grad_refine = grad_refine_node[target * kDim + tid];
  const float nonlinear_v = nonlinear[base + tid];
  const float bi = base_envelope[source * kDim + tid];
  const float bj = base_envelope[target * kDim + tid];
  float grad_nonlinear_value = grad_refine * bi * bj;
  if (has_direct_grad) {
    grad_nonlinear_value += grad_nonlinear_direct[base + tid];
  }
  atomicAdd(&grad_base[source * kDim + tid], grad_refine * nonlinear_v * bj);
  atomicAdd(&grad_base[target * kDim + tid], grad_refine * nonlinear_v * bi);

  const float core_v = core_pre[base + tid];
  const float gate_v = gate_pre[base + tid];
  s_reduce_core[tid] = core_v;
  s_reduce_gate[tid] = gate_v;
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

  const float core_centered = core_v - core_mean;
  const float gate_centered = gate_v - gate_mean;
  s_reduce_core[tid] = core_centered * core_centered;
  s_reduce_gate[tid] = gate_centered * gate_centered;
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
  const float core_ln = core_xhat * core_norm_weight[tid] + core_norm_bias[tid];
  const float gate_ln = gate_xhat * gate_norm_weight[tid] + gate_norm_bias[tid];
  const float core_sig = sigmoidf_stable_p64(core_ln);
  const float core_act = core_ln * core_sig;
  const float gate_act = sigmoidf_stable_p64(gate_ln);
  const float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
  const float grad_core_ln = grad_nonlinear_value * gate_act * core_silu_grad;
  const float grad_gate_ln = grad_nonlinear_value * core_act * gate_act * (1.0f - gate_act);
  const float grad_core_norm = grad_core_ln * core_norm_weight[tid];
  const float grad_gate_norm = grad_gate_ln * gate_norm_weight[tid];

  s_reduce_core[tid] = grad_core_norm;
  s_reduce_gate[tid] = grad_gate_norm;
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

  s_reduce_core[tid] = grad_core_norm * core_xhat;
  s_reduce_gate[tid] = grad_gate_norm * gate_xhat;
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

  const float inv_dim = 1.0f / static_cast<float>(kDim);
  s_core_grad_pre[tid] =
      (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
      core_rstd * inv_dim;
  s_gate_grad_pre[tid] =
      (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
      gate_rstd * inv_dim;
  __syncthreads();

  float core_grad = 0.0f;
  float gate_grad = 0.0f;
  for (int j = 0; j < kDim; ++j) {
    core_grad += s_core_grad_pre[j] *
                 static_cast<float>(core_q_weight[j * kDim + tid]) * core_weight_scale[j];
    gate_grad += s_gate_grad_pre[j] *
                 static_cast<float>(gate_q_weight[j * kDim + tid]) * gate_weight_scale[j];
  }
  const int64_t act_base = static_cast<int64_t>(row) * kOutDim;
  grad_act[act_base + tid] = core_grad;
  grad_act[act_base + kDim + tid] = gate_grad;
}

__device__ __forceinline__ void store_p65_refine_line_first_grad_value(
    float value,
    int64_t row,
    int col,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom) {
  if (col < kDim) {
    grad_edge[row * kDim + col] = value;
  } else if (col < 2 * kDim) {
    atomicAdd(&grad_atom[atom_index[row] * kDim + (col - kDim)], value);
  } else if (col < 3 * kDim) {
    atomicAdd(&grad_node[target_index[row] * kDim + (col - 2 * kDim)], value);
  } else {
    atomicAdd(&grad_node[source_index[row] * kDim + (col - 3 * kDim)], value);
  }
}

template <int kTileRows>
__global__ void refine_line_smooth_tail_first_silu_backward_tile_kernel(
    const float* __restrict__ grad_refine_node,
    const float* __restrict__ grad_nonlinear_direct,
    const float* __restrict__ nonlinear,
    const float* __restrict__ base_envelope,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ first_weight,
    const float* __restrict__ core_raw,
    const float* __restrict__ gate_raw,
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
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    float* __restrict__ grad_base,
    int64_t rows,
    bool has_direct_grad,
    float eps) {
  const int tid = threadIdx.x;
  const int64_t row0 = static_cast<int64_t>(blockIdx.x) * kTileRows;

  __shared__ float s_core_grad[kTileRows][kDim];
  __shared__ float s_gate_grad[kTileRows][kDim];
  __shared__ float s_reduce_core[kTailThreads];
  __shared__ float s_reduce_gate[kTailThreads];

  for (int local_row = 0; local_row < kTileRows; ++local_row) {
    const int64_t row = row0 + local_row;
    const bool valid_row = row < rows;
    const int64_t base = row * kDim;
    int64_t source = 0;
    int64_t target = 0;
    if (valid_row) {
      source = source_index[row];
      target = target_index[row];
    }

    float grad_nonlinear_value = 0.0f;
    float core_v = 0.0f;
    float gate_v = 0.0f;
    if (tid < kDim && valid_row) {
      const float grad_refine = grad_refine_node[target * kDim + tid];
      const float nonlinear_v = nonlinear[base + tid];
      const float bi = base_envelope[source * kDim + tid];
      const float bj = base_envelope[target * kDim + tid];
      grad_nonlinear_value = grad_refine * bi * bj;
      if (has_direct_grad) {
        grad_nonlinear_value += grad_nonlinear_direct[base + tid];
      }
      atomicAdd(&grad_base[source * kDim + tid], grad_refine * nonlinear_v * bj);
      atomicAdd(&grad_base[target * kDim + tid], grad_refine * nonlinear_v * bi);
      core_v = core_pre[base + tid];
      gate_v = gate_pre[base + tid];
    }

    if (tid < kTailThreads) {
      s_reduce_core[tid] = (tid < kDim && valid_row) ? core_v : 0.0f;
      s_reduce_gate[tid] = (tid < kDim && valid_row) ? gate_v : 0.0f;
    }
    __syncthreads();
    for (int stride = kTailThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce_core[tid] += s_reduce_core[tid + stride];
        s_reduce_gate[tid] += s_reduce_gate[tid + stride];
      }
      __syncthreads();
    }
    const float core_mean = s_reduce_core[0] * (1.0f / static_cast<float>(kDim));
    const float gate_mean = s_reduce_gate[0] * (1.0f / static_cast<float>(kDim));
    __syncthreads();

    const float core_centered = (tid < kDim && valid_row) ? core_v - core_mean : 0.0f;
    const float gate_centered = (tid < kDim && valid_row) ? gate_v - gate_mean : 0.0f;
    if (tid < kTailThreads) {
      s_reduce_core[tid] = core_centered * core_centered;
      s_reduce_gate[tid] = gate_centered * gate_centered;
    }
    __syncthreads();
    for (int stride = kTailThreads / 2; stride > 0; stride >>= 1) {
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
    if (tid < kDim && valid_row) {
      const float core_ln = core_xhat * core_norm_weight[tid] + core_norm_bias[tid];
      const float gate_ln = gate_xhat * gate_norm_weight[tid] + gate_norm_bias[tid];
      const float core_sig = sigmoidf_stable_p64(core_ln);
      const float core_act = core_ln * core_sig;
      const float gate_act = sigmoidf_stable_p64(gate_ln);
      const float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
      const float grad_core_ln = grad_nonlinear_value * gate_act * core_silu_grad;
      const float grad_gate_ln = grad_nonlinear_value * core_act * gate_act * (1.0f - gate_act);
      grad_core_norm = grad_core_ln * core_norm_weight[tid];
      grad_gate_norm = grad_gate_ln * gate_norm_weight[tid];
    }

    if (tid < kTailThreads) {
      s_reduce_core[tid] = grad_core_norm;
      s_reduce_gate[tid] = grad_gate_norm;
    }
    __syncthreads();
    for (int stride = kTailThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce_core[tid] += s_reduce_core[tid + stride];
        s_reduce_gate[tid] += s_reduce_gate[tid + stride];
      }
      __syncthreads();
    }
    const float core_sum_grad = s_reduce_core[0];
    const float gate_sum_grad = s_reduce_gate[0];
    __syncthreads();

    if (tid < kTailThreads) {
      s_reduce_core[tid] = grad_core_norm * core_xhat;
      s_reduce_gate[tid] = grad_gate_norm * gate_xhat;
    }
    __syncthreads();
    for (int stride = kTailThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce_core[tid] += s_reduce_core[tid + stride];
        s_reduce_gate[tid] += s_reduce_gate[tid + stride];
      }
      __syncthreads();
    }
    const float core_sum_grad_xhat = s_reduce_core[0];
    const float gate_sum_grad_xhat = s_reduce_gate[0];
    __syncthreads();

    if (tid < kDim && valid_row) {
      const float inv_dim = 1.0f / static_cast<float>(kDim);
      const float grad_core_pre =
          (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
          core_rstd * inv_dim;
      const float grad_gate_pre =
          (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
          gate_rstd * inv_dim;
      s_reduce_core[tid] = grad_core_pre;
      s_reduce_gate[tid] = grad_gate_pre;
    } else if (tid < kDim) {
      s_reduce_core[tid] = 0.0f;
      s_reduce_gate[tid] = 0.0f;
    }
    __syncthreads();

    if (tid < kDim && valid_row) {
      float core_grad = 0.0f;
      float gate_grad = 0.0f;
      for (int j = 0; j < kDim; ++j) {
        core_grad += s_reduce_core[j] *
                     static_cast<float>(core_q_weight[j * kDim + tid]) * core_weight_scale[j];
        gate_grad += s_reduce_gate[j] *
                     static_cast<float>(gate_q_weight[j * kDim + tid]) * gate_weight_scale[j];
      }
      s_core_grad[local_row][tid] = core_grad * silu_grad_p64(core_raw[base + tid]);
      s_gate_grad[local_row][tid] = gate_grad * silu_grad_p64(gate_raw[base + tid]);
    } else if (tid < kDim) {
      s_core_grad[local_row][tid] = 0.0f;
      s_gate_grad[local_row][tid] = 0.0f;
    }
    __syncthreads();
  }

  for (int idx = tid; idx < kTileRows * kInDim; idx += blockDim.x) {
    const int local_row = idx / kInDim;
    const int col = idx - local_row * kInDim;
    const int64_t row = row0 + local_row;
    if (row >= rows) {
      continue;
    }
    float acc = 0.0f;
    for (int j = 0; j < kDim; ++j) {
      acc += s_core_grad[local_row][j] * first_weight[static_cast<int64_t>(j) * kInDim + col] +
             s_gate_grad[local_row][j] * first_weight[static_cast<int64_t>(kDim + j) * kInDim + col];
    }
    store_p65_refine_line_first_grad_value(
        acc, row, col, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
}

}  // namespace

std::vector<torch::Tensor> refine_line_edge_smooth_w8a8_backward_n128(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
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
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows,
    bool has_direct_grad,
    double eps) {
  TORCH_CHECK(grad_refine_node.is_cuda() && grad_nonlinear_direct.is_cuda() && nonlinear.is_cuda() &&
                  base_envelope.is_cuda() && atom_index.is_cuda() && source_index.is_cuda() &&
                  target_index.is_cuda() && first_weight.is_cuda() && core_raw.is_cuda() &&
                  gate_raw.is_cuda() && core_pre.is_cuda() && gate_pre.is_cuda() &&
                  core_q_weight.is_cuda() && gate_q_weight.is_cuda() && core_weight_scale.is_cuda() &&
                  gate_weight_scale.is_cuda() && core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                  gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
              "refine_line_edge_smooth_w8a8_backward_n128: all tensors must be CUDA");
  TORCH_CHECK(grad_refine_node.scalar_type() == torch::kFloat32 &&
                  grad_nonlinear_direct.scalar_type() == torch::kFloat32 &&
                  nonlinear.scalar_type() == torch::kFloat32 &&
                  base_envelope.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32 &&
                  core_raw.scalar_type() == torch::kFloat32 && gate_raw.scalar_type() == torch::kFloat32 &&
                  core_pre.scalar_type() == torch::kFloat32 && gate_pre.scalar_type() == torch::kFloat32 &&
                  core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_edge_smooth_w8a8_backward_n128: float tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_edge_smooth_w8a8_backward_n128: q weights must be int8");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_edge_smooth_w8a8_backward_n128: indices must be int64");
  TORCH_CHECK(grad_refine_node.dim() == 2 && grad_refine_node.size(1) == kDim,
              "refine_line_edge_smooth_w8a8_backward_n128: grad_refine_node must be [nodes, 128]");
  TORCH_CHECK(base_envelope.dim() == 2 && base_envelope.size(1) == kDim &&
                  base_envelope.size(0) == grad_refine_node.size(0),
              "refine_line_edge_smooth_w8a8_backward_n128: base_envelope must match node rows");
  TORCH_CHECK(nonlinear.dim() == 2 && nonlinear.size(1) == kDim,
              "refine_line_edge_smooth_w8a8_backward_n128: nonlinear must be [rows, 128]");
  TORCH_CHECK(!has_direct_grad || grad_nonlinear_direct.sizes() == nonlinear.sizes(),
              "refine_line_edge_smooth_w8a8_backward_n128: direct grad shape mismatch");
  TORCH_CHECK(core_raw.sizes() == nonlinear.sizes() && gate_raw.sizes() == nonlinear.sizes() &&
                  core_pre.sizes() == nonlinear.sizes() && gate_pre.sizes() == nonlinear.sizes(),
              "refine_line_edge_smooth_w8a8_backward_n128: saved row tensors must match nonlinear");
  TORCH_CHECK(first_weight.dim() == 2 && first_weight.size(0) == kOutDim && first_weight.size(1) == kInDim,
              "refine_line_edge_smooth_w8a8_backward_n128: first_weight must be [256, 512]");
  TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2 &&
                  core_q_weight.size(0) == kDim && core_q_weight.size(1) == kDim &&
                  gate_q_weight.size(0) == kDim && gate_q_weight.size(1) == kDim,
              "refine_line_edge_smooth_w8a8_backward_n128: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim &&
                  core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "refine_line_edge_smooth_w8a8_backward_n128: scale/norm tensors must be [128]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == nonlinear.size(0) && source_index.size(0) == nonlinear.size(0) &&
                  target_index.size(0) == nonlinear.size(0),
              "refine_line_edge_smooth_w8a8_backward_n128: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows >= 0 && atom_rows >= 0,
              "refine_line_edge_smooth_w8a8_backward_n128: output rows must be non-negative");

  auto grad_refine_node_c = grad_refine_node.contiguous();
  auto grad_nonlinear_direct_c = grad_nonlinear_direct.contiguous();
  auto nonlinear_c = nonlinear.contiguous();
  auto base_c = base_envelope.contiguous();
  auto atom_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto core_raw_c = core_raw.contiguous();
  auto gate_raw_c = gate_raw.contiguous();
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

  auto grad_node = nonlinear_c.new_zeros({node_rows, kDim});
  auto grad_edge = nonlinear_c.new_empty({edge_rows, kDim});
  auto grad_atom = nonlinear_c.new_zeros({atom_rows, kDim});
  auto grad_base = base_c.new_zeros(base_c.sizes());
  const int64_t rows = nonlinear_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge.zero_(), grad_atom, grad_base};
  }
  TORCH_CHECK(edge_rows == rows,
              "refine_line_edge_smooth_w8a8_backward_n128: edge_rows must match nonlinear rows");

  refine_line_edge_smooth_w8a8_backward_n128_kernel<<<static_cast<unsigned int>(rows), kThreads>>>(
      grad_refine_node_c.data_ptr<float>(),
      has_direct_grad ? grad_nonlinear_direct_c.data_ptr<float>() : nullptr,
      nonlinear_c.data_ptr<float>(),
      base_c.data_ptr<float>(),
      atom_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      first_weight_c.data_ptr<float>(),
      core_raw_c.data_ptr<float>(),
      gate_raw_c.data_ptr<float>(),
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
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      grad_atom.data_ptr<float>(),
      grad_base.data_ptr<float>(),
      rows,
      has_direct_grad,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge, grad_atom, grad_base};
}

std::vector<torch::Tensor> refine_line_smooth_w8a8_tail_input_grad_backward_n128(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
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
    bool has_direct_grad,
    double eps) {
  TORCH_CHECK(grad_refine_node.is_cuda() && grad_nonlinear_direct.is_cuda() && nonlinear.is_cuda() &&
                  base_envelope.is_cuda() && source_index.is_cuda() && target_index.is_cuda() &&
                  core_pre.is_cuda() && gate_pre.is_cuda() && core_q_weight.is_cuda() &&
                  gate_q_weight.is_cuda() && core_weight_scale.is_cuda() && gate_weight_scale.is_cuda() &&
                  core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() &&
                  gate_norm_bias.is_cuda(),
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: all tensors must be CUDA");
  TORCH_CHECK(grad_refine_node.scalar_type() == torch::kFloat32 &&
                  grad_nonlinear_direct.scalar_type() == torch::kFloat32 &&
                  nonlinear.scalar_type() == torch::kFloat32 &&
                  base_envelope.scalar_type() == torch::kFloat32 &&
                  core_pre.scalar_type() == torch::kFloat32 && gate_pre.scalar_type() == torch::kFloat32 &&
                  core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: float tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: q weights must be int8");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: indices must be int64");
  TORCH_CHECK(grad_refine_node.dim() == 2 && grad_refine_node.size(1) == kDim,
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: grad_refine_node must be [nodes, 128]");
  TORCH_CHECK(base_envelope.dim() == 2 && base_envelope.size(1) == kDim &&
                  base_envelope.size(0) == grad_refine_node.size(0),
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: base_envelope must match node rows");
  TORCH_CHECK(nonlinear.dim() == 2 && nonlinear.size(1) == kDim,
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: nonlinear must be [rows, 128]");
  TORCH_CHECK(!has_direct_grad || grad_nonlinear_direct.sizes() == nonlinear.sizes(),
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: direct grad shape mismatch");
  TORCH_CHECK(core_pre.sizes() == nonlinear.sizes() && gate_pre.sizes() == nonlinear.sizes(),
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: pre-tail tensors must match rows");
  TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2 &&
                  core_q_weight.size(0) == kDim && core_q_weight.size(1) == kDim &&
                  gate_q_weight.size(0) == kDim && gate_q_weight.size(1) == kDim,
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim &&
                  core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: scale/norm tensors must be [128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == nonlinear.size(0) && target_index.size(0) == nonlinear.size(0),
              "refine_line_smooth_w8a8_tail_input_grad_backward_n128: index sizes must match rows");

  auto grad_refine_node_c = grad_refine_node.contiguous();
  auto grad_nonlinear_direct_c = grad_nonlinear_direct.contiguous();
  auto nonlinear_c = nonlinear.contiguous();
  auto base_c = base_envelope.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
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

  auto grad_core = nonlinear_c.new_empty(nonlinear_c.sizes());
  auto grad_gate = nonlinear_c.new_empty(nonlinear_c.sizes());
  auto grad_base = base_c.new_zeros(base_c.sizes());
  const int64_t rows = nonlinear_c.size(0);
  if (rows == 0) {
    return {grad_core.zero_(), grad_gate.zero_(), grad_base};
  }

  refine_line_smooth_w8a8_tail_input_grad_backward_n128_kernel<<<static_cast<unsigned int>(rows), kTailThreads>>>(
      grad_refine_node_c.data_ptr<float>(),
      has_direct_grad ? grad_nonlinear_direct_c.data_ptr<float>() : nullptr,
      nonlinear_c.data_ptr<float>(),
      base_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
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
      grad_base.data_ptr<float>(),
      rows,
      has_direct_grad,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate, grad_base};
}

std::vector<torch::Tensor> refine_line_smooth_w8a8_tail_actgrad_backward_n128(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
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
    bool has_direct_grad,
    double eps) {
  TORCH_CHECK(grad_refine_node.is_cuda() && grad_nonlinear_direct.is_cuda() && nonlinear.is_cuda() &&
                  base_envelope.is_cuda() && source_index.is_cuda() && target_index.is_cuda() &&
                  core_pre.is_cuda() && gate_pre.is_cuda() && core_q_weight.is_cuda() &&
                  gate_q_weight.is_cuda() && core_weight_scale.is_cuda() && gate_weight_scale.is_cuda() &&
                  core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() &&
                  gate_norm_bias.is_cuda(),
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: all tensors must be CUDA");
  TORCH_CHECK(grad_refine_node.scalar_type() == torch::kFloat32 &&
                  grad_nonlinear_direct.scalar_type() == torch::kFloat32 &&
                  nonlinear.scalar_type() == torch::kFloat32 &&
                  base_envelope.scalar_type() == torch::kFloat32 &&
                  core_pre.scalar_type() == torch::kFloat32 && gate_pre.scalar_type() == torch::kFloat32 &&
                  core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: float tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: q weights must be int8");
  TORCH_CHECK(source_index.scalar_type() == torch::kInt64 && target_index.scalar_type() == torch::kInt64,
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: indices must be int64");
  TORCH_CHECK(grad_refine_node.dim() == 2 && grad_refine_node.size(1) == kDim,
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: grad_refine_node must be [nodes, 128]");
  TORCH_CHECK(base_envelope.dim() == 2 && base_envelope.size(1) == kDim &&
                  base_envelope.size(0) == grad_refine_node.size(0),
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: base_envelope must match node rows");
  TORCH_CHECK(nonlinear.dim() == 2 && nonlinear.size(1) == kDim,
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: nonlinear must be [rows, 128]");
  TORCH_CHECK(!has_direct_grad || grad_nonlinear_direct.sizes() == nonlinear.sizes(),
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: direct grad shape mismatch");
  TORCH_CHECK(core_pre.sizes() == nonlinear.sizes() && gate_pre.sizes() == nonlinear.sizes(),
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: pre-tail tensors must match rows");
  TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2 &&
                  core_q_weight.size(0) == kDim && core_q_weight.size(1) == kDim &&
                  gate_q_weight.size(0) == kDim && gate_q_weight.size(1) == kDim,
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim &&
                  core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: scale/norm tensors must be [128]");
  TORCH_CHECK(source_index.dim() == 1 && target_index.dim() == 1 &&
                  source_index.size(0) == nonlinear.size(0) && target_index.size(0) == nonlinear.size(0),
              "refine_line_smooth_w8a8_tail_actgrad_backward_n128: index sizes must match rows");

  auto grad_refine_node_c = grad_refine_node.contiguous();
  auto grad_nonlinear_direct_c = grad_nonlinear_direct.contiguous();
  auto nonlinear_c = nonlinear.contiguous();
  auto base_c = base_envelope.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
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

  const int64_t rows = nonlinear_c.size(0);
  auto grad_act = nonlinear_c.new_empty({rows, kOutDim});
  auto grad_base = base_c.new_zeros(base_c.sizes());
  if (rows == 0) {
    return {grad_act.zero_(), grad_base};
  }

  refine_line_smooth_w8a8_tail_actgrad_backward_n128_kernel<<<static_cast<unsigned int>(rows), kTailThreads>>>(
      grad_refine_node_c.data_ptr<float>(),
      has_direct_grad ? grad_nonlinear_direct_c.data_ptr<float>() : nullptr,
      nonlinear_c.data_ptr<float>(),
      base_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
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
      grad_act.data_ptr<float>(),
      grad_base.data_ptr<float>(),
      rows,
      has_direct_grad,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_act, grad_base};
}

std::vector<torch::Tensor> refine_line_smooth_tail_first_silu_backward_tile(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
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
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows,
    bool has_direct_grad,
    double eps,
    int64_t tile_rows) {
  TORCH_CHECK(grad_refine_node.is_cuda() && grad_nonlinear_direct.is_cuda() && nonlinear.is_cuda() &&
                  base_envelope.is_cuda() && atom_index.is_cuda() && source_index.is_cuda() &&
                  target_index.is_cuda() && first_weight.is_cuda() && core_raw.is_cuda() &&
                  gate_raw.is_cuda() && core_pre.is_cuda() && gate_pre.is_cuda() &&
                  core_q_weight.is_cuda() && gate_q_weight.is_cuda() && core_weight_scale.is_cuda() &&
                  gate_weight_scale.is_cuda() && core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                  gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
              "refine_line_smooth_tail_first_silu_backward_tile: all tensors must be CUDA");
  TORCH_CHECK(grad_refine_node.scalar_type() == torch::kFloat32 &&
                  grad_nonlinear_direct.scalar_type() == torch::kFloat32 &&
                  nonlinear.scalar_type() == torch::kFloat32 &&
                  base_envelope.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32 &&
                  core_raw.scalar_type() == torch::kFloat32 && gate_raw.scalar_type() == torch::kFloat32 &&
                  core_pre.scalar_type() == torch::kFloat32 && gate_pre.scalar_type() == torch::kFloat32 &&
                  core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_smooth_tail_first_silu_backward_tile: float tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_smooth_tail_first_silu_backward_tile: q weights must be int8");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_smooth_tail_first_silu_backward_tile: indices must be int64");
  TORCH_CHECK(grad_refine_node.dim() == 2 && grad_refine_node.size(1) == kDim,
              "refine_line_smooth_tail_first_silu_backward_tile: grad_refine_node must be [nodes, 128]");
  TORCH_CHECK(base_envelope.dim() == 2 && base_envelope.size(1) == kDim &&
                  base_envelope.size(0) == grad_refine_node.size(0),
              "refine_line_smooth_tail_first_silu_backward_tile: base_envelope must match node rows");
  TORCH_CHECK(nonlinear.dim() == 2 && nonlinear.size(1) == kDim,
              "refine_line_smooth_tail_first_silu_backward_tile: nonlinear must be [rows, 128]");
  TORCH_CHECK(!has_direct_grad || grad_nonlinear_direct.sizes() == nonlinear.sizes(),
              "refine_line_smooth_tail_first_silu_backward_tile: direct grad shape mismatch");
  TORCH_CHECK(core_raw.sizes() == nonlinear.sizes() && gate_raw.sizes() == nonlinear.sizes() &&
                  core_pre.sizes() == nonlinear.sizes() && gate_pre.sizes() == nonlinear.sizes(),
              "refine_line_smooth_tail_first_silu_backward_tile: saved row tensors must match nonlinear");
  TORCH_CHECK(first_weight.dim() == 2 && first_weight.size(0) == kOutDim && first_weight.size(1) == kInDim,
              "refine_line_smooth_tail_first_silu_backward_tile: first_weight must be [256, 512]");
  TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2 &&
                  core_q_weight.size(0) == kDim && core_q_weight.size(1) == kDim &&
                  gate_q_weight.size(0) == kDim && gate_q_weight.size(1) == kDim,
              "refine_line_smooth_tail_first_silu_backward_tile: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim &&
                  core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "refine_line_smooth_tail_first_silu_backward_tile: scale/norm tensors must be [128]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == nonlinear.size(0) && source_index.size(0) == nonlinear.size(0) &&
                  target_index.size(0) == nonlinear.size(0),
              "refine_line_smooth_tail_first_silu_backward_tile: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows == nonlinear.size(0) && atom_rows >= 0,
              "refine_line_smooth_tail_first_silu_backward_tile: invalid output rows");

  auto grad_refine_node_c = grad_refine_node.contiguous();
  auto grad_nonlinear_direct_c = grad_nonlinear_direct.contiguous();
  auto nonlinear_c = nonlinear.contiguous();
  auto base_c = base_envelope.contiguous();
  auto atom_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto core_raw_c = core_raw.contiguous();
  auto gate_raw_c = gate_raw.contiguous();
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

  auto grad_node = nonlinear_c.new_zeros({node_rows, kDim});
  auto grad_edge = nonlinear_c.new_empty({edge_rows, kDim});
  auto grad_atom = nonlinear_c.new_zeros({atom_rows, kDim});
  auto grad_base = base_c.new_zeros(base_c.sizes());
  const int64_t rows = nonlinear_c.size(0);
  if (rows == 0) {
    return {grad_node, grad_edge.zero_(), grad_atom, grad_base};
  }

  // Tile-32 matches the existing first-projection backward row schedule and
  // preserves numerical alignment on high-collision graph samples.
  const int selected_tile_rows = 32;
  const int threads = 256;
  if (selected_tile_rows == 4) {
    refine_line_smooth_tail_first_silu_backward_tile_kernel<4>
        <<<static_cast<unsigned int>((rows + 3) / 4), threads>>>(
            grad_refine_node_c.data_ptr<float>(),
            has_direct_grad ? grad_nonlinear_direct_c.data_ptr<float>() : nullptr,
            nonlinear_c.data_ptr<float>(),
            base_c.data_ptr<float>(),
            atom_c.data_ptr<int64_t>(),
            source_c.data_ptr<int64_t>(),
            target_c.data_ptr<int64_t>(),
            first_weight_c.data_ptr<float>(),
            core_raw_c.data_ptr<float>(),
            gate_raw_c.data_ptr<float>(),
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
            grad_node.data_ptr<float>(),
            grad_edge.data_ptr<float>(),
            grad_atom.data_ptr<float>(),
            grad_base.data_ptr<float>(),
            rows,
            has_direct_grad,
            static_cast<float>(eps));
  } else if (selected_tile_rows == 8) {
    refine_line_smooth_tail_first_silu_backward_tile_kernel<8>
        <<<static_cast<unsigned int>((rows + 7) / 8), threads>>>(
            grad_refine_node_c.data_ptr<float>(),
            has_direct_grad ? grad_nonlinear_direct_c.data_ptr<float>() : nullptr,
            nonlinear_c.data_ptr<float>(),
            base_c.data_ptr<float>(),
            atom_c.data_ptr<int64_t>(),
            source_c.data_ptr<int64_t>(),
            target_c.data_ptr<int64_t>(),
            first_weight_c.data_ptr<float>(),
            core_raw_c.data_ptr<float>(),
            gate_raw_c.data_ptr<float>(),
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
            grad_node.data_ptr<float>(),
            grad_edge.data_ptr<float>(),
            grad_atom.data_ptr<float>(),
            grad_base.data_ptr<float>(),
            rows,
            has_direct_grad,
            static_cast<float>(eps));
  } else if (selected_tile_rows == 32) {
    refine_line_smooth_tail_first_silu_backward_tile_kernel<32>
        <<<static_cast<unsigned int>((rows + 31) / 32), threads>>>(
            grad_refine_node_c.data_ptr<float>(),
            has_direct_grad ? grad_nonlinear_direct_c.data_ptr<float>() : nullptr,
            nonlinear_c.data_ptr<float>(),
            base_c.data_ptr<float>(),
            atom_c.data_ptr<int64_t>(),
            source_c.data_ptr<int64_t>(),
            target_c.data_ptr<int64_t>(),
            first_weight_c.data_ptr<float>(),
            core_raw_c.data_ptr<float>(),
            gate_raw_c.data_ptr<float>(),
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
            grad_node.data_ptr<float>(),
            grad_edge.data_ptr<float>(),
            grad_atom.data_ptr<float>(),
            grad_base.data_ptr<float>(),
            rows,
            has_direct_grad,
            static_cast<float>(eps));
  } else {
    refine_line_smooth_tail_first_silu_backward_tile_kernel<16>
        <<<static_cast<unsigned int>((rows + 15) / 16), threads>>>(
            grad_refine_node_c.data_ptr<float>(),
            has_direct_grad ? grad_nonlinear_direct_c.data_ptr<float>() : nullptr,
            nonlinear_c.data_ptr<float>(),
            base_c.data_ptr<float>(),
            atom_c.data_ptr<int64_t>(),
            source_c.data_ptr<int64_t>(),
            target_c.data_ptr<int64_t>(),
            first_weight_c.data_ptr<float>(),
            core_raw_c.data_ptr<float>(),
            gate_raw_c.data_ptr<float>(),
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
            grad_node.data_ptr<float>(),
            grad_edge.data_ptr<float>(),
            grad_atom.data_ptr<float>(),
            grad_base.data_ptr<float>(),
            rows,
            has_direct_grad,
            static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge, grad_atom, grad_base};
}
