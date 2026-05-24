#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <limits>

void launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(
    const float *core_input,
    const float *gate_input,
    const int8_t *core_q_weight,
    const int8_t *gate_q_weight,
    const float *core_weight_scale,
    const float *gate_weight_scale,
    const float *core_activation_scale,
    const float *gate_activation_scale,
    const float *core_bias,
    const float *gate_bias,
    const float *core_norm_weight,
    const float *core_norm_bias,
    const float *gate_norm_weight,
    const float *gate_norm_bias,
    float *core_pre_output,
    float *gate_pre_output,
    float *output,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps,
    cudaStream_t stream);

void launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(
    const float *core_input,
    const float *gate_input,
    const int8_t *core_q_weight,
    const int8_t *gate_q_weight,
    const float *core_weight_scale,
    const float *gate_weight_scale,
    const float *core_activation_scale,
    const float *gate_activation_scale,
    const float *core_bias,
    const float *gate_bias,
    const float *core_norm_weight,
    const float *core_norm_bias,
    const float *gate_norm_weight,
    const float *gate_norm_bias,
    float *core_pre_output,
    float *gate_pre_output,
    float *output,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps,
    cudaStream_t stream);

void launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_smooth_kernel(
    const float *core_input,
    const float *gate_input,
    const int8_t *core_q_weight,
    const int8_t *gate_q_weight,
    const float *core_weight_scale,
    const float *gate_weight_scale,
    const float *core_activation_scale,
    const float *gate_activation_scale,
    const float *core_bias,
    const float *gate_bias,
    const float *core_norm_weight,
    const float *core_norm_bias,
    const float *gate_norm_weight,
    const float *gate_norm_bias,
    const float *base_envelope,
    const int64_t *source_index,
    const int64_t *target_index,
    float *core_pre_output,
    float *gate_pre_output,
    float *output,
    float *refine_node,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps,
    cudaStream_t stream);

namespace {

using namespace nvcuda;

constexpr int kDim = 128;
constexpr int kInDim = 4 * kDim;
constexpr int kOutDim = 2 * kDim;
constexpr int kForwardTileM = 16;
constexpr int kForwardTileN = 16;
constexpr int kForwardTileK = 32;
constexpr int kProjectTile32M = 32;
constexpr int kProjectTile32N = 32;
constexpr int kProjectTile32K = 32;
constexpr int kScaleBlock = 256;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kTailN128Tiles = 8;

__device__ __forceinline__ float sigmoidf_stable_refine_line(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_refine_line(float x) {
  return x * sigmoidf_stable_refine_line(x);
}

__device__ __forceinline__ float silu_grad_refine_line(float x) {
  const float sig = sigmoidf_stable_refine_line(x);
  return sig * (1.0f + x * (1.0f - sig));
}

__device__ __forceinline__ int8_t quantize_s8_refine_line(float x, float scale) {
  float scaled = x / scale;
  int q = static_cast<int>(rintf(scaled));
  q = q > 127 ? 127 : q;
  q = q < -127 ? -127 : q;
  return static_cast<int8_t>(q);
}

__device__ __forceinline__ float half_warp_sum_refine_line(float value) {
  value += __shfl_down_sync(0xffffffffu, value, 8, 16);
  value += __shfl_down_sync(0xffffffffu, value, 4, 16);
  value += __shfl_down_sync(0xffffffffu, value, 2, 16);
  value += __shfl_down_sync(0xffffffffu, value, 1, 16);
  return value;
}

__device__ __forceinline__ float refine_line_input_value(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const float* __restrict__ atom_feat,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    int64_t row,
    int col) {
  const int dim = col & (kDim - 1);
  if (col < kDim) {
    return edge_feat[row * kDim + dim];
  }
  if (col < 2 * kDim) {
    return atom_feat[atom_index[row] * kDim + dim];
  }
  if (col < 3 * kDim) {
    return node_feat[target_index[row] * kDim + dim];
  }
  return node_feat[source_index[row] * kDim + dim];
}

__global__ void refine_line_first_silu_quant_forward_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const float* __restrict__ atom_feat,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    bool has_bias,
    const float* __restrict__ core_activation_scale,
    const float* __restrict__ gate_activation_scale,
    int8_t* __restrict__ core_q_act,
    int8_t* __restrict__ gate_q_act,
    int64_t rows) {
  __shared__ float s_x[kForwardTileM][kForwardTileK];
  __shared__ float s_w[kForwardTileK][kForwardTileN + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + ty;
  const int out_col = static_cast<int>(blockIdx.x) * kForwardTileN + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < kInDim; k0 += kForwardTileK) {
    for (int idx = tid; idx < kForwardTileM * kForwardTileK; idx += blockDim.x * blockDim.y) {
      const int r = idx / kForwardTileK;
      const int k = idx - r * kForwardTileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + r;
      const int global_k = k0 + k;
      s_x[r][k] = (global_row < rows && global_k < kInDim)
                      ? refine_line_input_value(
                            node_feat,
                            edge_feat,
                            atom_feat,
                            atom_index,
                            source_index,
                            target_index,
                            global_row,
                            global_k)
                      : 0.0f;
    }
    for (int idx = tid; idx < kForwardTileK * kForwardTileN; idx += blockDim.x * blockDim.y) {
      const int k = idx / kForwardTileN;
      const int c = idx - k * kForwardTileN;
      const int global_k = k0 + k;
      const int global_out = static_cast<int>(blockIdx.x) * kForwardTileN + c;
      s_w[k][c] = (global_k < kInDim && global_out < kOutDim)
                      ? weight[global_out * kInDim + global_k]
                      : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kForwardTileK; ++kk) {
      acc += s_x[ty][kk] * s_w[kk][tx];
    }
    __syncthreads();
  }

  if (row >= rows || out_col >= kOutDim) {
    return;
  }
  if (has_bias) {
    acc += bias[out_col];
  }
  const float act = silu_refine_line(acc);
  const int64_t out_base = row * kDim;
  if (out_col < kDim) {
    core_q_act[out_base + out_col] = quantize_s8_refine_line(act, core_activation_scale[0]);
  } else {
    const int dim = out_col - kDim;
    gate_q_act[out_base + dim] = quantize_s8_refine_line(act, gate_activation_scale[0]);
  }
}

__global__ void refine_line_first_silu_absmax_partial_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const float* __restrict__ atom_feat,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    bool has_bias,
    float* __restrict__ scale_partials,
    int64_t rows) {
  __shared__ float s_x[kForwardTileM][kForwardTileK];
  __shared__ float s_w[kForwardTileK][kForwardTileN + 1];
  __shared__ float s_core_abs[kForwardTileM * kForwardTileN];
  __shared__ float s_gate_abs[kForwardTileM * kForwardTileN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + ty;
  const int out_col = static_cast<int>(blockIdx.x) * kForwardTileN + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < kInDim; k0 += kForwardTileK) {
    for (int idx = tid; idx < kForwardTileM * kForwardTileK; idx += blockDim.x * blockDim.y) {
      const int r = idx / kForwardTileK;
      const int k = idx - r * kForwardTileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + r;
      const int global_k = k0 + k;
      s_x[r][k] = (global_row < rows && global_k < kInDim)
                      ? refine_line_input_value(
                            node_feat,
                            edge_feat,
                            atom_feat,
                            atom_index,
                            source_index,
                            target_index,
                            global_row,
                            global_k)
                      : 0.0f;
    }
    for (int idx = tid; idx < kForwardTileK * kForwardTileN; idx += blockDim.x * blockDim.y) {
      const int k = idx / kForwardTileN;
      const int c = idx - k * kForwardTileN;
      const int global_k = k0 + k;
      const int global_out = static_cast<int>(blockIdx.x) * kForwardTileN + c;
      s_w[k][c] = (global_k < kInDim && global_out < kOutDim)
                      ? weight[global_out * kInDim + global_k]
                      : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kForwardTileK; ++kk) {
      acc += s_x[ty][kk] * s_w[kk][tx];
    }
    __syncthreads();
  }

  float core_abs = 0.0f;
  float gate_abs = 0.0f;
  if (row < rows && out_col < kOutDim) {
    if (has_bias) {
      acc += bias[out_col];
    }
    const float act = silu_refine_line(acc);
    if (out_col < kDim) {
      core_abs = fabsf(act);
    } else {
      gate_abs = fabsf(act);
    }
  }
  s_core_abs[tid] = core_abs;
  s_gate_abs[tid] = gate_abs;
  __syncthreads();
  for (int offset = (kForwardTileM * kForwardTileN) >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_core_abs[tid] = fmaxf(s_core_abs[tid], s_core_abs[tid + offset]);
      s_gate_abs[tid] = fmaxf(s_gate_abs[tid], s_gate_abs[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int block_id = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int total_blocks = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
    scale_partials[block_id] = s_core_abs[0];
    scale_partials[total_blocks + block_id] = s_gate_abs[0];
  }
}

template <bool SaveRaw, bool WriteScalePartials>
__global__ void refine_line_first_silu_forward_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const float* __restrict__ atom_feat,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    bool has_bias,
    float* __restrict__ core_act,
    float* __restrict__ gate_act,
    float* __restrict__ core_raw,
    float* __restrict__ gate_raw,
    float* __restrict__ scale_partials,
    int64_t rows) {
  __shared__ float s_x[kForwardTileM][kForwardTileK];
  __shared__ float s_w[kForwardTileK][kForwardTileN + 1];
  __shared__ float s_core_abs[kForwardTileM * kForwardTileN];
  __shared__ float s_gate_abs[kForwardTileM * kForwardTileN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + ty;
  const int out_col = static_cast<int>(blockIdx.x) * kForwardTileN + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < kInDim; k0 += kForwardTileK) {
    for (int idx = tid; idx < kForwardTileM * kForwardTileK; idx += blockDim.x * blockDim.y) {
      const int r = idx / kForwardTileK;
      const int k = idx - r * kForwardTileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + r;
      const int global_k = k0 + k;
      s_x[r][k] = (global_row < rows && global_k < kInDim)
                      ? refine_line_input_value(
                            node_feat,
                            edge_feat,
                            atom_feat,
                            atom_index,
                            source_index,
                            target_index,
                            global_row,
                            global_k)
                      : 0.0f;
    }
    for (int idx = tid; idx < kForwardTileK * kForwardTileN; idx += blockDim.x * blockDim.y) {
      const int k = idx / kForwardTileN;
      const int c = idx - k * kForwardTileN;
      const int global_k = k0 + k;
      const int global_out = static_cast<int>(blockIdx.x) * kForwardTileN + c;
      s_w[k][c] = (global_k < kInDim && global_out < kOutDim)
                      ? weight[global_out * kInDim + global_k]
                      : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kForwardTileK; ++kk) {
      acc += s_x[ty][kk] * s_w[kk][tx];
    }
    __syncthreads();
  }

  float core_abs = 0.0f;
  float gate_abs = 0.0f;
  if (row < rows && out_col < kOutDim) {
    if (has_bias) {
      acc += bias[out_col];
    }
    const float act = silu_refine_line(acc);
    const int64_t out_base = row * kDim;
    if (out_col < kDim) {
      if constexpr (SaveRaw) {
        core_raw[out_base + out_col] = acc;
      }
      core_act[out_base + out_col] = act;
      core_abs = fabsf(act);
    } else {
      const int dim = out_col - kDim;
      if constexpr (SaveRaw) {
        gate_raw[out_base + dim] = acc;
      }
      gate_act[out_base + dim] = act;
      gate_abs = fabsf(act);
    }
  }

  if constexpr (WriteScalePartials) {
    s_core_abs[tid] = core_abs;
    s_gate_abs[tid] = gate_abs;
    __syncthreads();
    for (int offset = (kForwardTileM * kForwardTileN) >> 1; offset > 0; offset >>= 1) {
      if (tid < offset) {
        s_core_abs[tid] = fmaxf(s_core_abs[tid], s_core_abs[tid + offset]);
        s_gate_abs[tid] = fmaxf(s_gate_abs[tid], s_gate_abs[tid + offset]);
      }
      __syncthreads();
    }
    if (tid == 0) {
      const int block_id = static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
      const int total_blocks = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y);
      scale_partials[block_id] = s_core_abs[0];
      scale_partials[total_blocks + block_id] = s_gate_abs[0];
    }
  }
}

template <bool SaveRaw>
__global__ void refine_line_first_silu_forward_plain_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ edge_feat,
    const float* __restrict__ atom_feat,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    bool has_bias,
    float* __restrict__ core_act,
    float* __restrict__ gate_act,
    float* __restrict__ core_raw,
    float* __restrict__ gate_raw,
    int64_t rows) {
  __shared__ float s_x[kForwardTileM][kForwardTileK];
  __shared__ float s_w[kForwardTileK][kForwardTileN + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + ty;
  const int out_col = static_cast<int>(blockIdx.x) * kForwardTileN + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < kInDim; k0 += kForwardTileK) {
    for (int idx = tid; idx < kForwardTileM * kForwardTileK; idx += blockDim.x * blockDim.y) {
      const int r = idx / kForwardTileK;
      const int k = idx - r * kForwardTileK;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kForwardTileM + r;
      const int global_k = k0 + k;
      s_x[r][k] = (global_row < rows && global_k < kInDim)
                      ? refine_line_input_value(
                            node_feat,
                            edge_feat,
                            atom_feat,
                            atom_index,
                            source_index,
                            target_index,
                            global_row,
                            global_k)
                      : 0.0f;
    }
    for (int idx = tid; idx < kForwardTileK * kForwardTileN; idx += blockDim.x * blockDim.y) {
      const int k = idx / kForwardTileN;
      const int c = idx - k * kForwardTileN;
      const int global_k = k0 + k;
      const int global_out = static_cast<int>(blockIdx.x) * kForwardTileN + c;
      s_w[k][c] = (global_k < kInDim && global_out < kOutDim)
                      ? weight[global_out * kInDim + global_k]
                      : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kForwardTileK; ++kk) {
      acc += s_x[ty][kk] * s_w[kk][tx];
    }
    __syncthreads();
  }

  if (row >= rows || out_col >= kOutDim) {
    return;
  }
  if (has_bias) {
    acc += bias[out_col];
  }
  const int64_t out_base = row * kDim;
  if (out_col < kDim) {
    if constexpr (SaveRaw) {
      core_raw[out_base + out_col] = acc;
    }
    core_act[out_base + out_col] = silu_refine_line(acc);
  } else {
    const int dim = out_col - kDim;
    if constexpr (SaveRaw) {
      gate_raw[out_base + dim] = acc;
    }
    gate_act[out_base + dim] = silu_refine_line(acc);
  }
}

__global__ void refine_line_dual_absmax_partial_kernel(
    const float* __restrict__ core_act,
    const float* __restrict__ gate_act,
    float* __restrict__ partials,
    int64_t numel) {
  __shared__ float s_core[kScaleBlock];
  __shared__ float s_gate[kScaleBlock];
  const int tid = threadIdx.x;
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  float core_max = 0.0f;
  float gate_max = 0.0f;
  for (int64_t idx = start; idx < numel; idx += stride) {
    core_max = fmaxf(core_max, fabsf(core_act[idx]));
    gate_max = fmaxf(gate_max, fabsf(gate_act[idx]));
  }
  s_core[tid] = core_max;
  s_gate[tid] = gate_max;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_core[tid] = fmaxf(s_core[tid], s_core[tid + offset]);
      s_gate[tid] = fmaxf(s_gate[tid], s_gate[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    partials[blockIdx.x] = s_core[0];
    partials[gridDim.x + blockIdx.x] = s_gate[0];
  }
}

__global__ void refine_line_dual_absmax_scale_finalize_kernel(
    const float* __restrict__ partials,
    float* __restrict__ scales,
    int partial_blocks) {
  __shared__ float s_core[kScaleBlock];
  __shared__ float s_gate[kScaleBlock];
  const int tid = threadIdx.x;
  float core_max = 0.0f;
  float gate_max = 0.0f;
  for (int idx = tid; idx < partial_blocks; idx += blockDim.x) {
    core_max = fmaxf(core_max, partials[idx]);
    gate_max = fmaxf(gate_max, partials[partial_blocks + idx]);
  }
  s_core[tid] = core_max;
  s_gate[tid] = gate_max;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_core[tid] = fmaxf(s_core[tid], s_core[tid + offset]);
      s_gate[tid] = fmaxf(s_gate[tid], s_gate[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float inv_qmax = 1.0f / 127.0f;
    scales[0] = fmaxf(s_core[0], 1.0e-12f) * inv_qmax;
    scales[1] = fmaxf(s_gate[0], 1.0e-12f) * inv_qmax;
  }
}

__device__ __forceinline__ void store_refine_line_first_grad_value(
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

__global__ void refine_line_first_silu_backward_kernel(
    const float* __restrict__ grad_core,
    const float* __restrict__ grad_gate,
    const float* __restrict__ core_raw,
    const float* __restrict__ gate_raw,
    const float* __restrict__ weight,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  for (int k0 = 0; k0 < kOutDim; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kOutDim) {
        const int64_t base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_raw[base + global_k];
          value = grad_core[base + global_k] * silu_grad_refine_line(x);
        } else {
          const int dim = global_k - kDim;
          const float x = gate_raw[base + dim];
          value = grad_gate[base + dim] * silu_grad_refine_line(x);
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kOutDim && global_col < kInDim)
                           ? weight[global_k * kInDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 < rows && col0 < kInDim) {
    store_refine_line_first_grad_value(
        acc00, row0, col0, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 < rows && col0 + 1 < kInDim) {
    store_refine_line_first_grad_value(
        acc01, row0, col0 + 1, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 + 1 < rows && col0 < kInDim) {
    store_refine_line_first_grad_value(
        acc10, row0 + 1, col0, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 + 1 < rows && col0 + 1 < kInDim) {
    store_refine_line_first_grad_value(
        acc11, row0 + 1, col0 + 1, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
}

__global__ void refine_line_first_silu_backward_packed_kernel(
    const float* __restrict__ grad_act,
    const float* __restrict__ core_raw,
    const float* __restrict__ gate_raw,
    const float* __restrict__ weight,
    const int64_t* __restrict__ atom_index,
    const int64_t* __restrict__ source_index,
    const int64_t* __restrict__ target_index,
    float* __restrict__ grad_node,
    float* __restrict__ grad_edge,
    float* __restrict__ grad_atom,
    int64_t rows) {
  __shared__ float s_grad[kProjectTile32M][kProjectTile32K];
  __shared__ float s_weight[kProjectTile32K][kProjectTile32N + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + ty * 2;
  const int col0 = static_cast<int>(blockIdx.x) * kProjectTile32N + tx * 2;

  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;

  for (int k0 = 0; k0 < kOutDim; k0 += kProjectTile32K) {
    for (int idx = tid; idx < kProjectTile32M * kProjectTile32K; idx += blockDim.x * blockDim.y) {
      const int r = idx / kProjectTile32K;
      const int k = idx - r * kProjectTile32K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * kProjectTile32M + r;
      const int global_k = k0 + k;
      float value = 0.0f;
      if (global_row < rows && global_k < kOutDim) {
        const int64_t act_base = global_row * kOutDim;
        const int64_t raw_base = global_row * kDim;
        if (global_k < kDim) {
          const float x = core_raw[raw_base + global_k];
          value = grad_act[act_base + global_k] * silu_grad_refine_line(x);
        } else {
          const int dim = global_k - kDim;
          const float x = gate_raw[raw_base + dim];
          value = grad_act[act_base + global_k] * silu_grad_refine_line(x);
        }
      }
      s_grad[r][k] = value;
    }
    for (int idx = tid; idx < kProjectTile32K * kProjectTile32N; idx += blockDim.x * blockDim.y) {
      const int k = idx / kProjectTile32N;
      const int c = idx - k * kProjectTile32N;
      const int global_k = k0 + k;
      const int global_col = static_cast<int>(blockIdx.x) * kProjectTile32N + c;
      s_weight[k][c] = (global_k < kOutDim && global_col < kInDim)
                           ? weight[global_k * kInDim + global_col]
                           : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kProjectTile32K; ++kk) {
      const float a0 = s_grad[ty * 2][kk];
      const float a1 = s_grad[ty * 2 + 1][kk];
      const float b0 = s_weight[kk][tx * 2];
      const float b1 = s_weight[kk][tx * 2 + 1];
      acc00 += a0 * b0;
      acc01 += a0 * b1;
      acc10 += a1 * b0;
      acc11 += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 < rows && col0 < kInDim) {
    store_refine_line_first_grad_value(
        acc00, row0, col0, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 < rows && col0 + 1 < kInDim) {
    store_refine_line_first_grad_value(
        acc01, row0, col0 + 1, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 + 1 < rows && col0 < kInDim) {
    store_refine_line_first_grad_value(
        acc10, row0 + 1, col0, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
  if (row0 + 1 < rows && col0 + 1 < kInDim) {
    store_refine_line_first_grad_value(
        acc11, row0 + 1, col0 + 1, atom_index, source_index, target_index, grad_node, grad_edge, grad_atom);
  }
}

__global__ void refine_line_w8a8_packed_gated_tail_n128_parallel_kernel(
    const int8_t* __restrict__ core_q_input,
    const int8_t* __restrict__ gate_q_input,
    const int8_t* __restrict__ core_q_weight,
    const int8_t* __restrict__ gate_q_weight,
    const float* __restrict__ core_weight_scale,
    const float* __restrict__ gate_weight_scale,
    const float* __restrict__ core_activation_scale,
    const float* __restrict__ gate_activation_scale,
    const float* __restrict__ core_bias,
    const float* __restrict__ gate_bias,
    const float* __restrict__ core_norm_weight,
    const float* __restrict__ core_norm_bias,
    const float* __restrict__ gate_norm_weight,
    const float* __restrict__ gate_norm_bias,
    float* __restrict__ output,
    int64_t rows,
    bool core_has_bias,
    bool gate_has_bias,
    float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const bool is_gate = warp_id >= kTailN128Tiles;
  const int branch_warp = warp_id - (is_gate ? kTailN128Tiles : 0);
  const int branch_thread = branch_warp * warpSize + lane;
  const int tile_m = static_cast<int>(blockIdx.x) * kWmmaM;
  const int tile_n = branch_warp * kWmmaN;
  const float core_act_scale = core_activation_scale[0];
  const float gate_act_scale = gate_activation_scale[0];
  const int8_t* branch_input = is_gate ? gate_q_input : core_q_input;
  const int8_t* branch_q_weight = is_gate ? gate_q_weight : core_q_weight;

  __shared__ int8_t core_a_tile[kWmmaM * kWmmaK];
  __shared__ int8_t gate_a_tile[kWmmaM * kWmmaK];
  __shared__ int core_acc_tile[kTailN128Tiles][kWmmaM * kWmmaN];
  __shared__ int gate_acc_tile[kTailN128Tiles][kWmmaM * kWmmaN];
  __shared__ float core_mean[kWmmaM];
  __shared__ float gate_mean[kWmmaM];
  __shared__ float core_rstd[kWmmaM];
  __shared__ float gate_rstd[kWmmaM];

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int k0 = 0; k0 < kDim; k0 += kWmmaK) {
    int8_t* branch_a_tile = is_gate ? gate_a_tile : core_a_tile;
    for (int idx = branch_thread; idx < kWmmaM * kWmmaK; idx += kTailN128Tiles * warpSize) {
      const int local_m = idx / kWmmaK;
      const int local_k = idx % kWmmaK;
      const int row = tile_m + local_m;
      const int k = k0 + local_k;
      branch_a_tile[idx] = (row < rows && k < kDim) ? branch_input[static_cast<int64_t>(row) * kDim + k] : 0;
    }
    __syncthreads();

    const int8_t* b_tile = branch_q_weight + static_cast<int64_t>(tile_n) * kDim + k0;
    wmma::load_matrix_sync(a_frag, branch_a_tile, kWmmaK);
    wmma::load_matrix_sync(b_frag, b_tile, kDim);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __syncthreads();
  }

  if (is_gate) {
    wmma::store_matrix_sync(gate_acc_tile[branch_warp], acc_frag, kWmmaN, wmma::mem_row_major);
  } else {
    wmma::store_matrix_sync(core_acc_tile[branch_warp], acc_frag, kWmmaN, wmma::mem_row_major);
  }
  __syncthreads();

  const int half_warp = lane >> 4;
  const int half_lane = lane & 15;
  const int reduce_row = warp_id * 2 + half_warp;
  if (reduce_row < kWmmaM) {
    float core_sum = 0.0f;
    float gate_sum = 0.0f;
    for (int col = half_lane; col < kDim; col += 16) {
      const int n_tile = col / kWmmaN;
      const int local_n = col % kWmmaN;
      const int tile_idx = reduce_row * kWmmaN + local_n;
      float core = static_cast<float>(core_acc_tile[n_tile][tile_idx]) * core_act_scale * core_weight_scale[col];
      float gate = static_cast<float>(gate_acc_tile[n_tile][tile_idx]) * gate_act_scale * gate_weight_scale[col];
      if (core_has_bias) {
        core += core_bias[col];
      }
      if (gate_has_bias) {
        gate += gate_bias[col];
      }
      core_sum += core;
      gate_sum += gate;
    }
    core_sum = half_warp_sum_refine_line(core_sum);
    gate_sum = half_warp_sum_refine_line(gate_sum);
    float c_mean = core_sum * (1.0f / static_cast<float>(kDim));
    float g_mean = gate_sum * (1.0f / static_cast<float>(kDim));
    c_mean = __shfl_sync(0xffffffffu, c_mean, 0, 16);
    g_mean = __shfl_sync(0xffffffffu, g_mean, 0, 16);
    if (half_lane == 0) {
      core_mean[reduce_row] = c_mean;
      gate_mean[reduce_row] = g_mean;
    }

    float core_var = 0.0f;
    float gate_var = 0.0f;
    for (int col = half_lane; col < kDim; col += 16) {
      const int n_tile = col / kWmmaN;
      const int local_n = col % kWmmaN;
      const int tile_idx = reduce_row * kWmmaN + local_n;
      float core = static_cast<float>(core_acc_tile[n_tile][tile_idx]) * core_act_scale * core_weight_scale[col];
      float gate = static_cast<float>(gate_acc_tile[n_tile][tile_idx]) * gate_act_scale * gate_weight_scale[col];
      if (core_has_bias) {
        core += core_bias[col];
      }
      if (gate_has_bias) {
        gate += gate_bias[col];
      }
      const float c = core - c_mean;
      const float g = gate - g_mean;
      core_var += c * c;
      gate_var += g * g;
    }
    core_var = half_warp_sum_refine_line(core_var);
    gate_var = half_warp_sum_refine_line(gate_var);
    if (half_lane == 0) {
      core_rstd[reduce_row] = rsqrtf(core_var * (1.0f / static_cast<float>(kDim)) + eps);
      gate_rstd[reduce_row] = rsqrtf(gate_var * (1.0f / static_cast<float>(kDim)) + eps);
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kWmmaM * kDim; idx += blockDim.x) {
    const int local_m = idx / kDim;
    const int col = idx % kDim;
    const int row = tile_m + local_m;
    if (row < rows) {
      const int n_tile = col / kWmmaN;
      const int local_n = col % kWmmaN;
      const int tile_idx = local_m * kWmmaN + local_n;
      float core = static_cast<float>(core_acc_tile[n_tile][tile_idx]) * core_act_scale * core_weight_scale[col];
      float gate = static_cast<float>(gate_acc_tile[n_tile][tile_idx]) * gate_act_scale * gate_weight_scale[col];
      if (core_has_bias) {
        core += core_bias[col];
      }
      if (gate_has_bias) {
        gate += gate_bias[col];
      }
      core = (core - core_mean[local_m]) * core_rstd[local_m];
      gate = (gate - gate_mean[local_m]) * gate_rstd[local_m];
      core = core * core_norm_weight[col] + core_norm_bias[col];
      gate = gate * gate_norm_weight[col] + gate_norm_bias[col];
      const float core_sigmoid = 1.0f / (1.0f + expf(-core));
      const float gate_sigmoid = 1.0f / (1.0f + expf(-gate));
      output[static_cast<int64_t>(row) * kDim + col] = core * core_sigmoid * gate_sigmoid;
    }
  }
#endif
}

}  // namespace

std::vector<torch::Tensor> refine_line_first_silu_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    bool has_bias) {
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && atom_feat.is_cuda() && atom_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && weight.is_cuda(),
              "refine_line_first_silu_forward: all tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32 &&
                  atom_feat.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "refine_line_first_silu_forward: feature and weight tensors must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_silu_forward: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && atom_feat.dim() == 2 &&
                  node_feat.size(1) == kDim && edge_feat.size(1) == kDim && atom_feat.size(1) == kDim,
              "refine_line_first_silu_forward: features must be [rows, 128]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kOutDim && weight.size(1) == kInDim,
              "refine_line_first_silu_forward: weight must be [256, 512]");
  TORCH_CHECK(!has_bias || (bias.is_cuda() && bias.scalar_type() == torch::kFloat32 && bias.numel() == kOutDim),
              "refine_line_first_silu_forward: bias must be CUDA float32 [256]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == edge_feat.size(0) && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "refine_line_first_silu_forward: index sizes must match edge rows");

  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto atom_c = atom_feat.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto weight_c = weight.contiguous();
  auto bias_c = has_bias ? bias.contiguous() : weight_c.new_empty({0});

  const int64_t rows = edge_c.size(0);
  auto core_act = edge_c.new_empty({rows, kDim});
  auto gate_act = edge_c.new_empty({rows, kDim});
  auto core_raw = edge_c.new_empty({rows, kDim});
  auto gate_raw = edge_c.new_empty({rows, kDim});
  if (rows == 0) {
    return {core_act, gate_act, core_raw, gate_raw};
  }

  const dim3 block(kForwardTileN, kForwardTileM);
  const dim3 grid((kOutDim + kForwardTileN - 1) / kForwardTileN,
                  static_cast<unsigned int>((rows + kForwardTileM - 1) / kForwardTileM));
  refine_line_first_silu_forward_plain_kernel<true><<<grid, block>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      atom_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      weight_c.data_ptr<float>(),
      bias_c.data_ptr<float>(),
      has_bias,
      core_act.data_ptr<float>(),
      gate_act.data_ptr<float>(),
      core_raw.data_ptr<float>(),
      gate_raw.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {core_act, gate_act, core_raw, gate_raw};
}

std::vector<torch::Tensor> refine_line_first_silu_forward_acts(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    bool has_bias) {
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && atom_feat.is_cuda() && atom_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && weight.is_cuda(),
              "refine_line_first_silu_forward_acts: all tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32 &&
                  atom_feat.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32,
              "refine_line_first_silu_forward_acts: feature and weight tensors must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_silu_forward_acts: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && atom_feat.dim() == 2 &&
                  node_feat.size(1) == kDim && edge_feat.size(1) == kDim && atom_feat.size(1) == kDim,
              "refine_line_first_silu_forward_acts: features must be [rows, 128]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kOutDim && weight.size(1) == kInDim,
              "refine_line_first_silu_forward_acts: weight must be [256, 512]");
  TORCH_CHECK(!has_bias || (bias.is_cuda() && bias.scalar_type() == torch::kFloat32 && bias.numel() == kOutDim),
              "refine_line_first_silu_forward_acts: bias must be CUDA float32 [256]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == edge_feat.size(0) && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "refine_line_first_silu_forward_acts: index sizes must match edge rows");

  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto atom_c = atom_feat.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto weight_c = weight.contiguous();
  auto bias_c = has_bias ? bias.contiguous() : weight_c.new_empty({0});

  const int64_t rows = edge_c.size(0);
  auto core_act = edge_c.new_empty({rows, kDim});
  auto gate_act = edge_c.new_empty({rows, kDim});
  if (rows == 0) {
    return {core_act, gate_act};
  }

  const dim3 block(kForwardTileN, kForwardTileM);
  const dim3 grid((kOutDim + kForwardTileN - 1) / kForwardTileN,
                  static_cast<unsigned int>((rows + kForwardTileM - 1) / kForwardTileM));
  refine_line_first_silu_forward_plain_kernel<false><<<grid, block>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      atom_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      weight_c.data_ptr<float>(),
      bias_c.data_ptr<float>(),
      has_bias,
      core_act.data_ptr<float>(),
      gate_act.data_ptr<float>(),
      nullptr,
      nullptr,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {core_act, gate_act};
}

torch::Tensor refine_line_first_tail_w8a8_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
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
    double eps,
    bool use_parallel_tail,
    bool use_dynamic_activation_scale,
    bool use_tile_scale_partials) {
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && atom_feat.is_cuda() && atom_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && first_weight.is_cuda(),
              "refine_line_first_tail_w8a8_forward: first-stage tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() && core_weight_scale.is_cuda() &&
                  gate_weight_scale.is_cuda() && core_activation_scale.is_cuda() && gate_activation_scale.is_cuda() &&
                  core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() &&
                  gate_norm_bias.is_cuda(),
              "refine_line_first_tail_w8a8_forward: tail tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32 &&
                  atom_feat.scalar_type() == torch::kFloat32 && first_weight.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_forward: first-stage features/weight must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_first_tail_w8a8_forward: tail q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_activation_scale.scalar_type() == torch::kFloat32 &&
                  gate_activation_scale.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_forward: tail scales must be float32");
  TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_forward: norm params must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_tail_w8a8_forward: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && atom_feat.dim() == 2 &&
                  node_feat.size(1) == kDim && edge_feat.size(1) == kDim && atom_feat.size(1) == kDim,
              "refine_line_first_tail_w8a8_forward: features must be [rows, 128]");
  TORCH_CHECK(first_weight.dim() == 2 && first_weight.size(0) == kOutDim && first_weight.size(1) == kInDim,
              "refine_line_first_tail_w8a8_forward: first weight must be [256, 512]");
  TORCH_CHECK(!first_has_bias ||
                  (first_bias.is_cuda() && first_bias.scalar_type() == torch::kFloat32 && first_bias.numel() == kOutDim),
              "refine_line_first_tail_w8a8_forward: first bias must be CUDA float32 [256]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == edge_feat.size(0) && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "refine_line_first_tail_w8a8_forward: index sizes must match edge rows");
  TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2 &&
                  core_q_weight.size(0) == kDim && gate_q_weight.size(0) == kDim &&
                  core_q_weight.size(1) == kDim && gate_q_weight.size(1) == kDim,
              "refine_line_first_tail_w8a8_forward: tail q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim &&
                  core_activation_scale.numel() == 1 && gate_activation_scale.numel() == 1 &&
                  core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "refine_line_first_tail_w8a8_forward: tail scale/norm sizes are invalid");
  if (core_has_bias) {
    TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.numel() == kDim,
                "refine_line_first_tail_w8a8_forward: invalid core bias");
  }
  if (gate_has_bias) {
    TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.numel() == kDim,
                "refine_line_first_tail_w8a8_forward: invalid gate bias");
  }

  const c10::cuda::CUDAGuard device_guard(edge_feat.device());
  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto atom_c = atom_feat.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto first_bias_c = first_has_bias ? first_bias.contiguous() : first_weight_c.new_empty({0});
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();
  auto core_activation_scale_c = core_activation_scale.reshape({1}).contiguous();
  auto gate_activation_scale_c = gate_activation_scale.reshape({1}).contiguous();
  auto core_bias_c = core_has_bias ? core_bias.contiguous() : edge_c.new_empty({0});
  auto gate_bias_c = gate_has_bias ? gate_bias.contiguous() : edge_c.new_empty({0});
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();

  const int64_t rows = edge_c.size(0);
  auto core_act = edge_c.new_empty({rows, kDim});
  auto gate_act = edge_c.new_empty({rows, kDim});
  auto output = edge_c.new_empty({rows, kDim});
  if (rows == 0) {
    return output;
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 first_block(kForwardTileN, kForwardTileM);
  const dim3 first_grid((kOutDim + kForwardTileN - 1) / kForwardTileN,
                        static_cast<unsigned int>((rows + kForwardTileM - 1) / kForwardTileM));
  torch::Tensor dynamic_scales;
  torch::Tensor dynamic_scale_partials;
  float* dynamic_scale_partials_ptr = nullptr;
  int dynamic_scale_partial_blocks = 0;
  if (use_dynamic_activation_scale && use_tile_scale_partials) {
    const int64_t first_total_blocks =
        static_cast<int64_t>(first_grid.x) * static_cast<int64_t>(first_grid.y);
    TORCH_CHECK(first_total_blocks <= std::numeric_limits<int>::max(),
                "refine_line_first_tail_w8a8_forward: too many first-stage blocks for dynamic scale");
    dynamic_scale_partial_blocks = static_cast<int>(first_total_blocks);
    dynamic_scales = edge_c.new_empty({2});
    dynamic_scale_partials = edge_c.new_empty({2, dynamic_scale_partial_blocks});
    dynamic_scale_partials_ptr = dynamic_scale_partials.data_ptr<float>();
  }
  if (use_dynamic_activation_scale && use_tile_scale_partials) {
    refine_line_first_silu_forward_kernel<false, true><<<first_grid, first_block, 0, stream>>>(
        node_c.data_ptr<float>(),
        edge_c.data_ptr<float>(),
        atom_c.data_ptr<float>(),
        atom_index_c.data_ptr<int64_t>(),
        source_c.data_ptr<int64_t>(),
        target_c.data_ptr<int64_t>(),
        first_weight_c.data_ptr<float>(),
        first_bias_c.data_ptr<float>(),
        first_has_bias,
        core_act.data_ptr<float>(),
        gate_act.data_ptr<float>(),
        nullptr,
        nullptr,
        dynamic_scale_partials_ptr,
        rows);
  } else {
    refine_line_first_silu_forward_plain_kernel<false><<<first_grid, first_block, 0, stream>>>(
        node_c.data_ptr<float>(),
        edge_c.data_ptr<float>(),
        atom_c.data_ptr<float>(),
        atom_index_c.data_ptr<int64_t>(),
        source_c.data_ptr<int64_t>(),
        target_c.data_ptr<int64_t>(),
        first_weight_c.data_ptr<float>(),
        first_bias_c.data_ptr<float>(),
        first_has_bias,
        core_act.data_ptr<float>(),
        gate_act.data_ptr<float>(),
        nullptr,
        nullptr,
        rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const float* core_activation_scale_ptr = core_activation_scale_c.data_ptr<float>();
  const float* gate_activation_scale_ptr = gate_activation_scale_c.data_ptr<float>();
  if (use_dynamic_activation_scale) {
    if (!use_tile_scale_partials) {
      const int64_t numel = rows * kDim;
      const int partial_blocks =
          static_cast<int>(std::min<int64_t>((numel + kScaleBlock - 1) / kScaleBlock, 65535));
      dynamic_scales = edge_c.new_empty({2});
      dynamic_scale_partials = edge_c.new_empty({2, partial_blocks});
      dynamic_scale_partial_blocks = partial_blocks;
      refine_line_dual_absmax_partial_kernel<<<partial_blocks, kScaleBlock, 0, stream>>>(
          core_act.data_ptr<float>(),
          gate_act.data_ptr<float>(),
          dynamic_scale_partials.data_ptr<float>(),
          numel);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    refine_line_dual_absmax_scale_finalize_kernel<<<1, kScaleBlock, 0, stream>>>(
        dynamic_scale_partials.data_ptr<float>(),
        dynamic_scales.data_ptr<float>(),
        dynamic_scale_partial_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    core_activation_scale_ptr = dynamic_scales.data_ptr<float>();
    gate_activation_scale_ptr = dynamic_scales.data_ptr<float>() + 1;
  }

  if (use_parallel_tail) {
    launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(
        core_act.data_ptr<float>(),
        gate_act.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(),
        gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(),
        gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_ptr,
        gate_activation_scale_ptr,
        core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
        core_norm_weight_c.data_ptr<float>(),
        core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(),
        gate_norm_bias_c.data_ptr<float>(),
        nullptr,
        nullptr,
        output.data_ptr<float>(),
        rows,
        kDim,
        kDim,
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        stream);
  } else {
    launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(
        core_act.data_ptr<float>(),
        gate_act.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(),
        gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(),
        gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_ptr,
        gate_activation_scale_ptr,
        core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
        core_norm_weight_c.data_ptr<float>(),
        core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(),
        gate_norm_bias_c.data_ptr<float>(),
        nullptr,
        nullptr,
        output.data_ptr<float>(),
        rows,
        kDim,
        kDim,
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> refine_line_first_tail_w8a8_forward_with_pre(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
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
    double eps,
    bool use_parallel_tail) {
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && atom_feat.is_cuda() && atom_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && first_weight.is_cuda(),
              "refine_line_first_tail_w8a8_forward_with_pre: first-stage tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() && core_weight_scale.is_cuda() &&
                  gate_weight_scale.is_cuda() && core_activation_scale.is_cuda() && gate_activation_scale.is_cuda() &&
                  core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() &&
                  gate_norm_bias.is_cuda(),
              "refine_line_first_tail_w8a8_forward_with_pre: tail tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32 &&
                  atom_feat.scalar_type() == torch::kFloat32 && first_weight.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_forward_with_pre: first-stage features/weight must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_first_tail_w8a8_forward_with_pre: tail q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_activation_scale.scalar_type() == torch::kFloat32 &&
                  gate_activation_scale.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_forward_with_pre: tail scales must be float32");
  TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_forward_with_pre: norm params must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_tail_w8a8_forward_with_pre: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && atom_feat.dim() == 2 &&
                  node_feat.size(1) == kDim && edge_feat.size(1) == kDim && atom_feat.size(1) == kDim,
              "refine_line_first_tail_w8a8_forward_with_pre: features must be [rows, 128]");
  TORCH_CHECK(first_weight.dim() == 2 && first_weight.size(0) == kOutDim && first_weight.size(1) == kInDim,
              "refine_line_first_tail_w8a8_forward_with_pre: first weight must be [256, 512]");
  TORCH_CHECK(!first_has_bias ||
                  (first_bias.is_cuda() && first_bias.scalar_type() == torch::kFloat32 && first_bias.numel() == kOutDim),
              "refine_line_first_tail_w8a8_forward_with_pre: first bias must be CUDA float32 [256]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == edge_feat.size(0) && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "refine_line_first_tail_w8a8_forward_with_pre: index sizes must match edge rows");

  const c10::cuda::CUDAGuard device_guard(edge_feat.device());
  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto atom_c = atom_feat.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto first_bias_c = first_has_bias ? first_bias.contiguous() : first_weight_c.new_empty({0});
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();
  auto core_activation_scale_c = core_activation_scale.reshape({1}).contiguous();
  auto gate_activation_scale_c = gate_activation_scale.reshape({1}).contiguous();
  auto core_bias_c = core_has_bias ? core_bias.contiguous() : edge_c.new_empty({0});
  auto gate_bias_c = gate_has_bias ? gate_bias.contiguous() : edge_c.new_empty({0});
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();

  const int64_t rows = edge_c.size(0);
  auto core_act = edge_c.new_empty({rows, kDim});
  auto gate_act = edge_c.new_empty({rows, kDim});
  auto core_raw = edge_c.new_empty({rows, kDim});
  auto gate_raw = edge_c.new_empty({rows, kDim});
  auto core_pre = edge_c.new_empty({rows, kDim});
  auto gate_pre = edge_c.new_empty({rows, kDim});
  auto output = edge_c.new_empty({rows, kDim});
  if (rows == 0) {
    return {output, core_raw, gate_raw, core_pre, gate_pre};
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 first_block(kForwardTileN, kForwardTileM);
  const dim3 first_grid((kOutDim + kForwardTileN - 1) / kForwardTileN,
                        static_cast<unsigned int>((rows + kForwardTileM - 1) / kForwardTileM));
  refine_line_first_silu_forward_plain_kernel<true><<<first_grid, first_block, 0, stream>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      atom_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      first_weight_c.data_ptr<float>(),
      first_bias_c.data_ptr<float>(),
      first_has_bias,
      core_act.data_ptr<float>(),
      gate_act.data_ptr<float>(),
      core_raw.data_ptr<float>(),
      gate_raw.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (use_parallel_tail) {
    launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(
        core_act.data_ptr<float>(),
        gate_act.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(),
        gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(),
        gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_c.data_ptr<float>(),
        gate_activation_scale_c.data_ptr<float>(),
        core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
        core_norm_weight_c.data_ptr<float>(),
        core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(),
        gate_norm_bias_c.data_ptr<float>(),
        core_pre.data_ptr<float>(),
        gate_pre.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        kDim,
        kDim,
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        stream);
  } else {
    launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(
        core_act.data_ptr<float>(),
        gate_act.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(),
        gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(),
        gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_c.data_ptr<float>(),
        gate_activation_scale_c.data_ptr<float>(),
        core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
        core_norm_weight_c.data_ptr<float>(),
        core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(),
        gate_norm_bias_c.data_ptr<float>(),
        core_pre.data_ptr<float>(),
        gate_pre.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        kDim,
        kDim,
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, core_raw, gate_raw, core_pre, gate_pre};
}

std::vector<torch::Tensor> refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &base_envelope,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
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
    double eps,
    bool use_parallel_tail,
    int64_t num_nodes) {
  TORCH_CHECK(use_parallel_tail,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: only parallel tail is implemented");
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && atom_feat.is_cuda() && base_envelope.is_cuda() &&
                  atom_index.is_cuda() && source_index.is_cuda() && target_index.is_cuda() && first_weight.is_cuda(),
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: first-stage tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() && core_weight_scale.is_cuda() &&
                  gate_weight_scale.is_cuda() && core_activation_scale.is_cuda() && gate_activation_scale.is_cuda() &&
                  core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() &&
                  gate_norm_bias.is_cuda(),
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: tail tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32 &&
                  atom_feat.scalar_type() == torch::kFloat32 && base_envelope.scalar_type() == torch::kFloat32 &&
                  first_weight.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: features/weight must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: tail q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_activation_scale.scalar_type() == torch::kFloat32 &&
                  gate_activation_scale.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: tail scales must be float32");
  TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: norm params must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && atom_feat.dim() == 2 && base_envelope.dim() == 2 &&
                  node_feat.size(1) == kDim && edge_feat.size(1) == kDim && atom_feat.size(1) == kDim &&
                  base_envelope.size(1) == kDim,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: features must be [rows, 128]");
  TORCH_CHECK(base_envelope.size(0) == num_nodes,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: base rows must match num_nodes");
  TORCH_CHECK(first_weight.dim() == 2 && first_weight.size(0) == kOutDim && first_weight.size(1) == kInDim,
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: first weight must be [256, 512]");
  TORCH_CHECK(!first_has_bias ||
                  (first_bias.is_cuda() && first_bias.scalar_type() == torch::kFloat32 && first_bias.numel() == kOutDim),
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: first bias must be CUDA float32 [256]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == edge_feat.size(0) && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre: index sizes must match edge rows");

  const c10::cuda::CUDAGuard device_guard(edge_feat.device());
  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto atom_c = atom_feat.contiguous();
  auto base_c = base_envelope.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto first_bias_c = first_has_bias ? first_bias.contiguous() : first_weight_c.new_empty({0});
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();
  auto core_activation_scale_c = core_activation_scale.reshape({1}).contiguous();
  auto gate_activation_scale_c = gate_activation_scale.reshape({1}).contiguous();
  auto core_bias_c = core_has_bias ? core_bias.contiguous() : edge_c.new_empty({0});
  auto gate_bias_c = gate_has_bias ? gate_bias.contiguous() : edge_c.new_empty({0});
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();

  const int64_t rows = edge_c.size(0);
  auto core_act = edge_c.new_empty({rows, kDim});
  auto gate_act = edge_c.new_empty({rows, kDim});
  auto core_raw = edge_c.new_empty({rows, kDim});
  auto gate_raw = edge_c.new_empty({rows, kDim});
  auto core_pre = edge_c.new_empty({rows, kDim});
  auto gate_pre = edge_c.new_empty({rows, kDim});
  auto nonlinear = edge_c.new_empty({rows, kDim});
  auto refine_node = edge_c.new_zeros({num_nodes, kDim});
  if (rows == 0) {
    return {refine_node, nonlinear, core_raw, gate_raw, core_pre, gate_pre};
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 first_block(kForwardTileN, kForwardTileM);
  const dim3 first_grid((kOutDim + kForwardTileN - 1) / kForwardTileN,
                        static_cast<unsigned int>((rows + kForwardTileM - 1) / kForwardTileM));
  refine_line_first_silu_forward_plain_kernel<true><<<first_grid, first_block, 0, stream>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      atom_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      first_weight_c.data_ptr<float>(),
      first_bias_c.data_ptr<float>(),
      first_has_bias,
      core_act.data_ptr<float>(),
      gate_act.data_ptr<float>(),
      core_raw.data_ptr<float>(),
      gate_raw.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_smooth_kernel(
      core_act.data_ptr<float>(),
      gate_act.data_ptr<float>(),
      core_q_weight_c.data_ptr<int8_t>(),
      gate_q_weight_c.data_ptr<int8_t>(),
      core_weight_scale_c.data_ptr<float>(),
      gate_weight_scale_c.data_ptr<float>(),
      core_activation_scale_c.data_ptr<float>(),
      gate_activation_scale_c.data_ptr<float>(),
      core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
      gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
      core_norm_weight_c.data_ptr<float>(),
      core_norm_bias_c.data_ptr<float>(),
      gate_norm_weight_c.data_ptr<float>(),
      gate_norm_bias_c.data_ptr<float>(),
      base_c.data_ptr<float>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      core_pre.data_ptr<float>(),
      gate_pre.data_ptr<float>(),
      nonlinear.data_ptr<float>(),
      refine_node.data_ptr<float>(),
      rows,
      kDim,
      kDim,
      core_has_bias,
      gate_has_bias,
      static_cast<float>(eps),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {refine_node, nonlinear, core_raw, gate_raw, core_pre, gate_pre};
}

torch::Tensor refine_line_first_tail_w8a8_packed_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
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
    double eps,
    bool use_parallel_tail,
    bool use_dynamic_activation_scale,
    bool use_tile_scale_partials) {
  (void)use_parallel_tail;
  (void)use_tile_scale_partials;
  TORCH_CHECK(node_feat.is_cuda() && edge_feat.is_cuda() && atom_feat.is_cuda() && atom_index.is_cuda() &&
                  source_index.is_cuda() && target_index.is_cuda() && first_weight.is_cuda(),
              "refine_line_first_tail_w8a8_packed_forward: first-stage tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() && core_weight_scale.is_cuda() &&
                  gate_weight_scale.is_cuda() && core_activation_scale.is_cuda() && gate_activation_scale.is_cuda() &&
                  core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() &&
                  gate_norm_bias.is_cuda(),
              "refine_line_first_tail_w8a8_packed_forward: tail tensors must be CUDA");
  TORCH_CHECK(node_feat.scalar_type() == torch::kFloat32 && edge_feat.scalar_type() == torch::kFloat32 &&
                  atom_feat.scalar_type() == torch::kFloat32 && first_weight.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_packed_forward: first-stage features/weight must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "refine_line_first_tail_w8a8_packed_forward: tail q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_activation_scale.scalar_type() == torch::kFloat32 &&
                  gate_activation_scale.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_packed_forward: tail scales must be float32");
  TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
              "refine_line_first_tail_w8a8_packed_forward: norm params must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_tail_w8a8_packed_forward: indices must be int64");
  TORCH_CHECK(node_feat.dim() == 2 && edge_feat.dim() == 2 && atom_feat.dim() == 2 &&
                  node_feat.size(1) == kDim && edge_feat.size(1) == kDim && atom_feat.size(1) == kDim,
              "refine_line_first_tail_w8a8_packed_forward: features must be [rows, 128]");
  TORCH_CHECK(first_weight.dim() == 2 && first_weight.size(0) == kOutDim && first_weight.size(1) == kInDim,
              "refine_line_first_tail_w8a8_packed_forward: first weight must be [256, 512]");
  TORCH_CHECK(!first_has_bias ||
                  (first_bias.is_cuda() && first_bias.scalar_type() == torch::kFloat32 && first_bias.numel() == kOutDim),
              "refine_line_first_tail_w8a8_packed_forward: first bias must be CUDA float32 [256]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == edge_feat.size(0) && source_index.size(0) == edge_feat.size(0) &&
                  target_index.size(0) == edge_feat.size(0),
              "refine_line_first_tail_w8a8_packed_forward: index sizes must match edge rows");
  TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2 &&
                  core_q_weight.size(0) == kDim && gate_q_weight.size(0) == kDim &&
                  core_q_weight.size(1) == kDim && gate_q_weight.size(1) == kDim,
              "refine_line_first_tail_w8a8_packed_forward: tail q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim &&
                  core_activation_scale.numel() == 1 && gate_activation_scale.numel() == 1 &&
                  core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "refine_line_first_tail_w8a8_packed_forward: tail scale/norm sizes are invalid");
  if (core_has_bias) {
    TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.numel() == kDim,
                "refine_line_first_tail_w8a8_packed_forward: invalid core bias");
  }
  if (gate_has_bias) {
    TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.numel() == kDim,
                "refine_line_first_tail_w8a8_packed_forward: invalid gate bias");
  }

  const c10::cuda::CUDAGuard device_guard(edge_feat.device());
  auto node_c = node_feat.contiguous();
  auto edge_c = edge_feat.contiguous();
  auto atom_c = atom_feat.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();
  auto first_weight_c = first_weight.contiguous();
  auto first_bias_c = first_has_bias ? first_bias.contiguous() : first_weight_c.new_empty({0});
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();
  auto core_activation_scale_c = core_activation_scale.reshape({1}).contiguous();
  auto gate_activation_scale_c = gate_activation_scale.reshape({1}).contiguous();
  auto core_bias_c = core_has_bias ? core_bias.contiguous() : edge_c.new_empty({0});
  auto gate_bias_c = gate_has_bias ? gate_bias.contiguous() : edge_c.new_empty({0});
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();

  const int64_t rows = edge_c.size(0);
  auto output = edge_c.new_empty({rows, kDim});
  if (rows == 0) {
    return output;
  }
  auto core_q_act = torch::empty({rows, kDim}, edge_c.options().dtype(torch::kInt8));
  auto gate_q_act = torch::empty({rows, kDim}, edge_c.options().dtype(torch::kInt8));

  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 first_block(kForwardTileN, kForwardTileM);
  const dim3 first_grid((kOutDim + kForwardTileN - 1) / kForwardTileN,
                        static_cast<unsigned int>((rows + kForwardTileM - 1) / kForwardTileM));
  torch::Tensor dynamic_scales;
  torch::Tensor dynamic_scale_partials;
  const float* core_activation_scale_ptr = core_activation_scale_c.data_ptr<float>();
  const float* gate_activation_scale_ptr = gate_activation_scale_c.data_ptr<float>();
  if (use_dynamic_activation_scale) {
    const int64_t first_total_blocks =
        static_cast<int64_t>(first_grid.x) * static_cast<int64_t>(first_grid.y);
    TORCH_CHECK(first_total_blocks <= std::numeric_limits<int>::max(),
                "refine_line_first_tail_w8a8_packed_forward: too many first-stage blocks for dynamic scale");
    const int dynamic_scale_partial_blocks = static_cast<int>(first_total_blocks);
    dynamic_scales = edge_c.new_empty({2});
    dynamic_scale_partials = edge_c.new_empty({2, dynamic_scale_partial_blocks});
    refine_line_first_silu_absmax_partial_kernel<<<first_grid, first_block, 0, stream>>>(
        node_c.data_ptr<float>(),
        edge_c.data_ptr<float>(),
        atom_c.data_ptr<float>(),
        atom_index_c.data_ptr<int64_t>(),
        source_c.data_ptr<int64_t>(),
        target_c.data_ptr<int64_t>(),
        first_weight_c.data_ptr<float>(),
        first_bias_c.data_ptr<float>(),
        first_has_bias,
        dynamic_scale_partials.data_ptr<float>(),
        rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    refine_line_dual_absmax_scale_finalize_kernel<<<1, kScaleBlock, 0, stream>>>(
        dynamic_scale_partials.data_ptr<float>(),
        dynamic_scales.data_ptr<float>(),
        dynamic_scale_partial_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    core_activation_scale_ptr = dynamic_scales.data_ptr<float>();
    gate_activation_scale_ptr = dynamic_scales.data_ptr<float>() + 1;
  }
  refine_line_first_silu_quant_forward_kernel<<<first_grid, first_block, 0, stream>>>(
      node_c.data_ptr<float>(),
      edge_c.data_ptr<float>(),
      atom_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      first_weight_c.data_ptr<float>(),
      first_bias_c.data_ptr<float>(),
      first_has_bias,
      core_activation_scale_ptr,
      gate_activation_scale_ptr,
      core_q_act.data_ptr<int8_t>(),
      gate_q_act.data_ptr<int8_t>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 tail_block(kTailN128Tiles * 2 * 32);
  const dim3 tail_grid(static_cast<unsigned int>((rows + kWmmaM - 1) / kWmmaM));
  refine_line_w8a8_packed_gated_tail_n128_parallel_kernel<<<tail_grid, tail_block, 0, stream>>>(
      core_q_act.data_ptr<int8_t>(),
      gate_q_act.data_ptr<int8_t>(),
      core_q_weight_c.data_ptr<int8_t>(),
      gate_q_weight_c.data_ptr<int8_t>(),
      core_weight_scale_c.data_ptr<float>(),
      gate_weight_scale_c.data_ptr<float>(),
      core_activation_scale_ptr,
      gate_activation_scale_ptr,
      core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
      gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
      core_norm_weight_c.data_ptr<float>(),
      core_norm_bias_c.data_ptr<float>(),
      gate_norm_weight_c.data_ptr<float>(),
      gate_norm_bias_c.data_ptr<float>(),
      output.data_ptr<float>(),
      rows,
      core_has_bias,
      gate_has_bias,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> refine_line_first_silu_backward(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows) {
  TORCH_CHECK(grad_core.is_cuda() && grad_gate.is_cuda() && core_raw.is_cuda() && gate_raw.is_cuda() &&
                  weight.is_cuda() && atom_index.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "refine_line_first_silu_backward: all tensors must be CUDA");
  TORCH_CHECK(grad_core.scalar_type() == torch::kFloat32 && grad_gate.scalar_type() == torch::kFloat32 &&
                  core_raw.scalar_type() == torch::kFloat32 && gate_raw.scalar_type() == torch::kFloat32 &&
                  weight.scalar_type() == torch::kFloat32,
              "refine_line_first_silu_backward: feature and weight tensors must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_silu_backward: indices must be int64");
  TORCH_CHECK(grad_core.dim() == 2 && grad_gate.dim() == 2 && core_raw.dim() == 2 && gate_raw.dim() == 2 &&
                  grad_core.size(1) == kDim && grad_gate.size(1) == kDim && core_raw.size(1) == kDim &&
                  gate_raw.size(1) == kDim && grad_core.size(0) == grad_gate.size(0) &&
                  grad_core.size(0) == core_raw.size(0) && grad_core.size(0) == gate_raw.size(0),
              "refine_line_first_silu_backward: grad/raw tensors must be matching [rows, 128]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kOutDim && weight.size(1) == kInDim,
              "refine_line_first_silu_backward: weight must be [256, 512]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == grad_core.size(0) && source_index.size(0) == grad_core.size(0) &&
                  target_index.size(0) == grad_core.size(0),
              "refine_line_first_silu_backward: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows == grad_core.size(0) && atom_rows >= 0,
              "refine_line_first_silu_backward: invalid output rows");

  auto grad_core_c = grad_core.contiguous();
  auto grad_gate_c = grad_gate.contiguous();
  auto core_raw_c = core_raw.contiguous();
  auto gate_raw_c = gate_raw.contiguous();
  auto weight_c = weight.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();

  const int64_t rows = grad_core_c.size(0);
  auto grad_node = grad_core_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_core_c.new_empty({edge_rows, kDim});
  auto grad_atom = grad_core_c.new_zeros({atom_rows, kDim});
  if (rows == 0) {
    return {grad_node, grad_edge.zero_(), grad_atom};
  }

  const dim3 block(16, 16);
  const dim3 grid((kInDim + kProjectTile32N - 1) / kProjectTile32N,
                  static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  refine_line_first_silu_backward_kernel<<<grid, block>>>(
      grad_core_c.data_ptr<float>(),
      grad_gate_c.data_ptr<float>(),
      core_raw_c.data_ptr<float>(),
      gate_raw_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      grad_atom.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge, grad_atom};
}

std::vector<torch::Tensor> refine_line_first_silu_backward_packed(
    const torch::Tensor &grad_act,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows) {
  TORCH_CHECK(grad_act.is_cuda() && core_raw.is_cuda() && gate_raw.is_cuda() &&
                  weight.is_cuda() && atom_index.is_cuda() && source_index.is_cuda() && target_index.is_cuda(),
              "refine_line_first_silu_backward_packed: all tensors must be CUDA");
  TORCH_CHECK(grad_act.scalar_type() == torch::kFloat32 &&
                  core_raw.scalar_type() == torch::kFloat32 && gate_raw.scalar_type() == torch::kFloat32 &&
                  weight.scalar_type() == torch::kFloat32,
              "refine_line_first_silu_backward_packed: feature and weight tensors must be float32");
  TORCH_CHECK(atom_index.scalar_type() == torch::kInt64 && source_index.scalar_type() == torch::kInt64 &&
                  target_index.scalar_type() == torch::kInt64,
              "refine_line_first_silu_backward_packed: indices must be int64");
  TORCH_CHECK(grad_act.dim() == 2 && core_raw.dim() == 2 && gate_raw.dim() == 2 &&
                  grad_act.size(1) == kOutDim && core_raw.size(1) == kDim && gate_raw.size(1) == kDim &&
                  grad_act.size(0) == core_raw.size(0) && grad_act.size(0) == gate_raw.size(0),
              "refine_line_first_silu_backward_packed: grad_act must be [rows, 256] and raw tensors [rows, 128]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kOutDim && weight.size(1) == kInDim,
              "refine_line_first_silu_backward_packed: weight must be [256, 512]");
  TORCH_CHECK(atom_index.dim() == 1 && source_index.dim() == 1 && target_index.dim() == 1 &&
                  atom_index.size(0) == grad_act.size(0) && source_index.size(0) == grad_act.size(0) &&
                  target_index.size(0) == grad_act.size(0),
              "refine_line_first_silu_backward_packed: index sizes must match rows");
  TORCH_CHECK(node_rows >= 0 && edge_rows == grad_act.size(0) && atom_rows >= 0,
              "refine_line_first_silu_backward_packed: invalid output rows");

  auto grad_act_c = grad_act.contiguous();
  auto core_raw_c = core_raw.contiguous();
  auto gate_raw_c = gate_raw.contiguous();
  auto weight_c = weight.contiguous();
  auto atom_index_c = atom_index.contiguous();
  auto source_c = source_index.contiguous();
  auto target_c = target_index.contiguous();

  const int64_t rows = grad_act_c.size(0);
  auto grad_node = grad_act_c.new_zeros({node_rows, kDim});
  auto grad_edge = grad_act_c.new_empty({edge_rows, kDim});
  auto grad_atom = grad_act_c.new_zeros({atom_rows, kDim});
  if (rows == 0) {
    return {grad_node, grad_edge.zero_(), grad_atom};
  }

  const dim3 block(16, 16);
  const dim3 grid((kInDim + kProjectTile32N - 1) / kProjectTile32N,
                  static_cast<unsigned int>((rows + kProjectTile32M - 1) / kProjectTile32M));
  refine_line_first_silu_backward_packed_kernel<<<grid, block>>>(
      grad_act_c.data_ptr<float>(),
      core_raw_c.data_ptr<float>(),
      gate_raw_c.data_ptr<float>(),
      weight_c.data_ptr<float>(),
      atom_index_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      target_c.data_ptr<int64_t>(),
      grad_node.data_ptr<float>(),
      grad_edge.data_ptr<float>(),
      grad_atom.data_ptr<float>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_node, grad_edge, grad_atom};
}
