#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace {

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int FIXED_N128_TILES = 8;

__device__ __forceinline__ int8_t quantize_s8(float x, float scale) {
    float scaled = x / scale;
    int q = static_cast<int>(rintf(scaled));
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    return static_cast<int8_t>(q);
}

__device__ __forceinline__ float half_warp_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 8, 16);
    value += __shfl_down_sync(0xffffffffu, value, 4, 16);
    value += __shfl_down_sync(0xffffffffu, value, 2, 16);
    value += __shfl_down_sync(0xffffffffu, value, 1, 16);
    return value;
}

__global__ void quant_linear_w8a8_wmma_kernel(const float *__restrict__ input,
                                              const int8_t *__restrict__ q_weight,
                                              const float *__restrict__ weight_scale,
                                              const float *__restrict__ activation_scale,
                                              const float *__restrict__ bias,
                                              float *__restrict__ output,
                                              int64_t rows,
                                              int64_t out_features,
                                              int64_t in_features,
                                              bool has_bias) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int tile_m = static_cast<int>(blockIdx.y) * WMMA_M;
    const int tile_n = static_cast<int>(blockIdx.x) * WMMA_N;
    const int lane = threadIdx.x;
    const float act_scale = activation_scale[0];

    __shared__ int8_t a_tile[WMMA_M * WMMA_K];
    __shared__ int acc_tile[WMMA_M * WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
    wmma::fill_fragment(acc_frag, 0);

    for (int k0 = 0; k0 < in_features; k0 += WMMA_K) {
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += warpSize) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int row = tile_m + local_m;
            const int k = k0 + local_k;
            float x = 0.0f;
            if (row < rows && k < in_features) {
                x = input[static_cast<int64_t>(row) * in_features + k];
            }
            a_tile[idx] = quantize_s8(x, act_scale);
        }
        __syncwarp();

        const int8_t *b_tile = q_weight + static_cast<int64_t>(tile_n) * in_features + k0;
        wmma::load_matrix_sync(a_frag, a_tile, WMMA_K);
        wmma::load_matrix_sync(b_frag, b_tile, in_features);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(acc_tile, acc_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += warpSize) {
        const int local_m = idx / WMMA_N;
        const int local_n = idx % WMMA_N;
        const int row = tile_m + local_m;
        const int col = tile_n + local_n;
        if (row < rows && col < out_features) {
            float out = static_cast<float>(acc_tile[idx]) * act_scale * weight_scale[col];
            if (has_bias) {
                out += bias[col];
            }
            output[static_cast<int64_t>(row) * out_features + col] = out;
        }
    }
#endif
}

__global__ void quant_linear_w8a8_wmma_dual_kernel(const float *__restrict__ core_input,
                                                   const float *__restrict__ gate_input,
                                                   const int8_t *__restrict__ core_q_weight,
                                                   const int8_t *__restrict__ gate_q_weight,
                                                   const float *__restrict__ core_weight_scale,
                                                   const float *__restrict__ gate_weight_scale,
                                                   const float *__restrict__ core_activation_scale,
                                                   const float *__restrict__ gate_activation_scale,
                                                   const float *__restrict__ core_bias,
                                                   const float *__restrict__ gate_bias,
                                                   float *__restrict__ core_output,
                                                   float *__restrict__ gate_output,
                                                   int64_t rows,
                                                   int64_t core_out_features,
                                                   int64_t gate_out_features,
                                                   int64_t core_in_features,
                                                   int64_t gate_in_features,
                                                   bool core_has_bias,
                                                   bool gate_has_bias) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const bool is_gate = blockIdx.z == 1;
    const float *input = is_gate ? gate_input : core_input;
    const int8_t *q_weight = is_gate ? gate_q_weight : core_q_weight;
    const float *weight_scale = is_gate ? gate_weight_scale : core_weight_scale;
    const float *activation_scale = is_gate ? gate_activation_scale : core_activation_scale;
    const float *bias = is_gate ? gate_bias : core_bias;
    float *output = is_gate ? gate_output : core_output;
    const int64_t out_features = is_gate ? gate_out_features : core_out_features;
    const int64_t in_features = is_gate ? gate_in_features : core_in_features;
    const bool has_bias = is_gate ? gate_has_bias : core_has_bias;

    const int tile_m = static_cast<int>(blockIdx.y) * WMMA_M;
    const int tile_n = static_cast<int>(blockIdx.x) * WMMA_N;
    const int lane = threadIdx.x;
    const float act_scale = activation_scale[0];

    __shared__ int8_t a_tile[WMMA_M * WMMA_K];
    __shared__ int acc_tile[WMMA_M * WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
    wmma::fill_fragment(acc_frag, 0);

    for (int k0 = 0; k0 < in_features; k0 += WMMA_K) {
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += warpSize) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int row = tile_m + local_m;
            const int k = k0 + local_k;
            float x = 0.0f;
            if (row < rows && k < in_features) {
                x = input[static_cast<int64_t>(row) * in_features + k];
            }
            a_tile[idx] = quantize_s8(x, act_scale);
        }
        __syncwarp();

        const int8_t *b_tile = q_weight + static_cast<int64_t>(tile_n) * in_features + k0;
        wmma::load_matrix_sync(a_frag, a_tile, WMMA_K);
        wmma::load_matrix_sync(b_frag, b_tile, in_features);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(acc_tile, acc_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += warpSize) {
        const int local_m = idx / WMMA_N;
        const int local_n = idx % WMMA_N;
        const int row = tile_m + local_m;
        const int col = tile_n + local_n;
        if (row < rows && col < out_features) {
            float out = static_cast<float>(acc_tile[idx]) * act_scale * weight_scale[col];
            if (has_bias) {
                out += bias[col];
            }
            output[static_cast<int64_t>(row) * out_features + col] = out;
        }
    }
#endif
}

__global__ void quant_linear_w8a8_grad_input_kernel(const float *__restrict__ grad_output,
                                                    const int8_t *__restrict__ q_weight,
                                                    const float *__restrict__ weight_scale,
                                                    float *__restrict__ grad_input,
                                                    int64_t rows,
                                                    int64_t out_features,
                                                    int64_t in_features) {
    const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int k = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || k >= in_features) {
        return;
    }

    float acc = 0.0f;
    for (int64_t n = 0; n < out_features; ++n) {
        const float w = static_cast<float>(q_weight[n * in_features + k]) * weight_scale[n];
        acc += grad_output[static_cast<int64_t>(row) * out_features + n] * w;
    }
    grad_input[static_cast<int64_t>(row) * in_features + k] = acc;
}

__global__ void quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(
    const float *__restrict__ core_input,
    const float *__restrict__ gate_input,
    const int8_t *__restrict__ core_q_weight,
    const int8_t *__restrict__ gate_q_weight,
    const float *__restrict__ core_weight_scale,
    const float *__restrict__ gate_weight_scale,
    const float *__restrict__ core_activation_scale,
    const float *__restrict__ gate_activation_scale,
    const float *__restrict__ core_bias,
    const float *__restrict__ gate_bias,
    const float *__restrict__ core_norm_weight,
    const float *__restrict__ core_norm_bias,
    const float *__restrict__ gate_norm_weight,
    const float *__restrict__ gate_norm_bias,
    float *__restrict__ output,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int warp_id = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int tile_m = static_cast<int>(blockIdx.x) * WMMA_M;
    const int tile_n = warp_id * WMMA_N;
    const float core_act_scale = core_activation_scale[0];
    const float gate_act_scale = gate_activation_scale[0];

    __shared__ int8_t core_a_tile[WMMA_M * WMMA_K];
    __shared__ int8_t gate_a_tile[WMMA_M * WMMA_K];
    __shared__ int core_acc_tile[FIXED_N128_TILES][WMMA_M * WMMA_N];
    __shared__ int gate_acc_tile[FIXED_N128_TILES][WMMA_M * WMMA_N];
    __shared__ float core_mean[WMMA_M];
    __shared__ float gate_mean[WMMA_M];
    __shared__ float core_rstd[WMMA_M];
    __shared__ float gate_rstd[WMMA_M];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> core_a_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> gate_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> core_b_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> gate_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> core_acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> gate_acc_frag;
    wmma::fill_fragment(core_acc_frag, 0);
    wmma::fill_fragment(gate_acc_frag, 0);

    for (int64_t k0 = 0; k0 < core_in_features; k0 += WMMA_K) {
        for (int idx = threadIdx.x; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int row = tile_m + local_m;
            const int64_t core_k = k0 + local_k;
            float x = 0.0f;
            if (row < rows && core_k < core_in_features) {
                x = core_input[static_cast<int64_t>(row) * core_in_features + core_k];
            }
            core_a_tile[idx] = quantize_s8(x, core_act_scale);
        }
        __syncthreads();

        const int8_t *core_b_tile = core_q_weight + static_cast<int64_t>(tile_n) * core_in_features + k0;
        wmma::load_matrix_sync(core_a_frag, core_a_tile, WMMA_K);
        wmma::load_matrix_sync(core_b_frag, core_b_tile, core_in_features);
        wmma::mma_sync(core_acc_frag, core_a_frag, core_b_frag, core_acc_frag);
        __syncthreads();
    }

    for (int64_t k0 = 0; k0 < gate_in_features; k0 += WMMA_K) {
        for (int idx = threadIdx.x; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int row = tile_m + local_m;
            const int64_t gate_k = k0 + local_k;
            float x = 0.0f;
            if (row < rows && gate_k < gate_in_features) {
                x = gate_input[static_cast<int64_t>(row) * gate_in_features + gate_k];
            }
            gate_a_tile[idx] = quantize_s8(x, gate_act_scale);
        }
        __syncthreads();

        const int8_t *gate_b_tile = gate_q_weight + static_cast<int64_t>(tile_n) * gate_in_features + k0;
        wmma::load_matrix_sync(gate_a_frag, gate_a_tile, WMMA_K);
        wmma::load_matrix_sync(gate_b_frag, gate_b_tile, gate_in_features);
        wmma::mma_sync(gate_acc_frag, gate_a_frag, gate_b_frag, gate_acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(core_acc_tile[warp_id], core_acc_frag, WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(gate_acc_tile[warp_id], gate_acc_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    const int half_warp = lane >> 4;
    const int half_lane = lane & 15;
    const int reduce_row = warp_id * 2 + half_warp;
    if (reduce_row < WMMA_M) {
        float core_sum = 0.0f;
        float gate_sum = 0.0f;
        for (int col = half_lane; col < 128; col += 16) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = reduce_row * WMMA_N + local_n;
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
        core_sum = half_warp_sum(core_sum);
        gate_sum = half_warp_sum(gate_sum);
        float c_mean = core_sum * (1.0f / 128.0f);
        float g_mean = gate_sum * (1.0f / 128.0f);
        c_mean = __shfl_sync(0xffffffffu, c_mean, 0, 16);
        g_mean = __shfl_sync(0xffffffffu, g_mean, 0, 16);
        if (half_lane == 0) {
            core_mean[reduce_row] = c_mean;
            gate_mean[reduce_row] = g_mean;
        }

        float core_var = 0.0f;
        float gate_var = 0.0f;
        for (int col = half_lane; col < 128; col += 16) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = reduce_row * WMMA_N + local_n;
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
        core_var = half_warp_sum(core_var);
        gate_var = half_warp_sum(gate_var);
        if (half_lane == 0) {
            core_rstd[reduce_row] = rsqrtf(core_var * (1.0f / 128.0f) + eps);
            gate_rstd[reduce_row] = rsqrtf(gate_var * (1.0f / 128.0f) + eps);
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < WMMA_M * 128; idx += blockDim.x) {
        const int local_m = idx / 128;
        const int col = idx % 128;
        const int row = tile_m + local_m;
        if (row < rows) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = local_m * WMMA_N + local_n;
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
            output[static_cast<int64_t>(row) * 128 + col] = core * core_sigmoid * gate_sigmoid;
        }
    }
#endif
}

__global__ void quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(
    const float *__restrict__ core_input,
    const float *__restrict__ gate_input,
    const int8_t *__restrict__ core_q_weight,
    const int8_t *__restrict__ gate_q_weight,
    const float *__restrict__ core_weight_scale,
    const float *__restrict__ gate_weight_scale,
    const float *__restrict__ core_activation_scale,
    const float *__restrict__ gate_activation_scale,
    const float *__restrict__ core_bias,
    const float *__restrict__ gate_bias,
    const float *__restrict__ core_norm_weight,
    const float *__restrict__ core_norm_bias,
    const float *__restrict__ gate_norm_weight,
    const float *__restrict__ gate_norm_bias,
    float *__restrict__ output,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int warp_id = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const bool is_gate = warp_id >= FIXED_N128_TILES;
    const int branch_warp = warp_id - (is_gate ? FIXED_N128_TILES : 0);
    const int branch_thread = branch_warp * warpSize + lane;
    const int tile_m = static_cast<int>(blockIdx.x) * WMMA_M;
    const int tile_n = branch_warp * WMMA_N;
    const float core_act_scale = core_activation_scale[0];
    const float gate_act_scale = gate_activation_scale[0];
    const int64_t branch_in_features = is_gate ? gate_in_features : core_in_features;
    const int64_t max_in_features = core_in_features > gate_in_features ? core_in_features : gate_in_features;
    const float *branch_input = is_gate ? gate_input : core_input;
    const int8_t *branch_q_weight = is_gate ? gate_q_weight : core_q_weight;
    const float branch_act_scale = is_gate ? gate_act_scale : core_act_scale;

    __shared__ int8_t core_a_tile[WMMA_M * WMMA_K];
    __shared__ int8_t gate_a_tile[WMMA_M * WMMA_K];
    __shared__ int core_acc_tile[FIXED_N128_TILES][WMMA_M * WMMA_N];
    __shared__ int gate_acc_tile[FIXED_N128_TILES][WMMA_M * WMMA_N];
    __shared__ float core_mean[WMMA_M];
    __shared__ float gate_mean[WMMA_M];
    __shared__ float core_rstd[WMMA_M];
    __shared__ float gate_rstd[WMMA_M];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
    wmma::fill_fragment(acc_frag, 0);

    for (int64_t k0 = 0; k0 < max_in_features; k0 += WMMA_K) {
        int8_t *branch_a_tile = is_gate ? gate_a_tile : core_a_tile;
        if (k0 < branch_in_features) {
            for (int idx = branch_thread; idx < WMMA_M * WMMA_K; idx += FIXED_N128_TILES * warpSize) {
                const int local_m = idx / WMMA_K;
                const int local_k = idx % WMMA_K;
                const int row = tile_m + local_m;
                const int64_t k = k0 + local_k;
                float x = 0.0f;
                if (row < rows && k < branch_in_features) {
                    x = branch_input[static_cast<int64_t>(row) * branch_in_features + k];
                }
                branch_a_tile[idx] = quantize_s8(x, branch_act_scale);
            }
        }
        __syncthreads();

        if (k0 < branch_in_features) {
            const int8_t *b_tile = branch_q_weight + static_cast<int64_t>(tile_n) * branch_in_features + k0;
            wmma::load_matrix_sync(a_frag, branch_a_tile, WMMA_K);
            wmma::load_matrix_sync(b_frag, b_tile, branch_in_features);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    if (is_gate) {
        wmma::store_matrix_sync(gate_acc_tile[branch_warp], acc_frag, WMMA_N, wmma::mem_row_major);
    } else {
        wmma::store_matrix_sync(core_acc_tile[branch_warp], acc_frag, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    const int half_warp = lane >> 4;
    const int half_lane = lane & 15;
    const int reduce_row = warp_id * 2 + half_warp;
    if (reduce_row < WMMA_M) {
        float core_sum = 0.0f;
        float gate_sum = 0.0f;
        for (int col = half_lane; col < 128; col += 16) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = reduce_row * WMMA_N + local_n;
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
        core_sum = half_warp_sum(core_sum);
        gate_sum = half_warp_sum(gate_sum);
        float c_mean = core_sum * (1.0f / 128.0f);
        float g_mean = gate_sum * (1.0f / 128.0f);
        c_mean = __shfl_sync(0xffffffffu, c_mean, 0, 16);
        g_mean = __shfl_sync(0xffffffffu, g_mean, 0, 16);
        if (half_lane == 0) {
            core_mean[reduce_row] = c_mean;
            gate_mean[reduce_row] = g_mean;
        }

        float core_var = 0.0f;
        float gate_var = 0.0f;
        for (int col = half_lane; col < 128; col += 16) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = reduce_row * WMMA_N + local_n;
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
        core_var = half_warp_sum(core_var);
        gate_var = half_warp_sum(gate_var);
        if (half_lane == 0) {
            core_rstd[reduce_row] = rsqrtf(core_var * (1.0f / 128.0f) + eps);
            gate_rstd[reduce_row] = rsqrtf(gate_var * (1.0f / 128.0f) + eps);
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < WMMA_M * 128; idx += blockDim.x) {
        const int local_m = idx / 128;
        const int col = idx % 128;
        const int row = tile_m + local_m;
        if (row < rows) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = local_m * WMMA_N + local_n;
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
            output[static_cast<int64_t>(row) * 128 + col] = core * core_sigmoid * gate_sigmoid;
        }
    }
#endif
}

__global__ void quant_linear_w8a8_wmma_dual_gated_tail_n128_aux_kernel(
    const float *__restrict__ core_input,
    const float *__restrict__ gate_input,
    const int8_t *__restrict__ core_q_weight,
    const int8_t *__restrict__ gate_q_weight,
    const float *__restrict__ core_weight_scale,
    const float *__restrict__ gate_weight_scale,
    const float *__restrict__ core_activation_scale,
    const float *__restrict__ gate_activation_scale,
    const float *__restrict__ core_bias,
    const float *__restrict__ gate_bias,
    const float *__restrict__ core_norm_weight,
    const float *__restrict__ core_norm_bias,
    const float *__restrict__ gate_norm_weight,
    const float *__restrict__ gate_norm_bias,
    float *__restrict__ output,
    float *__restrict__ core_linear,
    float *__restrict__ gate_linear,
    float *__restrict__ core_norm_output,
    float *__restrict__ gate_norm_output,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int warp_id = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const int tile_m = static_cast<int>(blockIdx.x) * WMMA_M;
    const int tile_n = warp_id * WMMA_N;
    const float core_act_scale = core_activation_scale[0];
    const float gate_act_scale = gate_activation_scale[0];

    __shared__ int8_t core_a_tile[WMMA_M * WMMA_K];
    __shared__ int8_t gate_a_tile[WMMA_M * WMMA_K];
    __shared__ int core_acc_tile[FIXED_N128_TILES][WMMA_M * WMMA_N];
    __shared__ int gate_acc_tile[FIXED_N128_TILES][WMMA_M * WMMA_N];
    __shared__ float core_mean[WMMA_M];
    __shared__ float gate_mean[WMMA_M];
    __shared__ float core_rstd[WMMA_M];
    __shared__ float gate_rstd[WMMA_M];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> core_a_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::row_major> gate_a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> core_b_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, signed char, wmma::col_major> gate_b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> core_acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> gate_acc_frag;
    wmma::fill_fragment(core_acc_frag, 0);
    wmma::fill_fragment(gate_acc_frag, 0);

    for (int64_t k0 = 0; k0 < core_in_features; k0 += WMMA_K) {
        for (int idx = threadIdx.x; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int row = tile_m + local_m;
            const int64_t core_k = k0 + local_k;
            float x = 0.0f;
            if (row < rows && core_k < core_in_features) {
                x = core_input[static_cast<int64_t>(row) * core_in_features + core_k];
            }
            core_a_tile[idx] = quantize_s8(x, core_act_scale);
        }
        __syncthreads();
        const int8_t *core_b_tile = core_q_weight + static_cast<int64_t>(tile_n) * core_in_features + k0;
        wmma::load_matrix_sync(core_a_frag, core_a_tile, WMMA_K);
        wmma::load_matrix_sync(core_b_frag, core_b_tile, core_in_features);
        wmma::mma_sync(core_acc_frag, core_a_frag, core_b_frag, core_acc_frag);
        __syncthreads();
    }

    for (int64_t k0 = 0; k0 < gate_in_features; k0 += WMMA_K) {
        for (int idx = threadIdx.x; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const int row = tile_m + local_m;
            const int64_t gate_k = k0 + local_k;
            float x = 0.0f;
            if (row < rows && gate_k < gate_in_features) {
                x = gate_input[static_cast<int64_t>(row) * gate_in_features + gate_k];
            }
            gate_a_tile[idx] = quantize_s8(x, gate_act_scale);
        }
        __syncthreads();
        const int8_t *gate_b_tile = gate_q_weight + static_cast<int64_t>(tile_n) * gate_in_features + k0;
        wmma::load_matrix_sync(gate_a_frag, gate_a_tile, WMMA_K);
        wmma::load_matrix_sync(gate_b_frag, gate_b_tile, gate_in_features);
        wmma::mma_sync(gate_acc_frag, gate_a_frag, gate_b_frag, gate_acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(core_acc_tile[warp_id], core_acc_frag, WMMA_N, wmma::mem_row_major);
    wmma::store_matrix_sync(gate_acc_tile[warp_id], gate_acc_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    const int half_warp = lane >> 4;
    const int half_lane = lane & 15;
    const int reduce_row = warp_id * 2 + half_warp;
    if (reduce_row < WMMA_M) {
        float core_sum = 0.0f;
        float gate_sum = 0.0f;
        for (int col = half_lane; col < 128; col += 16) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = reduce_row * WMMA_N + local_n;
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
        core_sum = half_warp_sum(core_sum);
        gate_sum = half_warp_sum(gate_sum);
        float c_mean = core_sum * (1.0f / 128.0f);
        float g_mean = gate_sum * (1.0f / 128.0f);
        c_mean = __shfl_sync(0xffffffffu, c_mean, 0, 16);
        g_mean = __shfl_sync(0xffffffffu, g_mean, 0, 16);
        if (half_lane == 0) {
            core_mean[reduce_row] = c_mean;
            gate_mean[reduce_row] = g_mean;
        }

        float core_var = 0.0f;
        float gate_var = 0.0f;
        for (int col = half_lane; col < 128; col += 16) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = reduce_row * WMMA_N + local_n;
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
        core_var = half_warp_sum(core_var);
        gate_var = half_warp_sum(gate_var);
        if (half_lane == 0) {
            core_rstd[reduce_row] = rsqrtf(core_var * (1.0f / 128.0f) + eps);
            gate_rstd[reduce_row] = rsqrtf(gate_var * (1.0f / 128.0f) + eps);
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < WMMA_M * 128; idx += blockDim.x) {
        const int local_m = idx / 128;
        const int col = idx % 128;
        const int row = tile_m + local_m;
        if (row < rows) {
            const int n_tile = col / WMMA_N;
            const int local_n = col % WMMA_N;
            const int tile_idx = local_m * WMMA_N + local_n;
            float core = static_cast<float>(core_acc_tile[n_tile][tile_idx]) * core_act_scale * core_weight_scale[col];
            float gate = static_cast<float>(gate_acc_tile[n_tile][tile_idx]) * gate_act_scale * gate_weight_scale[col];
            if (core_has_bias) {
                core += core_bias[col];
            }
            if (gate_has_bias) {
                gate += gate_bias[col];
            }
            const int64_t out_idx = static_cast<int64_t>(row) * 128 + col;
            core_linear[out_idx] = core;
            gate_linear[out_idx] = gate;
            core = (core - core_mean[local_m]) * core_rstd[local_m];
            gate = (gate - gate_mean[local_m]) * gate_rstd[local_m];
            core = core * core_norm_weight[col] + core_norm_bias[col];
            gate = gate * gate_norm_weight[col] + gate_norm_bias[col];
            core_norm_output[out_idx] = core;
            gate_norm_output[out_idx] = gate;
            const float core_sigmoid = 1.0f / (1.0f + expf(-core));
            const float gate_sigmoid = 1.0f / (1.0f + expf(-gate));
            output[out_idx] = core * core_sigmoid * gate_sigmoid;
        }
    }
#endif
}

}  // namespace

void launch_quant_linear_w8a8_wmma_kernel(const float *input,
                                          const int8_t *q_weight,
                                          const float *weight_scale,
                                          const float *activation_scale,
                                          const float *bias,
                                          float *output,
                                          int64_t rows,
                                          int64_t out_features,
                                          int64_t in_features,
                                          bool has_bias,
                                          cudaStream_t stream) {
    const dim3 block(32);
    const dim3 grid((out_features + WMMA_N - 1) / WMMA_N,
                    (rows + WMMA_M - 1) / WMMA_M);

    quant_linear_w8a8_wmma_kernel<<<grid, block, 0, stream>>>(
        input,
        q_weight,
        weight_scale,
        activation_scale,
        bias,
        output,
        rows,
        out_features,
        in_features,
        has_bias);
}

void launch_quant_linear_w8a8_grad_input_kernel(const float *grad_output,
                                                const int8_t *q_weight,
                                                const float *weight_scale,
                                                float *grad_input,
                                                int64_t rows,
                                                int64_t out_features,
                                                int64_t in_features,
                                                cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((in_features + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    quant_linear_w8a8_grad_input_kernel<<<grid, block, 0, stream>>>(
        grad_output,
        q_weight,
        weight_scale,
        grad_input,
        rows,
        out_features,
        in_features);
}

void launch_quant_linear_w8a8_wmma_dual_kernel(const float *core_input,
                                               const float *gate_input,
                                               const int8_t *core_q_weight,
                                               const int8_t *gate_q_weight,
                                               const float *core_weight_scale,
                                               const float *gate_weight_scale,
                                               const float *core_activation_scale,
                                               const float *gate_activation_scale,
                                               const float *core_bias,
                                               const float *gate_bias,
                                               float *core_output,
                                               float *gate_output,
                                               int64_t rows,
                                               int64_t core_out_features,
                                               int64_t gate_out_features,
                                               int64_t core_in_features,
                                               int64_t gate_in_features,
                                               bool core_has_bias,
                                               bool gate_has_bias,
                                               cudaStream_t stream) {
    const int64_t max_out = core_out_features > gate_out_features ? core_out_features : gate_out_features;
    const dim3 block(32);
    const dim3 grid((max_out + WMMA_N - 1) / WMMA_N,
                    (rows + WMMA_M - 1) / WMMA_M,
                    2);

    quant_linear_w8a8_wmma_dual_kernel<<<grid, block, 0, stream>>>(
        core_input,
        gate_input,
        core_q_weight,
        gate_q_weight,
        core_weight_scale,
        gate_weight_scale,
        core_activation_scale,
        gate_activation_scale,
        core_bias,
        gate_bias,
        core_output,
        gate_output,
        rows,
        core_out_features,
        gate_out_features,
        core_in_features,
        gate_in_features,
        core_has_bias,
        gate_has_bias);
}

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
    float *output,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps,
    cudaStream_t stream) {
    const dim3 block(FIXED_N128_TILES * 32);
    const dim3 grid((rows + WMMA_M - 1) / WMMA_M);
    quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel<<<grid, block, 0, stream>>>(
        core_input,
        gate_input,
        core_q_weight,
        gate_q_weight,
        core_weight_scale,
        gate_weight_scale,
        core_activation_scale,
        gate_activation_scale,
        core_bias,
        gate_bias,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        output,
        rows,
        core_in_features,
        gate_in_features,
        core_has_bias,
        gate_has_bias,
        eps);
}

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
    float *output,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps,
    cudaStream_t stream) {
    const dim3 block(FIXED_N128_TILES * 2 * 32);
    const dim3 grid((rows + WMMA_M - 1) / WMMA_M);
    quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel<<<grid, block, 0, stream>>>(
        core_input,
        gate_input,
        core_q_weight,
        gate_q_weight,
        core_weight_scale,
        gate_weight_scale,
        core_activation_scale,
        gate_activation_scale,
        core_bias,
        gate_bias,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        output,
        rows,
        core_in_features,
        gate_in_features,
        core_has_bias,
        gate_has_bias,
        eps);
}

void launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_aux_kernel(
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
    float *output,
    float *core_linear,
    float *gate_linear,
    float *core_norm,
    float *gate_norm,
    int64_t rows,
    int64_t core_in_features,
    int64_t gate_in_features,
    bool core_has_bias,
    bool gate_has_bias,
    float eps,
    cudaStream_t stream) {
    const dim3 block(FIXED_N128_TILES * 32);
    const dim3 grid((rows + WMMA_M - 1) / WMMA_M);
    quant_linear_w8a8_wmma_dual_gated_tail_n128_aux_kernel<<<grid, block, 0, stream>>>(
        core_input,
        gate_input,
        core_q_weight,
        gate_q_weight,
        core_weight_scale,
        gate_weight_scale,
        core_activation_scale,
        gate_activation_scale,
        core_bias,
        gate_bias,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        output,
        core_linear,
        gate_linear,
        core_norm,
        gate_norm,
        rows,
        core_in_features,
        gate_in_features,
        core_has_bias,
        gate_has_bias,
        eps);
}
