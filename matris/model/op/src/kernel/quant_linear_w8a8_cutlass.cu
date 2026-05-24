#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace {

__global__ void quantize_activation_s8_kernel(const float *__restrict__ input,
                                              int8_t *__restrict__ q_input,
                                              const float *__restrict__ activation_scale,
                                              int64_t rows,
                                              int64_t in_features) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * in_features;
    if (idx >= total) {
        return;
    }
    float scaled = input[idx] / activation_scale[0];
    int q = static_cast<int>(rintf(scaled));
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    q_input[idx] = static_cast<int8_t>(q);
}

__global__ void dequantize_s32_kernel(const int32_t *__restrict__ acc,
                                      const float *__restrict__ weight_scale,
                                      const float *__restrict__ activation_scale,
                                      const float *__restrict__ bias,
                                      float *__restrict__ output,
                                      int64_t rows,
                                      int64_t out_features,
                                      bool has_bias) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * out_features;
    if (idx >= total) {
        return;
    }
    const int64_t col = idx % out_features;
    float out = static_cast<float>(acc[idx]) * activation_scale[0] * weight_scale[col];
    if (has_bias) {
        out += bias[col];
    }
    output[idx] = out;
}

__global__ void quantize_activation_s8_dual_kernel(const float *__restrict__ core_input,
                                                   const float *__restrict__ gate_input,
                                                   int8_t *__restrict__ core_q_input,
                                                   int8_t *__restrict__ gate_q_input,
                                                   const float *__restrict__ core_activation_scale,
                                                   const float *__restrict__ gate_activation_scale,
                                                   int64_t rows,
                                                   int64_t core_in_features,
                                                   int64_t gate_in_features) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t core_total = rows * core_in_features;
    const int64_t gate_total = rows * gate_in_features;
    const int64_t total = core_total + gate_total;
    if (idx >= total) {
        return;
    }

    const bool is_core = idx < core_total;
    const int64_t local_idx = is_core ? idx : idx - core_total;
    const float *input = is_core ? core_input : gate_input;
    int8_t *q_input = is_core ? core_q_input : gate_q_input;
    const float scale = is_core ? core_activation_scale[0] : gate_activation_scale[0];
    float scaled = input[local_idx] / scale;
    int q = static_cast<int>(rintf(scaled));
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    q_input[local_idx] = static_cast<int8_t>(q);
}

__global__ void quantize_activation_s8_dual_meta_kernel(const float *__restrict__ core_input,
                                                        const float *__restrict__ gate_input,
                                                        int8_t *__restrict__ core_q_input,
                                                        int8_t *__restrict__ gate_q_input,
                                                        const int8_t *__restrict__ core_q_weight,
                                                        const int8_t *__restrict__ gate_q_weight,
                                                        int32_t *__restrict__ core_acc,
                                                        int32_t *__restrict__ gate_acc,
                                                        int32_t *__restrict__ problem_sizes,
                                                        int64_t *__restrict__ ptr_a,
                                                        int64_t *__restrict__ ptr_b,
                                                        int64_t *__restrict__ ptr_c,
                                                        int64_t *__restrict__ ptr_d,
                                                        int64_t *__restrict__ lda,
                                                        int64_t *__restrict__ ldb,
                                                        int64_t *__restrict__ ldc,
                                                        int64_t *__restrict__ ldd,
                                                        const float *__restrict__ core_activation_scale,
                                                        const float *__restrict__ gate_activation_scale,
                                                        int64_t rows,
                                                        int64_t core_in_features,
                                                        int64_t gate_in_features,
                                                        int64_t core_out_features,
                                                        int64_t gate_out_features) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx == 0) {
        problem_sizes[0] = static_cast<int32_t>(rows);
        problem_sizes[1] = static_cast<int32_t>(core_out_features);
        problem_sizes[2] = static_cast<int32_t>(core_in_features);
        problem_sizes[3] = static_cast<int32_t>(rows);
        problem_sizes[4] = static_cast<int32_t>(gate_out_features);
        problem_sizes[5] = static_cast<int32_t>(gate_in_features);

        ptr_a[0] = reinterpret_cast<int64_t>(core_q_input);
        ptr_a[1] = reinterpret_cast<int64_t>(gate_q_input);
        ptr_b[0] = reinterpret_cast<int64_t>(core_q_weight);
        ptr_b[1] = reinterpret_cast<int64_t>(gate_q_weight);
        ptr_c[0] = reinterpret_cast<int64_t>(core_acc);
        ptr_c[1] = reinterpret_cast<int64_t>(gate_acc);
        ptr_d[0] = reinterpret_cast<int64_t>(core_acc);
        ptr_d[1] = reinterpret_cast<int64_t>(gate_acc);

        lda[0] = core_in_features;
        lda[1] = gate_in_features;
        ldb[0] = core_in_features;
        ldb[1] = gate_in_features;
        ldc[0] = core_out_features;
        ldc[1] = gate_out_features;
        ldd[0] = core_out_features;
        ldd[1] = gate_out_features;
    }

    const int64_t core_total = rows * core_in_features;
    const int64_t gate_total = rows * gate_in_features;
    const int64_t total = core_total + gate_total;
    if (idx >= total) {
        return;
    }

    const bool is_core = idx < core_total;
    const int64_t local_idx = is_core ? idx : idx - core_total;
    const float *input = is_core ? core_input : gate_input;
    int8_t *q_input = is_core ? core_q_input : gate_q_input;
    const float scale = is_core ? core_activation_scale[0] : gate_activation_scale[0];
    float scaled = input[local_idx] / scale;
    int q = static_cast<int>(rintf(scaled));
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    q_input[local_idx] = static_cast<int8_t>(q);
}

__global__ void quantize_activation_s8_pair_meta_kernel(const float *__restrict__ input_a,
                                                        const float *__restrict__ input_b,
                                                        int8_t *__restrict__ q_input_a,
                                                        int8_t *__restrict__ q_input_b,
                                                        const int8_t *__restrict__ q_weight_a,
                                                        const int8_t *__restrict__ q_weight_b,
                                                        int32_t *__restrict__ acc_a,
                                                        int32_t *__restrict__ acc_b,
                                                        int32_t *__restrict__ problem_sizes,
                                                        int64_t *__restrict__ ptr_a,
                                                        int64_t *__restrict__ ptr_b,
                                                        int64_t *__restrict__ ptr_c,
                                                        int64_t *__restrict__ ptr_d,
                                                        int64_t *__restrict__ lda,
                                                        int64_t *__restrict__ ldb,
                                                        int64_t *__restrict__ ldc,
                                                        int64_t *__restrict__ ldd,
                                                        const float *__restrict__ activation_scale_a,
                                                        const float *__restrict__ activation_scale_b,
                                                        int64_t rows_a,
                                                        int64_t rows_b,
                                                        int64_t in_features_a,
                                                        int64_t in_features_b,
                                                        int64_t out_features_a,
                                                        int64_t out_features_b) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx == 0) {
        problem_sizes[0] = static_cast<int32_t>(rows_a);
        problem_sizes[1] = static_cast<int32_t>(out_features_a);
        problem_sizes[2] = static_cast<int32_t>(in_features_a);
        problem_sizes[3] = static_cast<int32_t>(rows_b);
        problem_sizes[4] = static_cast<int32_t>(out_features_b);
        problem_sizes[5] = static_cast<int32_t>(in_features_b);

        ptr_a[0] = reinterpret_cast<int64_t>(q_input_a);
        ptr_a[1] = reinterpret_cast<int64_t>(q_input_b);
        ptr_b[0] = reinterpret_cast<int64_t>(q_weight_a);
        ptr_b[1] = reinterpret_cast<int64_t>(q_weight_b);
        ptr_c[0] = reinterpret_cast<int64_t>(acc_a);
        ptr_c[1] = reinterpret_cast<int64_t>(acc_b);
        ptr_d[0] = reinterpret_cast<int64_t>(acc_a);
        ptr_d[1] = reinterpret_cast<int64_t>(acc_b);

        lda[0] = in_features_a;
        lda[1] = in_features_b;
        ldb[0] = in_features_a;
        ldb[1] = in_features_b;
        ldc[0] = out_features_a;
        ldc[1] = out_features_b;
        ldd[0] = out_features_a;
        ldd[1] = out_features_b;
    }

    const int64_t total_a = rows_a * in_features_a;
    const int64_t total_b = rows_b * in_features_b;
    const int64_t total = total_a + total_b;
    if (idx >= total) {
        return;
    }

    const bool is_a = idx < total_a;
    const int64_t local_idx = is_a ? idx : idx - total_a;
    const float *input = is_a ? input_a : input_b;
    int8_t *q_input = is_a ? q_input_a : q_input_b;
    const float scale = is_a ? activation_scale_a[0] : activation_scale_b[0];
    float scaled = input[local_idx] / scale;
    int q = static_cast<int>(rintf(scaled));
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    q_input[local_idx] = static_cast<int8_t>(q);
}

__global__ void dequantize_s32_dual_kernel(const int32_t *__restrict__ core_acc,
                                           const int32_t *__restrict__ gate_acc,
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
                                           bool core_has_bias,
                                           bool gate_has_bias) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t core_total = rows * core_out_features;
    const int64_t gate_total = rows * gate_out_features;
    const int64_t total = core_total + gate_total;
    if (idx >= total) {
        return;
    }

    const bool is_core = idx < core_total;
    const int64_t local_idx = is_core ? idx : idx - core_total;
    const int64_t out_features = is_core ? core_out_features : gate_out_features;
    const int64_t col = local_idx % out_features;
    const int32_t *acc = is_core ? core_acc : gate_acc;
    const float *weight_scale = is_core ? core_weight_scale : gate_weight_scale;
    const float *activation_scale = is_core ? core_activation_scale : gate_activation_scale;
    const float *bias = is_core ? core_bias : gate_bias;
    float *output = is_core ? core_output : gate_output;
    const bool has_bias = is_core ? core_has_bias : gate_has_bias;

    float out = static_cast<float>(acc[local_idx]) * activation_scale[0] * weight_scale[col];
    if (has_bias) {
        out += bias[col];
    }
    output[local_idx] = out;
}

__global__ void dequantize_s32_pair_kernel(const int32_t *__restrict__ acc_a,
                                           const int32_t *__restrict__ acc_b,
                                           const float *__restrict__ weight_scale_a,
                                           const float *__restrict__ weight_scale_b,
                                           const float *__restrict__ activation_scale_a,
                                           const float *__restrict__ activation_scale_b,
                                           const float *__restrict__ bias_a,
                                           const float *__restrict__ bias_b,
                                           float *__restrict__ output_a,
                                           float *__restrict__ output_b,
                                           int64_t rows_a,
                                           int64_t rows_b,
                                           int64_t out_features_a,
                                           int64_t out_features_b,
                                           bool has_bias_a,
                                           bool has_bias_b) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_a = rows_a * out_features_a;
    const int64_t total_b = rows_b * out_features_b;
    const int64_t total = total_a + total_b;
    if (idx >= total) {
        return;
    }

    if (idx < total_a) {
        const int64_t col = idx % out_features_a;
        float out = static_cast<float>(acc_a[idx]) * activation_scale_a[0] * weight_scale_a[col];
        if (has_bias_a) {
            out += bias_a[col];
        }
        output_a[idx] = out;
        return;
    }

    const int64_t local_idx = idx - total_a;
    const int64_t col = local_idx % out_features_b;
    float out = static_cast<float>(acc_b[local_idx]) * activation_scale_b[0] * weight_scale_b[col];
    if (has_bias_b) {
        out += bias_b[col];
    }
    output_b[local_idx] = out;
}

__global__ void dequantize_silu_quant_pair_meta_kernel(const int32_t *__restrict__ acc_a,
                                                       const int32_t *__restrict__ acc_b,
                                                       const float *__restrict__ weight_scale_a,
                                                       const float *__restrict__ weight_scale_b,
                                                       const float *__restrict__ activation_scale_a,
                                                       const float *__restrict__ activation_scale_b,
                                                       const float *__restrict__ bias_a,
                                                       const float *__restrict__ bias_b,
                                                       float *__restrict__ hidden_a,
                                                       float *__restrict__ hidden_b,
                                                       int8_t *__restrict__ q_hidden_a,
                                                       int8_t *__restrict__ q_hidden_b,
                                                       const int8_t *__restrict__ q_weight2_a,
                                                       const int8_t *__restrict__ q_weight2_b,
                                                       int32_t *__restrict__ acc2_a,
                                                       int32_t *__restrict__ acc2_b,
                                                       int32_t *__restrict__ problem_sizes,
                                                       int64_t *__restrict__ ptr_a,
                                                       int64_t *__restrict__ ptr_b,
                                                       int64_t *__restrict__ ptr_c,
                                                       int64_t *__restrict__ ptr_d,
                                                       int64_t *__restrict__ lda,
                                                       int64_t *__restrict__ ldb,
                                                       int64_t *__restrict__ ldc,
                                                       int64_t *__restrict__ ldd,
                                                       const float *__restrict__ hidden_activation_scale_a,
                                                       const float *__restrict__ hidden_activation_scale_b,
                                                       int64_t rows_a,
                                                       int64_t rows_b,
                                                       int64_t hidden_features_a,
                                                       int64_t hidden_features_b,
                                                       int64_t out_features_a,
                                                       int64_t out_features_b,
                                                       bool has_bias_a,
                                                       bool has_bias_b) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx == 0) {
        problem_sizes[0] = static_cast<int32_t>(rows_a);
        problem_sizes[1] = static_cast<int32_t>(out_features_a);
        problem_sizes[2] = static_cast<int32_t>(hidden_features_a);
        problem_sizes[3] = static_cast<int32_t>(rows_b);
        problem_sizes[4] = static_cast<int32_t>(out_features_b);
        problem_sizes[5] = static_cast<int32_t>(hidden_features_b);

        ptr_a[0] = reinterpret_cast<int64_t>(q_hidden_a);
        ptr_a[1] = reinterpret_cast<int64_t>(q_hidden_b);
        ptr_b[0] = reinterpret_cast<int64_t>(q_weight2_a);
        ptr_b[1] = reinterpret_cast<int64_t>(q_weight2_b);
        ptr_c[0] = reinterpret_cast<int64_t>(acc2_a);
        ptr_c[1] = reinterpret_cast<int64_t>(acc2_b);
        ptr_d[0] = reinterpret_cast<int64_t>(acc2_a);
        ptr_d[1] = reinterpret_cast<int64_t>(acc2_b);

        lda[0] = hidden_features_a;
        lda[1] = hidden_features_b;
        ldb[0] = hidden_features_a;
        ldb[1] = hidden_features_b;
        ldc[0] = out_features_a;
        ldc[1] = out_features_b;
        ldd[0] = out_features_a;
        ldd[1] = out_features_b;
    }

    const int64_t total_a = rows_a * hidden_features_a;
    const int64_t total_b = rows_b * hidden_features_b;
    const int64_t total = total_a + total_b;
    if (idx >= total) {
        return;
    }

    const bool is_a = idx < total_a;
    const int64_t local_idx = is_a ? idx : idx - total_a;
    const int64_t features = is_a ? hidden_features_a : hidden_features_b;
    const int64_t col = local_idx % features;
    const int32_t *acc = is_a ? acc_a : acc_b;
    const float *weight_scale = is_a ? weight_scale_a : weight_scale_b;
    const float *activation_scale = is_a ? activation_scale_a : activation_scale_b;
    const float *bias = is_a ? bias_a : bias_b;
    const bool has_bias = is_a ? has_bias_a : has_bias_b;
    float *hidden = is_a ? hidden_a : hidden_b;
    int8_t *q_hidden = is_a ? q_hidden_a : q_hidden_b;
    const float hidden_scale = is_a ? hidden_activation_scale_a[0] : hidden_activation_scale_b[0];

    float pre = static_cast<float>(acc[local_idx]) * activation_scale[0] * weight_scale[col];
    if (has_bias) {
        pre += bias[col];
    }
    hidden[local_idx] = pre;
    const float sigmoid = 1.0f / (1.0f + expf(-pre));
    const float activated = pre * sigmoid;
    int q = static_cast<int>(rintf(activated / hidden_scale));
    q = q > 127 ? 127 : q;
    q = q < -127 ? -127 : q;
    q_hidden[local_idx] = static_cast<int8_t>(q);
}

__global__ void dequantize_s32_dual_gated_tail_kernel(const int32_t *__restrict__ core_acc,
                                                      const int32_t *__restrict__ gate_acc,
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
                                                      int64_t out_features,
                                                      bool core_has_bias,
                                                      bool gate_has_bias,
                                                      float eps) {
    extern __shared__ float shared[];
    float *core_reduce = shared;
    float *gate_reduce = shared + blockDim.x;

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    float core_sum = 0.0f;
    float gate_sum = 0.0f;
    for (int64_t col = tid; col < out_features; col += blockDim.x) {
        const int64_t idx = row * out_features + col;
        float core = static_cast<float>(core_acc[idx]) * core_activation_scale[0] * core_weight_scale[col];
        float gate = static_cast<float>(gate_acc[idx]) * gate_activation_scale[0] * gate_weight_scale[col];
        if (core_has_bias) {
            core += core_bias[col];
        }
        if (gate_has_bias) {
            gate += gate_bias[col];
        }
        core_sum += core;
        gate_sum += gate;
    }
    core_reduce[tid] = core_sum;
    gate_reduce[tid] = gate_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            core_reduce[tid] += core_reduce[tid + stride];
            gate_reduce[tid] += gate_reduce[tid + stride];
        }
        __syncthreads();
    }

    const float inv_d = 1.0f / static_cast<float>(out_features);
    const float core_mean = core_reduce[0] * inv_d;
    const float gate_mean = gate_reduce[0] * inv_d;

    float core_var_sum = 0.0f;
    float gate_var_sum = 0.0f;
    for (int64_t col = tid; col < out_features; col += blockDim.x) {
        const int64_t idx = row * out_features + col;
        float core = static_cast<float>(core_acc[idx]) * core_activation_scale[0] * core_weight_scale[col];
        float gate = static_cast<float>(gate_acc[idx]) * gate_activation_scale[0] * gate_weight_scale[col];
        if (core_has_bias) {
            core += core_bias[col];
        }
        if (gate_has_bias) {
            gate += gate_bias[col];
        }
        const float core_centered = core - core_mean;
        const float gate_centered = gate - gate_mean;
        core_var_sum += core_centered * core_centered;
        gate_var_sum += gate_centered * gate_centered;
    }
    core_reduce[tid] = core_var_sum;
    gate_reduce[tid] = gate_var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            core_reduce[tid] += core_reduce[tid + stride];
            gate_reduce[tid] += gate_reduce[tid + stride];
        }
        __syncthreads();
    }

    const float core_rstd = rsqrtf(core_reduce[0] * inv_d + eps);
    const float gate_rstd = rsqrtf(gate_reduce[0] * inv_d + eps);

    for (int64_t col = tid; col < out_features; col += blockDim.x) {
        const int64_t idx = row * out_features + col;
        float core = static_cast<float>(core_acc[idx]) * core_activation_scale[0] * core_weight_scale[col];
        float gate = static_cast<float>(gate_acc[idx]) * gate_activation_scale[0] * gate_weight_scale[col];
        if (core_has_bias) {
            core += core_bias[col];
        }
        if (gate_has_bias) {
            gate += gate_bias[col];
        }

        float core_norm = (core - core_mean) * core_rstd;
        float gate_norm = (gate - gate_mean) * gate_rstd;
        core_norm = core_norm * core_norm_weight[col] + core_norm_bias[col];
        gate_norm = gate_norm * gate_norm_weight[col] + gate_norm_bias[col];

        const float core_sigmoid = 1.0f / (1.0f + expf(-core_norm));
        const float gate_sigmoid = 1.0f / (1.0f + expf(-gate_norm));
        output[idx] = core_norm * core_sigmoid * gate_sigmoid;
    }
}

}  // namespace

void launch_quantize_activation_s8_kernel(const float *input,
                                          int8_t *q_input,
                                          const float *activation_scale,
                                          int64_t rows,
                                          int64_t in_features,
                                          cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows * in_features;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    quantize_activation_s8_kernel<<<blocks, threads, 0, stream>>>(
        input,
        q_input,
        activation_scale,
        rows,
        in_features);
}

void launch_dequantize_s32_kernel(const int32_t *acc,
                                  const float *weight_scale,
                                  const float *activation_scale,
                                  const float *bias,
                                  float *output,
                                  int64_t rows,
                                  int64_t out_features,
                                  bool has_bias,
                                  cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows * out_features;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    dequantize_s32_kernel<<<blocks, threads, 0, stream>>>(
        acc,
        weight_scale,
        activation_scale,
        bias,
        output,
        rows,
        out_features,
        has_bias);
}

void launch_quantize_activation_s8_dual_kernel(const float *core_input,
                                               const float *gate_input,
                                               int8_t *core_q_input,
                                               int8_t *gate_q_input,
                                               const float *core_activation_scale,
                                               const float *gate_activation_scale,
                                               int64_t rows,
                                               int64_t core_in_features,
                                               int64_t gate_in_features,
                                               cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows * (core_in_features + gate_in_features);
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    quantize_activation_s8_dual_kernel<<<blocks, threads, 0, stream>>>(
        core_input,
        gate_input,
        core_q_input,
        gate_q_input,
        core_activation_scale,
        gate_activation_scale,
        rows,
        core_in_features,
        gate_in_features);
}

void launch_quantize_activation_s8_dual_meta_kernel(const float *core_input,
                                                    const float *gate_input,
                                                    int8_t *core_q_input,
                                                    int8_t *gate_q_input,
                                                    const int8_t *core_q_weight,
                                                    const int8_t *gate_q_weight,
                                                    int32_t *core_acc,
                                                    int32_t *gate_acc,
                                                    int32_t *problem_sizes,
                                                    int64_t *ptr_a,
                                                    int64_t *ptr_b,
                                                    int64_t *ptr_c,
                                                    int64_t *ptr_d,
                                                    int64_t *lda,
                                                    int64_t *ldb,
                                                    int64_t *ldc,
                                                    int64_t *ldd,
                                                    const float *core_activation_scale,
                                                    const float *gate_activation_scale,
                                                    int64_t rows,
                                                    int64_t core_in_features,
                                                    int64_t gate_in_features,
                                                    int64_t core_out_features,
                                                    int64_t gate_out_features,
                                                    cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows * (core_in_features + gate_in_features);
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    quantize_activation_s8_dual_meta_kernel<<<blocks, threads, 0, stream>>>(
        core_input,
        gate_input,
        core_q_input,
        gate_q_input,
        core_q_weight,
        gate_q_weight,
        core_acc,
        gate_acc,
        problem_sizes,
        ptr_a,
        ptr_b,
        ptr_c,
        ptr_d,
        lda,
        ldb,
        ldc,
        ldd,
        core_activation_scale,
        gate_activation_scale,
        rows,
        core_in_features,
        gate_in_features,
        core_out_features,
        gate_out_features);
}

void launch_quantize_activation_s8_pair_meta_kernel(const float *input_a,
                                                    const float *input_b,
                                                    int8_t *q_input_a,
                                                    int8_t *q_input_b,
                                                    const int8_t *q_weight_a,
                                                    const int8_t *q_weight_b,
                                                    int32_t *acc_a,
                                                    int32_t *acc_b,
                                                    int32_t *problem_sizes,
                                                    int64_t *ptr_a,
                                                    int64_t *ptr_b,
                                                    int64_t *ptr_c,
                                                    int64_t *ptr_d,
                                                    int64_t *lda,
                                                    int64_t *ldb,
                                                    int64_t *ldc,
                                                    int64_t *ldd,
                                                    const float *activation_scale_a,
                                                    const float *activation_scale_b,
                                                    int64_t rows_a,
                                                    int64_t rows_b,
                                                    int64_t in_features_a,
                                                    int64_t in_features_b,
                                                    int64_t out_features_a,
                                                    int64_t out_features_b,
                                                    cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows_a * in_features_a + rows_b * in_features_b;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    quantize_activation_s8_pair_meta_kernel<<<blocks, threads, 0, stream>>>(
        input_a,
        input_b,
        q_input_a,
        q_input_b,
        q_weight_a,
        q_weight_b,
        acc_a,
        acc_b,
        problem_sizes,
        ptr_a,
        ptr_b,
        ptr_c,
        ptr_d,
        lda,
        ldb,
        ldc,
        ldd,
        activation_scale_a,
        activation_scale_b,
        rows_a,
        rows_b,
        in_features_a,
        in_features_b,
        out_features_a,
        out_features_b);
}

void launch_dequantize_s32_dual_kernel(const int32_t *core_acc,
                                       const int32_t *gate_acc,
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
                                       bool core_has_bias,
                                       bool gate_has_bias,
                                       cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows * (core_out_features + gate_out_features);
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    dequantize_s32_dual_kernel<<<blocks, threads, 0, stream>>>(
        core_acc,
        gate_acc,
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
        core_has_bias,
        gate_has_bias);
}

void launch_dequantize_s32_pair_kernel(const int32_t *acc_a,
                                       const int32_t *acc_b,
                                       const float *weight_scale_a,
                                       const float *weight_scale_b,
                                       const float *activation_scale_a,
                                       const float *activation_scale_b,
                                       const float *bias_a,
                                       const float *bias_b,
                                       float *output_a,
                                       float *output_b,
                                       int64_t rows_a,
                                       int64_t rows_b,
                                       int64_t out_features_a,
                                       int64_t out_features_b,
                                       bool has_bias_a,
                                       bool has_bias_b,
                                       cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows_a * out_features_a + rows_b * out_features_b;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    dequantize_s32_pair_kernel<<<blocks, threads, 0, stream>>>(
        acc_a,
        acc_b,
        weight_scale_a,
        weight_scale_b,
        activation_scale_a,
        activation_scale_b,
        bias_a,
        bias_b,
        output_a,
        output_b,
        rows_a,
        rows_b,
        out_features_a,
        out_features_b,
        has_bias_a,
        has_bias_b);
}

void launch_dequantize_silu_quant_pair_meta_kernel(const int32_t *acc_a,
                                                   const int32_t *acc_b,
                                                   const float *weight_scale_a,
                                                   const float *weight_scale_b,
                                                   const float *activation_scale_a,
                                                   const float *activation_scale_b,
                                                   const float *bias_a,
                                                   const float *bias_b,
                                                   float *hidden_a,
                                                   float *hidden_b,
                                                   int8_t *q_hidden_a,
                                                   int8_t *q_hidden_b,
                                                   const int8_t *q_weight2_a,
                                                   const int8_t *q_weight2_b,
                                                   int32_t *acc2_a,
                                                   int32_t *acc2_b,
                                                   int32_t *problem_sizes,
                                                   int64_t *ptr_a,
                                                   int64_t *ptr_b,
                                                   int64_t *ptr_c,
                                                   int64_t *ptr_d,
                                                   int64_t *lda,
                                                   int64_t *ldb,
                                                   int64_t *ldc,
                                                   int64_t *ldd,
                                                   const float *hidden_activation_scale_a,
                                                   const float *hidden_activation_scale_b,
                                                   int64_t rows_a,
                                                   int64_t rows_b,
                                                   int64_t hidden_features_a,
                                                   int64_t hidden_features_b,
                                                   int64_t out_features_a,
                                                   int64_t out_features_b,
                                                   bool has_bias_a,
                                                   bool has_bias_b,
                                                   cudaStream_t stream) {
    const int threads = 256;
    const int64_t total = rows_a * hidden_features_a + rows_b * hidden_features_b;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    dequantize_silu_quant_pair_meta_kernel<<<blocks, threads, 0, stream>>>(
        acc_a,
        acc_b,
        weight_scale_a,
        weight_scale_b,
        activation_scale_a,
        activation_scale_b,
        bias_a,
        bias_b,
        hidden_a,
        hidden_b,
        q_hidden_a,
        q_hidden_b,
        q_weight2_a,
        q_weight2_b,
        acc2_a,
        acc2_b,
        problem_sizes,
        ptr_a,
        ptr_b,
        ptr_c,
        ptr_d,
        lda,
        ldb,
        ldc,
        ldd,
        hidden_activation_scale_a,
        hidden_activation_scale_b,
        rows_a,
        rows_b,
        hidden_features_a,
        hidden_features_b,
        out_features_a,
        out_features_b,
        has_bias_a,
        has_bias_b);
}

void launch_dequantize_s32_dual_gated_tail_kernel(const int32_t *core_acc,
                                                  const int32_t *gate_acc,
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
                                                  int64_t out_features,
                                                  bool core_has_bias,
                                                  bool gate_has_bias,
                                                  float eps,
                                                  cudaStream_t stream) {
    int threads = 1;
    while (threads < out_features && threads < 256) {
        threads <<= 1;
    }
    const size_t shared_bytes = static_cast<size_t>(threads) * 2 * sizeof(float);
    dequantize_s32_dual_gated_tail_kernel<<<static_cast<int>(rows), threads, shared_bytes, stream>>>(
        core_acc,
        gate_acc,
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
        out_features,
        core_has_bias,
        gate_has_bias,
        eps);
}

cutlass::Status launch_cutlass_s8s8s32_gemm(const int8_t *q_input,
                                            const int8_t *q_weight,
                                            int32_t *acc,
                                            int64_t rows,
                                            int64_t out_features,
                                            int64_t in_features,
                                            cudaStream_t stream) {
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        ElementInputA,
        cutlass::layout::RowMajor,
        ElementInputB,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombinationClamp<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;

    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {static_cast<int>(rows), static_cast<int>(out_features), static_cast<int>(in_features)},
        {q_input, static_cast<int>(in_features)},
        {q_weight, static_cast<int>(in_features)},
        {acc, static_cast<int>(out_features)},
        {acc, static_cast<int>(out_features)},
        {ElementCompute(1), ElementCompute(0)});

    return gemm_op(args, nullptr, stream);
}

cutlass::Status launch_cutlass_s8s8s32_grouped_gemm(const int32_t *problem_sizes_device,
                                                    const int32_t *problem_sizes_host,
                                                    int8_t **ptr_a,
                                                    int8_t **ptr_b,
                                                    int32_t **ptr_c,
                                                    int32_t **ptr_d,
                                                    int64_t *lda,
                                                    int64_t *ldb,
                                                    int64_t *ldc,
                                                    int64_t *ldd,
                                                    int problem_count,
                                                    cudaStream_t stream) {
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementInputA,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        8,
        ElementInputB,
        cutlass::layout::ColumnMajor,
        cutlass::ComplexTransform::kNone,
        8,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombinationClamp<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementCompute>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        3,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using GroupedGemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    GroupedGemm gemm_op;
    auto problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord *>(const_cast<int32_t *>(problem_sizes_device));
    auto host_problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord *>(const_cast<int32_t *>(problem_sizes_host));
    int threadblock_count = GroupedGemm::sufficient(host_problem_sizes, problem_count);
    if (threadblock_count <= 0) {
        return cutlass::Status::kErrorInternal;
    }

    typename GroupedGemm::EpilogueOutputOp::Params epilogue_op(ElementCompute(1), ElementCompute(0));
    typename GroupedGemm::Arguments args(
        problem_sizes,
        problem_count,
        threadblock_count,
        epilogue_op,
        ptr_a,
        ptr_b,
        ptr_c,
        ptr_d,
        lda,
        ldb,
        ldc,
        ldd,
        host_problem_sizes);

    size_t workspace_size = gemm_op.get_workspace_size(args);
    void *workspace = nullptr;
    cudaError_t cuda_status = cudaSuccess;
    if (workspace_size > 0) {
        cuda_status = cudaMalloc(&workspace, workspace_size);
        if (cuda_status != cudaSuccess) {
            return cutlass::Status::kErrorInternal;
        }
    }

    cutlass::Status status = gemm_op.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op.run(stream);
    }
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    return status;
}
