#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;

__global__ void quant_linear_w8a32_kernel(const float *__restrict__ input,
                                          const int8_t *__restrict__ q_weight,
                                          const float *__restrict__ scale,
                                          const float *__restrict__ bias,
                                          float *__restrict__ output,
                                          int64_t rows,
                                          int64_t out_features,
                                          int64_t in_features,
                                          bool has_bias) {
    const int64_t row = static_cast<int64_t>(blockIdx.y) * TILE_M + threadIdx.y;
    const int64_t out_col = static_cast<int64_t>(blockIdx.x) * TILE_N + threadIdx.x;

    if (row >= rows || out_col >= out_features) {
        return;
    }

    float acc = has_bias ? bias[out_col] : 0.0f;
    const float row_scale = scale[out_col];
    const float *input_row = input + row * in_features;
    const int8_t *weight_row = q_weight + out_col * in_features;

    for (int64_t k = 0; k < in_features; ++k) {
        acc += input_row[k] * (static_cast<float>(weight_row[k]) * row_scale);
    }

    output[row * out_features + out_col] = acc;
}

}  // namespace

void launch_quant_linear_w8a32_kernel(const float *input,
                                      const int8_t *q_weight,
                                      const float *scale,
                                      const float *bias,
                                      float *output,
                                      int64_t rows,
                                      int64_t out_features,
                                      int64_t in_features,
                                      bool has_bias,
                                      cudaStream_t stream) {
    const dim3 block(TILE_N, TILE_M);
    const dim3 grid((out_features + TILE_N - 1) / TILE_N,
                    (rows + TILE_M - 1) / TILE_M);

    quant_linear_w8a32_kernel<<<grid, block, 0, stream>>>(
        input,
        q_weight,
        scale,
        bias,
        output,
        rows,
        out_features,
        in_features,
        has_bias);
}
