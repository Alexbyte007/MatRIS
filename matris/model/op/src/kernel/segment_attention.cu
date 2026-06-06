#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kDim = 128;

__global__ void target_attention_sum_forward_kernel(const float *logits,
                                                    const float *values,
                                                    const int64_t *lengths,
                                                    const int64_t *offsets,
                                                    float *out,
                                                    float *alpha,
                                                    int64_t num_segments) {
    const int64_t seg = blockIdx.x;
    const int d = threadIdx.x;
    if (seg >= num_segments || d >= kDim) {
        return;
    }

    const int64_t len = lengths[seg];
    const int64_t start = offsets[seg];
    if (len <= 0) {
        out[seg * kDim + d] = 0.0f;
        return;
    }

    float max_v = -CUDART_INF_F;
    for (int64_t r = 0; r < len; ++r) {
        const float v = logits[(start + r) * kDim + d];
        max_v = fmaxf(max_v, v);
    }

    float sum_exp = 0.0f;
    for (int64_t r = 0; r < len; ++r) {
        const float e = expf(logits[(start + r) * kDim + d] - max_v);
        alpha[(start + r) * kDim + d] = e;
        sum_exp += e;
    }

    const float inv_sum = 1.0f / sum_exp;
    float accum = 0.0f;
    for (int64_t r = 0; r < len; ++r) {
        const int64_t idx = (start + r) * kDim + d;
        const float a = alpha[idx] * inv_sum;
        alpha[idx] = a;
        accum += a * values[idx];
    }
    out[seg * kDim + d] = accum;
}

__global__ void target_attention_sum_forward_no_alpha_kernel(const float *logits,
                                                             const float *values,
                                                             const int64_t *lengths,
                                                             const int64_t *offsets,
                                                             float *out,
                                                             int64_t num_segments) {
    const int64_t seg = blockIdx.x;
    const int d = threadIdx.x;
    if (seg >= num_segments || d >= kDim) {
        return;
    }

    const int64_t len = lengths[seg];
    const int64_t start = offsets[seg];
    if (len <= 0) {
        out[seg * kDim + d] = 0.0f;
        return;
    }

    float max_v = -CUDART_INF_F;
    for (int64_t r = 0; r < len; ++r) {
        max_v = fmaxf(max_v, logits[(start + r) * kDim + d]);
    }

    float sum_exp = 0.0f;
    float accum = 0.0f;
    for (int64_t r = 0; r < len; ++r) {
        const int64_t idx = (start + r) * kDim + d;
        const float e = expf(logits[idx] - max_v);
        sum_exp += e;
        accum += e * values[idx];
    }
    out[seg * kDim + d] = sum_exp > 0.0f ? accum / sum_exp : 0.0f;
}

__global__ void target_attention_sum_backward_kernel(const float *grad_out,
                                                     const float *values,
                                                     const float *out,
                                                     const float *alpha,
                                                     const int64_t *lengths,
                                                     const int64_t *offsets,
                                                     float *grad_logits,
                                                     float *grad_values,
                                                     int64_t num_segments) {
    const int64_t seg = blockIdx.x;
    const int d = threadIdx.x;
    if (seg >= num_segments || d >= kDim) {
        return;
    }

    const int64_t len = lengths[seg];
    const int64_t start = offsets[seg];
    if (len <= 0) {
        return;
    }

    const float go = grad_out[seg * kDim + d];
    const float segment_out = out[seg * kDim + d];
    for (int64_t r = 0; r < len; ++r) {
        const int64_t idx = (start + r) * kDim + d;
        const float a = alpha[idx];
        grad_values[idx] = a * go;
        grad_logits[idx] = a * go * (values[idx] - segment_out);
    }
}

__global__ void target_attention_sum_backward_recompute_kernel(const float *grad_out,
                                                               const float *logits,
                                                               const float *values,
                                                               const float *out,
                                                               const int64_t *lengths,
                                                               const int64_t *offsets,
                                                               float *grad_logits,
                                                               float *grad_values,
                                                               int64_t num_segments) {
    const int64_t seg = blockIdx.x;
    const int d = threadIdx.x;
    if (seg >= num_segments || d >= kDim) {
        return;
    }

    const int64_t len = lengths[seg];
    const int64_t start = offsets[seg];
    if (len <= 0) {
        return;
    }

    float max_v = -CUDART_INF_F;
    for (int64_t r = 0; r < len; ++r) {
        max_v = fmaxf(max_v, logits[(start + r) * kDim + d]);
    }

    float sum_exp = 0.0f;
    for (int64_t r = 0; r < len; ++r) {
        sum_exp += expf(logits[(start + r) * kDim + d] - max_v);
    }
    const float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    const float go = grad_out[seg * kDim + d];
    const float segment_out = out[seg * kDim + d];
    for (int64_t r = 0; r < len; ++r) {
        const int64_t idx = (start + r) * kDim + d;
        const float a = expf(logits[idx] - max_v) * inv_sum;
        grad_values[idx] = a * go;
        grad_logits[idx] = a * go * (values[idx] - segment_out);
    }
}

}  // namespace

void launch_target_attention_sum_forward_kernel(const float *logits,
                                                const float *values,
                                                const int64_t *lengths,
                                                const int64_t *offsets,
                                                float *out,
                                                float *alpha,
                                                int64_t num_segments,
                                                cudaStream_t stream) {
    dim3 block(kDim);
    dim3 grid(num_segments);
    target_attention_sum_forward_kernel<<<grid, block, 0, stream>>>(
        logits,
        values,
        lengths,
        offsets,
        out,
        alpha,
        num_segments);
}

void launch_target_attention_sum_forward_no_alpha_kernel(const float *logits,
                                                         const float *values,
                                                         const int64_t *lengths,
                                                         const int64_t *offsets,
                                                         float *out,
                                                         int64_t num_segments,
                                                         cudaStream_t stream) {
    dim3 block(kDim);
    dim3 grid(num_segments);
    target_attention_sum_forward_no_alpha_kernel<<<grid, block, 0, stream>>>(
        logits,
        values,
        lengths,
        offsets,
        out,
        num_segments);
}

void launch_target_attention_sum_backward_kernel(const float *grad_out,
                                                 const float *values,
                                                 const float *out,
                                                 const float *alpha,
                                                 const int64_t *lengths,
                                                 const int64_t *offsets,
                                                 float *grad_logits,
                                                 float *grad_values,
                                                 int64_t num_segments,
                                                 cudaStream_t stream) {
    dim3 block(kDim);
    dim3 grid(num_segments);
    target_attention_sum_backward_kernel<<<grid, block, 0, stream>>>(
        grad_out,
        values,
        out,
        alpha,
        lengths,
        offsets,
        grad_logits,
        grad_values,
        num_segments);
}

void launch_target_attention_sum_backward_recompute_kernel(const float *grad_out,
                                                           const float *logits,
                                                           const float *values,
                                                           const float *out,
                                                           const int64_t *lengths,
                                                           const int64_t *offsets,
                                                           float *grad_logits,
                                                           float *grad_values,
                                                           int64_t num_segments,
                                                           cudaStream_t stream) {
    dim3 block(kDim);
    dim3 grid(num_segments);
    target_attention_sum_backward_recompute_kernel<<<grid, block, 0, stream>>>(
        grad_out,
        logits,
        values,
        out,
        lengths,
        offsets,
        grad_logits,
        grad_values,
        num_segments);
}
