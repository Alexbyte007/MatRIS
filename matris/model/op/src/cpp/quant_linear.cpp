#include "../Opdefine.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace cutlass {
enum class Status;
}

void launch_quant_linear_w8a32_kernel(const float *input,
                                      const int8_t *q_weight,
                                      const float *scale,
                                      const float *bias,
                                      float *output,
                                      int64_t rows,
                                      int64_t out_features,
                                      int64_t in_features,
                                      bool has_bias,
                                      cudaStream_t stream);

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
                                          cudaStream_t stream);

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
                                               cudaStream_t stream);

void launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(const float *core_input,
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

void launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(const float *core_input,
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

void launch_quantize_activation_s8_kernel(const float *input,
                                          int8_t *q_input,
                                          const float *activation_scale,
                                          int64_t rows,
                                          int64_t in_features,
                                          cudaStream_t stream);

void launch_dequantize_s32_kernel(const int32_t *acc,
                                  const float *weight_scale,
                                  const float *activation_scale,
                                  const float *bias,
                                  float *output,
                                  int64_t rows,
                                  int64_t out_features,
                                  bool has_bias,
                                  cudaStream_t stream);

void launch_quantize_activation_s8_dual_kernel(const float *core_input,
                                               const float *gate_input,
                                               int8_t *core_q_input,
                                               int8_t *gate_q_input,
                                               const float *core_activation_scale,
                                               const float *gate_activation_scale,
                                               int64_t rows,
                                               int64_t core_in_features,
                                               int64_t gate_in_features,
                                               cudaStream_t stream);

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
                                                    cudaStream_t stream);

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
                                                    cudaStream_t stream);

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
                                       cudaStream_t stream);

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
                                       cudaStream_t stream);

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
                                                   cudaStream_t stream);

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
                                                  cudaStream_t stream);

cutlass::Status launch_cutlass_s8s8s32_gemm(const int8_t *q_input,
                                            const int8_t *q_weight,
                                            int32_t *acc,
                                            int64_t rows,
                                            int64_t out_features,
                                            int64_t in_features,
                                            cudaStream_t stream);

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
                                                    cudaStream_t stream);

namespace {

int64_t w8a8_tail_n128_auto_parallel_rows() {
    const char *env = std::getenv("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_ROWS");
    if (env == nullptr) {
        env = std::getenv("MATRIS_W8A8_TAIL_N128_AUTO_THRESHOLD");
    }
    if (env == nullptr) {
        return 4096;
    }
    char *end = nullptr;
    const long long value = std::strtoll(env, &end, 10);
    if (end == env || value <= 0) {
        return 4096;
    }
    return static_cast<int64_t>(value);
}

bool use_w8a8_tail_n128_parallel_backend(int64_t rows) {
    const char *backend = std::getenv("MATRIS_W8A8_BACKEND");
    if (backend == nullptr) {
        return false;
    }
    if (std::strcmp(backend, "cuda_wmma_tail_n128_parallel") == 0) {
        return true;
    }
    if (std::strcmp(backend, "cuda_wmma_tail_n128_auto") == 0) {
        const int64_t threshold = w8a8_tail_n128_auto_parallel_rows();
        const char *when = std::getenv("MATRIS_W8A8_TAIL_N128_AUTO_PARALLEL_WHEN");
        if (when != nullptr && std::strcmp(when, "lt") == 0) {
            return rows < threshold;
        }
        if (when != nullptr && std::strcmp(when, "always") == 0) {
            return true;
        }
        if (when != nullptr && std::strcmp(when, "never") == 0) {
            return false;
        }
        return rows >= threshold;
    }
    return false;
}

}  // namespace

torch::Tensor quant_linear_w8a32(const torch::Tensor &input,
                                 const torch::Tensor &q_weight,
                                 const torch::Tensor &scale,
                                 const torch::Tensor &bias,
                                 bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a32: input must be a CUDA tensor");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a32: q_weight must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "quant_linear_w8a32: scale must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "quant_linear_w8a32: input must be float32");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a32: q_weight must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "quant_linear_w8a32: scale must be float32");
    TORCH_CHECK(input.dim() == 2, "quant_linear_w8a32: input must be 2D [M, K]");
    TORCH_CHECK(q_weight.dim() == 2, "quant_linear_w8a32: q_weight must be 2D [N, K]");
    TORCH_CHECK(scale.dim() == 1, "quant_linear_w8a32: scale must be 1D [N]");
    TORCH_CHECK(input.size(1) == q_weight.size(1), "quant_linear_w8a32: input K must match weight K");
    TORCH_CHECK(scale.size(0) == q_weight.size(0), "quant_linear_w8a32: scale N must match weight N");

    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a32: bias must be a CUDA tensor when has_bias=true");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a32: bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "quant_linear_w8a32: bias must be 1D [N]");
        TORCH_CHECK(bias.size(0) == q_weight.size(0), "quant_linear_w8a32: bias N must match weight N");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto q_weight_contig = q_weight.contiguous();
    auto scale_contig = scale.contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : bias;
    auto output = torch::empty({input.size(0), q_weight.size(0)}, input.options());

    const float *bias_ptr = has_bias ? bias_contig.data_ptr<float>() : nullptr;
    launch_quant_linear_w8a32_kernel(input_contig.data_ptr<float>(),
                                     q_weight_contig.data_ptr<int8_t>(),
                                     scale_contig.data_ptr<float>(),
                                     bias_ptr,
                                     output.data_ptr<float>(),
                                     input.size(0),
                                     q_weight.size(0),
                                     input.size(1),
                                     has_bias,
                                     c10::cuda::getCurrentCUDAStream());
    return output;
}

torch::Tensor quant_linear_w8a8_static_cutlass(const torch::Tensor &input,
                                               const torch::Tensor &q_weight,
                                               const torch::Tensor &weight_scale,
                                               const torch::Tensor &activation_scale,
                                               const torch::Tensor &bias,
                                               bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a8_static_cutlass: input must be CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a8_static_cutlass: q_weight must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "quant_linear_w8a8_static_cutlass: weight_scale must be CUDA");
    TORCH_CHECK(activation_scale.is_cuda(), "quant_linear_w8a8_static_cutlass: activation_scale must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass: input must be float32");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a8_static_cutlass: q_weight must be int8");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass: weight_scale must be float32");
    TORCH_CHECK(activation_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass: activation_scale must be float32");
    TORCH_CHECK(input.dim() == 2, "quant_linear_w8a8_static_cutlass: input must be 2D [M, K]");
    TORCH_CHECK(q_weight.dim() == 2, "quant_linear_w8a8_static_cutlass: q_weight must be 2D [N, K]");
    TORCH_CHECK(input.size(1) == q_weight.size(1), "quant_linear_w8a8_static_cutlass: input K must match weight K");
    TORCH_CHECK(weight_scale.size(0) == q_weight.size(0), "quant_linear_w8a8_static_cutlass: weight scale N mismatch");
    TORCH_CHECK(input.size(1) % 32 == 0, "quant_linear_w8a8_static_cutlass: K must be multiple of 32");

    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a8_static_cutlass: bias must be CUDA when has_bias=true");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == q_weight.size(0), "quant_linear_w8a8_static_cutlass: bias N mismatch");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto q_weight_contig = q_weight.contiguous();
    auto weight_scale_contig = weight_scale.contiguous();
    auto activation_scale_contig = activation_scale.reshape({1}).contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : bias;
    auto q_input = torch::empty(input_contig.sizes(), input_contig.options().dtype(torch::kInt8));
    auto acc = torch::empty({input.size(0), q_weight.size(0)}, input_contig.options().dtype(torch::kInt32));
    auto output = torch::empty({input.size(0), q_weight.size(0)}, input_contig.options());
    auto stream = c10::cuda::getCurrentCUDAStream();

    launch_quantize_activation_s8_kernel(
        input_contig.data_ptr<float>(),
        q_input.data_ptr<int8_t>(),
        activation_scale_contig.data_ptr<float>(),
        input.size(0),
        input.size(1),
        stream);
    auto status = launch_cutlass_s8s8s32_gemm(
        q_input.data_ptr<int8_t>(),
        q_weight_contig.data_ptr<int8_t>(),
        acc.data_ptr<int32_t>(),
        input.size(0),
        q_weight.size(0),
        input.size(1),
        stream);
    TORCH_CHECK(static_cast<int>(status) == 0, "quant_linear_w8a8_static_cutlass: CUTLASS GEMM failed");
    launch_dequantize_s32_kernel(
        acc.data_ptr<int32_t>(),
        weight_scale_contig.data_ptr<float>(),
        activation_scale_contig.data_ptr<float>(),
        has_bias ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        input.size(0),
        q_weight.size(0),
        has_bias,
        stream);
    return output;
}

torch::Tensor quant_linear_w8a8_static_wmma_dual_gated_tail_n128(const torch::Tensor &core_input,
                                                                 const torch::Tensor &gate_input,
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
                                                                 double eps) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: inputs must be CUDA");
    TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: weights must be CUDA");
    TORCH_CHECK(core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: scales must be CUDA");
    TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: activation scales must be CUDA");
    TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: norm params must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: scales must be float32");
    TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                    gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: norm params must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: inputs must be 2D");
    TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: weights must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: row count must match");
    TORCH_CHECK(core_input.size(1) == core_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: core K mismatch");
    TORCH_CHECK(gate_input.size(1) == gate_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: gate K mismatch");
    TORCH_CHECK(core_q_weight.size(0) == 128 && gate_q_weight.size(0) == 128,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: only N=128 is supported");
    TORCH_CHECK(core_input.size(1) % 16 == 0 && gate_input.size(1) % 16 == 0,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: K must be multiple of 16");
    TORCH_CHECK(core_weight_scale.numel() == 128 && gate_weight_scale.numel() == 128 &&
                    core_norm_weight.numel() == 128 && core_norm_bias.numel() == 128 &&
                    gate_norm_weight.numel() == 128 && gate_norm_bias.numel() == 128,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: scale/norm size mismatch");
    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.numel() == 128,
                    "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.numel() == 128,
                    "quant_linear_w8a8_static_wmma_dual_gated_tail_n128: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_q_weight_contig = core_q_weight.contiguous();
    auto gate_q_weight_contig = gate_q_weight.contiguous();
    auto core_weight_scale_contig = core_weight_scale.contiguous();
    auto gate_weight_scale_contig = gate_weight_scale.contiguous();
    auto core_activation_scale_contig = core_activation_scale.reshape({1}).contiguous();
    auto gate_activation_scale_contig = gate_activation_scale.reshape({1}).contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_norm_weight_contig = core_norm_weight.contiguous();
    auto core_norm_bias_contig = core_norm_bias.contiguous();
    auto gate_norm_weight_contig = gate_norm_weight.contiguous();
    auto gate_norm_bias_contig = gate_norm_bias.contiguous();
    auto output = torch::empty({core_input.size(0), 128}, core_input_contig.options());

    const bool use_parallel = use_w8a8_tail_n128_parallel_backend(core_input.size(0));
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (use_parallel) {
        launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(
            core_input_contig.data_ptr<float>(),
            gate_input_contig.data_ptr<float>(),
            core_q_weight_contig.data_ptr<int8_t>(),
            gate_q_weight_contig.data_ptr<int8_t>(),
            core_weight_scale_contig.data_ptr<float>(),
            gate_weight_scale_contig.data_ptr<float>(),
            core_activation_scale_contig.data_ptr<float>(),
            gate_activation_scale_contig.data_ptr<float>(),
            core_has_bias ? core_bias_contig.data_ptr<float>() : nullptr,
            gate_has_bias ? gate_bias_contig.data_ptr<float>() : nullptr,
            core_norm_weight_contig.data_ptr<float>(),
            core_norm_bias_contig.data_ptr<float>(),
            gate_norm_weight_contig.data_ptr<float>(),
            gate_norm_bias_contig.data_ptr<float>(),
            nullptr,
            nullptr,
            output.data_ptr<float>(),
            core_input.size(0),
            core_input.size(1),
            gate_input.size(1),
            core_has_bias,
            gate_has_bias,
            static_cast<float>(eps),
            stream);
    } else {
        launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(
            core_input_contig.data_ptr<float>(),
            gate_input_contig.data_ptr<float>(),
            core_q_weight_contig.data_ptr<int8_t>(),
            gate_q_weight_contig.data_ptr<int8_t>(),
            core_weight_scale_contig.data_ptr<float>(),
            gate_weight_scale_contig.data_ptr<float>(),
            core_activation_scale_contig.data_ptr<float>(),
            gate_activation_scale_contig.data_ptr<float>(),
            core_has_bias ? core_bias_contig.data_ptr<float>() : nullptr,
            gate_has_bias ? gate_bias_contig.data_ptr<float>() : nullptr,
            core_norm_weight_contig.data_ptr<float>(),
            core_norm_bias_contig.data_ptr<float>(),
            gate_norm_weight_contig.data_ptr<float>(),
            gate_norm_bias_contig.data_ptr<float>(),
            nullptr,
            nullptr,
            output.data_ptr<float>(),
            core_input.size(0),
            core_input.size(1),
            gate_input.size(1),
            core_has_bias,
            gate_has_bias,
            static_cast<float>(eps),
            stream);
    }
    return output;
}

torch::Tensor quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast(const torch::Tensor &core_input,
                                                                      const torch::Tensor &gate_input,
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
                                                                      double eps) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: inputs must be CUDA");
    TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: weights must be CUDA");
    TORCH_CHECK(core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: scales must be CUDA");
    TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: activation scales must be CUDA");
    TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: norm params must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32 &&
                    core_activation_scale.scalar_type() == torch::kFloat32 && gate_activation_scale.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: scales must be float32");
    TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                    gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: norm params must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: inputs must be 2D");
    TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: weights must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: row count must match");
    TORCH_CHECK(core_input.size(1) == core_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: core K mismatch");
    TORCH_CHECK(gate_input.size(1) == gate_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: gate K mismatch");
    TORCH_CHECK(core_q_weight.size(0) == 128 && gate_q_weight.size(0) == 128,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: only N=128 is supported");
    TORCH_CHECK(core_input.size(1) % 16 == 0 && gate_input.size(1) % 16 == 0,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: K must be multiple of 16");
    TORCH_CHECK(core_weight_scale.numel() == 128 && gate_weight_scale.numel() == 128 &&
                    core_activation_scale.numel() == 1 && gate_activation_scale.numel() == 1 &&
                    core_norm_weight.numel() == 128 && core_norm_bias.numel() == 128 &&
                    gate_norm_weight.numel() == 128 && gate_norm_bias.numel() == 128,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: scale/norm size mismatch");
    TORCH_CHECK(core_input.is_contiguous(), "core must be contiguous");
    TORCH_CHECK(gate_input.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(core_q_weight.is_contiguous(), "core_q_weight must be contiguous");
    TORCH_CHECK(gate_q_weight.is_contiguous(), "gate_q_weight must be contiguous");
    TORCH_CHECK(core_weight_scale.is_contiguous(), "core_weight_scale must be contiguous");
    TORCH_CHECK(gate_weight_scale.is_contiguous(), "gate_weight_scale must be contiguous");
    TORCH_CHECK(core_activation_scale.is_contiguous(), "core_activation_scale must be contiguous");
    TORCH_CHECK(gate_activation_scale.is_contiguous(), "gate_activation_scale must be contiguous");
    TORCH_CHECK(core_norm_weight.is_contiguous(), "core_norm_weight must be contiguous");
    TORCH_CHECK(core_norm_bias.is_contiguous(), "core_norm_bias must be contiguous");
    TORCH_CHECK(gate_norm_weight.is_contiguous(), "gate_norm_weight must be contiguous");
    TORCH_CHECK(gate_norm_bias.is_contiguous(), "gate_norm_bias must be contiguous");
    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.numel() == 128,
                    "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: invalid core bias");
        TORCH_CHECK(core_bias.is_contiguous(), "core_bias must be contiguous");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.numel() == 128,
                    "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast: invalid gate bias");
        TORCH_CHECK(gate_bias.is_contiguous(), "gate_bias must be contiguous");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto output = torch::empty({core_input.size(0), 128}, core_input.options());

    const bool use_parallel = use_w8a8_tail_n128_parallel_backend(core_input.size(0));
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (use_parallel) {
        launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(
            core_input.data_ptr<float>(),
            gate_input.data_ptr<float>(),
            core_q_weight.data_ptr<int8_t>(),
            gate_q_weight.data_ptr<int8_t>(),
            core_weight_scale.data_ptr<float>(),
            gate_weight_scale.data_ptr<float>(),
            core_activation_scale.data_ptr<float>(),
            gate_activation_scale.data_ptr<float>(),
            core_has_bias ? core_bias.data_ptr<float>() : nullptr,
            gate_has_bias ? gate_bias.data_ptr<float>() : nullptr,
            core_norm_weight.data_ptr<float>(),
            core_norm_bias.data_ptr<float>(),
            gate_norm_weight.data_ptr<float>(),
            gate_norm_bias.data_ptr<float>(),
            nullptr,
            nullptr,
            output.data_ptr<float>(),
            core_input.size(0),
            core_input.size(1),
            gate_input.size(1),
            core_has_bias,
            gate_has_bias,
            static_cast<float>(eps),
            stream);
    } else {
        launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(
            core_input.data_ptr<float>(),
            gate_input.data_ptr<float>(),
            core_q_weight.data_ptr<int8_t>(),
            gate_q_weight.data_ptr<int8_t>(),
            core_weight_scale.data_ptr<float>(),
            gate_weight_scale.data_ptr<float>(),
            core_activation_scale.data_ptr<float>(),
            gate_activation_scale.data_ptr<float>(),
            core_has_bias ? core_bias.data_ptr<float>() : nullptr,
            gate_has_bias ? gate_bias.data_ptr<float>() : nullptr,
            core_norm_weight.data_ptr<float>(),
            core_norm_bias.data_ptr<float>(),
            gate_norm_weight.data_ptr<float>(),
            gate_norm_bias.data_ptr<float>(),
            nullptr,
            nullptr,
            output.data_ptr<float>(),
            core_input.size(0),
            core_input.size(1),
            gate_input.size(1),
            core_has_bias,
            gate_has_bias,
            static_cast<float>(eps),
            stream);
    }
    return output;
}

std::vector<torch::Tensor> quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre(
    const torch::Tensor &core_input,
    const torch::Tensor &gate_input,
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
    double eps) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "w8a8 gated_tail_with_pre: inputs must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32,
                "w8a8 gated_tail_with_pre: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
                "w8a8 gated_tail_with_pre: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32 &&
                    core_activation_scale.scalar_type() == torch::kFloat32 && gate_activation_scale.scalar_type() == torch::kFloat32,
                "w8a8 gated_tail_with_pre: scales must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2 && core_input.size(1) == 128 && gate_input.size(1) == 128,
                "w8a8 gated_tail_with_pre: inputs must be [rows, 128]");
    TORCH_CHECK(core_q_weight.sizes() == torch::IntArrayRef({128, 128}) &&
                    gate_q_weight.sizes() == torch::IntArrayRef({128, 128}),
                "w8a8 gated_tail_with_pre: q weights must be [128, 128]");
    TORCH_CHECK(core_weight_scale.numel() == 128 && gate_weight_scale.numel() == 128 &&
                    core_activation_scale.numel() == 1 && gate_activation_scale.numel() == 1,
                "w8a8 gated_tail_with_pre: invalid scale shapes");
    TORCH_CHECK(core_norm_weight.numel() == 128 && core_norm_bias.numel() == 128 &&
                    gate_norm_weight.numel() == 128 && gate_norm_bias.numel() == 128,
                "w8a8 gated_tail_with_pre: invalid norm shapes");

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_c = core_input.contiguous();
    auto gate_input_c = gate_input.contiguous();
    auto core_q_weight_c = core_q_weight.contiguous();
    auto gate_q_weight_c = gate_q_weight.contiguous();
    auto core_weight_scale_c = core_weight_scale.contiguous();
    auto gate_weight_scale_c = gate_weight_scale.contiguous();
    auto core_activation_scale_c = core_activation_scale.reshape({1}).contiguous();
    auto gate_activation_scale_c = gate_activation_scale.reshape({1}).contiguous();
    auto core_bias_c = core_has_bias ? core_bias.contiguous() : core_input_c.new_empty({0});
    auto gate_bias_c = gate_has_bias ? gate_bias.contiguous() : gate_input_c.new_empty({0});
    auto core_norm_weight_c = core_norm_weight.contiguous();
    auto core_norm_bias_c = core_norm_bias.contiguous();
    auto gate_norm_weight_c = gate_norm_weight.contiguous();
    auto gate_norm_bias_c = gate_norm_bias.contiguous();
    auto output = torch::empty({core_input_c.size(0), 128}, core_input_c.options());
    auto core_pre = torch::empty_like(output);
    auto gate_pre = torch::empty_like(output);

    const bool use_parallel = use_w8a8_tail_n128_parallel_backend(core_input_c.size(0));
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (use_parallel) {
        launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_parallel_kernel(
            core_input_c.data_ptr<float>(), gate_input_c.data_ptr<float>(),
            core_q_weight_c.data_ptr<int8_t>(), gate_q_weight_c.data_ptr<int8_t>(),
            core_weight_scale_c.data_ptr<float>(), gate_weight_scale_c.data_ptr<float>(),
            core_activation_scale_c.data_ptr<float>(), gate_activation_scale_c.data_ptr<float>(),
            core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
            gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
            core_norm_weight_c.data_ptr<float>(), core_norm_bias_c.data_ptr<float>(),
            gate_norm_weight_c.data_ptr<float>(), gate_norm_bias_c.data_ptr<float>(),
            core_pre.data_ptr<float>(), gate_pre.data_ptr<float>(), output.data_ptr<float>(),
            core_input_c.size(0), core_input_c.size(1), gate_input_c.size(1),
            core_has_bias, gate_has_bias, static_cast<float>(eps), stream);
    } else {
        launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_kernel(
            core_input_c.data_ptr<float>(), gate_input_c.data_ptr<float>(),
            core_q_weight_c.data_ptr<int8_t>(), gate_q_weight_c.data_ptr<int8_t>(),
            core_weight_scale_c.data_ptr<float>(), gate_weight_scale_c.data_ptr<float>(),
            core_activation_scale_c.data_ptr<float>(), gate_activation_scale_c.data_ptr<float>(),
            core_has_bias ? core_bias_c.data_ptr<float>() : nullptr,
            gate_has_bias ? gate_bias_c.data_ptr<float>() : nullptr,
            core_norm_weight_c.data_ptr<float>(), core_norm_bias_c.data_ptr<float>(),
            gate_norm_weight_c.data_ptr<float>(), gate_norm_bias_c.data_ptr<float>(),
            core_pre.data_ptr<float>(), gate_pre.data_ptr<float>(), output.data_ptr<float>(),
            core_input_c.size(0), core_input_c.size(1), gate_input_c.size(1),
            core_has_bias, gate_has_bias, static_cast<float>(eps), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, core_pre, gate_pre};
}

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_dual(const torch::Tensor &core_input,
                                                                 const torch::Tensor &gate_input,
                                                                 const torch::Tensor &core_q_weight,
                                                                 const torch::Tensor &gate_q_weight,
                                                                 const torch::Tensor &core_weight_scale,
                                                                 const torch::Tensor &gate_weight_scale,
                                                                 const torch::Tensor &core_activation_scale,
                                                                 const torch::Tensor &gate_activation_scale,
                                                                 const torch::Tensor &core_bias,
                                                                 const torch::Tensor &gate_bias,
                                                                 bool core_has_bias,
                                                                 bool gate_has_bias) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "quant_linear_w8a8_static_cutlass_dual: inputs must be CUDA");
    TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda(), "quant_linear_w8a8_static_cutlass_dual: weights must be CUDA");
    TORCH_CHECK(core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(), "quant_linear_w8a8_static_cutlass_dual: weight scales must be CUDA");
    TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(), "quant_linear_w8a8_static_cutlass_dual: activation scales must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass_dual: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a8_static_cutlass_dual: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass_dual: scales must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2, "quant_linear_w8a8_static_cutlass_dual: inputs must be 2D");
    TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2, "quant_linear_w8a8_static_cutlass_dual: weights must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0), "quant_linear_w8a8_static_cutlass_dual: row count must match");
    TORCH_CHECK(core_input.size(1) == core_q_weight.size(1), "quant_linear_w8a8_static_cutlass_dual: core K mismatch");
    TORCH_CHECK(gate_input.size(1) == gate_q_weight.size(1), "quant_linear_w8a8_static_cutlass_dual: gate K mismatch");
    TORCH_CHECK(core_weight_scale.size(0) == core_q_weight.size(0), "quant_linear_w8a8_static_cutlass_dual: core scale mismatch");
    TORCH_CHECK(gate_weight_scale.size(0) == gate_q_weight.size(0), "quant_linear_w8a8_static_cutlass_dual: gate scale mismatch");
    TORCH_CHECK(core_input.size(1) % 32 == 0 && gate_input.size(1) % 32 == 0, "quant_linear_w8a8_static_cutlass_dual: K must be multiple of 32");

    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.size(0) == core_q_weight.size(0),
                    "quant_linear_w8a8_static_cutlass_dual: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.size(0) == gate_q_weight.size(0),
                    "quant_linear_w8a8_static_cutlass_dual: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_q_weight_contig = core_q_weight.contiguous();
    auto gate_q_weight_contig = gate_q_weight.contiguous();
    auto core_weight_scale_contig = core_weight_scale.contiguous();
    auto gate_weight_scale_contig = gate_weight_scale.contiguous();
    auto core_activation_scale_contig = core_activation_scale.reshape({1}).contiguous();
    auto gate_activation_scale_contig = gate_activation_scale.reshape({1}).contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_q_input = torch::empty(core_input_contig.sizes(), core_input_contig.options().dtype(torch::kInt8));
    auto gate_q_input = torch::empty(gate_input_contig.sizes(), gate_input_contig.options().dtype(torch::kInt8));
    auto core_acc = torch::empty({core_input.size(0), core_q_weight.size(0)}, core_input_contig.options().dtype(torch::kInt32));
    auto gate_acc = torch::empty({gate_input.size(0), gate_q_weight.size(0)}, gate_input_contig.options().dtype(torch::kInt32));
    auto core_output = torch::empty({core_input.size(0), core_q_weight.size(0)}, core_input_contig.options());
    auto gate_output = torch::empty({gate_input.size(0), gate_q_weight.size(0)}, gate_input_contig.options());
    auto stream = c10::cuda::getCurrentCUDAStream();

    launch_quantize_activation_s8_dual_kernel(
        core_input_contig.data_ptr<float>(),
        gate_input_contig.data_ptr<float>(),
        core_q_input.data_ptr<int8_t>(),
        gate_q_input.data_ptr<int8_t>(),
        core_activation_scale_contig.data_ptr<float>(),
        gate_activation_scale_contig.data_ptr<float>(),
        core_input.size(0),
        core_input.size(1),
        gate_input.size(1),
        stream);
    auto core_status = launch_cutlass_s8s8s32_gemm(
        core_q_input.data_ptr<int8_t>(),
        core_q_weight_contig.data_ptr<int8_t>(),
        core_acc.data_ptr<int32_t>(),
        core_input.size(0),
        core_q_weight.size(0),
        core_input.size(1),
        stream);
    TORCH_CHECK(static_cast<int>(core_status) == 0, "quant_linear_w8a8_static_cutlass_dual: core CUTLASS GEMM failed");
    auto gate_status = launch_cutlass_s8s8s32_gemm(
        gate_q_input.data_ptr<int8_t>(),
        gate_q_weight_contig.data_ptr<int8_t>(),
        gate_acc.data_ptr<int32_t>(),
        gate_input.size(0),
        gate_q_weight.size(0),
        gate_input.size(1),
        stream);
    TORCH_CHECK(static_cast<int>(gate_status) == 0, "quant_linear_w8a8_static_cutlass_dual: gate CUTLASS GEMM failed");
    launch_dequantize_s32_dual_kernel(
        core_acc.data_ptr<int32_t>(),
        gate_acc.data_ptr<int32_t>(),
        core_weight_scale_contig.data_ptr<float>(),
        gate_weight_scale_contig.data_ptr<float>(),
        core_activation_scale_contig.data_ptr<float>(),
        gate_activation_scale_contig.data_ptr<float>(),
        core_has_bias ? core_bias_contig.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_contig.data_ptr<float>() : nullptr,
        core_output.data_ptr<float>(),
        gate_output.data_ptr<float>(),
        core_input.size(0),
        core_q_weight.size(0),
        gate_q_weight.size(0),
        core_has_bias,
        gate_has_bias,
        stream);
    return {core_output, gate_output};
}

torch::Tensor quant_linear_w8a8_static_cutlass_dual_gated_tail(const torch::Tensor &core_input,
                                                               const torch::Tensor &gate_input,
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
                                                               double eps) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "quant_linear_w8a8_static_cutlass_dual_gated_tail: inputs must be CUDA");
    TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda(), "quant_linear_w8a8_static_cutlass_dual_gated_tail: weights must be CUDA");
    TORCH_CHECK(core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(), "quant_linear_w8a8_static_cutlass_dual_gated_tail: weight scales must be CUDA");
    TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(), "quant_linear_w8a8_static_cutlass_dual_gated_tail: activation scales must be CUDA");
    TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
                "quant_linear_w8a8_static_cutlass_dual_gated_tail: norm params must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_cutlass_dual_gated_tail: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
                "quant_linear_w8a8_static_cutlass_dual_gated_tail: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_cutlass_dual_gated_tail: scales must be float32");
    TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                    gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_cutlass_dual_gated_tail: norm params must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2, "quant_linear_w8a8_static_cutlass_dual_gated_tail: inputs must be 2D");
    TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2, "quant_linear_w8a8_static_cutlass_dual_gated_tail: weights must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0), "quant_linear_w8a8_static_cutlass_dual_gated_tail: row count must match");
    TORCH_CHECK(core_input.size(1) == core_q_weight.size(1), "quant_linear_w8a8_static_cutlass_dual_gated_tail: core K mismatch");
    TORCH_CHECK(gate_input.size(1) == gate_q_weight.size(1), "quant_linear_w8a8_static_cutlass_dual_gated_tail: gate K mismatch");
    TORCH_CHECK(core_q_weight.size(0) == gate_q_weight.size(0), "quant_linear_w8a8_static_cutlass_dual_gated_tail: core/gate N must match");
    TORCH_CHECK(core_weight_scale.size(0) == core_q_weight.size(0), "quant_linear_w8a8_static_cutlass_dual_gated_tail: core scale mismatch");
    TORCH_CHECK(gate_weight_scale.size(0) == gate_q_weight.size(0), "quant_linear_w8a8_static_cutlass_dual_gated_tail: gate scale mismatch");
    TORCH_CHECK(core_norm_weight.numel() == core_q_weight.size(0) && core_norm_bias.numel() == core_q_weight.size(0) &&
                    gate_norm_weight.numel() == gate_q_weight.size(0) && gate_norm_bias.numel() == gate_q_weight.size(0),
                "quant_linear_w8a8_static_cutlass_dual_gated_tail: norm size mismatch");
    TORCH_CHECK(core_input.size(1) % 32 == 0 && gate_input.size(1) % 32 == 0,
                "quant_linear_w8a8_static_cutlass_dual_gated_tail: K must be multiple of 32");

    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.size(0) == core_q_weight.size(0),
                    "quant_linear_w8a8_static_cutlass_dual_gated_tail: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.size(0) == gate_q_weight.size(0),
                    "quant_linear_w8a8_static_cutlass_dual_gated_tail: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_q_weight_contig = core_q_weight.contiguous();
    auto gate_q_weight_contig = gate_q_weight.contiguous();
    auto core_weight_scale_contig = core_weight_scale.contiguous();
    auto gate_weight_scale_contig = gate_weight_scale.contiguous();
    auto core_activation_scale_contig = core_activation_scale.reshape({1}).contiguous();
    auto gate_activation_scale_contig = gate_activation_scale.reshape({1}).contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_norm_weight_contig = core_norm_weight.contiguous();
    auto core_norm_bias_contig = core_norm_bias.contiguous();
    auto gate_norm_weight_contig = gate_norm_weight.contiguous();
    auto gate_norm_bias_contig = gate_norm_bias.contiguous();
    auto core_q_input = torch::empty(core_input_contig.sizes(), core_input_contig.options().dtype(torch::kInt8));
    auto gate_q_input = torch::empty(gate_input_contig.sizes(), gate_input_contig.options().dtype(torch::kInt8));
    auto core_acc = torch::empty({core_input.size(0), core_q_weight.size(0)}, core_input_contig.options().dtype(torch::kInt32));
    auto gate_acc = torch::empty({gate_input.size(0), gate_q_weight.size(0)}, gate_input_contig.options().dtype(torch::kInt32));
    auto output = torch::empty({core_input.size(0), core_q_weight.size(0)}, core_input_contig.options());
    auto stream = c10::cuda::getCurrentCUDAStream();

    launch_quantize_activation_s8_dual_kernel(
        core_input_contig.data_ptr<float>(),
        gate_input_contig.data_ptr<float>(),
        core_q_input.data_ptr<int8_t>(),
        gate_q_input.data_ptr<int8_t>(),
        core_activation_scale_contig.data_ptr<float>(),
        gate_activation_scale_contig.data_ptr<float>(),
        core_input.size(0),
        core_input.size(1),
        gate_input.size(1),
        stream);
    auto core_status = launch_cutlass_s8s8s32_gemm(
        core_q_input.data_ptr<int8_t>(),
        core_q_weight_contig.data_ptr<int8_t>(),
        core_acc.data_ptr<int32_t>(),
        core_input.size(0),
        core_q_weight.size(0),
        core_input.size(1),
        stream);
    TORCH_CHECK(static_cast<int>(core_status) == 0, "quant_linear_w8a8_static_cutlass_dual_gated_tail: core CUTLASS GEMM failed");
    auto gate_status = launch_cutlass_s8s8s32_gemm(
        gate_q_input.data_ptr<int8_t>(),
        gate_q_weight_contig.data_ptr<int8_t>(),
        gate_acc.data_ptr<int32_t>(),
        gate_input.size(0),
        gate_q_weight.size(0),
        gate_input.size(1),
        stream);
    TORCH_CHECK(static_cast<int>(gate_status) == 0, "quant_linear_w8a8_static_cutlass_dual_gated_tail: gate CUTLASS GEMM failed");

    launch_dequantize_s32_dual_gated_tail_kernel(
        core_acc.data_ptr<int32_t>(),
        gate_acc.data_ptr<int32_t>(),
        core_weight_scale_contig.data_ptr<float>(),
        gate_weight_scale_contig.data_ptr<float>(),
        core_activation_scale_contig.data_ptr<float>(),
        gate_activation_scale_contig.data_ptr<float>(),
        core_has_bias ? core_bias_contig.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_contig.data_ptr<float>() : nullptr,
        core_norm_weight_contig.data_ptr<float>(),
        core_norm_bias_contig.data_ptr<float>(),
        gate_norm_weight_contig.data_ptr<float>(),
        gate_norm_bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        core_input.size(0),
        core_q_weight.size(0),
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        stream);
    return output;
}

std::vector<torch::Tensor> quant_linear_w8a8_static_wmma_dual(const torch::Tensor &core_input,
                                                              const torch::Tensor &gate_input,
                                                              const torch::Tensor &core_q_weight,
                                                              const torch::Tensor &gate_q_weight,
                                                              const torch::Tensor &core_weight_scale,
                                                              const torch::Tensor &gate_weight_scale,
                                                              const torch::Tensor &core_activation_scale,
                                                              const torch::Tensor &gate_activation_scale,
                                                              const torch::Tensor &core_bias,
                                                              const torch::Tensor &gate_bias,
                                                              bool core_has_bias,
                                                              bool gate_has_bias) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "quant_linear_w8a8_static_wmma_dual: inputs must be CUDA");
    TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda(), "quant_linear_w8a8_static_wmma_dual: weights must be CUDA");
    TORCH_CHECK(core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual: weight scales must be CUDA");
    TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual: activation scales must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_wmma_dual: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a8_static_wmma_dual: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_wmma_dual: scales must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2, "quant_linear_w8a8_static_wmma_dual: inputs must be 2D");
    TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2, "quant_linear_w8a8_static_wmma_dual: weights must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0), "quant_linear_w8a8_static_wmma_dual: row count must match");
    TORCH_CHECK(core_input.size(1) == core_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual: core K mismatch");
    TORCH_CHECK(gate_input.size(1) == gate_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual: gate K mismatch");
    TORCH_CHECK(core_weight_scale.size(0) == core_q_weight.size(0), "quant_linear_w8a8_static_wmma_dual: core scale mismatch");
    TORCH_CHECK(gate_weight_scale.size(0) == gate_q_weight.size(0), "quant_linear_w8a8_static_wmma_dual: gate scale mismatch");
    TORCH_CHECK(core_input.size(1) % 16 == 0 && gate_input.size(1) % 16 == 0, "quant_linear_w8a8_static_wmma_dual: K must be multiple of 16");
    TORCH_CHECK(core_q_weight.size(0) % 16 == 0 && gate_q_weight.size(0) % 16 == 0, "quant_linear_w8a8_static_wmma_dual: N must be multiple of 16");

    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.size(0) == core_q_weight.size(0),
                    "quant_linear_w8a8_static_wmma_dual: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.size(0) == gate_q_weight.size(0),
                    "quant_linear_w8a8_static_wmma_dual: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_q_weight_contig = core_q_weight.contiguous();
    auto gate_q_weight_contig = gate_q_weight.contiguous();
    auto core_weight_scale_contig = core_weight_scale.contiguous();
    auto gate_weight_scale_contig = gate_weight_scale.contiguous();
    auto core_activation_scale_contig = core_activation_scale.reshape({1}).contiguous();
    auto gate_activation_scale_contig = gate_activation_scale.reshape({1}).contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_output = torch::empty({core_input.size(0), core_q_weight.size(0)}, core_input.options());
    auto gate_output = torch::empty({gate_input.size(0), gate_q_weight.size(0)}, gate_input.options());

    launch_quant_linear_w8a8_wmma_dual_kernel(
        core_input_contig.data_ptr<float>(),
        gate_input_contig.data_ptr<float>(),
        core_q_weight_contig.data_ptr<int8_t>(),
        gate_q_weight_contig.data_ptr<int8_t>(),
        core_weight_scale_contig.data_ptr<float>(),
        gate_weight_scale_contig.data_ptr<float>(),
        core_activation_scale_contig.data_ptr<float>(),
        gate_activation_scale_contig.data_ptr<float>(),
        core_has_bias ? core_bias_contig.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_contig.data_ptr<float>() : nullptr,
        core_output.data_ptr<float>(),
        gate_output.data_ptr<float>(),
        core_input.size(0),
        core_q_weight.size(0),
        gate_q_weight.size(0),
        core_input.size(1),
        gate_input.size(1),
        core_has_bias,
        gate_has_bias,
        c10::cuda::getCurrentCUDAStream());
    return {core_output, gate_output};
}

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_grouped_dual(const torch::Tensor &core_input,
                                                                         const torch::Tensor &gate_input,
                                                                         const torch::Tensor &core_q_weight,
                                                                         const torch::Tensor &gate_q_weight,
                                                                         const torch::Tensor &core_weight_scale,
                                                                         const torch::Tensor &gate_weight_scale,
                                                                         const torch::Tensor &core_activation_scale,
                                                                         const torch::Tensor &gate_activation_scale,
                                                                         const torch::Tensor &core_bias,
                                                                         const torch::Tensor &gate_bias,
                                                                         bool core_has_bias,
                                                                         bool gate_has_bias) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_dual: inputs must be CUDA");
    TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_dual: weights must be CUDA");
    TORCH_CHECK(core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_dual: weight scales must be CUDA");
    TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_dual: activation scales must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass_grouped_dual: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a8_static_cutlass_grouped_dual: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_cutlass_grouped_dual: scales must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2, "quant_linear_w8a8_static_cutlass_grouped_dual: inputs must be 2D");
    TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2, "quant_linear_w8a8_static_cutlass_grouped_dual: weights must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0), "quant_linear_w8a8_static_cutlass_grouped_dual: row count must match");
    TORCH_CHECK(core_input.size(1) == core_q_weight.size(1), "quant_linear_w8a8_static_cutlass_grouped_dual: core K mismatch");
    TORCH_CHECK(gate_input.size(1) == gate_q_weight.size(1), "quant_linear_w8a8_static_cutlass_grouped_dual: gate K mismatch");
    TORCH_CHECK(core_weight_scale.size(0) == core_q_weight.size(0), "quant_linear_w8a8_static_cutlass_grouped_dual: core scale mismatch");
    TORCH_CHECK(gate_weight_scale.size(0) == gate_q_weight.size(0), "quant_linear_w8a8_static_cutlass_grouped_dual: gate scale mismatch");
    TORCH_CHECK(core_input.size(1) % 32 == 0 && gate_input.size(1) % 32 == 0, "quant_linear_w8a8_static_cutlass_grouped_dual: K must be multiple of 32");

    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.size(0) == core_q_weight.size(0),
                    "quant_linear_w8a8_static_cutlass_grouped_dual: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.size(0) == gate_q_weight.size(0),
                    "quant_linear_w8a8_static_cutlass_grouped_dual: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_q_weight_contig = core_q_weight.contiguous();
    auto gate_q_weight_contig = gate_q_weight.contiguous();
    auto core_weight_scale_contig = core_weight_scale.contiguous();
    auto gate_weight_scale_contig = gate_weight_scale.contiguous();
    auto core_activation_scale_contig = core_activation_scale.reshape({1}).contiguous();
    auto gate_activation_scale_contig = gate_activation_scale.reshape({1}).contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_q_input = torch::empty(core_input_contig.sizes(), core_input_contig.options().dtype(torch::kInt8));
    auto gate_q_input = torch::empty(gate_input_contig.sizes(), gate_input_contig.options().dtype(torch::kInt8));
    auto core_acc = torch::empty({core_input.size(0), core_q_weight.size(0)}, core_input_contig.options().dtype(torch::kInt32));
    auto gate_acc = torch::empty({gate_input.size(0), gate_q_weight.size(0)}, gate_input_contig.options().dtype(torch::kInt32));
    auto core_output = torch::empty({core_input.size(0), core_q_weight.size(0)}, core_input_contig.options());
    auto gate_output = torch::empty({gate_input.size(0), gate_q_weight.size(0)}, gate_input_contig.options());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto int32_options = core_input.options().dtype(torch::kInt32);
    auto int64_options = core_input.options().dtype(torch::kInt64);
    auto problem_sizes_device = torch::empty({2, 3}, int32_options);
    auto ptr_a_device = torch::empty({2}, int64_options);
    auto ptr_b_device = torch::empty({2}, int64_options);
    auto ptr_c_device = torch::empty({2}, int64_options);
    auto ptr_d_device = torch::empty({2}, int64_options);
    auto lda_device = torch::empty({2}, int64_options);
    auto ldb_device = torch::empty({2}, int64_options);
    auto ldc_device = torch::empty({2}, int64_options);
    auto ldd_device = torch::empty({2}, int64_options);

    int32_t problem_sizes_host[6] = {
        static_cast<int32_t>(core_input.size(0)),
        static_cast<int32_t>(core_q_weight.size(0)),
        static_cast<int32_t>(core_input.size(1)),
        static_cast<int32_t>(gate_input.size(0)),
        static_cast<int32_t>(gate_q_weight.size(0)),
        static_cast<int32_t>(gate_input.size(1)),
    };
    launch_quantize_activation_s8_dual_meta_kernel(
        core_input_contig.data_ptr<float>(),
        gate_input_contig.data_ptr<float>(),
        core_q_input.data_ptr<int8_t>(),
        gate_q_input.data_ptr<int8_t>(),
        core_q_weight_contig.data_ptr<int8_t>(),
        gate_q_weight_contig.data_ptr<int8_t>(),
        core_acc.data_ptr<int32_t>(),
        gate_acc.data_ptr<int32_t>(),
        problem_sizes_device.data_ptr<int32_t>(),
        ptr_a_device.data_ptr<int64_t>(),
        ptr_b_device.data_ptr<int64_t>(),
        ptr_c_device.data_ptr<int64_t>(),
        ptr_d_device.data_ptr<int64_t>(),
        lda_device.data_ptr<int64_t>(),
        ldb_device.data_ptr<int64_t>(),
        ldc_device.data_ptr<int64_t>(),
        ldd_device.data_ptr<int64_t>(),
        core_activation_scale_contig.data_ptr<float>(),
        gate_activation_scale_contig.data_ptr<float>(),
        core_input.size(0),
        core_input.size(1),
        gate_input.size(1),
        core_q_weight.size(0),
        gate_q_weight.size(0),
        stream);

    auto status = launch_cutlass_s8s8s32_grouped_gemm(
        problem_sizes_device.data_ptr<int32_t>(),
        problem_sizes_host,
        reinterpret_cast<int8_t **>(ptr_a_device.data_ptr<int64_t>()),
        reinterpret_cast<int8_t **>(ptr_b_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_c_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_d_device.data_ptr<int64_t>()),
        lda_device.data_ptr<int64_t>(),
        ldb_device.data_ptr<int64_t>(),
        ldc_device.data_ptr<int64_t>(),
        ldd_device.data_ptr<int64_t>(),
        2,
        stream);
    TORCH_CHECK(static_cast<int>(status) == 0, "quant_linear_w8a8_static_cutlass_grouped_dual: grouped CUTLASS GEMM failed");

    launch_dequantize_s32_dual_kernel(
        core_acc.data_ptr<int32_t>(),
        gate_acc.data_ptr<int32_t>(),
        core_weight_scale_contig.data_ptr<float>(),
        gate_weight_scale_contig.data_ptr<float>(),
        core_activation_scale_contig.data_ptr<float>(),
        gate_activation_scale_contig.data_ptr<float>(),
        core_has_bias ? core_bias_contig.data_ptr<float>() : nullptr,
        gate_has_bias ? gate_bias_contig.data_ptr<float>() : nullptr,
        core_output.data_ptr<float>(),
        gate_output.data_ptr<float>(),
        core_input.size(0),
        core_q_weight.size(0),
        gate_q_weight.size(0),
        core_has_bias,
        gate_has_bias,
        stream);
    return {core_output, gate_output};
}

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_grouped_pair(const torch::Tensor &input_a,
                                                                         const torch::Tensor &input_b,
                                                                         const torch::Tensor &q_weight_a,
                                                                         const torch::Tensor &q_weight_b,
                                                                         const torch::Tensor &weight_scale_a,
                                                                         const torch::Tensor &weight_scale_b,
                                                                         const torch::Tensor &activation_scale_a,
                                                                         const torch::Tensor &activation_scale_b,
                                                                         const torch::Tensor &bias_a,
                                                                         const torch::Tensor &bias_b,
                                                                         bool has_bias_a,
                                                                         bool has_bias_b) {
    TORCH_CHECK(input_a.is_cuda() && input_b.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_pair: inputs must be CUDA");
    TORCH_CHECK(q_weight_a.is_cuda() && q_weight_b.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_pair: weights must be CUDA");
    TORCH_CHECK(weight_scale_a.is_cuda() && weight_scale_b.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_pair: weight scales must be CUDA");
    TORCH_CHECK(activation_scale_a.is_cuda() && activation_scale_b.is_cuda(), "quant_linear_w8a8_static_cutlass_grouped_pair: activation scales must be CUDA");
    TORCH_CHECK(input_a.scalar_type() == torch::kFloat32 && input_b.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_cutlass_grouped_pair: inputs must be float32");
    TORCH_CHECK(q_weight_a.scalar_type() == torch::kInt8 && q_weight_b.scalar_type() == torch::kInt8,
                "quant_linear_w8a8_static_cutlass_grouped_pair: weights must be int8");
    TORCH_CHECK(weight_scale_a.scalar_type() == torch::kFloat32 && weight_scale_b.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_cutlass_grouped_pair: scales must be float32");
    TORCH_CHECK(input_a.dim() == 2 && input_b.dim() == 2, "quant_linear_w8a8_static_cutlass_grouped_pair: inputs must be 2D");
    TORCH_CHECK(q_weight_a.dim() == 2 && q_weight_b.dim() == 2, "quant_linear_w8a8_static_cutlass_grouped_pair: weights must be 2D");
    TORCH_CHECK(input_a.size(1) == q_weight_a.size(1), "quant_linear_w8a8_static_cutlass_grouped_pair: A K mismatch");
    TORCH_CHECK(input_b.size(1) == q_weight_b.size(1), "quant_linear_w8a8_static_cutlass_grouped_pair: B K mismatch");
    TORCH_CHECK(weight_scale_a.size(0) == q_weight_a.size(0), "quant_linear_w8a8_static_cutlass_grouped_pair: A scale mismatch");
    TORCH_CHECK(weight_scale_b.size(0) == q_weight_b.size(0), "quant_linear_w8a8_static_cutlass_grouped_pair: B scale mismatch");
    TORCH_CHECK(input_a.size(1) % 32 == 0 && input_b.size(1) % 32 == 0,
                "quant_linear_w8a8_static_cutlass_grouped_pair: K must be multiple of 32");

    if (has_bias_a) {
        TORCH_CHECK(bias_a.is_cuda() && bias_a.scalar_type() == torch::kFloat32 && bias_a.size(0) == q_weight_a.size(0),
                    "quant_linear_w8a8_static_cutlass_grouped_pair: invalid A bias");
    }
    if (has_bias_b) {
        TORCH_CHECK(bias_b.is_cuda() && bias_b.scalar_type() == torch::kFloat32 && bias_b.size(0) == q_weight_b.size(0),
                    "quant_linear_w8a8_static_cutlass_grouped_pair: invalid B bias");
    }

    const c10::cuda::CUDAGuard device_guard(input_a.device());
    auto input_a_contig = input_a.contiguous();
    auto input_b_contig = input_b.contiguous();
    auto q_weight_a_contig = q_weight_a.contiguous();
    auto q_weight_b_contig = q_weight_b.contiguous();
    auto weight_scale_a_contig = weight_scale_a.contiguous();
    auto weight_scale_b_contig = weight_scale_b.contiguous();
    auto activation_scale_a_contig = activation_scale_a.reshape({1}).contiguous();
    auto activation_scale_b_contig = activation_scale_b.reshape({1}).contiguous();
    auto bias_a_contig = has_bias_a ? bias_a.contiguous() : bias_a;
    auto bias_b_contig = has_bias_b ? bias_b.contiguous() : bias_b;
    auto q_input_a = torch::empty(input_a_contig.sizes(), input_a_contig.options().dtype(torch::kInt8));
    auto q_input_b = torch::empty(input_b_contig.sizes(), input_b_contig.options().dtype(torch::kInt8));
    auto acc_a = torch::empty({input_a.size(0), q_weight_a.size(0)}, input_a_contig.options().dtype(torch::kInt32));
    auto acc_b = torch::empty({input_b.size(0), q_weight_b.size(0)}, input_b_contig.options().dtype(torch::kInt32));
    auto output_a = torch::empty({input_a.size(0), q_weight_a.size(0)}, input_a_contig.options());
    auto output_b = torch::empty({input_b.size(0), q_weight_b.size(0)}, input_b_contig.options());
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto int32_options = input_a.options().dtype(torch::kInt32);
    auto int64_options = input_a.options().dtype(torch::kInt64);
    auto problem_sizes_device = torch::empty({2, 3}, int32_options);
    auto ptr_a_device = torch::empty({2}, int64_options);
    auto ptr_b_device = torch::empty({2}, int64_options);
    auto ptr_c_device = torch::empty({2}, int64_options);
    auto ptr_d_device = torch::empty({2}, int64_options);
    auto lda_device = torch::empty({2}, int64_options);
    auto ldb_device = torch::empty({2}, int64_options);
    auto ldc_device = torch::empty({2}, int64_options);
    auto ldd_device = torch::empty({2}, int64_options);

    int32_t problem_sizes_host[6] = {
        static_cast<int32_t>(input_a.size(0)),
        static_cast<int32_t>(q_weight_a.size(0)),
        static_cast<int32_t>(input_a.size(1)),
        static_cast<int32_t>(input_b.size(0)),
        static_cast<int32_t>(q_weight_b.size(0)),
        static_cast<int32_t>(input_b.size(1)),
    };
    launch_quantize_activation_s8_pair_meta_kernel(
        input_a_contig.data_ptr<float>(),
        input_b_contig.data_ptr<float>(),
        q_input_a.data_ptr<int8_t>(),
        q_input_b.data_ptr<int8_t>(),
        q_weight_a_contig.data_ptr<int8_t>(),
        q_weight_b_contig.data_ptr<int8_t>(),
        acc_a.data_ptr<int32_t>(),
        acc_b.data_ptr<int32_t>(),
        problem_sizes_device.data_ptr<int32_t>(),
        ptr_a_device.data_ptr<int64_t>(),
        ptr_b_device.data_ptr<int64_t>(),
        ptr_c_device.data_ptr<int64_t>(),
        ptr_d_device.data_ptr<int64_t>(),
        lda_device.data_ptr<int64_t>(),
        ldb_device.data_ptr<int64_t>(),
        ldc_device.data_ptr<int64_t>(),
        ldd_device.data_ptr<int64_t>(),
        activation_scale_a_contig.data_ptr<float>(),
        activation_scale_b_contig.data_ptr<float>(),
        input_a.size(0),
        input_b.size(0),
        input_a.size(1),
        input_b.size(1),
        q_weight_a.size(0),
        q_weight_b.size(0),
        stream);

    auto status = launch_cutlass_s8s8s32_grouped_gemm(
        problem_sizes_device.data_ptr<int32_t>(),
        problem_sizes_host,
        reinterpret_cast<int8_t **>(ptr_a_device.data_ptr<int64_t>()),
        reinterpret_cast<int8_t **>(ptr_b_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_c_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_d_device.data_ptr<int64_t>()),
        lda_device.data_ptr<int64_t>(),
        ldb_device.data_ptr<int64_t>(),
        ldc_device.data_ptr<int64_t>(),
        ldd_device.data_ptr<int64_t>(),
        2,
        stream);
    TORCH_CHECK(static_cast<int>(status) == 0, "quant_linear_w8a8_static_cutlass_grouped_pair: grouped CUTLASS GEMM failed");

    launch_dequantize_s32_pair_kernel(
        acc_a.data_ptr<int32_t>(),
        acc_b.data_ptr<int32_t>(),
        weight_scale_a_contig.data_ptr<float>(),
        weight_scale_b_contig.data_ptr<float>(),
        activation_scale_a_contig.data_ptr<float>(),
        activation_scale_b_contig.data_ptr<float>(),
        has_bias_a ? bias_a_contig.data_ptr<float>() : nullptr,
        has_bias_b ? bias_b_contig.data_ptr<float>() : nullptr,
        output_a.data_ptr<float>(),
        output_b.data_ptr<float>(),
        input_a.size(0),
        input_b.size(0),
        q_weight_a.size(0),
        q_weight_b.size(0),
        has_bias_a,
        has_bias_b,
        stream);
    return {output_a, output_b};
}

std::vector<torch::Tensor> quant_ffn_w8a8_static_cutlass_grouped_pair(const torch::Tensor &input_a,
                                                                      const torch::Tensor &input_b,
                                                                      const torch::Tensor &q_weight1_a,
                                                                      const torch::Tensor &q_weight1_b,
                                                                      const torch::Tensor &weight_scale1_a,
                                                                      const torch::Tensor &weight_scale1_b,
                                                                      const torch::Tensor &activation_scale1_a,
                                                                      const torch::Tensor &activation_scale1_b,
                                                                      const torch::Tensor &bias1_a,
                                                                      const torch::Tensor &bias1_b,
                                                                      bool has_bias1_a,
                                                                      bool has_bias1_b,
                                                                      const torch::Tensor &q_weight2_a,
                                                                      const torch::Tensor &q_weight2_b,
                                                                      const torch::Tensor &weight_scale2_a,
                                                                      const torch::Tensor &weight_scale2_b,
                                                                      const torch::Tensor &activation_scale2_a,
                                                                      const torch::Tensor &activation_scale2_b,
                                                                      const torch::Tensor &bias2_a,
                                                                      const torch::Tensor &bias2_b,
                                                                      bool has_bias2_a,
                                                                      bool has_bias2_b) {
    TORCH_CHECK(input_a.is_cuda() && input_b.is_cuda(), "quant_ffn_w8a8_static_cutlass_grouped_pair: inputs must be CUDA");
    TORCH_CHECK(q_weight1_a.is_cuda() && q_weight1_b.is_cuda() && q_weight2_a.is_cuda() && q_weight2_b.is_cuda(),
                "quant_ffn_w8a8_static_cutlass_grouped_pair: weights must be CUDA");
    TORCH_CHECK(weight_scale1_a.is_cuda() && weight_scale1_b.is_cuda() && weight_scale2_a.is_cuda() && weight_scale2_b.is_cuda(),
                "quant_ffn_w8a8_static_cutlass_grouped_pair: weight scales must be CUDA");
    TORCH_CHECK(activation_scale1_a.is_cuda() && activation_scale1_b.is_cuda() && activation_scale2_a.is_cuda() && activation_scale2_b.is_cuda(),
                "quant_ffn_w8a8_static_cutlass_grouped_pair: activation scales must be CUDA");
    TORCH_CHECK(input_a.scalar_type() == torch::kFloat32 && input_b.scalar_type() == torch::kFloat32,
                "quant_ffn_w8a8_static_cutlass_grouped_pair: inputs must be float32");
    TORCH_CHECK(q_weight1_a.scalar_type() == torch::kInt8 && q_weight1_b.scalar_type() == torch::kInt8 &&
                    q_weight2_a.scalar_type() == torch::kInt8 && q_weight2_b.scalar_type() == torch::kInt8,
                "quant_ffn_w8a8_static_cutlass_grouped_pair: weights must be int8");
    TORCH_CHECK(input_a.dim() == 2 && input_b.dim() == 2, "quant_ffn_w8a8_static_cutlass_grouped_pair: inputs must be 2D");
    TORCH_CHECK(q_weight1_a.dim() == 2 && q_weight1_b.dim() == 2 && q_weight2_a.dim() == 2 && q_weight2_b.dim() == 2,
                "quant_ffn_w8a8_static_cutlass_grouped_pair: weights must be 2D");
    TORCH_CHECK(input_a.size(1) == q_weight1_a.size(1), "quant_ffn_w8a8_static_cutlass_grouped_pair: A first K mismatch");
    TORCH_CHECK(input_b.size(1) == q_weight1_b.size(1), "quant_ffn_w8a8_static_cutlass_grouped_pair: B first K mismatch");
    TORCH_CHECK(q_weight1_a.size(0) == q_weight2_a.size(1), "quant_ffn_w8a8_static_cutlass_grouped_pair: A hidden mismatch");
    TORCH_CHECK(q_weight1_b.size(0) == q_weight2_b.size(1), "quant_ffn_w8a8_static_cutlass_grouped_pair: B hidden mismatch");
    TORCH_CHECK(weight_scale1_a.size(0) == q_weight1_a.size(0) && weight_scale1_b.size(0) == q_weight1_b.size(0) &&
                    weight_scale2_a.size(0) == q_weight2_a.size(0) && weight_scale2_b.size(0) == q_weight2_b.size(0),
                "quant_ffn_w8a8_static_cutlass_grouped_pair: weight scale mismatch");
    TORCH_CHECK(input_a.size(1) % 32 == 0 && input_b.size(1) % 32 == 0 &&
                    q_weight2_a.size(1) % 32 == 0 && q_weight2_b.size(1) % 32 == 0,
                "quant_ffn_w8a8_static_cutlass_grouped_pair: K must be multiple of 32");
    if (has_bias1_a) {
        TORCH_CHECK(bias1_a.is_cuda() && bias1_a.scalar_type() == torch::kFloat32 && bias1_a.size(0) == q_weight1_a.size(0),
                    "quant_ffn_w8a8_static_cutlass_grouped_pair: invalid A first bias");
    }
    if (has_bias1_b) {
        TORCH_CHECK(bias1_b.is_cuda() && bias1_b.scalar_type() == torch::kFloat32 && bias1_b.size(0) == q_weight1_b.size(0),
                    "quant_ffn_w8a8_static_cutlass_grouped_pair: invalid B first bias");
    }
    if (has_bias2_a) {
        TORCH_CHECK(bias2_a.is_cuda() && bias2_a.scalar_type() == torch::kFloat32 && bias2_a.size(0) == q_weight2_a.size(0),
                    "quant_ffn_w8a8_static_cutlass_grouped_pair: invalid A second bias");
    }
    if (has_bias2_b) {
        TORCH_CHECK(bias2_b.is_cuda() && bias2_b.scalar_type() == torch::kFloat32 && bias2_b.size(0) == q_weight2_b.size(0),
                    "quant_ffn_w8a8_static_cutlass_grouped_pair: invalid B second bias");
    }

    const c10::cuda::CUDAGuard device_guard(input_a.device());
    auto input_a_contig = input_a.contiguous();
    auto input_b_contig = input_b.contiguous();
    auto q_weight1_a_contig = q_weight1_a.contiguous();
    auto q_weight1_b_contig = q_weight1_b.contiguous();
    auto q_weight2_a_contig = q_weight2_a.contiguous();
    auto q_weight2_b_contig = q_weight2_b.contiguous();
    auto weight_scale1_a_contig = weight_scale1_a.contiguous();
    auto weight_scale1_b_contig = weight_scale1_b.contiguous();
    auto weight_scale2_a_contig = weight_scale2_a.contiguous();
    auto weight_scale2_b_contig = weight_scale2_b.contiguous();
    auto activation_scale1_a_contig = activation_scale1_a.reshape({1}).contiguous();
    auto activation_scale1_b_contig = activation_scale1_b.reshape({1}).contiguous();
    auto activation_scale2_a_contig = activation_scale2_a.reshape({1}).contiguous();
    auto activation_scale2_b_contig = activation_scale2_b.reshape({1}).contiguous();
    auto bias1_a_contig = has_bias1_a ? bias1_a.contiguous() : bias1_a;
    auto bias1_b_contig = has_bias1_b ? bias1_b.contiguous() : bias1_b;
    auto bias2_a_contig = has_bias2_a ? bias2_a.contiguous() : bias2_a;
    auto bias2_b_contig = has_bias2_b ? bias2_b.contiguous() : bias2_b;

    auto q_input_a = torch::empty(input_a_contig.sizes(), input_a_contig.options().dtype(torch::kInt8));
    auto q_input_b = torch::empty(input_b_contig.sizes(), input_b_contig.options().dtype(torch::kInt8));
    auto hidden_a = torch::empty({input_a.size(0), q_weight1_a.size(0)}, input_a_contig.options());
    auto hidden_b = torch::empty({input_b.size(0), q_weight1_b.size(0)}, input_b_contig.options());
    auto q_hidden_a = torch::empty(hidden_a.sizes(), input_a_contig.options().dtype(torch::kInt8));
    auto q_hidden_b = torch::empty(hidden_b.sizes(), input_b_contig.options().dtype(torch::kInt8));
    auto acc1_a = torch::empty(hidden_a.sizes(), input_a_contig.options().dtype(torch::kInt32));
    auto acc1_b = torch::empty(hidden_b.sizes(), input_b_contig.options().dtype(torch::kInt32));
    auto acc2_a = torch::empty({input_a.size(0), q_weight2_a.size(0)}, input_a_contig.options().dtype(torch::kInt32));
    auto acc2_b = torch::empty({input_b.size(0), q_weight2_b.size(0)}, input_b_contig.options().dtype(torch::kInt32));
    auto output_a = torch::empty({input_a.size(0), q_weight2_a.size(0)}, input_a_contig.options());
    auto output_b = torch::empty({input_b.size(0), q_weight2_b.size(0)}, input_b_contig.options());

    auto int32_options = input_a.options().dtype(torch::kInt32);
    auto int64_options = input_a.options().dtype(torch::kInt64);
    auto problem_sizes1_device = torch::empty({2, 3}, int32_options);
    auto problem_sizes2_device = torch::empty({2, 3}, int32_options);
    auto ptr_a1_device = torch::empty({2}, int64_options);
    auto ptr_b1_device = torch::empty({2}, int64_options);
    auto ptr_c1_device = torch::empty({2}, int64_options);
    auto ptr_d1_device = torch::empty({2}, int64_options);
    auto lda1_device = torch::empty({2}, int64_options);
    auto ldb1_device = torch::empty({2}, int64_options);
    auto ldc1_device = torch::empty({2}, int64_options);
    auto ldd1_device = torch::empty({2}, int64_options);
    auto ptr_a2_device = torch::empty({2}, int64_options);
    auto ptr_b2_device = torch::empty({2}, int64_options);
    auto ptr_c2_device = torch::empty({2}, int64_options);
    auto ptr_d2_device = torch::empty({2}, int64_options);
    auto lda2_device = torch::empty({2}, int64_options);
    auto ldb2_device = torch::empty({2}, int64_options);
    auto ldc2_device = torch::empty({2}, int64_options);
    auto ldd2_device = torch::empty({2}, int64_options);

    int32_t problem_sizes1_host[6] = {
        static_cast<int32_t>(input_a.size(0)),
        static_cast<int32_t>(q_weight1_a.size(0)),
        static_cast<int32_t>(input_a.size(1)),
        static_cast<int32_t>(input_b.size(0)),
        static_cast<int32_t>(q_weight1_b.size(0)),
        static_cast<int32_t>(input_b.size(1)),
    };
    int32_t problem_sizes2_host[6] = {
        static_cast<int32_t>(input_a.size(0)),
        static_cast<int32_t>(q_weight2_a.size(0)),
        static_cast<int32_t>(q_weight2_a.size(1)),
        static_cast<int32_t>(input_b.size(0)),
        static_cast<int32_t>(q_weight2_b.size(0)),
        static_cast<int32_t>(q_weight2_b.size(1)),
    };
    auto stream = c10::cuda::getCurrentCUDAStream();

    launch_quantize_activation_s8_pair_meta_kernel(
        input_a_contig.data_ptr<float>(),
        input_b_contig.data_ptr<float>(),
        q_input_a.data_ptr<int8_t>(),
        q_input_b.data_ptr<int8_t>(),
        q_weight1_a_contig.data_ptr<int8_t>(),
        q_weight1_b_contig.data_ptr<int8_t>(),
        acc1_a.data_ptr<int32_t>(),
        acc1_b.data_ptr<int32_t>(),
        problem_sizes1_device.data_ptr<int32_t>(),
        ptr_a1_device.data_ptr<int64_t>(),
        ptr_b1_device.data_ptr<int64_t>(),
        ptr_c1_device.data_ptr<int64_t>(),
        ptr_d1_device.data_ptr<int64_t>(),
        lda1_device.data_ptr<int64_t>(),
        ldb1_device.data_ptr<int64_t>(),
        ldc1_device.data_ptr<int64_t>(),
        ldd1_device.data_ptr<int64_t>(),
        activation_scale1_a_contig.data_ptr<float>(),
        activation_scale1_b_contig.data_ptr<float>(),
        input_a.size(0),
        input_b.size(0),
        input_a.size(1),
        input_b.size(1),
        q_weight1_a.size(0),
        q_weight1_b.size(0),
        stream);
    auto status1 = launch_cutlass_s8s8s32_grouped_gemm(
        problem_sizes1_device.data_ptr<int32_t>(),
        problem_sizes1_host,
        reinterpret_cast<int8_t **>(ptr_a1_device.data_ptr<int64_t>()),
        reinterpret_cast<int8_t **>(ptr_b1_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_c1_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_d1_device.data_ptr<int64_t>()),
        lda1_device.data_ptr<int64_t>(),
        ldb1_device.data_ptr<int64_t>(),
        ldc1_device.data_ptr<int64_t>(),
        ldd1_device.data_ptr<int64_t>(),
        2,
        stream);
    TORCH_CHECK(static_cast<int>(status1) == 0, "quant_ffn_w8a8_static_cutlass_grouped_pair: first grouped CUTLASS GEMM failed");

    launch_dequantize_silu_quant_pair_meta_kernel(
        acc1_a.data_ptr<int32_t>(),
        acc1_b.data_ptr<int32_t>(),
        weight_scale1_a_contig.data_ptr<float>(),
        weight_scale1_b_contig.data_ptr<float>(),
        activation_scale1_a_contig.data_ptr<float>(),
        activation_scale1_b_contig.data_ptr<float>(),
        has_bias1_a ? bias1_a_contig.data_ptr<float>() : nullptr,
        has_bias1_b ? bias1_b_contig.data_ptr<float>() : nullptr,
        hidden_a.data_ptr<float>(),
        hidden_b.data_ptr<float>(),
        q_hidden_a.data_ptr<int8_t>(),
        q_hidden_b.data_ptr<int8_t>(),
        q_weight2_a_contig.data_ptr<int8_t>(),
        q_weight2_b_contig.data_ptr<int8_t>(),
        acc2_a.data_ptr<int32_t>(),
        acc2_b.data_ptr<int32_t>(),
        problem_sizes2_device.data_ptr<int32_t>(),
        ptr_a2_device.data_ptr<int64_t>(),
        ptr_b2_device.data_ptr<int64_t>(),
        ptr_c2_device.data_ptr<int64_t>(),
        ptr_d2_device.data_ptr<int64_t>(),
        lda2_device.data_ptr<int64_t>(),
        ldb2_device.data_ptr<int64_t>(),
        ldc2_device.data_ptr<int64_t>(),
        ldd2_device.data_ptr<int64_t>(),
        activation_scale2_a_contig.data_ptr<float>(),
        activation_scale2_b_contig.data_ptr<float>(),
        input_a.size(0),
        input_b.size(0),
        q_weight1_a.size(0),
        q_weight1_b.size(0),
        q_weight2_a.size(0),
        q_weight2_b.size(0),
        has_bias1_a,
        has_bias1_b,
        stream);
    auto status2 = launch_cutlass_s8s8s32_grouped_gemm(
        problem_sizes2_device.data_ptr<int32_t>(),
        problem_sizes2_host,
        reinterpret_cast<int8_t **>(ptr_a2_device.data_ptr<int64_t>()),
        reinterpret_cast<int8_t **>(ptr_b2_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_c2_device.data_ptr<int64_t>()),
        reinterpret_cast<int32_t **>(ptr_d2_device.data_ptr<int64_t>()),
        lda2_device.data_ptr<int64_t>(),
        ldb2_device.data_ptr<int64_t>(),
        ldc2_device.data_ptr<int64_t>(),
        ldd2_device.data_ptr<int64_t>(),
        2,
        stream);
    TORCH_CHECK(static_cast<int>(status2) == 0, "quant_ffn_w8a8_static_cutlass_grouped_pair: second grouped CUTLASS GEMM failed");

    launch_dequantize_s32_pair_kernel(
        acc2_a.data_ptr<int32_t>(),
        acc2_b.data_ptr<int32_t>(),
        weight_scale2_a_contig.data_ptr<float>(),
        weight_scale2_b_contig.data_ptr<float>(),
        activation_scale2_a_contig.data_ptr<float>(),
        activation_scale2_b_contig.data_ptr<float>(),
        has_bias2_a ? bias2_a_contig.data_ptr<float>() : nullptr,
        has_bias2_b ? bias2_b_contig.data_ptr<float>() : nullptr,
        output_a.data_ptr<float>(),
        output_b.data_ptr<float>(),
        input_a.size(0),
        input_b.size(0),
        q_weight2_a.size(0),
        q_weight2_b.size(0),
        has_bias2_a,
        has_bias2_b,
        stream);
    return {output_a, output_b, hidden_a, hidden_b};
}

torch::Tensor quant_linear_w8a8_static_wmma(const torch::Tensor &input,
                                            const torch::Tensor &q_weight,
                                            const torch::Tensor &weight_scale,
                                            const torch::Tensor &activation_scale,
                                            const torch::Tensor &bias,
                                            bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a8_static_wmma: input must be a CUDA tensor");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a8_static_wmma: q_weight must be a CUDA tensor");
    TORCH_CHECK(weight_scale.is_cuda(), "quant_linear_w8a8_static_wmma: weight_scale must be a CUDA tensor");
    TORCH_CHECK(activation_scale.is_cuda(), "quant_linear_w8a8_static_wmma: activation_scale must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_wmma: input must be float32");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a8_static_wmma: q_weight must be int8");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_wmma: weight_scale must be float32");
    TORCH_CHECK(activation_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_wmma: activation_scale must be float32");
    TORCH_CHECK(input.dim() == 2, "quant_linear_w8a8_static_wmma: input must be 2D [M, K]");
    TORCH_CHECK(q_weight.dim() == 2, "quant_linear_w8a8_static_wmma: q_weight must be 2D [N, K]");
    TORCH_CHECK(weight_scale.dim() == 1, "quant_linear_w8a8_static_wmma: weight_scale must be 1D [N]");
    TORCH_CHECK(activation_scale.numel() == 1, "quant_linear_w8a8_static_wmma: activation_scale must be scalar-like");
    TORCH_CHECK(input.size(1) == q_weight.size(1), "quant_linear_w8a8_static_wmma: input K must match weight K");
    TORCH_CHECK(weight_scale.size(0) == q_weight.size(0), "quant_linear_w8a8_static_wmma: weight_scale N must match weight N");
    TORCH_CHECK(input.size(1) % 16 == 0, "quant_linear_w8a8_static_wmma: K must be a multiple of 16");
    TORCH_CHECK(q_weight.size(0) % 16 == 0, "quant_linear_w8a8_static_wmma: N must be a multiple of 16");

    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a8_static_wmma: bias must be CUDA when has_bias=true");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_wmma: bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "quant_linear_w8a8_static_wmma: bias must be 1D [N]");
        TORCH_CHECK(bias.size(0) == q_weight.size(0), "quant_linear_w8a8_static_wmma: bias N must match weight N");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto q_weight_contig = q_weight.contiguous();
    auto weight_scale_contig = weight_scale.contiguous();
    auto activation_scale_contig = activation_scale.reshape({1}).contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : bias;
    auto output = torch::empty({input.size(0), q_weight.size(0)}, input.options());

    const float *bias_ptr = has_bias ? bias_contig.data_ptr<float>() : nullptr;
    launch_quant_linear_w8a8_wmma_kernel(input_contig.data_ptr<float>(),
                                         q_weight_contig.data_ptr<int8_t>(),
                                         weight_scale_contig.data_ptr<float>(),
                                         activation_scale_contig.data_ptr<float>(),
                                         bias_ptr,
                                         output.data_ptr<float>(),
                                         input.size(0),
                                         q_weight.size(0),
                                         input.size(1),
                                         has_bias,
                                         c10::cuda::getCurrentCUDAStream());
    return output;
}
