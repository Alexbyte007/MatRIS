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

void launch_quant_linear_w8a_lowp_kernel(const void *input,
                                         const int8_t *q_weight,
                                         const float *scale,
                                         const float *bias,
                                         float *output,
                                         int64_t rows,
                                         int64_t out_features,
                                         int64_t in_features,
                                         bool has_bias,
                                         int dtype_code,
                                         cudaStream_t stream);

void launch_quant_linear_w8a_lowp_t_kernel(const void *input,
                                           const int8_t *q_weight_t,
                                           const float *scale,
                                           const float *bias,
                                           float *output,
                                           int64_t rows,
                                           int64_t out_features,
                                           int64_t in_features,
                                           bool has_bias,
                                           int dtype_code,
                                           cudaStream_t stream);

void launch_quant_linear_w8a_lowp_wmma_t_kernel(const void *input,
                                                const int8_t *q_weight_t,
                                                const float *scale,
                                                const float *bias,
                                                float *output,
                                                int64_t rows,
                                                int64_t out_features,
                                                int64_t in_features,
                                                bool has_bias,
                                                int dtype_code,
                                                cudaStream_t stream);

cutlass::Status launch_cutlass_w8a_lowp_mixed_gemm(const void *input,
                                                   const int8_t *q_weight,
                                                   const float *scale,
                                                   const float *bias,
                                                   float *output,
                                                   int64_t rows,
                                                   bool has_bias,
                                                   int dtype_code,
                                                   cudaStream_t stream);

void launch_quant_linear_w8a_lowp_grad_input_kernel(const float *grad_output,
                                                    const int8_t *q_weight,
                                                    const float *scale,
                                                    float *grad_input,
                                                    int64_t rows,
                                                    int64_t out_features,
                                                    int64_t in_features,
                                                    cudaStream_t stream);

void launch_quant_linear_w8a_lowp_dual_cached_wmma_kernel(const void *core_input,
                                                          const void *gate_input,
                                                          const void *core_weight,
                                                          const void *gate_weight,
                                                          const void *core_bias,
                                                          const void *gate_bias,
                                                          float *core_output,
                                                          float *gate_output,
                                                          int64_t rows,
                                                          bool core_has_bias,
                                                          bool gate_has_bias,
                                                          int dtype_code,
                                                          cudaStream_t stream);

void launch_quant_linear_w8a_lowp_dual_cached_tail_wmma_kernel(const void *core_input,
                                                               const void *gate_input,
                                                               const void *core_weight,
                                                               const void *gate_weight,
                                                               const void *core_bias,
                                                               const void *gate_bias,
                                                               const float *core_norm_weight,
                                                               const float *core_norm_bias,
                                                               const float *gate_norm_weight,
                                                               const float *gate_norm_bias,
                                                               float *output,
                                                               float *core_linear_aux,
                                                               float *gate_linear_aux,
                                                               float *core_norm_aux,
                                                               float *gate_norm_aux,
                                                               int64_t rows,
                                                               bool core_has_bias,
                                                               bool gate_has_bias,
                                                               float eps,
                                                               int dtype_code,
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

void launch_quant_linear_w8a8_grad_input_kernel(const float *grad_output,
                                                const int8_t *q_weight,
                                                const float *weight_scale,
                                                float *grad_input,
                                                int64_t rows,
                                                int64_t out_features,
                                                int64_t in_features,
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
                                                                        float *output,
                                                                        int64_t rows,
                                                                        int64_t core_in_features,
                                                                        int64_t gate_in_features,
                                                                        bool core_has_bias,
                                                                        bool gate_has_bias,
                                                                        float eps,
                                                                        cudaStream_t stream);

void launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_aux_kernel(const float *core_input,
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

torch::Tensor quant_linear_w8a_lowp_forward(const torch::Tensor &input,
                                            const torch::Tensor &q_weight,
                                            const torch::Tensor &scale,
                                            const torch::Tensor &bias,
                                            bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a_lowp_forward: input must be a CUDA tensor");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a_lowp_forward: q_weight must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "quant_linear_w8a_lowp_forward: scale must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_forward: input must be float16 or bfloat16");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a_lowp_forward: q_weight must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_forward: scale must be float32");
    TORCH_CHECK(input.dim() == 2, "quant_linear_w8a_lowp_forward: input must be 2D [M, K]");
    TORCH_CHECK(q_weight.dim() == 2, "quant_linear_w8a_lowp_forward: q_weight must be 2D [N, K]");
    TORCH_CHECK(scale.dim() == 1, "quant_linear_w8a_lowp_forward: scale must be 1D [N]");
    TORCH_CHECK(input.size(1) == q_weight.size(1), "quant_linear_w8a_lowp_forward: input K must match weight K");
    TORCH_CHECK(scale.size(0) == q_weight.size(0), "quant_linear_w8a_lowp_forward: scale N must match weight N");
    TORCH_CHECK(input.size(1) == 128 && q_weight.size(0) == 128,
                "quant_linear_w8a_lowp_forward: v0 only supports K=128 and N=128");

    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a_lowp_forward: bias must be a CUDA tensor when has_bias=true");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "quant_linear_w8a_lowp_forward: bias must be 1D [N]");
        TORCH_CHECK(bias.size(0) == q_weight.size(0), "quant_linear_w8a_lowp_forward: bias N must match weight N");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto q_weight_contig = q_weight.contiguous();
    auto scale_contig = scale.contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : bias;
    auto output = torch::empty({input.size(0), q_weight.size(0)}, input.options().dtype(torch::kFloat32));

    const int dtype_code = input_contig.scalar_type() == torch::kFloat16 ? 0 : 1;
    const float *bias_ptr = has_bias ? bias_contig.data_ptr<float>() : nullptr;
    launch_quant_linear_w8a_lowp_kernel(input_contig.data_ptr(),
                                        q_weight_contig.data_ptr<int8_t>(),
                                        scale_contig.data_ptr<float>(),
                                        bias_ptr,
                                        output.data_ptr<float>(),
                                        input.size(0),
                                        q_weight.size(0),
                                        input.size(1),
                                        has_bias,
                                        dtype_code,
                                        c10::cuda::getCurrentCUDAStream());
    return output;
}

torch::Tensor quant_linear_w8a_lowp_t_forward(const torch::Tensor &input,
                                              const torch::Tensor &q_weight_t,
                                              const torch::Tensor &scale,
                                              const torch::Tensor &bias,
                                              bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a_lowp_t_forward: input must be a CUDA tensor");
    TORCH_CHECK(q_weight_t.is_cuda(), "quant_linear_w8a_lowp_t_forward: q_weight_t must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "quant_linear_w8a_lowp_t_forward: scale must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_t_forward: input must be float16 or bfloat16");
    TORCH_CHECK(q_weight_t.scalar_type() == torch::kInt8, "quant_linear_w8a_lowp_t_forward: q_weight_t must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_t_forward: scale must be float32");
    TORCH_CHECK(input.dim() == 2, "quant_linear_w8a_lowp_t_forward: input must be 2D [M, K]");
    TORCH_CHECK(q_weight_t.dim() == 2, "quant_linear_w8a_lowp_t_forward: q_weight_t must be 2D [K, N]");
    TORCH_CHECK(scale.dim() == 1, "quant_linear_w8a_lowp_t_forward: scale must be 1D [N]");
    TORCH_CHECK(input.size(1) == q_weight_t.size(0), "quant_linear_w8a_lowp_t_forward: input K must match weight K");
    TORCH_CHECK(scale.size(0) == q_weight_t.size(1), "quant_linear_w8a_lowp_t_forward: scale N must match weight N");
    TORCH_CHECK(input.size(1) == 128 && q_weight_t.size(1) == 128,
                "quant_linear_w8a_lowp_t_forward: v1 only supports K=128 and N=128");

    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a_lowp_t_forward: bias must be a CUDA tensor when has_bias=true");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_t_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "quant_linear_w8a_lowp_t_forward: bias must be 1D [N]");
        TORCH_CHECK(bias.size(0) == q_weight_t.size(1), "quant_linear_w8a_lowp_t_forward: bias N must match weight N");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto q_weight_t_contig = q_weight_t.contiguous();
    auto scale_contig = scale.contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : bias;
    auto output = torch::empty({input.size(0), q_weight_t.size(1)}, input.options().dtype(torch::kFloat32));

    const int dtype_code = input_contig.scalar_type() == torch::kFloat16 ? 0 : 1;
    const float *bias_ptr = has_bias ? bias_contig.data_ptr<float>() : nullptr;
    launch_quant_linear_w8a_lowp_t_kernel(input_contig.data_ptr(),
                                          q_weight_t_contig.data_ptr<int8_t>(),
                                          scale_contig.data_ptr<float>(),
                                          bias_ptr,
                                          output.data_ptr<float>(),
                                          input.size(0),
                                          q_weight_t.size(1),
                                          input.size(1),
                                          has_bias,
                                          dtype_code,
                                          c10::cuda::getCurrentCUDAStream());
    return output;
}

torch::Tensor quant_linear_w8a_lowp_wmma_t_forward(const torch::Tensor &input,
                                                   const torch::Tensor &q_weight_t,
                                                   const torch::Tensor &scale,
                                                   const torch::Tensor &bias,
                                                   bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a_lowp_wmma_t_forward: input must be CUDA");
    TORCH_CHECK(q_weight_t.is_cuda(), "quant_linear_w8a_lowp_wmma_t_forward: q_weight_t must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "quant_linear_w8a_lowp_wmma_t_forward: scale must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_wmma_t_forward: input must be float16 or bfloat16");
    TORCH_CHECK(q_weight_t.scalar_type() == torch::kInt8, "quant_linear_w8a_lowp_wmma_t_forward: q_weight_t must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_wmma_t_forward: scale must be float32");
    TORCH_CHECK(input.dim() == 2 && q_weight_t.dim() == 2 && scale.dim() == 1,
                "quant_linear_w8a_lowp_wmma_t_forward: input [M,K], q_weight_t [K,N], scale [N]");
    TORCH_CHECK(input.size(1) == q_weight_t.size(0), "quant_linear_w8a_lowp_wmma_t_forward: K mismatch");
    TORCH_CHECK(scale.size(0) == q_weight_t.size(1), "quant_linear_w8a_lowp_wmma_t_forward: N mismatch");
    TORCH_CHECK(input.size(1) == 128 && q_weight_t.size(1) == 128,
                "quant_linear_w8a_lowp_wmma_t_forward: v0 only supports K=N=128");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a_lowp_wmma_t_forward: bias must be CUDA when present");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_wmma_t_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == q_weight_t.size(1),
                    "quant_linear_w8a_lowp_wmma_t_forward: bias N mismatch");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto q_weight_t_contig = q_weight_t.contiguous();
    auto scale_contig = scale.contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : bias;
    auto output = torch::empty({input.size(0), q_weight_t.size(1)}, input.options().dtype(torch::kFloat32));
    const int dtype_code = input_contig.scalar_type() == torch::kFloat16 ? 0 : 1;
    launch_quant_linear_w8a_lowp_wmma_t_kernel(input_contig.data_ptr(),
                                               q_weight_t_contig.data_ptr<int8_t>(),
                                               scale_contig.data_ptr<float>(),
                                               has_bias ? bias_contig.data_ptr<float>() : nullptr,
                                               output.data_ptr<float>(),
                                               input.size(0),
                                               q_weight_t.size(1),
                                               input.size(1),
                                               has_bias,
                                               dtype_code,
                                               c10::cuda::getCurrentCUDAStream());
    return output;
}

torch::Tensor quant_linear_w8a_lowp_cutlass_mixed_forward(const torch::Tensor &input,
                                                          const torch::Tensor &q_weight,
                                                          const torch::Tensor &scale,
                                                          const torch::Tensor &bias,
                                                          bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_forward: input must be CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_forward: q_weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_forward: scale must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_cutlass_mixed_forward: input must be float16 or bfloat16");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a_lowp_cutlass_mixed_forward: q_weight must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_cutlass_mixed_forward: scale must be float32");
    TORCH_CHECK(input.dim() == 2 && q_weight.dim() == 2 && scale.dim() == 1,
                "quant_linear_w8a_lowp_cutlass_mixed_forward: input [M,K], q_weight [N,K], scale [N]");
    TORCH_CHECK(input.size(1) == q_weight.size(1), "quant_linear_w8a_lowp_cutlass_mixed_forward: K mismatch");
    TORCH_CHECK(scale.size(0) == q_weight.size(0), "quant_linear_w8a_lowp_cutlass_mixed_forward: N mismatch");
    TORCH_CHECK(input.size(1) == 128 && q_weight.size(0) == 128,
                "quant_linear_w8a_lowp_cutlass_mixed_forward: v0 only supports K=N=128");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_forward: bias must be CUDA when present");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_cutlass_mixed_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == q_weight.size(0),
                    "quant_linear_w8a_lowp_cutlass_mixed_forward: bias N mismatch");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto q_weight_contig = q_weight.contiguous();
    auto scale_contig = scale.contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : bias;
    auto output = torch::empty({input.size(0), q_weight.size(0)}, input.options().dtype(torch::kFloat32));
    const int dtype_code = input_contig.scalar_type() == torch::kFloat16 ? 0 : 1;
    const float *bias_ptr = has_bias ? bias_contig.data_ptr<float>() : nullptr;
    auto status = launch_cutlass_w8a_lowp_mixed_gemm(input_contig.data_ptr(),
                                                     q_weight_contig.data_ptr<int8_t>(),
                                                     scale_contig.data_ptr<float>(),
                                                     bias_ptr,
                                                     output.data_ptr<float>(),
                                                     input.size(0),
                                                     has_bias,
                                                     dtype_code,
                                                     c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(static_cast<int>(status) == 0, "quant_linear_w8a_lowp_cutlass_mixed_forward: CUTLASS mixed GEMM failed");
    return output;
}

torch::Tensor quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward(const torch::Tensor &input,
                                                                   const torch::Tensor &q_weight,
                                                                   const torch::Tensor &scale,
                                                                   const torch::Tensor &bias,
                                                                   bool has_bias) {
    TORCH_CHECK(input.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: input must be CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: q_weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: scale must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: input must be contiguous");
    TORCH_CHECK(q_weight.is_contiguous(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: q_weight must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: scale must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: input must be float16 or bfloat16");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: q_weight must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: scale must be float32");
    TORCH_CHECK(input.dim() == 2 && q_weight.dim() == 2 && scale.dim() == 1,
                "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: input [M,K], q_weight [N,K], scale [N]");
    TORCH_CHECK(input.size(1) == q_weight.size(1), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: K mismatch");
    TORCH_CHECK(scale.size(0) == q_weight.size(0), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: N mismatch");
    TORCH_CHECK(input.size(1) == 128 && q_weight.size(0) == 128,
                "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: v1 only supports K=N=128");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: bias must be CUDA when present");
        TORCH_CHECK(bias.is_contiguous(), "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: bias must be contiguous");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == q_weight.size(0),
                    "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: bias N mismatch");
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto output = torch::empty({input.size(0), q_weight.size(0)}, input.options().dtype(torch::kFloat32));
    const int dtype_code = input.scalar_type() == torch::kFloat16 ? 0 : 1;
    const float *bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;
    auto status = launch_cutlass_w8a_lowp_mixed_gemm(input.data_ptr(),
                                                     q_weight.data_ptr<int8_t>(),
                                                     scale.data_ptr<float>(),
                                                     bias_ptr,
                                                     output.data_ptr<float>(),
                                                     input.size(0),
                                                     has_bias,
                                                     dtype_code,
                                                     c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(static_cast<int>(status) == 0, "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward: CUTLASS mixed GEMM failed");
    return output;
}

torch::Tensor quant_linear_w8a_lowp_grad_input(const torch::Tensor &grad_output,
                                               const torch::Tensor &q_weight,
                                               const torch::Tensor &scale) {
    TORCH_CHECK(grad_output.is_cuda(), "quant_linear_w8a_lowp_grad_input: grad_output must be CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a_lowp_grad_input: q_weight must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "quant_linear_w8a_lowp_grad_input: scale must be CUDA");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_grad_input: grad_output must be float32");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a_lowp_grad_input: q_weight must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "quant_linear_w8a_lowp_grad_input: scale must be float32");
    TORCH_CHECK(grad_output.dim() == 2 && q_weight.dim() == 2 && scale.dim() == 1,
                "quant_linear_w8a_lowp_grad_input: grad_output [M,N], q_weight [N,K], scale [N]");
    TORCH_CHECK(grad_output.size(1) == q_weight.size(0), "quant_linear_w8a_lowp_grad_input: N mismatch");
    TORCH_CHECK(scale.size(0) == q_weight.size(0), "quant_linear_w8a_lowp_grad_input: scale N mismatch");
    TORCH_CHECK(q_weight.size(0) == 128 && q_weight.size(1) == 128,
                "quant_linear_w8a_lowp_grad_input: v0 only supports N=K=128");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());
    auto grad_output_contig = grad_output.contiguous();
    auto q_weight_contig = q_weight.contiguous();
    auto scale_contig = scale.contiguous();
    auto grad_input = torch::empty({grad_output.size(0), q_weight.size(1)}, grad_output.options());
    launch_quant_linear_w8a_lowp_grad_input_kernel(
        grad_output_contig.data_ptr<float>(),
        q_weight_contig.data_ptr<int8_t>(),
        scale_contig.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_output.size(0),
        q_weight.size(0),
        q_weight.size(1),
        c10::cuda::getCurrentCUDAStream());
    return grad_input;
}

std::vector<torch::Tensor> quant_linear_w8a_lowp_dual_cached_forward(const torch::Tensor &core_input,
                                                                     const torch::Tensor &gate_input,
                                                                     const torch::Tensor &core_weight,
                                                                     const torch::Tensor &gate_weight,
                                                                     const torch::Tensor &core_bias,
                                                                     const torch::Tensor &gate_bias,
                                                                     bool core_has_bias,
                                                                     bool gate_has_bias) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_forward: inputs must be CUDA");
    TORCH_CHECK(core_weight.is_cuda() && gate_weight.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_forward: weights must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat16 || core_input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_dual_cached_forward: core input must be float16 or bfloat16");
    TORCH_CHECK(core_input.scalar_type() == gate_input.scalar_type() &&
                    core_input.scalar_type() == core_weight.scalar_type() &&
                    core_input.scalar_type() == gate_weight.scalar_type(),
                "quant_linear_w8a_lowp_dual_cached_forward: inputs and weights must share dtype");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2 &&
                    core_weight.dim() == 2 && gate_weight.dim() == 2,
                "quant_linear_w8a_lowp_dual_cached_forward: tensors must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0),
                "quant_linear_w8a_lowp_dual_cached_forward: row count must match");
    TORCH_CHECK(core_input.size(1) == 128 && gate_input.size(1) == 128 &&
                    core_weight.size(0) == 128 && core_weight.size(1) == 128 &&
                    gate_weight.size(0) == 128 && gate_weight.size(1) == 128,
                "quant_linear_w8a_lowp_dual_cached_forward: v0 only supports K=N=128");
    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == core_input.scalar_type() &&
                        core_bias.dim() == 1 && core_bias.size(0) == 128,
                    "quant_linear_w8a_lowp_dual_cached_forward: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == core_input.scalar_type() &&
                        gate_bias.dim() == 1 && gate_bias.size(0) == 128,
                    "quant_linear_w8a_lowp_dual_cached_forward: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_weight_contig = core_weight.contiguous();
    auto gate_weight_contig = gate_weight.contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_output = torch::empty({core_input.size(0), 128}, core_input.options().dtype(torch::kFloat32));
    auto gate_output = torch::empty({gate_input.size(0), 128}, gate_input.options().dtype(torch::kFloat32));
    const int dtype_code = core_input_contig.scalar_type() == torch::kFloat16 ? 0 : 1;
    launch_quant_linear_w8a_lowp_dual_cached_wmma_kernel(
        core_input_contig.data_ptr(),
        gate_input_contig.data_ptr(),
        core_weight_contig.data_ptr(),
        gate_weight_contig.data_ptr(),
        core_has_bias ? core_bias_contig.data_ptr() : nullptr,
        gate_has_bias ? gate_bias_contig.data_ptr() : nullptr,
        core_output.data_ptr<float>(),
        gate_output.data_ptr<float>(),
        core_input.size(0),
        core_has_bias,
        gate_has_bias,
        dtype_code,
        c10::cuda::getCurrentCUDAStream());
    return {core_output, gate_output};
}

torch::Tensor quant_linear_w8a_lowp_dual_cached_tail_forward(const torch::Tensor &core_input,
                                                             const torch::Tensor &gate_input,
                                                             const torch::Tensor &core_weight,
                                                             const torch::Tensor &gate_weight,
                                                             const torch::Tensor &core_bias,
                                                             const torch::Tensor &gate_bias,
                                                             bool core_has_bias,
                                                             bool gate_has_bias,
                                                             const torch::Tensor &core_norm_weight,
                                                             const torch::Tensor &core_norm_bias,
                                                             const torch::Tensor &gate_norm_weight,
                                                             const torch::Tensor &gate_norm_bias,
                                                             double eps) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward: inputs must be CUDA");
    TORCH_CHECK(core_weight.is_cuda() && gate_weight.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward: weights must be CUDA");
    TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                    gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward: norm params must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat16 || core_input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_dual_cached_tail_forward: core input must be float16 or bfloat16");
    TORCH_CHECK(core_input.scalar_type() == gate_input.scalar_type() &&
                    core_input.scalar_type() == core_weight.scalar_type() &&
                    core_input.scalar_type() == gate_weight.scalar_type(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward: inputs and weights must share dtype");
    TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 &&
                    core_norm_bias.scalar_type() == torch::kFloat32 &&
                    gate_norm_weight.scalar_type() == torch::kFloat32 &&
                    gate_norm_bias.scalar_type() == torch::kFloat32,
                "quant_linear_w8a_lowp_dual_cached_tail_forward: norm params must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2 &&
                    core_weight.dim() == 2 && gate_weight.dim() == 2,
                "quant_linear_w8a_lowp_dual_cached_tail_forward: tensors must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0),
                "quant_linear_w8a_lowp_dual_cached_tail_forward: row count must match");
    TORCH_CHECK(core_input.size(1) == 128 && gate_input.size(1) == 128 &&
                    core_weight.size(0) == 128 && core_weight.size(1) == 128 &&
                    gate_weight.size(0) == 128 && gate_weight.size(1) == 128,
                "quant_linear_w8a_lowp_dual_cached_tail_forward: v0 only supports K=N=128");
    TORCH_CHECK(core_norm_weight.numel() == 128 && core_norm_bias.numel() == 128 &&
                    gate_norm_weight.numel() == 128 && gate_norm_bias.numel() == 128,
                "quant_linear_w8a_lowp_dual_cached_tail_forward: norm params must be size 128");
    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == core_input.scalar_type() &&
                        core_bias.dim() == 1 && core_bias.size(0) == 128,
                    "quant_linear_w8a_lowp_dual_cached_tail_forward: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == core_input.scalar_type() &&
                        gate_bias.dim() == 1 && gate_bias.size(0) == 128,
                    "quant_linear_w8a_lowp_dual_cached_tail_forward: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_weight_contig = core_weight.contiguous();
    auto gate_weight_contig = gate_weight.contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_norm_weight_contig = core_norm_weight.contiguous();
    auto core_norm_bias_contig = core_norm_bias.contiguous();
    auto gate_norm_weight_contig = gate_norm_weight.contiguous();
    auto gate_norm_bias_contig = gate_norm_bias.contiguous();
    auto output = torch::empty({core_input.size(0), 128}, core_input.options().dtype(torch::kFloat32));
    const int dtype_code = core_input_contig.scalar_type() == torch::kFloat16 ? 0 : 1;
    launch_quant_linear_w8a_lowp_dual_cached_tail_wmma_kernel(
        core_input_contig.data_ptr(),
        gate_input_contig.data_ptr(),
        core_weight_contig.data_ptr(),
        gate_weight_contig.data_ptr(),
        core_has_bias ? core_bias_contig.data_ptr() : nullptr,
        gate_has_bias ? gate_bias_contig.data_ptr() : nullptr,
        core_norm_weight_contig.data_ptr<float>(),
        core_norm_bias_contig.data_ptr<float>(),
        gate_norm_weight_contig.data_ptr<float>(),
        gate_norm_bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        core_input.size(0),
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        dtype_code,
        c10::cuda::getCurrentCUDAStream());
    return output;
}

std::vector<torch::Tensor> quant_linear_w8a_lowp_dual_cached_tail_forward_aux(const torch::Tensor &core_input,
                                                                              const torch::Tensor &gate_input,
                                                                              const torch::Tensor &core_weight,
                                                                              const torch::Tensor &gate_weight,
                                                                              const torch::Tensor &core_bias,
                                                                              const torch::Tensor &gate_bias,
                                                                              bool core_has_bias,
                                                                              bool gate_has_bias,
                                                                              const torch::Tensor &core_norm_weight,
                                                                              const torch::Tensor &core_norm_bias,
                                                                              const torch::Tensor &gate_norm_weight,
                                                                              const torch::Tensor &gate_norm_bias,
                                                                              double eps) {
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: inputs must be CUDA");
    TORCH_CHECK(core_weight.is_cuda() && gate_weight.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: weights must be CUDA");
    TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                    gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: norm params must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat16 || core_input.scalar_type() == torch::kBFloat16,
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: core input must be float16 or bfloat16");
    TORCH_CHECK(core_input.scalar_type() == gate_input.scalar_type() &&
                    core_input.scalar_type() == core_weight.scalar_type() &&
                    core_input.scalar_type() == gate_weight.scalar_type(),
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: inputs and weights must share dtype");
    TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 &&
                    core_norm_bias.scalar_type() == torch::kFloat32 &&
                    gate_norm_weight.scalar_type() == torch::kFloat32 &&
                    gate_norm_bias.scalar_type() == torch::kFloat32,
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: norm params must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2 &&
                    core_weight.dim() == 2 && gate_weight.dim() == 2,
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: tensors must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0),
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: row count must match");
    TORCH_CHECK(core_input.size(1) == 128 && gate_input.size(1) == 128 &&
                    core_weight.size(0) == 128 && core_weight.size(1) == 128 &&
                    gate_weight.size(0) == 128 && gate_weight.size(1) == 128,
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: v0 only supports K=N=128");
    TORCH_CHECK(core_norm_weight.numel() == 128 && core_norm_bias.numel() == 128 &&
                    gate_norm_weight.numel() == 128 && gate_norm_bias.numel() == 128,
                "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: norm params must be size 128");
    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == core_input.scalar_type() &&
                        core_bias.dim() == 1 && core_bias.size(0) == 128,
                    "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == core_input.scalar_type() &&
                        gate_bias.dim() == 1 && gate_bias.size(0) == 128,
                    "quant_linear_w8a_lowp_dual_cached_tail_forward_aux: invalid gate bias");
    }

    const c10::cuda::CUDAGuard device_guard(core_input.device());
    auto core_input_contig = core_input.contiguous();
    auto gate_input_contig = gate_input.contiguous();
    auto core_weight_contig = core_weight.contiguous();
    auto gate_weight_contig = gate_weight.contiguous();
    auto core_bias_contig = core_has_bias ? core_bias.contiguous() : core_bias;
    auto gate_bias_contig = gate_has_bias ? gate_bias.contiguous() : gate_bias;
    auto core_norm_weight_contig = core_norm_weight.contiguous();
    auto core_norm_bias_contig = core_norm_bias.contiguous();
    auto gate_norm_weight_contig = gate_norm_weight.contiguous();
    auto gate_norm_bias_contig = gate_norm_bias.contiguous();
    auto output = torch::empty({core_input.size(0), 128}, core_input.options().dtype(torch::kFloat32));
    auto core_linear = torch::empty({core_input.size(0), 128}, core_input.options().dtype(torch::kFloat32));
    auto gate_linear = torch::empty({core_input.size(0), 128}, core_input.options().dtype(torch::kFloat32));
    auto core_norm = torch::empty({core_input.size(0), 128}, core_input.options().dtype(torch::kFloat32));
    auto gate_norm = torch::empty({core_input.size(0), 128}, core_input.options().dtype(torch::kFloat32));
    const int dtype_code = core_input_contig.scalar_type() == torch::kFloat16 ? 0 : 1;
    launch_quant_linear_w8a_lowp_dual_cached_tail_wmma_kernel(
        core_input_contig.data_ptr(),
        gate_input_contig.data_ptr(),
        core_weight_contig.data_ptr(),
        gate_weight_contig.data_ptr(),
        core_has_bias ? core_bias_contig.data_ptr() : nullptr,
        gate_has_bias ? gate_bias_contig.data_ptr() : nullptr,
        core_norm_weight_contig.data_ptr<float>(),
        core_norm_bias_contig.data_ptr<float>(),
        gate_norm_weight_contig.data_ptr<float>(),
        gate_norm_bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        core_linear.data_ptr<float>(),
        gate_linear.data_ptr<float>(),
        core_norm.data_ptr<float>(),
        gate_norm.data_ptr<float>(),
        core_input.size(0),
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        dtype_code,
        c10::cuda::getCurrentCUDAStream());
    return {output, core_linear, gate_linear, core_norm, gate_norm};
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

    const char *backend = std::getenv("MATRIS_W8A8_BACKEND");
    const bool use_parallel_backend = backend != nullptr && std::strcmp(backend, "cuda_wmma_tail_n128_parallel") == 0;
    const bool use_auto_backend = backend != nullptr && std::strcmp(backend, "cuda_wmma_tail_n128_auto") == 0;
    // On H100, the parallel branch is consistently faster for the small/medium
    // line-graph rows seen by p8e, but slower once rows get large enough. Keep
    // the switch conservative and default-off via MATRIS_W8A8_BACKEND.
    const bool use_parallel = use_parallel_backend || (use_auto_backend && core_input.size(0) < 4096);
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

std::vector<torch::Tensor> quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux(const torch::Tensor &core_input,
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
    TORCH_CHECK(core_input.is_cuda() && gate_input.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: inputs must be CUDA");
    TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: weights must be CUDA");
    TORCH_CHECK(core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: scales must be CUDA");
    TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: activation scales must be CUDA");
    TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() && gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: norm params must be CUDA");
    TORCH_CHECK(core_input.scalar_type() == torch::kFloat32 && gate_input.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: inputs must be float32");
    TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: weights must be int8");
    TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 && gate_weight_scale.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: scales must be float32");
    TORCH_CHECK(core_norm_weight.scalar_type() == torch::kFloat32 && core_norm_bias.scalar_type() == torch::kFloat32 &&
                    gate_norm_weight.scalar_type() == torch::kFloat32 && gate_norm_bias.scalar_type() == torch::kFloat32,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: norm params must be float32");
    TORCH_CHECK(core_input.dim() == 2 && gate_input.dim() == 2, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: inputs must be 2D");
    TORCH_CHECK(core_q_weight.dim() == 2 && gate_q_weight.dim() == 2, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: weights must be 2D");
    TORCH_CHECK(core_input.size(0) == gate_input.size(0), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: row count must match");
    TORCH_CHECK(core_input.size(1) == core_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: core K mismatch");
    TORCH_CHECK(gate_input.size(1) == gate_q_weight.size(1), "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: gate K mismatch");
    TORCH_CHECK(core_q_weight.size(0) == 128 && gate_q_weight.size(0) == 128,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: only N=128 is supported");
    TORCH_CHECK(core_input.size(1) % 16 == 0 && gate_input.size(1) % 16 == 0,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: K must be multiple of 16");
    TORCH_CHECK(core_weight_scale.numel() == 128 && gate_weight_scale.numel() == 128 &&
                    core_norm_weight.numel() == 128 && core_norm_bias.numel() == 128 &&
                    gate_norm_weight.numel() == 128 && gate_norm_bias.numel() == 128,
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: scale/norm size mismatch");
    if (core_has_bias) {
        TORCH_CHECK(core_bias.is_cuda() && core_bias.scalar_type() == torch::kFloat32 && core_bias.numel() == 128,
                    "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: invalid core bias");
    }
    if (gate_has_bias) {
        TORCH_CHECK(gate_bias.is_cuda() && gate_bias.scalar_type() == torch::kFloat32 && gate_bias.numel() == 128,
                    "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: invalid gate bias");
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
    auto core_linear = torch::empty_like(output);
    auto gate_linear = torch::empty_like(output);
    auto core_norm = torch::empty_like(output);
    auto gate_norm = torch::empty_like(output);

    const char *backend = std::getenv("MATRIS_W8A8_BACKEND");
    TORCH_CHECK(!(backend != nullptr && std::strcmp(backend, "cuda_wmma_tail_n128_parallel") == 0),
                "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux: parallel backend is not supported");
    auto stream = c10::cuda::getCurrentCUDAStream();
    launch_quant_linear_w8a8_wmma_dual_gated_tail_n128_aux_kernel(
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
        output.data_ptr<float>(),
        core_linear.data_ptr<float>(),
        gate_linear.data_ptr<float>(),
        core_norm.data_ptr<float>(),
        gate_norm.data_ptr<float>(),
        core_input.size(0),
        core_input.size(1),
        gate_input.size(1),
        core_has_bias,
        gate_has_bias,
        static_cast<float>(eps),
        stream);
    return {output, core_linear, gate_linear, core_norm, gate_norm};
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

torch::Tensor quant_linear_w8a8_static_grad_input(const torch::Tensor &grad_output,
                                                  const torch::Tensor &q_weight,
                                                  const torch::Tensor &weight_scale) {
    TORCH_CHECK(grad_output.is_cuda(), "quant_linear_w8a8_static_grad_input: grad_output must be CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "quant_linear_w8a8_static_grad_input: q_weight must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "quant_linear_w8a8_static_grad_input: weight_scale must be CUDA");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_grad_input: grad_output must be float32");
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt8, "quant_linear_w8a8_static_grad_input: q_weight must be int8");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, "quant_linear_w8a8_static_grad_input: weight_scale must be float32");
    TORCH_CHECK(grad_output.dim() == 2, "quant_linear_w8a8_static_grad_input: grad_output must be 2D [M, N]");
    TORCH_CHECK(q_weight.dim() == 2, "quant_linear_w8a8_static_grad_input: q_weight must be 2D [N, K]");
    TORCH_CHECK(weight_scale.dim() == 1, "quant_linear_w8a8_static_grad_input: weight_scale must be 1D [N]");
    TORCH_CHECK(grad_output.size(1) == q_weight.size(0), "quant_linear_w8a8_static_grad_input: grad N mismatch");
    TORCH_CHECK(weight_scale.size(0) == q_weight.size(0), "quant_linear_w8a8_static_grad_input: scale N mismatch");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());
    auto grad_output_contig = grad_output.contiguous();
    auto q_weight_contig = q_weight.contiguous();
    auto weight_scale_contig = weight_scale.contiguous();
    auto grad_input = torch::empty({grad_output.size(0), q_weight.size(1)}, grad_output.options());

    launch_quant_linear_w8a8_grad_input_kernel(
        grad_output_contig.data_ptr<float>(),
        q_weight_contig.data_ptr<int8_t>(),
        weight_scale_contig.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_output.size(0),
        q_weight.size(0),
        q_weight.size(1),
        c10::cuda::getCurrentCUDAStream());
    return grad_input;
}
