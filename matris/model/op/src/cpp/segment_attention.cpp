#include "../Opdefine.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

void launch_target_attention_sum_forward_kernel(const float *logits,
                                                const float *values,
                                                const int64_t *lengths,
                                                const int64_t *offsets,
                                                float *out,
                                                float *alpha,
                                                int64_t num_segments,
                                                cudaStream_t stream);

void launch_target_attention_sum_backward_kernel(const float *grad_out,
                                                 const float *values,
                                                 const float *out,
                                                 const float *alpha,
                                                 const int64_t *lengths,
                                                 const int64_t *offsets,
                                                 float *grad_logits,
                                                 float *grad_values,
                                                 int64_t num_segments,
                                                 cudaStream_t stream);

std::vector<torch::Tensor> target_attention_sum_forward(const torch::Tensor &logits,
                                                        const torch::Tensor &values,
                                                        const torch::Tensor &lengths) {
    TORCH_CHECK(logits.is_cuda() && values.is_cuda() && lengths.is_cuda(),
                "target_attention_sum_forward: tensors must be CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && values.scalar_type() == torch::kFloat32,
                "target_attention_sum_forward: logits and values must be float32");
    TORCH_CHECK(lengths.scalar_type() == torch::kInt64, "target_attention_sum_forward: lengths must be int64");
    TORCH_CHECK(logits.dim() == 2 && values.dim() == 2, "target_attention_sum_forward: logits/values must be 2D");
    TORCH_CHECK(logits.size(1) == 128 && values.size(1) == 128,
                "target_attention_sum_forward: feature dim must be 128");
    TORCH_CHECK(logits.sizes() == values.sizes(), "target_attention_sum_forward: logits/values shape mismatch");
    TORCH_CHECK(lengths.dim() == 1, "target_attention_sum_forward: lengths must be 1D");

    const c10::cuda::CUDAGuard device_guard(logits.device());
    auto logits_contig = logits.contiguous();
    auto values_contig = values.contiguous();
    auto lengths_contig = lengths.contiguous();
    auto offsets = torch::cumsum(lengths_contig, 0) - lengths_contig;
    auto out = torch::empty({lengths.size(0), 128}, logits_contig.options());
    auto alpha = torch::empty_like(logits_contig);

    launch_target_attention_sum_forward_kernel(
        logits_contig.data_ptr<float>(),
        values_contig.data_ptr<float>(),
        lengths_contig.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lengths.size(0),
        c10::cuda::getCurrentCUDAStream());
    return {out, alpha};
}

std::vector<torch::Tensor> target_attention_sum_backward(const torch::Tensor &grad_out,
                                                         const torch::Tensor &values,
                                                         const torch::Tensor &out,
                                                         const torch::Tensor &alpha,
                                                         const torch::Tensor &lengths) {
    TORCH_CHECK(grad_out.is_cuda() && values.is_cuda() && out.is_cuda() && alpha.is_cuda() && lengths.is_cuda(),
                "target_attention_sum_backward: tensors must be CUDA");
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 && values.scalar_type() == torch::kFloat32 &&
                    out.scalar_type() == torch::kFloat32 && alpha.scalar_type() == torch::kFloat32,
                "target_attention_sum_backward: float tensors must be float32");
    TORCH_CHECK(lengths.scalar_type() == torch::kInt64, "target_attention_sum_backward: lengths must be int64");
    TORCH_CHECK(values.dim() == 2 && alpha.dim() == 2 && grad_out.dim() == 2 && out.dim() == 2,
                "target_attention_sum_backward: tensors must be 2D");
    TORCH_CHECK(values.size(1) == 128 && alpha.size(1) == 128 && grad_out.size(1) == 128 && out.size(1) == 128,
                "target_attention_sum_backward: feature dim must be 128");
    TORCH_CHECK(values.sizes() == alpha.sizes(), "target_attention_sum_backward: values/alpha shape mismatch");
    TORCH_CHECK(grad_out.sizes() == out.sizes(), "target_attention_sum_backward: grad_out/out shape mismatch");
    TORCH_CHECK(lengths.size(0) == out.size(0), "target_attention_sum_backward: segment count mismatch");

    const c10::cuda::CUDAGuard device_guard(grad_out.device());
    auto grad_out_contig = grad_out.contiguous();
    auto values_contig = values.contiguous();
    auto out_contig = out.contiguous();
    auto alpha_contig = alpha.contiguous();
    auto lengths_contig = lengths.contiguous();
    auto offsets = torch::cumsum(lengths_contig, 0) - lengths_contig;
    auto grad_logits = torch::empty_like(values_contig);
    auto grad_values = torch::empty_like(values_contig);

    launch_target_attention_sum_backward_kernel(
        grad_out_contig.data_ptr<float>(),
        values_contig.data_ptr<float>(),
        out_contig.data_ptr<float>(),
        alpha_contig.data_ptr<float>(),
        lengths_contig.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        grad_logits.data_ptr<float>(),
        grad_values.data_ptr<float>(),
        lengths.size(0),
        c10::cuda::getCurrentCUDAStream());
    return {grad_logits, grad_values};
}
