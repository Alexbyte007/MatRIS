#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kDim = 128;
constexpr int kThreads = 256;

__device__ __forceinline__ float sigmoidf_stable(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float clamp_round_int8_as_float(float x) {
  float q = nearbyintf(x);
  q = fminf(127.0f, fmaxf(-127.0f, q));
  return q;
}

template <bool CoreHasBias, bool GateHasBias>
__global__ void w8a8_dual_gated_tail_input_grad_backward_n128_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ core_input,
    const float* __restrict__ gate_input,
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
    float* __restrict__ grad_core_input,
    float* __restrict__ grad_gate_input,
    int64_t rows,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float s_core_out[kDim];
  __shared__ float s_gate_out[kDim];
  __shared__ float s_core_grad_proj[kDim];
  __shared__ float s_gate_grad_proj[kDim];
  __shared__ float s_reduce_core[kThreads];
  __shared__ float s_reduce_gate[kThreads];

  const int64_t base = static_cast<int64_t>(row) * kDim;
  const float core_act_scale = core_activation_scale[0];
  const float gate_act_scale = gate_activation_scale[0];

  if (tid < kDim) {
    float core_acc = CoreHasBias ? core_bias[tid] : 0.0f;
    float gate_acc = GateHasBias ? gate_bias[tid] : 0.0f;
    for (int k = 0; k < kDim; ++k) {
      float core_qx = clamp_round_int8_as_float(core_input[base + k] / core_act_scale);
      float gate_qx = clamp_round_int8_as_float(gate_input[base + k] / gate_act_scale);
      core_acc += core_qx * core_act_scale *
                  static_cast<float>(core_q_weight[tid * kDim + k]) * core_weight_scale[tid];
      gate_acc += gate_qx * gate_act_scale *
                  static_cast<float>(gate_q_weight[tid * kDim + k]) * gate_weight_scale[tid];
    }
    s_core_out[tid] = core_acc;
    s_gate_out[tid] = gate_acc;
  }
  __syncthreads();

  float core_sum = tid < kDim ? s_core_out[tid] : 0.0f;
  float gate_sum = tid < kDim ? s_gate_out[tid] : 0.0f;
  s_reduce_core[tid] = core_sum;
  s_reduce_gate[tid] = gate_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_mean = s_reduce_core[0] * (1.0f / static_cast<float>(kDim));
  const float gate_mean = s_reduce_gate[0] * (1.0f / static_cast<float>(kDim));

  float core_centered = tid < kDim ? s_core_out[tid] - core_mean : 0.0f;
  float gate_centered = tid < kDim ? s_gate_out[tid] - gate_mean : 0.0f;
  s_reduce_core[tid] = tid < kDim ? core_centered * core_centered : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? gate_centered * gate_centered : 0.0f;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_rstd = rsqrtf(s_reduce_core[0] * (1.0f / static_cast<float>(kDim)) + eps);
  const float gate_rstd = rsqrtf(s_reduce_gate[0] * (1.0f / static_cast<float>(kDim)) + eps);

  float core_xhat = core_centered * core_rstd;
  float gate_xhat = gate_centered * gate_rstd;
  float grad_core_norm = 0.0f;
  float grad_gate_norm = 0.0f;
  if (tid < kDim) {
    float core_ln = core_xhat * core_norm_weight[tid] + core_norm_bias[tid];
    float gate_ln = gate_xhat * gate_norm_weight[tid] + gate_norm_bias[tid];
    float core_sig = sigmoidf_stable(core_ln);
    float core_act = core_ln * core_sig;
    float gate_act = sigmoidf_stable(gate_ln);
    float grad = grad_out[base + tid];
    float core_silu_grad = core_sig * (1.0f + core_ln * (1.0f - core_sig));
    float grad_core_ln = grad * gate_act * core_silu_grad;
    float grad_gate_ln = grad * core_act * gate_act * (1.0f - gate_act);
    grad_core_norm = grad_core_ln * core_norm_weight[tid];
    grad_gate_norm = grad_gate_ln * gate_norm_weight[tid];
  }

  s_reduce_core[tid] = tid < kDim ? grad_core_norm : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm : 0.0f;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce_core[tid] += s_reduce_core[tid + stride];
      s_reduce_gate[tid] += s_reduce_gate[tid + stride];
    }
    __syncthreads();
  }
  const float core_sum_grad = s_reduce_core[0];
  const float gate_sum_grad = s_reduce_gate[0];

  s_reduce_core[tid] = tid < kDim ? grad_core_norm * core_xhat : 0.0f;
  s_reduce_gate[tid] = tid < kDim ? grad_gate_norm * gate_xhat : 0.0f;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
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
    s_core_grad_proj[tid] =
        (grad_core_norm * static_cast<float>(kDim) - core_sum_grad - core_xhat * core_sum_grad_xhat) *
        core_rstd * inv_dim;
    s_gate_grad_proj[tid] =
        (grad_gate_norm * static_cast<float>(kDim) - gate_sum_grad - gate_xhat * gate_sum_grad_xhat) *
        gate_rstd * inv_dim;
  }
  __syncthreads();

  if (tid < kDim) {
    float core_grad = 0.0f;
    float gate_grad = 0.0f;
    for (int j = 0; j < kDim; ++j) {
      core_grad += s_core_grad_proj[j] *
                   static_cast<float>(core_q_weight[j * kDim + tid]) * core_weight_scale[j];
      gate_grad += s_gate_grad_proj[j] *
                   static_cast<float>(gate_q_weight[j * kDim + tid]) * gate_weight_scale[j];
    }
    grad_core_input[base + tid] = core_grad;
    grad_gate_input[base + tid] = gate_grad;
  }
}

}  // namespace

std::vector<torch::Tensor> w8a8_dual_gated_tail_input_grad_backward_n128(
    const torch::Tensor &grad_out,
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
  TORCH_CHECK(grad_out.is_cuda() && core_input.is_cuda() && gate_input.is_cuda(),
              "w8a8 gated-tail bwd: tensors must be CUDA");
  TORCH_CHECK(core_q_weight.is_cuda() && gate_q_weight.is_cuda() &&
                  core_weight_scale.is_cuda() && gate_weight_scale.is_cuda(),
              "w8a8 gated-tail bwd: weights/scales must be CUDA");
  TORCH_CHECK(core_activation_scale.is_cuda() && gate_activation_scale.is_cuda(),
              "w8a8 gated-tail bwd: activation scales must be CUDA");
  TORCH_CHECK(core_norm_weight.is_cuda() && core_norm_bias.is_cuda() &&
                  gate_norm_weight.is_cuda() && gate_norm_bias.is_cuda(),
              "w8a8 gated-tail bwd: norm tensors must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32 &&
                  core_input.scalar_type() == torch::kFloat32 &&
                  gate_input.scalar_type() == torch::kFloat32,
              "w8a8 gated-tail bwd: float tensors must be float32");
  TORCH_CHECK(core_q_weight.scalar_type() == torch::kInt8 && gate_q_weight.scalar_type() == torch::kInt8,
              "w8a8 gated-tail bwd: q weights must be int8");
  TORCH_CHECK(core_weight_scale.scalar_type() == torch::kFloat32 &&
                  gate_weight_scale.scalar_type() == torch::kFloat32 &&
                  core_activation_scale.scalar_type() == torch::kFloat32 &&
                  gate_activation_scale.scalar_type() == torch::kFloat32 &&
                  core_norm_weight.scalar_type() == torch::kFloat32 &&
                  core_norm_bias.scalar_type() == torch::kFloat32 &&
                  gate_norm_weight.scalar_type() == torch::kFloat32 &&
                  gate_norm_bias.scalar_type() == torch::kFloat32,
              "w8a8 gated-tail bwd: scales/norm tensors must be float32");
  TORCH_CHECK(core_input.dim() == 2 && gate_input.sizes() == core_input.sizes(),
              "w8a8 gated-tail bwd: inputs must be matching 2D tensors");
  TORCH_CHECK(grad_out.sizes() == core_input.sizes(), "w8a8 gated-tail bwd: grad_out shape mismatch");
  TORCH_CHECK(core_input.size(1) == kDim, "w8a8 gated-tail bwd: input dim must be 128");
  TORCH_CHECK(core_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}) &&
                  gate_q_weight.sizes() == torch::IntArrayRef({kDim, kDim}),
              "w8a8 gated-tail bwd: q weights must be [128, 128]");
  TORCH_CHECK(core_weight_scale.numel() == kDim && gate_weight_scale.numel() == kDim,
              "w8a8 gated-tail bwd: weight scales must be [128]");
  TORCH_CHECK(core_activation_scale.numel() == 1 && gate_activation_scale.numel() == 1,
              "w8a8 gated-tail bwd: activation scales must be scalar");
  TORCH_CHECK(core_norm_weight.numel() == kDim && core_norm_bias.numel() == kDim &&
                  gate_norm_weight.numel() == kDim && gate_norm_bias.numel() == kDim,
              "w8a8 gated-tail bwd: norm tensors must be [128]");

  auto grad_out_c = grad_out.contiguous();
  auto core_input_c = core_input.contiguous();
  auto gate_input_c = gate_input.contiguous();
  auto core_q_weight_c = core_q_weight.contiguous();
  auto gate_q_weight_c = gate_q_weight.contiguous();
  auto core_weight_scale_c = core_weight_scale.contiguous();
  auto gate_weight_scale_c = gate_weight_scale.contiguous();
  auto core_activation_scale_c = core_activation_scale.reshape({}).contiguous();
  auto gate_activation_scale_c = gate_activation_scale.reshape({}).contiguous();
  auto core_norm_weight_c = core_norm_weight.contiguous();
  auto core_norm_bias_c = core_norm_bias.contiguous();
  auto gate_norm_weight_c = gate_norm_weight.contiguous();
  auto gate_norm_bias_c = gate_norm_bias.contiguous();
  auto core_bias_c = core_has_bias ? core_bias.contiguous() : core_input_c.new_empty({0});
  auto gate_bias_c = gate_has_bias ? gate_bias.contiguous() : gate_input_c.new_empty({0});

  TORCH_CHECK(!core_has_bias || (core_bias_c.is_cuda() && core_bias_c.scalar_type() == torch::kFloat32 &&
                                 core_bias_c.numel() == kDim),
              "w8a8 gated-tail bwd: core bias must be float32 [128]");
  TORCH_CHECK(!gate_has_bias || (gate_bias_c.is_cuda() && gate_bias_c.scalar_type() == torch::kFloat32 &&
                                 gate_bias_c.numel() == kDim),
              "w8a8 gated-tail bwd: gate bias must be float32 [128]");

  auto grad_core = torch::empty_like(core_input_c);
  auto grad_gate = torch::empty_like(gate_input_c);
  int64_t rows = core_input_c.size(0);

  dim3 grid(rows);
  dim3 block(kThreads);
  if (core_has_bias && gate_has_bias) {
    w8a8_dual_gated_tail_input_grad_backward_n128_kernel<true, true><<<grid, block>>>(
        grad_out_c.data_ptr<float>(), core_input_c.data_ptr<float>(), gate_input_c.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(), gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(), gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_c.data_ptr<float>(), gate_activation_scale_c.data_ptr<float>(),
        core_bias_c.data_ptr<float>(), gate_bias_c.data_ptr<float>(),
        core_norm_weight_c.data_ptr<float>(), core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(), gate_norm_bias_c.data_ptr<float>(),
        grad_core.data_ptr<float>(), grad_gate.data_ptr<float>(), rows, static_cast<float>(eps));
  } else if (core_has_bias) {
    w8a8_dual_gated_tail_input_grad_backward_n128_kernel<true, false><<<grid, block>>>(
        grad_out_c.data_ptr<float>(), core_input_c.data_ptr<float>(), gate_input_c.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(), gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(), gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_c.data_ptr<float>(), gate_activation_scale_c.data_ptr<float>(),
        core_bias_c.data_ptr<float>(), gate_bias_c.data_ptr<float>(),
        core_norm_weight_c.data_ptr<float>(), core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(), gate_norm_bias_c.data_ptr<float>(),
        grad_core.data_ptr<float>(), grad_gate.data_ptr<float>(), rows, static_cast<float>(eps));
  } else if (gate_has_bias) {
    w8a8_dual_gated_tail_input_grad_backward_n128_kernel<false, true><<<grid, block>>>(
        grad_out_c.data_ptr<float>(), core_input_c.data_ptr<float>(), gate_input_c.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(), gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(), gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_c.data_ptr<float>(), gate_activation_scale_c.data_ptr<float>(),
        core_bias_c.data_ptr<float>(), gate_bias_c.data_ptr<float>(),
        core_norm_weight_c.data_ptr<float>(), core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(), gate_norm_bias_c.data_ptr<float>(),
        grad_core.data_ptr<float>(), grad_gate.data_ptr<float>(), rows, static_cast<float>(eps));
  } else {
    w8a8_dual_gated_tail_input_grad_backward_n128_kernel<false, false><<<grid, block>>>(
        grad_out_c.data_ptr<float>(), core_input_c.data_ptr<float>(), gate_input_c.data_ptr<float>(),
        core_q_weight_c.data_ptr<int8_t>(), gate_q_weight_c.data_ptr<int8_t>(),
        core_weight_scale_c.data_ptr<float>(), gate_weight_scale_c.data_ptr<float>(),
        core_activation_scale_c.data_ptr<float>(), gate_activation_scale_c.data_ptr<float>(),
        core_bias_c.data_ptr<float>(), gate_bias_c.data_ptr<float>(),
        core_norm_weight_c.data_ptr<float>(), core_norm_bias_c.data_ptr<float>(),
        gate_norm_weight_c.data_ptr<float>(), gate_norm_bias_c.data_ptr<float>(),
        grad_core.data_ptr<float>(), grad_gate.data_ptr<float>(), rows, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_core, grad_gate};
}
