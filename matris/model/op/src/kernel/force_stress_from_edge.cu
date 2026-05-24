#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;
constexpr float kStressScale = 160.21766208f;

__global__ void force_stress_from_edge_kernel(
    const float* __restrict__ grad_edge,
    const float* __restrict__ edge_vectors,
    const int64_t* __restrict__ target,
    const int64_t* __restrict__ source,
    const int64_t* __restrict__ atom_segment,
    float* __restrict__ force,
    float* __restrict__ raw_stress,
    int64_t edges) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= edges) {
    return;
  }

  int64_t t = target[e];
  int64_t s = source[e];
  int64_t graph = atom_segment[t];
  const float gx = grad_edge[e * 3 + 0];
  const float gy = grad_edge[e * 3 + 1];
  const float gz = grad_edge[e * 3 + 2];
  const float vx = edge_vectors[e * 3 + 0];
  const float vy = edge_vectors[e * 3 + 1];
  const float vz = edge_vectors[e * 3 + 2];

  atomicAdd(force + t * 3 + 0, -gx);
  atomicAdd(force + t * 3 + 1, -gy);
  atomicAdd(force + t * 3 + 2, -gz);
  atomicAdd(force + s * 3 + 0, gx);
  atomicAdd(force + s * 3 + 1, gy);
  atomicAdd(force + s * 3 + 2, gz);

  float v[3] = {vx, vy, vz};
  float g[3] = {gx, gy, gz};
  int64_t base = graph * 9;
  #pragma unroll
  for (int a = 0; a < 3; ++a) {
    #pragma unroll
    for (int b = 0; b < 3; ++b) {
      float value = 0.5f * (v[a] * g[b] + v[b] * g[a]);
      atomicAdd(raw_stress + base + a * 3 + b, value);
    }
  }
}

__global__ void scale_stress_kernel(
    const float* __restrict__ raw_stress,
    const float* __restrict__ volumes,
    float* __restrict__ stress,
    int64_t graphs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(graphs * 9);
  if (idx >= total) {
    return;
  }
  int graph = idx / 9;
  stress[idx] = raw_stress[idx] * (kStressScale / volumes[graph]);
}

}  // namespace

std::vector<torch::Tensor> force_stress_from_edge_vectors(
    const torch::Tensor &grad_edge,
    const torch::Tensor &edge_vectors,
    const torch::Tensor &target,
    const torch::Tensor &source,
    const torch::Tensor &atom_segment,
    const torch::Tensor &volumes) {
  TORCH_CHECK(grad_edge.is_cuda() && edge_vectors.is_cuda() && target.is_cuda() &&
                  source.is_cuda() && atom_segment.is_cuda() && volumes.is_cuda(),
              "force_stress_from_edge_vectors: tensors must be CUDA");
  TORCH_CHECK(grad_edge.scalar_type() == torch::kFloat32 &&
                  edge_vectors.scalar_type() == torch::kFloat32 &&
                  volumes.scalar_type() == torch::kFloat32,
              "force_stress_from_edge_vectors: float tensors must be float32");
  TORCH_CHECK(target.scalar_type() == torch::kInt64 &&
                  source.scalar_type() == torch::kInt64 &&
                  atom_segment.scalar_type() == torch::kInt64,
              "force_stress_from_edge_vectors: indices must be int64");
  TORCH_CHECK(grad_edge.dim() == 2 && grad_edge.size(1) == 3,
              "force_stress_from_edge_vectors: grad_edge must be [E, 3]");
  TORCH_CHECK(edge_vectors.dim() == 2 && edge_vectors.size(1) == 3,
              "force_stress_from_edge_vectors: edge_vectors must be [E, 3]");
  TORCH_CHECK(target.dim() == 1 && source.dim() == 1 &&
                  target.size(0) == grad_edge.size(0) &&
                  source.size(0) == grad_edge.size(0),
              "force_stress_from_edge_vectors: target/source must be [E]");
  TORCH_CHECK(atom_segment.dim() == 1,
              "force_stress_from_edge_vectors: atom_segment must be [N]");
  TORCH_CHECK(volumes.dim() == 3 && volumes.size(1) == 1 && volumes.size(2) == 1,
              "force_stress_from_edge_vectors: volumes must be [G, 1, 1]");

  auto grad_edge_c = grad_edge.contiguous();
  auto edge_vectors_c = edge_vectors.contiguous();
  auto target_c = target.contiguous();
  auto source_c = source.contiguous();
  auto atom_segment_c = atom_segment.contiguous();
  auto volumes_c = volumes.contiguous();

  int64_t edges = grad_edge_c.size(0);
  int64_t num_atoms = atom_segment_c.size(0);
  int64_t num_graphs = volumes_c.size(0);
  auto force = torch::zeros({num_atoms, 3}, grad_edge.options());
  auto raw_stress = torch::zeros({num_graphs, 3, 3}, grad_edge.options());
  auto stress = torch::empty({num_graphs, 3, 3}, grad_edge.options());

  int force_blocks = static_cast<int>((edges + kThreads - 1) / kThreads);
  force_stress_from_edge_kernel<<<force_blocks, kThreads>>>(
      grad_edge_c.data_ptr<float>(),
      edge_vectors_c.data_ptr<float>(),
      target_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      atom_segment_c.data_ptr<int64_t>(),
      force.data_ptr<float>(),
      raw_stress.data_ptr<float>(),
      edges);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int stress_blocks = static_cast<int>((num_graphs * 9 + kThreads - 1) / kThreads);
  scale_stress_kernel<<<stress_blocks, kThreads>>>(
      raw_stress.data_ptr<float>(),
      volumes_c.data_ptr<float>(),
      stress.data_ptr<float>(),
      num_graphs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {force, stress};
}
