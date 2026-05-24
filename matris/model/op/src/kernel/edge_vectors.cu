#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;

__global__ void edge_vectors_forward_kernel(
    const float* __restrict__ coords,
    const float* __restrict__ lattice,
    const float* __restrict__ image,
    const int64_t* __restrict__ target,
    const int64_t* __restrict__ source,
    float* __restrict__ edge_vectors,
    int64_t edges,
    int64_t lattice_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(edges * 3);
  if (idx >= total) {
    return;
  }
  int e = idx / 3;
  int c = idx - e * 3;
  int64_t t = target[e];
  int64_t s = source[e];
  float image_lattice = 0.0f;
  for (int64_t k = 0; k < lattice_rows; ++k) {
    image_lattice += image[e * lattice_rows + k] * lattice[k * 3 + c];
  }
  edge_vectors[idx] = coords[t * 3 + c] - coords[s * 3 + c] - image_lattice;
}

__global__ void edge_vectors_backward_coords_kernel(
    const float* __restrict__ grad_edge,
    const int64_t* __restrict__ target,
    const int64_t* __restrict__ source,
    float* __restrict__ grad_coords,
    int64_t edges) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(edges * 3);
  if (idx >= total) {
    return;
  }
  int e = idx / 3;
  int c = idx - e * 3;
  float g = grad_edge[idx];
  atomicAdd(grad_coords + target[e] * 3 + c, g);
  atomicAdd(grad_coords + source[e] * 3 + c, -g);
}

__global__ void edge_vectors_backward_lattice_kernel(
    const float* __restrict__ grad_edge,
    const float* __restrict__ image,
    float* __restrict__ grad_lattice,
    int64_t edges,
    int64_t lattice_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = static_cast<int>(edges * lattice_rows * 3);
  if (idx >= total) {
    return;
  }
  int c = idx % 3;
  int tmp = idx / 3;
  int k = tmp % static_cast<int>(lattice_rows);
  int e = tmp / static_cast<int>(lattice_rows);
  float img = image[e * lattice_rows + k];
  if (img != 0.0f) {
    atomicAdd(grad_lattice + k * 3 + c, -img * grad_edge[e * 3 + c]);
  }
}

}  // namespace

torch::Tensor edge_vectors_forward(const torch::Tensor &coords,
                                   const torch::Tensor &lattice,
                                   const torch::Tensor &image,
                                   const torch::Tensor &target,
                                   const torch::Tensor &source) {
  TORCH_CHECK(coords.is_cuda() && lattice.is_cuda() && image.is_cuda() && target.is_cuda() && source.is_cuda(),
              "edge_vectors_forward: tensors must be CUDA");
  TORCH_CHECK(coords.scalar_type() == torch::kFloat32 && lattice.scalar_type() == torch::kFloat32 &&
                  image.scalar_type() == torch::kFloat32,
              "edge_vectors_forward: float tensors must be float32");
  TORCH_CHECK(target.scalar_type() == torch::kInt64 && source.scalar_type() == torch::kInt64,
              "edge_vectors_forward: indices must be int64");
  TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 3, "edge_vectors_forward: coords must be [N, 3]");
  TORCH_CHECK(lattice.dim() == 2 && lattice.size(1) == 3, "edge_vectors_forward: lattice must be [K, 3]");
  TORCH_CHECK(image.dim() == 2 && image.size(1) == lattice.size(0), "edge_vectors_forward: image must be [E, K]");
  TORCH_CHECK(target.dim() == 1 && source.dim() == 1 && target.size(0) == source.size(0),
              "edge_vectors_forward: indices must be [E]");
  TORCH_CHECK(image.size(0) == target.size(0), "edge_vectors_forward: image/indices mismatch");

  auto coords_c = coords.contiguous();
  auto lattice_c = lattice.contiguous();
  auto image_c = image.contiguous();
  auto target_c = target.contiguous();
  auto source_c = source.contiguous();
  auto out = torch::empty({target.size(0), 3}, coords.options());
  int64_t edges = target.size(0);
  int blocks = static_cast<int>((edges * 3 + kThreads - 1) / kThreads);
  edge_vectors_forward_kernel<<<blocks, kThreads>>>(
      coords_c.data_ptr<float>(),
      lattice_c.data_ptr<float>(),
      image_c.data_ptr<float>(),
      target_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      edges,
      lattice.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> edge_vectors_backward(const torch::Tensor &grad_edge,
                                                 const torch::Tensor &image,
                                                 const torch::Tensor &target,
                                                 const torch::Tensor &source,
                                                 int64_t num_coords,
                                                 int64_t lattice_rows) {
  TORCH_CHECK(grad_edge.is_cuda() && image.is_cuda() && target.is_cuda() && source.is_cuda(),
              "edge_vectors_backward: tensors must be CUDA");
  TORCH_CHECK(grad_edge.scalar_type() == torch::kFloat32 && image.scalar_type() == torch::kFloat32,
              "edge_vectors_backward: float tensors must be float32");
  TORCH_CHECK(target.scalar_type() == torch::kInt64 && source.scalar_type() == torch::kInt64,
              "edge_vectors_backward: indices must be int64");
  TORCH_CHECK(grad_edge.dim() == 2 && grad_edge.size(1) == 3, "edge_vectors_backward: grad_edge must be [E, 3]");

  auto grad_edge_c = grad_edge.contiguous();
  auto image_c = image.contiguous();
  auto target_c = target.contiguous();
  auto source_c = source.contiguous();
  auto grad_coords = torch::zeros({num_coords, 3}, grad_edge.options());
  auto grad_lattice = torch::zeros({lattice_rows, 3}, grad_edge.options());
  int64_t edges = grad_edge.size(0);
  int coord_blocks = static_cast<int>((edges * 3 + kThreads - 1) / kThreads);
  int lattice_blocks = static_cast<int>((edges * lattice_rows * 3 + kThreads - 1) / kThreads);
  edge_vectors_backward_coords_kernel<<<coord_blocks, kThreads>>>(
      grad_edge_c.data_ptr<float>(),
      target_c.data_ptr<int64_t>(),
      source_c.data_ptr<int64_t>(),
      grad_coords.data_ptr<float>(),
      edges);
  edge_vectors_backward_lattice_kernel<<<lattice_blocks, kThreads>>>(
      grad_edge_c.data_ptr<float>(),
      image_c.data_ptr<float>(),
      grad_lattice.data_ptr<float>(),
      edges,
      lattice_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_coords, grad_lattice};
}
