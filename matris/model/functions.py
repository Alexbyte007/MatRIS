from __future__ import annotations

from collections.abc import Sequence
import json
import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path
from .op import fused_silu, fused_sigmoid

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional.
    triton = None
    tl = None

class FusedSiLU(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_silu(x)
        else:
            return torch.nn.functional.silu(x) 

class FusedSigmoid(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_sigmoid(x)
        else:
            return torch.nn.functional.sigmoid(x)

def get_activation(name: str) -> nn.Module:
    """Return an activation function"""
    activation_map = {
        "relu": nn.ReLU,
        "silu": FusedSiLU,  # Using fused version for better performance
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "sigmoid": FusedSigmoid,  # Using fused version for better performance
        "tanh": nn.Tanh,
    }
    
    name_lower = name.lower()
    if name_lower not in activation_map:
        raise NotImplementedError(
            f"Activation '{name}' is not implemented. "
            f"Supported activations: {list(activation_map.keys())}"
        )
    return activation_map[name_lower]()

def get_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function"""
    if name is None:
        return None
        
    normalization_map = {
        "layer": nn.LayerNorm(dim),
        "rms": nn.RMSNorm(dim), # torch >= 2.6.0
        "batch": nn.BatchNorm1d(dim),
    }
    name_lower = name.lower()
    return normalization_map[name_lower]


def _graph_op_profile_path() -> str:
    return os.environ.get("MATRIS_GRAPH_OP_PROFILE_PATH", "").strip()


def _profiled_cuda_start(data: Tensor):
    if not _graph_op_profile_path() or data.device.type != "cuda":
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _profiled_cuda_finish(
    handle,
    *,
    op: str,
    name: str | None,
    data: Tensor,
    segment: Tensor,
    num_segment: int | None,
    average: bool | None = None,
) -> None:
    path = _graph_op_profile_path()
    if not path:
        return
    elapsed_ms = None
    if handle is not None:
        start, end = handle
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))

    with torch.no_grad():
        segment_cpu = segment.detach()
        segment_numel = int(segment_cpu.numel())
        segment_max_plus_one = 0
        if segment_numel:
            segment_max_plus_one = int(segment_cpu.max().item()) + 1
        resolved_num_segment = int(num_segment) if num_segment is not None else segment_max_plus_one
        record = {
            "op": op,
            "name": name or op,
            "elapsed_ms": elapsed_ms,
            "data_shape": list(data.shape),
            "data_dtype": str(data.dtype),
            "segment_numel": segment_numel,
            "segment_max_plus_one": segment_max_plus_one,
            "num_segment": resolved_num_segment,
            "feature_dim": int(data.shape[1]) if data.ndim == 2 else None,
            "average": average,
            "requires_grad": bool(data.requires_grad),
        }
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


if triton is not None:

    @triton.jit
    def _segment_softmax_sorted_forward_kernel(
        x_ptr,
        offsets_ptr,
        lengths_ptr,
        out_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        x = tl.load(
            x_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        max_v = tl.max(x, axis=0)
        exp_v = tl.exp(x - max_v[None, :])
        exp_v = tl.where(offs_r[:, None] < length, exp_v, 0.0)
        sum_v = tl.sum(exp_v, axis=0)
        y = exp_v / sum_v[None, :]
        tl.store(
            out_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            y,
            mask=mask,
        )

    @triton.jit
    def _segment_softmax_sorted_backward_kernel(
        grad_out_ptr,
        out_ptr,
        offsets_ptr,
        lengths_ptr,
        grad_x_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        y = tl.load(
            out_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            grad_out_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(grad_out * y, axis=0)
        grad_x = y * (grad_out - dot[None, :])
        tl.store(
            grad_x_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            grad_x,
            mask=mask,
        )

    @triton.jit
    def _segment_softmax_weighted_sum_forward_kernel(
        alpha_logits_ptr,
        values_ptr,
        offsets_ptr,
        lengths_ptr,
        alpha_ptr,
        out_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        valid_segment = length > 0
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        logits = tl.load(
            alpha_logits_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        max_v = tl.max(logits, axis=0)
        max_v = tl.where(valid_segment, max_v, 0.0)
        exp_v = tl.exp(logits - max_v[None, :])
        exp_v = tl.where(offs_r[:, None] < length, exp_v, 0.0)
        sum_v = tl.sum(exp_v, axis=0)
        alpha = exp_v / sum_v[None, :]
        alpha = tl.where(valid_segment, alpha, 0.0)
        values = tl.load(
            values_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(alpha * values, axis=0)
        tl.store(
            alpha_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            alpha,
            mask=mask,
        )
        tl.store(out_ptr + pid_s * dim + offs_d, out, mask=offs_d < dim)

    @triton.jit
    def _segment_softmax_weighted_sum_backward_kernel(
        grad_out_ptr,
        values_ptr,
        alpha_ptr,
        out_ptr,
        offsets_ptr,
        lengths_ptr,
        grad_alpha_logits_ptr,
        grad_values_ptr,
        num_segments: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_r = tl.arange(0, BLOCK_R)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        start = tl.load(offsets_ptr + pid_s)
        length = tl.load(lengths_ptr + pid_s)
        mask = (offs_r[:, None] < length) & (offs_d[None, :] < dim)
        grad_out = tl.load(grad_out_ptr + pid_s * dim + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        out = tl.load(out_ptr + pid_s * dim + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        values = tl.load(
            values_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        alpha = tl.load(
            alpha_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_values = alpha * grad_out[None, :]
        grad_alpha_logits = alpha * grad_out[None, :] * (values - out[None, :])
        tl.store(
            grad_values_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            grad_values,
            mask=mask,
        )
        tl.store(
            grad_alpha_logits_ptr + (start + offs_r[:, None]) * dim + offs_d[None, :],
            grad_alpha_logits,
            mask=mask,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (max(value, 1) - 1).bit_length()


def _use_triton_sorted_segment_softmax() -> bool:
    return os.environ.get("MATRIS_USE_TRITON_SORTED_SEGMENT_SOFTMAX", "0") == "1"


def _use_torch_sorted_segment_reduce() -> bool:
    return os.environ.get("MATRIS_USE_TORCH_SORTED_SEGMENT_REDUCE", "0") == "1"


def _use_triton_target_attention_sum() -> bool:
    return os.environ.get("MATRIS_USE_TRITON_TARGET_ATTENTION_SUM", "0") == "1"


def _use_cuda_target_attention_sum() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_TARGET_ATTENTION_SUM", "0") == "1"


def _use_cuda_target_attention_bwd() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_TARGET_ATTENTION_BWD", "0") == "1"


def _use_cuda_directed2undirected_average() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE", "0") == "1"


def use_cuda_edge_vectors() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_EDGE_VECTORS", "0") == "1"


def _use_cuda_fused_line_attention() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_FUSED_LINE_ATTENTION", "0") == "1"


def _use_cuda_fused_atom_attention() -> bool:
    return os.environ.get("MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION", "0") == "1"


def _load_matris_op():
    try:
        import matris_op  # type: ignore

        return matris_op
    except Exception:
        op_src = Path(__file__).resolve().parent / "op" / "src"
        if op_src.exists() and str(op_src) not in sys.path:
            sys.path.insert(0, str(op_src))
        try:
            import matris_op  # type: ignore

            return matris_op
        except Exception:
            return None


class _TritonSortedSegmentSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feas: Tensor, bin_count: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        if feas.ndim != 2:
            raise RuntimeError("sorted segment softmax expects a 2D tensor")
        if not feas.is_cuda or not bin_count.is_cuda:
            raise RuntimeError("sorted segment softmax expects CUDA tensors")
        num_rows, dim = feas.shape
        if dim != 128:
            raise RuntimeError("sorted segment softmax currently specializes dim=128")

        lengths = bin_count.to(device=feas.device, dtype=torch.int64).contiguous()
        offsets = torch.cumsum(lengths, dim=0) - lengths
        max_len = int(lengths.max().item()) if lengths.numel() else 1
        block_r = _next_power_of_2(max_len)
        if block_r > 128:
            raise RuntimeError("sorted segment softmax currently supports max segment length <= 128")
        block_d = 32
        out = torch.empty_like(feas)
        grid = (lengths.numel(), triton.cdiv(dim, block_d))
        _segment_softmax_sorted_forward_kernel[grid](
            feas,
            offsets,
            lengths,
            out,
            lengths.numel(),
            dim,
            block_r,
            block_d,
            num_warps=4,
        )
        ctx.save_for_backward(out, offsets, lengths)
        ctx.dim = dim
        ctx.block_r = block_r
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        out, offsets, lengths = ctx.saved_tensors
        grad_x = torch.empty_like(grad_out)
        grid = (lengths.numel(), triton.cdiv(ctx.dim, ctx.block_d))
        _segment_softmax_sorted_backward_kernel[grid](
            grad_out.contiguous(),
            out,
            offsets,
            lengths,
            grad_x,
            lengths.numel(),
            ctx.dim,
            ctx.block_r,
            ctx.block_d,
            num_warps=4,
        )
        return grad_x, None


class _TritonSegmentSoftmaxWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_logits: Tensor, values: Tensor, lengths: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        if alpha_logits.ndim != 2 or values.ndim != 2 or alpha_logits.shape != values.shape:
            raise RuntimeError("weighted segment softmax expects matching 2D tensors")
        if not alpha_logits.is_cuda or not values.is_cuda or not lengths.is_cuda:
            raise RuntimeError("weighted segment softmax expects CUDA tensors")
        _, dim = alpha_logits.shape
        if dim != 128:
            raise RuntimeError("weighted segment softmax currently specializes dim=128")

        lengths = lengths.to(device=alpha_logits.device, dtype=torch.int64).contiguous()
        offsets = torch.cumsum(lengths, dim=0) - lengths
        max_len = int(lengths.max().item()) if lengths.numel() else 1
        block_r = _next_power_of_2(max_len)
        if block_r > 128:
            raise RuntimeError("weighted segment softmax currently supports max segment length <= 128")
        block_d = 32
        alpha = torch.empty_like(alpha_logits)
        out = torch.empty((lengths.numel(), dim), device=alpha_logits.device, dtype=alpha_logits.dtype)
        grid = (lengths.numel(), triton.cdiv(dim, block_d))
        _segment_softmax_weighted_sum_forward_kernel[grid](
            alpha_logits,
            values,
            offsets,
            lengths,
            alpha,
            out,
            lengths.numel(),
            dim,
            block_r,
            block_d,
            num_warps=4,
        )
        ctx.save_for_backward(values, alpha, out, offsets, lengths)
        ctx.dim = dim
        ctx.block_r = block_r
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        values, alpha, out, offsets, lengths = ctx.saved_tensors
        grad_alpha_logits = torch.empty_like(alpha)
        grad_values = torch.empty_like(values)
        grid = (lengths.numel(), triton.cdiv(ctx.dim, ctx.block_d))
        _segment_softmax_weighted_sum_backward_kernel[grid](
            grad_out.contiguous(),
            values,
            alpha,
            out,
            offsets,
            lengths,
            grad_alpha_logits,
            grad_values,
            lengths.numel(),
            ctx.dim,
            ctx.block_r,
            ctx.block_d,
            num_warps=4,
        )
        return grad_alpha_logits, grad_values, None


class _CudaTargetAttentionSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_logits: Tensor, values: Tensor, lengths: Tensor) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "target_attention_sum_forward"):
            raise RuntimeError("matris_op.target_attention_sum_forward is unavailable")
        out, alpha = matris_op.target_attention_sum_forward(alpha_logits.contiguous(), values.contiguous(), lengths)
        ctx.save_for_backward(values, out, alpha, lengths)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        values, out, alpha, lengths = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "target_attention_sum_backward"):
            raise RuntimeError("matris_op.target_attention_sum_backward is unavailable")
        grad_logits, grad_values = matris_op.target_attention_sum_backward(
            grad_out.contiguous(),
            values,
            out,
            alpha,
            lengths,
        )
        return grad_logits, grad_values, None


class _CudaTargetAttentionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_logits: Tensor, values: Tensor, segment: Tensor, lengths: Tensor) -> Tensor:
        alpha_logits = alpha_logits.contiguous()
        values = values.contiguous()
        alpha = Dimwise_softmax(alpha_logits, segment)
        safe_lengths = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
        out = aggregate(alpha * values, segment, bin_count=safe_lengths, average=False, num_segment=lengths.numel())
        ctx.save_for_backward(values, out, alpha, lengths)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        values, out, alpha, lengths = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "target_attention_sum_backward"):
            raise RuntimeError("matris_op.target_attention_sum_backward is unavailable")
        grad_logits, grad_values = matris_op.target_attention_sum_backward(
            grad_out.contiguous(),
            values,
            out,
            alpha,
            lengths,
        )
        return grad_logits, grad_values, None, None


class _CudaDirected2UndirectedAverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: Tensor, segment: Tensor, num_segment: int) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed2undirected_average_forward"):
            raise RuntimeError("matris_op.directed2undirected_average_forward is unavailable")
        segment = segment.contiguous()
        out = matris_op.directed2undirected_average_forward(data.contiguous(), segment, int(num_segment))
        ctx.save_for_backward(segment)
        ctx.rows = int(data.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (segment,) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "directed2undirected_average_backward"):
            raise RuntimeError("matris_op.directed2undirected_average_backward is unavailable")
        grad_data = matris_op.directed2undirected_average_backward(grad_out.contiguous(), segment, ctx.rows)
        return grad_data, None, None


def directed2undirected_average_or_none(
    data: Tensor,
    segment: Tensor,
    num_segment,
    enable_hint: bool,
) -> Tensor | None:
    if not _use_cuda_directed2undirected_average() or not enable_hint:
        return None
    if data.ndim != 2 or data.shape[1] != 128 or data.dtype != torch.float32 or not data.is_cuda:
        return None
    if segment.ndim != 1 or segment.numel() != data.shape[0] or segment.dtype != torch.int64 or not segment.is_cuda:
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    if data.shape[0] != resolved_num_segment * 2:
        return None
    try:
        return _CudaDirected2UndirectedAverage.apply(data, segment, resolved_num_segment)
    except RuntimeError:
        return None


class _CudaEdgeVectors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords: Tensor, lattice: Tensor, image: Tensor, target: Tensor, source: Tensor) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "edge_vectors_forward"):
            raise RuntimeError("matris_op.edge_vectors_forward is unavailable")
        image = image.contiguous()
        target = target.contiguous()
        source = source.contiguous()
        out = matris_op.edge_vectors_forward(coords.contiguous(), lattice.contiguous(), image, target, source)
        ctx.save_for_backward(image, target, source)
        ctx.num_coords = int(coords.shape[0])
        ctx.lattice_rows = int(lattice.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        image, target, source = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "edge_vectors_backward"):
            raise RuntimeError("matris_op.edge_vectors_backward is unavailable")
        grad_coords, grad_lattice = matris_op.edge_vectors_backward(
            grad_out.contiguous(),
            image,
            target,
            source,
            ctx.num_coords,
            ctx.lattice_rows,
        )
        return grad_coords, grad_lattice, None, None, None


def edge_vectors_or_none(
    coords: Tensor,
    lattice: Tensor,
    image: Tensor,
    target: Tensor,
    source: Tensor,
) -> Tensor | None:
    if not use_cuda_edge_vectors():
        return None
    if (
        coords.ndim != 2
        or lattice.ndim != 2
        or image.ndim != 2
        or coords.shape[1] != 3
        or lattice.shape[1] != 3
        or image.shape[1] != lattice.shape[0]
        or coords.dtype != torch.float32
        or lattice.dtype != torch.float32
        or image.dtype != torch.float32
        or target.dtype != torch.int64
        or source.dtype != torch.int64
        or not coords.is_cuda
        or not lattice.is_cuda
        or not image.is_cuda
        or not target.is_cuda
        or not source.is_cuda
    ):
        return None
    try:
        return _CudaEdgeVectors.apply(coords, lattice, image, target, source)
    except RuntimeError:
        return None


class _CudaFusedLineAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source_logits: Tensor,
        target_logits: Tensor,
        values: Tensor,
        source_index: Tensor,
        target_index: Tensor,
        num_segments: int,
    ):
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "fused_line_attention_forward"):
            raise RuntimeError("matris_op.fused_line_attention_forward is unavailable")
        source_index = source_index.contiguous()
        target_index = target_index.contiguous()
        source_out, target_out, source_alpha, target_alpha = matris_op.fused_line_attention_forward(
            source_logits.contiguous(),
            target_logits.contiguous(),
            values.contiguous(),
            source_index,
            target_index,
            int(num_segments),
        )
        ctx.save_for_backward(values, source_out, target_out, source_alpha, target_alpha, source_index, target_index)
        return source_out, target_out

    @staticmethod
    def backward(ctx, grad_source_out: Tensor, grad_target_out: Tensor):
        values, source_out, target_out, source_alpha, target_alpha, source_index, target_index = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "fused_line_attention_backward"):
            raise RuntimeError("matris_op.fused_line_attention_backward is unavailable")
        grad_source_logits, grad_target_logits, grad_values = matris_op.fused_line_attention_backward(
            grad_source_out.contiguous(),
            grad_target_out.contiguous(),
            values,
            source_out,
            target_out,
            source_alpha,
            target_alpha,
            source_index,
            target_index,
        )
        return grad_source_logits, grad_target_logits, grad_values, None, None, None


def fused_line_attention_or_none(
    source_logits: Tensor,
    target_logits: Tensor,
    values: Tensor,
    source_index: Tensor,
    target_index: Tensor,
    num_segments: int,
    *,
    enable_hint: bool,
    atom_graph: bool = False,
) -> tuple[Tensor, Tensor] | None:
    enabled = _use_cuda_fused_atom_attention() if atom_graph else _use_cuda_fused_line_attention()
    if not enabled or not enable_hint:
        return None
    if (
        source_logits.ndim != 2
        or source_logits.shape != target_logits.shape
        or source_logits.shape != values.shape
        or source_logits.shape[1] != 128
        or source_logits.dtype != torch.float32
        or target_logits.dtype != torch.float32
        or values.dtype != torch.float32
        or source_index.dtype != torch.int64
        or target_index.dtype != torch.int64
        or not source_logits.is_cuda
        or not target_logits.is_cuda
        or not values.is_cuda
        or not source_index.is_cuda
        or not target_index.is_cuda
    ):
        return None
    try:
        source_out, target_out = _CudaFusedLineAttention.apply(
            source_logits,
            target_logits,
            values,
            source_index,
            target_index,
            int(num_segments),
        )
        return source_out, target_out
    except RuntimeError:
        return None


def _triton_sorted_segment_softmax_or_none(
    feas: Tensor,
    segment: Tensor,
    num_segment,
    bin_count: Tensor | None,
) -> Tensor | None:
    if not _use_triton_sorted_segment_softmax() or bin_count is None or triton is None:
        return None
    if feas.ndim != 2 or feas.shape[1] != 128 or not feas.is_cuda:
        return None
    if segment.numel() > 1 and not bool(torch.all(segment[1:] >= segment[:-1]).item()):
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    true_bin_count = torch.bincount(segment, minlength=resolved_num_segment).to(torch.int64)
    try:
        return _TritonSortedSegmentSoftmax.apply(feas, true_bin_count)
    except RuntimeError:
        return None


def _sorted_segment_reduce_or_none(
    data: Tensor,
    segment: Tensor,
    num_segment,
    average: bool,
    enable_hint: bool,
) -> Tensor | None:
    if not _use_torch_sorted_segment_reduce() or not enable_hint:
        return None
    if data.ndim != 2 or not data.is_cuda or segment.numel() == 0:
        return None
    if segment.numel() > 1 and not bool(torch.all(segment[1:] >= segment[:-1]).item()):
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    lengths = torch.bincount(segment, minlength=resolved_num_segment).to(torch.int64)
    out = torch.segment_reduce(data, "sum", lengths=lengths)
    if average:
        safe_lengths = lengths.where(lengths != 0, lengths.new_ones(1)).to(data.dtype)
        out = out / safe_lengths.reshape(-1, 1)
    return out


def segment_softmax_weighted_sum_sorted_or_none(
    alpha_logits: Tensor,
    values: Tensor,
    segment: Tensor,
    num_segment,
    enable_hint: bool,
) -> Tensor | None:
    if not _use_triton_target_attention_sum() or not enable_hint or triton is None:
        if not (_use_cuda_target_attention_sum() or _use_cuda_target_attention_bwd()) or not enable_hint:
            return None
    if alpha_logits.ndim != 2 or alpha_logits.shape != values.shape or alpha_logits.shape[1] != 128:
        return None
    if not alpha_logits.is_cuda or segment.numel() == 0:
        return None
    if segment.numel() > 1 and not bool(torch.all(segment[1:] >= segment[:-1]).item()):
        return None
    resolved_num_segment = int(num_segment) if num_segment is not None else int(segment.max().item()) + 1
    lengths = torch.bincount(segment, minlength=resolved_num_segment).to(torch.int64)
    if _use_cuda_target_attention_sum():
        try:
            return _CudaTargetAttentionSum.apply(alpha_logits, values, lengths)
        except RuntimeError:
            return None
    if _use_cuda_target_attention_bwd():
        try:
            return _CudaTargetAttentionBackward.apply(alpha_logits, values, segment, lengths)
        except RuntimeError:
            return None
    if not _use_triton_target_attention_sum() or triton is None:
        return None
    try:
        return _TritonSegmentSoftmaxWeightedSum.apply(alpha_logits, values, lengths)
    except RuntimeError:
        return None

class SwishLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to use bias in the linear layer. Default: True.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act = get_activation("silu")
    
    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas: shape (feas_num, in_dim)
            
        Returns:
            output: shape (feas_num, out_dim)
        """
        return self.act(self.linear(feas))

def Dimwise_softmax(
    feas: Tensor,
    segment: Tensor,
    num_segment=None,
    profile_name: str | None = None,
    bin_count: Tensor | None = None,
) -> Tensor:
    """Computes a sparsely evaluated softmax.
    
    Args:
        feas: The source tensor. shape: [num, dim]
        segment: specify the segment of each row [num, 1] 
    """
    triton_result = _triton_sorted_segment_softmax_or_none(feas, segment, num_segment, bin_count)
    if triton_result is not None:
        return triton_result

    profile_handle = _profiled_cuda_start(feas)
    num, dim = feas.shape
    original_dtype = feas.dtype
    reduce_dtype = torch.float32 if original_dtype in (torch.float16, torch.bfloat16) else original_dtype
    feas_for_reduce = feas.to(reduce_dtype)
    if num_segment is None:
        num_segment = int(segment.max()) + 1
    
    segment_expanded = segment.unsqueeze(1).expand(-1, dim) # [num, dim]
    
    feas_max = torch.empty(num_segment, dim, dtype=reduce_dtype, device=feas.device)
    feas_max.fill_(float("-inf"))
    feas_max = feas_max.scatter_reduce(
        0, segment_expanded, feas_for_reduce, reduce='amax', include_self=False,
    ) #[num_segment, dim]
    # Gather: [num_segment, dim] -> [num, dim]
    feas_max = feas_max[segment]
    out = (feas_for_reduce - feas_max).exp()
    
    # =========== scatter sum ============
    out_sum = torch.zeros(num_segment, dim, device=feas.device, dtype=reduce_dtype)
    out_sum = out_sum.scatter_reduce(
        0, segment_expanded, out, reduce='sum', include_self=False
    )
    # Gather: [num_segment, dim] -> [num, dim]
    out_sum = out_sum[segment]
    score = out / out_sum
    result = score.to(original_dtype)
    _profiled_cuda_finish(
        profile_handle,
        op="Dimwise_softmax",
        name=profile_name,
        data=feas,
        segment=segment,
        num_segment=num_segment,
        average=None,
    )
    return result

def aggregate(data: torch.Tensor, 
              segment: torch.Tensor, 
              bin_count: torch.Tensor = None, 
              average=True, 
              num_segment=None,
              profile_name: str | None = None) -> torch.Tensor:
    """Aggregate rows in data by specifying the segment.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        segment (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    used_precomputed_bin_count = bin_count is not None
    segment_reduce_output = _sorted_segment_reduce_or_none(
        data,
        segment,
        num_segment,
        average,
        enable_hint=bin_count is not None or average,
    )
    if segment_reduce_output is not None:
        _record_aggregate_bincount_stat(
            profile_name,
            path="sorted_segment_reduce",
            used_precomputed_bin_count=used_precomputed_bin_count,
            runtime_bincount=False,
            data=data,
            segment=segment,
            bin_count=bin_count,
            average=average,
            num_segment=num_segment,
        )
        return segment_reduce_output

    profile_handle = _profiled_cuda_start(data)
    runtime_bincount = False
    if bin_count is None:
        runtime_bincount = True
        bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

    if (num_segment is not None) and (bin_count.shape[0] != num_segment):
        difference = num_segment - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, segment, data)
    if average:
        output = (output.T / bin_count).T
    _profiled_cuda_finish(
        profile_handle,
        op="aggregate",
        name=profile_name,
        data=data,
        segment=segment,
        num_segment=num_segment,
        average=average,
    )
    _record_aggregate_bincount_stat(
        profile_name,
        path="index_add",
        used_precomputed_bin_count=used_precomputed_bin_count,
        runtime_bincount=runtime_bincount,
        data=data,
        segment=segment,
        bin_count=bin_count,
        average=average,
        num_segment=num_segment,
    )
    return output


def use_precomputed_aggregate_bincount() -> bool:
    return os.environ.get("MATRIS_USE_PRECOMPUTED_AGGREGATE_BINCOUNT", "0") == "1"


def use_directed2undirected_select_fastpath() -> bool:
    return os.environ.get("MATRIS_USE_DIRECTED2UNDIRECTED_SELECT_FASTPATH", "0") == "1"


_AGGREGATE_BINCOUNT_STATS: dict[str, object] = {
    "calls": 0,
    "runtime_bincount_calls": 0,
    "precomputed_bincount_calls": 0,
    "by_name": {},
}


def _record_aggregate_bincount_stat(
    profile_name: str | None,
    *,
    path: str,
    used_precomputed_bin_count: bool,
    runtime_bincount: bool,
    data: Tensor,
    segment: Tensor,
    bin_count: Tensor | None,
    average: bool,
    num_segment: int | None,
) -> None:
    stats_path = os.environ.get("MATRIS_AGGREGATE_BINCOUNT_STATS_PATH", "").strip()
    if not stats_path:
        return
    name = profile_name or "aggregate.unnamed"
    by_name = _AGGREGATE_BINCOUNT_STATS.setdefault("by_name", {})
    entry = by_name.setdefault(
        name,
        {
            "calls": 0,
            "runtime_bincount_calls": 0,
            "precomputed_bincount_calls": 0,
            "paths": {},
            "average": bool(average),
            "last_data_shape": None,
            "last_segment_shape": None,
            "last_bincount_shape": None,
            "last_num_segment": None,
        },
    )
    _AGGREGATE_BINCOUNT_STATS["calls"] = int(_AGGREGATE_BINCOUNT_STATS.get("calls", 0)) + 1
    entry["calls"] = int(entry.get("calls", 0)) + 1
    paths = entry.setdefault("paths", {})
    paths[path] = int(paths.get(path, 0)) + 1
    if runtime_bincount:
        _AGGREGATE_BINCOUNT_STATS["runtime_bincount_calls"] = int(
            _AGGREGATE_BINCOUNT_STATS.get("runtime_bincount_calls", 0)
        ) + 1
        entry["runtime_bincount_calls"] = int(entry.get("runtime_bincount_calls", 0)) + 1
    if used_precomputed_bin_count:
        _AGGREGATE_BINCOUNT_STATS["precomputed_bincount_calls"] = int(
            _AGGREGATE_BINCOUNT_STATS.get("precomputed_bincount_calls", 0)
        ) + 1
        entry["precomputed_bincount_calls"] = int(entry.get("precomputed_bincount_calls", 0)) + 1
    entry["last_data_shape"] = list(data.shape)
    entry["last_segment_shape"] = list(segment.shape)
    entry["last_bincount_shape"] = list(bin_count.shape) if bin_count is not None else None
    entry["last_num_segment"] = int(num_segment) if num_segment is not None else None
    path_obj = Path(stats_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(_AGGREGATE_BINCOUNT_STATS, indent=2), encoding="utf-8")


_LINE_REFINE_SMOOTH_AGG_STATS = {"enabled_calls": 0, "fallback_calls": 0}


def _record_line_refine_smooth_agg_stat(key: str, reason: str | None = None) -> None:
    _LINE_REFINE_SMOOTH_AGG_STATS[key] = _LINE_REFINE_SMOOTH_AGG_STATS.get(key, 0) + 1
    if reason:
        reason_key = f"reason_{reason}"
        _LINE_REFINE_SMOOTH_AGG_STATS[reason_key] = _LINE_REFINE_SMOOTH_AGG_STATS.get(reason_key, 0) + 1
    stats_path = os.environ.get("MATRIS_LINE_REFINE_SMOOTH_AGG_STATS_PATH", "").strip()
    if not stats_path:
        return
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_LINE_REFINE_SMOOTH_AGG_STATS, indent=2), encoding="utf-8")


class _CudaLineRefineSmoothAgg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, smooth: Tensor, target: Tensor, bin_count: Tensor, num_segment: int) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_refine_smooth_agg_forward"):
            raise RuntimeError("matris_op.line_refine_smooth_agg_forward is unavailable")
        out = matris_op.line_refine_smooth_agg_forward(
            x.contiguous(),
            smooth.contiguous(),
            target.contiguous(),
            bin_count.contiguous(),
            int(num_segment),
        )
        ctx.save_for_backward(x, smooth, target)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, smooth, target = ctx.saved_tensors
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "line_refine_smooth_agg_backward"):
            raise RuntimeError("matris_op.line_refine_smooth_agg_backward is unavailable")
        grad_x, grad_smooth = matris_op.line_refine_smooth_agg_backward(
            grad_out.contiguous(),
            x.contiguous(),
            smooth.contiguous(),
            target.contiguous(),
        )
        return grad_x, grad_smooth, None, None, None


def line_refine_smooth_agg_or_none(
    x: Tensor,
    smooth: Tensor,
    target: Tensor,
    bin_count: Tensor | None,
    num_segment,
    *,
    enable_hint: bool,
) -> Tensor | None:
    if os.environ.get("MATRIS_USE_LINE_REFINE_SMOOTH_AGG_FUSION", "0") != "1" or not enable_hint:
        _record_line_refine_smooth_agg_stat("fallback_calls", "disabled")
        return None
    if x.ndim != 2 or smooth.ndim != 2 or x.shape != smooth.shape:
        _record_line_refine_smooth_agg_stat("fallback_calls", "shape")
        return None
    if x.shape[1] != 128:
        _record_line_refine_smooth_agg_stat("fallback_calls", "hidden")
        return None
    if x.dtype != torch.float32 or smooth.dtype != torch.float32:
        _record_line_refine_smooth_agg_stat("fallback_calls", "dtype")
        return None
    if target.ndim != 1 or target.numel() != x.shape[0] or target.dtype != torch.int64:
        _record_line_refine_smooth_agg_stat("fallback_calls", "target")
        return None
    if bin_count is None or bin_count.ndim != 1 or bin_count.dtype != torch.int64:
        _record_line_refine_smooth_agg_stat("fallback_calls", "bin_count")
        return None
    if not x.is_cuda or not smooth.is_cuda or not target.is_cuda:
        _record_line_refine_smooth_agg_stat("fallback_calls", "not_cuda")
        return None
    if not bin_count.is_cuda:
        _record_line_refine_smooth_agg_stat("fallback_calls", "not_cuda")
        return None
    try:
        resolved_num_segment = int(num_segment)
    except Exception:
        _record_line_refine_smooth_agg_stat("fallback_calls", "num_segment")
        return None
    if resolved_num_segment < 0:
        _record_line_refine_smooth_agg_stat("fallback_calls", "num_segment")
        return None
    if int(bin_count.shape[0]) != resolved_num_segment:
        _record_line_refine_smooth_agg_stat("fallback_calls", "bin_count")
        return None
    if target.numel() > 1 and not bool(torch.all(target[1:] >= target[:-1]).item()):
        _record_line_refine_smooth_agg_stat("fallback_calls", "unsorted")
        return None
    matris_op = _load_matris_op()
    if (
        matris_op is None
        or not hasattr(matris_op, "line_refine_smooth_agg_forward")
        or not hasattr(matris_op, "line_refine_smooth_agg_backward")
    ):
        _record_line_refine_smooth_agg_stat("fallback_calls", "missing_op")
        return None
    try:
        _record_line_refine_smooth_agg_stat("enabled_calls")
        return _CudaLineRefineSmoothAgg.apply(x, smooth, target, bin_count, resolved_num_segment)
    except RuntimeError:
        _record_line_refine_smooth_agg_stat("fallback_calls", "runtime")
        return None


_GRAPH_FEATURE_CONSTRUCTION_STATS: dict[str, int] = {}


def _record_graph_feature_construction_stat(key: str, reason: str | None = None) -> None:
    _GRAPH_FEATURE_CONSTRUCTION_STATS[key] = _GRAPH_FEATURE_CONSTRUCTION_STATS.get(key, 0) + 1
    if reason:
        reason_key = f"reason_{reason}"
        _GRAPH_FEATURE_CONSTRUCTION_STATS[reason_key] = (
            _GRAPH_FEATURE_CONSTRUCTION_STATS.get(reason_key, 0) + 1
        )
    stats_path = os.environ.get("MATRIS_GRAPH_FEATURE_CONSTRUCTION_STATS_PATH", "").strip()
    if not stats_path:
        return
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_GRAPH_FEATURE_CONSTRUCTION_STATS, indent=2), encoding="utf-8")


class _CudaGraphFeatureConstruction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_feat: Tensor, node_feat: Tensor, target: Tensor, source: Tensor) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "graph_feature_construction_forward"):
            raise RuntimeError("matris_op.graph_feature_construction_forward is unavailable")
        out = matris_op.graph_feature_construction_forward(
            edge_feat.contiguous(),
            node_feat.contiguous(),
            target.contiguous(),
            source.contiguous(),
        )
        ctx.save_for_backward(target, source)
        ctx.num_nodes = int(node_feat.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        target, source = ctx.saved_tensors
        grad_out_contig = grad_out.contiguous()
        grad_edge = grad_out_contig[:, :128].contiguous()
        grad_node = grad_out_contig.new_zeros((int(ctx.num_nodes), 128))
        grad_node.index_add_(0, target, grad_out_contig[:, 128:256].contiguous())
        grad_node.index_add_(0, source, grad_out_contig[:, 256:384].contiguous())
        return grad_edge, grad_node, None, None


def graph_feature_construction_or_none(
    edge_feat: Tensor,
    node_feat: Tensor,
    target: Tensor,
    source: Tensor,
    *,
    enable_hint: bool,
) -> Tensor | None:
    if os.environ.get("MATRIS_USE_GRAPH_FEATURE_CONSTRUCTION_FUSION", "0") != "1" or not enable_hint:
        _record_graph_feature_construction_stat("fallback_calls", "disabled")
        return None
    if edge_feat.ndim != 2 or node_feat.ndim != 2:
        _record_graph_feature_construction_stat("fallback_calls", "shape")
        return None
    if edge_feat.shape[1] != 128 or node_feat.shape[1] != 128:
        _record_graph_feature_construction_stat("fallback_calls", "hidden")
        return None
    if edge_feat.dtype != torch.float32 or node_feat.dtype != torch.float32:
        _record_graph_feature_construction_stat("fallback_calls", "dtype")
        return None
    if target.ndim != 1 or source.ndim != 1 or target.shape[0] != edge_feat.shape[0] or source.shape[0] != edge_feat.shape[0]:
        _record_graph_feature_construction_stat("fallback_calls", "index_shape")
        return None
    if target.dtype != torch.int64 or source.dtype != torch.int64:
        _record_graph_feature_construction_stat("fallback_calls", "index_dtype")
        return None
    if not edge_feat.is_cuda or not node_feat.is_cuda or not target.is_cuda or not source.is_cuda:
        _record_graph_feature_construction_stat("fallback_calls", "not_cuda")
        return None
    if edge_feat.device != node_feat.device or target.device != edge_feat.device or source.device != edge_feat.device:
        _record_graph_feature_construction_stat("fallback_calls", "device")
        return None
    matris_op = _load_matris_op()
    if (
        matris_op is None
        or not hasattr(matris_op, "graph_feature_construction_forward")
        or not hasattr(matris_op, "graph_feature_construction_backward")
    ):
        _record_graph_feature_construction_stat("fallback_calls", "missing_op")
        return None
    try:
        _record_graph_feature_construction_stat("enabled_calls")
        return _CudaGraphFeatureConstruction.apply(edge_feat, node_feat, target, source)
    except RuntimeError:
        _record_graph_feature_construction_stat("fallback_calls", "runtime")
        return None


_LINE_ATTENTION_FEATURE_FIRST_LINEAR_STATS: dict[str, int] = {}


def _record_line_attention_feature_first_linear_stat(key: str, reason: str | None = None) -> None:
    _LINE_ATTENTION_FEATURE_FIRST_LINEAR_STATS[key] = (
        _LINE_ATTENTION_FEATURE_FIRST_LINEAR_STATS.get(key, 0) + 1
    )
    if reason:
        reason_key = f"reason_{reason}"
        _LINE_ATTENTION_FEATURE_FIRST_LINEAR_STATS[reason_key] = (
            _LINE_ATTENTION_FEATURE_FIRST_LINEAR_STATS.get(reason_key, 0) + 1
        )
    stats_path = os.environ.get("MATRIS_LINE_ATTENTION_FEATURE_FIRST_LINEAR_STATS_PATH", "").strip()
    if not stats_path:
        return
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_LINE_ATTENTION_FEATURE_FIRST_LINEAR_STATS, indent=2),
        encoding="utf-8",
    )


def line_attention_fused_feature_first_linear_or_none(
    edge_feat: Tensor,
    node_feat: Tensor,
    target: Tensor,
    source: Tensor,
    gated_mlp: nn.Module,
    *,
    enable_hint: bool,
) -> Tensor | None:
    enabled = (
        os.environ.get("MATRIS_USE_LINE_ATTENTION_FUSED_FEATURE_FIRST_LINEAR", "0") == "1"
        or os.environ.get("MATRIS_USE_LINE_FEATURE_FIRST_PROJECTION_FUSION", "0") == "1"
    )
    if not enabled or not enable_hint:
        _record_line_attention_feature_first_linear_stat("fallback_calls", "disabled")
        return None
    if not isinstance(gated_mlp, FusedInputGatedMLP):
        _record_line_attention_feature_first_linear_stat("fallback_calls", "module")
        return None
    out = gated_mlp.forward_line_attention_feature_first_linear(
        edge_feat,
        node_feat,
        target,
        source,
    )
    if out is None:
        _record_line_attention_feature_first_linear_stat("fallback_calls", "conditions")
        return None
    _record_line_attention_feature_first_linear_stat("enabled_calls")
    return out


_FUSED_ATTENTION_WEIGHT_LINEAR_STATS: dict[str, int] = {}


def _record_fused_attention_weight_linear_stat(key: str, reason: str | None = None) -> None:
    _FUSED_ATTENTION_WEIGHT_LINEAR_STATS[key] = _FUSED_ATTENTION_WEIGHT_LINEAR_STATS.get(key, 0) + 1
    if reason:
        reason_key = f"reason_{reason}"
        _FUSED_ATTENTION_WEIGHT_LINEAR_STATS[reason_key] = (
            _FUSED_ATTENTION_WEIGHT_LINEAR_STATS.get(reason_key, 0) + 1
        )
    stats_path = os.environ.get("MATRIS_FUSED_ATTENTION_WEIGHT_LINEAR_STATS_PATH", "").strip()
    if not stats_path:
        return
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_FUSED_ATTENTION_WEIGHT_LINEAR_STATS, indent=2), encoding="utf-8")


def fused_attention_weight_linear_or_none(
    x: Tensor,
    source_linear: nn.Linear,
    target_linear: nn.Linear,
    *,
    enable_hint: bool,
) -> tuple[Tensor, Tensor] | None:
    if os.environ.get("MATRIS_USE_FUSED_ATTENTION_WEIGHT_LINEAR", "0") != "1" or not enable_hint:
        _record_fused_attention_weight_linear_stat("fallback_calls", "disabled")
        return None
    if not isinstance(source_linear, nn.Linear) or not isinstance(target_linear, nn.Linear):
        _record_fused_attention_weight_linear_stat("fallback_calls", "module")
        return None
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda:
        _record_fused_attention_weight_linear_stat("fallback_calls", "input")
        return None
    if source_linear.in_features != target_linear.in_features:
        _record_fused_attention_weight_linear_stat("fallback_calls", "in_features")
        return None
    if source_linear.out_features != target_linear.out_features:
        _record_fused_attention_weight_linear_stat("fallback_calls", "out_features")
        return None
    if x.shape[-1] != source_linear.in_features:
        _record_fused_attention_weight_linear_stat("fallback_calls", "shape")
        return None
    source_weight = source_linear.weight
    target_weight = target_linear.weight
    if source_weight.dtype != torch.float32 or target_weight.dtype != torch.float32:
        _record_fused_attention_weight_linear_stat("fallback_calls", "weight_dtype")
        return None
    if not source_weight.is_cuda or not target_weight.is_cuda:
        _record_fused_attention_weight_linear_stat("fallback_calls", "weight_device")
        return None
    if source_weight.device != x.device or target_weight.device != x.device:
        _record_fused_attention_weight_linear_stat("fallback_calls", "device")
        return None
    if source_weight.shape != target_weight.shape:
        _record_fused_attention_weight_linear_stat("fallback_calls", "weight_shape")
        return None

    source_bias = source_linear.bias
    target_bias = target_linear.bias
    fused_bias = None
    if source_bias is not None or target_bias is not None:
        if source_bias is not None and (source_bias.dtype != torch.float32 or not source_bias.is_cuda):
            _record_fused_attention_weight_linear_stat("fallback_calls", "bias")
            return None
        if target_bias is not None and (target_bias.dtype != torch.float32 or not target_bias.is_cuda):
            _record_fused_attention_weight_linear_stat("fallback_calls", "bias")
            return None
        if source_bias is not None and source_bias.device != x.device:
            _record_fused_attention_weight_linear_stat("fallback_calls", "bias_device")
            return None
        if target_bias is not None and target_bias.device != x.device:
            _record_fused_attention_weight_linear_stat("fallback_calls", "bias_device")
            return None
        if source_bias is None:
            source_bias = torch.zeros(source_linear.out_features, device=x.device, dtype=x.dtype)
        if target_bias is None:
            target_bias = torch.zeros(target_linear.out_features, device=x.device, dtype=x.dtype)
        fused_bias = torch.cat((source_bias, target_bias), dim=0)

    try:
        fused_weight = torch.cat((source_weight, target_weight), dim=0)
        projected = F.linear(x, fused_weight, fused_bias)
        source_alpha, target_alpha = projected.split(source_linear.out_features, dim=-1)
    except RuntimeError:
        _record_fused_attention_weight_linear_stat("fallback_calls", "runtime")
        return None
    _record_fused_attention_weight_linear_stat("enabled_calls")
    return source_alpha, target_alpha


def _linear_cache_key(source_linear: nn.Linear, target_linear: nn.Linear) -> tuple[int, int, int, int, int, int]:
    source_bias = source_linear.bias
    target_bias = target_linear.bias
    return (
        int(source_linear.weight.data_ptr()),
        int(target_linear.weight.data_ptr()),
        int(getattr(source_linear.weight, "_version", 0)),
        int(getattr(target_linear.weight, "_version", 0)),
        int(source_bias.data_ptr()) if source_bias is not None else 0,
        int(target_bias.data_ptr()) if target_bias is not None else 0,
    )


def cached_fused_attention_weight_linear_or_none(
    x: Tensor,
    source_linear: nn.Linear,
    target_linear: nn.Linear,
    *,
    enable_hint: bool,
) -> tuple[Tensor, Tensor] | None:
    if os.environ.get("MATRIS_USE_CACHED_FUSED_ATTENTION_WEIGHT_LINEAR", "0") != "1" or not enable_hint:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "disabled")
        return None
    if os.environ.get("MATRIS_FREEZE_MODEL_PARAMS_FOR_EFS", "0") != "1":
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "not_freeze")
        return None
    if not isinstance(source_linear, nn.Linear) or not isinstance(target_linear, nn.Linear):
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "module")
        return None
    if source_linear.weight.requires_grad or target_linear.weight.requires_grad:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "requires_grad")
        return None
    if source_linear.bias is not None and source_linear.bias.requires_grad:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "requires_grad")
        return None
    if target_linear.bias is not None and target_linear.bias.requires_grad:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "requires_grad")
        return None
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "input")
        return None
    if source_linear.in_features != target_linear.in_features or source_linear.out_features != target_linear.out_features:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "features")
        return None
    if x.shape[-1] != source_linear.in_features:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "shape")
        return None
    source_weight = source_linear.weight
    target_weight = target_linear.weight
    if source_weight.dtype != torch.float32 or target_weight.dtype != torch.float32:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "weight_dtype")
        return None
    if not source_weight.is_cuda or not target_weight.is_cuda:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "weight_device")
        return None
    if source_weight.device != x.device or target_weight.device != x.device or source_weight.shape != target_weight.shape:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "weight_shape")
        return None

    source_bias = source_linear.bias
    target_bias = target_linear.bias
    if source_bias is not None and (source_bias.dtype != torch.float32 or source_bias.device != x.device):
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "bias")
        return None
    if target_bias is not None and (target_bias.dtype != torch.float32 or target_bias.device != x.device):
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "bias")
        return None

    cache_owner = source_linear
    cache_key = _linear_cache_key(source_linear, target_linear)
    cached_key = getattr(cache_owner, "_matris_cached_attention_weight_key", None)
    fused_weight = getattr(cache_owner, "_matris_cached_attention_weight", None)
    fused_bias = getattr(cache_owner, "_matris_cached_attention_bias", None)
    if cached_key != cache_key or fused_weight is None:
        with torch.no_grad():
            fused_weight = torch.cat((source_weight.detach(), target_weight.detach()), dim=0).contiguous()
            if source_bias is None and target_bias is None:
                fused_bias = None
            else:
                if source_bias is None:
                    source_bias = torch.zeros(source_linear.out_features, device=x.device, dtype=x.dtype)
                if target_bias is None:
                    target_bias = torch.zeros(target_linear.out_features, device=x.device, dtype=x.dtype)
                fused_bias = torch.cat((source_bias.detach(), target_bias.detach()), dim=0).contiguous()
        cache_owner._matris_cached_attention_weight_key = cache_key
        cache_owner._matris_cached_attention_weight = fused_weight
        cache_owner._matris_cached_attention_bias = fused_bias
        _record_fused_attention_weight_linear_stat("cached_rebuilds")

    try:
        projected = F.linear(x, fused_weight, fused_bias)
        source_alpha, target_alpha = projected.split(source_linear.out_features, dim=-1)
    except RuntimeError:
        _record_fused_attention_weight_linear_stat("cached_fallback_calls", "runtime")
        return None
    _record_fused_attention_weight_linear_stat("cached_enabled_calls")
    return source_alpha, target_alpha


class MLP(nn.Module):
        
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MLP layer.
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Number of hidden units. Can be an integer for a single
                hidden layer, a sequence of integers for multiple hidden layers,
                or None for no hidden layers. Default: (128, 128).
            output_dim: Dimension of output predictions. Default: 128.
            dropout: Dropout rate applied before each linear layer. Default: 0.0.
            activation: Activation function. Supported: "relu", "silu", "tanh", "gelu".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Dropout rate must be in [0.0, 1.0), got {dropout}")
        
        self.use_fp16 = use_fp16
        activation_func = get_activation(activation)

        layers = []
        if hidden_dim in (None, 0):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        elif isinstance(hidden_dim, int):
            # Single hidden layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim, bias=bias),
                activation_func,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim, bias=bias),
            ])
        elif isinstance(hidden_dim, Sequence):
            # Multiple hidden layers
            layers.extend([
                nn.Linear(input_dim, hidden_dim[0], bias=bias),
                activation_func,
            ])
            # Additional hidden layers
            for i in range(len(hidden_dim) - 1):
                layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=bias),
                    activation_func,
                ])
            # Output layer
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim[-1], output_dim, bias=bias)])
        else:
            raise TypeError(
                f"hidden_dim must be an integer, sequence of integers, or None, "
                f"got {type(hidden_dim).__name__}"
            )
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, feas: Tensor) -> Tensor:
        """
            Args:
                feas: Input tensor of shape (features, input_dim)
            Returns:
                Output tensor of shape (features, output_dim)
        """
        if self.use_fp16 and feas.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self.layers(feas)
            out = out.to(torch.float32)
        else:
            out = self.layers(feas) 
        return out


class _W8ALowPrecisionDualCachedSecondFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: Tensor,
        gate: Tensor,
        core_weight: Tensor,
        gate_weight: Tensor,
        core_bias: Tensor,
        gate_bias: Tensor,
        core_has_bias: bool,
        gate_has_bias: bool,
        compute_dtype: torch.dtype,
    ):
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a_lowp_dual_cached_forward"):
            raise RuntimeError("matris_op.quant_linear_w8a_lowp_dual_cached_forward is unavailable")
        core_out, gate_out = matris_op.quant_linear_w8a_lowp_dual_cached_forward(
            core.to(compute_dtype),
            gate.to(compute_dtype),
            core_weight,
            gate_weight,
            core_bias,
            gate_bias,
            core_has_bias,
            gate_has_bias,
        )
        ctx.save_for_backward(core_weight, gate_weight)
        ctx.compute_dtype = compute_dtype
        return core_out, gate_out

    @staticmethod
    def backward(ctx, grad_core_out: Tensor, grad_gate_out: Tensor):
        core_weight, gate_weight = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        grad_core = F.linear(grad_core_out.to(compute_dtype), core_weight.t()).float()
        grad_gate = F.linear(grad_gate_out.to(compute_dtype), gate_weight.t()).float()
        return grad_core, grad_gate, None, None, None, None, None, None, None


class GatedMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm_type: str = "layer",
        bias: bool = True,
        use_fp16: bool = False,
    ) -> None:
        """
        Args:
            input_dim: The input dimension.
            hidden_dim: A list of integers or a single integer representing the number 
                of hidden units in each layer of the MLP. Default: None.
            output_dim: The output dimension.
            dropout: The dropout rate. Default: 0.0.
            activation: The name of the activation function. Must be one of "relu", 
                "silu", "tanh", or "gelu". Default: "silu".
            norm_type: The name of the normalization layer to use. Must be one of 
                "layer", "rms", "batch", "group", or None. Default: "layer".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        self.use_fp16 = use_fp16
        self.activation_func = get_activation(activation)
        self.activation_gate = get_activation("sigmoid")
        self.gate_norm = get_normalization(name=norm_type, dim=output_dim)
        self.core_norm = get_normalization(name=norm_type, dim=output_dim)
        self.mlp_core = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )

    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas (Tensor): shape (feas_num, input_dim)
        Returns:
            output: shape (feas_num, output_dim)
        """
        if self.gate_norm is None:
            core = self.activation_func(self.mlp_core(feas))
            gate = self.activation_gate(self.mlp_gate(feas))
        else:
            core = self.activation_func(self.core_norm(self.mlp_core(feas)))
            gate = self.activation_gate(self.gate_norm(self.mlp_gate(feas)))
        out = core * gate # gate mul
        return out


class _CudaFusedGatedMLPTailFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: Tensor,
        gate: Tensor,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
    ) -> Tensor:
        core_norm = F.layer_norm(
            core,
            core.shape[-1:],
            core_norm_weight,
            core_norm_bias,
            eps,
        )
        gate_norm = F.layer_norm(
            gate,
            gate.shape[-1:],
            gate_norm_weight,
            gate_norm_bias,
            eps,
        )
        ctx.eps = eps
        ctx.save_for_backward(
            core,
            gate,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
        )
        return F.silu(core_norm) * torch.sigmoid(gate_norm)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            core,
            gate,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
        ) = ctx.saved_tensors
        return _CudaFusedGatedMLPTailFn._torch_backward(
            grad_out,
            core,
            gate,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
            ctx.eps,
        )

    @staticmethod
    def _torch_backward(
        grad_out: Tensor,
        core: Tensor,
        gate: Tensor,
        core_norm: Tensor,
        gate_norm: Tensor,
        core_norm_weight: Tensor,
        gate_norm_weight: Tensor,
        eps: float,
    ):
        sigmoid_gate = torch.sigmoid(gate_norm)
        sigmoid_core = torch.sigmoid(core_norm)
        silu_core = core_norm * sigmoid_core
        d_core_norm = grad_out * sigmoid_gate * sigmoid_core * (
            1.0 + core_norm * (1.0 - sigmoid_core)
        )
        d_gate_norm = grad_out * silu_core * sigmoid_gate * (1.0 - sigmoid_gate)

        def layernorm_grad_input(x: Tensor, grad_norm: Tensor, weight: Tensor) -> Tensor:
            mean = x.mean(dim=-1, keepdim=True)
            centered = x - mean
            rstd = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + eps)
            xhat = centered * rstd
            dxhat = grad_norm * weight
            cols = x.shape[-1]
            return (
                rstd
                / cols
                * (
                    cols * dxhat
                    - dxhat.sum(dim=-1, keepdim=True)
                    - xhat * (dxhat * xhat).sum(dim=-1, keepdim=True)
                )
            )

        grad_core = layernorm_grad_input(core, d_core_norm, core_norm_weight)
        grad_gate = layernorm_grad_input(gate, d_gate_norm, gate_norm_weight)
        return grad_core, grad_gate, None, None, None, None, None


class _W8ALowPrecisionSecondTailInputGradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core_input: Tensor,
        gate_input: Tensor,
        core_weight: Tensor,
        gate_weight: Tensor,
        core_bias: Tensor,
        gate_bias: Tensor,
        core_has_bias: bool,
        gate_has_bias: bool,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
    ) -> Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a_lowp_dual_cached_tail_forward"):
            raise RuntimeError("matris_op.quant_linear_w8a_lowp_dual_cached_tail_forward is unavailable")

        compute_dtype = core_weight.dtype
        core_low = core_input.to(compute_dtype)
        gate_low = gate_input.to(compute_dtype)
        use_aux = False
        if use_aux:
            out, core_linear, gate_linear, core_norm, gate_norm = (
                matris_op.quant_linear_w8a_lowp_dual_cached_tail_forward_aux(
                    core_low,
                    gate_low,
                    core_weight,
                    gate_weight,
                    core_bias,
                    gate_bias,
                    core_has_bias,
                    gate_has_bias,
                    core_norm_weight.float(),
                    core_norm_bias.float(),
                    gate_norm_weight.float(),
                    gate_norm_bias.float(),
                    float(eps),
                )
            )
        else:
            out = matris_op.quant_linear_w8a_lowp_dual_cached_tail_forward(
                core_low,
                gate_low,
                core_weight,
                gate_weight,
                core_bias,
                gate_bias,
                core_has_bias,
                gate_has_bias,
                core_norm_weight.float(),
                core_norm_bias.float(),
                gate_norm_weight.float(),
                gate_norm_bias.float(),
                float(eps),
            )
            core_linear = F.linear(core_low, core_weight, core_bias if core_has_bias else None).float()
            gate_linear = F.linear(gate_low, gate_weight, gate_bias if gate_has_bias else None).float()
            core_norm = F.layer_norm(
                core_linear,
                core_linear.shape[-1:],
                core_norm_weight.float(),
                core_norm_bias.float(),
                float(eps),
            )
            gate_norm = F.layer_norm(
                gate_linear,
                gate_linear.shape[-1:],
                gate_norm_weight.float(),
                gate_norm_bias.float(),
                float(eps),
            )
        ctx.eps = float(eps)
        ctx.compute_dtype = compute_dtype
        ctx.save_for_backward(
            core_weight,
            gate_weight,
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_norm_weight.float(),
            gate_norm_weight.float(),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            core_weight,
            gate_weight,
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
        ) = ctx.saved_tensors
        grad_core_linear, grad_gate_linear, *_ = _CudaFusedGatedMLPTailFn._torch_backward(
            grad_out,
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
            float(ctx.eps),
        )

        compute_dtype = ctx.compute_dtype
        grad_core_input = grad_core_linear.to(compute_dtype).matmul(core_weight).float()
        grad_gate_input = grad_gate_linear.to(compute_dtype).matmul(gate_weight).float()
        return (
            grad_core_input,
            grad_gate_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FP32SecondTailInputGradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core_input: Tensor,
        gate_input: Tensor,
        core_weight: Tensor,
        gate_weight: Tensor,
        core_bias: Tensor,
        gate_bias: Tensor,
        core_has_bias: bool,
        gate_has_bias: bool,
        core_norm_weight: Tensor,
        core_norm_bias: Tensor,
        gate_norm_weight: Tensor,
        gate_norm_bias: Tensor,
        eps: float,
    ) -> Tensor:
        core_linear = F.linear(core_input, core_weight, core_bias if core_has_bias else None)
        gate_linear = F.linear(gate_input, gate_weight, gate_bias if gate_has_bias else None)
        core_norm = F.layer_norm(
            core_linear,
            core_linear.shape[-1:],
            core_norm_weight,
            core_norm_bias,
            float(eps),
        )
        gate_norm = F.layer_norm(
            gate_linear,
            gate_linear.shape[-1:],
            gate_norm_weight,
            gate_norm_bias,
            float(eps),
        )
        ctx.eps = float(eps)
        ctx.save_for_backward(
            core_weight,
            gate_weight,
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
        )
        return F.silu(core_norm) * torch.sigmoid(gate_norm)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            core_weight,
            gate_weight,
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
        ) = ctx.saved_tensors
        grad_core_linear, grad_gate_linear, *_ = _CudaFusedGatedMLPTailFn._torch_backward(
            grad_out,
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_norm_weight,
            gate_norm_weight,
            float(ctx.eps),
        )

        grad_core_input = grad_core_linear.matmul(core_weight)
        grad_gate_input = grad_gate_linear.matmul(gate_weight)
        return (
            grad_core_input,
            grad_gate_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_GATED_TAIL_BWD_STATS = {
    "enabled": 0,
    "fallback_min_rows": 0,
    "fallback_module": 0,
    "enabled_rows": {},
    "fallback_min_rows_rows": {},
    "enabled_by_module": {},
    "fallback_module_by_module": {},
}
_LINE_EDGE_GATED_MLP_BWD_REF_STATS = {
    "enabled_calls": 0,
    "fallback_calls": 0,
    "enabled_by_module": {},
    "fallback_reasons": {},
}


def _record_gated_tail_bwd_stat(
    key: str,
    rows: int | None = None,
    module_name: str | None = None,
) -> None:
    _GATED_TAIL_BWD_STATS[key] = int(_GATED_TAIL_BWD_STATS.get(key, 0)) + 1
    if rows is not None:
        row_key = f"{key}_rows"
        row_hist = _GATED_TAIL_BWD_STATS.setdefault(row_key, {})
        row_hist[str(int(rows))] = int(row_hist.get(str(int(rows)), 0)) + 1
    if module_name:
        module_key = f"{key}_by_module"
        module_hist = _GATED_TAIL_BWD_STATS.setdefault(module_key, {})
        module_hist[module_name] = int(module_hist.get(module_name, 0)) + 1
    stats_path = os.environ.get("MATRIS_FUSED_GATED_TAIL_BWD_STATS_PATH")
    if not stats_path:
        return
    try:
        import json
        from pathlib import Path

        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_GATED_TAIL_BWD_STATS, indent=2), encoding="utf-8")
    except Exception:
        pass


def _record_line_edge_gated_mlp_bwd_ref_stat(
    key: str,
    *,
    module_name: str | None = None,
    reason: str | None = None,
) -> None:
    if os.environ.get("MATRIS_LINE_EDGE_GATED_MLP_BWD_REF_STATS") != "1":
        return
    if key == "enabled":
        _LINE_EDGE_GATED_MLP_BWD_REF_STATS["enabled_calls"] = (
            int(_LINE_EDGE_GATED_MLP_BWD_REF_STATS.get("enabled_calls", 0)) + 1
        )
        if module_name:
            enabled = _LINE_EDGE_GATED_MLP_BWD_REF_STATS.setdefault("enabled_by_module", {})
            enabled[module_name] = int(enabled.get(module_name, 0)) + 1
    else:
        _LINE_EDGE_GATED_MLP_BWD_REF_STATS["fallback_calls"] = (
            int(_LINE_EDGE_GATED_MLP_BWD_REF_STATS.get("fallback_calls", 0)) + 1
        )
        fallback_reasons = _LINE_EDGE_GATED_MLP_BWD_REF_STATS.setdefault("fallback_reasons", {})
        fallback_key = reason or key
        fallback_reasons[fallback_key] = int(fallback_reasons.get(fallback_key, 0)) + 1
    stats_path = os.environ.get("MATRIS_LINE_EDGE_GATED_MLP_BWD_REF_STATS_PATH")
    if not stats_path:
        return
    try:
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_LINE_EDGE_GATED_MLP_BWD_REF_STATS, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass


class FusedGatedMLPTail(nn.Module):
    """GatedMLP tail wrapper for norm, activation, sigmoid gate, and multiply."""

    def __init__(
        self,
        core_norm: nn.Module | None,
        gate_norm: nn.Module | None,
        activation_func: nn.Module,
        activation_gate: nn.Module,
        module_name: str = "",
    ) -> None:
        super().__init__()
        self.core_norm = core_norm
        self.gate_norm = gate_norm
        self.activation_func = activation_func
        self.activation_gate = activation_gate
        self.module_name = module_name

    def forward(self, core: Tensor, gate: Tensor) -> Tensor:
        if (
            not torch.is_grad_enabled()
            and core.is_cuda
            and gate.is_cuda
            and core.ndim == 2
            and gate.shape == core.shape
            and isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and self.core_norm.normalized_shape == self.gate_norm.normalized_shape
            and self.core_norm.eps == self.gate_norm.eps
            and self.core_norm.elementwise_affine
            and self.gate_norm.elementwise_affine
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
        ):
            if os.environ.get("MATRIS_DISABLE_TRITON_GATED_TAIL") != "1":
                from quant.layers import triton_gated_mlp_tail

                out = triton_gated_mlp_tail(
                    core,
                    gate,
                    self.core_norm.weight,
                    self.core_norm.bias,
                    self.gate_norm.weight,
                    self.gate_norm.bias,
                    self.core_norm.eps,
                )
                if out is not None:
                    return out

        if self.core_norm is not None:
            core = self.core_norm(core)
        if self.gate_norm is not None:
            gate = self.gate_norm(gate)
        return self.activation_func(core) * self.activation_gate(gate)


class FusedInputGatedMLP(nn.Module):
    """GatedMLP variant that fuses core/gate projection pairs."""

    def __init__(
        self,
        core_first: nn.Module,
        gate_first: nn.Module,
        fused_first: nn.Linear | None,
        core_tail: nn.Sequential,
        gate_tail: nn.Sequential,
        activation_func: nn.Module | None,
        activation_gate: nn.Module | None,
        core_norm: nn.Module | None,
        gate_norm: nn.Module | None,
        core_second: nn.Module | None = None,
        gate_second: nn.Module | None = None,
        fused_second: nn.Linear | None = None,
        core_second_prefix: nn.Sequential | None = None,
        gate_second_prefix: nn.Sequential | None = None,
        core_post_second_tail: nn.Sequential | None = None,
        gate_post_second_tail: nn.Sequential | None = None,
        fused_tail: FusedGatedMLPTail | None = None,
        use_fp16: bool = False,
        module_name: str = "",
    ) -> None:
        super().__init__()
        self.core_first = core_first
        self.gate_first = gate_first
        self.fused_first = fused_first
        self.core_tail = core_tail
        self.gate_tail = gate_tail
        self.core_second = core_second
        self.gate_second = gate_second
        self.fused_second = fused_second
        self.core_second_prefix = core_second_prefix
        self.gate_second_prefix = gate_second_prefix
        self.core_post_second_tail = core_post_second_tail
        self.gate_post_second_tail = gate_post_second_tail
        self.fused_tail = fused_tail
        self.activation_func = activation_func
        self.activation_gate = activation_gate
        self.core_norm = core_norm
        self.gate_norm = gate_norm
        self.use_fp16 = use_fp16
        self.module_name = module_name
        self.core_hidden_dim = self._linear_like_out_features(core_first)
        self.gate_hidden_dim = self._linear_like_out_features(gate_first)
        self.core_output_dim = self._linear_like_out_features(core_second) if core_second is not None else None
        self.gate_output_dim = self._linear_like_out_features(gate_second) if gate_second is not None else None

    @staticmethod
    def _linear_like_out_features(module: nn.Module) -> int:
        if hasattr(module, "weight"):
            return module.weight.shape[0]
        if hasattr(module, "q_weight"):
            return module.q_weight.shape[0]
        if hasattr(module, "weight_low"):
            return module.weight_low.shape[0]
        raise AttributeError(f"Cannot infer out_features for {type(module).__name__}")

    @staticmethod
    def _is_linear_like(module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) or (
            hasattr(module, "weight") and hasattr(module, "_fake_quant_weight")
        ) or (
            hasattr(module, "q_weight") and hasattr(module, "scale")
        ) or (
            hasattr(module, "weight_low")
        )

    @staticmethod
    def can_fuse(original: nn.Module) -> bool:
        if not all(hasattr(original, name) for name in ("mlp_core", "mlp_gate")):
            return False
        core_layers = getattr(original.mlp_core, "layers", None)
        gate_layers = getattr(original.mlp_gate, "layers", None)
        if not isinstance(core_layers, nn.Sequential) or not isinstance(gate_layers, nn.Sequential):
            return False
        if len(core_layers) < 2 or len(core_layers) != len(gate_layers):
            return False
        return FusedInputGatedMLP._is_linear_like(core_layers[0]) and FusedInputGatedMLP._is_linear_like(gate_layers[0])

    @staticmethod
    def _is_matching_stateless_activation(core_layer: nn.Module, gate_layer: nn.Module) -> bool:
        return type(core_layer) is type(gate_layer) and len(list(core_layer.parameters())) == 0

    @staticmethod
    def _is_disabled_dropout(core_layer: nn.Module, gate_layer: nn.Module) -> bool:
        return (
            isinstance(core_layer, nn.Dropout)
            and isinstance(gate_layer, nn.Dropout)
            and core_layer.p == 0.0
            and gate_layer.p == 0.0
        )

    @staticmethod
    def can_fuse_second_tail(core_tail: nn.Sequential, gate_tail: nn.Sequential) -> bool:
        if len(core_tail) < 3 or len(core_tail) != len(gate_tail):
            return False
        core_layers = list(core_tail.children())
        gate_layers = list(gate_tail.children())
        return (
            FusedInputGatedMLP._is_matching_stateless_activation(core_layers[0], gate_layers[0])
            and FusedInputGatedMLP._is_disabled_dropout(core_layers[1], gate_layers[1])
            and FusedInputGatedMLP._is_linear_like(core_layers[2])
            and FusedInputGatedMLP._is_linear_like(gate_layers[2])
        )

    @classmethod
    def from_gated_mlp(
        cls,
        original: nn.Module,
        module_name: str = "",
        fuse_second: bool = False,
        fuse_tail: bool = False,
    ) -> "FusedInputGatedMLP":
        if not cls.can_fuse(original):
            raise ValueError(f"Unsupported GatedMLP structure for input fusion: {module_name}")

        core_layers = list(original.mlp_core.layers.children())
        gate_layers = list(original.mlp_gate.layers.children())
        fused_first = cls._make_fused_linear(core_layers[0], gate_layers[0])
        core_tail = nn.Sequential(*core_layers[1:])
        gate_tail = nn.Sequential(*gate_layers[1:])

        core_second = None
        gate_second = None
        fused_second = None
        core_second_prefix = None
        gate_second_prefix = None
        core_post_second_tail = None
        gate_post_second_tail = None
        if fuse_second and cls.can_fuse_second_tail(core_tail, gate_tail):
            core_tail_layers = list(core_tail.children())
            gate_tail_layers = list(gate_tail.children())
            core_second = core_tail_layers[2]
            gate_second = gate_tail_layers[2]
            fused_second = cls._make_block_diagonal_fused_linear(core_second, gate_second)
            core_second_prefix = nn.Sequential(*core_tail_layers[:2])
            gate_second_prefix = nn.Sequential(*gate_tail_layers[:2])
            core_post_second_tail = nn.Sequential(*core_tail_layers[3:])
            gate_post_second_tail = nn.Sequential(*gate_tail_layers[3:])
            core_tail = nn.Sequential()
            gate_tail = nn.Sequential()

        fused_tail = None
        activation_func = original.activation_func
        activation_gate = original.activation_gate
        core_norm = original.core_norm
        gate_norm = original.gate_norm
        if fuse_tail:
            fused_tail = FusedGatedMLPTail(
                core_norm=original.core_norm,
                gate_norm=original.gate_norm,
                activation_func=original.activation_func,
                activation_gate=original.activation_gate,
                module_name=module_name,
            )
            activation_func = None
            activation_gate = None
            core_norm = None
            gate_norm = None

        return cls(
            core_first=core_layers[0],
            gate_first=gate_layers[0],
            fused_first=fused_first,
            core_tail=core_tail,
            gate_tail=gate_tail,
            core_second=core_second,
            gate_second=gate_second,
            fused_second=fused_second,
            core_second_prefix=core_second_prefix,
            gate_second_prefix=gate_second_prefix,
            core_post_second_tail=core_post_second_tail,
            gate_post_second_tail=gate_post_second_tail,
            fused_tail=fused_tail,
            activation_func=activation_func,
            activation_gate=activation_gate,
            core_norm=core_norm,
            gate_norm=gate_norm,
            use_fp16=getattr(original, "use_fp16", False),
            module_name=module_name,
        )

    @staticmethod
    def _make_fused_linear(core_first: nn.Module, gate_first: nn.Module) -> nn.Linear | None:
        if not isinstance(core_first, nn.Linear) or not isinstance(gate_first, nn.Linear):
            return None
        if core_first.in_features != gate_first.in_features:
            return None

        bias = core_first.bias is not None or gate_first.bias is not None
        fused = nn.Linear(
            core_first.in_features,
            core_first.out_features + gate_first.out_features,
            bias=bias,
            device=core_first.weight.device,
            dtype=core_first.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat([core_first.weight, gate_first.weight], dim=0))
            if fused.bias is not None:
                core_bias = (
                    core_first.bias
                    if core_first.bias is not None
                    else core_first.weight.new_zeros(core_first.out_features)
                )
                gate_bias = (
                    gate_first.bias
                    if gate_first.bias is not None
                    else gate_first.weight.new_zeros(gate_first.out_features)
                )
                fused.bias.copy_(torch.cat([core_bias, gate_bias], dim=0))
        return fused

    @staticmethod
    def _make_block_diagonal_fused_linear(core_second: nn.Module, gate_second: nn.Module) -> nn.Linear | None:
        if not isinstance(core_second, nn.Linear) or not isinstance(gate_second, nn.Linear):
            return None

        bias = core_second.bias is not None or gate_second.bias is not None
        fused = nn.Linear(
            core_second.in_features + gate_second.in_features,
            core_second.out_features + gate_second.out_features,
            bias=bias,
            device=core_second.weight.device,
            dtype=core_second.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.zero_()
            fused.weight[: core_second.out_features, : core_second.in_features].copy_(core_second.weight)
            fused.weight[
                core_second.out_features :,
                core_second.in_features :,
            ].copy_(gate_second.weight)
            if fused.bias is not None:
                core_bias = (
                    core_second.bias
                    if core_second.bias is not None
                    else core_second.weight.new_zeros(core_second.out_features)
                )
                gate_bias = (
                    gate_second.bias
                    if gate_second.bias is not None
                    else gate_second.weight.new_zeros(gate_second.out_features)
                )
                fused.bias.copy_(torch.cat([core_bias, gate_bias], dim=0))
        return fused

    @staticmethod
    def _weight_bias_stats(module: nn.Module) -> tuple[Tensor, Tensor | None, dict | None]:
        if isinstance(module, nn.Linear):
            return module.weight, module.bias, None

        dq_weight, stats = module._fake_quant_weight()
        bias = module.bias.float() if getattr(module, "bias", None) is not None else None
        return dq_weight, bias, stats

    @staticmethod
    def _set_fake_quant_stats(
        module: nn.Module,
        stats: dict | None,
        x: Tensor,
        out: Tensor,
    ) -> None:
        if stats is None:
            return
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        module.last_stats = stats

    @staticmethod
    def _maybe_fake_quant_activation(
        module: nn.Module,
        x: Tensor,
        stats: dict | None,
    ) -> tuple[Tensor, dict | None]:
        if stats is None or not hasattr(module, "_fake_quant_activation"):
            return x, stats
        dq_x, activation_stats = module._fake_quant_activation(x)
        stats.update(activation_stats)
        return dq_x, stats

    @staticmethod
    def _is_triton_w8a32(module: nn.Module) -> bool:
        return (
            hasattr(module, "q_weight")
            and hasattr(module, "scale")
            and getattr(module, "last_stats", {}).get("backend") == "triton_w8a32"
        )

    @staticmethod
    def _is_cached_low_precision(module: nn.Module) -> bool:
        return hasattr(module, "weight_low") and getattr(module, "last_stats", {}).get(
            "backend"
        ) == "linear_low_precision_cached"

    @staticmethod
    def _is_w8a_low_precision(module: nn.Module) -> bool:
        backend = getattr(module, "last_stats", {}).get("backend")
        return (
            hasattr(module, "q_weight")
            and hasattr(module, "scale")
            and isinstance(backend, str)
            and (backend == "w8abf16_reference" or backend == "w8a16_reference"
                 or backend.startswith("w8abf16_reference_") or backend.startswith("w8a16_reference_"))
        )

    @staticmethod
    def _triton_w8a32_bias(module: nn.Module, rows: int) -> Tensor:
        bias = getattr(module, "bias", None)
        if bias is None:
            return module.scale.new_zeros(rows)
        return bias.float()

    def _triton_w8a32_fused_first(self, x: Tensor) -> tuple[Tensor, Tensor] | None:
        if not (self._is_triton_w8a32(self.core_first) and self._is_triton_w8a32(self.gate_first)):
            return None
        if self.core_first.q_weight.shape[1] != self.gate_first.q_weight.shape[1]:
            return None

        cache_key = "_triton_w8a32_first_cache"
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            q_weight = torch.cat([self.core_first.q_weight, self.gate_first.q_weight], dim=0).contiguous()
            scale = torch.cat([self.core_first.scale, self.gate_first.scale], dim=0).contiguous()
            bias = torch.cat(
                [
                    self._triton_w8a32_bias(self.core_first, self.core_first.q_weight.shape[0]),
                    self._triton_w8a32_bias(self.gate_first, self.gate_first.q_weight.shape[0]),
                ],
                dim=0,
            ).contiguous()
            cache = {
                "device": x.device,
                "q_weight": q_weight,
                "scale": scale,
                "bias": bias,
            }
            setattr(self, cache_key, cache)

        from quant.layers import triton_w8a32_linear

        projected = triton_w8a32_linear(
            x.float(),
            cache["q_weight"],
            cache["scale"],
            cache["bias"],
        )
        core_hidden, gate_hidden = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        self._set_fake_quant_stats(self.core_first, dict(self.core_first.last_stats), x, core_hidden)
        self._set_fake_quant_stats(self.gate_first, dict(self.gate_first.last_stats), x, gate_hidden)
        return core_hidden, gate_hidden

    def _triton_w8a32_fused_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if self.core_second is None or self.gate_second is None:
            return None
        if not (self._is_triton_w8a32(self.core_second) and self._is_triton_w8a32(self.gate_second)):
            return None

        cache_key = "_triton_w8a32_second_cache"
        x = torch.cat([core.float(), gate.float()], dim=-1)
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            core_out, core_in = self.core_second.q_weight.shape
            gate_out, gate_in = self.gate_second.q_weight.shape
            q_weight = self.core_second.q_weight.new_zeros((core_out + gate_out, core_in + gate_in))
            q_weight[:core_out, :core_in] = self.core_second.q_weight
            q_weight[core_out:, core_in:] = self.gate_second.q_weight
            q_weight = q_weight.contiguous()
            scale = torch.cat([self.core_second.scale, self.gate_second.scale], dim=0).contiguous()
            bias = torch.cat(
                [
                    self._triton_w8a32_bias(self.core_second, core_out),
                    self._triton_w8a32_bias(self.gate_second, gate_out),
                ],
                dim=0,
            ).contiguous()
            cache = {
                "device": x.device,
                "q_weight": q_weight,
                "scale": scale,
                "bias": bias,
            }
            setattr(self, cache_key, cache)

        from quant.layers import triton_w8a32_linear

        projected = triton_w8a32_linear(
            x,
            cache["q_weight"],
            cache["scale"],
            cache["bias"],
        )
        core_out, gate_out = projected.split([self.core_output_dim, self.gate_output_dim], dim=-1)
        self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
        self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
        return core_out, gate_out

    @staticmethod
    def _is_triton_w8a8_static(module: nn.Module) -> bool:
        return (
            hasattr(module, "q_weight")
            and hasattr(module, "q_weight_t")
            and hasattr(module, "scale")
            and hasattr(module, "_activation_scale_for")
            and getattr(module, "last_stats", {}).get("backend") == "triton_w8a8_static"
        )

    def _triton_w8a8_static_fused_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if self.core_second is None or self.gate_second is None:
            return None
        if not (
            self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
        ):
            return None

        import os

        from quant.layers import triton_w8a8_static_dual_linear

        core_scale = self.core_second._activation_scale_for(core)
        gate_scale = self.gate_second._activation_scale_for(gate)
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_wmma":
            from quant.layers import cuda_w8a8_static_wmma_dual_linear

            dual_out = cuda_w8a8_static_wmma_dual_linear(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if dual_out is not None:
                core_out, gate_out = dual_out
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_cutlass_dual":
            from quant.layers import cuda_w8a8_static_cutlass_dual_linear

            dual_out = cuda_w8a8_static_cutlass_dual_linear(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if dual_out is not None:
                core_out, gate_out = dual_out
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_cutlass_grouped":
            from quant.layers import cuda_w8a8_static_cutlass_grouped_dual_linear

            dual_out = cuda_w8a8_static_cutlass_grouped_dual_linear(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if dual_out is not None:
                core_out, gate_out = dual_out
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out
        if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_cutlass":
            from quant.layers import cuda_w8a8_static_cutlass_linear

            core_out = cuda_w8a8_static_cutlass_linear(
                core,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
            )
            gate_out = cuda_w8a8_static_cutlass_linear(
                gate,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
            )
            if core_out is not None and gate_out is not None:
                if getattr(self.core_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
                if getattr(self.gate_second, "collect_runtime_stats", False):
                    self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
                return core_out, gate_out

        core_out, gate_out = triton_w8a8_static_dual_linear(
            core,
            gate,
            self.core_second.q_weight,
            self.core_second.q_weight_t,
            self.core_second.scale,
            core_scale,
            core_bias,
            self.gate_second.q_weight,
            self.gate_second.q_weight_t,
            self.gate_second.scale,
            gate_scale,
            gate_bias,
        )
        if getattr(self.core_second, "collect_runtime_stats", False):
            self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
        if getattr(self.gate_second, "collect_runtime_stats", False):
            self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
        return core_out, gate_out

    @staticmethod
    def _is_silu_dropout0_prefix(prefix: nn.Sequential | None) -> bool:
        if prefix is None or len(prefix) != 2:
            return False
        layers = list(prefix.children())
        return isinstance(layers[0], FusedSiLU) and isinstance(layers[1], nn.Dropout) and layers[1].p == 0.0

    def _triton_w8a8_static_fused_silu_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if torch.is_grad_enabled():
            return None
        import os

        if os.environ.get("MATRIS_ENABLE_TRITON_SILU_SECOND") != "1":
            return None
        if self.core_second is None or self.gate_second is None:
            return None
        if not (
            self._is_silu_dropout0_prefix(self.core_second_prefix)
            and self._is_silu_dropout0_prefix(self.gate_second_prefix)
        ):
            return None
        if not (
            self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
        ):
            return None

        from quant.layers import triton_w8a8_static_dual_silu_linear

        core_scale = (
            self.core_second.activation_static_scale.to(device=core.device, dtype=torch.float32).reshape(())
            if getattr(self.core_second, "activation_static_calibrated", False)
            else self.core_second._activation_scale_for(F.silu(core))
        )
        gate_scale = (
            self.gate_second.activation_static_scale.to(device=gate.device, dtype=torch.float32).reshape(())
            if getattr(self.gate_second, "activation_static_calibrated", False)
            else self.gate_second._activation_scale_for(F.silu(gate))
        )
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        return triton_w8a8_static_dual_silu_linear(
            core,
            gate,
            self.core_second.q_weight,
            self.core_second.q_weight_t,
            self.core_second.scale,
            core_scale,
            core_bias,
            self.gate_second.q_weight,
            self.gate_second.q_weight_t,
            self.gate_second.scale,
            gate_scale,
            gate_bias,
        )

    def _cuda_cutlass_w8a8_static_fused_second_tail(self, core: Tensor, gate: Tensor) -> Tensor | None:
        def _record_backend_stat(
            *,
            branch: str,
            input_tensor: Tensor | None,
            weight_tensor: Tensor | None,
            backend_selected: str | None,
            used_cuda: bool,
            fallback_reason: str | None,
            activation_scale: Tensor | None = None,
            weight_scale: Tensor | None = None,
        ) -> None:
            try:
                from quant.layers import record_w8a8_backend_stat

                suffix = "mlp_core.layers.3" if branch == "core" else "mlp_gate.layers.3"
                base_name = self.module_name or "unknown_gated_mlp"
                module_name = f"{base_name}.{suffix}"
                record_w8a8_backend_stat(
                    module_name=module_name,
                    branch=branch,
                    input_shape=tuple(input_tensor.shape) if input_tensor is not None else None,
                    weight_shape=tuple(weight_tensor.shape) if weight_tensor is not None else None,
                    backend_selected=backend_selected,
                    used_cuda_wmma_tail_n128=used_cuda,
                    fallback_reason=fallback_reason,
                    activation_scale=activation_scale,
                    weight_scale=weight_scale,
                )
            except Exception:
                pass

        def _record_pair(
            reason: str,
            *,
            used_cuda: bool = False,
            core_tensor: Tensor | None = None,
            gate_tensor: Tensor | None = None,
            core_scale: Tensor | None = None,
            gate_scale: Tensor | None = None,
        ) -> None:
            backend_selected = os.environ.get("MATRIS_W8A8_BACKEND")
            core_weight = getattr(getattr(self, "core_second", None), "q_weight", None)
            gate_weight = getattr(getattr(self, "gate_second", None), "q_weight", None)
            core_weight_scale = getattr(getattr(self, "core_second", None), "scale", None)
            gate_weight_scale = getattr(getattr(self, "gate_second", None), "scale", None)
            _record_backend_stat(
                branch="core",
                input_tensor=core_tensor,
                weight_tensor=core_weight,
                backend_selected=backend_selected,
                used_cuda=used_cuda,
                fallback_reason=reason,
                activation_scale=core_scale,
                weight_scale=core_weight_scale,
            )
            _record_backend_stat(
                branch="gate",
                input_tensor=gate_tensor,
                weight_tensor=gate_weight,
                backend_selected=backend_selected,
                used_cuda=used_cuda,
                fallback_reason=reason,
                activation_scale=gate_scale,
                weight_scale=gate_weight_scale,
            )

        if self.training:
            _record_pair("training")
            return None

        backend = os.environ.get("MATRIS_W8A8_BACKEND")
        if backend not in ("cuda_cutlass_tail", "cuda_wmma_tail_n128", "cuda_wmma_tail_n128_parallel", "cuda_wmma_tail_n128_auto"):
            _record_pair("backend_not_selected")
            return None
        if self.core_second is None or self.gate_second is None or self.fused_tail is None:
            _record_pair("missing_second_or_tail")
            return None
        if self.core_second_prefix is None or self.gate_second_prefix is None:
            _record_pair("missing_second_prefix")
            return None
        if self.core_post_second_tail is None or self.gate_post_second_tail is None:
            _record_pair("missing_post_second_tail")
            return None
        if not (
            self._is_triton_w8a8_static(self.core_second)
            and self._is_triton_w8a8_static(self.gate_second)
        ):
            _record_pair("second_not_w8a8_static")
            return None
        if len(self.core_post_second_tail) != 0 or len(self.gate_post_second_tail) != 0:
            _record_pair("nonempty_post_second_tail")
            return None
        if (
            os.environ.get("MATRIS_USE_CUDA_W8A8_SECOND_GRAD_INPUT") == "1"
            and os.environ.get("MATRIS_FREEZE_MODEL_PARAMS_FOR_EFS") == "1"
            and torch.is_grad_enabled()
        ):
            _record_pair("second_grad_input_guard")
            return None

        tail = self.fused_tail
        if not (
            isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == tail.gate_norm.normalized_shape
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
        ):
            _record_pair("unsupported_tail_structure")
            return None
        if self.core_second.q_weight.shape[0] != self.gate_second.q_weight.shape[0]:
            _record_pair("core_gate_output_mismatch")
            return None

        core = self.core_second_prefix(core)
        gate = self.gate_second_prefix(gate)
        stats_path = os.environ.get("MATRIS_W8A8_SHAPE_STATS_PATH")
        if stats_path:
            import atexit
            import json
            from pathlib import Path

            stats = getattr(FusedInputGatedMLP, "_w8a8_shape_stats", None)
            if stats is None:
                stats = {}
                setattr(FusedInputGatedMLP, "_w8a8_shape_stats", stats)

            if not getattr(FusedInputGatedMLP, "_w8a8_shape_stats_atexit", False):
                def _dump_shape_stats() -> None:
                    path = Path(stats_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ordered = [
                        {"shape": key, "count": count}
                        for key, count in sorted(stats.items(), key=lambda item: (-item[1], item[0]))
                    ]
                    path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

                atexit.register(_dump_shape_stats)
                setattr(FusedInputGatedMLP, "_w8a8_shape_stats_atexit", True)

            rows = int(core.reshape(-1, core.shape[-1]).shape[0])
            key = (
                f"rows={rows},core_k={int(core.shape[-1])},gate_k={int(gate.shape[-1])},"
                f"core_n={int(self.core_second.q_weight.shape[0])},gate_n={int(self.gate_second.q_weight.shape[0])}"
            )
            stats[key] = stats.get(key, 0) + 1

        core_scale = self.core_second._activation_scale_for(core)
        gate_scale = self.gate_second._activation_scale_for(gate)
        core_bias = self.core_second.bias.float() if self.core_second.bias is not None else None
        gate_bias = self.gate_second.bias.float() if self.gate_second.bias is not None else None
        if backend in ("cuda_wmma_tail_n128", "cuda_wmma_tail_n128_parallel", "cuda_wmma_tail_n128_auto"):
            from quant.layers import (
                cuda_w8a8_static_wmma_dual_gated_tail_n128,
                cuda_w8a8_static_wmma_dual_gated_tail_n128_autograd,
            )

            if (
                os.environ.get("MATRIS_USE_CUDA_W8A8_FUSED_SECOND_TAIL_AUTOGRAD") == "1"
                and torch.is_grad_enabled()
                and backend == "cuda_wmma_tail_n128"
            ):
                autograd_out = cuda_w8a8_static_wmma_dual_gated_tail_n128_autograd(
                    core,
                    gate,
                    self.core_second.q_weight,
                    self.core_second.scale,
                    core_scale,
                    core_bias,
                    self.gate_second.q_weight,
                    self.gate_second.scale,
                    gate_scale,
                    gate_bias,
                    tail.core_norm.weight,
                    tail.core_norm.bias,
                    tail.gate_norm.weight,
                    tail.gate_norm.bias,
                    tail.core_norm.eps,
                )
                if autograd_out is not None:
                    _record_pair(
                        "enabled_autograd",
                        used_cuda=True,
                        core_tensor=core,
                        gate_tensor=gate,
                        core_scale=core_scale,
                        gate_scale=gate_scale,
                    )
                    return autograd_out

            out = cuda_w8a8_static_wmma_dual_gated_tail_n128(
                core,
                gate,
                self.core_second.q_weight,
                self.core_second.scale,
                core_scale,
                core_bias,
                self.gate_second.q_weight,
                self.gate_second.scale,
                gate_scale,
                gate_bias,
                tail.core_norm.weight,
                tail.core_norm.bias,
                tail.gate_norm.weight,
                tail.gate_norm.bias,
                tail.core_norm.eps,
            )
            _record_pair(
                "enabled" if out is not None else "cuda_helper_returned_none",
                used_cuda=out is not None,
                core_tensor=core,
                gate_tensor=gate,
                core_scale=core_scale,
                gate_scale=gate_scale,
            )
            return out

        from quant.layers import cuda_w8a8_static_cutlass_dual_gated_tail

        out = cuda_w8a8_static_cutlass_dual_gated_tail(
            core,
            gate,
            self.core_second.q_weight,
            self.core_second.scale,
            core_scale,
            core_bias,
            self.gate_second.q_weight,
            self.gate_second.scale,
            gate_scale,
            gate_bias,
            tail.core_norm.weight,
            tail.core_norm.bias,
            tail.gate_norm.weight,
            tail.gate_norm.bias,
            tail.core_norm.eps,
        )
        _record_pair(
            "enabled" if out is not None else "cuda_helper_returned_none",
            used_cuda=out is not None,
            core_tensor=core,
            gate_tensor=gate,
            core_scale=core_scale,
            gate_scale=gate_scale,
        )
        return out

    def _w8a_low_precision_fused_second_tail_forward(self, core: Tensor, gate: Tensor) -> Tensor | None:
        return None

    def _fp32_second_tail_input_grad(self, core: Tensor, gate: Tensor) -> Tensor | None:
        legacy_requested = os.environ.get("MATRIS_USE_FP32_SECOND_TAIL_INPUT_GRAD") == "1"
        family_requested = os.environ.get("MATRIS_USE_LINE_EDGE_GATED_MLP_BWD_INPUT_GRAD_FAMILY_REF") == "1"
        if not (legacy_requested or family_requested):
            return None
        if os.environ.get("MATRIS_FREEZE_MODEL_PARAMS_FOR_EFS") != "1":
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat("fallback", reason="freeze_not_enabled")
            return None
        if not (not self.training and torch.is_grad_enabled()):
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat("fallback", reason="training_or_no_grad")
            return None
        if legacy_requested and "attn_block_line_graph.edge_nonlinear_update" not in self.module_name:
            return None
        if family_requested and not (
            "attn_block_line_graph.edge_nonlinear_update" in self.module_name
            or "refine_block_line_graph.edge_nonlinear_update" in self.module_name
        ):
            _record_line_edge_gated_mlp_bwd_ref_stat(
                "fallback",
                module_name=self.module_name,
                reason="module_not_line_edge",
            )
            return None
        if self.core_second is None or self.gate_second is None or self.fused_tail is None:
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat(
                    "fallback",
                    module_name=self.module_name,
                    reason="missing_second_or_tail",
                )
            return None
        if self.core_second_prefix is None or self.gate_second_prefix is None:
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat(
                    "fallback",
                    module_name=self.module_name,
                    reason="missing_second_prefix",
                )
            return None
        if self.core_post_second_tail is None or self.gate_post_second_tail is None:
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat(
                    "fallback",
                    module_name=self.module_name,
                    reason="missing_post_second_tail",
                )
            return None
        if len(self.core_post_second_tail) != 0 or len(self.gate_post_second_tail) != 0:
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat(
                    "fallback",
                    module_name=self.module_name,
                    reason="nonempty_post_second_tail",
                )
            return None
        if not (isinstance(self.core_second, nn.Linear) and isinstance(self.gate_second, nn.Linear)):
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat(
                    "fallback",
                    module_name=self.module_name,
                    reason="second_not_fp32_linear",
                )
            return None
        tail = self.fused_tail
        if not (
            isinstance(tail.core_norm, nn.LayerNorm)
            and isinstance(tail.gate_norm, nn.LayerNorm)
            and tail.core_norm.normalized_shape == tail.gate_norm.normalized_shape
            and tail.core_norm.eps == tail.gate_norm.eps
            and tail.core_norm.elementwise_affine
            and tail.gate_norm.elementwise_affine
            and isinstance(tail.activation_func, FusedSiLU)
            and isinstance(tail.activation_gate, FusedSigmoid)
        ):
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat(
                    "fallback",
                    module_name=self.module_name,
                    reason="unsupported_tail_structure",
                )
            return None

        core_prefix = self.core_second_prefix(core)
        gate_prefix = self.gate_second_prefix(gate)
        if not (
            core_prefix.ndim == 2
            and gate_prefix.ndim == 2
            and core_prefix.is_cuda
            and gate_prefix.is_cuda
            and core_prefix.dtype == torch.float32
            and gate_prefix.dtype == torch.float32
            and core_prefix.shape == gate_prefix.shape
            and self.core_second.weight.dtype == torch.float32
            and self.gate_second.weight.dtype == torch.float32
            and self.core_second.weight.is_cuda
            and self.gate_second.weight.is_cuda
        ):
            if family_requested:
                _record_line_edge_gated_mlp_bwd_ref_stat(
                    "fallback",
                    module_name=self.module_name,
                    reason="shape_dtype_device_guard",
                )
            return None

        empty_bias = self.core_second.weight.new_empty(0)
        core_bias = self.core_second.bias if self.core_second.bias is not None else empty_bias
        gate_bias = self.gate_second.bias if self.gate_second.bias is not None else empty_bias
        if family_requested:
            _record_line_edge_gated_mlp_bwd_ref_stat(
                "enabled",
                module_name=self.module_name,
            )
        return _FP32SecondTailInputGradFn.apply(
            core_prefix,
            gate_prefix,
            self.core_second.weight,
            self.gate_second.weight,
            core_bias,
            gate_bias,
            self.core_second.bias is not None,
            self.gate_second.bias is not None,
            tail.core_norm.weight,
            tail.core_norm.bias,
            tail.gate_norm.weight,
            tail.gate_norm.bias,
            float(tail.core_norm.eps),
        )

    @staticmethod
    def _cached_low_precision_bias(module: nn.Module, rows: int) -> Tensor:
        bias = getattr(module, "bias_low", None)
        if bias is None:
            return module.weight_low.new_zeros(rows)
        return bias

    def _cached_low_precision_fused_first(self, x: Tensor) -> tuple[Tensor, Tensor] | None:
        if not (self._is_cached_low_precision(self.core_first) and self._is_cached_low_precision(self.gate_first)):
            return None
        if self.core_first.weight_low.shape[1] != self.gate_first.weight_low.shape[1]:
            return None

        cache_key = "_cached_low_precision_first_cache"
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            weight = torch.cat([self.core_first.weight_low, self.gate_first.weight_low], dim=0).contiguous()
            bias = torch.cat(
                [
                    self._cached_low_precision_bias(self.core_first, self.core_first.weight_low.shape[0]),
                    self._cached_low_precision_bias(self.gate_first, self.gate_first.weight_low.shape[0]),
                ],
                dim=0,
            ).contiguous()
            cache = {"device": x.device, "weight": weight, "bias": bias, "dtype": weight.dtype}
            setattr(self, cache_key, cache)

        projected = F.linear(x.to(cache["dtype"]), cache["weight"], cache["bias"]).float()
        core_hidden, gate_hidden = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        self._set_fake_quant_stats(self.core_first, dict(self.core_first.last_stats), x, core_hidden)
        self._set_fake_quant_stats(self.gate_first, dict(self.gate_first.last_stats), x, gate_hidden)
        return core_hidden, gate_hidden

    def _cached_low_precision_fused_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        if self.core_second is None or self.gate_second is None:
            return None
        if not (self._is_cached_low_precision(self.core_second) and self._is_cached_low_precision(self.gate_second)):
            return None

        x = torch.cat([core, gate], dim=-1)
        cache_key = "_cached_low_precision_second_cache"
        cache = getattr(self, cache_key, None)
        if cache is None or cache["device"] != x.device:
            core_out, core_in = self.core_second.weight_low.shape
            gate_out, gate_in = self.gate_second.weight_low.shape
            weight = self.core_second.weight_low.new_zeros((core_out + gate_out, core_in + gate_in))
            weight[:core_out, :core_in] = self.core_second.weight_low
            weight[core_out:, core_in:] = self.gate_second.weight_low
            weight = weight.contiguous()
            bias = torch.cat(
                [
                    self._cached_low_precision_bias(self.core_second, core_out),
                    self._cached_low_precision_bias(self.gate_second, gate_out),
                ],
                dim=0,
            ).contiguous()
            cache = {"device": x.device, "weight": weight, "bias": bias, "dtype": weight.dtype}
            setattr(self, cache_key, cache)

        projected = F.linear(x.to(cache["dtype"]), cache["weight"], cache["bias"]).float()
        core_out, gate_out = projected.split([self.core_output_dim, self.gate_output_dim], dim=-1)
        self._set_fake_quant_stats(self.core_second, dict(self.core_second.last_stats), core, core_out)
        self._set_fake_quant_stats(self.gate_second, dict(self.gate_second.last_stats), gate, gate_out)
        return core_out, gate_out

    def _w8a_low_precision_fused_second(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor] | None:
        return None

    def _fused_first_projection(self, feas: Tensor) -> tuple[Tensor, Tensor]:
        if self.fused_first is not None:
            projected = self.fused_first(feas)
            return projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)

        low_precision_projected = self._cached_low_precision_fused_first(feas)
        if low_precision_projected is not None:
            return low_precision_projected

        triton_projected = self._triton_w8a32_fused_first(feas)
        if triton_projected is not None:
            return triton_projected

        core_weight, core_bias, core_stats = self._weight_bias_stats(self.core_first)
        gate_weight, gate_bias, gate_stats = self._weight_bias_stats(self.gate_first)
        weight = torch.cat([core_weight.float(), gate_weight.float()], dim=0)

        bias = None
        if core_bias is not None or gate_bias is not None:
            if core_bias is None:
                core_bias = weight.new_zeros(core_weight.shape[0])
            if gate_bias is None:
                gate_bias = weight.new_zeros(gate_weight.shape[0])
            bias = torch.cat([core_bias.float(), gate_bias.float()], dim=0)

        x = feas.float() if core_stats is not None or gate_stats is not None else feas
        raw_x = x
        if core_stats is not None:
            x, core_stats = self._maybe_fake_quant_activation(self.core_first, raw_x, core_stats)
        if gate_stats is not None and gate_stats is not core_stats:
            _, gate_stats = self._maybe_fake_quant_activation(self.gate_first, raw_x, gate_stats)
        projected = F.linear(x, weight, bias)
        core_hidden, gate_hidden = projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)
        self._set_fake_quant_stats(self.core_first, core_stats, x, core_hidden)
        self._set_fake_quant_stats(self.gate_first, gate_stats, x, gate_hidden)
        return core_hidden, gate_hidden

    def _fused_first_projection_from_line_attention_features(
        self,
        edge_feat: Tensor,
        node_feat: Tensor,
        target: Tensor,
        source: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        if self.fused_first is None or not isinstance(self.fused_first, nn.Linear):
            return None
        if edge_feat.ndim != 2 or node_feat.ndim != 2:
            return None
        if edge_feat.shape[1] != 128 or node_feat.shape[1] != 128:
            return None
        if self.fused_first.in_features != 384:
            return None
        if edge_feat.dtype != torch.float32 or node_feat.dtype != torch.float32:
            return None
        if not edge_feat.is_cuda or not node_feat.is_cuda:
            return None
        if target.ndim != 1 or source.ndim != 1:
            return None
        if target.shape[0] != edge_feat.shape[0] or source.shape[0] != edge_feat.shape[0]:
            return None
        if target.dtype != torch.int64 or source.dtype != torch.int64:
            return None
        if not target.is_cuda or not source.is_cuda:
            return None
        if edge_feat.device != node_feat.device or target.device != edge_feat.device or source.device != edge_feat.device:
            return None

        weight = self.fused_first.weight
        bias = self.fused_first.bias
        if weight.dtype != torch.float32 or not weight.is_cuda or weight.device != edge_feat.device:
            return None
        if bias is not None and (bias.dtype != torch.float32 or not bias.is_cuda or bias.device != edge_feat.device):
            return None
        if weight.shape[1] != 384 or weight.shape[0] != self.core_hidden_dim + self.gate_hidden_dim:
            return None

        target_node_feat = torch.index_select(node_feat, 0, target)
        source_node_feat = torch.index_select(node_feat, 0, source)
        projected = (
            F.linear(edge_feat, weight[:, :128], bias)
            + F.linear(target_node_feat, weight[:, 128:256], None)
            + F.linear(source_node_feat, weight[:, 256:384], None)
        )
        return projected.split([self.core_hidden_dim, self.gate_hidden_dim], dim=-1)

    def _forward_from_first_projection(self, core: Tensor, gate: Tensor) -> Tensor:
        if self.core_second is None:
            core = self.core_tail(core)
            gate = self.gate_tail(gate)
        else:
            fused_second_tail = self._cuda_cutlass_w8a8_static_fused_second_tail(core, gate)
            if fused_second_tail is not None:
                return fused_second_tail
            fused_silu_second = self._triton_w8a8_static_fused_silu_second(core, gate)
            if fused_silu_second is None:
                core = self.core_second_prefix(core)
                gate = self.gate_second_prefix(gate)
                core, gate = self._fused_second_projection(core, gate)
            else:
                core, gate = fused_silu_second
            core = self.core_post_second_tail(core)
            gate = self.gate_post_second_tail(gate)
        if self.fused_tail is not None:
            return self.fused_tail(core, gate)
        if self.core_norm is not None:
            core = self.core_norm(core)
        if self.gate_norm is not None:
            gate = self.gate_norm(gate)
        if self.activation_func is None or self.activation_gate is None:
            raise RuntimeError("GatedMLP tail is missing activation modules.")
        return self.activation_func(core) * self.activation_gate(gate)

    def forward_line_attention_feature_first_linear(
        self,
        edge_feat: Tensor,
        node_feat: Tensor,
        target: Tensor,
        source: Tensor,
    ) -> Tensor | None:
        if self.use_fp16:
            return None
        projected = self._fused_first_projection_from_line_attention_features(
            edge_feat,
            node_feat,
            target,
            source,
        )
        if projected is None:
            return None
        core, gate = projected
        return self._forward_from_first_projection(core, gate)

    def _fused_second_projection(self, core: Tensor, gate: Tensor) -> tuple[Tensor, Tensor]:
        if self.core_second is None or self.gate_second is None:
            raise RuntimeError("Second projection fusion requested without second projection modules.")

        if self.fused_second is not None:
            projected = self.fused_second(torch.cat([core, gate], dim=-1))
            return projected.split([self.core_output_dim, self.gate_output_dim], dim=-1)

        low_precision_projected = self._cached_low_precision_fused_second(core, gate)
        if low_precision_projected is not None:
            return low_precision_projected

        triton_w8a8_projected = self._triton_w8a8_static_fused_second(core, gate)
        if triton_w8a8_projected is not None:
            return triton_w8a8_projected

        triton_projected = self._triton_w8a32_fused_second(core, gate)
        if triton_projected is not None:
            return triton_projected

        core_weight, core_bias, core_stats = self._weight_bias_stats(self.core_second)
        gate_weight, gate_bias, gate_stats = self._weight_bias_stats(self.gate_second)
        core_x = core.float() if core_stats is not None else core
        gate_x = gate.float() if gate_stats is not None else gate
        if core_stats is not None:
            core_x, core_stats = self._maybe_fake_quant_activation(self.core_second, core_x, core_stats)
        if gate_stats is not None and gate_stats is not core_stats:
            gate_x, gate_stats = self._maybe_fake_quant_activation(self.gate_second, gate_x, gate_stats)
        x = torch.cat([core_x, gate_x], dim=-1)
        weight = x.new_zeros(
            core_weight.shape[0] + gate_weight.shape[0],
            core_weight.shape[1] + gate_weight.shape[1],
        )
        weight[: core_weight.shape[0], : core_weight.shape[1]] = core_weight.float()
        weight[core_weight.shape[0] :, core_weight.shape[1] :] = gate_weight.float()

        bias = None
        if core_bias is not None or gate_bias is not None:
            if core_bias is None:
                core_bias = weight.new_zeros(core_weight.shape[0])
            if gate_bias is None:
                gate_bias = weight.new_zeros(gate_weight.shape[0])
            bias = torch.cat([core_bias.float(), gate_bias.float()], dim=0)

        projected = F.linear(x, weight, bias)
        core_out, gate_out = projected.split([core_weight.shape[0], gate_weight.shape[0]], dim=-1)
        self._set_fake_quant_stats(self.core_second, core_stats, core_x, core_out)
        self._set_fake_quant_stats(self.gate_second, gate_stats, gate_x, gate_out)
        return core_out, gate_out

    def _forward_impl(self, feas: Tensor) -> Tensor:
        core, gate = self._fused_first_projection(feas)
        return self._forward_from_first_projection(core, gate)

    def forward(self, feas: Tensor) -> Tensor:
        if self.use_fp16 and feas.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self._forward_impl(feas)
            return out.to(torch.float32)
        return self._forward_impl(feas)


class MOE_Layer(nn.Module):
    
    def __init__(
        self,
        num_expert: int = 64,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MOE layer.

        Args:
            
        """
        super().__init__()
        
        raise NotImplementedError
         
    def forward(self, feas: Tensor) -> Tensor:
        return None


class GraphPooling(nn.Module):
    def __init__(self, average: bool = False) -> None:
        
        super().__init__()
        self.average = average

    def forward(self, node_feat: Tensor, segment: Tensor) -> Tensor:
        """
        Args:
            atom_feat (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, node_feat_dim or 1]
            segment (Tensor): graph indices for each atom.
                [num_batch_atoms]
        
        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, node_feat_dim or 1]
        """
        bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

        output = node_feat.new_zeros([bin_count.shape[0], node_feat.shape[1]])
        output = output.index_add_(0, segment, node_feat)
        if self.average:
            output = (output.T / bin_count).T
        return output


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        change_mat = torch.tensor(
            [
                [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                [
                    -(6 ** (-0.5)),
                    0,
                    0,
                    0,
                    2 * 6 ** (-0.5),
                    0,
                    0,
                    0,
                    -(6 ** (-0.5)),
                ],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
            ],
            device=device,
        ).detach()

    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1

    return total


def reshape_stress(L0out, L2out, batch_size=1):
    _max_rank = 2
    pred_irreps = torch.zeros(
        (batch_size, irreps_sum(_max_rank)),
        device = L0out.device,
    )
    # L=0
    L=0
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L0out.view(batch_size, -1)
    
    L=2
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L2out.view(batch_size, -1) 
    
    pred = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(_max_rank, device = L0out.device),
        pred_irreps,
    )
    
    return pred.view(batch_size, 3,3)


class Sphere(nn.Module):
    
    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.lmax = lmax
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_0_0 = torch.ones_like(x)
        if lmax == 0:
            return torch.stack([ sh_0_0, ], dim=-1)
        
        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)
