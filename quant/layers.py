from __future__ import annotations

import fnmatch
import json
import warnings
import os
import sys
import atexit
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only checks.
    triton = None
    tl = None


_QUANT_LINEAR_KERNEL_WARNED = False
_W8A_LOWP_KERNEL_WARNED = False
_TRITON_W8A32_WARNED = False
_TRITON_W8A8_WARNED = False
_TRITON_W8A_LOWP_WARNED = False
_W8A8_SECOND_GRAD_INPUT_STATS = {
    "enabled_calls": 0,
    "fallback_calls": 0,
    "cuda_forward_calls": 0,
}
_W8A8_BACKEND_STATS: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
_W8A8_BACKEND_STATS_ATEXIT = False


def _parse_w8a8_module_name(module_name: str) -> dict[str, object]:
    parts = module_name.split(".") if module_name else []
    block_id = None
    graph_type = "other"
    block_type = "other"
    update_type = "other"
    branch = "none"
    layer_index = "other"
    for idx, part in enumerate(parts):
        if part == "interaction_block" and idx + 1 < len(parts):
            try:
                block_id = int(parts[idx + 1])
            except ValueError:
                block_id = None
        if part == "attn_block_line_graph":
            graph_type = "line"
            block_type = "attn"
        elif part == "refine_block_line_graph":
            graph_type = "line"
            block_type = "refine"
        elif part == "attn_block_atom_graph":
            graph_type = "atom"
            block_type = "attn"
        elif part == "refine_block_atom_graph":
            graph_type = "atom"
            block_type = "refine"
        if part == "edge_nonlinear_update":
            update_type = "edge"
        elif part == "node_nonlinear_update":
            update_type = "node"
        elif part == "source_weight_linear" or part == "target_weight_linear":
            update_type = "attention_weight"
        if part == "mlp_core":
            branch = "core"
        elif part == "mlp_gate":
            branch = "gate"
        if part == "layers" and idx + 1 < len(parts):
            if parts[idx + 1] == "0":
                layer_index = "first"
            elif parts[idx + 1] == "3":
                layer_index = "second"
    return {
        "block_id": block_id,
        "graph_type": graph_type,
        "block_type": block_type,
        "update_type": update_type,
        "branch": branch,
        "layer_index": layer_index,
    }


def _dump_w8a8_backend_stats() -> None:
    stats_path = os.environ.get("MATRIS_W8A8_BACKEND_STATS_PATH")
    if not stats_path:
        stats_path = "results/w8a8_backend_stats.jsonl"
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for record in _W8A8_BACKEND_STATS.values():
        out = dict(record)
        forward_calls = int(out.get("forward_calls", 0) or 0)
        rows_sum = int(out.pop("_rows_sum", 0) or 0)
        out["avg_rows"] = (rows_sum / forward_calls) if forward_calls else None
        records.append(out)
    records.sort(
        key=lambda item: (
            str(item.get("module_name", "")),
            str(item.get("branch", "")),
            str(item.get("backend_selected", "")),
            str(item.get("fallback_reason", "")),
        )
    )
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def record_w8a8_backend_stat(
    *,
    module_name: str,
    branch: str,
    input_shape: tuple[int, ...] | list[int] | None,
    weight_shape: tuple[int, ...] | list[int] | None,
    backend_selected: str | None,
    used_cuda_wmma_tail_n128: bool,
    fallback_reason: str | None = None,
    activation_scale: torch.Tensor | float | None = None,
    weight_scale: torch.Tensor | None = None,
) -> None:
    if os.environ.get("MATRIS_W8A8_COLLECT_BACKEND_STATS") != "1":
        return
    global _W8A8_BACKEND_STATS_ATEXIT
    if not _W8A8_BACKEND_STATS_ATEXIT:
        atexit.register(_dump_w8a8_backend_stats)
        _W8A8_BACKEND_STATS_ATEXIT = True

    input_shape_tuple = tuple(int(x) for x in input_shape) if input_shape is not None else ()
    weight_shape_tuple = tuple(int(x) for x in weight_shape) if weight_shape is not None else ()
    backend = backend_selected or ""
    reason = fallback_reason or ("enabled" if used_cuda_wmma_tail_n128 else "fallback_unknown")
    meta = _parse_w8a8_module_name(module_name)
    parsed_branch = str(meta.get("branch") or "none")
    actual_branch = branch or parsed_branch
    key = (module_name, actual_branch, backend, reason, str(input_shape_tuple), str(weight_shape_tuple))
    if key not in _W8A8_BACKEND_STATS:
        rows = int(input_shape_tuple[0]) if input_shape_tuple else None
        record: dict[str, object] = {
            "module_name": module_name,
            **meta,
            "branch": actual_branch,
            "input_shape": list(input_shape_tuple),
            "weight_shape": list(weight_shape_tuple),
            "backend_selected": backend,
            "used_cuda_wmma_tail_n128": bool(used_cuda_wmma_tail_n128),
            "fallback_reason": reason,
            "forward_calls": 0,
            "backward_calls": 0,
            "kernel_calls": 0,
            "fallback_calls": 0,
            "min_rows": rows,
            "max_rows": rows,
            "_rows_sum": 0,
            "activation_scale": None,
            "weight_scale_shape": list(weight_scale.shape) if weight_scale is not None else None,
        }
        _W8A8_BACKEND_STATS[key] = record
    record = _W8A8_BACKEND_STATS[key]
    record["forward_calls"] = int(record.get("forward_calls", 0) or 0) + 1
    if used_cuda_wmma_tail_n128:
        record["kernel_calls"] = int(record.get("kernel_calls", 0) or 0) + 1
    else:
        record["fallback_calls"] = int(record.get("fallback_calls", 0) or 0) + 1
    if input_shape_tuple:
        rows = int(input_shape_tuple[0])
        record["_rows_sum"] = int(record.get("_rows_sum", 0) or 0) + rows
        min_rows = record.get("min_rows")
        max_rows = record.get("max_rows")
        record["min_rows"] = rows if min_rows is None else min(int(min_rows), rows)
        record["max_rows"] = rows if max_rows is None else max(int(max_rows), rows)
    if activation_scale is not None and record.get("activation_scale") is None:
        if isinstance(activation_scale, torch.Tensor):
            try:
                record["activation_scale"] = float(activation_scale.detach().float().mean().cpu().item())
            except Exception:
                record["activation_scale"] = "tensor"
        else:
            record["activation_scale"] = float(activation_scale)


def _load_matris_op():
    try:
        import matris_op  # type: ignore

        return matris_op
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        op_src = repo_root / "matris" / "model" / "op" / "src"
        if op_src.exists() and str(op_src) not in sys.path:
            sys.path.insert(0, str(op_src))
        try:
            import matris_op  # type: ignore

            return matris_op
        except Exception:
            return None


def _record_w8a8_second_grad_input_stat(key: str) -> None:
    _W8A8_SECOND_GRAD_INPUT_STATS[key] = _W8A8_SECOND_GRAD_INPUT_STATS.get(key, 0) + 1
    stats_path = os.environ.get("MATRIS_W8A8_SECOND_GRAD_INPUT_STATS_PATH")
    if not stats_path:
        return
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_W8A8_SECOND_GRAD_INPUT_STATS, indent=2), encoding="utf-8")


def _cuda_w8a8_static_grad_input(
    grad_output_2d: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor | None:
    if os.environ.get("MATRIS_USE_CUDA_W8A8_SECOND_GRAD_INPUT") != "1":
        _record_w8a8_second_grad_input_stat("fallback_calls")
        return None
    if os.environ.get("MATRIS_FREEZE_MODEL_PARAMS_FOR_EFS") != "1":
        _record_w8a8_second_grad_input_stat("fallback_calls")
        return None
    if not grad_output_2d.is_cuda or grad_output_2d.dtype != torch.float32:
        _record_w8a8_second_grad_input_stat("fallback_calls")
        return None
    if q_weight.ndim != 2 or weight_scale.ndim != 1:
        _record_w8a8_second_grad_input_stat("fallback_calls")
        return None
    if grad_output_2d.ndim != 2 or grad_output_2d.shape[1] != q_weight.shape[0]:
        _record_w8a8_second_grad_input_stat("fallback_calls")
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_grad_input"):
        _record_w8a8_second_grad_input_stat("fallback_calls")
        return None
    _record_w8a8_second_grad_input_stat("enabled_calls")
    return matris_op.quant_linear_w8a8_static_grad_input(
        grad_output_2d.contiguous(),
        q_weight.contiguous(),
        weight_scale.float().contiguous(),
    )


def _activation_static_scale_multiplier() -> float:
    value = os.environ.get("MATRIS_ACTIVATION_STATIC_SCALE_MULTIPLIER", "1.0")
    try:
        multiplier = float(value)
    except ValueError:
        warnings.warn(
            f"Invalid MATRIS_ACTIVATION_STATIC_SCALE_MULTIPLIER={value!r}; using 1.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0
    return max(multiplier, 1.0e-6)


def _parse_activation_static_rules(env_name: str) -> list[tuple[str, str]]:
    raw = os.environ.get(env_name, "")
    rules: list[tuple[str, str]] = []
    for part in raw.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            warnings.warn(
                f"Ignoring invalid {env_name} rule {item!r}; expected pattern=value.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        pattern, value = item.split("=", 1)
        pattern = pattern.strip()
        value = value.strip()
        if pattern and value:
            rules.append((pattern, value))
    return rules


def _activation_static_rule_value_for(env_name: str, module_name: str) -> str | None:
    matched: str | None = None
    for pattern, value in _parse_activation_static_rules(env_name):
        if fnmatch.fnmatch(module_name, pattern):
            matched = value
    return matched


def _activation_static_scale_multiplier_for(module_name: str) -> float:
    multiplier = _activation_static_scale_multiplier()
    rule_value = _activation_static_rule_value_for(
        "MATRIS_ACTIVATION_STATIC_SCALE_MULTIPLIER_RULES",
        module_name,
    )
    if rule_value is not None:
        try:
            return max(float(rule_value), 1.0e-6)
        except ValueError:
            warnings.warn(
                f"Invalid activation scale multiplier rule value {rule_value!r} for {module_name}; using fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
    globs = [
        part.strip()
        for part in os.environ.get("MATRIS_ACTIVATION_STATIC_SCALE_MULTIPLIER_GLOBS", "").split(",")
        if part.strip()
    ]
    if not globs or not any(fnmatch.fnmatch(module_name, pattern) for pattern in globs):
        return multiplier

    value = os.environ.get("MATRIS_ACTIVATION_STATIC_SCALE_MULTIPLIER_MATCHED", "")
    if not value:
        return multiplier
    try:
        matched_multiplier = float(value)
    except ValueError:
        warnings.warn(
            f"Invalid MATRIS_ACTIVATION_STATIC_SCALE_MULTIPLIER_MATCHED={value!r}; using global multiplier.",
            RuntimeWarning,
            stacklevel=2,
        )
        return multiplier
    return max(matched_multiplier, 1.0e-6)


def _normalize_activation_static_calibration_mode(mode: str) -> str:
    mode = mode.strip().lower()
    aliases = {
        "absmax": "max",
        "max_abs": "max",
        "clip": "percentile",
        "clipping": "percentile",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"max", "percentile"}:
        warnings.warn(
            f"Invalid MATRIS_ACTIVATION_STATIC_CALIBRATION={mode!r}; using max.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "max"
    return mode


def _activation_static_calibration_mode() -> str:
    return _normalize_activation_static_calibration_mode(
        os.environ.get("MATRIS_ACTIVATION_STATIC_CALIBRATION", "max")
    )


def _activation_static_calibration_mode_for(module_name: str) -> str:
    rule_value = _activation_static_rule_value_for(
        "MATRIS_ACTIVATION_STATIC_CALIBRATION_RULES",
        module_name,
    )
    if rule_value is not None:
        return _normalize_activation_static_calibration_mode(rule_value)
    return _activation_static_calibration_mode()


def _parse_activation_static_percentile(value: str) -> float:
    try:
        percentile = float(value)
    except ValueError:
        warnings.warn(
            f"Invalid activation static percentile {value!r}; using 0.999.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.999
    if percentile > 1.0:
        percentile /= 100.0
    return min(max(percentile, 1.0e-6), 1.0)


def _activation_static_percentile() -> float:
    return _parse_activation_static_percentile(
        os.environ.get("MATRIS_ACTIVATION_STATIC_PERCENTILE", "0.999")
    )


def _activation_static_percentile_for(module_name: str) -> float:
    rule_value = _activation_static_rule_value_for(
        "MATRIS_ACTIVATION_STATIC_PERCENTILE_RULES",
        module_name,
    )
    if rule_value is not None:
        return _parse_activation_static_percentile(rule_value)
    return _activation_static_percentile()


def _activation_static_sample_limit() -> int:
    value = os.environ.get("MATRIS_ACTIVATION_STATIC_SAMPLE_LIMIT", "262144")
    try:
        limit = int(value)
    except ValueError:
        warnings.warn(
            f"Invalid MATRIS_ACTIVATION_STATIC_SAMPLE_LIMIT={value!r}; using 262144.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 262144
    return max(limit, 0)


def _sample_activation_abs_values(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    flat = x.detach().float().abs().reshape(-1)
    if max_samples <= 0 or flat.numel() <= max_samples:
        return flat.cpu()
    stride = (flat.numel() + max_samples - 1) // max_samples
    return flat[::stride][:max_samples].cpu()


def _fake_quantize_weight(
    weight: torch.Tensor,
    weight_bits: int,
    scale_granularity: str,
    module_name: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
    qmax = 2 ** (weight_bits - 1) - 1
    weight_fp32 = weight.float()

    if scale_granularity == "per_tensor":
        scale = weight_fp32.abs().max().clamp_min(1e-12) / qmax
    elif scale_granularity == "per_channel":
        scale = weight_fp32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    else:
        raise ValueError(f"Unknown scale_granularity: {scale_granularity}")

    q_weight = torch.round(weight_fp32 / scale).clamp(-qmax, qmax)
    dq_weight = q_weight * scale
    error = (dq_weight - weight_fp32).abs()

    stats = {
        "module_name": module_name,
        "weight_abs_max": weight_fp32.abs().max().detach(),
        "scale_mean": scale.mean().detach(),
        "scale_min": scale.min().detach(),
        "scale_max": scale.max().detach(),
        "weight_quant_error_mae": error.mean().detach(),
        "weight_quant_error_max": error.max().detach(),
        "saturation_ratio": (q_weight.abs() >= qmax).float().mean().detach(),
    }
    return dq_weight, stats


def _fake_quantize_activation_dynamic_per_tensor(
    x: torch.Tensor,
    activation_bits: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    qmax = 2 ** (activation_bits - 1) - 1
    x_fp32 = x.float()
    scale = x_fp32.abs().max().clamp_min(1e-12) / qmax
    q_x = torch.round(x_fp32 / scale).clamp(-qmax, qmax)
    dq_x = q_x * scale
    ste_dq_x = x_fp32 + (dq_x - x_fp32).detach()
    error = (dq_x - x_fp32).abs()
    stats = {
        "activation_abs_max": x_fp32.abs().max().detach(),
        "activation_scale": scale.detach(),
        "activation_quant_error_mae": error.mean().detach(),
        "activation_quant_error_max": error.max().detach(),
        "activation_saturation_ratio": (q_x.abs() >= qmax).float().mean().detach(),
    }
    return ste_dq_x, stats


def _fake_quantize_activation_static_per_tensor(
    x: torch.Tensor,
    activation_bits: int,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    qmax = 2 ** (activation_bits - 1) - 1
    x_fp32 = x.float()
    safe_scale = scale.float().clamp_min(1e-12)
    q_x = torch.round(x_fp32 / safe_scale).clamp(-qmax, qmax)
    dq_x = q_x * safe_scale
    ste_dq_x = x_fp32 + (dq_x - x_fp32).detach()
    error = (dq_x - x_fp32).abs()
    stats = {
        "activation_abs_max": x_fp32.abs().max().detach(),
        "activation_scale": safe_scale.detach(),
        "activation_quant_error_mae": error.mean().detach(),
        "activation_quant_error_max": error.max().detach(),
        "activation_saturation_ratio": (q_x.abs() >= qmax).float().mean().detach(),
    }
    return ste_dq_x, stats


def _fake_quantize_activation_dynamic_m_block(
    x: torch.Tensor,
    activation_bits: int,
    block_m: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    qmax = 2 ** (activation_bits - 1) - 1
    original_shape = x.shape
    x_2d = x.float().reshape(-1, original_shape[-1])
    rows, cols = x_2d.shape
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    if padded_rows != rows:
        padded = x_2d.new_zeros((padded_rows, cols))
        padded[:rows] = x_2d
    else:
        padded = x_2d
    blocks = padded.reshape(-1, block_m, cols)
    scales = blocks.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12) / qmax
    q_blocks = torch.round(blocks / scales).clamp(-qmax, qmax)
    dq_2d = (q_blocks * scales).reshape(padded_rows, cols)[:rows]
    dq_x = dq_2d.reshape(original_shape)
    ste_dq_x = x.float() + (dq_x - x.float()).detach()
    error = (dq_x - x.float()).abs()
    scale_tensor = scales.reshape(-1).detach()
    stats = {
        "activation_abs_max": x.float().abs().max().detach(),
        "activation_scale_mean": scale_tensor.mean().detach() if scale_tensor.numel() else x_2d.new_tensor(0.0),
        "activation_scale_min": scale_tensor.min().detach() if scale_tensor.numel() else x_2d.new_tensor(0.0),
        "activation_scale_max": scale_tensor.max().detach() if scale_tensor.numel() else x_2d.new_tensor(0.0),
        "activation_quant_error_mae": error.mean().detach(),
        "activation_quant_error_max": error.max().detach(),
        "activation_saturation_ratio": (q_blocks.reshape(padded_rows, cols)[:rows].abs() >= qmax).float().mean().detach(),
    }
    return ste_dq_x, stats


class _SelectiveFakeQuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        forward_weight: torch.Tensor,
        backward_weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_fp32 = x.float()
        forward_weight_fp32 = forward_weight.float()
        bias_fp32 = bias.float() if bias is not None else None
        ctx.save_for_backward(backward_weight.float())
        ctx.original_shape = x_fp32.shape
        return F.linear(x_fp32, forward_weight_fp32, bias_fp32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (backward_weight,) = ctx.saved_tensors
        grad_output_2d = grad_output.float().reshape(-1, backward_weight.shape[0])
        grad_x_2d = F.linear(grad_output_2d, backward_weight.t())
        grad_x = grad_x_2d.reshape(ctx.original_shape)
        return grad_x, None, None, None


class _QuantLinearW8A32Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        use_cuda_kernel: bool,
    ) -> torch.Tensor:
        global _QUANT_LINEAR_KERNEL_WARNED

        x_fp32 = x.float()
        q_weight = q_weight.contiguous()
        scale = scale.float().contiguous()
        bias_fp32 = bias.float().contiguous() if bias is not None else None
        original_shape = x_fp32.shape
        x_2d = x_fp32.reshape(-1, original_shape[-1]).contiguous()

        out_2d = None
        if use_cuda_kernel and x_2d.is_cuda:
            matris_op = _load_matris_op()
            if matris_op is not None and hasattr(matris_op, "quant_linear_w8a32"):
                bias_arg = bias_fp32 if bias_fp32 is not None else x_2d.new_empty(0)
                out_2d = matris_op.quant_linear_w8a32(
                    x_2d,
                    q_weight,
                    scale,
                    bias_arg,
                    bias_fp32 is not None,
                )
            elif not _QUANT_LINEAR_KERNEL_WARNED:
                warnings.warn(
                    "matris_op.quant_linear_w8a32 is unavailable; falling back to torch dequantized linear.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _QUANT_LINEAR_KERNEL_WARNED = True

        if out_2d is None:
            dq_weight = q_weight.float() * scale.reshape(-1, 1)
            out_2d = F.linear(x_2d, dq_weight, bias_fp32)

        ctx.save_for_backward(q_weight, scale)
        ctx.original_shape = original_shape
        return out_2d.reshape(*original_shape[:-1], q_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q_weight, scale = ctx.saved_tensors
        grad_output_2d = grad_output.float().reshape(-1, q_weight.shape[0])
        dq_weight = q_weight.float() * scale.reshape(-1, 1)
        grad_x_2d = F.linear(grad_output_2d, dq_weight.t())
        grad_x = grad_x_2d.reshape(ctx.original_shape)
        return grad_x, None, None, None, None


class _W8ALowPrecisionCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
        use_transposed_weight: bool,
    ) -> torch.Tensor:
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a_lowp_forward"):
            raise RuntimeError("matris_op.quant_linear_w8a_lowp_forward is unavailable")

        original_shape = x.shape
        x_2d = x.to(compute_dtype).reshape(-1, original_shape[-1]).contiguous()
        q_weight = q_weight.contiguous()
        scale = scale.float().contiguous()
        bias_fp32 = bias.float().contiguous() if bias is not None else None
        bias_arg = bias_fp32 if bias_fp32 is not None else x_2d.new_empty(0, dtype=torch.float32)
        if use_transposed_weight and use_transposed_weight != "wmma":
            if not hasattr(matris_op, "quant_linear_w8a_lowp_t_forward"):
                raise RuntimeError("matris_op.quant_linear_w8a_lowp_t_forward is unavailable")
            out_2d = matris_op.quant_linear_w8a_lowp_t_forward(
                x_2d,
                q_weight_t.contiguous(),
                scale,
                bias_arg,
                bias_fp32 is not None,
            )
        elif use_transposed_weight in {"cutlass_mixed", "cutlass_mixed_nocontig"}:
            op_name = (
                "quant_linear_w8a_lowp_cutlass_mixed_nocontig_forward"
                if use_transposed_weight == "cutlass_mixed_nocontig"
                else "quant_linear_w8a_lowp_cutlass_mixed_forward"
            )
            if not hasattr(matris_op, op_name):
                raise RuntimeError(f"matris_op.{op_name} is unavailable")
            out_2d = getattr(matris_op, op_name)(
                x_2d,
                q_weight,
                scale,
                bias_arg,
                bias_fp32 is not None,
            )
        elif use_transposed_weight == "wmma":
            if not hasattr(matris_op, "quant_linear_w8a_lowp_wmma_t_forward"):
                raise RuntimeError("matris_op.quant_linear_w8a_lowp_wmma_t_forward is unavailable")
            out_2d = matris_op.quant_linear_w8a_lowp_wmma_t_forward(
                x_2d,
                q_weight_t.contiguous(),
                scale,
                bias_arg,
                bias_fp32 is not None,
            )
        else:
            out_2d = matris_op.quant_linear_w8a_lowp_forward(
                x_2d,
                q_weight,
                scale,
                bias_arg,
                bias_fp32 is not None,
            )
        ctx.save_for_backward(q_weight, scale)
        ctx.original_shape = original_shape
        ctx.compute_dtype = compute_dtype
        return out_2d.reshape(*original_shape[:-1], q_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q_weight, scale = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        grad_output_2d_fp32 = grad_output.float().reshape(-1, q_weight.shape[0])
        if os.environ.get("MATRIS_W8A_LOWP_CUDA_GRAD_INPUT") == "1":
            matris_op = _load_matris_op()
            if (
                matris_op is not None
                and hasattr(matris_op, "quant_linear_w8a_lowp_grad_input")
                and grad_output_2d_fp32.is_cuda
                and q_weight.is_cuda
                and q_weight.shape == (128, 128)
            ):
                grad_x_2d = matris_op.quant_linear_w8a_lowp_grad_input(
                    grad_output_2d_fp32.contiguous(),
                    q_weight,
                    scale.float(),
                )
                grad_x = grad_x_2d.reshape(ctx.original_shape).float()
                return grad_x, None, None, None, None, None, None
        grad_output_2d = grad_output_2d_fp32.to(compute_dtype)
        dq_weight = (q_weight.float() * scale.float().reshape(-1, 1)).to(compute_dtype)
        grad_x_2d = F.linear(grad_output_2d, dq_weight.t())
        grad_x = grad_x_2d.reshape(ctx.original_shape).float()
        return grad_x, None, None, None, None, None, None


class _W8ALowPrecisionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        out_2d = _triton_w8a_lowp_t_forward(
            x_2d,
            q_weight_t,
            scale,
            bias,
            compute_dtype,
        )
        if out_2d is None:
            dq_weight = (q_weight.float() * scale.float().reshape(-1, 1)).to(compute_dtype)
            bias_low = bias.to(compute_dtype) if bias is not None else None
            out_2d = F.linear(x_2d.to(compute_dtype), dq_weight, bias_low).float()

        ctx.save_for_backward(q_weight, scale)
        ctx.original_shape = original_shape
        ctx.compute_dtype = compute_dtype
        return out_2d.reshape(*original_shape[:-1], q_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q_weight, scale = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        grad_output_2d = grad_output.to(compute_dtype).reshape(-1, q_weight.shape[0])
        dq_weight = (q_weight.float() * scale.float().reshape(-1, 1)).to(compute_dtype)
        grad_x_2d = F.linear(grad_output_2d, dq_weight.t())
        grad_x = grad_x_2d.reshape(ctx.original_shape).float()
        return grad_x, None, None, None, None, None


if triton is not None:

    @triton.jit
    def _triton_w8a32_dequant_matmul_kernel(
        x_ptr,
        q_weight_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            q_weight = tl.load(
                q_weight_ptr + offs_n[None, :] * K + k_idxs[:, None],
                mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                other=0,
            ).to(tl.float32)
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
            weight = q_weight * scale[None, :]
            acc += tl.dot(x, weight, input_precision="tf32")

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _triton_w8a8_static_quant_matmul_dequant_kernel(
        x_ptr,
        q_weight_t_ptr,
        weight_scale_ptr,
        activation_scale_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        activation_scale = tl.load(activation_scale_ptr)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            scaled = x / activation_scale
            q_pos = tl.floor(scaled + 0.5)
            q_neg = tl.ceil(scaled - 0.5)
            a = tl.where(scaled >= 0.0, q_pos, q_neg)
            a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
            b = tl.load(
                q_weight_t_ptr + k_idxs[:, None] * N + offs_n[None, :],
                mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            )
            acc += tl.dot(a, b, out_dtype=tl.int32)

        weight_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            out += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _triton_w8a_lowp_t_dequant_matmul_kernel(
        x_ptr,
        q_weight_t_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WEIGHT_DTYPE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            q_weight = tl.load(
                q_weight_t_ptr + k_idxs[:, None] * N + offs_n[None, :],
                mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            ).to(tl.float32)
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
            weight = q_weight * scale[None, :]
            if WEIGHT_DTYPE == 0:
                weight = weight.to(tl.float16)
            else:
                weight = weight.to(tl.bfloat16)
            acc += tl.dot(x, weight, out_dtype=tl.float32)

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _triton_w8a8_static_smooth_quant_matmul_dequant_kernel(
        x_ptr,
        q_weight_t_ptr,
        weight_scale_ptr,
        activation_scale_ptr,
        smooth_scale_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        activation_scale = tl.load(activation_scale_ptr)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            smooth = tl.load(smooth_scale_ptr + k_idxs, mask=k_idxs < K, other=1.0).to(tl.float32)
            scaled = x / (smooth[None, :] * activation_scale)
            q_pos = tl.floor(scaled + 0.5)
            q_neg = tl.ceil(scaled - 0.5)
            a = tl.where(scaled >= 0.0, q_pos, q_neg)
            a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
            b = tl.load(
                q_weight_t_ptr + k_idxs[:, None] * N + offs_n[None, :],
                mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            )
            acc += tl.dot(a, b, out_dtype=tl.int32)

        weight_scale = tl.load(weight_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            out += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _triton_w8a8_static_dual_matmul_dequant_kernel(
        core_ptr,
        gate_ptr,
        core_weight_t_ptr,
        gate_weight_t_ptr,
        core_weight_scale_ptr,
        gate_weight_scale_ptr,
        core_activation_scale_ptr,
        gate_activation_scale_ptr,
        core_bias_ptr,
        gate_bias_ptr,
        core_out_ptr,
        gate_out_ptr,
        M: tl.constexpr,
        K_CORE: tl.constexpr,
        K_GATE: tl.constexpr,
        N_CORE: tl.constexpr,
        N_GATE: tl.constexpr,
        core_has_bias: tl.constexpr,
        gate_has_bias: tl.constexpr,
        MAX_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_branch = tl.program_id(2)
        is_gate = pid_branch == 1

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        k_size = tl.where(is_gate, K_GATE, K_CORE)
        n_size = tl.where(is_gate, N_GATE, N_CORE)
        activation_scale = tl.load(tl.where(is_gate, gate_activation_scale_ptr, core_activation_scale_ptr))

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, MAX_K, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                tl.where(is_gate, gate_ptr, core_ptr) + offs_m[:, None] * k_size + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < k_size),
                other=0.0,
            ).to(tl.float32)
            scaled = x / activation_scale
            q_pos = tl.floor(scaled + 0.5)
            q_neg = tl.ceil(scaled - 0.5)
            a = tl.where(scaled >= 0.0, q_pos, q_neg)
            a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
            b = tl.load(
                tl.where(is_gate, gate_weight_t_ptr, core_weight_t_ptr) + k_idxs[:, None] * n_size + offs_n[None, :],
                mask=(k_idxs[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0,
            )
            acc += tl.dot(a, b, out_dtype=tl.int32)

        weight_scale = tl.load(
            tl.where(is_gate, gate_weight_scale_ptr, core_weight_scale_ptr) + offs_n,
            mask=offs_n < n_size,
            other=0.0,
        )
        out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
        has_bias = tl.where(is_gate, gate_has_bias, core_has_bias)
        if has_bias:
            bias = tl.load(
                tl.where(is_gate, gate_bias_ptr, core_bias_ptr) + offs_n,
                mask=offs_n < n_size,
                other=0.0,
            )
            out += bias[None, :]
        tl.store(
            tl.where(is_gate, gate_out_ptr, core_out_ptr) + offs_m[:, None] * n_size + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < n_size),
        )

    @triton.jit
    def _triton_w8a8_static_dual_silu_matmul_dequant_kernel(
        core_ptr,
        gate_ptr,
        core_weight_t_ptr,
        gate_weight_t_ptr,
        core_weight_scale_ptr,
        gate_weight_scale_ptr,
        core_activation_scale_ptr,
        gate_activation_scale_ptr,
        core_bias_ptr,
        gate_bias_ptr,
        core_out_ptr,
        gate_out_ptr,
        M: tl.constexpr,
        K_CORE: tl.constexpr,
        K_GATE: tl.constexpr,
        N_CORE: tl.constexpr,
        N_GATE: tl.constexpr,
        core_has_bias: tl.constexpr,
        gate_has_bias: tl.constexpr,
        MAX_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_branch = tl.program_id(2)
        is_gate = pid_branch == 1

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        k_size = tl.where(is_gate, K_GATE, K_CORE)
        n_size = tl.where(is_gate, N_GATE, N_CORE)
        activation_scale = tl.load(tl.where(is_gate, gate_activation_scale_ptr, core_activation_scale_ptr))

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, MAX_K, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                tl.where(is_gate, gate_ptr, core_ptr) + offs_m[:, None] * k_size + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < k_size),
                other=0.0,
            ).to(tl.float32)
            x = x * tl.sigmoid(x)
            scaled = x / activation_scale
            q_pos = tl.floor(scaled + 0.5)
            q_neg = tl.ceil(scaled - 0.5)
            a = tl.where(scaled >= 0.0, q_pos, q_neg)
            a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
            b = tl.load(
                tl.where(is_gate, gate_weight_t_ptr, core_weight_t_ptr) + k_idxs[:, None] * n_size + offs_n[None, :],
                mask=(k_idxs[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0,
            )
            acc += tl.dot(a, b, out_dtype=tl.int32)

        weight_scale = tl.load(
            tl.where(is_gate, gate_weight_scale_ptr, core_weight_scale_ptr) + offs_n,
            mask=offs_n < n_size,
            other=0.0,
        )
        out = acc.to(tl.float32) * (activation_scale * weight_scale[None, :])
        has_bias = tl.where(is_gate, gate_has_bias, core_has_bias)
        if has_bias:
            bias = tl.load(
                tl.where(is_gate, gate_bias_ptr, core_bias_ptr) + offs_n,
                mask=offs_n < n_size,
                other=0.0,
            )
            out += bias[None, :]
        tl.store(
            tl.where(is_gate, gate_out_ptr, core_out_ptr) + offs_m[:, None] * n_size + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < n_size),
        )

    @triton.jit
    def _triton_gated_mlp_tail_kernel(
        core_ptr,
        gate_ptr,
        core_norm_weight_ptr,
        core_norm_bias_ptr,
        gate_norm_weight_ptr,
        gate_norm_bias_ptr,
        out_ptr,
        M: tl.constexpr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        core = tl.load(core_ptr + pid_m * D + offs_d, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + pid_m * D + offs_d, mask=mask, other=0.0).to(tl.float32)

        core_sum = tl.sum(tl.where(mask, core, 0.0), axis=0)
        gate_sum = tl.sum(tl.where(mask, gate, 0.0), axis=0)
        core_mean = core_sum / D
        gate_mean = gate_sum / D
        core_centered = tl.where(mask, core - core_mean, 0.0)
        gate_centered = tl.where(mask, gate - gate_mean, 0.0)
        core_var = tl.sum(core_centered * core_centered, axis=0) / D
        gate_var = tl.sum(gate_centered * gate_centered, axis=0) / D

        core_norm = core_centered * tl.rsqrt(core_var + EPS)
        gate_norm = gate_centered * tl.rsqrt(gate_var + EPS)
        core_weight = tl.load(core_norm_weight_ptr + offs_d, mask=mask, other=1.0).to(tl.float32)
        core_bias = tl.load(core_norm_bias_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
        gate_weight = tl.load(gate_norm_weight_ptr + offs_d, mask=mask, other=1.0).to(tl.float32)
        gate_bias = tl.load(gate_norm_bias_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
        core_norm = core_norm * core_weight + core_bias
        gate_norm = gate_norm * gate_weight + gate_bias

        core_act = core_norm * tl.sigmoid(core_norm)
        gate_act = tl.sigmoid(gate_norm)
        out = core_act * gate_act
        tl.store(out_ptr + pid_m * D + offs_d, out, mask=mask)


def _select_triton_w8a32_config(m: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    if k == 256 and m <= 2300:
        return 64, 128, 64, 8, 3
    if k == 256 and m <= 3000:
        return 32, 64, 64, 4, 3
    return 32, 128, 64, 4, 3


def _select_triton_w8a8_config(m: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    if k <= 256 and m <= 3000:
        return 32, 64, 64, 4, 3
    if k <= 256:
        return 32, 128, 64, 4, 3
    return 32, 128, 64, 4, 3


def _select_triton_w8a_lowp_config(m: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    if k == 128 and n == 128:
        return 128, 64, 64, 4, 3
    return 32, 64, 64, 4, 3


def _triton_w8a32_forward(
    x_2d: torch.Tensor,
    q_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if triton is None or not x_2d.is_cuda:
        return None
    m, k = x_2d.shape
    n = q_weight.shape[0]
    block_m, block_n, block_k, num_warps, num_stages = _select_triton_w8a32_config(m, k, n)
    out = torch.empty((m, n), device=x_2d.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias_arg = bias if bias is not None else x_2d.new_empty(0)
    _triton_w8a32_dequant_matmul_kernel[grid](
        x_2d,
        q_weight,
        scale,
        bias_arg,
        out,
        m,
        k,
        n,
        bias is not None,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _triton_w8a_lowp_t_forward(
    x_2d: torch.Tensor,
    q_weight_t: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    compute_dtype: torch.dtype,
) -> torch.Tensor | None:
    if triton is None or not x_2d.is_cuda:
        return None
    if compute_dtype not in (torch.float16, torch.bfloat16):
        return None
    m, k = x_2d.shape
    if q_weight_t.dim() != 2:
        return None
    n = q_weight_t.shape[1]
    if k != 128 or n != 128 or q_weight_t.shape[0] != 128:
        return None
    x_low = x_2d.to(compute_dtype).contiguous()
    q_weight_t = q_weight_t.contiguous()
    scale = scale.float().contiguous()
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    block_m, block_n, block_k, num_warps, num_stages = _select_triton_w8a_lowp_config(m, k, n)
    out = torch.empty((m, n), device=x_2d.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias_arg = bias_fp32 if bias_fp32 is not None else x_2d.new_empty(0, dtype=torch.float32)
    _triton_w8a_lowp_t_dequant_matmul_kernel[grid](
        x_low,
        q_weight_t,
        scale,
        bias_arg,
        out,
        m,
        k,
        n,
        bias_fp32 is not None,
        block_m,
        block_n,
        block_k,
        0 if compute_dtype == torch.float16 else 1,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _triton_w8a8_static_forward(
    x_2d: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if triton is None or not x_2d.is_cuda:
        return None
    m, k = x_2d.shape
    n = q_weight_t.shape[1]
    block_m, block_n, block_k, num_warps, num_stages = _select_triton_w8a8_config(m, k, n)
    out = torch.empty((m, n), device=x_2d.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias_arg = bias if bias is not None else x_2d.new_empty(0)
    _triton_w8a8_static_quant_matmul_dequant_kernel[grid](
        x_2d,
        q_weight_t,
        weight_scale,
        activation_scale,
        bias_arg,
        out,
        m,
        k,
        n,
        bias is not None,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _cuda_w8a8_static_wmma_forward(
    x_2d: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if not x_2d.is_cuda:
        return None
    m, k = x_2d.shape
    n = q_weight.shape[0]
    if k % 16 != 0 or n % 16 != 0:
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma"):
        return None
    bias_arg = bias if bias is not None else x_2d.new_empty(0)
    return matris_op.quant_linear_w8a8_static_wmma(
        x_2d,
        q_weight.contiguous(),
        weight_scale.float().contiguous(),
        activation_scale.float().reshape(()).contiguous(),
        bias_arg,
        bias is not None,
    )


def _cuda_w8a8_static_cutlass_forward(
    x_2d: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if not x_2d.is_cuda:
        return None
    _, k = x_2d.shape
    if k % 32 != 0:
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_cutlass"):
        return None
    bias_arg = bias if bias is not None else x_2d.new_empty(0)
    return matris_op.quant_linear_w8a8_static_cutlass(
        x_2d,
        q_weight.contiguous(),
        weight_scale.float().contiguous(),
        activation_scale.float().reshape(()).contiguous(),
        bias_arg,
        bias is not None,
    )


def _triton_w8a8_static_smooth_forward(
    x_2d: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    smooth_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if triton is None or not x_2d.is_cuda:
        return None
    m, k = x_2d.shape
    n = q_weight_t.shape[1]
    block_m, block_n, block_k, num_warps, num_stages = _select_triton_w8a8_config(m, k, n)
    out = torch.empty((m, n), device=x_2d.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias_arg = bias if bias is not None else x_2d.new_empty(0)
    _triton_w8a8_static_smooth_quant_matmul_dequant_kernel[grid](
        x_2d,
        q_weight_t,
        weight_scale,
        activation_scale,
        smooth_scale,
        bias_arg,
        out,
        m,
        k,
        n,
        bias is not None,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _triton_w8a8_static_dual_forward(
    core_2d: torch.Tensor,
    gate_2d: torch.Tensor,
    core_q_weight_t: torch.Tensor,
    gate_q_weight_t: torch.Tensor,
    core_weight_scale: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or not core_2d.is_cuda:
        return None
    m, k_core = core_2d.shape
    m_gate, k_gate = gate_2d.shape
    if m != m_gate:
        raise ValueError("core and gate inputs must have the same row count")
    n_core = core_q_weight_t.shape[1]
    n_gate = gate_q_weight_t.shape[1]
    block_m, block_n, block_k, num_warps, num_stages = _select_triton_w8a8_config(
        m,
        max(k_core, k_gate),
        max(n_core, n_gate),
    )
    core_out = torch.empty((m, n_core), device=core_2d.device, dtype=torch.float32)
    gate_out = torch.empty((m, n_gate), device=core_2d.device, dtype=torch.float32)
    grid = (
        triton.cdiv(m, block_m),
        max(triton.cdiv(n_core, block_n), triton.cdiv(n_gate, block_n)),
        2,
    )
    core_bias_arg = core_bias if core_bias is not None else core_2d.new_empty(0)
    gate_bias_arg = gate_bias if gate_bias is not None else gate_2d.new_empty(0)
    _triton_w8a8_static_dual_matmul_dequant_kernel[grid](
        core_2d,
        gate_2d,
        core_q_weight_t,
        gate_q_weight_t,
        core_weight_scale,
        gate_weight_scale,
        core_activation_scale,
        gate_activation_scale,
        core_bias_arg,
        gate_bias_arg,
        core_out,
        gate_out,
        m,
        k_core,
        k_gate,
        n_core,
        n_gate,
        core_bias is not None,
        gate_bias is not None,
        max(k_core, k_gate),
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return core_out, gate_out


def _triton_w8a8_static_dual_silu_forward(
    core_2d: torch.Tensor,
    gate_2d: torch.Tensor,
    core_q_weight_t: torch.Tensor,
    gate_q_weight_t: torch.Tensor,
    core_weight_scale: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or not core_2d.is_cuda:
        return None
    m, k_core = core_2d.shape
    m_gate, k_gate = gate_2d.shape
    if m != m_gate:
        raise ValueError("core and gate inputs must have the same row count")
    n_core = core_q_weight_t.shape[1]
    n_gate = gate_q_weight_t.shape[1]
    block_m, block_n, block_k, num_warps, num_stages = _select_triton_w8a8_config(
        m,
        max(k_core, k_gate),
        max(n_core, n_gate),
    )
    core_out = torch.empty((m, n_core), device=core_2d.device, dtype=torch.float32)
    gate_out = torch.empty((m, n_gate), device=gate_2d.device, dtype=torch.float32)
    grid = (
        triton.cdiv(m, block_m),
        max(triton.cdiv(n_core, block_n), triton.cdiv(n_gate, block_n)),
        2,
    )
    core_bias_arg = core_bias if core_bias is not None else core_2d.new_empty(0)
    gate_bias_arg = gate_bias if gate_bias is not None else gate_2d.new_empty(0)
    _triton_w8a8_static_dual_silu_matmul_dequant_kernel[grid](
        core_2d,
        gate_2d,
        core_q_weight_t,
        gate_q_weight_t,
        core_weight_scale,
        gate_weight_scale,
        core_activation_scale,
        gate_activation_scale,
        core_bias_arg,
        gate_bias_arg,
        core_out,
        gate_out,
        m,
        k_core,
        k_gate,
        n_core,
        n_gate,
        core_bias is not None,
        gate_bias is not None,
        max(k_core, k_gate),
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return core_out, gate_out


def triton_gated_mlp_tail(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_norm_weight: torch.Tensor,
    core_norm_bias: torch.Tensor,
    gate_norm_weight: torch.Tensor,
    gate_norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if triton is None or not core.is_cuda or not gate.is_cuda:
        return None
    if core.ndim != 2 or gate.ndim != 2 or core.shape != gate.shape:
        return None

    m, d = core.shape
    if d <= 0 or d > 1024:
        return None

    block_d = 1 << (d - 1).bit_length()
    if block_d > 1024:
        return None

    core_2d = core.float().contiguous()
    gate_2d = gate.float().contiguous()
    out = torch.empty_like(core_2d)
    _triton_gated_mlp_tail_kernel[(m,)](
        core_2d,
        gate_2d,
        core_norm_weight.float().contiguous(),
        core_norm_bias.float().contiguous(),
        gate_norm_weight.float().contiguous(),
        gate_norm_bias.float().contiguous(),
        out,
        m,
        d,
        float(eps),
        block_d,
        num_warps=4,
    )
    return out


def triton_w8a32_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return _TritonW8A32Function.apply(x, q_weight, scale, bias)


def triton_w8a8_static_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    q_weight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return _TritonW8A8StaticFunction.apply(x, q_weight, q_weight_t, weight_scale, activation_scale, bias)


def cuda_w8a8_static_wmma_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    x_fp32 = x.float()
    original_shape = x_fp32.shape
    x_2d = x_fp32.reshape(-1, original_shape[-1]).contiguous()
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    out_2d = _cuda_w8a8_static_wmma_forward(
        x_2d,
        q_weight.contiguous(),
        weight_scale.float().contiguous(),
        activation_scale.float().reshape(()).contiguous(),
        bias_fp32,
    )
    if out_2d is None:
        return None
    return out_2d.reshape(*original_shape[:-1], q_weight.shape[0])


def cuda_w8a8_static_cutlass_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    x_fp32 = x.float()
    original_shape = x_fp32.shape
    x_2d = x_fp32.reshape(-1, original_shape[-1]).contiguous()
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    out_2d = _cuda_w8a8_static_cutlass_forward(
        x_2d,
        q_weight.contiguous(),
        weight_scale.float().contiguous(),
        activation_scale.float().reshape(()).contiguous(),
        bias_fp32,
    )
    if out_2d is None:
        return None
    return out_2d.reshape(*original_shape[:-1], q_weight.shape[0])


def cuda_w8a8_static_wmma_dual_linear(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not core.is_cuda or not gate.is_cuda:
        return None
    core_fp32 = core.float()
    gate_fp32 = gate.float()
    core_shape = core_fp32.shape
    gate_shape = gate_fp32.shape
    core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
    gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
    if core_2d.shape[0] != gate_2d.shape[0]:
        return None
    if core_2d.shape[1] % 16 != 0 or gate_2d.shape[1] % 16 != 0:
        return None
    if core_q_weight.shape[0] % 16 != 0 or gate_q_weight.shape[0] % 16 != 0:
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual"):
        return None
    core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
    gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
    core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
    gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
    out = matris_op.quant_linear_w8a8_static_wmma_dual(
        core_2d,
        gate_2d,
        core_q_weight.contiguous(),
        gate_q_weight.contiguous(),
        core_weight_scale.float().contiguous(),
        gate_weight_scale.float().contiguous(),
        core_activation_scale.float().reshape(()).contiguous(),
        gate_activation_scale.float().reshape(()).contiguous(),
        core_bias_arg,
        gate_bias_arg,
        core_bias_fp32 is not None,
        gate_bias_fp32 is not None,
    )
    core_out, gate_out = out
    return (
        core_out.reshape(*core_shape[:-1], core_q_weight.shape[0]),
        gate_out.reshape(*gate_shape[:-1], gate_q_weight.shape[0]),
    )


def cuda_w8a8_static_cutlass_dual_linear(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not core.is_cuda or not gate.is_cuda:
        return None
    core_fp32 = core.float()
    gate_fp32 = gate.float()
    core_shape = core_fp32.shape
    gate_shape = gate_fp32.shape
    core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
    gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
    if core_2d.shape[0] != gate_2d.shape[0]:
        return None
    if core_2d.shape[1] % 32 != 0 or gate_2d.shape[1] % 32 != 0:
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_cutlass_dual"):
        return None
    core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
    gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
    core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
    gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
    out = matris_op.quant_linear_w8a8_static_cutlass_dual(
        core_2d,
        gate_2d,
        core_q_weight.contiguous(),
        gate_q_weight.contiguous(),
        core_weight_scale.float().contiguous(),
        gate_weight_scale.float().contiguous(),
        core_activation_scale.float().reshape(()).contiguous(),
        gate_activation_scale.float().reshape(()).contiguous(),
        core_bias_arg,
        gate_bias_arg,
        core_bias_fp32 is not None,
        gate_bias_fp32 is not None,
    )
    core_out, gate_out = out
    return (
        core_out.reshape(*core_shape[:-1], core_q_weight.shape[0]),
        gate_out.reshape(*gate_shape[:-1], gate_q_weight.shape[0]),
    )


def cuda_w8a8_static_cutlass_dual_gated_tail(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
    core_norm_weight: torch.Tensor,
    core_norm_bias: torch.Tensor,
    gate_norm_weight: torch.Tensor,
    gate_norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if not core.is_cuda or not gate.is_cuda:
        return None
    core_fp32 = core.float()
    gate_fp32 = gate.float()
    core_shape = core_fp32.shape
    gate_shape = gate_fp32.shape
    core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
    gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
    if core_2d.shape[0] != gate_2d.shape[0]:
        return None
    if core_2d.shape[1] % 32 != 0 or gate_2d.shape[1] % 32 != 0:
        return None
    if core_q_weight.shape[0] != gate_q_weight.shape[0]:
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_cutlass_dual_gated_tail"):
        return None
    core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
    gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
    core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
    gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
    out = matris_op.quant_linear_w8a8_static_cutlass_dual_gated_tail(
        core_2d,
        gate_2d,
        core_q_weight.contiguous(),
        gate_q_weight.contiguous(),
        core_weight_scale.float().contiguous(),
        gate_weight_scale.float().contiguous(),
        core_activation_scale.float().reshape(()).contiguous(),
        gate_activation_scale.float().reshape(()).contiguous(),
        core_bias_arg,
        gate_bias_arg,
        core_bias_fp32 is not None,
        gate_bias_fp32 is not None,
        core_norm_weight.float().contiguous(),
        core_norm_bias.float().contiguous(),
        gate_norm_weight.float().contiguous(),
        gate_norm_bias.float().contiguous(),
        float(eps),
    )
    return out.reshape(*core_shape[:-1], core_q_weight.shape[0])


def cuda_w8a8_static_wmma_dual_gated_tail_n128(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
    core_norm_weight: torch.Tensor,
    core_norm_bias: torch.Tensor,
    gate_norm_weight: torch.Tensor,
    gate_norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if not core.is_cuda or not gate.is_cuda:
        return None
    if core_q_weight.shape[0] != 128 or gate_q_weight.shape[0] != 128:
        return None
    core_fp32 = core.float()
    gate_fp32 = gate.float()
    core_shape = core_fp32.shape
    gate_shape = gate_fp32.shape
    core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
    gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
    if core_2d.shape[0] != gate_2d.shape[0]:
        return None
    if core_2d.shape[1] % 16 != 0 or gate_2d.shape[1] % 16 != 0:
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128"):
        return None
    core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
    gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
    core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
    gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
    out = matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128(
        core_2d,
        gate_2d,
        core_q_weight.contiguous(),
        gate_q_weight.contiguous(),
        core_weight_scale.float().contiguous(),
        gate_weight_scale.float().contiguous(),
        core_activation_scale.float().reshape(()).contiguous(),
        gate_activation_scale.float().reshape(()).contiguous(),
        core_bias_arg,
        gate_bias_arg,
        core_bias_fp32 is not None,
        gate_bias_fp32 is not None,
        core_norm_weight.float().contiguous(),
        core_norm_bias.float().contiguous(),
        gate_norm_weight.float().contiguous(),
        gate_norm_bias.float().contiguous(),
        float(eps),
    )
    return out.reshape(*core_shape[:-1], 128)


def _cuda_input_grad_only_gated_tail_bwd_enabled() -> bool:
    return (
        (
            os.environ.get("MATRIS_USE_CUDA_INPUT_GRAD_ONLY_GATED_TAIL_BWD") == "1"
            or os.environ.get("MATRIS_USE_CUDA_FUSED_GATED_TAIL_BWD") == "1"
        )
        and os.environ.get("MATRIS_FREEZE_MODEL_PARAMS_FOR_EFS") == "1"
    )


class _CudaW8A8SecondTailAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: torch.Tensor,
        gate: torch.Tensor,
        core_q_weight: torch.Tensor,
        core_weight_scale: torch.Tensor,
        core_activation_scale: torch.Tensor,
        core_bias: torch.Tensor | None,
        gate_q_weight: torch.Tensor,
        gate_weight_scale: torch.Tensor,
        gate_activation_scale: torch.Tensor,
        gate_bias: torch.Tensor | None,
        core_norm_weight: torch.Tensor,
        core_norm_bias: torch.Tensor,
        gate_norm_weight: torch.Tensor,
        gate_norm_bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        core_fp32 = core.float()
        gate_fp32 = gate.float()
        core_shape = core_fp32.shape
        gate_shape = gate_fp32.shape
        core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
        gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
        if core_2d.shape[0] != gate_2d.shape[0]:
            raise RuntimeError("core/gate row count mismatch")
        if core_q_weight.shape[0] != 128 or gate_q_weight.shape[0] != 128:
            raise RuntimeError("only N=128 is supported")
        if core_2d.shape[1] % 16 != 0 or gate_2d.shape[1] % 16 != 0:
            raise RuntimeError("K must be multiple of 16")

        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux"):
            raise RuntimeError("matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux is unavailable")

        core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
        gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
        core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
        gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
        out, core_linear, gate_linear, core_norm, gate_norm = matris_op.quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux(
            core_2d,
            gate_2d,
            core_q_weight.contiguous(),
            gate_q_weight.contiguous(),
            core_weight_scale.float().contiguous(),
            gate_weight_scale.float().contiguous(),
            core_activation_scale.float().reshape(()).contiguous(),
            gate_activation_scale.float().reshape(()).contiguous(),
            core_bias_arg,
            gate_bias_arg,
            core_bias_fp32 is not None,
            gate_bias_fp32 is not None,
            core_norm_weight.float().contiguous(),
            core_norm_bias.float().contiguous(),
            gate_norm_weight.float().contiguous(),
            gate_norm_bias.float().contiguous(),
            float(eps),
        )
        ctx.save_for_backward(
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_q_weight.contiguous(),
            core_weight_scale.float().contiguous(),
            gate_q_weight.contiguous(),
            gate_weight_scale.float().contiguous(),
            core_norm_weight.float().contiguous(),
            gate_norm_weight.float().contiguous(),
        )
        ctx.core_shape = core_shape
        ctx.gate_shape = gate_shape
        ctx.eps = float(eps)
        return out.reshape(*core_shape[:-1], 128)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            core_linear,
            gate_linear,
            core_norm,
            gate_norm,
            core_q_weight,
            core_weight_scale,
            gate_q_weight,
            gate_weight_scale,
            core_norm_weight,
            gate_norm_weight,
        ) = ctx.saved_tensors
        matris_op = _load_matris_op()
        if (
            _cuda_input_grad_only_gated_tail_bwd_enabled()
            and matris_op is not None
            and hasattr(matris_op, "fused_gated_mlp_tail_backward")
        ):
            grad_core_linear, grad_gate_linear = matris_op.fused_gated_mlp_tail_backward(
                grad_out.float().reshape(-1, 128).contiguous(),
                core_linear,
                gate_linear,
                core_norm,
                gate_norm,
                core_norm_weight,
                gate_norm_weight,
                float(ctx.eps),
            )
        else:
            sigmoid_gate = torch.sigmoid(gate_norm)
            sigmoid_core = torch.sigmoid(core_norm)
            silu_core = core_norm * sigmoid_core
            d_core_norm = grad_out.float().reshape(-1, 128) * sigmoid_gate * sigmoid_core * (
                1.0 + core_norm * (1.0 - sigmoid_core)
            )
            d_gate_norm = grad_out.float().reshape(-1, 128) * silu_core * sigmoid_gate * (1.0 - sigmoid_gate)

            def layernorm_grad_input(x: torch.Tensor, grad_norm: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
                mean = x.mean(dim=-1, keepdim=True)
                centered = x - mean
                rstd = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + ctx.eps)
                xhat = centered * rstd
                dxhat = grad_norm * weight
                cols = x.shape[-1]
                return rstd / cols * (cols * dxhat - dxhat.sum(dim=-1, keepdim=True) - xhat * (dxhat * xhat).sum(dim=-1, keepdim=True))

            grad_core_linear = layernorm_grad_input(core_linear, d_core_norm, core_norm_weight)
            grad_gate_linear = layernorm_grad_input(gate_linear, d_gate_norm, gate_norm_weight)

        core_dq_weight = core_q_weight.float() * core_weight_scale.reshape(-1, 1)
        gate_dq_weight = gate_q_weight.float() * gate_weight_scale.reshape(-1, 1)
        grad_core = F.linear(grad_core_linear.float(), core_dq_weight.t()).reshape(ctx.core_shape)
        grad_gate = F.linear(grad_gate_linear.float(), gate_dq_weight.t()).reshape(ctx.gate_shape)
        return grad_core, grad_gate, None, None, None, None, None, None, None, None, None, None, None, None, None


def cuda_w8a8_static_wmma_dual_gated_tail_n128_autograd(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
    core_norm_weight: torch.Tensor,
    core_norm_bias: torch.Tensor,
    gate_norm_weight: torch.Tensor,
    gate_norm_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if not core.is_cuda or not gate.is_cuda:
        return None
    if core.ndim != 2 or gate.ndim != 2 or core.shape[0] != gate.shape[0]:
        return None
    if core_q_weight.shape[0] != 128 or gate_q_weight.shape[0] != 128:
        return None
    if core.shape[-1] % 16 != 0 or gate.shape[-1] % 16 != 0:
        return None
    if os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_wmma_tail_n128_parallel":
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128_aux"):
        return None
    return _CudaW8A8SecondTailAutogradFunction.apply(
        core,
        gate,
        core_q_weight,
        core_weight_scale,
        core_activation_scale,
        core_bias,
        gate_q_weight,
        gate_weight_scale,
        gate_activation_scale,
        gate_bias,
        core_norm_weight,
        core_norm_bias,
        gate_norm_weight,
        gate_norm_bias,
        eps,
    )


def cuda_w8a8_static_cutlass_grouped_dual_linear(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not core.is_cuda or not gate.is_cuda:
        return None
    core_fp32 = core.float()
    gate_fp32 = gate.float()
    core_shape = core_fp32.shape
    gate_shape = gate_fp32.shape
    core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
    gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
    if core_2d.shape[0] != gate_2d.shape[0]:
        return None
    if core_2d.shape[1] % 32 != 0 or gate_2d.shape[1] % 32 != 0:
        return None
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_cutlass_grouped_dual"):
        return None
    core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
    gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
    core_bias_arg = core_bias_fp32 if core_bias_fp32 is not None else core_2d.new_empty(0)
    gate_bias_arg = gate_bias_fp32 if gate_bias_fp32 is not None else gate_2d.new_empty(0)
    out = matris_op.quant_linear_w8a8_static_cutlass_grouped_dual(
        core_2d,
        gate_2d,
        core_q_weight.contiguous(),
        gate_q_weight.contiguous(),
        core_weight_scale.float().contiguous(),
        gate_weight_scale.float().contiguous(),
        core_activation_scale.float().reshape(()).contiguous(),
        gate_activation_scale.float().reshape(()).contiguous(),
        core_bias_arg,
        gate_bias_arg,
        core_bias_fp32 is not None,
        gate_bias_fp32 is not None,
    )
    core_out, gate_out = out
    return (
        core_out.reshape(*core_shape[:-1], core_q_weight.shape[0]),
        gate_out.reshape(*gate_shape[:-1], gate_q_weight.shape[0]),
    )


def triton_w8a8_static_dual_linear(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_q_weight_t: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_q_weight_t: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _TritonW8A8StaticDualFunction.apply(
        core,
        gate,
        core_q_weight,
        core_q_weight_t,
        core_weight_scale,
        core_activation_scale,
        core_bias,
        gate_q_weight,
        gate_q_weight_t,
        gate_weight_scale,
        gate_activation_scale,
        gate_bias,
    )


def triton_w8a8_static_dual_silu_linear(
    core: torch.Tensor,
    gate: torch.Tensor,
    core_q_weight: torch.Tensor,
    core_q_weight_t: torch.Tensor,
    core_weight_scale: torch.Tensor,
    core_activation_scale: torch.Tensor,
    core_bias: torch.Tensor | None,
    gate_q_weight: torch.Tensor,
    gate_q_weight_t: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_activation_scale: torch.Tensor,
    gate_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_fp32 = core.float()
    gate_fp32 = gate.float()
    core_shape = core_fp32.shape
    gate_shape = gate_fp32.shape
    core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
    gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
    core_activation_scale = core_activation_scale.float().reshape(()).contiguous()
    gate_activation_scale = gate_activation_scale.float().reshape(()).contiguous()
    core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
    gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None
    out = _triton_w8a8_static_dual_silu_forward(
        core_2d,
        gate_2d,
        core_q_weight_t.contiguous(),
        gate_q_weight_t.contiguous(),
        core_weight_scale.float().contiguous(),
        gate_weight_scale.float().contiguous(),
        core_activation_scale,
        gate_activation_scale,
        core_bias_fp32,
        gate_bias_fp32,
    )
    if out is None:
        core_act = F.silu(core_2d)
        gate_act = F.silu(gate_2d)
        core_q = torch.round(core_act / core_activation_scale).clamp(-127, 127)
        gate_q = torch.round(gate_act / gate_activation_scale).clamp(-127, 127)
        core_dq_weight = core_q_weight.float() * core_weight_scale.float().reshape(-1, 1)
        gate_dq_weight = gate_q_weight.float() * gate_weight_scale.float().reshape(-1, 1)
        core_out = F.linear(core_q * core_activation_scale, core_dq_weight, core_bias_fp32)
        gate_out = F.linear(gate_q * gate_activation_scale, gate_dq_weight, gate_bias_fp32)
    else:
        core_out, gate_out = out
    return (
        core_out.reshape(*core_shape[:-1], core_q_weight.shape[0]),
        gate_out.reshape(*gate_shape[:-1], gate_q_weight.shape[0]),
    )


class _TritonW8A32Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        global _TRITON_W8A32_WARNED

        x_fp32 = x.float()
        q_weight = q_weight.contiguous()
        scale = scale.float().contiguous()
        bias_fp32 = bias.float().contiguous() if bias is not None else None
        original_shape = x_fp32.shape
        x_2d = x_fp32.reshape(-1, original_shape[-1]).contiguous()

        out_2d = _triton_w8a32_forward(x_2d, q_weight, scale, bias_fp32)
        if out_2d is None:
            if not _TRITON_W8A32_WARNED:
                warnings.warn(
                    "Triton W8A32 is unavailable; falling back to torch dequantized linear.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _TRITON_W8A32_WARNED = True
            dq_weight = q_weight.float() * scale.reshape(-1, 1)
            out_2d = F.linear(x_2d, dq_weight, bias_fp32)

        ctx.save_for_backward(q_weight, scale)
        ctx.original_shape = original_shape
        return out_2d.reshape(*original_shape[:-1], q_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q_weight, scale = ctx.saved_tensors
        grad_output_2d = grad_output.float().reshape(-1, q_weight.shape[0])
        dq_weight = q_weight.float() * scale.reshape(-1, 1)
        grad_x_2d = F.linear(grad_output_2d, dq_weight.t())
        grad_x = grad_x_2d.reshape(ctx.original_shape)
        return grad_x, None, None, None


class _TritonW8A8StaticFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        q_weight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        global _TRITON_W8A8_WARNED

        x_fp32 = x.float()
        q_weight = q_weight.contiguous()
        q_weight_t = q_weight_t.contiguous()
        weight_scale = weight_scale.float().contiguous()
        activation_scale = activation_scale.float().reshape(()).contiguous()
        bias_fp32 = bias.float().contiguous() if bias is not None else None
        original_shape = x_fp32.shape
        x_2d = x_fp32.reshape(-1, original_shape[-1]).contiguous()

        out_2d = None
        if (
            os.environ.get("MATRIS_USE_CUDA_W8A8_SECOND_GRAD_INPUT") == "1"
            and os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_wmma_tail_n128"
        ):
            out_2d = _cuda_w8a8_static_wmma_forward(
                x_2d,
                q_weight,
                weight_scale,
                activation_scale,
                bias_fp32,
            )
            if out_2d is not None:
                _record_w8a8_second_grad_input_stat("cuda_forward_calls")
        if out_2d is None:
            out_2d = _triton_w8a8_static_forward(
                x_2d,
                q_weight_t,
                weight_scale,
                activation_scale,
                bias_fp32,
            )
        if out_2d is None:
            if not _TRITON_W8A8_WARNED:
                warnings.warn(
                    "Triton W8A8 static is unavailable; falling back to torch fake-quant linear.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _TRITON_W8A8_WARNED = True
            q_x = torch.round(x_2d / activation_scale).clamp(-127, 127)
            dq_x = q_x * activation_scale
            q_weight = q_weight_t.t().contiguous()
            dq_weight = q_weight.float() * weight_scale.reshape(-1, 1)
            out_2d = F.linear(dq_x, dq_weight, bias_fp32)

        ctx.save_for_backward(q_weight, weight_scale)
        ctx.original_shape = original_shape
        return out_2d.reshape(*original_shape[:-1], q_weight_t.shape[1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q_weight, weight_scale = ctx.saved_tensors
        grad_output_2d = grad_output.float().reshape(-1, q_weight.shape[0])
        grad_x_2d = _cuda_w8a8_static_grad_input(grad_output_2d.contiguous(), q_weight, weight_scale)
        if grad_x_2d is None:
            dq_weight = q_weight.float() * weight_scale.reshape(-1, 1)
            grad_x_2d = F.linear(grad_output_2d, dq_weight.t())
        grad_x = grad_x_2d.reshape(ctx.original_shape)
        return grad_x, None, None, None, None, None


class _TritonW8A8StaticDualFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        core: torch.Tensor,
        gate: torch.Tensor,
        core_q_weight: torch.Tensor,
        core_q_weight_t: torch.Tensor,
        core_weight_scale: torch.Tensor,
        core_activation_scale: torch.Tensor,
        core_bias: torch.Tensor | None,
        gate_q_weight: torch.Tensor,
        gate_q_weight_t: torch.Tensor,
        gate_weight_scale: torch.Tensor,
        gate_activation_scale: torch.Tensor,
        gate_bias: torch.Tensor | None,
    ):
        core_fp32 = core.float()
        gate_fp32 = gate.float()
        core_shape = core_fp32.shape
        gate_shape = gate_fp32.shape
        core_2d = core_fp32.reshape(-1, core_shape[-1]).contiguous()
        gate_2d = gate_fp32.reshape(-1, gate_shape[-1]).contiguous()
        core_activation_scale = core_activation_scale.float().reshape(()).contiguous()
        gate_activation_scale = gate_activation_scale.float().reshape(()).contiguous()
        core_bias_fp32 = core_bias.float().contiguous() if core_bias is not None else None
        gate_bias_fp32 = gate_bias.float().contiguous() if gate_bias is not None else None

        out = None
        if (
            os.environ.get("MATRIS_USE_CUDA_W8A8_SECOND_GRAD_INPUT") == "1"
            and os.environ.get("MATRIS_W8A8_BACKEND") == "cuda_wmma_tail_n128"
        ):
            out = cuda_w8a8_static_wmma_dual_linear(
                core_2d,
                gate_2d,
                core_q_weight,
                core_weight_scale,
                core_activation_scale,
                core_bias_fp32,
                gate_q_weight,
                gate_weight_scale,
                gate_activation_scale,
                gate_bias_fp32,
            )
            if out is not None:
                _record_w8a8_second_grad_input_stat("cuda_forward_calls")
        if out is None:
            out = _triton_w8a8_static_dual_forward(
                core_2d,
                gate_2d,
                core_q_weight_t.contiguous(),
                gate_q_weight_t.contiguous(),
                core_weight_scale.float().contiguous(),
                gate_weight_scale.float().contiguous(),
                core_activation_scale,
                gate_activation_scale,
                core_bias_fp32,
                gate_bias_fp32,
            )
        if out is None:
            core_q = torch.round(core_2d / core_activation_scale).clamp(-127, 127)
            gate_q = torch.round(gate_2d / gate_activation_scale).clamp(-127, 127)
            core_dq_weight = core_q_weight.float() * core_weight_scale.float().reshape(-1, 1)
            gate_dq_weight = gate_q_weight.float() * gate_weight_scale.float().reshape(-1, 1)
            core_out = F.linear(core_q * core_activation_scale, core_dq_weight, core_bias_fp32)
            gate_out = F.linear(gate_q * gate_activation_scale, gate_dq_weight, gate_bias_fp32)
        else:
            core_out, gate_out = out

        ctx.save_for_backward(
            core_q_weight.contiguous(),
            core_weight_scale.float().contiguous(),
            gate_q_weight.contiguous(),
            gate_weight_scale.float().contiguous(),
        )
        ctx.core_shape = core_shape
        ctx.gate_shape = gate_shape
        return (
            core_out.reshape(*core_shape[:-1], core_q_weight.shape[0]),
            gate_out.reshape(*gate_shape[:-1], gate_q_weight.shape[0]),
        )

    @staticmethod
    def backward(ctx, grad_core_out: torch.Tensor, grad_gate_out: torch.Tensor):
        core_q_weight, core_weight_scale, gate_q_weight, gate_weight_scale = ctx.saved_tensors
        grad_core_2d = grad_core_out.float().reshape(-1, core_q_weight.shape[0])
        grad_gate_2d = grad_gate_out.float().reshape(-1, gate_q_weight.shape[0])
        grad_core_2d_out = _cuda_w8a8_static_grad_input(grad_core_2d.contiguous(), core_q_weight, core_weight_scale)
        grad_gate_2d_out = _cuda_w8a8_static_grad_input(grad_gate_2d.contiguous(), gate_q_weight, gate_weight_scale)
        if grad_core_2d_out is None:
            core_dq_weight = core_q_weight.float() * core_weight_scale.reshape(-1, 1)
            grad_core_2d_out = F.linear(grad_core_2d, core_dq_weight.t())
        if grad_gate_2d_out is None:
            gate_dq_weight = gate_q_weight.float() * gate_weight_scale.reshape(-1, 1)
            grad_gate_2d_out = F.linear(grad_gate_2d, gate_dq_weight.t())
        grad_core = grad_core_2d_out.reshape(ctx.core_shape)
        grad_gate = grad_gate_2d_out.reshape(ctx.gate_shape)
        return grad_core, grad_gate, None, None, None, None, None, None, None, None, None, None


class FakeQuantLinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        weight_bits: int = 8,
        scale_granularity: str = "per_channel",
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.weight_bits = weight_bits
        self.scale_granularity = scale_granularity

        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone())

        self.last_stats: dict[str, torch.Tensor | str] = {}

    def _fake_quant_weight(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        return _fake_quantize_weight(
            self.weight,
            self.weight_bits,
            self.scale_granularity,
            self.module_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq_weight, stats = self._fake_quant_weight()
        x_fp32 = x.float()
        bias = self.bias.float() if self.bias is not None else None
        out = F.linear(x_fp32, dq_weight, bias)

        stats["input_abs_max"] = x_fp32.abs().max().detach()
        stats["output_abs_max"] = out.abs().max().detach()
        self.last_stats = stats
        return out


class ActivationWeightFakeQuantLinear(FakeQuantLinear):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        weight_bits: int = 8,
        activation_bits: int = 8,
        scale_granularity: str = "per_channel",
        activation_scale_granularity: str = "dynamic_per_tensor",
        activation_block_m: int = 32,
    ) -> None:
        super().__init__(
            original,
            module_name=module_name,
            weight_bits=weight_bits,
            scale_granularity=scale_granularity,
        )
        if activation_scale_granularity not in (
            "dynamic_per_tensor",
            "static_per_tensor",
            "dynamic_per_m_block",
        ):
            raise ValueError(
                "ActivationWeightFakeQuantLinear currently supports "
                "activation_scale_granularity in "
                "{dynamic_per_tensor, static_per_tensor, dynamic_per_m_block}"
            )
        self.activation_bits = activation_bits
        self.activation_scale_granularity = activation_scale_granularity
        self.activation_block_m = activation_block_m
        self.calibrating_activation = False
        self.register_buffer("activation_observed_abs_max", torch.tensor(0.0))
        self.register_buffer("activation_static_scale", torch.tensor(0.0))
        self.activation_static_calibrated = False

    def reset_activation_calibration(self) -> None:
        self.activation_observed_abs_max.zero_()
        self.activation_static_scale.zero_()
        self.activation_static_calibrated = False

    def finalize_activation_calibration(self) -> None:
        qmax = 2 ** (self.activation_bits - 1) - 1
        scale = self.activation_observed_abs_max.clamp_min(1e-12) / qmax
        scale_multiplier = _activation_static_scale_multiplier_for(self.module_name)
        self.activation_static_scale.copy_(scale * scale_multiplier)
        self.activation_static_calibrated = True

    def _fake_quant_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.activation_scale_granularity == "dynamic_per_tensor":
            return _fake_quantize_activation_dynamic_per_tensor(
                x,
                self.activation_bits,
            )
        if self.activation_scale_granularity == "dynamic_per_m_block":
            return _fake_quantize_activation_dynamic_m_block(
                x,
                self.activation_bits,
                self.activation_block_m,
            )

        if self.activation_static_calibrated:
            scale = self.activation_static_scale.to(device=x.device, dtype=torch.float32)
            return _fake_quantize_activation_static_per_tensor(
                x,
                self.activation_bits,
                scale,
            )

        x_abs_max = x.float().abs().max().detach()
        if self.calibrating_activation:
            self.activation_observed_abs_max.copy_(
                torch.maximum(self.activation_observed_abs_max.to(x_abs_max.device), x_abs_max).cpu()
            )
            return _fake_quantize_activation_dynamic_per_tensor(
                x,
                self.activation_bits,
            )

        if not self.activation_static_calibrated and self.activation_static_scale.item() == 0.0:
            scale = x_abs_max.clamp_min(1e-12) / (2 ** (self.activation_bits - 1) - 1)
        else:
            scale = self.activation_static_scale.to(device=x.device, dtype=torch.float32)
        return _fake_quantize_activation_static_per_tensor(
            x,
            self.activation_bits,
            scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq_weight, stats = self._fake_quant_weight()
        dq_x, activation_stats = self._fake_quant_activation(x)
        bias = self.bias.float() if self.bias is not None else None
        out = F.linear(dq_x, dq_weight, bias)

        stats.update(activation_stats)
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        stats["activation_bits"] = torch.tensor(self.activation_bits)
        stats["activation_scale_granularity"] = self.activation_scale_granularity
        stats["activation_block_m"] = torch.tensor(self.activation_block_m)
        if self.activation_scale_granularity == "static_per_tensor":
            stats["activation_static_calibrated"] = str(self.activation_static_calibrated)
            stats["activation_static_scale"] = self.activation_static_scale.detach()
        stats["activation_fake_quant_backward"] = "ste"
        self.last_stats = stats
        return out


class SmoothActivationWeightFakeQuantLinear(ActivationWeightFakeQuantLinear):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        weight_bits: int = 8,
        activation_bits: int = 8,
        scale_granularity: str = "per_channel",
        activation_scale_granularity: str = "static_per_tensor",
        activation_block_m: int = 32,
        smooth_alpha: float = 0.5,
        smooth_scale_clamp: float = 1.0e4,
    ) -> None:
        super().__init__(
            original,
            module_name=module_name,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            scale_granularity=scale_granularity,
            activation_scale_granularity=activation_scale_granularity,
            activation_block_m=activation_block_m,
        )
        if activation_scale_granularity != "static_per_tensor":
            raise ValueError("SmoothActivationWeightFakeQuantLinear currently expects static_per_tensor activation scale")
        if not 0.0 <= smooth_alpha <= 1.0:
            raise ValueError(f"smooth_alpha must be in [0, 1], got {smooth_alpha}")
        self.smooth_alpha = smooth_alpha
        self.smooth_scale_clamp = smooth_scale_clamp
        in_features = self.weight.shape[1]
        self.register_buffer("activation_observed_abs_max_channel", torch.zeros(in_features))
        self.register_buffer("smooth_scale", torch.ones(in_features))
        self.smooth_calibrated = False

    def reset_activation_calibration(self) -> None:
        super().reset_activation_calibration()
        self.activation_observed_abs_max_channel.zero_()
        self.smooth_scale.fill_(1.0)
        self.smooth_calibrated = False

    def finalize_activation_calibration(self) -> None:
        qmax = 2 ** (self.activation_bits - 1) - 1
        act = self.activation_observed_abs_max_channel.float().clamp_min(1e-12)
        weight_col = self.weight.detach().float().abs().amax(dim=0).cpu().clamp_min(1e-12)
        smooth = (act.pow(self.smooth_alpha) / weight_col.pow(1.0 - self.smooth_alpha)).clamp(
            1.0 / self.smooth_scale_clamp,
            self.smooth_scale_clamp,
        )
        self.smooth_scale.copy_(smooth)
        smoothed_act_abs_max = act / smooth
        scale = smoothed_act_abs_max.max().clamp_min(1e-12) / qmax
        scale_multiplier = _activation_static_scale_multiplier_for(self.module_name)
        self.activation_static_scale.copy_(scale * scale_multiplier)
        self.activation_static_calibrated = True
        self.smooth_calibrated = True

    def _smooth_scale_for(self, x: torch.Tensor) -> torch.Tensor:
        if not self.smooth_calibrated:
            return torch.ones(self.weight.shape[1], device=x.device, dtype=torch.float32)
        return self.smooth_scale.to(device=x.device, dtype=torch.float32)

    def _fake_quant_weight(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        smooth = self.smooth_scale.to(device=self.weight.device, dtype=torch.float32)
        smoothed_weight = self.weight.float() * smooth.reshape(1, -1)
        dq_weight, stats = _fake_quantize_weight(
            smoothed_weight,
            self.weight_bits,
            self.scale_granularity,
            self.module_name,
        )
        stats["backend"] = "smooth_fake_w8a8"
        stats["smooth_alpha"] = torch.tensor(self.smooth_alpha)
        stats["smooth_scale_min"] = smooth.min().detach()
        stats["smooth_scale_mean"] = smooth.mean().detach()
        stats["smooth_scale_max"] = smooth.max().detach()
        stats["smooth_calibrated"] = str(self.smooth_calibrated)
        return dq_weight, stats

    def _fake_quant_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_fp32 = x.float()
        if self.calibrating_activation:
            x_2d = x_fp32.reshape(-1, x_fp32.shape[-1])
            channel_abs_max = x_2d.abs().amax(dim=0).detach().cpu()
            self.activation_observed_abs_max_channel.copy_(
                torch.maximum(self.activation_observed_abs_max_channel, channel_abs_max)
            )
            x_abs_max = x_fp32.abs().max().detach()
            self.activation_observed_abs_max.copy_(
                torch.maximum(self.activation_observed_abs_max.to(x_abs_max.device), x_abs_max).cpu()
            )
            return _fake_quantize_activation_dynamic_per_tensor(x_fp32, self.activation_bits)

        smooth = self._smooth_scale_for(x_fp32)
        smoothed_x = x_fp32 / smooth.reshape(*([1] * (x_fp32.ndim - 1)), -1)
        if self.activation_static_calibrated:
            scale = self.activation_static_scale.to(device=x.device, dtype=torch.float32)
        elif self.activation_static_scale.item() == 0.0:
            scale = smoothed_x.abs().max().detach().clamp_min(1e-12) / (2 ** (self.activation_bits - 1) - 1)
        else:
            scale = self.activation_static_scale.to(device=x.device, dtype=torch.float32)
        dq_x, stats = _fake_quantize_activation_static_per_tensor(
            smoothed_x,
            self.activation_bits,
            scale,
        )
        stats["smooth_activation"] = torch.tensor(1)
        return dq_x, stats


class LowPrecisionLinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        compute_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        if compute_dtype == "bf16":
            self.compute_dtype = torch.bfloat16
        elif compute_dtype == "fp16":
            self.compute_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

        self.module_name = module_name
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone())

        self.last_stats: dict[str, torch.Tensor | str] = {
            "module_name": module_name,
            "backend": "linear_low_precision",
            "compute_dtype": str(self.compute_dtype),
            "master_weight_dtype": str(self.weight.dtype),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x.to(self.compute_dtype)
        weight_low = self.weight.to(self.compute_dtype)
        bias_low = self.bias.to(self.compute_dtype) if self.bias is not None else None
        out = F.linear(x_low, weight_low, bias_low).float()

        stats = dict(self.last_stats)
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.abs().max().detach()
        stats["input_dtype"] = str(x.dtype)
        stats["output_dtype"] = str(out.dtype)
        self.last_stats = stats
        return out


class CachedLowPrecisionLinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        compute_dtype: str = "bf16",
    ) -> None:
        super().__init__()
        if compute_dtype == "bf16":
            self.compute_dtype = torch.bfloat16
        elif compute_dtype == "fp16":
            self.compute_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

        self.module_name = module_name
        self.register_buffer("weight_low", original.weight.detach().to(self.compute_dtype).contiguous())
        if original.bias is None:
            self.register_buffer("bias_low", None)
        else:
            self.register_buffer("bias_low", original.bias.detach().to(self.compute_dtype).contiguous())

        self.last_stats: dict[str, torch.Tensor | str] = {
            "module_name": module_name,
            "backend": "linear_low_precision_cached",
            "compute_dtype": str(self.compute_dtype),
            "cached_weight_dtype": str(self.weight_low.dtype),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x.to(self.compute_dtype)
        out = F.linear(x_low, self.weight_low, self.bias_low).float()

        stats = dict(self.last_stats)
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.abs().max().detach()
        stats["input_dtype"] = str(x.dtype)
        stats["output_dtype"] = str(out.dtype)
        self.last_stats = stats
        return out


class ForwardOnlyFakeQuantLinear(FakeQuantLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq_weight, stats = self._fake_quant_weight()
        bias = self.bias.float() if self.bias is not None else None
        out = _SelectiveFakeQuantLinearFunction.apply(
            x,
            dq_weight,
            self.weight.float(),
            bias,
        )

        x_fp32 = x.float()
        stats["input_abs_max"] = x_fp32.abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        stats["fake_quant_scope"] = "forward_only"
        self.last_stats = stats
        return out


class BackwardFakeQuantLinear(FakeQuantLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq_weight, stats = self._fake_quant_weight()
        bias = self.bias.float() if self.bias is not None else None
        out = _SelectiveFakeQuantLinearFunction.apply(
            x,
            self.weight.float(),
            dq_weight,
            bias,
        )

        x_fp32 = x.float()
        stats["input_abs_max"] = x_fp32.abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        stats["fake_quant_scope"] = "backward_only"
        self.last_stats = stats
        return out


class PackedW8A32Linear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        weight_bits: int = 8,
        scale_granularity: str = "per_channel",
        use_cuda_kernel: bool = True,
    ) -> None:
        super().__init__()
        if weight_bits != 8:
            raise ValueError("PackedW8A32Linear currently supports weight_bits=8 only")
        if scale_granularity not in ("per_tensor", "per_channel"):
            raise ValueError(f"Unknown scale_granularity: {scale_granularity}")

        self.module_name = module_name
        self.weight_bits = weight_bits
        self.scale_granularity = scale_granularity
        self.use_cuda_kernel = use_cuda_kernel

        q_weight, scale, stats = self._pack_weight(original.weight.detach())
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)

        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)

        self.last_stats: dict[str, torch.Tensor | str] = stats

    def _pack_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str]]:
        qmax = 2 ** (self.weight_bits - 1) - 1
        weight_fp32 = weight.float()

        if self.scale_granularity == "per_tensor":
            scale = weight_fp32.abs().max().clamp_min(1e-12) / qmax
            scale = scale.reshape(1).expand(weight_fp32.shape[0]).contiguous()
        else:
            scale = weight_fp32.abs().amax(dim=1).clamp_min(1e-12) / qmax

        q_weight = torch.round(weight_fp32 / scale.reshape(-1, 1)).clamp(-qmax, qmax).to(torch.int8)
        dq_weight = q_weight.float() * scale.reshape(-1, 1)
        error = (dq_weight - weight_fp32).abs()

        stats = {
            "module_name": self.module_name,
            "backend": "packed_w8a32",
            "weight_abs_max": weight_fp32.abs().max().detach(),
            "scale_mean": scale.mean().detach(),
            "scale_min": scale.min().detach(),
            "scale_max": scale.max().detach(),
            "weight_quant_error_mae": error.mean().detach(),
            "weight_quant_error_max": error.max().detach(),
            "saturation_ratio": (q_weight.abs() >= qmax).float().mean().detach(),
        }
        return q_weight.contiguous(), scale.contiguous(), stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _QuantLinearW8A32Function.apply(
            x,
            self.q_weight,
            self.scale,
            self.bias,
            self.use_cuda_kernel,
        )
        stats = dict(self.last_stats)
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        stats["kernel_requested"] = "cuda" if self.use_cuda_kernel else "torch_fallback"
        self.last_stats = stats
        return out


class TritonW8A32Linear(PackedW8A32Linear):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        weight_bits: int = 8,
        scale_granularity: str = "per_channel",
    ) -> None:
        super().__init__(
            original,
            module_name=module_name,
            weight_bits=weight_bits,
            scale_granularity=scale_granularity,
            use_cuda_kernel=False,
        )
        self.last_stats = {
            **self.last_stats,
            "backend": "triton_w8a32",
            "backward": "torch_dequantized_grad_x",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.float() if self.bias is not None else None
        out = _TritonW8A32Function.apply(x, self.q_weight, self.scale, bias)

        stats = dict(self.last_stats)
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        self.last_stats = stats
        return out


class W8ALowPrecisionLinear(PackedW8A32Linear):
    """Reference W8A16/W8ABF16 path: int8 per-channel weight, half/bfloat16 activation."""

    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        compute_dtype: str = "bf16",
        weight_bits: int = 8,
        scale_granularity: str = "per_channel",
        backend_preference: str = "reference",
    ) -> None:
        super().__init__(
            original,
            module_name=module_name,
            weight_bits=weight_bits,
            scale_granularity=scale_granularity,
            use_cuda_kernel=False,
        )
        if compute_dtype == "bf16":
            self.compute_dtype = torch.bfloat16
            backend = "w8abf16_reference"
        elif compute_dtype == "fp16":
            self.compute_dtype = torch.float16
            backend = "w8a16_reference"
        else:
            raise ValueError(f"Unsupported W8A low precision compute_dtype: {compute_dtype}")
        self.backend_preference = backend_preference
        self.register_buffer("q_weight_t", self.q_weight.t().contiguous())
        cached_dequant_weight = (self.q_weight.float() * self.scale.float().reshape(-1, 1)).to(self.compute_dtype)
        self.register_buffer("cached_dequant_weight_lowp", cached_dequant_weight.contiguous())
        if self.bias is not None:
            self.register_buffer("cached_bias_lowp", self.bias.to(self.compute_dtype).contiguous())
        else:
            self.cached_bias_lowp = None
        self.last_stats = {
            **self.last_stats,
            "backend": f"{backend}_{backend_preference}" if backend_preference != "reference" else backend,
            "compute_dtype": str(self.compute_dtype),
            "weight_packing": "int8_per_channel",
            "backward": "torch_autograd_low_precision_grad_x",
        }

    def _fake_quant_weight(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        dq_weight = self.q_weight.float() * self.scale.float().reshape(-1, 1)
        stats = dict(self.last_stats)
        stats["dequant_weight_dtype"] = str(dq_weight.dtype)
        return dq_weight, stats

    def _cuda_forward(self, x: torch.Tensor) -> torch.Tensor | None:
        global _W8A_LOWP_KERNEL_WARNED

        env_backend = os.environ.get("MATRIS_W8A_LOWP_BACKEND")
        supported_backends = (
            "cuda_v0",
            "cuda_v1",
            "wmma_v0",
            "triton_v0",
            "cutlass_mixed_v0",
            "cutlass_mixed_nocontig_v1",
            "cached_dequant_lowp_v0",
        )
        if self.backend_preference not in supported_backends and env_backend not in supported_backends:
            return None
        if x.device.type != "cuda" or x.shape[-1] != 128 or self.q_weight.shape != (128, 128):
            return None
        if self.compute_dtype not in (torch.float16, torch.bfloat16):
            return None
        if self.backend_preference == "cached_dequant_lowp_v0" or env_backend == "cached_dequant_lowp_v0":
            return F.linear(
                x.to(self.compute_dtype),
                self.cached_dequant_weight_lowp,
                self.cached_bias_lowp,
            ).float()
        if self.backend_preference == "triton_v0" or env_backend == "triton_v0":
            return _W8ALowPrecisionTritonFunction.apply(
                x,
                self.q_weight,
                self.q_weight_t,
                self.scale,
                self.bias,
                self.compute_dtype,
            )
        matris_op = _load_matris_op()
        if matris_op is None or not hasattr(matris_op, "quant_linear_w8a_lowp_forward"):
            if not _W8A_LOWP_KERNEL_WARNED:
                warnings.warn(
                    "matris_op.quant_linear_w8a_lowp_forward is unavailable; falling back to reference W8A low precision linear.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _W8A_LOWP_KERNEL_WARNED = True
            return None

        return _W8ALowPrecisionCudaFunction.apply(
            x,
            self.q_weight,
            self.q_weight_t,
            self.scale,
            self.bias,
            self.compute_dtype,
            "cutlass_mixed_nocontig" if (self.backend_preference == "cutlass_mixed_nocontig_v1" or env_backend == "cutlass_mixed_nocontig_v1") else (
                "cutlass_mixed" if (self.backend_preference == "cutlass_mixed_v0" or env_backend == "cutlass_mixed_v0") else (
                "wmma" if (self.backend_preference == "wmma_v0" or env_backend == "wmma_v0") else (
                    self.backend_preference == "cuda_v1" or env_backend == "cuda_v1"
                )
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_cuda = self._cuda_forward(x)
        if out_cuda is not None:
            stats = dict(self.last_stats)
            if os.environ.get("MATRIS_W8A_LOWP_COLLECT_TENSOR_STATS") == "1":
                stats["input_abs_max"] = x.float().abs().max().detach()
                stats["output_abs_max"] = out_cuda.float().abs().max().detach()
            stats["input_dtype"] = str(x.dtype)
            stats["activation_compute_dtype"] = str(self.compute_dtype)
            stats["backend_runtime"] = os.environ.get("MATRIS_W8A_LOWP_BACKEND") or self.backend_preference
            self.last_stats = stats
            return out_cuda

        x_low = x.to(self.compute_dtype)
        weight_low = (self.q_weight.float() * self.scale.float().reshape(-1, 1)).to(self.compute_dtype)
        bias_low = self.bias.to(self.compute_dtype) if self.bias is not None else None
        out = F.linear(x_low, weight_low, bias_low).float()

        stats = dict(self.last_stats)
        if os.environ.get("MATRIS_W8A_LOWP_COLLECT_TENSOR_STATS") == "1":
            stats["input_abs_max"] = x.float().abs().max().detach()
            stats["output_abs_max"] = out.float().abs().max().detach()
        stats["input_dtype"] = str(x.dtype)
        stats["activation_compute_dtype"] = str(self.compute_dtype)
        self.last_stats = stats
        return out


class TritonW8A8StaticLinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        weight_bits: int = 8,
        activation_bits: int = 8,
        scale_granularity: str = "per_channel",
    ) -> None:
        super().__init__()
        if weight_bits != 8 or activation_bits != 8:
            raise ValueError("TritonW8A8StaticLinear currently supports W8A8 only")
        if scale_granularity != "per_channel":
            raise ValueError("TritonW8A8StaticLinear currently supports per-channel weight scale only")

        self.module_name = module_name
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.scale_granularity = scale_granularity
        self.activation_scale_granularity = "static_per_tensor"

        q_weight, weight_scale, weight_stats = self._pack_weight(original.weight.detach())
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("q_weight_t", q_weight.t().contiguous())
        self.register_buffer("scale", weight_scale)
        self.register_buffer("activation_observed_abs_max", torch.tensor(0.0))
        self.register_buffer("activation_static_scale", torch.tensor(0.0))
        self.activation_static_calibrated = False
        self.calibrating_activation = False
        self.collect_runtime_stats = False
        self.activation_static_sample_count = 0
        self.activation_static_sample_chunks: list[torch.Tensor] = []

        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)

        self.last_stats: dict[str, torch.Tensor | str] = {
            **weight_stats,
            "module_name": module_name,
            "backend": "triton_w8a8_static",
            "activation_scale_granularity": "static_per_tensor",
            "backward": "torch_dequantized_grad_x",
        }

    def _pack_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str]]:
        qmax = 2 ** (self.weight_bits - 1) - 1
        weight_fp32 = weight.float()
        scale = weight_fp32.abs().amax(dim=1).clamp_min(1e-12) / qmax
        q_weight = torch.round(weight_fp32 / scale.reshape(-1, 1)).clamp(-qmax, qmax).to(torch.int8)
        dq_weight = q_weight.float() * scale.reshape(-1, 1)
        error = (dq_weight - weight_fp32).abs()
        stats = {
            "weight_abs_max": weight_fp32.abs().max().detach(),
            "scale_mean": scale.mean().detach(),
            "scale_min": scale.min().detach(),
            "scale_max": scale.max().detach(),
            "weight_quant_error_mae": error.mean().detach(),
            "weight_quant_error_max": error.max().detach(),
            "saturation_ratio": (q_weight.abs() >= qmax).float().mean().detach(),
        }
        return q_weight.contiguous(), scale.contiguous(), stats

    def reset_activation_calibration(self) -> None:
        self.activation_observed_abs_max.zero_()
        self.activation_static_scale.zero_()
        self.activation_static_calibrated = False
        self.activation_static_sample_count = 0
        self.activation_static_sample_chunks.clear()

    def finalize_activation_calibration(self) -> None:
        qmax = 2 ** (self.activation_bits - 1) - 1
        observed_abs_max = self.activation_observed_abs_max.float().clamp_min(1e-12)
        calibration_mode = _activation_static_calibration_mode_for(self.module_name)
        calibration_value = observed_abs_max
        percentile_value = None
        if calibration_mode == "percentile" and self.activation_static_sample_chunks:
            samples = torch.cat(self.activation_static_sample_chunks).float()
            percentile = _activation_static_percentile_for(self.module_name)
            percentile_value = torch.quantile(samples, percentile).clamp_min(1e-12)
            calibration_value = torch.minimum(observed_abs_max.cpu(), percentile_value.cpu()).to(observed_abs_max.device)
        scale = calibration_value.clamp_min(1e-12) / qmax
        scale_multiplier = _activation_static_scale_multiplier_for(self.module_name)
        self.activation_static_scale.copy_(scale * scale_multiplier)
        self.activation_static_calibrated = True
        self.last_stats = {
            **self.last_stats,
            "activation_static_calibration": calibration_mode,
            "activation_observed_abs_max": observed_abs_max.detach(),
            "activation_calibration_abs_value": calibration_value.detach(),
            "activation_static_scale_multiplier": torch.tensor(scale_multiplier),
            "activation_calibration_percentile": torch.tensor(_activation_static_percentile_for(self.module_name)),
            "activation_calibration_sample_count": torch.tensor(self.activation_static_sample_count),
        }
        if percentile_value is not None:
            self.last_stats["activation_calibration_percentile_abs"] = percentile_value.detach()

    def _observe_activation_for_static_calibration(self, x: torch.Tensor, x_abs_max: torch.Tensor) -> None:
        self.activation_observed_abs_max.copy_(
            torch.maximum(self.activation_observed_abs_max.to(x_abs_max.device), x_abs_max).cpu()
        )
        if _activation_static_calibration_mode_for(self.module_name) != "percentile":
            return
        sample_limit = _activation_static_sample_limit()
        remaining = sample_limit - self.activation_static_sample_count
        if remaining <= 0:
            return
        samples = _sample_activation_abs_values(x, remaining)
        if samples.numel() == 0:
            return
        self.activation_static_sample_chunks.append(samples)
        self.activation_static_sample_count += samples.numel()

    def _activation_scale_for(self, x: torch.Tensor) -> torch.Tensor:
        qmax = 2 ** (self.activation_bits - 1) - 1
        if self.activation_static_calibrated:
            return self.activation_static_scale.to(device=x.device, dtype=torch.float32).reshape(())
        x_abs_max = x.float().abs().max().detach()
        if self.calibrating_activation:
            self._observe_activation_for_static_calibration(x, x_abs_max)
            return x_abs_max.clamp_min(1e-12) / qmax
        if not self.activation_static_calibrated and self.activation_static_scale.item() == 0.0:
            return x_abs_max.clamp_min(1e-12) / qmax
        return self.activation_static_scale.to(device=x.device, dtype=torch.float32).reshape(())

    def _fake_quant_weight(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        dq_weight = self.q_weight.float() * self.scale.reshape(-1, 1)
        return dq_weight, dict(self.last_stats)

    def _fake_quant_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return _fake_quantize_activation_static_per_tensor(
            x,
            self.activation_bits,
            self._activation_scale_for(x),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_scale = self._activation_scale_for(x)
        bias = self.bias.float() if self.bias is not None else None
        if not torch.is_grad_enabled():
            original_shape = x.shape
            x_2d = x.float().reshape(-1, original_shape[-1]).contiguous()
            backend = os.environ.get("MATRIS_W8A8_BACKEND")
            if backend == "cuda_wmma":
                out_2d = _cuda_w8a8_static_wmma_forward(
                    x_2d,
                    self.q_weight,
                    self.scale,
                    activation_scale,
                    bias,
                )
            elif backend == "cuda_cutlass":
                out_2d = _cuda_w8a8_static_cutlass_forward(
                    x_2d,
                    self.q_weight,
                    self.scale,
                    activation_scale,
                    bias,
                )
            else:
                out_2d = _triton_w8a8_static_forward(
                    x_2d,
                    self.q_weight_t,
                    self.scale,
                    activation_scale,
                    bias,
                )
            if out_2d is None:
                q_x = torch.round(x_2d / activation_scale).clamp(-127, 127)
                dq_x = q_x * activation_scale
                dq_weight = self.q_weight.float() * self.scale.reshape(-1, 1)
                out_2d = F.linear(dq_x, dq_weight, bias)
            return out_2d.reshape(*original_shape[:-1], self.q_weight.shape[0])

        out = triton_w8a8_static_linear(
            x,
            self.q_weight,
            self.q_weight_t,
            self.scale,
            activation_scale,
            bias,
        )

        stats = dict(self.last_stats)
        stats["activation_scale"] = activation_scale.detach()
        stats["activation_static_calibrated"] = str(self.activation_static_calibrated)
        stats["activation_static_scale"] = self.activation_static_scale.detach()
        if self.collect_runtime_stats:
            x_fp32 = x.float()
            q_x = torch.round(x_fp32 / activation_scale).clamp(-127, 127)
            dq_x = q_x * activation_scale
            activation_error = (dq_x - x_fp32).abs()
            stats["input_abs_max"] = x_fp32.abs().max().detach()
            stats["output_abs_max"] = out.float().abs().max().detach()
            stats["activation_quant_error_mae"] = activation_error.mean().detach()
            stats["activation_quant_error_max"] = activation_error.max().detach()
            stats["activation_saturation_ratio"] = (q_x.abs() >= 127).float().mean().detach()
        self.last_stats = stats
        return out


class _TorchFP8StaticFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight_col_major: torch.Tensor,
        dq_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.float().reshape(-1, original_shape[-1]).contiguous()
        activation_scale = activation_scale.float().reshape(()).contiguous()
        weight_scale = weight_scale.float().reshape(()).contiguous()
        q_x = (x_2d / activation_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        out_2d = torch._scaled_mm(
            q_x,
            q_weight_col_major,
            scale_a=activation_scale,
            scale_b=weight_scale,
            out_dtype=torch.float32,
        )
        if bias is not None:
            out_2d = out_2d + bias.float()
        ctx.save_for_backward(dq_weight)
        ctx.original_shape = original_shape
        ctx.has_bias = bias is not None
        return out_2d.reshape(*original_shape[:-1], q_weight_col_major.shape[1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dq_weight,) = ctx.saved_tensors
        grad_2d = grad_output.float().reshape(-1, grad_output.shape[-1])
        grad_x = grad_2d.matmul(dq_weight.float()).reshape(ctx.original_shape)
        grad_bias = grad_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, None, None, None, None, grad_bias


class TorchFP8StaticLinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        fp8_dtype: str = "e4m3fn",
    ) -> None:
        super().__init__()
        if fp8_dtype != "e4m3fn":
            raise ValueError("TorchFP8StaticLinear currently supports fp8_dtype=e4m3fn only")
        self.module_name = module_name
        self.fp8_dtype = fp8_dtype
        self.activation_scale_granularity = "static_per_tensor"

        q_weight, q_weight_col_major, dq_weight, weight_scale, weight_stats = self._pack_weight(
            original.weight.detach()
        )
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("q_weight_col_major", q_weight_col_major)
        self.register_buffer("dq_weight", dq_weight)
        self.register_buffer("scale", weight_scale)
        self.register_buffer("activation_observed_abs_max", torch.tensor(0.0))
        self.register_buffer("activation_static_scale", torch.tensor(0.0))
        self.activation_static_calibrated = False
        self.calibrating_activation = False
        self.collect_runtime_stats = False
        self.activation_static_sample_count = 0
        self.activation_static_sample_chunks: list[torch.Tensor] = []

        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)

        self.last_stats: dict[str, torch.Tensor | str] = {
            **weight_stats,
            "module_name": module_name,
            "backend": "torch_scaled_mm_fp8_static",
            "fp8_dtype": "float8_e4m3fn",
            "activation_scale_granularity": "static_per_tensor",
            "weight_scale_granularity": "per_tensor",
            "backward": "torch_dequantized_grad_x",
        }

    def _pack_weight(
        self,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str]]:
        fp8_max = 448.0
        weight_fp32 = weight.float()
        scale = weight_fp32.abs().max().clamp_min(1e-12) / fp8_max
        q_weight = (weight_fp32 / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()
        dq_weight = q_weight.float() * scale
        # torch._scaled_mm expects row-major A and column-major B. q_weight.t()
        # has shape [in_features, out_features] with column-major strides.
        q_weight_col_major = q_weight.t()
        error = (dq_weight - weight_fp32).abs()
        stats = {
            "weight_abs_max": weight_fp32.abs().max().detach(),
            "scale_mean": scale.detach(),
            "scale_min": scale.detach(),
            "scale_max": scale.detach(),
            "weight_quant_error_mae": error.mean().detach(),
            "weight_quant_error_max": error.max().detach(),
            "saturation_ratio": (q_weight.float().abs() >= fp8_max).float().mean().detach(),
        }
        return q_weight, q_weight_col_major, dq_weight.contiguous(), scale.reshape(()).contiguous(), stats

    def reset_activation_calibration(self) -> None:
        self.activation_observed_abs_max.zero_()
        self.activation_static_scale.zero_()
        self.activation_static_calibrated = False
        self.activation_static_sample_count = 0
        self.activation_static_sample_chunks.clear()

    def finalize_activation_calibration(self) -> None:
        fp8_max = 448.0
        observed_abs_max = self.activation_observed_abs_max.float().clamp_min(1e-12)
        calibration_mode = _activation_static_calibration_mode_for(self.module_name)
        calibration_value = observed_abs_max
        percentile_value = None
        if calibration_mode == "percentile" and self.activation_static_sample_chunks:
            samples = torch.cat(self.activation_static_sample_chunks).float()
            percentile = _activation_static_percentile_for(self.module_name)
            percentile_value = torch.quantile(samples, percentile).clamp_min(1e-12)
            calibration_value = torch.minimum(observed_abs_max.cpu(), percentile_value.cpu()).to(observed_abs_max.device)
        scale = calibration_value.clamp_min(1e-12) / fp8_max
        scale_multiplier = _activation_static_scale_multiplier_for(self.module_name)
        self.activation_static_scale.copy_(scale * scale_multiplier)
        self.activation_static_calibrated = True
        self.last_stats = {
            **self.last_stats,
            "activation_static_calibration": calibration_mode,
            "activation_observed_abs_max": observed_abs_max.detach(),
            "activation_calibration_abs_value": calibration_value.detach(),
            "activation_static_scale_multiplier": torch.tensor(scale_multiplier),
            "activation_calibration_percentile": torch.tensor(_activation_static_percentile_for(self.module_name)),
            "activation_calibration_sample_count": torch.tensor(self.activation_static_sample_count),
        }
        if percentile_value is not None:
            self.last_stats["activation_calibration_percentile_abs"] = percentile_value.detach()

    def _observe_activation_for_static_calibration(self, x: torch.Tensor, x_abs_max: torch.Tensor) -> None:
        self.activation_observed_abs_max.copy_(
            torch.maximum(self.activation_observed_abs_max.to(x_abs_max.device), x_abs_max).cpu()
        )
        if _activation_static_calibration_mode_for(self.module_name) != "percentile":
            return
        sample_limit = _activation_static_sample_limit()
        remaining = sample_limit - self.activation_static_sample_count
        if remaining <= 0:
            return
        samples = _sample_activation_abs_values(x, remaining)
        if samples.numel() == 0:
            return
        self.activation_static_sample_chunks.append(samples)
        self.activation_static_sample_count += samples.numel()

    def _activation_scale_for(self, x: torch.Tensor) -> torch.Tensor:
        fp8_max = 448.0
        if self.activation_static_calibrated:
            return self.activation_static_scale.to(device=x.device, dtype=torch.float32).reshape(())
        x_abs_max = x.float().abs().max().detach()
        if self.calibrating_activation:
            self._observe_activation_for_static_calibration(x, x_abs_max)
            return x_abs_max.clamp_min(1e-12) / fp8_max
        if not self.activation_static_calibrated and self.activation_static_scale.item() == 0.0:
            return x_abs_max.clamp_min(1e-12) / fp8_max
        return self.activation_static_scale.to(device=x.device, dtype=torch.float32).reshape(())

    def _fake_quant_weight(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        return self.dq_weight, dict(self.last_stats)

    def _fake_quant_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        activation_scale = self._activation_scale_for(x)
        q_x = (x.float() / activation_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        dq_x = q_x.float() * activation_scale
        error = (dq_x - x.float()).abs()
        return dq_x, {
            "activation_abs_max": x.float().abs().max().detach(),
            "activation_scale": activation_scale.detach(),
            "activation_quant_error_mae": error.mean().detach(),
            "activation_quant_error_max": error.max().detach(),
        }

    def _can_use_scaled_mm(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and hasattr(torch, "_scaled_mm")
            and hasattr(torch, "float8_e4m3fn")
            and x.shape[-1] == self.q_weight_col_major.shape[0]
            and self.q_weight_col_major.shape[0] % 16 == 0
            and self.q_weight_col_major.shape[1] % 16 == 0
        )

    def fp8_linear(self, x: torch.Tensor) -> torch.Tensor:
        activation_scale = self._activation_scale_for(x)
        bias = self.bias.float() if self.bias is not None else None
        if self._can_use_scaled_mm(x):
            out = _TorchFP8StaticFunction.apply(
                x,
                self.q_weight_col_major,
                self.dq_weight,
                self.scale,
                activation_scale,
                bias,
            )
        else:
            dq_x, _ = self._fake_quant_activation(x)
            out = F.linear(dq_x, self.dq_weight.float(), bias)
        stats = dict(self.last_stats)
        stats["activation_scale"] = activation_scale.detach()
        stats["activation_static_calibrated"] = str(self.activation_static_calibrated)
        stats["activation_static_scale"] = self.activation_static_scale.detach()
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        self.last_stats = stats
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fp8_linear(x)


class TransformerEngineFP8Linear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        fp8_format: str = "HYBRID",
        amax_history_len: int = 16,
        amax_compute_algo: str = "max",
    ) -> None:
        super().__init__()
        try:
            import transformer_engine.pytorch as te  # type: ignore
            from transformer_engine.common import recipe  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "TransformerEngineFP8Linear requires transformer_engine[pytorch]."
            ) from exc

        self.module_name = module_name
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.fp8_format = fp8_format
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.te = te
        fmt = getattr(recipe.Format, fp8_format)
        self.fp8_recipe = recipe.DelayedScaling(
            fp8_format=fmt,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )
        self.te_linear = te.Linear(
            original.in_features,
            original.out_features,
            bias=original.bias is not None,
            device=original.weight.device,
            params_dtype=original.weight.dtype,
            name=module_name,
        )
        with torch.no_grad():
            self.te_linear.weight.copy_(original.weight.detach())
            if original.bias is not None and self.te_linear.bias is not None:
                self.te_linear.bias.copy_(original.bias.detach())
        self.freeze_params = os.environ.get("MATRIS_TE_FP8_FREEZE_PARAMS", "1") != "0"
        if self.freeze_params:
            self.te_linear.weight.requires_grad_(False)
            if self.te_linear.bias is not None:
                self.te_linear.bias.requires_grad_(False)
        self.total_calls = 0
        self.te_fp8_calls = 0
        self.fp32_fallback_calls = 0

        self.last_stats: dict[str, torch.Tensor | str] = {
            "module_name": module_name,
            "backend": "transformer_engine_fp8",
            "fp8_format": fp8_format,
            "amax_history_len": torch.tensor(amax_history_len),
            "amax_compute_algo": amax_compute_algo,
            "fallback": "fp32_linear_when_not_cuda_or_te_fp8_shape_incompatible",
            "freeze_params": str(self.freeze_params),
        }

    @property
    def weight(self) -> torch.Tensor:
        return self.te_linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.te_linear.bias

    def _fake_quant_weight(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        return self.te_linear.weight.float(), dict(self.last_stats)

    @staticmethod
    def _collect_stats_enabled() -> bool:
        return os.environ.get("MATRIS_TE_FP8_COLLECT_STATS", "0") == "1"

    @staticmethod
    def _pad_rows_enabled() -> bool:
        return os.environ.get("MATRIS_TE_FP8_PAD_ROWS", "0") == "1"

    @staticmethod
    def _max_pad_overhead() -> float:
        try:
            return float(os.environ.get("MATRIS_TE_FP8_MAX_PAD_OVERHEAD", "0.25"))
        except ValueError:
            return 0.25

    @staticmethod
    def _stats_path() -> Path:
        return Path(
            os.environ.get(
                "MATRIS_TE_FP8_STATS_PATH",
                "results/te_fp8_perf_probe_20260504/te_fp8_shape_stats.jsonl",
            )
        )

    def _shape_plan(self, x: torch.Tensor) -> dict[str, int | float | bool | str]:
        leading_rows = int(x.numel() // max(int(x.shape[-1]), 1)) if x.ndim > 0 else 0
        pad_to_16_rows = ((leading_rows + 15) // 16) * 16 if leading_rows else 0
        pad_overhead_ratio = (
            float(pad_to_16_rows - leading_rows) / float(leading_rows)
            if leading_rows > 0
            else 0.0
        )
        reason = ""
        can_use = True
        if not x.is_cuda:
            can_use = False
            reason = "not_cuda"
        elif x.ndim != 2:
            can_use = False
            reason = "not_2d"
        elif x.shape[-1] != self.in_features:
            can_use = False
            reason = "last_dim_mismatch"
        elif self.in_features % 16 != 0 or self.out_features % 16 != 0:
            can_use = False
            reason = "feature_dim_not_multiple_of_16"
        elif leading_rows % 16 != 0:
            if not self._pad_rows_enabled():
                can_use = False
                reason = "leading_rows_not_multiple_of_16"
            elif pad_overhead_ratio > self._max_pad_overhead():
                can_use = False
                reason = "pad_overhead_too_high"
            else:
                reason = "pad_rows_to_16"
        else:
            reason = "te_fp8_direct"
        return {
            "can_use_te_fp8": can_use,
            "fallback_reason": "" if can_use else reason,
            "execution_plan": reason,
            "leading_rows": leading_rows,
            "leading_rows_mod_8": leading_rows % 8 if leading_rows else 0,
            "leading_rows_mod_16": leading_rows % 16 if leading_rows else 0,
            "pad_to_16_rows": pad_to_16_rows,
            "pad_overhead_ratio": pad_overhead_ratio,
        }

    def _write_shape_stats(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        plan: dict[str, int | float | bool | str],
        used_te_fp8: bool,
    ) -> None:
        if not self._collect_stats_enabled():
            return
        path = self._stats_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "module_name": self.module_name,
            "total_calls": self.total_calls,
            "te_fp8_calls": self.te_fp8_calls,
            "fp32_fallback_calls": self.fp32_fallback_calls,
            "input_shape": list(x.shape),
            "output_shape": list(out.shape),
            "leading_rows": plan["leading_rows"],
            "leading_rows_mod_8": plan["leading_rows_mod_8"],
            "leading_rows_mod_16": plan["leading_rows_mod_16"],
            "pad_to_16_rows": plan["pad_to_16_rows"],
            "pad_overhead_ratio": plan["pad_overhead_ratio"],
            "used_te_fp8_last_call": used_te_fp8,
            "fallback_reason": plan["fallback_reason"],
            "execution_plan": plan["execution_plan"],
            "freeze_params": self.freeze_params,
            "pad_rows_enabled": self._pad_rows_enabled(),
        }
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def te_fp8_linear(self, x: torch.Tensor) -> torch.Tensor:
        self.total_calls += 1
        plan = self._shape_plan(x)
        used_te_fp8 = bool(plan["can_use_te_fp8"])
        if not used_te_fp8:
            self.fp32_fallback_calls += 1
            out = F.linear(
                x,
                self.te_linear.weight.to(dtype=x.dtype),
                None if self.te_linear.bias is None else self.te_linear.bias.to(dtype=x.dtype),
            )
        else:
            self.te_fp8_calls += 1
            original_rows = int(x.shape[0])
            planned_rows = int(plan["pad_to_16_rows"])
            if plan["execution_plan"] == "pad_rows_to_16" and planned_rows > original_rows:
                x_for_te = F.pad(x, (0, 0, 0, planned_rows - original_rows))
            else:
                x_for_te = x
            with self.te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                out = self.te_linear(x_for_te)
            if out.shape[0] != original_rows:
                out = out[:original_rows]
        stats = dict(self.last_stats)
        stats["input_abs_max"] = x.float().abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        stats["used_te_fp8_last_call"] = str(used_te_fp8)
        stats["fallback_reason"] = str(plan["fallback_reason"])
        stats["execution_plan"] = str(plan["execution_plan"])
        stats["total_calls"] = torch.tensor(self.total_calls)
        stats["te_fp8_calls"] = torch.tensor(self.te_fp8_calls)
        stats["fp32_fallback_calls"] = torch.tensor(self.fp32_fallback_calls)
        self.last_stats = stats
        self._write_shape_stats(x, out, plan, used_te_fp8)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.te_fp8_linear(x)


class TritonSmoothW8A8StaticLinear(TritonW8A8StaticLinear):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        weight_bits: int = 8,
        activation_bits: int = 8,
        scale_granularity: str = "per_channel",
        smooth_alpha: float = 0.5,
        smooth_scale_clamp: float = 1.0e4,
    ) -> None:
        if not 0.0 <= smooth_alpha <= 1.0:
            raise ValueError(f"smooth_alpha must be in [0, 1], got {smooth_alpha}")
        super().__init__(
            original,
            module_name=module_name,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            scale_granularity=scale_granularity,
        )
        self.smooth_alpha = smooth_alpha
        self.smooth_scale_clamp = smooth_scale_clamp
        self.register_buffer("weight_fp32", original.weight.detach().float().clone())
        self.register_buffer("activation_observed_abs_max_channel", torch.zeros(original.in_features))
        self.register_buffer("smooth_scale", torch.ones(original.in_features))
        self.smooth_calibrated = False
        self.last_stats = {
            **self.last_stats,
            "backend": "triton_smooth_w8a8_static",
            "smooth_alpha": torch.tensor(self.smooth_alpha),
            "smooth_scale_clamp": torch.tensor(self.smooth_scale_clamp),
        }

    def reset_activation_calibration(self) -> None:
        super().reset_activation_calibration()
        self.activation_observed_abs_max_channel.zero_()
        self.smooth_scale.fill_(1.0)
        self.smooth_calibrated = False

    def finalize_activation_calibration(self) -> None:
        qmax = 2 ** (self.activation_bits - 1) - 1
        act = self.activation_observed_abs_max_channel.float().clamp_min(1e-12)
        weight_col = self.weight_fp32.abs().amax(dim=0).to(act.device).clamp_min(1e-12)
        smooth = (act.pow(self.smooth_alpha) / weight_col.pow(1.0 - self.smooth_alpha)).clamp(
            1.0 / self.smooth_scale_clamp,
            self.smooth_scale_clamp,
        )
        self.smooth_scale.copy_(smooth.to(self.smooth_scale.device))
        smoothed_act_abs_max = act / smooth
        scale = smoothed_act_abs_max.max().clamp_min(1e-12) / qmax
        scale_multiplier = _activation_static_scale_multiplier_for(self.module_name)
        self.activation_static_scale.copy_(scale * scale_multiplier)
        smoothed_weight = self.weight_fp32.to(smooth.device) * smooth.reshape(1, -1)
        q_weight, weight_scale, weight_stats = self._pack_weight(smoothed_weight)
        self.q_weight.copy_(q_weight.to(self.q_weight.device))
        self.q_weight_t.copy_(q_weight.t().contiguous().to(self.q_weight_t.device))
        self.scale.copy_(weight_scale.to(self.scale.device))
        self.activation_static_calibrated = True
        self.smooth_calibrated = True
        self.last_stats = {
            **self.last_stats,
            **weight_stats,
            "smooth_scale_min": self.smooth_scale.min().detach(),
            "smooth_scale_mean": self.smooth_scale.mean().detach(),
            "smooth_scale_max": self.smooth_scale.max().detach(),
            "smooth_calibrated": str(self.smooth_calibrated),
            "activation_static_scale_multiplier": torch.tensor(scale_multiplier),
        }

    def _smooth_scale_for(self, x: torch.Tensor) -> torch.Tensor:
        if not self.smooth_calibrated:
            return torch.ones(self.weight_fp32.shape[1], device=x.device, dtype=torch.float32)
        return self.smooth_scale.to(device=x.device, dtype=torch.float32)

    def _smooth_input(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self._smooth_scale_for(x)
        return x.float() / smooth.reshape(*([1] * (x.ndim - 1)), -1)

    def _activation_scale_for(self, x: torch.Tensor) -> torch.Tensor:
        qmax = 2 ** (self.activation_bits - 1) - 1
        x_fp32 = x.float()
        if self.calibrating_activation:
            x_2d = x_fp32.reshape(-1, x_fp32.shape[-1])
            channel_abs_max = x_2d.abs().amax(dim=0).detach()
            self.activation_observed_abs_max_channel.copy_(
                torch.maximum(
                    self.activation_observed_abs_max_channel.to(channel_abs_max.device),
                    channel_abs_max,
                ).to(self.activation_observed_abs_max_channel.device)
            )
            x_abs_max = x_fp32.abs().max().detach()
            self.activation_observed_abs_max.copy_(
                torch.maximum(self.activation_observed_abs_max.to(x_abs_max.device), x_abs_max).cpu()
            )
            return x_abs_max.clamp_min(1e-12) / qmax
        if self.activation_static_calibrated:
            return self.activation_static_scale.to(device=x.device, dtype=torch.float32).reshape(())
        smoothed_x = self._smooth_input(x_fp32)
        if not self.activation_static_calibrated and self.activation_static_scale.item() == 0.0:
            return smoothed_x.abs().max().detach().clamp_min(1e-12) / qmax
        return self.activation_static_scale.to(device=x.device, dtype=torch.float32).reshape(())

    def _fake_quant_weight(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        dq_weight = self.q_weight.float() * self.scale.reshape(-1, 1)
        return dq_weight, dict(self.last_stats)

    def _fake_quant_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        smoothed_x = self._smooth_input(x)
        dq_x, stats = _fake_quantize_activation_static_per_tensor(
            smoothed_x,
            self.activation_bits,
            self._activation_scale_for(x),
        )
        stats["smooth_activation"] = torch.tensor(1)
        return dq_x, stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_scale = self._activation_scale_for(x)
        bias = self.bias.float() if self.bias is not None else None
        if not torch.is_grad_enabled():
            original_shape = x.shape
            x_2d = x.float().reshape(-1, original_shape[-1]).contiguous()
            smooth = self._smooth_scale_for(x_2d).contiguous()
            out_2d = _triton_w8a8_static_smooth_forward(
                x_2d,
                self.q_weight_t,
                self.scale,
                activation_scale,
                smooth,
                bias,
            )
            if out_2d is None:
                smoothed_x = x_2d / smooth.reshape(1, -1)
                q_x = torch.round(smoothed_x / activation_scale).clamp(-127, 127)
                dq_x = q_x * activation_scale
                dq_weight = self.q_weight.float() * self.scale.reshape(-1, 1)
                out_2d = F.linear(dq_x, dq_weight, bias)
            return out_2d.reshape(*original_shape[:-1], self.q_weight.shape[0])

        smoothed_x = self._smooth_input(x)
        out = triton_w8a8_static_linear(
            smoothed_x,
            self.q_weight,
            self.q_weight_t,
            self.scale,
            activation_scale,
            bias,
        )

        stats = dict(self.last_stats)
        stats["activation_scale"] = activation_scale.detach()
        stats["activation_static_calibrated"] = str(self.activation_static_calibrated)
        stats["activation_static_scale"] = self.activation_static_scale.detach()
        stats["smooth_scale_min"] = self.smooth_scale.min().detach()
        stats["smooth_scale_mean"] = self.smooth_scale.mean().detach()
        stats["smooth_scale_max"] = self.smooth_scale.max().detach()
        stats["smooth_calibrated"] = str(self.smooth_calibrated)
        if self.collect_runtime_stats:
            q_x = torch.round(smoothed_x / activation_scale).clamp(-127, 127)
            dq_x = q_x * activation_scale
            activation_error = (dq_x - smoothed_x).abs()
            stats["input_abs_max"] = x.float().abs().max().detach()
            stats["smoothed_input_abs_max"] = smoothed_x.abs().max().detach()
            stats["output_abs_max"] = out.float().abs().max().detach()
            stats["activation_quant_error_mae"] = activation_error.mean().detach()
            stats["activation_quant_error_max"] = activation_error.max().detach()
            stats["activation_saturation_ratio"] = (q_x.abs() >= 127).float().mean().detach()
        self.last_stats = stats
        return out


class TorchAOInt8WeightOnlyLinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        module_name: str,
        scale_granularity: str = "per_channel",
        set_inductor_config: bool = False,
    ) -> None:
        super().__init__()
        if scale_granularity != "per_channel":
            raise ValueError("TorchAOInt8WeightOnlyLinear currently supports per_channel only")

        self.module_name = module_name
        self.scale_granularity = scale_granularity
        self.backend = "torchao_int8_weight_only"

        weight_fp32 = original.weight.detach().float()
        stats = self._compute_weight_stats(weight_fp32)

        self.linear = nn.Linear(
            original.in_features,
            original.out_features,
            bias=original.bias is not None,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        self.linear.weight.data.copy_(original.weight.detach())
        if original.bias is not None:
            self.linear.bias.data.copy_(original.bias.detach())

        try:
            from torchao.quantization import Int8WeightOnlyConfig, quantize_
        except Exception as exc:
            raise ImportError(
                "TorchAOInt8WeightOnlyLinear requires torchao. Install it with: "
                "python -m pip install -i https://pypi.org/simple torchao"
            ) from exc

        quantize_(
            self.linear,
            Int8WeightOnlyConfig(
                group_size=None,
                set_inductor_config=set_inductor_config,
            ),
        )
        self.last_stats: dict[str, torch.Tensor | str] = stats

    def _compute_weight_stats(self, weight: torch.Tensor) -> dict[str, torch.Tensor | str]:
        qmax = 127
        scale = weight.abs().amax(dim=1).clamp_min(1e-12) / qmax
        q_weight = torch.round(weight / scale.reshape(-1, 1)).clamp(-qmax, qmax)
        dq_weight = q_weight * scale.reshape(-1, 1)
        error = (dq_weight - weight).abs()
        return {
            "module_name": self.module_name,
            "backend": self.backend,
            "weight_abs_max": weight.abs().max().detach(),
            "scale_mean": scale.mean().detach(),
            "scale_min": scale.min().detach(),
            "scale_max": scale.max().detach(),
            "weight_quant_error_mae": error.mean().detach(),
            "weight_quant_error_max": error.max().detach(),
            "saturation_ratio": (q_weight.abs() >= qmax).float().mean().detach(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        out = self.linear(x_fp32)
        stats = dict(self.last_stats)
        stats["input_abs_max"] = x_fp32.abs().max().detach()
        stats["output_abs_max"] = out.float().abs().max().detach()
        stats["kernel_requested"] = "torchao"
        self.last_stats = stats
        return out
