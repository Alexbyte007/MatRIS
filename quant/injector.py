from __future__ import annotations

import fnmatch
import os

from torch import nn

from .layers import (
    ActivationWeightFakeQuantLinear,
    BackwardFakeQuantLinear,
    CachedLowPrecisionLinear,
    FakeQuantLinear,
    ForwardOnlyFakeQuantLinear,
    LowPrecisionLinear,
    PackedW8A32Linear,
    SmoothActivationWeightFakeQuantLinear,
    TorchAOInt8WeightOnlyLinear,
    TritonW8A32Linear,
    TritonSmoothW8A8StaticLinear,
    TritonW8A8StaticLinear,
    TorchFP8StaticLinear,
    TransformerEngineFP8Linear,
    W8ALowPrecisionLinear,
)


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _env_globs(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _passes_env_filters(name: str) -> bool:
    include_globs = _env_globs("MATRIS_QUANT_INCLUDE_GLOBS")
    exclude_globs = _env_globs("MATRIS_QUANT_EXCLUDE_GLOBS")
    if include_globs and not _matches_any(name, include_globs):
        return False
    if exclude_globs and _matches_any(name, exclude_globs):
        return False
    return True


def _replace_linear_children(
    module: nn.Module,
    prefix: str,
    quant_config: dict,
    replaced: list[str],
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Linear):
            if not _passes_env_filters(full_name):
                continue
            setattr(module, child_name, _make_quant_linear(child, full_name, quant_config))
            replaced.append(full_name)
        else:
            _replace_linear_children(child, full_name, quant_config, replaced)


def _make_quant_linear(original: nn.Linear, module_name: str, quant_config: dict) -> nn.Module:
    if quant_config.get("fake_quant", True):
        if quant_config.get("activation_bits") is not None:
            if quant_config.get("smooth_quant", False):
                return SmoothActivationWeightFakeQuantLinear(
                    original,
                    module_name=module_name,
                    weight_bits=quant_config.get("weight_bits", 8),
                    activation_bits=quant_config.get("activation_bits", 8),
                    scale_granularity=quant_config.get("scale_granularity", "per_channel"),
                    activation_scale_granularity=quant_config.get(
                        "activation_scale_granularity",
                        "static_per_tensor",
                    ),
                    activation_block_m=quant_config.get("activation_block_m", 32),
                    smooth_alpha=quant_config.get("smooth_alpha", 0.5),
                    smooth_scale_clamp=quant_config.get("smooth_scale_clamp", 1.0e4),
                )
            return ActivationWeightFakeQuantLinear(
                original,
                module_name=module_name,
                weight_bits=quant_config.get("weight_bits", 8),
                activation_bits=quant_config.get("activation_bits", 8),
                scale_granularity=quant_config.get("scale_granularity", "per_channel"),
                activation_scale_granularity=quant_config.get(
                    "activation_scale_granularity",
                    "dynamic_per_tensor",
                ),
                activation_block_m=quant_config.get("activation_block_m", 32),
            )
        fake_quant_scope = quant_config.get("fake_quant_scope", "forward_backward")
        layer_cls = {
            "forward_backward": FakeQuantLinear,
            "forward_only": ForwardOnlyFakeQuantLinear,
            "backward_only": BackwardFakeQuantLinear,
        }.get(fake_quant_scope)
        if layer_cls is None:
            raise ValueError(f"Unsupported fake_quant_scope: {fake_quant_scope}")
        return layer_cls(
            original,
            module_name=module_name,
            weight_bits=quant_config.get("weight_bits", 8),
            scale_granularity=quant_config.get("scale_granularity", "per_channel"),
        )

    kernel = quant_config.get("kernel")
    if kernel == "w8a32_packed":
        return PackedW8A32Linear(
            original,
            module_name=module_name,
            weight_bits=quant_config.get("weight_bits", 8),
            scale_granularity=quant_config.get("scale_granularity", "per_channel"),
            use_cuda_kernel=quant_config.get("use_cuda_kernel", True),
        )

    if kernel == "triton_w8a32":
        return TritonW8A32Linear(
            original,
            module_name=module_name,
            weight_bits=quant_config.get("weight_bits", 8),
            scale_granularity=quant_config.get("scale_granularity", "per_channel"),
        )

    if kernel == "triton_w8a8_static":
        if quant_config.get("smooth_quant", False):
            return TritonSmoothW8A8StaticLinear(
                original,
                module_name=module_name,
                weight_bits=quant_config.get("weight_bits", 8),
                activation_bits=quant_config.get("activation_bits", 8),
                scale_granularity=quant_config.get("scale_granularity", "per_channel"),
                smooth_alpha=quant_config.get("smooth_alpha", 0.5),
                smooth_scale_clamp=quant_config.get("smooth_scale_clamp", 1.0e4),
            )
        return TritonW8A8StaticLinear(
            original,
            module_name=module_name,
            weight_bits=quant_config.get("weight_bits", 8),
            activation_bits=quant_config.get("activation_bits", 8),
            scale_granularity=quant_config.get("scale_granularity", "per_channel"),
        )

    if kernel == "torch_scaled_mm_fp8_static":
        return TorchFP8StaticLinear(
            original,
            module_name=module_name,
            fp8_dtype=quant_config.get("fp8_dtype", "e4m3fn"),
        )

    if kernel == "transformer_engine_fp8":
        return TransformerEngineFP8Linear(
            original,
            module_name=module_name,
            fp8_format=quant_config.get("fp8_format", "HYBRID"),
            amax_history_len=quant_config.get("amax_history_len", 16),
            amax_compute_algo=quant_config.get("amax_compute_algo", "max"),
        )

    if kernel == "torchao_int8_weight_only":
        return TorchAOInt8WeightOnlyLinear(
            original,
            module_name=module_name,
            scale_granularity=quant_config.get("scale_granularity", "per_channel"),
            set_inductor_config=quant_config.get("set_inductor_config", False),
        )

    if kernel == "linear_low_precision":
        return LowPrecisionLinear(
            original,
            module_name=module_name,
            compute_dtype=quant_config.get("compute_dtype", "bf16"),
        )

    if kernel == "linear_low_precision_cached":
        return CachedLowPrecisionLinear(
            original,
            module_name=module_name,
            compute_dtype=quant_config.get("compute_dtype", "bf16"),
        )

    if kernel == "w8a_low_precision_reference":
        return W8ALowPrecisionLinear(
            original,
            module_name=module_name,
            compute_dtype=quant_config.get("compute_dtype", "bf16"),
            weight_bits=quant_config.get("weight_bits", 8),
            scale_granularity=quant_config.get("scale_granularity", "per_channel"),
            backend_preference=quant_config.get("mixed_backend", "reference"),
        )

    raise ValueError(f"Unsupported real quant kernel: {kernel}")


def _apply_quant_spec(
    model: nn.Module,
    targets: list[str],
    quant_config: dict,
    replaced: list[str],
) -> None:
    modules = dict(model.named_modules())
    replaced_set = set(replaced)

    for module_name, module in list(modules.items()):
        if module_name in replaced_set:
            continue
        if not _matches_any(module_name, targets):
            continue

        if isinstance(module, nn.Linear):
            if not _passes_env_filters(module_name):
                continue
            if "." in module_name:
                parent_name, child_name = module_name.rsplit(".", 1)
                parent = modules[parent_name]
            else:
                parent = model
                child_name = module_name
            setattr(parent, child_name, _make_quant_linear(module, module_name, quant_config))
            replaced.append(module_name)
            replaced_set.add(module_name)
        else:
            before = len(replaced)
            _replace_linear_children(module, module_name, quant_config, replaced)
            replaced_set.update(replaced[before:])


def apply_quant_config(model: nn.Module, quant_config: dict | None) -> list[str]:
    if quant_config is None:
        print("[quant] mode=none replaced 0 Linear modules")
        return []

    replaced: list[str] = []
    include_globs = _env_globs("MATRIS_QUANT_INCLUDE_GLOBS")
    exclude_globs = _env_globs("MATRIS_QUANT_EXCLUDE_GLOBS")
    if include_globs:
        print(f"[quant] include_globs={include_globs}")
    if exclude_globs:
        print(f"[quant] exclude_globs={exclude_globs}")

    target_specs = quant_config.get("target_specs")
    if target_specs:
        for spec in target_specs:
            spec_config = {**quant_config, **spec}
            targets = spec_config.get("targets", [])
            _apply_quant_spec(model, targets, spec_config, replaced)
    else:
        targets = quant_config.get("targets", [])
        _apply_quant_spec(model, targets, quant_config, replaced)

    print(f"[quant] mode={quant_config.get('mode')} replaced {len(replaced)} Linear modules")
    for name in replaced:
        print(f"[quant] replaced: {name}")

    return replaced
