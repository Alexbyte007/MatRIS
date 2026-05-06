from __future__ import annotations

import fnmatch
import os

from torch import nn

from .layers import TritonW8A8StaticLinear


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


def _make_quant_linear(original: nn.Linear, module_name: str, quant_config: dict) -> nn.Module:
    kernel = quant_config.get("kernel")
    if kernel != "triton_w8a8_static":
        raise ValueError(f"Unsupported stable quant kernel: {kernel}")
    return TritonW8A8StaticLinear(
        original,
        module_name=module_name,
        weight_bits=quant_config.get("weight_bits", 8),
        activation_bits=quant_config.get("activation_bits", 8),
        scale_granularity=quant_config.get("scale_granularity", "per_channel"),
    )


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
            _apply_quant_spec(model, spec_config.get("targets", []), spec_config, replaced)
    else:
        _apply_quant_spec(model, quant_config.get("targets", []), quant_config, replaced)

    print(f"[quant] mode={quant_config.get('mode')} replaced {len(replaced)} Linear modules")
    for name in replaced:
        print(f"[quant] replaced: {name}")

    return replaced
