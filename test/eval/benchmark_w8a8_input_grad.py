from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _load_matris_op():
    try:
        from quant.layers import _load_matris_op as load_op

        return load_op()
    except Exception:
        return None


def _pack_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 127
    weight_fp32 = weight.float().contiguous()
    scale = weight_fp32.abs().amax(dim=1).clamp_min(1.0e-12) / qmax
    q_weight = torch.round(weight_fp32 / scale[:, None]).clamp(-qmax, qmax).to(torch.int8).contiguous()
    return q_weight, scale.contiguous()


def _sync():
    torch.cuda.synchronize()


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / iters


def _bench_case(
    matris_op,
    rows: int,
    in_dim: int,
    out_dim: int,
    warmup: int,
    iters: int,
    dtype: torch.dtype,
) -> dict[str, float | int]:
    torch.manual_seed(1234 + rows + in_dim * 17 + out_dim * 31)
    device = torch.device("cuda")

    weight = (torch.randn(out_dim, in_dim, device=device, dtype=torch.float32) * 0.03).contiguous()
    q_weight, weight_scale = _pack_per_channel(weight)
    dq_weight = (q_weight.float() * weight_scale[:, None]).contiguous()

    grad_out = torch.randn(rows, out_dim, device=device, dtype=torch.float32).contiguous()
    if dtype != torch.float32:
        grad_out_for_fp = grad_out.to(dtype)
        dq_weight_for_fp = dq_weight.to(dtype)
    else:
        grad_out_for_fp = grad_out
        dq_weight_for_fp = dq_weight

    # Backward-specific quantized weight for grad_x = grad_out @ dq_weight.
    # Existing forward W8A8 op expects output-channel scales. After transpose, the
    # original weight scales sit on the reduction dimension, so we repack
    # dequantized W.T per grad_x output channel for this prototype.
    bwd_q_weight, bwd_weight_scale = _pack_per_channel(dq_weight.t().contiguous())
    grad_scale = grad_out.abs().amax().clamp_min(1.0e-12) / 127.0
    grad_scale = grad_scale.reshape(()).contiguous()
    empty_bias = torch.empty(0, device=device, dtype=torch.float32)

    def ref():
        return grad_out_for_fp.matmul(dq_weight_for_fp)

    def w8a8():
        return matris_op.quant_linear_w8a8_static_cutlass(
            grad_out,
            bwd_q_weight,
            bwd_weight_scale,
            grad_scale,
            empty_bias,
            False,
        )

    ref_out = ref().float()
    w8a8_out = w8a8().float()
    _sync()
    diff = (w8a8_out - ref_out).abs()
    ref_abs = ref_out.abs().clamp_min(1.0e-12)

    ref_ms = _time_ms(ref, warmup, iters)
    w8a8_ms = _time_ms(w8a8, warmup, iters)

    return {
        "rows": rows,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "ref_ms": ref_ms,
        "w8a8_bwd_ms": w8a8_ms,
        "speedup_vs_ref": ref_ms / w8a8_ms if w8a8_ms > 0 else 0.0,
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "mean_rel_error": (diff / ref_abs).mean().item(),
        "grad_scale": grad_scale.item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="4096,8192,16384,65536")
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", choices=["fp32", "tf32"], default="fp32")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = args.dtype == "tf32"
    matris_op = _load_matris_op()
    if matris_op is None or not hasattr(matris_op, "quant_linear_w8a8_static_cutlass"):
        raise SystemExit("matris_op.quant_linear_w8a8_static_cutlass is unavailable")

    rows_list = [int(x.strip()) for x in args.rows.split(",") if x.strip()]
    results = [
        _bench_case(
            matris_op,
            rows=rows,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            warmup=args.warmup,
            iters=args.iters,
            dtype=torch.float32,
        )
        for rows in rows_list
    ]

    print("| rows | ref ms | W8A8 input-grad ms | speedup | max abs err | mean abs err |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['rows']} | {r['ref_ms']:.4f} | {r['w8a8_bwd_ms']:.4f} | "
            f"{r['speedup_vs_ref']:.3f}x | {r['max_abs_error']:.6e} | {r['mean_abs_error']:.6e} |"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": results, "env": dict(os.environ)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
