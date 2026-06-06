import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.model.functions import _load_matris_op


class TargetAttentionSavedAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, values, lengths):
        matris_op = _load_matris_op()
        out, alpha = matris_op.target_attention_sum_forward(
            logits.contiguous(),
            values.contiguous(),
            lengths.contiguous(),
        )
        ctx.save_for_backward(values.contiguous(), out, alpha, lengths.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad_out):
        values, out, alpha, lengths = ctx.saved_tensors
        matris_op = _load_matris_op()
        grad_logits, grad_values = matris_op.target_attention_sum_backward(
            grad_out.contiguous(),
            values,
            out,
            alpha,
            lengths,
        )
        return grad_logits, grad_values, None


class TargetAttentionRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, values, lengths):
        matris_op = _load_matris_op()
        out = matris_op.target_attention_sum_forward_no_alpha(
            logits.contiguous(),
            values.contiguous(),
            lengths.contiguous(),
        )
        ctx.save_for_backward(logits.contiguous(), values.contiguous(), out, lengths.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad_out):
        logits, values, out, lengths = ctx.saved_tensors
        matris_op = _load_matris_op()
        grad_logits, grad_values = matris_op.target_attention_sum_backward_recompute(
            grad_out.contiguous(),
            logits,
            values,
            out,
            lengths,
        )
        return grad_logits, grad_values, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark saved-alpha vs recompute-alpha target attention CUDA kernels."
    )
    parser.add_argument("--rows", type=int, default=262144)
    parser.add_argument("--segments", type=int, default=8192)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def make_lengths(args, device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    lengths = torch.randint(
        args.min_len,
        args.max_len + 1,
        (args.segments,),
        generator=gen,
        dtype=torch.int64,
    )
    scale = float(args.rows) / float(lengths.sum().item())
    lengths = torch.clamp((lengths.float() * scale).round().to(torch.int64), min=1)
    diff = int(args.rows - lengths.sum().item())
    idx = 0
    while diff != 0:
        pos = idx % args.segments
        if diff > 0:
            lengths[pos] += 1
            diff -= 1
        elif lengths[pos] > 1:
            lengths[pos] -= 1
            diff += 1
        idx += 1
    return lengths.to(device=device)


def run_once(fn, logits, values, lengths, grad_out):
    logits_i = logits.detach().clone().requires_grad_(True)
    values_i = values.detach().clone().requires_grad_(True)
    out = fn.apply(logits_i, values_i, lengths)
    grad_logits, grad_values = torch.autograd.grad(
        out,
        (logits_i, values_i),
        grad_out,
        retain_graph=False,
        create_graph=False,
    )
    return out, grad_logits, grad_values


def time_fn(fn, logits, values, lengths, grad_out, warmup, iters):
    for _ in range(warmup):
        run_once(fn, logits, values, lengths, grad_out)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        run_once(fn, logits, values, lengths, grad_out)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / float(iters)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return elapsed_ms, peak_mb


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    matris_op = _load_matris_op()
    required = (
        "target_attention_sum_forward",
        "target_attention_sum_backward",
        "target_attention_sum_forward_no_alpha",
        "target_attention_sum_backward_recompute",
    )
    missing = [name for name in required if not hasattr(matris_op, name)]
    if missing:
        raise RuntimeError(f"matris_op is missing {missing}; rebuild the extension first")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    lengths = make_lengths(args, device)
    rows = int(lengths.sum().item())
    logits = torch.randn((rows, 128), device=device, dtype=torch.float32)
    values = torch.randn_like(logits)
    grad_out = torch.randn((args.segments, 128), device=device, dtype=torch.float32)

    ref_out, ref_grad_logits, ref_grad_values = run_once(
        TargetAttentionSavedAlpha,
        logits,
        values,
        lengths,
        grad_out,
    )
    cand_out, cand_grad_logits, cand_grad_values = run_once(
        TargetAttentionRecompute,
        logits,
        values,
        lengths,
        grad_out,
    )
    torch.cuda.synchronize()

    correctness = {
        "out_max_abs": float((ref_out - cand_out).abs().max().item()),
        "grad_logits_max_abs": float((ref_grad_logits - cand_grad_logits).abs().max().item()),
        "grad_values_max_abs": float((ref_grad_values - cand_grad_values).abs().max().item()),
        "out_mean_abs": float((ref_out - cand_out).abs().mean().item()),
        "grad_logits_mean_abs": float((ref_grad_logits - cand_grad_logits).abs().mean().item()),
        "grad_values_mean_abs": float((ref_grad_values - cand_grad_values).abs().mean().item()),
    }

    saved_alpha_ms, saved_alpha_peak = time_fn(
        TargetAttentionSavedAlpha,
        logits,
        values,
        lengths,
        grad_out,
        args.warmup,
        args.iters,
    )
    recompute_ms, recompute_peak = time_fn(
        TargetAttentionRecompute,
        logits,
        values,
        lengths,
        grad_out,
        args.warmup,
        args.iters,
    )

    bytes_per_edge_tensor = rows * 128 * 4
    result = {
        "rows": rows,
        "segments": args.segments,
        "min_len": int(lengths.min().item()),
        "max_len": int(lengths.max().item()),
        "mean_len": float(lengths.float().mean().item()),
        "correctness": correctness,
        "saved_alpha_ms": saved_alpha_ms,
        "recompute_ms": recompute_ms,
        "speedup_vs_saved_alpha": saved_alpha_ms / recompute_ms,
        "saved_alpha_peak_mb": saved_alpha_peak,
        "recompute_peak_mb": recompute_peak,
        "one_edge_feature_tensor_mb": bytes_per_edge_tensor / (1024**2),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
