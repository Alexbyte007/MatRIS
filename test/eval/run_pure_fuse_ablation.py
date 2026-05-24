from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"

PURE_FUSE_QUANT_MODE = "p71_latency_pruned_fusion_only"
FULL_FUSION_MODE = "p28_p26_all_ffn_mlp_input_grad_only"

KNOWN_MATRIS_ENV_KEYS = (
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY",
    "MATRIS_P29_MLP_BWD_KERNEL",
    "MATRIS_W8A8_BACKEND",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE",
)

OPTIMAL_ENV_FLAGS = {
    "MATRIS_P26_TAIL_INPUT_GRAD_ONLY": "1",
    "MATRIS_P28_MLP_INPUT_GRAD_ONLY": "1",
    "MATRIS_P29_MLP_BWD_KERNEL": "1",
    "MATRIS_USE_CUDA_FUSED_LINE_ATTENTION": "1",
    "MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION": "1",
    "MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE": "1",
}

LEGACY_P71_ENV_FLAGS = {
    **OPTIMAL_ENV_FLAGS,
    "MATRIS_W8A8_BACKEND": "cuda_wmma_tail_n128_parallel",
    "MATRIS_W8A8_DISABLE_FAST_WRAPPER": "1",
}

GATED_MLP_GROUPS = (
    "attn_atom_node",
    "attn_atom_edge",
    "attn_line_node",
    "attn_line_edge",
    "refine_atom_edge",
    "refine_line_edge",
)

MLP_GROUPS = (
    "refine_line_node_ffn",
    "refine_line_edge_ffn",
    "refine_atom_node_ffn",
    "refine_atom_edge_ffn",
)


@dataclass(frozen=True)
class Experiment:
    idx: int
    label: str
    quant_mode: str
    fusion_mode: str
    category: str
    env_flags: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def env_without(*keys: str) -> dict[str, str]:
    flags = dict(OPTIMAL_ENV_FLAGS)
    for key in keys:
        flags.pop(key, None)
    return flags


def build_experiments() -> list[Experiment]:
    experiments = [
        Experiment(
            0,
            "fp32 baseline none/none",
            "none",
            "none",
            "anchor",
            {},
            "plain FP32 endpoint baseline",
        ),
        Experiment(
            1,
            "full pure fuse no W8A8 residual env",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "anchor",
            dict(OPTIMAL_ENV_FLAGS),
            "current pure fuse reference; W8A8 residual env removed",
        ),
        Experiment(
            2,
            "legacy p71 with W8A8 residual env",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "env_sanity",
            dict(LEGACY_P71_ENV_FLAGS),
            "legacy replay sanity only; not an optimal candidate",
        ),
        Experiment(
            3,
            "no P26 GatedMLP tail input-grad",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "env_leave_one_out",
            env_without("MATRIS_P26_TAIL_INPUT_GRAD_ONLY"),
        ),
        Experiment(
            4,
            "no P28 MLP input-grad",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "env_leave_one_out",
            env_without("MATRIS_P28_MLP_INPUT_GRAD_ONLY"),
        ),
        Experiment(
            5,
            "no P29 MLP backward kernel",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "env_leave_one_out",
            env_without("MATRIS_P29_MLP_BWD_KERNEL"),
        ),
        Experiment(
            6,
            "no CUDA fused line attention",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "env_leave_one_out",
            env_without("MATRIS_USE_CUDA_FUSED_LINE_ATTENTION"),
        ),
        Experiment(
            7,
            "no CUDA fused atom attention",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "env_leave_one_out",
            env_without("MATRIS_USE_CUDA_FUSED_ATOM_ATTENTION"),
        ),
        Experiment(
            8,
            "no CUDA directed2undirected average",
            PURE_FUSE_QUANT_MODE,
            FULL_FUSION_MODE,
            "env_leave_one_out",
            env_without("MATRIS_USE_CUDA_DIRECTED2UNDIRECTED_AVERAGE"),
        ),
        Experiment(
            9,
            "no module fusion but full env",
            PURE_FUSE_QUANT_MODE,
            "none",
            "fusion_total",
            dict(OPTIMAL_ENV_FLAGS),
            "isolates total module-replacement contribution",
        ),
    ]

    idx = len(experiments)
    for group in GATED_MLP_GROUPS:
        experiments.append(
            Experiment(
                idx,
                f"no GatedMLP group {group}",
                PURE_FUSE_QUANT_MODE,
                f"{FULL_FUSION_MODE}_no_gmlp_{group}",
                "gated_mlp_leave_one_out",
                dict(OPTIMAL_ENV_FLAGS),
            )
        )
        idx += 1

    for group in MLP_GROUPS:
        experiments.append(
            Experiment(
                idx,
                f"no MLP group {group}",
                PURE_FUSE_QUANT_MODE,
                f"{FULL_FUSION_MODE}_no_mlp_{group}",
                "mlp_leave_one_out",
                dict(OPTIMAL_ENV_FLAGS),
            )
        )
        idx += 1

    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure FP32 fuse endpoint ablations.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", default="fp32")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--activation-calibration-limit", type=int, default=64)
    parser.add_argument("--activation-calibration-seed", type=int, default=43)
    parser.add_argument("--cuda-visible-devices", default="4")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_ROOT / "pure_fuse_ablation_500_20260524"),
    )
    parser.add_argument(
        "--summary-path",
        default=str(RESULTS_ROOT / "pure_fuse_ablation_500_20260524_summary.md"),
    )
    parser.add_argument("--force", action="store_true", help="Re-run experiments even if summary.json exists.")
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    return (
        label.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "plus")
    )


def run_experiment(args: argparse.Namespace, experiment: Experiment, out_dir: Path) -> dict:
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in KNOWN_MATRIS_ENV_KEYS:
        env.pop(key, None)
    env.update(experiment.env_flags)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        args.python_bin,
        str(REPO_ROOT / "test" / "eval" / "infer_salex_lmdb_quant.py"),
        "--dataset-src",
        args.dataset_src,
        "--model",
        args.model,
        "--task",
        args.task,
        "--device",
        args.device,
        "--precision-mode",
        args.precision_mode,
        "--quant-mode",
        experiment.quant_mode,
        "--fusion-mode",
        experiment.fusion_mode,
        "--limit",
        str(args.limit),
        "--sample-seed",
        str(args.sample_seed),
        "--activation-calibration-limit",
        str(args.activation_calibration_limit),
        "--activation-calibration-seed",
        str(args.activation_calibration_seed),
        "--measure-time",
        "--output-json",
        str(summary_path),
    ]

    print(f"\n=== [{experiment.idx:02d}] {experiment.label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    elapsed_s = time.perf_counter() - start
    payload = json.loads(summary_path.read_text())
    payload.setdefault("ablation_metadata", {})
    payload["ablation_metadata"].update(
        {
            "idx": experiment.idx,
            "label": experiment.label,
            "category": experiment.category,
            "notes": experiment.notes,
            "env_flags": experiment.env_flags,
            "wall_time_s": elapsed_s,
        }
    )
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def row_for(experiment: Experiment, payload: dict, full_latency: float | None) -> dict:
    timing = payload.get("timing", {})
    metadata = payload.get("metadata", {})
    latency = float(timing.get("latency_ms_mean", 0.0))
    delta_ms = None if full_latency is None else latency - full_latency
    return {
        "idx": experiment.idx,
        "label": experiment.label,
        "category": experiment.category,
        "success": f"{metadata.get('num_success', 0)}/{metadata.get('sample_count', 0)}",
        "latency_ms_mean": latency,
        "structures_per_s": float(timing.get("throughput_structures_per_s", 0.0)),
        "delta_vs_full_ms": delta_ms,
        "speedup_vs_full": None if full_latency is None or latency == 0 else full_latency / latency,
        "quant_replaced": len(metadata.get("quant_replaced_modules", [])),
        "fused_modules": len(metadata.get("fused_modules", [])),
        "energy_mae_natoms": payload.get("res", {}).get("energy_mae_natoms", [None])[0],
        "force_mae": payload.get("res", {}).get("force_mae", [None])[0],
        "stress_mae": payload.get("res", {}).get("stress_mae", [None])[0],
    }


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_summary(summary_path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pure FP32 Fuse Ablation 500",
        "",
        f"- limit: `{args.limit}`",
        f"- sample_seed: `{args.sample_seed}`",
        f"- task: `{args.task}`",
        f"- dataset: `{args.dataset_src}`",
        f"- device: `CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}`",
        "",
        "Interpretation: `delta_vs_full_ms > 0` means removing that point made latency slower, so the removed point is likely helpful. `delta_vs_full_ms < 0` means removing it made latency faster, so the removed point is a possible negative optimization.",
        "",
        "| idx | experiment | category | success | latency ms | structures/s | delta vs full ms | full/this | fused modules |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["idx"]),
                    f"`{row['label']}`",
                    row["category"],
                    row["success"],
                    format_float(row["latency_ms_mean"]),
                    format_float(row["structures_per_s"]),
                    format_float(row["delta_vs_full_ms"]),
                    format_float(row["speedup_vs_full"]),
                    str(row["fused_modules"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Accuracy Snapshot",
            "",
            "| idx | experiment | energy_mae_natoms | force_mae | stress_mae |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["idx"]),
                    f"`{row['label']}`",
                    format_float(row["energy_mae_natoms"], 12),
                    format_float(row["force_mae"], 12),
                    format_float(row["stress_mae"], 12),
                ]
            )
            + " |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    experiments = build_experiments()
    payloads: dict[int, dict] = {}
    for experiment in experiments:
        out_dir = output_root / f"{experiment.idx:02d}_{sanitize_label(experiment.label)}" / "eval"
        payloads[experiment.idx] = run_experiment(args, experiment, out_dir)

    full_latency = payloads[1]["timing"]["latency_ms_mean"]
    rows = [row_for(experiment, payloads[experiment.idx], full_latency) for experiment in experiments]
    write_summary(Path(args.summary_path), rows, args)

    print(f"\nWrote summary: {args.summary_path}", flush=True)
    print("Top possible negative optimizations (latency faster after removal):", flush=True)
    for row in sorted(
        [r for r in rows if r["idx"] not in (0, 1) and r["delta_vs_full_ms"] is not None],
        key=lambda r: r["delta_vs_full_ms"],
    )[:5]:
        print(
            f"  [{row['idx']:02d}] {row['label']}: "
            f"latency={row['latency_ms_mean']:.6f}ms, "
            f"delta_vs_full={row['delta_vs_full_ms']:.6f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
