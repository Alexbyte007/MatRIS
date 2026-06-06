import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from fairchem.core.datasets import AseDBDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INFER_SCRIPT = REPO_ROOT / "test" / "eval" / "infer_salex_lmdb_quant.py"
SPEC = importlib.util.spec_from_file_location("infer_salex_lmdb_quant", INFER_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {INFER_SCRIPT}")
INFER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFER)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile P201 alpha+attention recompute boundary.")
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", default="fp32")
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--graph-id", type=int, default=355248)
    parser.add_argument("--large-supercell-repeat", default="6,3,2")
    parser.add_argument("--large-supercell-max-atoms", type=int, default=3100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-table", default="")
    parser.add_argument("--chrome-trace", default="")
    return parser.parse_args()


def cuda_time_us(event) -> float:
    return float(
        getattr(event, "self_device_time_total", 0.0)
        or getattr(event, "self_cuda_time_total", 0.0)
        or 0.0
    )


def cuda_total_us(event) -> float:
    return float(
        getattr(event, "device_time_total", 0.0)
        or getattr(event, "cuda_time_total", 0.0)
        or 0.0
    )


def memory_bytes(event) -> int:
    return int(
        getattr(event, "self_cuda_memory_usage", 0)
        or getattr(event, "cuda_memory_usage", 0)
        or getattr(event, "self_device_memory_usage", 0)
        or getattr(event, "device_memory_usage", 0)
        or 0
    )


def event_row(event):
    return {
        "name": event.key,
        "count": int(event.count),
        "self_cuda_ms": cuda_time_us(event) / 1000.0,
        "cuda_total_ms": cuda_total_us(event) / 1000.0,
        "self_device_memory_mb": memory_bytes(event) / (1024**2),
    }


def run_endpoint(calculator, atom, args):
    calculator.reset()
    atom.calc = calculator
    if args.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    energy = atom.get_potential_energy()
    forces = atom.get_forces() if "f" in args.task else None
    stress = atom.get_stress() if "s" in args.task else None
    if args.device == "cuda":
        torch.cuda.synchronize()
    return {
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "energy": float(energy),
        "force_shape": list(forces.shape) if forces is not None else None,
        "stress_shape": list(stress.shape) if stress is not None else None,
    }


def main():
    args = parse_args()
    os.environ["MATRIS_P201_ALPHA_ATTENTION_RECOMPUTE"] = "1"
    os.environ["MATRIS_P201_RECORD_FUNCTION"] = "1"
    INFER.configure_precision(args.device, args.precision_mode)
    repeat = INFER.parse_supercell_repeat(args.large_supercell_repeat)
    structures = AseDBDataset(config={"src": args.dataset_src})
    calculator = INFER.build_calculator(args)
    atom, *_ = INFER.load_atoms_with_labels(
        structures,
        args.graph_id,
        repeat,
        args.large_supercell_max_atoms,
    )

    for _ in range(max(0, args.warmup)):
        run_endpoint(calculator, atom, args)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function(f"P201.profile.graph_{args.graph_id}"):
            record = run_endpoint(calculator, atom, args)

    if args.device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mb = 0.0

    events = [event_row(event) for event in prof.key_averages()]
    p201_events = sorted(
        [row for row in events if row["name"].startswith("P201.")],
        key=lambda row: row["cuda_total_ms"],
        reverse=True,
    )
    kernel_events = sorted(
        [
            row
            for row in events
            if "kernel" in row["name"].lower()
            or "sgemm" in row["name"].lower()
            or "gemm" in row["name"].lower()
            or "matmul" in row["name"].lower()
            or "fused_line_attention" in row["name"]
        ],
        key=lambda row: row["self_cuda_ms"],
        reverse=True,
    )
    payload = {
        "metadata": {
            "dataset_src": str(Path(args.dataset_src).resolve()),
            "graph_id": args.graph_id,
            "n_atoms": len(atom),
            "large_supercell_repeat": list(repeat),
            "large_supercell_max_atoms": args.large_supercell_max_atoms,
            "precision_mode": args.precision_mode,
            "task": args.task,
            "peak_mem_mb": peak_mb,
        },
        "record": record,
        "p201_events": p201_events,
        "top_kernel_events": kernel_events[:30],
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.chrome_trace:
        prof.export_chrome_trace(args.chrome_trace)

    lines = [
        f"# P201 profile graph_id={args.graph_id}",
        "",
        f"- n_atoms: {len(atom)}",
        f"- latency_ms: {record['latency_ms']:.3f}",
        f"- peak_mem_mb: {peak_mb:.3f}",
        "",
        "## P201 Ranges",
        "",
        "| name | count | cuda total ms | self cuda ms | self mem MB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in p201_events:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {row['cuda_total_ms']:.3f} | "
            f"{row['self_cuda_ms']:.3f} | {row['self_device_memory_mb']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top Kernel Events",
            "",
            "| name | count | self cuda ms | cuda total ms | self mem MB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in kernel_events[:20]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {row['self_cuda_ms']:.3f} | "
            f"{row['cuda_total_ms']:.3f} | {row['self_device_memory_mb']:.3f} |"
        )
    table_text = "\n".join(lines) + "\n"
    if args.output_table:
        table = Path(args.output_table)
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text(table_text, encoding="utf-8")
    print(table_text)


if __name__ == "__main__":
    main()
