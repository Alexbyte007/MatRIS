import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_INFER_SCRIPT = REPO_ROOT / "test" / "eval" / "infer_salex_lmdb_quant.py"
_INFER_SPEC = importlib.util.spec_from_file_location("infer_salex_lmdb_quant", _INFER_SCRIPT)
if _INFER_SPEC is None or _INFER_SPEC.loader is None:
    raise RuntimeError(f"Cannot load {_INFER_SCRIPT}")
_INFER_MOD = importlib.util.module_from_spec(_INFER_SPEC)
_INFER_SPEC.loader.exec_module(_INFER_MOD)

build_calculator = _INFER_MOD.build_calculator
configure_precision = _INFER_MOD.configure_precision
load_atoms_with_labels = _INFER_MOD.load_atoms_with_labels
parse_supercell_repeat = _INFER_MOD.parse_supercell_repeat
select_group_aligned_keys = _INFER_MOD.select_group_aligned_keys
summarize_timing = _INFER_MOD.summarize_timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe torch.compile scopes on the MatRIS large-graph endpoint path."
    )
    parser.add_argument("--dataset-src", default="/home/lht/lab/sAlex/val")
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="efsm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", default="fp32")
    parser.add_argument("--quant-mode", default="none")
    parser.add_argument("--fusion-mode", default="none")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--graph-ids",
        default="",
        help="Optional comma-separated graph ids. When set, overrides limit/sample-seed selection.",
    )
    parser.add_argument("--large-supercell-repeat", default="6,3,2")
    parser.add_argument("--large-supercell-max-atoms", type=int, default=3100)
    parser.add_argument(
        "--compile-scope",
        default="none",
        choices=["none", "full_model", "interaction_blocks", "force_stress_head"],
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def apply_compile_scope(calculator, args: argparse.Namespace) -> dict:
    if args.compile_scope == "none":
        return {"enabled": False, "scope": "none"}
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")

    compile_kwargs = {
        "dynamic": True,
        "mode": args.compile_mode,
        "fullgraph": args.compile_fullgraph,
    }
    model = calculator.model
    if args.compile_scope == "full_model":
        calculator.model = torch.compile(model, **compile_kwargs)
    elif args.compile_scope == "interaction_blocks":
        for idx, block in enumerate(model.interaction_block):
            model.interaction_block[idx] = torch.compile(block, **compile_kwargs)
    elif args.compile_scope == "force_stress_head":
        model.force_stress_head = torch.compile(model.force_stress_head, **compile_kwargs)
    else:
        raise ValueError(f"unsupported compile scope: {args.compile_scope}")
    return {
        "enabled": True,
        "scope": args.compile_scope,
        "mode": args.compile_mode,
        "dynamic": True,
        "fullgraph": args.compile_fullgraph,
    }


def run_one(calculator, structures, graph_id: int, repeat, max_atoms: int) -> tuple[float, int]:
    atom, *_ = load_atoms_with_labels(structures, graph_id, repeat, max_atoms)
    calculator.reset()
    atom.calc = calculator
    if calculator.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = atom.get_potential_energy()
    _ = atom.get_forces()
    _ = atom.get_stress()
    if calculator.device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, len(atom)


def main() -> None:
    args = parse_args()
    configure_precision(args.device, args.precision_mode)
    repeat = parse_supercell_repeat(args.large_supercell_repeat)
    structures = AseDBDataset(config={"src": args.dataset_src})
    if args.graph_ids.strip():
        keys = np.array([int(item.strip()) for item in args.graph_ids.split(",") if item.strip()])
    else:
        keys = select_group_aligned_keys(len(structures), args.limit, args.sample_seed)
    calculator = build_calculator(args)
    compile_info = apply_compile_scope(calculator, args)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    warmup_keys = keys[: max(0, min(args.warmup_count, len(keys)))]
    measured_keys = keys[max(0, min(args.warmup_count, len(keys))) :]

    warmup_records = []
    for idx, graph_id in enumerate(warmup_keys):
        try:
            latency_ms, n_atoms = run_one(
                calculator,
                structures,
                int(graph_id),
                repeat,
                args.large_supercell_max_atoms,
            )
            warmup_records.append(
                {
                    "sample_order": idx,
                    "graph_id": int(graph_id),
                    "n_atoms": n_atoms,
                    "latency_ms": latency_ms,
                }
            )
        except Exception as exc:
            warmup_records.append(
                {
                    "sample_order": idx,
                    "graph_id": int(graph_id),
                    "error": str(exc),
                }
            )

    latencies_ms = []
    n_atoms = []
    records = []
    for offset, graph_id in enumerate(measured_keys, start=len(warmup_keys)):
        try:
            latency_ms, atom_count = run_one(
                calculator,
                structures,
                int(graph_id),
                repeat,
                args.large_supercell_max_atoms,
            )
            latencies_ms.append(latency_ms)
            n_atoms.append(atom_count)
            records.append(
                {
                    "sample_order": offset,
                    "graph_id": int(graph_id),
                    "n_atoms": atom_count,
                    "latency_ms": latency_ms,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "sample_order": offset,
                    "graph_id": int(graph_id),
                    "error": str(exc),
                }
            )

    metadata = {
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "sample_count": len(keys),
        "num_success": len(latencies_ms),
        "num_warmup": len(warmup_records),
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "precision_mode": args.precision_mode,
        "quant_mode": args.quant_mode,
        "fusion_mode": args.fusion_mode,
        "large_supercell_repeat": list(repeat),
        "large_supercell_repeat_factor": int(np.prod(repeat)),
        "large_supercell_max_atoms": args.large_supercell_max_atoms,
        "compile": compile_info,
    }
    if args.device == "cuda":
        metadata["peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)

    payload = {
        "metadata": metadata,
        "warmup_records": warmup_records,
        "records": records,
        "timing": summarize_timing(latencies_ms, n_atoms),
        "top_slowest": sorted(
            [record for record in records if "latency_ms" in record],
            key=lambda item: item["latency_ms"],
            reverse=True,
        )[:10],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "compile_scope": args.compile_scope,
        "num_success": metadata["num_success"],
        "timing": payload["timing"],
        "peak_mem_mb": metadata.get("peak_mem_mb"),
    }, indent=2))


if __name__ == "__main__":
    main()
