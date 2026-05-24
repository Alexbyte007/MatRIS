import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets import AseDBDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matris.applications.base import MatRISCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OMAT24 force-consistency checks: random perturbation + finite difference."
    )
    parser.add_argument(
        "--dataset-src",
        default="/home/lht/lab/omat24/val/rattled-relax",
        help="Path to OMAT24 split directory containing data.aselmdb.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="matris_10m_oam")
    parser.add_argument("--task", default="ef", choices=("ef", "efs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", default="fp32", choices=("fp32", "tf32", "bf16", "fp16"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260424)
    parser.add_argument("--fd-step-ang", type=float, default=1e-3)
    parser.add_argument("--perturb-std-ang", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_precision(device: str, precision_mode: str) -> None:
    tf32_enabled = precision_mode == "tf32"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled


def autocast_context(device: str, precision_mode: str):
    if device != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    if precision_mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision_mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def build_calculator(args: argparse.Namespace) -> MatRISCalculator:
    return MatRISCalculator(model=args.model, task=args.task, device=args.device)


def select_indices(dataset_len: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= dataset_len:
        return list(range(dataset_len))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_len), limit)
    indices.sort()
    return indices


def predict(atoms, calc: MatRISCalculator, args: argparse.Namespace) -> tuple[float, np.ndarray]:
    atoms = atoms.copy()
    atoms.calc = calc
    with autocast_context(args.device, args.precision_mode):
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
    return energy, forces


def finite_difference_check(atoms, calc: MatRISCalculator, args: argparse.Namespace, rng: random.Random) -> dict:
    atom_idx = rng.randrange(len(atoms))
    axis = rng.randrange(3)
    h = args.fd_step_ang

    plus = atoms.copy()
    minus = atoms.copy()
    pos_plus = plus.get_positions()
    pos_minus = minus.get_positions()
    pos_plus[atom_idx, axis] += h
    pos_minus[atom_idx, axis] -= h
    plus.set_positions(pos_plus)
    minus.set_positions(pos_minus)

    e_plus, _ = predict(plus, calc, args)
    e_minus, _ = predict(minus, calc, args)
    _, forces = predict(atoms, calc, args)

    fd_force = -(e_plus - e_minus) / (2.0 * h)
    direct_force = float(forces[atom_idx, axis])
    return {
        "fd_atom_index": atom_idx,
        "fd_axis": axis,
        "fd_force_eVA": fd_force,
        "direct_force_eVA": direct_force,
        "fd_abs_error_eVA": abs(fd_force - direct_force),
    }


def random_perturbation_check(atoms, calc: MatRISCalculator, args: argparse.Namespace, rng: np.random.Generator) -> dict:
    e0, f0 = predict(atoms, calc, args)

    perturbed = atoms.copy()
    disp = rng.normal(loc=0.0, scale=args.perturb_std_ang, size=perturbed.positions.shape)
    perturbed.set_positions(perturbed.positions + disp)
    e1, f1 = predict(perturbed, calc, args)

    return {
        "energy_change_eV": abs(e1 - e0),
        "force_change_mae_eVA": float(np.abs(f1 - f0).mean()),
        "force_change_max_eVA": float(np.linalg.norm(f1 - f0, axis=1).max()),
        "perturb_rms_ang": float(np.sqrt((disp ** 2).mean())),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_precision(args.device, args.precision_mode)
    calc = build_calculator(args)
    dataset = AseDBDataset(config={"src": args.dataset_src})
    indices = select_indices(len(dataset), args.limit, args.sample_seed)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    records = []
    for idx, sample_index in enumerate(indices, start=1):
        item = dataset[sample_index]
        atoms = dataset.get_atoms(sample_index)
        fd = finite_difference_check(atoms, calc, args, rng)
        pert = random_perturbation_check(atoms, calc, args, np_rng)
        record = {
            "sample_index": sample_index,
            "sid": item["sid"] if "sid" in item else "",
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            **fd,
            **pert,
        }
        records.append(record)
        print(
            f"[{idx}/{len(indices)}] sample_index={sample_index} formula={record['formula']} "
            f"fd_abs_error={record['fd_abs_error_eVA']:.6e} eV/A "
            f"force_change_mae={record['force_change_mae_eVA']:.6e} eV/A"
        )

    summary = {
        "num_structures": len(records),
        "dataset_src": str(Path(args.dataset_src).resolve()),
        "dataset_size": len(dataset),
        "sample_seed": args.sample_seed,
        "fd_abs_error_eVA_mean": float(np.mean([r["fd_abs_error_eVA"] for r in records])) if records else 0.0,
        "fd_abs_error_eVA_max": float(np.max([r["fd_abs_error_eVA"] for r in records])) if records else 0.0,
        "force_change_mae_eVA_mean": float(np.mean([r["force_change_mae_eVA"] for r in records])) if records else 0.0,
        "force_change_max_eVA_max": float(np.max([r["force_change_max_eVA"] for r in records])) if records else 0.0,
    }

    with open(output_dir / "force_consistency_records.jsonl", "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "force_consistency_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6e}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
