import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a candidate sAlex static-eval run against an FP32 baseline run."
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def record_key(row: dict) -> tuple[str, str, str]:
    if "sample_index" in row:
        return (str(row["sample_index"]), "", "")
    return (row.get("material_id", ""), row.get("structure_role", ""), row.get("cif_path", ""))


def build_comparison_rows(baseline_rows: list[dict], candidate_rows: list[dict]) -> list[dict]:
    base_map = {record_key(row): row for row in baseline_rows}
    cand_map = {record_key(row): row for row in candidate_rows}

    shared_keys = sorted(set(base_map) & set(cand_map))
    rows = []
    for key in shared_keys:
        base = base_map[key]
        cand = cand_map[key]

        n_atoms = int(base.get("n_atoms") or base.get("nsites") or 1)
        baseline_latency = float(base.get("latency_ms", 0.0))
        candidate_latency = float(cand.get("latency_ms", 0.0))
        row = {
            "sample_index": base.get("sample_index", ""),
            "sid": base.get("sid", ""),
            "formula": base.get("formula", base.get("formula_pretty", "")),
            "n_atoms": n_atoms,
            "latency_ms_baseline": baseline_latency,
            "latency_ms_candidate": candidate_latency,
            "latency_speedup": baseline_latency / candidate_latency if candidate_latency > 0 else None,
            "pred_energy_abs_delta_eV": abs(float(cand["pred_energy_eV"]) - float(base["pred_energy_eV"])),
        }

        row["pred_energy_abs_delta_meV_per_atom"] = (
            row["pred_energy_abs_delta_eV"] * 1000.0 / max(n_atoms, 1)
        )

        metric_keys = [
            "energy_abs_error_eV",
            "energy_abs_error_per_atom_eV",
            "force_mae_eVA",
            "force_rmse_eVA",
            "stress_mae_eVA3",
            "stress_rmse_eVA3",
            "peak_mem_mb",
        ]
        for metric_key in metric_keys:
            if metric_key in base and metric_key in cand:
                baseline_value = float(base[metric_key])
                candidate_value = float(cand[metric_key])
                row[f"{metric_key}_baseline"] = baseline_value
                row[f"{metric_key}_candidate"] = candidate_value
                row[f"{metric_key}_delta"] = candidate_value - baseline_value
                row[f"{metric_key}_ratio"] = (
                    candidate_value / baseline_value if baseline_value != 0 else None
                )

        rows.append(row)
    return rows


def n_atoms_bin(n_atoms: int) -> str:
    if n_atoms <= 4:
        return "small(<=4)"
    if n_atoms <= 16:
        return "medium(5-16)"
    if n_atoms <= 32:
        return "large(17-32)"
    return "xlarge(>32)"


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}

    baseline_latency = np.array([r["latency_ms_baseline"] for r in rows], dtype=float)
    candidate_latency = np.array([r["latency_ms_candidate"] for r in rows], dtype=float)
    out = {
        "num_structures": len(rows),
        "latency_ms_baseline_mean": float(baseline_latency.mean()),
        "latency_ms_candidate_mean": float(candidate_latency.mean()),
        "speedup_vs_baseline": float(baseline_latency.mean() / candidate_latency.mean())
        if candidate_latency.mean() > 0
        else None,
        "pred_energy_mae_delta_meV_per_atom": float(
            np.mean([r["pred_energy_abs_delta_meV_per_atom"] for r in rows])
        ),
    }

    for metric_key in [
        "energy_abs_error_per_atom_eV",
        "force_mae_eVA",
        "force_rmse_eVA",
        "stress_mae_eVA3",
        "stress_rmse_eVA3",
        "peak_mem_mb",
    ]:
        delta_key = f"{metric_key}_delta"
        ratio_key = f"{metric_key}_ratio"
        if delta_key in rows[0]:
            out[f"{metric_key}_delta_mean"] = float(np.mean([r[delta_key] for r in rows]))
            ratios = [r[ratio_key] for r in rows if r.get(ratio_key) is not None]
            out[f"{metric_key}_ratio_mean"] = float(np.mean(ratios)) if ratios else None

    return out


def summarize_by_group(rows: list[dict]) -> dict:
    group_maps = {
        "n_atoms_bin": defaultdict(list),
    }
    for row in rows:
        group_maps["n_atoms_bin"][n_atoms_bin(int(row["n_atoms"]))].append(row)

    summary = {}
    for group_name, mapping in group_maps.items():
        summary[group_name] = {name: summarize_rows(group_rows) for name, group_rows in sorted(mapping.items())}
    return summary


def write_markdown_report(path: Path, baseline_cfg: dict, candidate_cfg: dict, overall: dict) -> None:
    lines = [
        "# Comparison Report",
        "",
        f"- baseline_dir: `{path.parent.parent / baseline_cfg.get('output_dir', '')}`" if baseline_cfg.get("output_dir") else "",
        f"- candidate_dir: `{path.parent.parent / candidate_cfg.get('output_dir', '')}`" if candidate_cfg.get("output_dir") else "",
        f"- baseline_precision: `{baseline_cfg.get('precision_mode', 'unknown')}`",
        f"- candidate_precision: `{candidate_cfg.get('precision_mode', 'unknown')}`",
        f"- candidate_compile: `{candidate_cfg.get('compile', False)}`",
        "",
        "## Overall",
        "",
    ]
    for key, value in overall.items():
        if isinstance(value, float):
            lines.append(f"- {key}: `{value:.6f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(line for line in lines if line != ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = load_json(baseline_dir / "run_config.json")
    candidate_cfg = load_json(candidate_dir / "run_config.json")
    baseline_rows = load_jsonl(baseline_dir / "per_structure_predictions.jsonl")
    candidate_rows = load_jsonl(candidate_dir / "per_structure_predictions.jsonl")

    comparison_rows = build_comparison_rows(baseline_rows, candidate_rows)
    overall = summarize_rows(comparison_rows)
    by_group = summarize_by_group(comparison_rows)

    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as fp:
        json.dump(overall, fp, ensure_ascii=False, indent=2)
    with open(output_dir / "comparison_by_group.json", "w", encoding="utf-8") as fp:
        json.dump(by_group, fp, ensure_ascii=False, indent=2)
    with open(output_dir / "comparison_rows.jsonl", "w", encoding="utf-8") as fp:
        for row in comparison_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_markdown_report(output_dir / "comparison_report.md", baseline_cfg, candidate_cfg, overall)

    print("=== Overall Comparison ===")
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
