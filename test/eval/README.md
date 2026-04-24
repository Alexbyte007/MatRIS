# Test Eval

本目录用于 MatRIS 的静态性能评测、物理一致性检查和 WBM discovery 评测。

当前主线静态测试数据源为 OMAT24 `rattled-relax` validation：

```text
/home/lht/lab/omat24/val/rattled-relax
```

原先基于本地 `mp_testset` / CIF metadata 的静态评测入口已移除，避免后续误用。OMAT24 静态测试用于量化优化阶段的快速回归；WBM 用于后续任务级 discovery 验证。

## OMAT24 静态评测

`evaluate_omat24_static_metrics.py`
- 用途：从 OMAT24 ASE LMDB 中固定抽样，逐结构运行 MatRIS 单点 E/F/S 预测。
- 默认数据：`rattled-relax` validation。
- 默认规模：`limit=500`，`sample_seed=20260424`。
- 主要输出：
  - `run_config.json`
  - `sample_indices.json`
  - `per_structure_predictions.jsonl`
  - `summary_overall.json`
- 主要指标：
  - `latency_ms_mean`
  - `throughput_structures_per_s`
  - `throughput_atoms_per_s`
  - `energy_mae_per_atom`
  - `force_mae_eVA`
  - `stress_mae_eVA3`
  - `peak_mem_mb_max`

启动脚本：

```bash
bash test/eval/sh/omat24/run_omat24_static_eval.sh
```

结果默认写入：

```text
/home/lht/lab/MatRIS/results/static_eval_omat24/fp32_baseline
```

后续量化版本应使用相同 `sample_seed`，或复用同一份 `sample_indices.json` 对齐样本。

## OMAT24 结果比较

`compare_eval_runs.py`
- 用途：比较 FP32 baseline 与候选版本的 OMAT24 静态评测结果。
- 对齐方式：优先按 `sample_index` 对齐。
- 主要比较：
  - latency speedup
  - predicted energy delta
  - E/F/S 误差相对 FP32 的变化
  - peak memory 变化
  - 按 `n_atoms` 分桶汇总

启动脚本：

```bash
BASELINE_NAME=fp32_baseline CANDIDATE_NAME=candidate \
bash test/eval/sh/omat24/run_compare_eval.sh
```

默认读取：

```text
/home/lht/lab/MatRIS/results/static_eval_omat24/${RUN_NAME}
```

默认输出：

```text
/home/lht/lab/MatRIS/results/comparisons_omat24/
```

## OMAT24 Pipeline Profile

`profile_omat24_pipeline.py`
- 用途：在 OMAT24 固定样本上拆分 MatRIS pipeline 各阶段耗时。
- 默认数据：`rattled-relax` validation。
- 默认规模：`limit=50`，`warmup_steps=5`，`sample_seed=20260424`。
- 主要阶段：
  - dataset get item / get atoms
  - ASE Atoms -> pymatgen Structure
  - graph converter
  - graph to device
  - process_graphs
  - embedding
  - interaction blocks
  - energy head
  - force autograd
  - stress autograd
  - cpu output
- 主要输出：
  - `run_config.json`
  - `sample_indices.json`
  - `pipeline_profile_records.jsonl`
  - `pipeline_profile_summary.json`
- 当前只统计到 pipeline stage 层级，`interaction blocks` 作为整体耗时记录。

启动脚本：

```bash
bash test/eval/sh/omat24/run_omat24_pipeline_profile.sh
```

默认输出：

```text
/home/lht/lab/MatRIS/results/pipeline_profile_omat24/fp32_pipeline_profile
```

## OMAT24 平滑性/力一致性检查

`check_force_consistency.py`
- 用途：从 OMAT24 中抽样，做轻量物理一致性检查。
- 当前包含：
  - 随机扰动测试
  - 有限差分 `F ≈ -dE/dR`
- 主要输出：
  - `force_consistency_records.jsonl`
  - `force_consistency_summary.json`

启动脚本：

```bash
bash test/eval/sh/omat24/run_force_consistency.sh
```

默认输出：

```text
/home/lht/lab/MatRIS/results/consistency_omat24/fp32_baseline
```

## WBM Discovery

`run_wbm_eval.py`
- 用途：对 WBM initial structures 做 relax，输出 relaxed energy。
- 当前抽样：从 `unique_prototype` 中按 full WBM 稳定比例抽 representative subset。
- 默认规模：500。

`join_wbm_predictions_with_summary.py`
- 用途：把 MatRIS relaxed energy 结果与 WBM summary 对齐，并转换 formation energy per atom。

`calc_wbm_discovery_metrics.py`
- 用途：计算 WBM discovery 指标。
- 当前输出包括：
  - MAE / RMSE / R2
  - TP / FP / TN / FN
  - Precision
  - Recall / TPR
  - TNR
  - Accuracy
  - F1
  - DAF

启动顺序：

```bash
bash test/eval/sh/wbm/run_wbm_eval.sh
bash test/eval/sh/wbm/join_wbm_predictions_with_summary.sh
bash test/eval/sh/wbm/calc_wbm_discovery_metrics.sh
```
