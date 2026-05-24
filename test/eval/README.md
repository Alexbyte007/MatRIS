# Test Eval

本目录用于 MatRIS 的静态性能评测和 WBM discovery 评测。

当前主线静态测试数据源为 sAlex validation：

```text
/home/lht/lab/sAlex/val
```

原先基于本地 `mp_testset` / CIF metadata 的静态评测入口已移除，避免后续误用。sAlex 静态测试用于量化优化阶段的快速回归；WBM 用于后续任务级 discovery 验证。

## sAlex 数据准备

官方 sAlex 包含 `train` 和 `val` 两个 split。当前评测默认使用 `val`：

```bash
bash test/eval/sh/salex/download_salex.sh
```

如需下载训练 split：

```bash
SPLIT=train bash test/eval/sh/salex/download_salex.sh
```

## sAlex 静态评测

`infer_salex_lmdb_quant.py`
- 用途：对齐组内 `infer_lmdb` 脚本口径，在 sAlex val 上固定抽样 500 个结构，输出 E/F/S MAE/RMSE。
- 默认口径：`sample_seed=42`，shuffle 后取前 `limit=500`，`task=efsm`。
- 核心输出字段保持组内脚本一致：
  - `energy_mae`
  - `energy_rmse`
  - `energy_mae_natoms`
  - `energy_rmse_natoms`
  - `force_mae`
  - `force_rmse`
  - `stress_mae`
  - `stress_rmse`
- 可选补充：`--quant-mode`、`--precision-mode`、`--measure-time`、`--output-json`、`--save-predictions`。

启动脚本：

```bash
bash test/eval/sh/salex/run_infer_salex_lmdb_quant.sh
```

量化版本示例：

```bash
QUANT_MODE=node_ffn_w8a32 RUN_NAME=node_ffn_w8a32 \
bash test/eval/sh/salex/run_infer_salex_lmdb_quant.sh
```

比较组内对齐口径结果：

```bash
BASELINE_NAME=fp32_baseline CANDIDATE_NAME=node_ffn_w8a32 \
bash test/eval/sh/salex/run_compare_lmdb_quant.sh
```

按论文 E/F/S 误差指标检查 fake quant 稳定组合：

```bash
bash test/eval/sh/salex/run_paper_stable_fake_quant_eval.sh
```

默认测试两个 static PASS 候选：

```text
p0_line_gate_paper_stable_w8a32
all_single_pass_w8a32
```

`p0_line_gate_paper_stable_w8a32` 即：

```text
p0_line_gate_stable_w8a32
+ interaction_block.*.attn_block_atom_graph.edge_nonlinear_update.mlp_core
```

`all_single_pass_w8a32` 覆盖当前清单中所有单项 static PASS 模块的组合。

这个检查只看 `energy_*`、`force_*`、`stress_*` 的 MAE/RMSE ratio，默认要求每项 `candidate / baseline <= 1.005`。可用 `MAX_RATIO=...` 调整阈值。

一键跑 P1 单模块和组合 static eval：

```bash
bash test/eval/sh/salex/run_p1_static_eval.sh
```

如只想跑某几个 P1 模式：

```bash
P1_MODES="atom_edge_mlp_w8a32 atom_node_mlp_w8a32" \
bash test/eval/sh/salex/run_p1_static_eval.sh
```

`evaluate_salex_static_metrics.py`
- 用途：从 sAlex ASE LMDB 中固定抽样，逐结构运行 MatRIS 单点 E/F/S 预测。
- 默认数据：sAlex `val` split。
- 默认规模：`limit=500`，`sample_seed=42`。
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
bash test/eval/sh/salex/run_salex_static_eval.sh
```

结果默认写入：

```text
/home/lht/lab/MatRIS/results/static_eval_salex/fp32_baseline
```

后续量化版本应使用相同 `sample_seed`，或复用同一份 `sample_indices.json` 对齐样本。

## sAlex 结果比较

`compare_eval_runs.py`
- 用途：比较 FP32 baseline 与候选版本的 sAlex 静态评测结果。
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
bash test/eval/sh/salex/run_compare_eval.sh
```

默认读取：

```text
/home/lht/lab/MatRIS/results/static_eval_salex/${RUN_NAME}
```

默认输出：

```text
/home/lht/lab/MatRIS/results/comparisons_salex/
```

## sAlex Pipeline Profile

`profile_salex_pipeline.py`
- 用途：在 sAlex 固定样本上拆分 MatRIS pipeline 各阶段耗时。
- 默认数据：sAlex `val` split。
- 默认规模：`limit=50`，`warmup_steps=5`，`sample_seed=42`。
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
bash test/eval/sh/salex/run_salex_pipeline_profile.sh
```

默认输出：

```text
/home/lht/lab/MatRIS/results/pipeline_profile_salex/fp32_pipeline_profile
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
