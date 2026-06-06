# MatRIS 大图优化任务清单

## 目标

从大原子数 / 大图场景重新评估 MatRIS 推理优化路线。小中图最佳 fuse 不直接作为大图默认路径，先用不开任何优化的 FP32 baseline 找到 50 样本下的可运行结构规模上限，再逐项做大图专用消融和优化。

## 固定测试口径

- 数据集：`/home/lht/lab/sAlex/val`
- 模型：`matris_10m_oam`
- 任务：`efsm`
- 精度：`fp32`
- 样本数：`--limit 50`
- 采样：`--sample-seed 42`
- 脚本：`test/eval/infer_salex_lmdb_quant.py`
- 结构放大：使用 `--large-supercell-repeat a,b,c`
- 安全上限：使用 `--large-supercell-max-atoms`
- 计时：`--measure-time`

## Baseline 定义

不开任何 fuse / quant / macro 优化：

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_HOME=/home/lht/miniconda3/envs/matris311 \
PATH=/home/lht/miniconda3/envs/matris311/bin:$PATH \
LD_LIBRARY_PATH=/home/lht/miniconda3/envs/matris311/lib:$LD_LIBRARY_PATH \
/home/lht/miniconda3/envs/matris311/bin/python test/eval/infer_salex_lmdb_quant.py \
  --dataset-src /home/lht/lab/sAlex/val \
  --model matris_10m_oam \
  --task efsm \
  --device cuda \
  --precision-mode fp32 \
  --quant-mode none \
  --fusion-mode none \
  --limit 50 \
  --sample-seed 42 \
  --activation-calibration-limit 0 \
  --measure-time \
  --large-supercell-repeat <repeat> \
  --large-supercell-max-atoms <max_atoms> \
  --output-json <output_dir>/summary.json \
  --save-predictions <output_dir>/predictions.jsonl
```

运行 baseline 时不设置任何 `MATRIS_*` 优化环境变量。

## 已知背景

- 50 样本原始原子数分布：`min 2 / mean 10.7 / max 80`。
- 小中图最佳 fuse 在 1280 atoms 档并不是大图最佳：
  - baseline：`408.431 ms/sample`
  - 小中图 all-on fuse：`427.608 ms/sample`
  - 大图消融最佳候选暂为 attention + GC + P35/P120：`394.433 ms/sample`
- 因此大图路线从 baseline 重新开始，不默认沿用 P26/P28/P29/P113/P114/P115。

## 当前阶段

1. Baseline 大图上限测试
   - 从不开优化的 FP32 baseline 出发。
   - 逐档增大 `--large-supercell-repeat`。
   - 每档跑 50 样本。
   - 记录成功数、最大原子数、峰值显存、总时间、平均 latency、OOM 样本。

2. 大图优化消融
   - 在 baseline 上逐组加入候选优化。
   - 优先关注 attention / scatter / reduce / graph construction。
   - 暂缓 MLP/GatedMLP input-grad 和 second-tail macro，除非大图 profile 证明其收益。

3. 大图专用分流策略
   - 小中图继续用当前 fuse 最佳。
   - 大图使用单独 env gate 或 rows/atoms threshold。
   - threshold 必须基于真实 repeat sweep 和消融结果确定。

## Baseline Sweep 记录

测试时间：2026-06-05

口径：不开任何 `MATRIS_*` 优化环境变量，`--quant-mode none --fusion-mode none`，50 样本，`fp32 efsm`。

| repeat | factor | 理论最大原子数 | 成功数 | 峰值显存 | 总时间 | 平均 latency | 最大 latency | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `5,2,2` | 20 | 1600 | 50/50 | 40.52 GB | 24.678 s | 493.562 ms | 3919.400 ms | 成功 |
| `6,2,2` | 24 | 1920 | 50/50 | 48.60 GB | 28.539 s | 570.776 ms | 4709.461 ms | 成功 |
| `4,3,2` | 24 | 1920 | 50/50 | 48.60 GB | 28.448 s | 568.965 ms | 4684.084 ms | 成功 |
| `5,3,2` | 30 | 2400 | 50/50 | 60.70 GB | 34.398 s | 687.968 ms | 5848.232 ms | 成功 |
| `4,4,2` | 32 | 2560 | 50/50 | 64.75 GB | 36.295 s | 725.898 ms | 6352.088 ms | 成功 |
| `6,3,2` | 36 | 2880 | 50/50 | 72.84 GB | 40.454 s | 809.088 ms | 7343.988 ms | 当前 baseline 安全最大档 |
| `5,4,2` | 40 | 3200 | 48/50 | 78.60 GB | 36.237 s | 754.937 ms | 4609.515 ms | OOM，失败于原始 80 atoms 样本 |
| `4,4,3` | 48 | 3840 | 48/50 | 78.52 GB | 43.335 s | 902.816 ms | 5509.292 ms | OOM，失败于原始 80 atoms 样本 |

阶段结论：

- 不开优化的 baseline 在 80GB GPU 上，50 样本当前安全最大档为 `6,3,2`，对应最大样本约 `2880 atoms`。
- `5,4,2` 和 `4,4,3` 都在 sample order 27 / graph_id `355248` 失败，该样本原始为 `80 atoms`，放大后分别为 `3200 atoms` 和 `3840 atoms`。
- 后续大图优化应优先以 `6,3,2` 作为稳定 benchmark 档；如果要测试极限，可以单独针对 `5,4,2` 做显存优化或跳过最大样本分析。

## 2880 Baseline Profile

测试档位：`repeat=6,3,2`，50 样本，最大样本 `2880 atoms`，不开任何优化。

输出文件：

- Stage profile：`results/large_graph_baseline_sweep_20260605/profile_baseline_repeat_6x3x2_stage.json`
- 最大样本 torch profiler top 表：`results/large_graph_baseline_sweep_20260605/profile_baseline_repeat_6x3x2_graph355248_torch_table.txt`

阶段热点：

| 阶段 | 总时间 | 平均时间 | 最大样本时间 | 说明 |
|---|---:|---:|---:|---|
| `calculator_calculate_total_ms` | 40519.705 ms | 810.394 ms | 7347.427 ms | 端到端 calculator 内部总时间 |
| `model_forward_total_ms` | 32451.967 ms | 649.039 ms | 5775.551 ms | 模型主体，包含 interaction 和 force/stress |
| `model.force_stress.autograd_grad_ms` | 20252.332 ms | 405.047 ms | 3986.820 ms | 最大热点，force/stress 对输入求导反传 |
| `model.interaction_blocks_ms` | 11586.756 ms | 231.735 ms | 1774.405 ms | interaction block 主体 |
| `graph_converter_ms` | 7886.664 ms | 157.733 ms | 1551.285 ms | Python/CPU 图构建 |

最大样本 `graph_id=355248` / `2880 atoms` 的 torch profiler 热点：

- GEMM / Linear：`aten::linear`、`aten::addmm`、`aten::mm`、`ampere_sgemm_*` 是最大 CUDA 计算热点。
- Elementwise backward：`MulBackward0`、`aten::mul` 很重，说明 MLP/GatedMLP、attention 权重乘法、force/stress 反传里的乘法链仍明显。
- Scatter / gather：`aten::index_add_`、`aten::scatter_reduce`、`ScatterReduceBackward0`、`aten::gather`、`aten::index_select`、`IndexSelectBackward0` 占比明显，是图结构数据搬运和聚合热点。
- Norm / cat：`aten::native_layer_norm`、`aten::cat` 也在 top 列表中，属于可考虑融合或减少中间 tensor 的 glue。

初步优化方向：

1. 第一优先级：force/stress autograd 穿过 interaction 的 input-gradient 路径。
2. 第二优先级：attention / scatter / reduce / gather 数据组织。
3. 第三优先级：graph_converter 大图构建。
4. GEMM 本体不建议手写普通 CUDA 替代，应考虑 cuBLASLt/CUTLASS/grouped GEMM 或只优化 GEMM 周围 glue。

### NVTX 标记

已加入 env-gated NVTX range：

- `MATRIS_NVTX_RANGES=1`
- calculator stage：`calculator.<stage>`
- model stage：`model.<stage>`
- readout / force-stress stage：`readout.<stage>`
- attention detail stage：`interaction_block.N.attn_line.<stage>`、`interaction_block.N.attn_atom.<stage>`

默认不开启，不影响正常 endpoint。

当前环境可用 `torch.cuda.nvtx`，但 PATH 中未发现 `nsys/ncu`，因此本轮先用 PyTorch profiler / stage profile 表做分析。后续如在 shell 中提供 Nsight 工具，可直接用 `MATRIS_NVTX_RANGES=1` 捕获 NVTX trace。

### 最大样本 attention 细分

最大样本：`graph_id=355248`，`2880 atoms`。

输出文件：`results/large_graph_baseline_sweep_20260605/profile_baseline_repeat_6x3x2_graph355248_attention_detail.json`

attention detail 聚合：

| 子阶段 | 总时间 | 占 interaction blocks | 占样本 wall time |
|---|---:|---:|---:|
| `attn_line.edge_update` | 413.894 ms | 22.34% | 5.56% |
| `attn_line.attention_reduce` | 383.469 ms | 20.70% | 5.15% |
| `attn_line.alpha_projection` | 58.767 ms | 3.17% | 0.79% |
| `attn_line.gather_concat` | 52.123 ms | 2.81% | 0.70% |
| `attn_atom.edge_update` | 44.077 ms | 2.38% | 0.59% |
| `attn_atom.attention_reduce` | 43.374 ms | 2.34% | 0.58% |

结论：最大样本 forward 侧最值得看的局部边界是 `attn_line.edge_update + attention_reduce`，合计约 `797 ms`，占该样本 wall time 约 `10.7%`。不过 50 样本平均中，最大热点仍是 force/stress autograd backward。

### 2880 attention fuse 候选验证

测试候选：`attention + GC + P35/P120`，不开 MLP/GatedMLP input-grad 和 second-tail。

结果目录：`results/large_graph_baseline_sweep_20260605/attention_gc_p35_p120_repeat_6x3x2_limit50/eval`

| 项 | 结果 |
|---|---:|
| 成功数 | 40/50 |
| 峰值显存 | 78.22 GB |
| 成功样本最大原子数 | 1008 |
| 平均 latency，成功样本 | 453.752 ms |

结论：该候选在 1280 档有加速，但在 2880 档不稳定，多个大样本 OOM，不能作为 2880 baseline 的可用 fuse 路线。2880 档 fuse 之前必须先处理显存占用或做 rows/atoms 分流。

### 2880 target attention 低显存小边界

目标：先从 attention/scatter/reduce 中选择一个小而清楚的边界，验证低显存 fused 思路。当前实现新增：

- `target_attention_sum_forward_no_alpha`
- `target_attention_sum_backward_recompute`
- Python 开关：`MATRIS_USE_CUDA_TARGET_ATTENTION_RECOMPUTE=1`
- microbench：`test/eval/benchmark_target_attention_recompute.py`

该边界覆盖单方向 sorted target attention：

```text
target_logits
+ segment softmax
+ alpha * value
+ target reduce
```

普通 CUDA saved-alpha 路线会在 forward 保存 `alpha`，backward 直接使用 `alpha`。recompute 路线 forward 不保存 `alpha`，backward 从 `target_logits` 重新计算 softmax，用更多计算换更低显存。

microbench 结果：

| rows | segments | saved-alpha | recompute | 速度比 | recompute 少峰值显存 |
|---:|---:|---:|---:|---:|---:|
| 32768 | 1024 | 0.260 ms | 0.234 ms | 1.113x | 16 MB |
| 131072 | 4096 | 1.535 ms | 1.346 ms | 1.141x | 64 MB |
| 262144 | 8192 | 2.315 ms | 1.866 ms | 1.241x | 128 MB |

2880 endpoint 单独开 target attention CUDA：

| 候选 | 成功数 | 平均 latency | 相比 baseline | 峰值显存 | 显存下降 |
|---|---:|---:|---:|---:|---:|
| baseline | 50/50 | 809.088 ms | - | 72.84 GB | - |
| saved-alpha CUDA target attention | 50/50 | 784.210 ms | +24.878 ms | 67.96 GB | 4.88 GB |
| recompute CUDA target attention | 50/50 | 788.372 ms | +20.716 ms | 67.22 GB | 5.62 GB |

阶段结论：

- 如果目标是 latency，当前 saved-alpha target attention 更好。
- 如果目标是降低 2880/3200 档显存，recompute 路线更有价值。
- 下一步包大边界不应只替换 target attention，而应把 `source+target attention`、`alpha projection` 和 `attention backward` 一起考虑，避免保存 source/target 两份 alpha，并尽量减少 logits/alpha 中间 tensor。

### 2880 P201 alpha projection + attention recompute 大边界

在 target attention 小边界之后，继续尝试更大的 line attention 边界：

```text
edge_feat_0
+ source/target alpha projection
+ source attention reduce
+ target attention reduce
+ attention backward
+ alpha projection input-grad
```

实现方式：

- env gate：`MATRIS_P201_ALPHA_ATTENTION_RECOMPUTE=1`
- GEMM/Linear 本体仍走 PyTorch/cuBLAS，不手写普通 GEMM。
- CUDA 负责 source/target attention no-alpha forward 和 recompute backward。
- custom autograd 只返回 input-grad，不计算 alpha projection 的 weight/bias grad。
- target 方向使用连续 `target_offsets`；source 方向真实大图中不是连续分段，因此使用 atomic max/sum + recompute。

真实图检查：`graph_id=355248`、`repeat=6,3,2` 下，line graph `target_index` 连续且 offsets 正确；`source_index` 不连续，不能直接使用 source offsets。

2880 endpoint 结果：

| 候选 | 成功数 | 平均 latency | 相比 baseline | 峰值显存 | 显存下降 |
|---|---:|---:|---:|---:|---:|
| baseline | 50/50 | 809.088 ms | - | 72.84 GB | - |
| target saved-alpha CUDA | 50/50 | 784.210 ms | +24.878 ms | 67.96 GB | 4.88 GB |
| target recompute CUDA | 50/50 | 788.372 ms | +20.716 ms | 67.22 GB | 5.62 GB |
| P201 alpha+attention recompute | 50/50 | 696.327 ms | +112.761 ms | 69.80 GB | 3.04 GB |
| P201 low-memory backward v2 | 50/50 | 698.932 ms | +110.156 ms | 66.90 GB | 5.94 GB |

阶段结论：

- P201 是当前 2880 大图下更有希望的 attention 大边界：端到端约 `1.162x`，平均省 `112.761 ms/sample`。
- 它比 target-only 路线快很多，说明把 alpha projection input-grad 和 source/target attention 一起组织起来是有效的。
- P201 low-memory backward v2 牺牲约 `2.605 ms/sample`，但比 P201 v1 继续降低约 `2.90 GB` 峰值显存；如果目标是 2880 稳定低显存或后续冲 3200 atoms，它比 P201 v1 更值得作为低显存 gate。

#### P201 内部 profile

profile 样本：`graph_id=355248`、`repeat=6,3,2`、`2880 atoms`。输出文件：

- `results/large_graph_attention_recompute_20260605/profile_p201_graph355248.json`
- `results/large_graph_attention_recompute_20260605/profile_p201_graph355248.md`

P201 子阶段 CUDA 时间：

| 子阶段 | 调用数 | CUDA total |
|---|---:|---:|
| `P201.forward.attention_recompute_no_alpha` | 20 | 148.582 ms |
| `P201.backward.attention_recompute` | 10 | 99.645 ms |
| `P201.forward.source_alpha_projection` | 20 | 55.408 ms |
| `P201.forward.target_alpha_projection` | 20 | 55.404 ms |
| `P201.backward.recompute_source_logits` | 10 | 27.699 ms |
| `P201.backward.recompute_target_logits` | 10 | 27.700 ms |
| `P201.backward.source_alpha_input_grad_gemm` | 10 | 27.567 ms |
| `P201.backward.target_alpha_input_grad_gemm` | 10 | 27.568 ms |
| `P201.backward.grad_edge_add` | 10 | 13.190 ms |

P201 相关 kernel 中，source 方向 atomic/recompute 比 target offset 更重：

| kernel | 调用数 | self CUDA |
|---|---:|---:|
| `fused_line_attention_single_max_kernel` | 30 | 70.879 ms |
| `fused_line_attention_source_norm_out_no_alpha_kernel` | 20 | 44.781 ms |
| `fused_line_attention_source_exp_sum_no_alpha_kernel` | 30 | 39.013 ms |
| `fused_line_attention_target_backward_recompute_add_kernel` | 10 | 32.610 ms |
| `fused_line_attention_source_backward_recompute_kernel` | 10 | 28.616 ms |
| `fused_line_attention_target_offsets_forward_no_alpha_kernel` | 20 | 28.156 ms |

显存判断：

- 最大样本 line rows 约 `1.54M`，单个 `[rows,128] fp32` tensor 约 `752 MB`。
- P201 forward source/target alpha projection 会分别产生一份 logits。
- P201 backward 会重算 source/target logits，并产生 grad logits / grad edge 相关临时。
- 因此 P201 当前主要峰值来自多个 `[rows,128]` 级临时 tensor 同时存在，而不是 target attention 本身。

下一步优先级：

1. P201 low-memory backward v2 已完成验证：source 和 target 分阶段计算，避免 `grad_source_logits`、`grad_target_logits`、`grad_edge_feat`、`grad_target_edge_feat` 同时驻留。
2. source 方向优化已完成 P203 初版：用 `source_sort_order + source_segment_offsets` 改写 source attention reduce，避免 source atomic max/sum/out。
3. alpha projection forward 可以考虑 source/target 合成一个更宽 GEMM，但这主要优化 launch/GEMM 调度，对峰值显存帮助有限。

#### P201 low-memory backward v2

v2 目标是降低 backward 峰值显存，而不是优先追求更低 latency。原 P201 backward 会同时持有 source/target logits、source/target grad logits，以及两份 alpha projection input-grad 临时；v2 改为分阶段执行：

```text
recompute source logits
-> source attention backward
-> source alpha input-grad GEMM 得到 grad_edge_feat
-> 释放 source 临时
-> recompute target logits
-> target attention backward，并累加到 grad_values
-> target alpha input-grad GEMM add 到 grad_edge_feat
```

新增开关：

- `MATRIS_P201_ALPHA_ATTENTION_RECOMPUTE=1`
- `MATRIS_P201_LOW_MEMORY_BWD=1`

2880 endpoint 结果：

| 候选 | 平均 latency | 总时间 | 峰值显存 | 相比 P201 v1 |
|---|---:|---:|---:|---:|
| P201 v1 | 696.327 ms | 34.816 s | 69.80 GB | - |
| P201 low-memory backward v2 | 698.932 ms | 34.947 s | 66.90 GB | 慢 `2.605 ms/sample`，少 `2.90 GB` |

最大样本 `graph_id=355248` 的 P201 profile：

| 候选 | latency | peak memory |
|---|---:|---:|
| P201 v1 | 6089.809 ms | 71449.728 MB |
| P201 low-memory backward v2 | 6074.299 ms | 68477.759 MB |

数值一致性：v1/v2 的 50 样本输出基本一致，`energy_pred` 最大差异约 `3.05e-5`，`force_mae` 最大差异约 `2.81e-7`，`stress_mae` 最大差异约 `2.15e-8`。

阶段结论：v2 是有效的低显存版本。它没有明显端到端加速，但在保留 P201 大边界主要速度收益的同时进一步降低显存，适合作为后续 3200 atoms 或更大图尝试的基础。

#### P203 source-sorted attention offsets

P201 profile 显示 source 方向仍然明显更重，原因是 line graph 的 `source_index` 不连续，原实现只能用 atomic 做 source max/sum/out：

```text
source logits
-> atomic max by source_index
-> atomic sum by source_index
-> atomic out reduce by source_index
```

P203 在图构建阶段额外生成：

- `source_segment_offsets`
- `source_sort_order = argsort(source_index)`

然后 CUDA kernel 按 segment 遍历排序后的原始 row：

```text
for source segment:
  rows = source_sort_order[source_offsets[s]:source_offsets[s+1]]
  max / sum / out / backward recompute
```

这样输出仍写到原始 row 或原始 segment，不改变数学结果，但 source attention 不再需要 atomic reduce。target 方向保持 P201 的 target-offset 路径，GEMM/Linear 仍走 PyTorch/cuBLAS。

新增开关：

- `MATRIS_P201_ALPHA_ATTENTION_RECOMPUTE=1`
- `MATRIS_P203_SOURCE_SORTED_ATTENTION=1`

2880 endpoint 结果：

| 候选 | 成功数 | 平均 latency | 总时间 | 峰值显存 | 相比 baseline | 相比 P201 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 50/50 | 809.088 ms | 40.454 s | 72.84 GB | - | - |
| P201 v1 rerun | 50/50 | 696.835 ms | 34.842 s | 69.80 GB | +112.253 ms | - |
| P203 source-sorted | 50/50 | 686.230 ms | 34.312 s | 69.81 GB | +122.858 ms / 1.178x | +10.605 ms |
| P203 source-sorted + low-memory bwd | 50/50 | 688.525 ms | 34.426 s | 68.34 GB | +120.563 ms / 1.176x | 慢 2.295 ms，少 1.47 GB |

最大样本 `graph_id=355248` profile：

| 项 | P201 v1 历史 | P203 |
|---|---:|---:|
| 最大样本 latency | 6089.809 ms | 5953.367 ms |
| peak memory | 71449.728 MB | 71464.854 MB |
| `P201.forward.attention_recompute_no_alpha` | 148.582 ms | 58.436 ms |

source atomic 相关 kernel 被 P203 source-offset kernel 替换：

| kernel | P201 self CUDA | P203 self CUDA |
|---|---:|---:|
| `fused_line_attention_single_max_kernel` | 70.879 ms | 不再出现 |
| `fused_line_attention_source_norm_out_no_alpha_kernel` | 44.781 ms | 不再出现 |
| `fused_line_attention_source_exp_sum_no_alpha_kernel` | 39.013 ms | 不再出现 |
| `fused_line_attention_source_offsets_forward_no_alpha_kernel` | - | 30.062 ms |
| `fused_line_attention_source_offsets_backward_recompute_kernel` | - | 29.776 ms |

阶段结论：P203 是当前 2880 大图下新的最佳速度候选。它没有进一步降低峰值显存，但在 P201 已经有效的基础上继续减少 source atomic reduce 成本，端到端再省约 `10.6 ms/sample`。

P203 low-memory backward 版本新增：

- `MATRIS_P203_LOW_MEMORY_BWD=1`

该版本把 backward 中的：

```text
grad_edge_feat = grad_source_logits @ W_source
grad_target_edge_feat = grad_target_logits @ W_target
grad_edge_feat = grad_edge_feat + grad_target_edge_feat
```

改为：

```text
grad_edge_feat = grad_source_logits @ W_source
grad_edge_feat.addmm_(grad_target_logits, W_target)
```

它少一份 `grad_target_edge_feat` 临时 tensor 和一个 add kernel。50 样本结果显示速度略慢约 `2.3 ms/sample`，但峰值显存降低约 `1.47 GB`。因此它不作为当前速度最佳默认路径，但适合作为冲更大原子数时的低显存备选。

#### GatedMLP W8A8 static quantization

目标：针对大图 profile 中占比较高的 GatedMLP Linear 做 W8A8 量化。当前阶段只先替换 Linear 本体，把 activation quant、int8 GEMM 和 dequant 输出组织到同一路径；暂不把后续 tail / residual / scatter 继续并入量化算子。

覆盖模块：

- `attn_block_line_graph.node_nonlinear_update`
- `attn_block_line_graph.edge_nonlinear_update`
- `attn_block_atom_graph.node_nonlinear_update`
- `attn_block_atom_graph.edge_nonlinear_update`
- `refine_block_line_graph.edge_nonlinear_update`
- `refine_block_atom_graph.edge_nonlinear_update`

新增量化模式：

- `--quant-mode gated_mlp_all_w8a8_static`
- `MATRIS_W8A8_BACKEND=cuda_cutlass`

覆盖范围：6 类 GatedMLP × 10 个 interaction block × core/gate 两支 × first/second 两层 Linear，共替换 `240` 个 Linear。first projection 和 second projection 都进入 W8A8。

正式 endpoint 口径：`repeat=6,3,2`，`limit=50`，`fp32`，叠加 `nvalchemi tensor graph + P203 source-sorted attention`。

| 候选 | 成功数 | 平均 latency | 总时间 | 吞吐 | 峰值显存 | force MAE | 相比 no-quant P203 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-quant P203 + tensor graph | 50/50 | 579.098 ms | 28.955 s | 1.7268/s | 71.48 GB | 0.019543 | - |
| W8A8 CUTLASS GatedMLP | 50/50 | 496.864 ms | 24.843 s | 2.0126/s | 60.70 GB | 0.024581 | +82.233 ms / 1.166x |

相对不开任何优化的 pymatgen baseline `814.886 ms/sample`，当前 `tensor graph + P203 + W8A8 CUTLASS` 为 `496.864 ms/sample`，端到端约 `1.640x`，平均每个结构节省约 `318.022 ms`。

注意事项：

- `cuda_cutlass` 后端本轮 50 样本全部成功，是当前可用候选。
- `cuda_wmma` 后端在大样本上出现过 `CUDA error: invalid configuration argument`，不作为当前候选。
- 当前 W8A8 主要加速 forward Linear；force/stress input-grad backward 仍有较多 FP32/dequantized 路径，后续若继续量化，应优先看 selective first projection 量化或真正的 int8 input-grad backward。
- 量化带来一定误差变化：force MAE 从 `0.019543` 增至 `0.024581`，需要结合任务误差容忍度决定是否进入默认大图路径。

##### Refine FFN W8A8 扩展消融

目标：在当前 `GatedMLP W8A8` 的 `240` 个 Linear 基础上，继续量化 refine 普通 MLP/FFN：

- `refine_block_atom_graph.node_FFN`
- `refine_block_atom_graph.edge_FFN`
- `refine_block_line_graph.node_FFN`
- `refine_block_line_graph.edge_FFN`

新增量化模式：

- `--quant-mode gated_mlp_all_plus_refine_ffn_w8a8_static`

覆盖范围：当前 GatedMLP `240` 个 Linear + refine FFN `80` 个 Linear，共 `320` 个 Linear。新增 FFN Linear 均为 `Linear(128,128, bias=False)`。

同一 endpoint 口径：`repeat=6,3,2`，`limit=50`，`fp32`，叠加 `nvalchemi tensor graph + P203 source-sorted attention`。

| 候选 | 替换 Linear 数 | 成功数 | 平均 latency | 总时间 | 吞吐 | 峰值显存 | force MAE | stress MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-quant P203 + tensor graph | 0 | 50/50 | 579.098 ms | 28.955 s | 1.7268/s | 71.48 GB | 0.019543 | 0.001531 |
| GatedMLP W8A8 current | 240 | 50/50 | 488.615 ms | 24.431 s | 2.0466/s | 60.70 GB | 0.029409 | 0.002415 |
| GatedMLP + refine FFN W8A8 | 320 | 50/50 | 513.308 ms | 25.665 s | 1.9481/s | 58.47 GB | 0.037811 | 0.002454 |

结论：refine FFN W8A8 扩展可以继续降低峰值显存约 `2.22 GB`，但相比当前 `GatedMLP W8A8` 慢 `24.693 ms/sample`，force MAE 也从 `0.029409` 增至 `0.037811`。因此该路径不进入当前速度最佳候选，只作为低显存/量化消融记录。

##### Refine FFN W8A8 macro 专用路线

目标：验证 refine 普通 FFN 是否需要专门按 `Linear1 + SiLU + Linear2 + input-grad-only backward` 组织，而不是继续用单个 Linear 粒度扩大 W8A8 target。该实验复用已有 P67/P68 FFN macro 路径：

```text
forward:
  Linear1 W8A8 CUTLASS
  + SiLU
  + Linear2 W8A8 CUTLASS

backward input-grad-only:
  grad_hidden = grad_out @ W2
  + SiLU'(hidden)
  + grad_x = grad_hidden @ W1
```

新增接入口：

- `--fusion-mode refine_all_ffn_mlp_input_grad_only`
- `MATRIS_P67_FFN_FUSED_QUANT=1`
- `MATRIS_P67_FFN_FUSED_QUANT_BACKEND=cutlass`
- `MATRIS_P67_FFN_FUSED_QUANT_SCOPE=all_ffn`
- `MATRIS_P68_GROUPED_FFN_PAIR=1`
- `MATRIS_P68_GROUPED_FFN_PAIR_SCOPE=all_refine`

同一 endpoint 口径：`repeat=6,3,2`，`limit=50`，`fp32`，叠加 `nvalchemi tensor graph + P203 source-sorted attention + GatedMLP W8A8 CUTLASS`。

| 候选 | 平均 latency | 总时间 | 吞吐 | 峰值显存 | force MAE | stress MAE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 当前最佳 GatedMLP W8A8 | 488.615 ms | 24.431 s | 2.0466/s | 60.70 GB | 0.029409 | 0.002415 | 当前速度最佳 |
| refine FFN 单 Linear W8A8 | 513.308 ms | 25.665 s | 1.9481/s | 58.47 GB | 0.037811 | 0.002454 | 慢，精度变差 |
| P67 refine FFN two-linear W8A8 macro | 514.326 ms | 25.716 s | 1.9443/s | 58.48 GB | 0.029819 | 0.002389 | 精度恢复，但速度不赢 |
| P68 refine FFN grouped-pair W8A8 macro | 513.574 ms | 25.679 s | 1.9471/s | 58.49 GB | 0.029749 | 0.002392 | 精度恢复，但速度不赢 |

阶段结论：

- P67/P68 证明 FFN 专用 macro 可以把 force/stress 精度拉回当前 GatedMLP W8A8 同一档，不再像单 Linear FFN 量化那样明显恶化。
- 但 P67/P68 仍比当前最佳慢约 `25 ms/sample`，速度和单 Linear FFN 量化几乎同档，因此不进入当前默认候选。
- 这说明当前 FFN 量化瓶颈不只是 forward Linear 粒度问题；force/stress input-grad 路径中仍有 dequantized FP32 GEMM / 临时 tensor / 调度开销。后续若继续做 FFN 量化，应转向真正的 CUTLASS/cuBLASLt input-grad epilogue 或 grouped GEMM，而不是继续扩大普通 W8A8 Linear 覆盖范围。
- 如果目标是显存而非速度，P67/P68 可作为低显存备选：相对当前最佳少约 `2.2 GB` 峰值显存，但会牺牲约 `25 ms/sample` latency。

##### W8A8 input-grad backward 原型验证

目标：验证大图 force/stress backward 中的 `grad_x = grad_out @ W` 是否可以改成 W8A8/CUTLASS input-grad，而不是继续走 dequantized FP32 `F.linear`。

原型方式：

```text
当前参考：
  q_weight -> dequant W
  grad_x = grad_out @ dequant(W)

W8A8 input-grad 原型：
  预先把 dequant(W).T 重新 per-channel 打包成 backward q_weight
  backward 时量化 grad_out
  CUTLASS int8 GEMM
  dequant 输出 grad_x
```

新增实验开关：

- `MATRIS_W8A8_INPUT_GRAD_BACKEND=cutlass`
- `MATRIS_W8A8_INPUT_GRAD_MIN_ROWS=8192`

局部 microbench：`test/eval/benchmark_w8a8_input_grad.py`。

| 形状 | rows | FP/dequant input-grad | W8A8 input-grad | 局部速度比 | mean abs error |
|---|---:|---:|---:|---:|---:|
| 128 -> 128 | 8192 | 0.0307 ms | 0.0260 ms | 1.181x | 3.39e-3 |
| 128 -> 128 | 65536 | 0.1635 ms | 0.1399 ms | 1.169x | 3.73e-3 |
| 128 -> 384 | 8192 | 0.0865 ms | 0.1153 ms | 0.750x | 3.53e-3 |
| 128 -> 384 | 65536 | 0.8514 ms | 0.5713 ms | 1.490x | 3.68e-3 |
| 128 -> 512 | 8192 | 0.1435 ms | 0.0962 ms | 1.492x | 3.44e-3 |
| 128 -> 512 | 65536 | 1.0701 ms | 0.3485 ms | 3.071x | 3.58e-3 |

endpoint smoke，`repeat=6,3,2`，`limit=10`：

| 候选 | 平均 latency | 总时间 | 峰值显存 | force MAE | stress MAE | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 当前最佳 GatedMLP W8A8 | 416.325 ms | 4.163 s | 31.87 GB | 0.027243 | 0.001681 | limit10 对照 |
| + W8A8 input-grad prototype | 423.407 ms | 4.234 s | 31.87 GB | 0.027467 | 0.002127 | 慢 7.081 ms，stress 变差 |

阶段结论：

- 局部 GEMM 上 W8A8 input-grad 有一定潜力，尤其是更宽输出维度和大 rows 形态。
- 但直接在量化 Linear backward 中替换为 W8A8 input-grad，端到端没有转化为收益，`limit10` 慢约 `7.1 ms/sample`，stress MAE 也变差。
- 原因可能是：每个 Linear backward 单独量化 `grad_out` 的开销、额外 backward 专用 weight buffer、误差在 force/stress 反传中累积，以及没有把 SiLU/LayerNorm/gate glue 一起并进 GEMM epilogue。
- 因此该原型不进入当前最佳候选。后续如果继续做 backward 量化，应该走更大粒度的 CUTLASS/cuBLASLt epilogue 或 grouped GEMM：把 `grad_out quantize + GEMM + SiLU/LayerNorm/gate grad + grad_x combine` 组织成同一个 macro，而不是逐个 Linear 替换。

##### W8A8 周边 fuse 验证

本轮尝试在量化 Linear 周围继续合并一块边界，优先选择 GatedMLP 的 first projection：

```text
core first Linear
+ gate first Linear
-> CUTLASS dual W8A8 Linear
+ input-grad-only autograd wrapper
```

说明：该实验对应代码已清理，不再保留可启用开关，避免误接入当前最佳路径。

50 样本正式结果：

| 候选 | 覆盖范围 | 成功数 | 平均 latency | 总时间 | 峰值显存 | force MAE | stress MAE | 相比当前 W8A8 CUTLASS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W8A8 CUTLASS current | - | 50/50 | 496.864 ms | 24.843 s | 60.70 GB | 0.024581 | 0.001559 | - |
| P204 first-only fused | all 60 GatedMLP | 50/50 | 555.590 ms | 27.779 s | 60.72 GB | 0.024515 | 0.001567 | 慢 58.726 ms |
| P204 first-only fused | line/refine line edge | 50/50 | 590.785 ms | 29.539 s | 60.72 GB | 0.024605 | 0.001559 | 慢 93.921 ms |

结论：first projection dual CUTLASS fuse 数值正确，但端到端不加速。主要原因是 force/stress 场景需要 input-grad，当前 autograd wrapper 的 backward 仍走 dequantized FP32 `F.linear`，抵消了前向少一次调度的收益。因此 P204 只保留历史结果记录，不再保留可启用代码路径。

同时尝试过 second projection + gated tail 的更大边界：

```text
core/gate second W8A8 Linear
+ LayerNorm
+ SiLU/Sigmoid
+ core * gate
```

该路径虽然 latency 很低，但 force/stress 明显失真：

| 候选 | 平均 latency | 总时间 | 峰值显存 | force MAE | stress MAE | 结论 |
|---|---:|---:|---:|---:|---:|---|
| second-tail forward-only fuse | 216.547 ms | 10.827 s | 31.73 GB | 0.098337 | 0.007893 | 不可用，缺少正确 force/stress input-grad |
| second-tail CUDA backward fuse | 226.589 ms | 11.329 s | 36.16 GB | 0.098330 | 0.007888 | 不可用，CUDA backward 不等价 |
| second-tail reference backward | 1004.965 ms | 50.248 s | 69.73 GB | 0.024528 | 0.001565 | 精度恢复，但速度/显存不可用 |
| second-tail saved-pre-dq CUDA backward | 1832.590 ms | 91.629 s | 69.73 GB | 0.024473 | 0.001562 | 精度恢复，但端到端更慢 |

局部 correctness microbench 曾用于定位问题，实验脚本已随无效代码清理。记录保留如下，便于解释为什么不继续走该方向。

| 路径 | forward max error | grad_core max error | grad_gate max error | 局部时间 |
|---|---:|---:|---:|---:|
| fused CUDA backward vs current | 1.073e-6 | 1.640e-1 | 1.080e-1 | 1.442 ms |
| fused reference backward vs current | 1.073e-6 | 9.537e-7 | 4.768e-7 | 0.369 ms |
| fused saved-pre-dq CUDA backward vs current | 1.073e-6 | 9.537e-7 | 4.768e-7 | 0.370 ms |

阶段结论：

- forward fuse 本身几乎等价，`forward max error` 只有约 `1e-6`。
- force/stress 失真根因在原 `w8a8_dual_gated_tail_input_grad_backward_n128` 这条 CUDA backward：局部 `grad_core/grad_gate` 误差达到 `1e-1` 量级。
- reference backward 使用稳定路径：`tail backward + dequantized FP32 weight input-grad`，局部梯度可对齐，并且 endpoint force/stress 恢复正常。
- 但 reference backward 需要额外保存/重算 second Linear 输出并走 FP32 input-grad，端到端从当前 W8A8 CUTLASS 的 `496.864 ms` 变成 `1004.965 ms`，因此不作为性能候选。
- saved-pre-dq CUDA backward 也能对齐局部梯度，但端到端更慢，说明当前一行一个 block 的 128x128 input-grad kernel 在大图全模型里不如 cuBLAS/torch linear 的调度和吞吐。

后续如果继续围绕 W8A8 second-tail 做 fuse，重点应转为 cuBLASLt/CUTLASS epilogue 或 grouped GEMM 级别的等价 input-grad，而不是一行一个 block 的普通 CUDA 128x128 matmul。另一个可选方向是只在 energy-only 推理模式下启用 second-tail forward fuse。

#### P202 line attention mixed macro 尝试

目标：继续扩大 P201 边界，把 line attention 外围的 gather/cat、edge_update、node_update、residual、scatter 汇总纳入同一个 manual VJP pipeline。实现上先不新增 CUDA kernel，而是复用现有 P101 attn_line pipeline、P201 no-alpha/recompute attention、P108 edge/alpha/project/scatter fused backward。

新增开关：

- `MATRIS_P101_A3_LITE_ATTN_LINE_VJP=1`
- `MATRIS_P202_ATTN_LINE_PIPELINE_RECOMPUTE=1`
- 可选配合：`MATRIS_P108_A_CUDA3_ATTN_LINE_DENSE_GEMM_OP_BWD=1`

2880 limit5 初筛：

| 候选 | block 范围 | 成功数 | 平均 latency | 峰值显存 | 结论 |
|---|---|---:|---:|---:|---|
| P201 v1 | all line attention P201 | 5/5 | 994.152 ms | 36.06 GB | 当前更优 |
| P202 pipeline recompute | block9 | 5/5 | 1137.747 ms | 39.96 GB | 慢，显存更高 |
| P202 + P108 dense-GEMM backward | block9 | 5/5 | 1134.189 ms | 39.96 GB | 仍慢 |
| P202 pipeline recompute | block8-9 | 5/5 | 1118.565 ms | 44.44 GB | 仍慢，显存更高 |
| P202 pipeline recompute | all 10 blocks | 4/5 | 360.610 ms | 77.02 GB | 第 5 个样本 OOM，不可用 |

阶段结论：

- P202 调度级包大边界没有转化为速度收益。主要原因是 manual VJP 需要保存 edge_update、node_update、attention、GatedMLP 中间状态，显存和 cache 压力超过了减少 PyTorch autograd glue 带来的收益。
- P108 edge/alpha/project/scatter fused backward 能略微降低 block9 latency，但幅度很小，不能扭转 P202 的整体劣势。
- 后续如果继续扩大边界，不能只靠 Python custom VJP 调度层；需要真正减少保存张量，或把 edge_update tail、attention backward、alpha projection input-grad、scatter 回写写成更紧的 CUDA/CUTLASS mixed macro。

### nvalchemi + Cython tensor graph builder

目标：构图阶段不再只替换 neighbor list，而是继续绕过 Python `Graph/Node/Edge` object 路径，直接生成 MatRIS 需要的图张量：

```text
nvalchemi neighbor_list
-> C create_graph 分组
-> atom_graph / directed2undirected / undirected2directed / line_graph tensor
```

新增开关：

- `MATRIS_GRAPH_BACKEND=nvalchemi`
- `MATRIS_NV_GRAPH_METHOD=cell_list`
- `MATRIS_NV_GRAPH_DEVICE=cuda`
- `MATRIS_NV_GRAPH_TENSOR_PATH=1`

实现位置：

- `matris/graph/cygraph.pyx`：新增 `build_graph_tensors_fast()`，复用已有 C `create_graph()`，直接返回 numpy arrays。
- `matris/graph/converter.py`：在 nvalchemi backend 下通过 env gate 接入 tensor path，默认路径不变。

2880 最大样本局部构图核心验证：

测试样本：`graph_id=355248`，`repeat=6,3,2`，`2880 atoms`，`157824` directed edges，`559872` line graph rows。

同一份 nvalchemi neighbor list 下，四个核心张量与旧 object path 完全一致：

- `atom_graph`
- `directed2undirected`
- `undirected2directed`
- `line_graph`

| 路径 | 中位时间 | 平均时间 | 说明 |
|---|---:|---:|---|
| 旧 object path | 1927.420 ms | 1848.949 ms | `Graph/Node/Edge` object + line graph |
| Cython tensor path | 160.208 ms | 161.324 ms | 直接生成四个图张量 |

endpoint smoke，`repeat=6,3,2`，`limit=5`：

| 路径 | 成功数 | 平均 latency | 总时间 | 说明 |
|---|---:|---:|---:|---|
| nvalchemi object path | 5/5 | 1521.869 ms | 7.609 s | 旧 Graph object 路径 |
| nvalchemi tensor path | 5/5 | 1353.959 ms | 6.770 s | 新 Cython tensor path |

正式 endpoint，`repeat=6,3,2`，`limit=50`：

| 路径 | 成功数 | 平均 latency | 总时间 | 吞吐 | 相比 pymatgen | 结论 |
|---|---:|---:|---:|---:|---:|---|
| pymatgen baseline | 50/50 | 814.886 ms | 40.744 s | 1.2272/s | - | baseline |
| nvalchemi object path | 50/50 | 848.041 ms | 42.402 s | 1.1792/s | -33.155 ms | 只换 neighbor list 不划算 |
| nvalchemi tensor path | 50/50 | 709.444 ms | 35.472 s | 1.4096/s | +105.442 ms | 有明确端到端收益 |

`nvalchemi tensor path` 相比 `pymatgen baseline` 端到端约 `1.149x`，平均每个结构节省约 `105.442 ms`；相比 `nvalchemi object path` 平均节省约 `138.597 ms`。

阶段结论：

- 这条路线已经证明可以显著压低大图构图核心开销；相比继续只优化 nvalchemi neighbor list，它更接近真正的大图构图瓶颈。
- endpoint `limit50` 已证明：只换 nvalchemi neighbor list 不够，绕过 Python Graph object / line graph path 后，构图优化能转成端到端收益。
- 当前实现仍然在 CPU/Cython 中构建 line graph。如果 50 样本结果仍显示构图明显，下一步再考虑更紧的 C/C++ 数据结构或 CUDA line graph builder。

## 2880 Torch Compile Probe

测试目的：用 `torch.compile` 探测大图下可融合长边界是否有端到端收益。该阶段不接入默认路径，只做独立 probe。

脚本：`test/eval/probe_large_graph_torch_compile.py`

测试范围：

- `full_model`：编译整个 MatRIS model。
- `interaction_blocks`：编译 10 个 interaction block。
- `force_stress_head`：编译 force/stress head。

结果：

| probe | compile scope | 成功数 | measured mean | 峰值显存 | 结论 |
|---|---|---:|---:|---:|---|
| smoke none limit2 | none | 2/2 | 427.741 ms | 13.61 GB | baseline probe 正常 |
| smoke full_model limit2 | full_model | 0/2 | 0.000 ms | 11.82 GB | 失败，autograd differentiated tensor unused |
| smoke interaction_blocks limit2 | interaction_blocks | 2/2 | 19424.922 ms | 11.84 GB | 严重变慢，重编译/graph break 过多 |
| smoke force_stress_head limit2 | force_stress_head | 2/2 | 1376.846 ms | 13.61 GB | 变慢，CUDA graph 基本为空 |
| max sample none warm1 | none | 1/1 | 7008.200 ms | 72.82 GB | 2880 atoms baseline |
| max sample force_stress_head warm1 | force_stress_head | 1/1 | 8255.669 ms | 72.81 GB | 同 shape steady-state 仍慢 |
| max sample interaction_blocks warm1 | interaction_blocks | 1/1 | 209171.081 ms | 67.95 GB | 不可用 |

主要问题：

- `Dimwise_softmax()` 中存在 `int(segment.max()) + 1`，触发 `Tensor.item()` graph break。
- line/atom/refine 路径中 `directed2undirected` 会在 Tensor / None 之间变化，导致 recompile limit。
- `profile_prefix` 等 Python 字符串分支也会导致 block forward 重编译。
- `force_stress_head` 里核心成本来自 `torch.autograd.grad` 穿过前面 interaction graph，直接 compile head 本身不能捕获这条长反传链。

阶段结论：

- 不建议做 `torch.compile(model)` 或 `torch.compile(interaction_blocks)` 作为大图 endpoint 主线。
- 如果继续用 compile，只适合作为更小局部边界的公式探测工具，例如 attention softmax/reduce、scatter/gather glue、residual/norm/elementwise 链。
- 大图真正优化仍应回到 CUDA/C++/cuBLASLt/CUTLASS 分流：attention/scatter/reduce 用专用 CUDA，GEMM 本体保留 cuBLAS/CUTLASS，graph_converter 走 C++/tensor 化方向。
