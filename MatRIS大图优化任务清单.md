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
