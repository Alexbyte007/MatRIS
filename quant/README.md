# P8E Stable Quantization

This branch keeps a single stable quantization entry point for the MatRIS
W8A8 path:

```bash
p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8
```

The mode quantizes the second projection of the core/gate MLP branches in:

- all `refine_block_line_graph.edge_nonlinear_update` blocks
- `attn_block_line_graph.edge_nonlinear_update` blocks 8 and 9

It uses the `triton_w8a8_static` backend with static per-tensor activation
scales and per-channel int8 weight scales.

## Usage

Apply the quantization config before evaluation:

```python
from quant.config import get_quant_config
from quant.injector import apply_quant_config
from quant.fusion import apply_gated_mlp_fusion

quant_mode = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"

model = MatRIS.load(model_name="matris_10m_oam", device="cuda")
apply_quant_config(model, get_quant_config(quant_mode))
apply_gated_mlp_fusion(model, "line_edge_gated_mlp_second_tail_fused_fp32")
model.eval()
```

For E/F/S evaluation where model parameters are frozen, use:

```bash
export MATRIS_FREEZE_MODEL_PARAMS_FOR_EFS=1
export MATRIS_W8A8_BACKEND=cuda_wmma_tail_n128
```

The quantized linear layers need activation calibration before the measured
evaluation pass. The existing evaluation/profile utilities call
`reset_activation_calibration()` and `finalize_activation_calibration()` through
the quantized modules.

## Sanity Checks

Basic import and syntax check:

```bash
python -m py_compile \
  matris/model/functions.py \
  quant/config.py \
  quant/injector.py \
  quant/layers.py \
  quant/stats.py
```

Minimal config check:

```bash
python - <<'PY'
from quant.config import get_quant_config

mode = "p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8"
cfg = get_quant_config(mode)
assert cfg["kernel"] == "triton_w8a8_static"
assert len(cfg["targets"]) == 6
print(cfg["mode"])
print(cfg["targets"])
PY
```

## P8E Test Commands

Set the dataset path once before running the CUDA tests:

```bash
export MATRIS_DATASET_SRC=/path/to/sAlex/val
export MATRIS_FREEZE_MODEL_PARAMS_FOR_EFS=1
export MATRIS_W8A8_BACKEND=cuda_wmma_tail_n128
```

Smoke test the real MatRIS path on one sAlex sample. This checks model loading,
quantized replacement, GatedMLP fusion, activation calibration, and E/F/S
autograd:

```bash
MATRIS_DATASET_SRC="${MATRIS_DATASET_SRC}" \
MATRIS_ALIGNMENT_STRICT_ASSERT=0 \
MATRIS_BASELINE_W8A8_BACKEND=cuda_wmma_tail_n128 \
MATRIS_CANDIDATE_W8A8_BACKEND=cuda_wmma_tail_n128 \
MATRIS_ALIGNMENT_SAMPLE_INDEX=26225 \
MATRIS_ALIGNMENT_OUTPUT=results/p8e_submit_sanity/w8a8_backend_same_backend_alignment.json \
python test/eval/check_w8a8_backend_matris_alignment.py
```

Run a small runtime profile:

```bash
python test/eval/profile_salex_pipeline.py \
  --dataset-src "${MATRIS_DATASET_SRC}" \
  --output-dir results/p8e_profile_smoke \
  --model matris_10m_oam \
  --task efs \
  --device cuda \
  --precision-mode fp32 \
  --quant-mode p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8 \
  --fusion-mode line_edge_gated_mlp_second_tail_fused_fp32 \
  --activation-calibration-limit 64 \
  --activation-calibration-seed 43 \
  --limit 10 \
  --warmup-steps 2 \
  --combined-force-stress-autograd
```

Run a small precision check:

```bash
python test/eval/evaluate_salex_static_metrics.py \
  --dataset-src "${MATRIS_DATASET_SRC}" \
  --output-dir results/p8e_precision_smoke \
  --model matris_10m_oam \
  --task efs \
  --device cuda \
  --precision-mode fp32 \
  --quant-mode p8d_refine_w8a8_attn_line_core_gate_second_blocks_8_9_w8a8 \
  --fusion-mode line_edge_gated_mlp_second_tail_fused_fp32 \
  --activation-calibration-limit 64 \
  --activation-calibration-seed 43 \
  --limit 50 \
  --warmup-steps 3
```

For a formal comparison, increase the precision `--limit` to the same sample
count used by the baseline run and compare against an FP32 run with
`--quant-mode none --fusion-mode none`.
