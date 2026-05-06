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

In a full MatRIS evaluation environment, also run a small CUDA sample with
activation calibration to confirm model loading, quantized replacement,
GatedMLP fusion, and force/stress autograd all execute successfully.
