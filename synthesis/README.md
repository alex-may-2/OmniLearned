# hls4ml conversion + synthesis

Notes for the agent or person running the FPGA step.

- Training, QAT and QONNX export happen on NERSC (Perlmutter).
- `convert.py` C-sim runs on Perlmutter or on SLAC rdsrv409.
- Vitis HLS synthesis runs only on rdsrv409.

Graph changes are made on the PyTorch side in `tools/quantize/qat_deepsets.py`, which does the Brevitas
wrapping (`network.py` stays Brevitas-free). Re-export with `tools/quantize/qat_deepsets_export_qonnx.py`.

## Input

Copy the graphs into `onnx_graphs/` from `/pscratch/sd/a/alexmay/omnilearned/qonnx/fpga/` on Perlmutter:

- `qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant_clean.onnx`: **current graph**, the `convert.py`
  default. It is bit-exact in HLS (see "Full-quant QAT").
- `qat_top_deepsets_distillnet_fpga_a05_T4_8bit_clean.onnx`: the first QAT graph. Only weights and MatMul
  inputs are quantized, and it reaches 97.5% HLS agreement. Kept for comparison (`--onnx`).

Model and graph:

- **Model:** DeepSets "distillnet" student (base_dim 32, phi 2 layers, rho 1 layer, ReLU, DynamicTanh
  norm), ~11k params. It is KD'd from `fine_tune_top_l`, then given 8-bit Brevitas QAT (power-of-2 scales,
  per-tensor fixed point).
- **Task:** top-tagging, 2 logits.
- **Input `global_in`:** `[64, 64, 4]` = (batch, particles, features). Particles are the leading-pT 64
  slots of the 150-slot dataset, with no mask. The batch dim is fixed to 64 by the export dummy input.
- **Output:** `[64, 2]` logits.
- **Full-quant ops (56 nodes):** Quant x27, Add x6, Transpose x5, MatMul x4, Gemm x3, Mul x3, Relu x3,
  Tanh x3, Flatten, GlobalAveragePool.
  - The mean pool is `Transpose` + `GlobalAveragePool` (not `ReduceMean`).
  - Every weight, bias (16-bit), MatMul/Gemm input and tanh input is `Quant`-ed, with power-of-2
    scales. Inputs that follow a ReLU are unsigned.
  - The residual stream (embed output, phi residual sum) and the pool output are 10-bit `Quant`s.
  - The only non-`Quant` ops are ReLU, the power-of-2 alpha `Mul`s, residual adds and the pool. All of
    them are exact in fixed point.
- **Export check:** onnx checker OK, IR version 9, and qonnx-executor vs PyTorch max |Δ logit| = 0.

Real-jet test set: `top_test_10k_n64.npz` (in this directory, and on Perlmutter next to the ONNX).

- `x`: float32 `[10000, 64, 4]`, the first 10k jets of the top-tagging test split, with the same
  preprocessing as training.
- `y`: int `[10000]` labels (5080 / 4920).
- Evaluated in batches of 64 (9984 jets = 156 full batches).

## Environment

- **rdsrv409:** conda env `/u1/alexmay/conda/envs/omnilearned-hls` (Python 3.11).
  - Packages: `qonnx==1.0.0`, `onnx==1.20.1`, `onnxruntime==1.30.0`, `onnxoptimizer==0.4.2`,
    scikit-learn, and the hls4ml fork
    `git+https://github.com/qibin2020/hls4ml.git@fix/onnx-frontend-shape-bugs` at `1d85133`.
  - Keep onnx < 1.21: newer versions write IR 14, which onnxruntime 1.30 rejects.
  - The fork is needed because stock hls4ml misreads rank-3 MatMul / Reshape shapes.
- **Perlmutter:** `/global/cfs/cdirs/m2616/alexm/conda/envs/omnilearned-fpga` has the same pins plus
  Brevitas, torch and the same hls4ml fork. It can do QAT, export and `convert.py` C-sim (g++ only, no
  `--synth`).
- **Vitis HLS 2024.1:** `source /afs/slac/g/reseng/xilinx/2024.1/Vitis_HLS/2024.1/settings64.sh`
  (`/u1/alexmay/setup.sh` only sources Vivado).
- Run synthesis only on rdsrv409, because of memory limits elsewhere. Kill synthesis if it runs longer
  than 1 hour.

## Running

On rdsrv409, from `synthesis/`:

```bash
source /sdf/group/atlas/sw/conda/etc/profile.d/conda.sh
conda activate /u1/alexmay/conda/envs/omnilearned-hls
source /afs/slac/g/reseng/xilinx/2024.1/Vitis_HLS/2024.1/settings64.sh   # only needed for --synth
python convert.py            # convert + C-sim: random smoke test + real-jet metrics
python convert.py --synth    # also Vitis HLS csynth
```

Options:

- `--onnx PATH` (default `onnx_graphs/..._8bit_fullQuant_clean.onnx`).
- `--io-type {io_stream,io_parallel}` (default `io_stream`).
- `--strategy {Resource,Latency}` (default `Resource`), applied to the model and every layer.
- `--reuse-factor N` (default 1).
- `--pf N` (default 16, `io_parallel` only): `ParallelizationFactor` of the four per-particle
  PointwiseConv1D layers (`Dense_MatMul_0..3`, configured via `MatMul_0..3`).
  - This is how many of the 64 particles are processed in parallel. It must divide 64.
  - 64 means fully unrolled (~400k multipliers, which will not fit the VU13P).
  - PF=1 would loop over the particles and give no latency gain.

io_parallel run. Check C-sim parity against the qonnx line first, then synthesize with a 1 h cap:

```bash
python convert.py --io-type io_parallel --strategy Latency --pf 16 2>&1 | tee logs/convert_fullQuant_parallel_pf16.txt
timeout 1h python convert.py --io-type io_parallel --strategy Latency --pf 16 --synth 2>&1 | tee logs/synth_fullQuant_parallel_pf16.txt
```

Each graph and option set writes its own project,
`hls_prj/deepsets_distillnet_8bit[<graph suffix>]_<io>_<strategy>_rf<N>[_pf<N>]`, so runs never overwrite
each other. The graph suffix is the part of the file name after `_8bit`, e.g. `_fullQuant`.

- Synthesis report: `<project>/deepsets_prj/solution1/syn/report/deepsets_csynth.rpt`, with per-layer
  reports in the same directory.
- The full hls4ml config used is saved as `<project>/hls4ml_config.yml`.
- `hls_prj/` and `logs/` are gitignored.

### `convert.py` config

- Vitis backend, part `xcvu13p-flga2577-2-e`. io type, strategy, reuse and PF come from the options above.
- Default precision is `fixed<16,6>`. It is only a fallback: every type on the full-quant graph is set
  from the graph or inferred by hls4ml.
- **Input:** the type of the input `Quant` (`fixed<8,*,RND_CONV,SAT>`, an 8-bit input port). The host
  conversion is then that Quant (round-half-even, like qonnx).
- **Tanh fed by a `Quant` and feeding a `Quant`:**
  - One LUT entry per input code: `TableSize = 8 / input scale`, which is 256 for scale 1/32.
  - `table_t` and result are the output `Quant`'s type, which makes the LUT exact.
  - Otherwise (the first QAT graph): `TableSize 4096`, `table_t fixed<18,2>`, result `fixed<16,2>`.
- **Pool:** the accumulator and result get 6 more fractional bits than the pooled tensor, so the /64 is
  exact. The accumulator also gets 6 more integer bits, so the 64-particle sum cannot overflow.
- **ReLU not fused with a following `Quant`:** gets its input's type. hls4ml does not infer ReLU types,
  and the `fixed<16,6>` default truncates. This happens when a power-of-2 alpha other than 1 sits between
  them.
- The pool and ReLU types are read from a first conversion (no compile), then the model is converted
  again.

### hls4ml bugs patched in `convert.py`

The graph does not convert correctly with the fork as-is. `convert.py` monkeypatches these (each is
commented in the script):

1. `ScaleDownAdd` rebuilds `ApplyAlpha` with `NamedType` precision attributes, which
   `WeightVariable.update_precision` rejects (crash).
2. `FuseBatchNormalization` multiplies a `[particles, n_out]` broadcast scale into the `[n_in, n_out]`
   weight of a rank-3 Dense.
3. `move_scales` passes copy the old node's output `TensorVariable` into the new `ApplyAlpha`, leaving an
   untyped variable (crash).
4. `ReplaceMultidimensionalDenseWithConv` rebuilds from the pre-fusion `weight_data`/`bias_data` and
   drops the quantizers (all per-particle biases lost).
5. `move_scales` passes push an `ApplyAlpha` below a consumer without checking fan-out (the residual
   branch lost the `Add_1` bias).
6. The `Transpose` before `GlobalAveragePool` is kept as a data reorder while the pool is configured
   channels-last, so the pool averaged the wrong elements. The Transpose is dropped.

Config gotchas:

- The Tanh table type must be set via layer-level `table_t`; `Precision["table"]` is silently ignored.
- hls4ml infers no types for `GlobalPooling1D` or ReLU; see the config section above.

## Full-quant QAT (2026-09-28/29)

Training: `tools/quantize/qat_deepsets.py --full-quant --tanh-in-max 4 [--res-bits N --relu-uint]`,
warm-started from the float `distill_top_deepsets_distillnet_fpga_a05_T4` with the same KD recipe as the
first QAT. Export: `qat_deepsets_export_qonnx.py` with the same quantization flags (the checkpoint keys
depend on them). `--resume-qat` continues from an existing full-quant QAT checkpoint.

Model changes relative to the first QAT graph:

- **Gamma folded:** each DynamicTanh's gamma is folded into the next Linear's weight columns. This
  removes 3 per-channel `Mul`s, two of them per-particle.
- **DynamicTanh becomes `tanh(Quant8(alpha_po2 * x))`:**
  - alpha is rounded to a power of 2 with a straight-through estimator, so the Mul is a shift.
  - The embed norm's alpha rounding is absorbed exactly into `embed.fc1`, since ReLU is positively
    homogeneous. Without that, the warm start drops from 0.93 to 0.74 accuracy.
  - The tanh input `Quant` has a fixed range [-4, 4) with scale 1/32. Clipping at 4 costs nothing,
    because 8-bit tanh(4) already rounds to the top output code.
  - A learned range drifted to scale 1/4 on the residual-stream norms (only ~16 codes across the steep
    part of tanh) and to 1/128 on the embed norm (a 1024-entry table with 256 entries used).
  - The tanh output goes straight into the next layer's input `Quant`.
- **Biases:** quantized with `Int16Bias` (scale = input scale x weight scale, i.e. on the accumulator
  grid).

Added in r6–r8 (2026-09-29), resumed from r4:

- **`--res-bits N`:** a power-of-2 `QuantIdentity` on the embed output, after the phi residual add (the
  pool input) and after the pool. Without it, hls4ml carries these at full accumulator width.
- **`--relu-uint`:** `phi.fc2` and `rho.fc2` inputs follow a ReLU, so they use
  `Uint8ActPerTensorFixedPoint` (one more bit of resolution at 8 bits).
- `convert.py` maps an unsigned `Quant` to `ufixed`. No other HLS-side changes were needed.

QAT runs used 1 GPU node each (~215–250 s/epoch). All are bit-exact in HLS C-sim (100% argmax agreement,
max |Δ logit| = 0, 9984 jets).

- Graphs: `..._8bit_fullQuant_r<N>[_clean].onnx` in the Perlmutter qonnx directory.
- Checkpoints: `/pscratch/sd/a/alexmay/omnilearned/checkpoints`. r1's checkpoint is the one without a
  suffix.

| run | residual / pool bits | schedule | acc | AUC | 1/eB @ eS=0.5 |
|-----|----------------------|----------|-----|-----|---------------|
| float model (no quant) | — | — | 0.9212 | 0.9770 | — |
| first QAT graph (`_8bit`, not bit-exact) | — | — | 0.9181 | 0.9755 | 169.3 |
| r1 (learned tanh range) | — | 15 ep, lr 5e-5 | 0.9169 | 0.9736 | 153.9 |
| r2 | — | 15 ep, lr 5e-5 | 0.9142 | 0.9746 | 123.9 |
| r3 | — | 30 ep, lr 1e-4 | 0.9185 | 0.9736 | 127.0 |
| r4 | — | r2 + 15 ep, lr 2e-5 (`--resume-qat`) | 0.9176 | 0.9745 | 127.0 |
| r5 | — | r4 + 9 ep, lr 2e-5, KD alpha/beta 0.2/0.8 | 0.9183 | 0.9742 | 130.3 |
| r6 | 12, `--relu-uint` | r4 + 15 ep, lr 2e-5 | 0.9161 | 0.9744 | 127.0 |
| **r7 (current)** | 10, `--relu-uint` | r4 + 15 ep, lr 2e-5 | 0.9173 | 0.9744 | 123.9 |
| r8 | 8, `--relu-uint` | r4 + 15 ep, lr 2e-5 | 0.9144 | 0.9724 | 130.3 |

All runs except r1 use the fixed tanh-input range [-4, 4). r7 is copied to
`..._8bit_fullQuant[_clean].onnx`; r4 is kept as `onnx_graphs/..._fullQuant_r4_clean.onnx` for comparison.

- The fixed tanh range recovers about half of the AUC lost in r1 (0.9736 to 0.9745).
- A 10-bit residual stream and pool cost nothing measurable (r7 vs r4); even untrained, r4 with 10-bit
  quantizers inserted gives AUC 0.9745. 8 bits costs 0.002 AUC and QAT does not recover it.
- The remaining 0.001 AUC gap to the first QAT graph did not close with longer training, a lower-lr
  continuation or more KD weight.
- At this level, the r2/r4/r5 differences are within checkpoint-selection noise (val loss jumps by
  ~0.01 between epochs).
- 1/eB at eS=0.5 rests on ~30 background jets, so its statistical error is ~±19%. AUC and argmax
  agreement are the better measures.

C-sim of the current graph, identical for io_stream/Resource and io_parallel/Latency PF=16:

| metric           | qonnx  | HLS C-sim |
|------------------|--------|-----------|
| accuracy         | 0.9173 | 0.9173    |
| AUC              | 0.9744 | 0.9744    |
| argmax agreement | —      | 100%      |
| max \|Δ logit\| | —      | 0 (bit-exact) |

The full-quant graph has not been synthesized yet. Resource savings are expected from:

- 3 fewer multiplier layers.
- Power-of-2 alpha, which is a shift.
- Three 256 x 8-bit tanh tables instead of 4096 x 18-bit.
- An 8-bit input.
- `fixed<8,*>` activations everywhere a `Quant` sits.
- The residual stream and pool (r4 → r7, io_stream types): residual stored as 10-bit instead of
  25-bit per particle, pool accumulator `fixed<22,10>` instead of `fixed<37,18>`, pool result 10-bit
  instead of 31-bit, rho residual add 24-bit instead of 32-bit.

## First QAT graph (2026-09-26, baseline)

C-sim, 9984 jets: HLS agreement 97.5%, mean / max |Δ logit| 0.26 / 1.94, and HLS AUC 0.9751 vs qonnx
0.9755. The mismatch came from the tensors this graph leaves in float (biases, alpha/gamma, the tanh LUT,
the input and pool types). DynamicTanh's alpha ≈ 8–10 and gamma up to 5.9 amplified those errors enough
to flip 8-bit `Quant` rounding. Widening every type to `fixed<48,20>` only reached 99.7%. This is what
the full-quant QAT fixes.

Vitis HLS csynth (xcvu13p-flga2577-2-e, 5 ns clock, io_stream, Resource, ReuseFactor 1, ~4.5 min):

- Latency: 120 cycles = 0.6 µs.
- II: 68 cycles (particles stream in).
- Estimated clock: 3.65 ns.
- Resources: BRAM_18K 628 (11%), DSP 1334 (10%), FF 124,650 (3%), LUT 339,799 (19%). That is
  46% / 43% / 14% / 78% of one SLR.

These are HLS estimates; no Vivado logic synthesis has been run yet. They are the baseline for the
full-quant graph.

## Next steps

Roughly in priority order.

### Synthesis of the full-quant graph

1. On rdsrv409, run `convert.py --synth` on the full-quant graph (io_stream, and io_parallel PF=16 with
   the 1 h cap). Compare against the baseline above.
2. Run Vivado logic synthesis (`hls_model.build(..., vsynth=True)`) for real post-synthesis resource and
   timing numbers instead of HLS estimates.

### Further quantization (NERSC side, needs QAT)

3. **Narrower pre-requantization sums.** The residual `Add`s are still computed at the MatMul accumulator
   width (24-bit) before the 10-bit `Quant`. An output quantizer on `phi.fc2` / `rho.fc2` would shrink
   the adders; check whether HLS already trims them first.
4. **7-bit unsigned inputs after ReLU** (same resolution as the old signed 8-bit).
5. **Optional output quantizer** on the logits (now `fixed<22,8>`), if the output port width matters
   downstream.
6. **Lower weight bit-widths** (e.g. 6-bit, or 4-bit where tolerated) for DSP/LUT savings. This is a
   separate precision-vs-AUC scan.

### Resources / latency (HLS side)

7. Raise `ReuseFactor` (e.g. 2, 4, 8) on the per-particle PointwiseConv1D layers to cut DSP/LUT, traded
   against latency and II.
8. Check whether the BRAM usage is mostly stream FIFOs. If so, run hls4ml FIFO depth optimization (the
   `fifo_depth_optimization` flow).
9. Try a different clock target if the application needs it.
10. Revisit the fixed batch dim of 64 in the export if the target interface wants per-jet streaming. The
    HLS model already processes one jet per call.

### Housekeeping

11. Report the six hls4ml bugs, the ignored `Precision["table"]` and the missing `GlobalPooling1D`/ReLU
    type inference upstream (or to the qibin2020 fork), so `convert.py` can drop its monkeypatches.
