# hls4ml conversion + synthesis (off-NERSC machine with Vitis/Vivado)

Notes for the agent/person running the FPGA step. Training, QAT and QONNX
export happen on NERSC; this machine (SLAC rdsrv409) only converts and
synthesizes. Graph changes must be made on the PyTorch side in
`src/omnilearned/network.py` and re-exported on NERSC with
`tools/quantize/qat_deepsets_export_qonnx.py`.

## Input

`onnx_graphs/qat_top_deepsets_distillnet_fpga_a05_T4_8bit_clean.onnx` (55 KB), copied from
`/pscratch/sd/a/alexmay/omnilearned/qonnx/fpga/` on Perlmutter.

- Model: DeepSets "distillnet" student (base_dim 32, phi 2 layers, rho 1 layer,
  ReLU, DynamicTanh norm), ~11k params, KD'd from `fine_tune_top_l`, then 8-bit
  Brevitas QAT (power-of-2 scales, per-tensor fixed point).
- Task: top-tagging, 2 logits.
- Input `global_in`: `[64, 64, 4]` = (batch, particles, features). Particles are
  the leading-pT 64 slots of the 150-slot dataset, no mask. Batch dim is fixed
  to 64 from the export dummy input.
- Output: `[64, 2]` logits.
- Ops (46 nodes): Quant x14, Add x6, Mul x6, Transpose x5, MatMul x4, Gemm x3,
  Relu x3, Tanh x3, Flatten, GlobalAveragePool. Mean pool is
  `Transpose` + `GlobalAveragePool` (not `ReduceMean`).
- Only weights (8-bit) and the inputs of each MatMul are `Quant`-ed. Biases,
  DynamicTanh alpha/gamma, ReLU/Tanh outputs, residual adds and the pool are
  float in the QONNX graph, so HLS must choose their fixed-point types.
- Verified on NERSC: onnx checker OK, IR version 9, every Quant 8-bit,
  qonnx-executor vs PyTorch max |Δ logit| = 0, 100% argmax agreement.

Real-jet test set: `top_test_10k_n64.npz` (also on Perlmutter next to the ONNX).
`x`: float32 `[10000, 64, 4]`, first 10k jets of the top-tagging test split,
same preprocessing as training. `y`: int `[10000]` labels (5080 / 4920).
Evaluated in batches of 64 (9984 jets = 156 full batches).

## Environment

- Conda env: `/u1/alexmay/conda/envs/omnilearned-hls` (Python 3.11,
  `qonnx==1.0.0`, `onnx==1.20.1`, `onnxruntime==1.30.0`,
  `onnxoptimizer==0.4.2`, scikit-learn, hls4ml fork
  `git+https://github.com/qibin2020/hls4ml.git@fix/onnx-frontend-shape-bugs`
  at `1d85133`). onnx>=1.21 writes IR 14, which onnxruntime 1.30 rejects.
  Stock hls4ml misreads rank-3 MatMul / Reshape shapes, hence the fork.
- Vitis HLS 2024.1: `source /afs/slac/g/reseng/xilinx/2024.1/Vitis_HLS/2024.1/settings64.sh`
  (`/u1/alexmay/setup.sh` only sources Vivado).
- Run synthesis only on rdsrv409 (memory limits elsewhere). Kill synthesis
  if it runs longer than 1 hour.

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

- `--io-type {io_stream,io_parallel}` (default `io_stream`)
- `--strategy {Resource,Latency}` (default `Resource`), applied to the model and every layer
- `--reuse-factor N` (default 1)
- `--pf N` (default 16, `io_parallel` only): `ParallelizationFactor` of the four
  per-particle PointwiseConv1D layers (`Dense_MatMul_0..3`, configured via
  `MatMul_0..3`), i.e. how many of the
  64 particles are processed in parallel. Must divide 64; 64 = fully unrolled
  (~400k multipliers, will not fit the VU13P). Default PF=1 would loop over
  particles and give no latency gain.

io_parallel run (check C-sim parity against the qonnx line first, then synthesize
with a 1 h cap):

```bash
python convert.py --io-type io_parallel --strategy Latency --pf 16 2>&1 | tee logs/convert_parallel_pf16.txt
timeout 1h python convert.py --io-type io_parallel --strategy Latency --pf 16 --synth 2>&1 | tee logs/synth_parallel_pf16.txt
```

Each option set writes its own project,
`hls_prj/deepsets_distillnet_8bit_<io>_<strategy>_rf<N>[_pf<N>]`, so runs never
overwrite each other (`hls_prj/` is not in git). The first run's project is kept
at `hls_prj/deepsets_distillnet_8bit`. Synthesis report:
`<project>/deepsets_prj/solution1/syn/report/deepsets_csynth.rpt` (per-layer
reports in the same directory); the full hls4ml config used is saved as
`<project>/hls4ml_config.yml`. Logs go in `logs/`.

`convert.py` config: Vitis backend, part `xcvu13p-flga2577-2-e`, io type /
strategy / reuse / PF from the options above (first run: `io_stream`, `Resource`), default `fixed<16,6>`, Tanh
`TableSize 4096` / `table_t fixed<18,2>` / result `fixed<16,2>`, pool
accumulator `fixed<32,19>`.

### hls4ml bugs patched in `convert.py`

The graph does not convert correctly with the fork as-is. `convert.py`
monkeypatches (each commented in the script):

1. `ScaleDownAdd` rebuilds `ApplyAlpha` with `NamedType` precision attributes,
   which `WeightVariable.update_precision` rejects (crash).
2. `FuseBatchNormalization` multiplies a `[particles, n_out]` broadcast scale
   into the `[n_in, n_out]` weight of a rank-3 Dense.
3. `move_scales` passes copy the old node's output `TensorVariable` into the
   new `ApplyAlpha`, leaving an untyped variable (crash).
4. `ReplaceMultidimensionalDenseWithConv` rebuilds from the pre-fusion
   `weight_data`/`bias_data` and drops the quantizers (all per-particle biases lost).
5. `move_scales` passes push an `ApplyAlpha` below a consumer without checking
   fan-out (the residual branch lost the `Add_1` bias).
6. The `Transpose` before `GlobalAveragePool` is kept as a data reorder while the
   pool is configured channels-last (pool averaged the wrong elements); the
   Transpose is dropped.

Config gotchas: the default `fixed<16,6>` pool accumulator overflows on the
64-particle sum (wraps, mean off by exactly 1.0); Tanh table precision must be
set via layer-level `table_t` (`Precision["table"]` is silently ignored).
These patches are worth reporting upstream.

## Status (first run, 2026-09-26)

C-sim, 9984 real jets (current `convert.py`):

| metric           | qonnx  | HLS C-sim |
|------------------|--------|-----------|
| accuracy         | 0.9181 | 0.9169    |
| AUC              | 0.9755 | 0.9751    |
| 1/eB @ eS=0.5    | 169.3  | 141.1     |
| argmax agreement | —      | 97.5%     |
| mean / max \|Δ logit\| | — | 0.26 / 1.94 |

1/eB at eS=0.5 rests on ~29 background jets, so its statistical error is ~±19%;
AUC and argmax agreement are the better parity measures.

Vitis HLS csynth (xcvu13p-flga2577-2-e, 5 ns clock, ReuseFactor 1, ~4.5 min):
latency 120 cycles = 0.6 µs, II 68 cycles (particles stream in), estimated
clock 3.65 ns. Resources: BRAM_18K 628 (11%), DSP 1334 (10%), FF 124,650 (3%),
LUT 339,799 (19%); 46% / 43% / 14% / 78% of one SLR. These are HLS estimates
(no Vivado logic synthesis yet). This run used the old Tanh table type
(`fixed<18,8>`); same width as now, so resources should barely change.

### Source of the remaining HLS vs qonnx discrepancy

No conversion bug remains; it is fixed-point precision on the tensors the
QONNX graph leaves in float:

- Biases, alpha/gamma constants, the input, `Relu_0` and pool outputs fall
  back to `fixed<16,6,TRN>` (up to ~0.001 truncation error each).
- Tanh is a lookup table (step 8/4096, floor indexing, saturates at |x| > 4),
  never bit-exact to float tanh.
- DynamicTanh multiplies these ~1e-3 errors by alpha ≈ 8 (`Mul_2`) and 9.85
  (`Mul_4`), then by gamma up to 5.9, so they flip 8-bit `Quant` rounding.

Ablation (9984 jets; "wide" = `fixed<48,20>` or 65536-entry 34-bit Tanh table):

| variant                          | mean \|Δ\| | argmax | acc    | AUC    |
|----------------------------------|-----------|--------|--------|--------|
| baseline                         | 0.263     | 97.48% | 0.9169 | 0.9751 |
| biases + alpha/gamma wide        | 0.208     | 98.16% | 0.9191 | 0.9755 |
| input/ReLU/pool outputs wide     | 0.220     | 97.88% | 0.9179 | 0.9753 |
| both of the above                | 0.146     | 98.75% | 0.9186 | 0.9756 |
| Tanh table only wide             | 0.157     | 98.77% | 0.9200 | 0.9754 |
| everything wide                  | 0.042     | 99.68% | 0.9175 | 0.9755 |

## Next steps

Roughly in priority order.

### Precision / parity (HLS side, no retraining)

1. Give the non-`Quant` types a few more fractional bits instead of the
   blanket `fixed<16,6>`: e.g. biases and alpha/gamma `fixed<18,4>` or wider,
   ReLU/pool outputs matched to their real range. Use rounding (`RND`) instead
   of truncation (`TRN`) on these types; truncation biases every value downward.
2. Tanh: try a finer table and `table_t` with rounding, and check how much
   the saturation at |x| > 4 matters (alpha ≈ 8 puts many inputs there).
3. Use `hls4ml.model.profiling` on the real jets to size each type to its
   actual range, then re-check parity. Target: AUC and argmax agreement match
   the "both wide" rows at minimal extra bits.
4. Re-run `convert.py --synth` after the precision changes (the current synth
   numbers predate the Tanh `table_t` fix).

### Retraining (NERSC side) for bit-exact HLS

5. Quantize biases in Brevitas QAT (`bias_quant`, power-of-2 scale) so QONNX
   pins their type.
6. Quantize DynamicTanh: `QuantIdentity` on `alpha * x` (8-bit, power-of-2
   scale, e.g. 1/32 covering [-4, 4)) and `qnn.QuantTanh` with a power-of-2
   output quantizer (`Int8ActPerTensorFixedPoint`). Then set HLS
   `TableSize = 8 / input_scale` (256 for scale 1/32) and `table_t` = the output
   quant type with rounding, which makes the Tanh LUT exact. Quantize gamma or
   fold it into the following `Quant` scale.
7. With 5 and 6, every float op between `Quant` nodes is gone and HLS should
   match qonnx bit-exactly (accumulators are already sized losslessly).
   Re-export and re-run `convert.py`; check the patches still apply.

### Resources / latency

8. Raise `ReuseFactor` (e.g. 2, 4, 8) on the per-particle PointwiseConv1D
   layers to cut DSP/LUT (LUT is 78% of one SLR); trade against latency and II.
9. Check whether the 628 BRAM are mostly stream FIFOs; run hls4ml FIFO depth
   optimization (`fifo_depth_optimization` flow) to shrink them.
10. Try `io_parallel` for comparison (latency vs resources), and a different
    clock target if the application needs it.
11. Run Vivado logic synthesis (`hls_model.build(..., vsynth=True)`) for real
    post-synthesis resource and timing numbers instead of HLS estimates.
12. Revisit the fixed batch dim of 64 in the export if the target interface
    wants per-jet streaming; the HLS model already processes one jet per call.

### Housekeeping

13. Report the six hls4ml bugs (and the ignored `Precision["table"]`) upstream
    or to the qibin2020 fork, so `convert.py` can drop its monkeypatches.
14. `hls_prj/` and `logs/` (`convert_log.txt`, `synth_log.txt`) are generated artifacts;
    add them to `.gitignore` if not wanted in the repo.
