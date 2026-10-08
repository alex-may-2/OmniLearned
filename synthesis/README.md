# hls4ml conversion + synthesis

Notes for the agent or person running the FPGA step.

- Training, QAT and QONNX export happen on NERSC (Perlmutter).
- `convert.py` C-sim runs on Perlmutter or on SLAC rdsrv409.
- Vitis HLS synthesis runs only on rdsrv409.

**Single-SLR design (one jet every 25 ns, one SLR):** `io_parallel_report.md`. Current design (2026-10-04):
**H4-d18s4stk**, d18p2r1m1 n16 (mlp_ratio 1), QAT `--round floor --bits 9 --in-bits 12 --tanh-in-max 2`, io_parallel
PF 2 at 360 MHz: II 22.2 ns, 69% of one SLR's LUT, 36% DSP, HLS AUC 0.9748 on all 404k test jets, C/RTL cosim exact
(2000 jets). Run log: `parallel_search_log.md`. Build flags: `convert.py --name <stem> --io-type io_parallel --strategy
Latency --pf 2 --clock 2.78 --mult-limit-fix --clone-fanout --reshape-channels --dsp-mult --auto-rewind`.
Previous single-SLR designs (H1g, H2-d16s4, H3-*) are in "Full test set comparison" below.

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

Full test set: `top_test_full_n64.npz` (gitignored, 417 MB; also in the Perlmutter qonnx directory).

- `x`: float32 `[404000, 64, 4]`, `data[:, :64]` of `/global/cfs/cdirs/m4567/www/top/test/test_ttbar.h5`
  (no clipping, same as the 10k file, whose rows are its first 10000). `y`: int64 `[404000]`.
- On 9984 jets, 1/eB at eS = 0.3 / 0.5 rests on only ~15 / ~30 background jets. Use the full set to rank models.

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
- `--name STEM`: project stem, `hls_prj/deepsets_<STEM>_<opts>` (see below). Pass it for every trained graph.
- `--clock NS` (default 5.0): target clock period. The single-SLR builds use 2.78 (360 MHz).
- `--mult-limit-fix`, `--clone-fanout` (io_parallel): the II fixes in `io_parallel_report.md` §3. Always on.
- `--reshape-channels` (io_parallel): `ARRAY_RESHAPE` instead of `ARRAY_PARTITION` on inter-layer arrays that no
  pointwise conv touches, so each layer boundary is one DATAFLOW channel instead of one FIFO per element. Pragmas
  only (C-sim unchanged); H1g 286k → 234k LUT, latency 100 → 34 cycles.
- `--dsp-mult` / `--dsp-add`: bind multiplies (and adds/subs) to DSPs via `config_op`. Synthesis only.
- `--auto-rewind` (io_parallel, 2026-10-03): fixes the C/RTL mismatch. Drops hls4ml's explicit conv `rewind`
  (`nnet_conv1d_latency.h`) and adds an input-capture process in front of the first layer, so Vitis auto-rewinds
  every conv. C-sim and II unchanged, 4-5% less LUT. Use it on every io_parallel build, and still run the cosim
  check below: one build (d16 r2 floor) mismatched even with it.
- The rounding of each activation type follows the QONNX `Quant` node's `rounding_mode`: `ROUND` →
  `AP_RND_CONV`, `FLOOR` → `AP_TRN` (QAT `--round floor`). No flag needed.

io_parallel run. Check C-sim parity against the qonnx line first, then synthesize with a 1 h cap:

```bash
python convert.py --io-type io_parallel --strategy Latency --pf 16 2>&1 | tee logs/convert_fullQuant_parallel_pf16.txt
timeout 1h python convert.py --io-type io_parallel --strategy Latency --pf 16 --synth 2>&1 | tee logs/synth_fullQuant_parallel_pf16.txt
```

Mini random-weight model for io_parallel sizing (see "Mini-model scan" below):

```bash
python mini_parallel.py --n 16 [--phi 64,32 --rho 32 --pf 16] [--synth]         # ReLU only (first scan)
python mini_parallel.py --full-quant --n 16 --phi 32,16 --rho 16 --pf 8           # + Int16 biases, quantized tanh
python mini_parallel.py --distillnet --n 32 --dim 16 --pf 2 [--phi-blocks 1 --rho-blocks 1]  # real student topology
```

Quant-recipe and debug flags of `mini_parallel.py --distillnet` (2026-10-03; defaults = the 8-bit round recipe):
`--w-bits`, `--a-bits`, `--in-bits`, `--tanh-bits`, `--tanh-in-max`, `--round {round,floor}` mirror the QAT flags
for HLS cost probes (project suffix only for non-defaults, e.g. `_w9a9i12tm2_fl`). `--cut <onnx tensor>` ends the
graph at that tensor (suffix `_cut<name>`), used to bisect C/RTL mismatches layer by layer. It also takes
`--reshape-channels --dsp-mult --dsp-add --auto-rewind` like `convert.py`.

Distillnet-topology synths need ~30 GB at n x dim = 512 and more than 55 GB at 1024. Run those alone.

Each graph and option set writes its own project, `hls_prj/deepsets_<stem>_<io>_<strategy>_rf<N>[_pf<N>][_clk<ns>][_mlf][_clone][_dsp][_dspadd][_rsh][_arw]`.

- `<stem>` is `--name` if given, else `distillnet_8bit<graph suffix>` (the part of the file name after `_8bit`, e.g. `_fullQuant`).
- All full-quant graphs share the suffix `_fullQuant`, so without `--name` a second graph overwrites the first one's project.
  Pass `--name` for every graph except r7. Convention: the training save tag without `qat_` and the epoch, e.g.
  `--name ps_d12p2r1m1_n16` gives `hls_prj/deepsets_ps_d12p2r1m1_n16_io_parallel_latency_rf1_pf2_clk2.78_mlf_clone`.
- Projects made before `--name` existed were renamed by hand (e.g. `deepsets_ps_d12p2r1m1_n16_io_parallel_pf2_clk2.78_mlf_clone`).

- Synthesis report: `<project>/deepsets_prj/solution1/syn/report/deepsets_csynth.rpt`, with per-layer
  reports in the same directory.
- The full hls4ml config used is saved as `<project>/hls4ml_config.yml`.
- `hls_prj/` and `logs/` are gitignored.

Full-test-set forward pass of a built project's C-sim library (io_parallel or io_stream; no hls4ml or Vitis needed):

```bash
python csim_forward.py hls_prj/<project> [--npz top_test_full_n64.npz]
```

- It reruns the project's `build_lib.sh` first: the library bakes in the absolute weights path, which a
  project rename breaks.
- It prints acc, AUC and 1/eB at eS = 0.3, 0.5, 0.7, and saves the logits as `<project>/csim_forward_<npz>.npz`.
- One core, about 50 s for 404k jets at n = 16 (io_parallel), a few minutes at n = 64 (io_stream).

C/RTL co-simulation (the hls4ml testbench always reports "Pass", so compare the logs instead):

```bash
/tmp/alexmay_ps/cosim.sh TAG hls_prj/<project> 200     # SYNTH=1 to csynth first
```

- It writes 200 full-test jets into `tb_data/`, sets csim + cosim in `build_opt.tcl`, runs Vitis under
  `synth.lock`, restores `build_opt.tcl` and compares C-sim against RTL (`cosim_tb.py detail` lists differing
  elements). Exact means `max|d| 0`, 200/200 rows equal.
- `/tmp/alexmay_ps/` can vanish; the scripts (`cosim.sh`, `cosim_tb.py`, `h2.sh`) are kept in the local
  `helpers/` folder of the launching machine. `h2.sh` runs convert + synth, `csim_forward.py` and the cosim in one go.

### `convert.py` config

- Vitis backend, part `xcvu13p-flga2577-2-e`. io type, strategy, reuse and PF come from the options above.
- Default precision is `fixed<16,6>`. It is only a fallback: every type on the full-quant graph is set
  from the graph or inferred by hls4ml.
- **Input:** the type of the input `Quant` (`fixed<W,*,RND_CONV,SAT>`, or `AP_TRN` for a `FLOOR` Quant; 8 bits in
  r7, 12 in H4). The host conversion is then that Quant, rounded like qonnx.
- **Tanh fed by a `Quant` and feeding a `Quant`:**
  - One LUT entry per input code: `TableSize = 8 / input scale`, which is 256 for scale 1/32.
  - `table_t` and result are the output `Quant`'s type, which makes the LUT exact.
  - Otherwise (the first QAT graph): `TableSize 4096`, `table_t fixed<18,2>`, result `fixed<16,2>`.
- **Pool:** the accumulator and result get 6 more fractional bits than the pooled tensor, so the /64 is
  exact. The accumulator also gets 6 more integer bits, so the 64-particle sum cannot overflow.
- **ReLU not fused with a following `Quant`:** gets its input's type. hls4ml does not infer ReLU types,
  and the `fixed<16,6>` default truncates. This happens when a power-of-2 alpha other than 1 sits between
  them. A floor-quantized ReLU output is also TRN, so the rule also checks that the next ONNX node is not a
  `Quant` (2026-10-03).
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
first QAT. r5-r8 continued from r4 with `--resume-qat` at lr 2e-5.

Since 2026-10-01 `qat_deepsets.py` is always full-quant: `--full-quant` and `--resume-qat` are gone, and the
defaults are `--tanh-in-max 4 --res-bits 10 --relu-uint`. QAT reads the model shape from the float checkpoint's
`arch_config`. Export (`qat_deepsets_export_qonnx.py --tag <qat_tag>`) and `qat_deepsets_eval.py` rebuild the
model from the QAT checkpoint's `arch_config`.

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

## io_parallel

The 64-particle graph does not fit io_parallel; the single-SLR design uses a smaller student. History (tanh blocker,
mini-model cost rules), the II and timing fixes, model knobs and the reproduce commands: `io_parallel_report.md`.

## Full test set comparison (2026-10-01, single-SLR rows updated 2026-10-04)

All 404k test jets (`top_test_full_n64.npz`). HLS rows are `csim_forward.py` C-sim, i.e. the exact
hardware output. Float rows are torch on the checkpoint; "Brevitas" is fake-quant QAT in torch.

| model | N | run as | acc | AUC | 1/eB @ eS=0.3 | @ 0.5 | @ 0.7 |
|---|---|---|---|---|---|---|---|
| twamorka `distill_top_deepsets_distillnet_fpga_a05_T4` (d32p2r1, float) | 64 | float | 0.9253 | 0.9786 | 706 | 190 | 58.6 |
| first QAT graph `_8bit` (same model, weights + Linear inputs only) | 64 | HLS | 0.9165 | 0.9756 | 523 | 143 | 46.3 |
| r7 (same model, full-quant QAT) | 64 | HLS | 0.9205 | 0.9757 | 540 | 143 | 47.7 |
| d12p2r1m1 float KD (`ps_d12p2r1m1_n16_e50`) | 16 | float | - | 0.9731 | 377 | 119 | 43.4 |
| **H4-d18s4stk** d18p2r1m1, QAT floor + 9-bit + 12-bit input + tanh ±2 (current design) | 16 | HLS | 0.9191 | **0.9748** | 404 | 130.8 | 46.3 |
| H4-d18r2s2stk d18p2r2m1, same QAT recipe | 16 | HLS | 0.9185 | 0.9745 | 388 | 125.3 | 45.7 |
| H3-d16flb9 d16p2r1m1, QAT floor + 9-bit | 16 | HLS | 0.9166 | 0.9734 | 367 | 115.8 | 44.0 |
| H3-d18s4 d18p2r1m1, QAT floor | 16 | HLS | 0.9151 | 0.9727 | 348 | 113.1 | 41.9 |
| H2-d16s4 d16p2r1m1, r7 recipe | 16 | HLS | 0.9132 | 0.9719 | 340 | 109.1 | 40.9 |
| H1g d12p2r1m1 full-quant (design until 2026-10-03) | 16 | HLS | 0.9106 | 0.9697 | 278 | 91.5 | 35.7 |
| d8p2r1 float KD (`ps_d8p2r1_n16_e50`) | 16 | float | - | 0.9685 | 274 | 88.1 | 33.0 |
| twamorka `qat_top_deepsets_mac_d8p2r1_n16_a05_T4_8bit_po2` | 16 | Brevitas | 0.9086 | 0.9677 | 262 | 85.3 | 31.4 |
| H1r d8p2r1 full-quant (smaller fallback) | 16 | HLS | 0.9074 | 0.9666 | 234 | 78.1 | 30.0 |
| H1f = H1r + `--dsp-mult` | 16 | HLS | 0.9074 | 0.9666 | 234 | 78.1 | 30.0 |

- The first QAT graph is not bit-exact in HLS (97.5% argmax agreement), yet on all 404k jets its HLS AUC equals
  r7's. Its C-sim runs at ~190 jets/s (wide types, 4096-entry tanh), so the full set takes ~35 min.
- Rejection falls much faster than AUC. From the float d32 n64 model to H1g, AUC drops 0.009 and
  1/eB at eS=0.5 roughly halves (190 to 91.5).
- Shrinking costs more than quantizing. d32 n64 float to d12 n16 float costs 37% of 1/eB@0.5;
  8-bit full-quant then costs another 23%.
- The twamorka mac checkpoint uses the partial recipe of the first QAT graph (weights and Linear inputs
  only; DyT, residual and pool in float). It beats H1r (same shape) in software but has no HLS number,
  and that recipe was not bit-exact in HLS. Its `po2` option is not recorded in `arch_config`.
- r7's input `Quant` (8-bit, 1/32) saturates at 3.97, so log pT / log E above that (typical values ~5)
  are clipped. H1g learned 1/16 (range +-8). Not measured how much this costs r7. The d16 r7-recipe graph also
  learned ±8; its 12-bit input (H4) keeps ±8 at step 1/256, so the 2026-10-04 input-bits gain is resolution, not range.
- Scripts for the non-HLS rows are outside the repo on Perlmutter, in
  `/pscratch/sd/a/alexmay/omnilearned/logs/ps/`: `eval_float_ckpt.py`, `eval_qat_ckpt.py`
  (strict-loads a non-full-quant QAT checkpoint from its `arch_config`), `qonnx_forward.py` (128-process
  QONNX run on a CPU node).

## Next steps

Roughly in priority order. Hardware next steps (Vivado P&R of H4, auto-rewind root cause): `io_parallel_report.md` §6.

### Further quantization (NERSC side, needs QAT)

Done 2026-10-03/04 (run log QP-QZ4, H3-*, H4-*): floor rounding (−10% LUT, AUC unchanged), 9-bit weights and
activations (+0.002 AUC), 12-bit input, tanh input range ±2. Fewer bits hurt: a7 −0.003, a6 −0.010, w6 −0.010,
w4 broken; `--res-bits` 8/12 and `--no-relu-uint` gave nothing; 10-bit = 9-bit.

1. **Bias clipping.** `Int16Bias` sits on the accumulator grid (`s_input × s_weight`). In H4 the 12-bit input
   makes embed fc1's grid 2^-17, so its bias range is ±0.25 and 1 of 18 biases clips; 3 of 18 phi.fc1 biases clip
   at ±1. QAT, C-sim and RTL all clip the same way (not a correctness bug). Fix: a `--bias-bits` QAT flag
   (default 16), e.g. 20; cost should be ~0 LUT.
2. **Input bits.** Only 8 / 10 / 12 were trained, on d16 (10 and 12 tie). Try `--in-bits 9` / `10` with the stk
   recipe on d18 s4.
3. **Per-layer precision** (more bits in rho/out, which are per jet and cheap; fewer in embed/phi). Not tried.
4. **Narrower pre-requantization sums.** The residual `Add`s are still computed at the MatMul accumulator
   width (24-bit) before the 10-bit `Quant`. An output quantizer on `phi.fc2` / `rho.fc2` would shrink
   the adders; check whether HLS already trims them first.
5. **Optional output quantizer** on the logits (now `fixed<24,8>`), if the output port width matters
   downstream.

### Housekeeping

6. Report the six hls4ml bugs, the explicit conv `rewind` C/RTL bug (worked around by `--auto-rewind`), the ignored `Precision["table"]` and the missing `GlobalPooling1D`/ReLU
    type inference upstream (or to the qibin2020 fork), so `convert.py` can drop its monkeypatches.
