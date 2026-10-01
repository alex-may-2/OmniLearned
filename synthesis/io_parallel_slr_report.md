# II ≈ 25 ns search for the full-quant DeepSets (2026-09-30/10-01, rdsrv409 + NERSC)

Plan: `synthesis/PARALLEL_SEARCH_PLAN.md`. Run log, one row per run: `synthesis/parallel_search_log.md`.

**Requirement.** L1T needs one jet every 25 ns. k copies of the tagger take turns on jets, one copy per SLR, so each
copy needs II ≤ 25·k ns. II ns = II cycles × clock period.

**What counts:**
- II and the one-SLR fit are hard limits.
- AUC is soft. Reference 0.9644 = r7 − 0.01.

**Caveat.** All hardware numbers are Vitis HLS 2024.1 csynth estimates: part `xcvu13p-flga2577-2-e`, % of one SLR
(432k LUT, 864k FF, 3072 DSP, 1344 BRAM_18K). There is no Vivado place and route. Clocks of 320 MHz and above are
**csynth-only, not P&R-confirmed**.

## Summary

All trained finalists are bit-exact (max |Δ logit| = 0) between qonnx and HLS C-sim on the 9984-jet test subset
used by `convert.py`. On that subset r7 scores AUC 0.9744.

| k | per-copy budget | design (trained) | family, clock | II cycles / ns | LUT / FF / DSP % SLR | clock est. | HLS AUC | status |
|---|---|---|---|---|---|---|---|---|
| 1 | 25 ns | d8p2r1, n 16 | io_parallel PF 2, 360 MHz | 8 / 22.2 | 59 / 18 / 0 | 1.96 ns | **0.9653** | pass |
| 1 | 25 ns | same graph | io_parallel PF 4, 200 MHz | 4 / 20.0 | 67 / 15 / 0 | 3.65 ns | **0.9653** | pass, more timing margin |
| 1 | 25 ns | same graph | io_parallel PF 2, 320 MHz | 8 / 25.0 | 57 / 17 / 0 | 2.28 ns | **0.9653** | pass |
| 2 | 50 ns | d8p2r1, n 16 (same graph) | io_parallel PF 1, 360 MHz | 16 / 44.5 | 55 / 17 / 0 | 1.96 ns | **0.9653** | pass |
| 2 | 50 ns | d12p1r1, n 32 | io_parallel PF 2, 360 MHz | 16 / 44.5 | 74 / 26 / 0 | 2.03 ns | 0.9566 | fits, but loses to the d8p2r1 n16 build above |
| 4 | 100 ns | d32p2r1, n 16 (r7 width) | io_stream, 240 MHz | 20 / 83.4 | 71 / 16 / 37 | 3.04 ns | **0.9712** | pass |

- **k=1 meets all three axes:**
  - II ≤ 25 ns with one copy in one SLR;
  - AUC 0.9653 ≥ 0.9644;
  - latency 99 cycles = 275 ns at 360 MHz (60 cycles = 300 ns at 200 MHz).
- **k=2 gains nothing over k=1.** A slower II does not shrink the elementwise cost, and the larger k=2-only shape
  (d12p1r1 n32, no phi block) trains worse. The best k=2 design is the k=1 graph at PF 1 (55% LUT, AUC 0.9653).
- **k=4 gives up 3 SLRs for +0.006 AUC:** io_stream keeps the r7 width; latency 384 ns.
- **Extra finalist d8p1r1 n32 (no residual phi block, 32 particles)** fits k=1 (II 22.2 ns, 51% LUT) but reaches only AUC 0.9532. Dropping the phi block costs about 0.005 AUC in float (0.9631 vs 0.9685) and loses more in QAT (about 0.01). The 9-epoch screens did not show this.
- **What made io_parallel work:**
  1. `--mult-limit-fix` (II = n/PF);
  2. fan-out clones (timing);
  3. a fast clock: at 360 MHz, PF 2 (8 cycles) needs half the parallel multipliers of PF 4 at 200 MHz (4 cycles),
     so it uses 59% vs 67% LUT.

## 1. r7 mirror (Phase 0)

`mini_parallel.py --distillnet` now builds the r7 graph element for element:
- input Quant 8b 1/32;
- embed alpha 1;
- 10-bit residual stream (embed out 1/128, phi out 1/64);
- 10-bit pool output 1/512;
- unsigned 8-bit block-ReLU Quants (phi 1/64, rho 1/128);
- 8-bit 1/128 before the output Linear.

It is C-sim bit-exact in io_parallel and io_stream.

Two conversion details had to change:
- **Embed Mul by 1.** In the mini graph, hls4ml dropped the tanh-input Quant after this Mul (max |Δ logit| 0.05).
  The mini graph therefore omits the Mul, and ReLU + Quant fuse as they do in r7.
- **Pool mean for non-power-of-2 n.** hls4ml computes `sum /= n`, truncated in the accumulator type. The exact
  mean is never closer than 1/(n·2^10) to a rounding tie of the 10-bit pool Quant. Six more fractional
  accumulator bits keep it bit-exact (checked at n = 12, 24, 28). `convert.py` does the same.

## 2. What sets II, and the fixes (Phase 1a, d8 n16)

| run | change | II cycles | clock est. | LUT % SLR | note |
|---|---|---|---|---|---|
| L0 | PF 2, 5 ns | 8 | 5.79 miss | 59 | II = n/PF |
| L1 | PF 4 | 8 | 5.79 miss | 64 | floor: 2 cycles per partition |
| L2 | PF 4 + `--mult-limit-fix` | 4 | 5.79 miss | 68 | II = n/PF again |
| L2c | L2 + fan-out clones | 4 | 3.65 | 69 | **k=1 at 200 MHz** |
| L6b | PF 2, 2.78 ns + fixes | 8 (22.2 ns) | 2.01 | 60 | **k=1 at 360 MHz** |
| L6c | PF 2, 3.125 ns + fixes | 8 (25.0 ns) | 2.28 | 58 | k=1 at 320 MHz |
| L4 | L2b + `--dsp-mult` | 4 | (pre-clone) | 60 (−11%) | 804 multiplies on DSP |
| L3 / L5 | PF = n, pipeline style | 5 / 2 | 3.5 / 3.6 | 102 / 98 | no FIFOs, but fully parallel |
| L7a/L7b | pipeline style, top II 5 / 8 | — | — | — | stalled in scheduling > 30 min with II violations; stopped |

- **II floor.** In `nnet_conv1d_latency.h`, the partition loop runs at II = RF under
  `ALLOCATION mul limit = n_chan·n_filt/RF`. That limit counts one particle, but each iteration does PF particles.
  - `--mult-limit-fix` scales the limit by `n_pixels`. II then equals n_partitions × RF.
  - Flag in `mini_parallel.py` and `convert.py`.
- **Clock miss.** Each residual skip array, and the pool output, has two readers in the DATAFLOW top (the DyT
  branch and the residual Add). Vitis then chains producer and consumers in one cycle: pool 3.0 ns plus alpha+Quant
  2.8 ns.
  - Giving each reader its own copy through a clone process removes the chain: 5.84 → 3.65 ns at 5 ns, and 2.01 ns
    at 2.78 ns.
  - Flag `--clone-fanout` in `convert.py`; always on in `mini_parallel.py` io_parallel.
  - Registering the pool output (`LATENCY min=1`) did not help.
- **Cost rule (io_parallel).** LUT is dominated by the elementwise layers and FIFOs.
  - Their cost does not drop when II is relaxed: d16 n16 at PF 1 (II 16) is still 127% of one SLR.
  - So k=2 gains no bigger io_parallel model unless operator sharing (L7) or model knobs cut it.

## 3. Model knobs (random weights, d8 n16, PF 2, 360 MHz)

| run | knob | LUT | vs L6b (263k) | keep? |
|---|---|---|---|---|
| M1 | p1: no residual phi block | 115k | −56% | hardware yes (`--size d<dim>p1r1`, no code), but costs ~0.012 AUC after full training (§5) |
| M3 | mlp_ratio 1 | 173k | −34% | yes, new size suffix `d<dim>p<p>r<r>m1` (nersc `utils.py`, +2 lines) |
| M2 | r0: no rho block | 244k | −7% | no |
| M4 | no embed DyT | 251k | −5% | no |

Shapes that the knobs let fit (random weights, 360 MHz):

| run | shape | PF | II | LUT % SLR | k |
|---|---|---|---|---|---|
| P2 | d8p1r1 n32 | 4 | 8 | 48 | 1 |
| Q1 | d12p2r1m1 n16 | 2 | 8 | 63 | 1 |
| P1 | d16p1r1 n16 | 2 | 8 | 62 | 1 |
| R2 | d12p1r1 n32 | 4 | 8 | 77 | 1 |
| R1 | d12p1r1 n32 | 2 | 16 | 68 | 2 |
| B6 | d12p2r1 n16 + dsp | 1 | 16 | 83 | 2 (over the 80% target) |
| Q2 / P3 / P4 / B1 | d16m1 n16 / d16p1 n32 / d24p1 n16 / d12 n16 | | | 92 / 97 / 94 / 97 | no |

Short screens trained 9 epochs on 1 GPU each (compare within this table only):

| model | AUC |
|---|---|
| d8p2r1 n16 | 0.9456 |
| d16p1r1 n16 | 0.9459 |
| d16p2r1m1 n16 | 0.9455 |
| d8p1r1 n32 | 0.9474 |
| d12p2r1m1 n16 | 0.9511 |

At 9 epochs the knobs showed no AUC cost, **but full training disagrees for p1**: d8p1r1 n32 reaches QAT AUC 0.9532
vs 0.9653 for d8p2r1 n16. Short single-GPU screens are too noisy and too undertrained to rank architectures; use
them only as a smoke test. mlp_ratio 1 (m1) has not been fully trained yet. It is the knob to try next, because it
keeps the phi block.

## 4. io_stream (k=4)

| run | shape | clock | II cycles / ns | LUT / DSP % SLR | clock est. |
|---|---|---|---|---|---|
| T1 | d32 n32 | 2.78 | 38 / 105.6 | 83 / 81 | 3.04 miss |
| T3 | d32 n24 | 3.125 | 32 / 99.8 | 83 / 86 | 3.29 miss |
| T4 | d32 n24, Latency strategy | 3.125 | 32 / 99.8 | 115 / 5 | 2.28 |
| T6 | T4 + `--dsp-mult` | 3.125 | 35 / 109.2 | 83 / 116 | 2.28 |
| T5 | d32 n16 | 4.17 | 20 / 83.4 | **82 / 81** | 3.04 |
| T7 | d32 n20 | 4.17 | 26 / 108.4 | 82 / 86 | 3.04 |
| T2b | d32 n12 (k=2) | 2.78 | 23 / 63.9 | 84 / 86 | 3.04 miss |

- io_stream II is n plus 4–11 cycles, and the overhead grows with the clock. So io_stream cannot reach k=2 (50 ns).
- k=4 works at 240 MHz with n 16.

## 5. Accuracy

| model | float AUC (full test, 404k jets) | QAT = HLS AUC (9984 jets) | acc (QAT) | 1/eB at eS 0.5 (QAT) |
|---|---|---|---|---|
| r7 (d32p2r1 n64) | — | 0.9744 | 0.9173 | 123.9 |
| d32p2r1 n64, 20 ep, 1 node (control) | 0.9708 | — | 0.9151 | 89.8 |
| d8p2r1 n16, 20 ep (screen) | 0.9672 | — | 0.9074 | 83.4 |
| **d8p2r1 n16, 50 ep + QAT (k=1)** | 0.9685 | **0.9653** | 0.9038 | 79.4 |
| **d32p2r1 n16, 50 ep + QAT (k=4)** | 0.9755 | **0.9712** | 0.9111 | 158.8 |
| d8p1r1 n32, 50 ep + QAT (k=1 extra) | 0.9631 | 0.9532 | 0.8930 | 37.4 |
| d12p1r1 n32, 50 ep + QAT (k=2 extra) | 0.9662 | 0.9566 | 0.8933 | 43.1 |

- The control is 0.006 below the fully trained 0.9770, so 20-epoch single-node screens undertrain by about that much.
- The QAT AUC comes from the exported QONNX graph via `convert.py`. `qat_deepsets_eval.py` is not used, because it
  rebuilds the model without the full-quant elements.
- 9984 jets give an AUC uncertainty of about ±0.002.
- The full-test-set float numbers and the 9984-jet QAT numbers are different samples.

## 6. Not covered

- The round-robin distributor and merger for k copies.
- Vivado synthesis and place-and-route (csynth timing only).
- Pipeline-style operator sharing (top-level II = C) was tried but did not finish scheduling (L7).

## 7. Next steps

1. **Vivado synthesis and place-and-route of the k=1 design** at 360 MHz (and the 200 MHz fallback), with a pblock
   on one SLR. csynth estimates 1.96 ns against 2.78 ns, but at 59% LUT the routed timing is the real question.
2. **Full training of the mlp_ratio-1 shape d12p2r1m1 n16** (k=1, 63% LUT). It keeps the phi block, which p1
   showed matters. It was launched 07:48 on nersc (interactive job 59160947, `ps_final.sbatch`, about 3 h). Its export
   will be `qonnx/fpga/qat_ps_d12p2r1m1_n16_e50_8bit_fullQuant_clean.onnx`; then run the k=1 `convert.py` line with
   that graph.
3. ~~Real-weight `--dsp-mult` on the k=1 build~~ done (H1f): 59 → 56% LUT with 328 DSPs (10%), same II, timing
   and AUC. `convert.py --dsp-mult` now exists.
4. **The round-robin distributor and merger** if k > 1 is ever used.

## 8. Reproduce

Training (nersc, branch `parallel-search`; on gpu_interactive, wrap the line in
`salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 bash scripts/ps_final.sbatch`):

```
SIZE=d8p2r1  N=16 sbatch scripts/ps_final.sbatch   # k=1   -> qat_ps_d8p2r1_n16_e50_8bit_fullQuant_clean.onnx
SIZE=d32p2r1 N=16 sbatch scripts/ps_final.sbatch   # k=4   -> qat_ps_d32p2r1_n16_e50_8bit_fullQuant_clean.onnx
SIZE=d12p1r1 N=32 sbatch scripts/ps_final.sbatch   # k=2
```

`ps_final.sbatch` runs three stages:
1. float KD: top_deepsets_distillnet_fpga recipe, `--act-layer relu --deepsets-fixed-n N`, 50 epochs on 1 node;
2. `qat_deepsets.py --full-quant --tanh-in-max 4 --res-bits 10 --relu-uint`, 15 epochs at 5e-5;
3. `qat_deepsets_export_qonnx.py`.

HLS (rdsrv, `synthesis/`):

```
python convert.py --onnx onnx_graphs/qat_ps_d8p2r1_n16_e50_8bit_fullQuant_clean.onnx --io-type io_parallel \
    --strategy Latency --pf 2 --mult-limit-fix --clone-fanout --clock 2.78 --synth          # k=1, 360 MHz
python convert.py --onnx onnx_graphs/qat_ps_d32p2r1_n16_e50_8bit_fullQuant_clean.onnx --io-type io_stream \
    --clock 4.17 --synth                                                                     # k=4, 240 MHz
```

The random-weight probes are `mini_parallel.py --distillnet ...` with the flags in the run log, and
`csynth_summary.py <project> [log]` prints one summary line per project.

**Note:** `convert.py` names projects from the graph suffix after `_8bit`, so every full-quant graph lands in
`hls_prj/deepsets_distillnet_8bit_fullQuant_<io>_...`. Synthesize one graph per option set at a time, or move the
project away first.
