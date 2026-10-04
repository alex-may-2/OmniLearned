# Single-SLR io_parallel design for the full-quant DeepSets top tagger (rdsrv409 + NERSC)

Run log, one row per run (history, including abandoned multi-copy builds): `synthesis/parallel_search_log.md`.

**Requirement.** L1T needs one jet every 25 ns from one copy of the tagger on one SLR:
- II ns = II cycles × clock period ≤ 25 ns;
- one SLR of `xcvu13p-flga2577-2-e`: 432k LUT (target < 80%), 864k FF, 3072 DSP, 1344 BRAM_18K;
- csynth estimated clock ≤ target clock.

II, the one-SLR fit and the clock are hard limits. AUC is soft (reference 0.9644 = r7 − 0.01 on the 9984-jet subset).

**C/RTL mismatch: found and fixed (2026-10-03).** C/RTL co-simulation of every io_parallel build (H1g, H2-d16s4) gave
RTL logits that differ from C-sim (max |d logit| ~3, argmax agreement 0.965 on 200 jets); C-sim matches QONNX bit for
bit and the r7 io_stream build was exact. Cause: the explicit `rewind` that hls4ml puts on the pointwise-conv
`PartitionLoop` (`nnet_conv1d_latency.h`, `PIPELINE II=RF rewind`). Cut graphs (`mini_parallel.py --cut`) showed the
first wrong layer is a conv output; without the rewind the RTL is exact but the first conv (the one reading the
top-level port) is not rewound and its II becomes n_partitions + depth (13 cycles). Pipeline styles, deeper FIFOs,
`ap_ctrl_chain`, start propagation, ping-pong channels and loop-local buffers all left the identical wrong values.
Fix, new flag `--auto-rewind` in both scripts (suffix `_arw`): drop the explicit rewind and give the input its own
capture process, so every conv reads an internal channel; Vitis then auto-rewinds every conv itself. II unchanged,
C-sim unchanged, cosim exact (mini BIS-b and BIS-d on 500 jets, H2-d16s4 and H1g-rsh on 200 real jets), and 4-5%
less LUT (H2-d16s4 318k to 300k). Use it on every io_parallel build. Run log rows CUT-*, VAR-*, D3-D8, FIX-*.

**Caveat.** All hardware numbers are Vitis HLS 2024.1 csynth estimates. There is no Vivado place and route. Clocks of
320 MHz and above are **csynth-only, not P&R-confirmed**.

## Summary

| design (trained, full-quant) | build | II cycles / ns | LUT / FF / DSP % SLR | clock est. | latency | AUC 9984 jets | AUC / 1/eB@0.5, all 404k jets |
|---|---|---|---|---|---|---|---|
| **H2-d16: d16p2r1m1, n 16** (2026-10-03) | PF 2, 360 MHz, `--reshape-channels --dsp-mult` | 8 / 22.2 | 74 / 25 / 18 | 1.96 ns | 34 cyc = 94 ns | 0.9699 | **0.9703 / 99.4** |
| H1g + `--reshape-channels` | PF 2, 360 MHz | 8 / 22.2 | 57 / 19 / 0 | 2.03 ns | 34 cyc = 94 ns | 0.9685 | 0.9697 / 91.5 |
| H1g: d12p2r1m1, n 16 (mlp_ratio 1) | PF 2, 360 MHz | 8 / 22.2 | 66 / 21 / 0 | 2.03 ns | 100 cyc = 278 ns | 0.9685 | 0.9697 / 91.5 |
| H1r: d8p2r1, n 16 | PF 2, 360 MHz | 8 / 22.2 | 59 / 18 / 0 | 1.96 ns | 275 ns | 0.9653 | 0.9666 / 78.1 |
| H1f: d8p2r1, n 16 + `--dsp-mult` | PF 2, 360 MHz | 8 / 22.2 | 56 / 18 / 10 | 1.96 ns | 275 ns | 0.9653 | 0.9666 / 78.1 (identical to H1r) |
| d8p2r1, n 16 | PF 4, 200 MHz | 4 / 20.0 | 67 / 15 / 0 | 3.65 ns | 300 ns | 0.9653 | |
| d8p2r1, n 16 | PF 2, 320 MHz | 8 / 25.0 | 57 / 17 / 0 | 2.28 ns | | 0.9653 | |

- **H2-d16 is the current design (2026-10-03).** The two LUT knobs of §3b free enough room for d16. It is one
  training seed: seed-to-seed spread at d12 is ~0.003 AUC and ~20% in 1/eB, so more seeds are in the run log.
- **H1g was the design until 2026-10-03.** It meets all three limits, and it is bit-exact (max |Δ logit| = 0) between QONNX and
  HLS C-sim. Its float model scores 0.9731 (all 404k jets).
- **H1r / H1f are the smaller fallback.** `--dsp-mult` moves 328 multiplies to DSPs for −5% LUT, with the same II,
  timing and AUC.
- **What made io_parallel work:**
  1. `--mult-limit-fix` (II = n/PF);
  2. `--clone-fanout` (timing);
  3. a fast clock: at 360 MHz, PF 2 (8 cycles) needs half the parallel multipliers of PF 4 at 200 MHz (4 cycles),
     so it uses 59% vs 67% LUT.
- **Model:** fewer particles (16) and narrower layers. mlp_ratio 1 (m1) frees LUT that buys width (d8 → d12).
  Dropping the residual phi block (p1) frees more LUT, but costs ~0.01 AUC after full training.
- Full-test-set comparison with r7 and the float models: `README.md`, "Full test set comparison".

## 1. Why the original model does not fit (2026-09-28/29)

The r7-size model (d32p2r1, 64 particles, ~410k MACs per jet) cannot run in io_parallel on one SLR.

| graph | io | result |
|---|---|---|
| first QAT (`_8bit`, float DynamicTanh) | io_stream, Resource, 5 ns | latency 0.6 µs, II 68 cyc = 340 ns, LUT 340k (78% SLR), DSP 1334, BRAM 628 |
| first QAT | io_parallel, Latency, PF 16 | C-sim fine; csynth killed at 1 h in unroll (110k instructions) |
| r7 (full-quant) | io_stream, Resource | II ~n cycles; bit-exact; kept as the reference project |

- **Tanh blocker (first QAT graph):** hls4ml's io_parallel `tanh` has a function-level `PIPELINE`, which unrolls the
  4096-entry table init, and 4096 parallel lookups into a 4096 x 18-bit table need ~7M LUT. The full-quant recipe
  fixes this: 256 x 8-bit exact tables (~28 LUT per lookup, no BRAM) and a power-of-2 alpha (a shift).
- **Size blocker:** in io_parallel, the elementwise layers (ReLU, alpha + Quant, tanh, residual Add) and the
  DATAFLOW FIFOs are fully parallel over all n particles, whatever the PF. PF only narrows the per-particle convs.

Random-weight mini scan (`mini_parallel.py`, 5 ns, every point bit-exact in C-sim):

| model | n | PF | II | LUT | note |
|---|---|---|---|---|---|
| ReLU-only, phi 32-16 / rho 16 | 16 | 8 | 40 ns | 366k (84% SLR) | |
| full-quant, phi 32-16 / rho 16 | 16 | 8 | 40 ns | 434k | +19% LUT, +7-8 cycles |
| distillnet d32 | 16 | 2 | 40 ns | 1.53M (354% SLR) | misses 5 ns; 33 GB |
| distillnet d16 | 32 | 2 | 80 ns | 996k (231% SLR) | 30 GB |
| PF = n (fully parallel) | 16 | 16 | — | — | Vitis OOM (> 58 GB) |

Cost rules (io_parallel):
- **elementwise + FIFO** ≈ 1.2k LUT × n × dim, independent of PF (relaxing II does not shrink it);
- **convs** ≈ 50-70 LUT per parallel multiply, DSP = 0 unless `--dsp-mult` (all 8-bit products go into LUTs);
- **II** = n / PF cycles once `--mult-limit-fix` is on;
- **Vitis memory** ≈ 30 GB at n × dim = 512, more than 55 GB at 1024. Run those synths alone.

So one SLR needs roughly n × dim ≤ ~200: a smaller student.

**PF gotcha:** set `ParallelizationFactor` on `MatMul_<i>`. hls4ml's `MatmulConstToDense` overwrites the
`Dense_MatMul_<i>` config with it. Check `firmware/parameters.h`: `n_partitions = n / PF`.

## 2. r7 mirror

`mini_parallel.py --distillnet` builds the r7 graph element for element:
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

## 3. What sets II, and the fixes (d8 n16, random weights)

| run | change | II cycles | clock est. | LUT % SLR | note |
|---|---|---|---|---|---|
| L0 | PF 2, 5 ns | 8 | 5.79 miss | 59 | II = n/PF |
| L1 | PF 4 | 8 | 5.79 miss | 64 | floor: 2 cycles per partition |
| L2 | PF 4 + `--mult-limit-fix` | 4 | 5.79 miss | 68 | II = n/PF again |
| L2c | L2 + fan-out clones | 4 | 3.65 | 69 | passes at 200 MHz |
| L6b | PF 2, 2.78 ns + fixes | 8 (22.2 ns) | 2.01 | 60 | passes at 360 MHz |
| L6c | PF 2, 3.125 ns + fixes | 8 (25.0 ns) | 2.28 | 58 | passes at 320 MHz |
| L4 | L2b (PF 4, mlf) + `--dsp-mult` | 4 | (pre-clone) | 60 (−11%) | 804 multiplies on DSP |
| L3 / L5 | PF = n, pipeline style | 5 / 2 | 3.5 / 3.6 | 102 / 98 | no FIFOs, but fully parallel |
| L7a/L7b | pipeline style, top II 5 / 8 | — | — | — | stalled in scheduling > 30 min; stopped |

- **II floor.** In `nnet_conv1d_latency.h`, the partition loop runs at II = RF under
  `ALLOCATION mul limit = n_chan·n_filt/RF`. That limit counts one particle, but each iteration does PF particles.
  `--mult-limit-fix` scales the limit by `n_pixels`. II then equals n_partitions × RF.
- **Clock miss.** Each residual skip array, and the pool output, has two readers in the DATAFLOW top (the DyT
  branch and the residual Add). Vitis then chains producer and consumers in one cycle: pool 3.0 ns plus alpha+Quant
  2.8 ns.
  - Giving each reader its own copy through a clone process removes the chain: 5.84 → 3.65 ns at 5 ns, and 2.01 ns
    at 2.78 ns.
  - `--clone-fanout` in `convert.py`; always on in `mini_parallel.py` io_parallel.
  - Registering the pool output (`LATENCY min=1`) did not help.

## 3b. LUT knobs (2026-10-03)

H1g's LUT (286k) split: multipliers 78k (27%), elementwise layers 107k (37%), DATAFLOW glue 100k (35%: FIFO control
64k for ~2,900 scalar depth-2 FIFOs, mux 25k, expressions 11k). Two opt-in flags in `mini_parallel.py` and
`convert.py` patch the generated project (csynth only; C-sim and outputs unchanged):

- `--reshape-channels`: `ARRAY_RESHAPE` instead of `ARRAY_PARTITION` on the inter-layer arrays, so each layer
  boundary is one channel, not one FIFO per element. Arrays a pointwise conv reads or writes stay partitioned: the
  conv indexes them per PF partition, and on a reshaped word that costs ~400k LUT per conv (K1).
- `--dsp-add`: `config_op add/sub -impl dsp`, with `--dsp-mult`. Moves adder trees and elementwise adds to DSPs.

| run | knobs (d12 n16 m1 PF2 360 MHz) | LUT | DSP | latency |
|---|---|---|---|---|
| Q1 | none (random weights) | 274k (63%) | 0 | 100 cyc |
| K0 | dsp-mult | 255k (59%) | 475 | 100 |
| K1b | reshape-channels | 232k (53%) | 0 | 34 |
| K3a | reshape + dsp-mult | 213k (49%) | 475 | 34 |
| K2 | dsp-mult + dsp-add | 220k (51%) | 2300 (74%) | 100 |
| K3b | all three | 178k (41%) | 2300 (74%) | 34 |
| H1g graph | reshape | 247k (57%) | 0 | 34 |
| H1g graph | all three | 194k (44%) | 2407 (78%) | 34 |

II, clock estimate and bit-exactness are unchanged in every run. `--dsp-add` runs out of DSPs past ~d13
(d16: 3694 DSP, d18: 4460), so the default set is `--reshape-channels --dsp-mult`.

Frontier with that set (random weights; trained graphs land 4-5 pts higher, so the gate is 77%):

| shape | II | LUT | | shape | II | LUT |
|---|---|---|---|---|---|---|
| d12 n16 | 8 | 49% | | d14 n20 (2.5 ns) | 10 | 71% |
| d12 n16 r2 / h2 | 8 | 52% / 51% | | d16 n16 | 8 | 70% |
| d12 n18 (2.77 ns) | 9 | 54% | | d16 n16 h2 / r2 | 8 | 75% / 76% |
| d12 n20 (2.5 ns) | 10 | 58% | | d16 n18 (2.77 ns) | 9 | 77% |
| d14 n16 | 8 | 59% | | d18 n16 | 8 | 81% (no) |
| d14 n16 h2 / r2 | 8 | 62% / 64% | | d16 n20 (2.5 ns) | 10 | 84% (no) |
| d12 n24 PF3 | 8 | 71% | | d20 n16 | 8 | 92% (no) |

Trained graphs (all bit-exact, 404k-jet C-sim = NERSC `qat_deepsets_eval.py`): d16 n16 74% (AUC 0.9703),
d12 n24 76% (0.9678), d14 n16 59% (0.9591, bad float seed), d12 n20 at 400 MHz 58% (0.9640), d12 h2 55% (0.9658).

## 4. Model knobs (random weights, d8 n16, PF 2, 360 MHz)

| run | knob | LUT | vs L6b (263k) | keep? |
|---|---|---|---|---|
| M1 | p1: no residual phi block | 115k | −56% | costs ~0.01 AUC after full training (§5) |
| M3 | mlp_ratio 1 | 173k | −34% | **yes**: size suffix `d<dim>p<p>r<r>m1` (nersc `utils.py`) |
| M2 | r0: no rho block | 244k | −7% | no |
| M4 | no embed DyT | 251k | −5% | no |

Shapes that fit one SLR at II 8 cycles (random weights, 360 MHz):

| run | shape | PF | LUT % SLR |
|---|---|---|---|
| P2 | d8p1r1 n32 | 4 | 48 |
| P1 | d16p1r1 n16 | 2 | 62 |
| Q1 | d12p2r1m1 n16 | 2 | 63 |
| R2 | d12p1r1 n32 | 4 | 77 |
| Q2 / P3 / P4 / B1 | d16p2r1m1 n16 / d16p1r1 n32 / d24p1r1 n16 / d12p2r1 n16 | | 92 / 97 / 94 / 97 (too big) |

## 5. Accuracy

| model | float AUC (all 404k jets) | QAT = HLS AUC (9984 jets) | 1/eB at eS 0.5 (9984 jets) |
|---|---|---|---|
| r7 (d32p2r1 n64) | 0.9786 (twamorka float) | 0.9744 | 123.9 |
| **d12p2r1m1 n16, 50 ep + QAT (H1g)** | 0.9731 | **0.9685** | 94.1 |
| d8p2r1 n16, 50 ep + QAT (H1r) | 0.9685 | 0.9653 | 79.4 |
| d8p1r1 n32, 50 ep + QAT | 0.9631 | 0.9532 | 37.4 |

- Training: float KD 50 epochs, then full-quant QAT 15 epochs (§7). QAT AUC comes from the exported QONNX graph via
  `convert.py`. Since 2026-10-01 `qat_deepsets_eval.py` rebuilds full-quant checkpoints from their `arch_config` and
  matches it: H1g gives acc 0.9106 / AUC 0.9697 / 1/eB@0.5 91.5 on all 404k jets, the same as HLS C-sim.
- 9-epoch single-GPU screens ranked d8p1r1 n32 first; full training put it last. Use screens only as smoke tests.
- 9984 jets give AUC ±0.002 and 1/eB@0.5 ±20%. Rank with `csim_forward.py` on all 404k jets (README).

## 6. Next steps

0. (2026-10-03) The RTL/C mismatch is fixed by `--auto-rewind` (above). Quantization scan: floor rounding (`--round
   floor`) frees ~10% LUT at no AUC cost; d18 and d16 r2 with floor are in training (run log QZ*, QP*, QF*).

1. **Vivado synthesis and place-and-route of H1g** at 360 MHz with a pblock on one SLR. csynth estimates 2.03 ns
   against 2.78 ns, but routed timing at 66% LUT is the real question. The 200 MHz PF 4 build is the fallback.
2. **Use the LUT headroom (66% → 80%):** d14p2r1m1 n16, or d12p2r1m1 at n 24 (PF 3). d16p2r1m1 n16 was 92% at
   random weights. `--dsp-mult` on H1g would free ~5% LUT more.
3. **Input range:** r7's input Quant saturates at 3.97 (log pT / log E are ~5); H1g learned ±8. Check whether a
   fixed wider input range helps the small models too.

## 7. Reproduce

Training (nersc, branch `synthesis`, on gpu_interactive):

```
SIZE=d12p2r1m1 N=16 FLOAT_TAG=ps_d12p2r1m1_n16_e50 setsid nohup salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
    --gpus-per-node 4 -A m3246 bash scripts/fullquant_chain.sbatch > <log> 2>&1 < /dev/null &
# -> qonnx/fpga/qat_ps_d12p2r1m1_n16_e50_8bit_fullQuant_clean.onnx
```

`fullquant_chain.sbatch` (formerly `ps_final.sbatch`) runs three stages:
1. float KD: top_deepsets_distillnet_fpga recipe, `--act-layer relu --deepsets-fixed-n N`, 50 epochs on 1 node;
2. `qat_deepsets.py` (always full-quant; its defaults are `--tanh-in-max 4 --res-bits 10 --relu-uint`), 15 epochs at 5e-5;
3. `qat_deepsets_export_qonnx.py --tag <qat_tag>` (shape and quant settings come from the checkpoint's `arch_config`).

HLS (rdsrv, `synthesis/`):

```
python convert.py --onnx onnx_graphs/qat_ps_d12p2r1m1_n16_e50_8bit_fullQuant_clean.onnx --name ps_d12p2r1m1_n16 \
    --io-type io_parallel --strategy Latency --pf 2 --mult-limit-fix --clone-fanout --clock 2.78 --synth
python csim_forward.py hls_prj/<project>          # all 404k test jets
python csynth_summary.py hls_prj/<project> [log]  # II, clock, % SLR
```

Random-weight probes: `mini_parallel.py --distillnet ...` with the flags in the run log.

Kept projects (`synthesis/hls_prj/`, not in git):
- `deepsets_ps_d12p2r1m1_n16_io_parallel_pf2_clk2.78_mlf_clone` (H1g, current design);
- `deepsets_ps_d8p2r1_n16_io_parallel_pf2_clk2.78_mlf_clone` (H1r);
- `deepsets_distillnet_8bit_fullQuant_io_parallel_latency_rf1_pf2_clk2.78_mlf_clone_dsp` (H1f, d8p2r1 graph);
- `deepsets_distillnet_8bit_fullQuant_r7_io_stream_resource_rf1` (r7 reference);
- `deepsets_distillnet_8bit` (first QAT graph, io_stream baseline).
