# io_parallel feasibility for the DeepSets top tagger (2026-09-28/29, rdsrv409)

Question: can the DeepSets student run fully parallel in hls4ml for an L1 trigger, and how many particles fit?

All numbers are Vitis HLS 2024.1 csynth estimates:
- part `xcvu13p-flga2577-2-e`, 5 ns clock, `Latency` strategy, RF 1;
- no Vivado place and route;
- device 1.73M LUT, one SLR 432k LUT.

Mini-model points use random weights from `mini_parallel.py`, and every one is bit-exact in C-sim vs qonnx.

## Summary

- **The real 64-particle graph does not synthesize in io_parallel.** Vitis was still unrolling at the 1 h cap. With
  io_stream it fits easily: latency 0.6 µs, II 68 cycles (0.34 µs), 19% LUT.
- **The full-quant model removes the tanh blocker, but not the size problem.**
  - Tanh becomes a 256 x 8-bit exact table (~28 LUT per lookup, no BRAM), and the po2 alpha becomes a shift.
  - On the same topology it still costs 13-19% more LUT than a ReLU-only mini, because it has more layers.
- **The particle count is limited by the elementwise layers and FIFOs, not by the multiplies.**
  - PF (ParallelizationFactor) only narrows the per-particle convs. ReLU, alpha+Quant, tanh, residual Add and
    the dataflow FIFOs are fully parallel over all n particles.
  - They cost ~1.2k LUT per particle per base channel.
- **Best real-topology point found:** dim 16, n 32, PF 2.
  - 57% LUT, II 80 ns, latency 0.53-0.55 µs.
  - The trained size (dim 32) only fits at n 16: 88% LUT, and it misses 5 ns. n 64 at dim 32 would need ~2.5M LUT.
- **Fully parallel (PF = n, II ≤ 25 ns) is out of reach on this machine.** Vitis needed more than 58 GB even for a
  10.5k-MAC mini.
- **For L1T, io_stream is the practical path.** The II needed is the TMUX period (150 ns at TMUX 6), not 25 ns.
  io_stream at 250 MHz with 2 copies fed round-robin should give ~135 ns effective II at ~0.5 µs latency
  (see below).

## 1. Real graph

| graph | io | result |
|---|---|---|
| first QAT (`_8bit`, float DynamicTanh) | io_stream, Resource | latency 120 cyc = 0.6 µs, II 68 cyc, LUT 340k (19%), DSP 1334, BRAM 628, clock est. 3.65 ns |
| first QAT | io_parallel, Latency, PF 16 | C-sim fine; csynth killed at 1 h in unroll (110k instructions) |
| full-quant (r4) | io_parallel PF 16 | C-sim bit-exact (run on NERSC); not synthesized |

Why the first graph fails in io_parallel:
- `tanh` has a function-level `PIPELINE` pragma, which unrolls the 4096-entry table init.
- 4096 parallel lookups into a 4096 x 18-bit table do not fit (~7M LUT).
- The model is ~410k MACs per jet.

**PF gotcha:** set `ParallelizationFactor` on `MatMul_<i>`. hls4ml's `MatmulConstToDense` overwrites the
`Dense_MatMul_<i>` config with it. Check `firmware/parameters.h`: `n_partitions = n / PF`.

## 2. Mini scan

`mini_parallel.py` modes:
- **default:** ReLU only, no biases, no tanh. phi = per-particle widths, rho = post-pool widths.
- **`--full-quant`:** same widths, with the full-quant elements: Int16 biases, and each hidden
  `ReLU -> Mul(po2 alpha) -> Quant(8b, 1/32) -> Tanh -> Quant(8b, 1/128)`.
- **`--distillnet`:** the real student topology with full-quant elements:
  - embed `4 -> 2d -> d` with DyT;
  - residual phi block(s);
  - mean pool;
  - residual rho block(s);
  - `-> 2` logits.

### ReLU-only vs full-quant, same topology (n 16, PF 8)

| phi / rho | variant | latency | II | LUT | FF | csynth time, peak RAM |
|---|---|---|---|---|---|---|
| 32-16 / 16 | ReLU only | 145-155 ns | 40 ns | 366k (21%, 84% SLR) | 124k | 4 min, 4 GB |
| 32-16 / 16 | full-quant | 185-195 ns | 40 ns | 434k (25%) | 196k | 6 min, 7 GB |
| 64-32 / 32 | ReLU only | 145-155 ns | 30 ns | 1.04M (60%) | 296k | 26 min, 14 GB |
| 64-32 / 32 | full-quant | 180-190 ns | 30 ns | 1.17M (67%) | 444k | 36 min, 23 GB |

Other ReLU-only points:
- n 16, phi 64-32, PF 4: 165-175 ns, II 40 ns, 743k LUT.
- n 32, phi 64-32, PF 8: 205-215 ns, II 60 ns, 1.39M LUT, 39 GB.

Fully parallel (PF 16) runs failed:
- phi 64-32: clang OOM after unroll (2.4M instructions).
- phi 32-16: 58 GB, OOM-killed.

LUT by layer kind, phi 64-32:

| layer | ReLU only | full-quant |
|---|---|---|
| per-particle convs | 691k | 673k |
| ReLU (ReLU only: fused with its 8-bit Quant) | 172k | 75k |
| alpha shift + tanh-input Quant | — | 108k |
| tanh (256-entry tables) | — | 43k |
| dataflow FIFOs | 71k | 139k |
| rho dense | 44k | 44k |

Full-quant does not make the multiplies more expensive. The extra cost comes from the steps:
- one elementwise layer becomes three (ReLU, alpha + Quant, tanh);
- each extra layer adds a dataflow stage with its own FIFO, which doubles the FIFO LUTs and adds 7-8 cycles;
- a Quant costs its rounding and saturation, a few tens of LUT per element, for every element in parallel.

### Real topology (`--distillnet`)

| dim | n | PF | MACs/jet | latency | II | LUT | clock est. | csynth time, peak RAM |
|---|---|---|---|---|---|---|---|---|
| 32 | 16 | 2 | 107k | 0.44-0.46 µs | 40 ns | 1.53M (88%) | **5.75 ns, misses 5 ns** | 30 min, 33 GB |
| 32 | 32 | 1 | 209k | — | — | — | — | killed at 55 GB |
| 16 | 32 | 2 | 54k | 0.53-0.55 µs | 80 ns | 996k (57%) | 4.20 ns | 24 min, 30 GB |

Both finished points have n x dim = 512, and both spend about 600k LUT outside the convs and the rho dense. That gives the scaling rule:
- **elementwise + FIFO** ≈ 1.2k LUT x n x dim, independent of PF;
- **convs** ≈ 50-70 LUT per parallel multiply, DSP = 0 (all 8-bit products go into LUTs);
- **II** ≈ n / PF cycles;
- **Vitis memory** ≈ 30 GB at n x dim = 512, and more than 55 GB at 1024. Run these synths alone.

**Clock:** at dim 32, the exact-mean pool (`fixed<31,12>`) and the next alpha + Quant are chained in one cycle
(2.3 + 3.5 ns). Narrow the pool result type, or register the pool output, to fix it.

## 3. io_stream for L1T (estimates, not synthesized)

io_stream reads one particle per clock, so its II is about n cycles.

| option | II | latency | LUT |
|---|---|---|---|
| now: n 64, 200 MHz | 340 ns | 600 ns | 19% |
| n 64, 250 MHz (clock est. 3.65 ns) | ~270 ns | ~480 ns | 19% |
| n 64, 250 MHz, 2 copies fed round-robin | ~135 ns effective | ~480 ns | ~40% |
| n 32, 250 MHz | ~145 ns | ~350 ns | ~10-15% |

Replicating the IP is standard in L1T, and io_stream is small enough to allow it. Confirm the TMUX period, the
latency budget and the clock with the trigger group.

## 4. Recommendations

1. **Synthesize the full-quant graph with io_stream.**
   - Run `convert.py --onnx <fullQuant_clean.onnx> --synth`. Compare with the first run: 340k LUT, 628 BRAM,
     1334 DSP.
   - Then add a `--clock` option to `convert.py` and try 4 ns.
2. **For io_parallel, train a smaller student,** with n x dim ≤ ~512 for the whole device, or ≤ ~200 for one SLR.
3. **Cut the per-element cost:**
   - fuse `ReLU -> shift -> Quant -> tanh` into one lookup;
   - drop DATAFLOW at the top (FIFOs are 12-23% of the LUT);
   - pack the conv multiplies into DSPs.
4. **Run Vivado synthesis and implementation** on the chosen point for real timing (multi-SLR).

Kept projects (`hls_prj/`, not in git): `mini_n16_phi32-16_rho16_pf8`, `mini_fq_n16_phi32-16_rho16_pf8`,
`mini_ds_d16r2_phi1_rho1_n32_pf2`. Logs are in `logs/mini_*.txt`.
