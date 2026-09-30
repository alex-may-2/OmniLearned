# io_parallel with the full-quant model: mini scan report (2026-09-29, rdsrv409)

This follows up `io_parallel_report.md` (the first scan). The full-quant QAT model on the `synthesis` branch is
`qat_deepsets.py --full-quant --tanh-in-max 4`. Its main changes:
- 8-bit tanh input on a 1/32 grid, giving an exact 256-entry table;
- power-of-2 alpha (a shift);
- gamma folded into the next weight;
- Int16 biases.

Question: does this lower the io_parallel resources and latency enough to allow more particles?

All numbers are Vitis HLS 2024.1 csynth estimates on `xcvu13p-flga2577-2-e`:
- 5 ns clock, `Latency` strategy, RF 1;
- random weights;
- no Vivado place and route.

Every point below is **bit-exact** in C-sim vs qonnx (max |Δ logit| = 0, 100% argmax agreement).

## TL;DR

- **No, it does not allow more particles in io_parallel.** On the same topology, the full-quant elements cost
  **13-19% more LUT** and **7-8 more cycles of latency** than the first scan's ReLU-only mini (same II). The first
  mini had no tanh and no biases, so it was already cheaper than any real model.
- **Where full-quant does help:**
  - The tanh no longer blocks io_parallel: a 256 x 8-bit table is ~28 LUT per lookup, no BRAM, compared with the
    4096 x 18-bit float-tanh table that made the real graph infeasible.
  - The po2 alpha costs no multiplier.
  - The input port is 8 bits (about 7k fewer LUT).
  - Everything is bit-exact.
- **The particle count is limited by the elementwise layers and FIFOs, not by the MACs.**
  - PF only narrows the per-particle convs. ReLU, alpha+Quant, tanh, residual Add and the dataflow FIFOs are fully
    parallel over all n particles.
  - They cost **~1.2k LUT per (particle x base channel)**: about 600k LUT at n x dim = 512.
  - Vitis needs ~30 GB for n x dim = 512 and > 55 GB for 1024 (the machine limit).
- **Largest real-topology point that synthesizes:** distillnet **dim 16, n = 32, PF 2**.
  - 57% LUT, latency 0.53-0.55 µs, II 16 cycles = 80 ns, clock estimate 4.2 ns.
  - The trained model (dim 32) fits only at n = 16: 88% LUT, and it misses the 5 ns clock.
  - n = 32 at dim 32 ran out of memory in Vitis, even at PF 1.
- The real graph (n = 64, dim 32) would need ~2.5M LUT for the elementwise layers alone, over the 1.73M on the
  device. **For 64 particles, stay with io_stream.** The full-quant graph should now be cheaper there than the
  first run (340k LUT, 628 BRAM, 1334 DSP), but that csynth has not been run yet.

## What changed in `mini_parallel.py`

The default mode is unchanged (first-scan points reproduce). Two new modes:

- `--full-quant`: same `--phi/--rho` widths as the first scan.
  - Every Linear gets an Int16 bias: a 16-bit `Quant` with scale s_in x s_w, then an `Add`.
  - Every hidden `Relu -> Quant` becomes `Relu -> Mul(alpha=2) -> Quant(8b, 1/32) -> Tanh -> Quant(8b, 1/128)`.
- `--distillnet`: the real student topology with full-quant elements.
  - embed `4 -> 2d ReLU DyT -> d`, then `--phi-blocks` residual blocks `h + fc2(Quant(ReLU(fc1(DyT(h)))))`;
  - mean pool;
  - `--rho-blocks` residual blocks, then `Quant -> Linear -> 2`;
  - alpha 2 (embed) and 8 (residual norms). `--dim 32 --ratio 2` is the trained model.
- All six hls4ml patches and the full-quant type settings are copied from `convert.py`:
  - tanh `TableSize = 8/scale` and `table_t` = the next Quant's type;
  - input type from the input Quant;
  - exact /n pool and unfused-ReLU types from a first no-compile conversion.

  `convert.py` itself is untouched.
- Prints the `n_filt` of every ApplyAlpha, and MACs/jet.

## A. Same topology as the first scan: old (ReLU only) vs full-quant

| config (n16, Latency) | variant | latency | II | LUT | FF | DSP / BRAM | clock est. | csynth time, peak RAM |
|---|---|---|---|---|---|---|---|---|
| phi 32-16, rho 16, PF 8 | old | 29-31 cyc = 145-155 ns | 8 cyc = 40 ns | 366k (21%) | 124k | 0 / 0 | 3.65 ns | 4 min, 4 GB |
| | **full-quant** | 37-39 cyc = 185-195 ns | 8 cyc = 40 ns | 434k (25%) | 196k | 0 / 0 | 3.65 ns | 6 min, 7 GB |
| phi 64-32, rho 32, PF 8 | old | 29-31 cyc = 145-155 ns | 6 cyc = 30 ns | 1.04M (60%) | 296k | 0 / 0 | 3.65 ns | 26 min, 14 GB |
| | **full-quant** | 36-38 cyc = 180-190 ns | 6 cyc = 30 ns | 1.17M (67%) | 444k | 0 / 0 | 3.65 ns | 36 min, 23 GB |
| phi 32-16, rho 16, PF 16 (fully parallel) | old | — | — | — | — | — | — | stopped by the low-memory guard (first scan); not rerun |
| | **full-quant** | — | — | — | — | — | — | **OOM-killed**: Vitis reached 58 GB of 61 GB during scheduling (14 min) |

LUT by layer kind (from the csynth instance table):

| layer kind | phi 32-16 old | phi 32-16 fq | phi 64-32 old | phi 64-32 fq |
|---|---|---|---|---|
| per-particle convs (`pointwise_conv_1d`) | 198k | 194k | 691k | 673k |
| ReLU (old: fused with its 8-bit Quant) | 85k | 37k | 172k | 75k |
| alpha shift + tanh-input Quant (`normalize`) | — | 53k | — | 108k |
| tanh, 256-entry tables | — | 22k | — | 43k |
| dataflow FIFOs | 36k | 69k | 71k | 139k |
| rho dense | 13k | 13k | 44k | 44k |
| input conversion (`linear`) | 8k | 1k | 9k | 2k |
| pool | 4k | 4k | 8k | 8k |

What the table shows:
- **Convs are unchanged.** They are the same 8x8 multiplies in both variants; the biases are nearly free.
- **The elementwise chain costs about 30% more:**
  - old: ReLU + Quant, 1 layer;
  - fq: ReLU, then alpha + Quant, then tanh, 3 layers.

  The LUT cost of `normalize` is the Quant rounding and saturation to 8 bits. The multiply by alpha = 2 is folded
  to a shift.
- **FIFOs double** because each extra layer is another dataflow stage with its own ping-pong buffer. This also adds
  the 7-8 cycles of latency.
- **Tanh:** 1568 lookups for 43k LUT (~28 LUT each), 0 BRAM.

## B. Real topology (`--distillnet`, full-quant), n scan

| model | n | PF (parallel particles) | MACs/jet | latency | II | LUT | FF | clock est. | csynth time, peak RAM |
|---|---|---|---|---|---|---|---|---|---|
| dim 32 (trained size) | 16 | 2 | 107k | 76-80 cyc = 0.44-0.46 µs | 8 cyc = 40 ns | 1.53M (88%) | 658k (19%) | **5.75 ns (misses 5 ns)** | 30 min, 33 GB |
| dim 32 | 32 | 1 | 209k | — | — | — | — | — | **killed at 55 GB** (after scheduling, 57 min) |
| dim 16 | 32 | 2 | 54k | 106-110 cyc = 0.53-0.55 µs | 16 cyc = 80 ns | 996k (57%) | 588k (17%) | 4.20 ns | 24 min, 30 GB |
| dim 32 | 64 | — | 419k | not run: ~2.5M LUT estimated for the elementwise layers alone | | | | | |

For reference, the first io_stream run of the real (not full-quant) graph at n 64, dim 32: latency 0.6 µs, II 68 cycles
(0.34 µs), 340k LUT (19%). io_parallel latency is about the same (0.44-0.55 µs vs 0.6 µs). It gains II, 4-8x, and
the cost is 3-4.5x the LUT for 2-4x fewer particles.

LUT by layer kind:

| layer kind | dim 32, n 16, PF 2 | dim 16, n 32, PF 2 |
|---|---|---|
| per-particle convs | 620k | 244k |
| dataflow FIFOs | 236k | 231k |
| rho dense (post pool) | 163k | 44k |
| ReLU | 159k | 136k |
| alpha + Quant (`normalize`) | 144k | 141k |
| tanh | 43k | 43k |
| residual Add | 26k | 25k |
| pool | 15k | 15k |

The two points have the same n x dim = 512, and **the same ~600k LUT outside the convs and the rho dense**. That gives
the scaling rule:
- elementwise + FIFO ≈ 1.2k LUT x n x dim, fully parallel regardless of PF;
- convs ≈ 50-70 LUT x PF x MACs/particle (all multiplies in LUTs, 0 DSP);
- Vitis memory ≈ 30 GB at n x dim = 512, > 55 GB at 1024.

II = n / PF cycles (one cycle per conv partition): 8 at n16 PF2, 16 at n32 PF2.

**Clock:** at dim 32 the critical path is the exact-mean pool (`fixed<31,12>` result) chained in the same cycle
into the rho block's alpha + Quant (2.3 + 3.5 ns). Both are zero-latency calls, so HLS puts no register between
them. At dim 16 the pool part is 0.7 ns and timing is met. To fix it, narrow the pool result type (the rho Quant
only needs 8 bits), or register the pool output.

## Status of the three open items from `io_parallel_report.md` section 3

1. **Smaller model:** still required, and now quantified. Keep n x dim ≲ 512, e.g. n16 x dim32 or n32 x dim16.
   n x dim = 1024 does not synthesize on this 61 GB machine, and n64 x dim32 = 2048 would not fit the device.
2. **Tanh table storage:** resolved. The 256 x 8-bit tables are built in LUT logic (0 BRAM), ~28 LUT per lookup,
   only 3-4% of the design.
3. **Per-particle broadcast constants:** harmless with full-quant. The alpha ApplyAlpha is still broadcast
   (`n_filt = -1`, n x channels constants). But the constant is a power of 2, so HLS folds it to a shift, and the
   remaining cost is the Quant rounding, which is needed per element anyway.

## Recommendations

1. **For the 64-particle model, use io_stream.** Copy the r4 full-quant graph from Perlmutter to `onnx_graphs/`,
   then run `convert.py --onnx <fullQuant_clean.onnx> --synth`, first with io_stream. Compare with the first run
   (340k LUT, 628 BRAM, 1334 DSP, II 68): this is where the full-quant savings should show (small tanh tables,
   no gamma multipliers, 8-bit input).
2. **For an L1-style io_parallel design,** train a smaller student: n x dim ≤ 512, e.g. n16 with dim 32, or n32
   with dim 16. dim16/n32/PF2 (57% LUT, II 80 ns, 0.54 µs) is the best fitting point found.
3. **Cut the per-element cost.** It is now the limit, not the MACs:
   - Merge `ReLU -> alpha shift -> Quant -> tanh` into one LUT per element. This needs an hls4ml layer fusion;
     today it is 3 layers and 3 FIFO stages.
   - Drop DATAFLOW at the top (FIFOs are 12-23% of the LUT).
   - Pack the conv multiplies into DSPs (0 of 12,288 used).
4. Narrow the pool result type in the full-quant path, so the pool and the following alpha+Quant stay under 5 ns
   at dim 32.
5. Fully parallel (PF = n) is out of reach on rdsrv409, even for the 10.5k-MAC phi 32-16 mini: Vitis needs
   more than 58 GB. It would need a bigger-memory machine.

## Can io_stream meet L1T requirements? (estimates, not synthesized)

The II needed is the time-multiplex (TMUX) period, not 25 ns. With TMUX 6 that is 150 ns per board. The latency
budget per algorithm is typically a few hundred ns to ~1 µs; confirm both numbers with the trigger group. io_stream
feeds one particle per clock, so its II is about n cycles.

| option | II | latency | cost |
|---|---|---|---|
| now: n64 at 200 MHz | 340 ns | 600 ns | 19% LUT |
| n64 at 250 MHz (HLS clock estimate 3.65 ns) | ~270 ns | ~480 ns | same |
| n64 at 250 MHz, 2 copies fed round-robin | ~135 ns effective | ~480 ns | ~40% LUT |
| n32 at 250 MHz | ~145 ns | ~350 ns | ~10-15% LUT |

Replicating the IP is standard in L1T and affordable with io_stream. io_parallel is only needed for a true 25 ns II
without TMUX. Next step: synthesize the r4 full-quant graph with io_stream at a 4 ns clock. That needs a `--clock`
option in `convert.py`.

## Files

- `mini_parallel.py`: new `--full-quant` and `--distillnet` modes (see above).
- Logs, in `logs/`, each ending with `time -v` peak RSS and the exit code:
  - `mini_fq_n16_phi32-16_pf8.txt`, `mini_fq_n16_phi32-16_pf16.txt`, `mini_fq_n16_phi64-32_pf8.txt`;
  - `mini_ds_n16_pf2.txt`, `mini_ds_n32_pf1.txt`, `mini_ds_d16_n32_pf2.txt`.
- Projects: `hls_prj/mini_fq_*` and `hls_prj/mini_ds_*` (not in git).
- Runs used a 2 h cap (none reached it). Two synths ran concurrently only for the small points; the distillnet
  points ran alone.
  - The first D16 attempt was killed at 30 GB by a low-memory watchdog while running next to phi 64-32 PF 8.
    The run was in its last RTL step; rerunning it alone completed it.
