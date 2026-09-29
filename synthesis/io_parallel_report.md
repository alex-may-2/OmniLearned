# io_parallel / L1-trigger feasibility report (2026-09-28, rdsrv409)

Question: can the DeepSets top tagger run fully parallel in hls4ml, fast enough for an LHC L1 trigger?
All numbers are Vitis HLS 2024.1 csynth estimates on `xcvu13p-flga2577-2-e`, 5 ns clock (200 MHz); no Vivado
place and route has been run.

## TL;DR

- The current 64-particle graph (`qat_top_deepsets_distillnet_fpga_a05_T4_8bit_clean.onnx`) only synthesizes with
  `io_stream`: **latency 0.6 µs, II 68 cycles (0.34 µs) per jet**, 19% LUT. With `io_parallel` it does not synthesize:
  Vitis was still unrolling after the 1 h cap.
- "25 ns" at the LHC is the **initiation interval** (one new event per bunch crossing, 5 cycles at 200 MHz), not the
  latency. L1 latency budgets are µs. Per-algorithm latency of ~100-200 ns is realistic.
- A random-weight mini DeepSets (ReLU, no tanh) scan with 16-32 particles gets **145-215 ns latency** and a
  best **II of 6 cycles = 30 ns** (16 particles, phi 4-64-32, PF 8, 60% LUT). With the smaller phi 4-32-16 it
  uses only 21% LUT (II 40 ns). 25 ns needs fully parallel conv layers (PF = n). The small-phi PF 16 run was
  stopped for low memory before it finished; rerun it alone.
- To reach it with the real model: fewer particles (16-32), no DynamicTanh (or the 8-bit exact tanh table from
  the full-quant QAT plan), a narrower phi, and multiplies moved into DSPs.

## 1. Baseline: current graph, io_stream (first run, 2026-09-26)

| | value |
|---|---|
| latency | 120 cycles = 0.600 µs (one jet, 64 particles streamed one per clock) |
| II | 68 cycles = 0.340 µs (about 2.9 M jets/s) |
| clock estimate | 3.65 ns (target 5 ns) |
| resources | BRAM 628 (11%), DSP 1334 (10%), FF 125k (3%), LUT 340k (19%) |

The HLS model processes **one jet per call**. The batch dimension of 64 in the ONNX export exists only in C-sim.
The latency is dominated by streaming 64 particles one per clock.

## 2. io_parallel on the current graph

Command: `convert.py --io-type io_parallel --strategy Latency --pf 16` (new flags, see below).

- **C-sim:** identical to io_stream (acc 0.9169, AUC 0.9751, argmax agreement 97.5% vs qonnx). The existing
  monkeypatches also work for io_parallel.
- **csynth:** killed at the 1 h cap, still in Vitis unroll/inline (110k instructions after compile/link).
- **Why it cannot work as-is.** In io_parallel every elementwise layer processes all 64 x 64 = 4096 values at once.
  PF only folds the 4 per-particle dense layers.
  - hls4ml's io_parallel `tanh` has `#pragma HLS PIPELINE` at function level. This fully unrolls the 4096-entry
    float table init (x3 tanh layers), and 4096 parallel lookups into a 4096 x 18-bit table need ~7M LUT (or
    ~2000 BRAM copies per layer). The device has 1.7M LUT and 5376 BRAM.
  - The DynamicTanh norm scale/bias (ApplyAlpha) are broadcast to particles x channels (4096 constants,
    `n_filt = -1`) instead of per channel.
  - The model itself is ~410k MACs per jet (64 particles x 4-64-32-64-32 phi).

**PF gotcha (fixed in convert.py):** `ParallelizationFactor` must be set on `MatMul_<i>`. hls4ml's
`MatmulConstToDense` copies the `MatMul_<i>` config over any `Dense_MatMul_<i>` entry, so PF set on the Dense name
is silently ignored. Check it in `firmware/parameters.h`: `n_partitions = 64 / PF`.

## 3. Does the full-quant QAT plan (running on NERSC) fix io_parallel?

Partly. The 8-bit tanh input gives a 256 x 8-bit exact table (~32 LUT per lookup instead of ~1150, ~200k LUT total
for the 6144 parallel lookups). It also folds gamma into the next layer and turns alpha into a power of 2 (a shift).
That removes the tanh blocker. Still needed for io_parallel:

1. **Smaller model.** 410k MACs/jet does not fit fully parallel; PF=16 needs ~100k parallel multipliers (~2M LUT).
   II is about (64 / PF) x 2-3 cycles. Needs fewer particles and/or a narrower phi (see section 4).
2. **Tanh table storage.** Check that HLS builds the 256-entry table as LUT logic and does not replicate it in
   BRAM. If needed, patch the template with `BIND_STORAGE ... impl=lutram`.
3. **Per-particle broadcast constants.** If any ApplyAlpha survives after a per-particle layer, collapse its
   scale/bias to per-channel (`n_filt` = channels) in `convert.py`.

## 4. Mini-model scan (`mini_parallel.py`, random weights)

A QONNX graph built in the script, shaped like the student but without DynamicTanh and biases. 8-bit power-of-2
`Quant` on the input, weights and every ReLU output; mean pool; io_parallel; `Latency` strategy; RF 1. Default phi
4-64-32 per particle, rho 32-2. C-sim is **bit-exact** vs qonnx (max |Δ| = 0) for every point. The graph and
project are saved in `hls_prj/mini_*`.

| particles | phi | PF | parallel particles | MACs/jet | latency | II | LUT (device / 1 SLR) | FF | DSP | csynth time, peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 64-32 | 16 | 16 | 38k | — | — | — | — | — | failed: clang OOM-killed after 40 min (2.4M instr. after unroll; 3 synths were running at once) |
| 16 | 64-32 | 4 | 4 | 38k | 33-35 cyc = 165-175 ns | 8 cyc = 40 ns | 743k (42% / 171%) | 246k (7%) | 0 | 14 min, 12 GB |
| 32 | 64-32 | 8 | 8 | 75k | 41-43 cyc = 205-215 ns | 12 cyc = 60 ns | 1.39M (80% / 321%) | 480k (13%) | 0 | 54 min, 39 GB |
| 16 | 64-32 | 8 | 8 | 38k | 29-31 cyc = 145-155 ns | **6 cyc = 30 ns** | 1.04M (60% / 240%) | 296k (8%) | 0 | 26 min, 14 GB |
| 16 | 32-16 (rho 16) | 8 | 8 | 10.5k | 29-31 cyc = 145-155 ns | 8 cyc = 40 ns | 366k (21% / 84%) | 124k (3%) | 0 | 4 min, 4 GB |
| 16 | 32-16 (rho 16) | 16 | 16 | 10.5k | — | — | — | — | — | not finished: stopped by Claude Code's low-memory guard during scheduling/binding (~6 GB used); rerun alone |

All points meet timing in HLS (estimated clock 3.65 ns vs 5 ns target).

Observations:
- **The II is set by the per-particle conv layers and does not scale down linearly with PF.** It is 6-8 cycles at
  2 partitions and 8-12 at 4, because hls4ml's io_parallel pointwise conv has a fixed per-call overhead. Every other
  layer has II 1. **An II ≤ 5 cycles (25 ns) therefore needs PF = n (1 partition, fully parallel).** That is the run
  that has not finished yet (n16 small phi); the n16 phi 64-32 fully parallel run is too big for Vitis on this machine.
- **Every multiply goes into LUTs (DSP = 0).** hls4ml `Latency` with 8-bit operands maps products to fabric, so
  the 12,288 DSPs sit idle. Forcing products into DSPs (e.g. `config_op mul -impl dsp`, or packing two 8-bit
  products per DSP) is the biggest remaining resource lever.
- **FIFO/ping-pong buffers between dataflow stages are ~10% of the LUTs.** Removing `DATAFLOW` from the top
  (a pure pipeline) would save them.
- **LUT counts are over one SLR** (432k LUT) for every point except the small-phi one (84% of one SLR). A multi-SLR design needs Vivado floorplanning, and
  timing will be harder than the 3.65 ns HLS estimate suggests.
- **Fully parallel with phi 4-64-32 (38k MACs) fails in Vitis before resources even matter.** The unroll reaches
  2.4M instructions, then clang runs out of memory on this 61 GB machine. The small phi (10.5k MACs) got further
  (into scheduling). Run one synth at a time.

## 5. Recommendations

1. For an L1 design, retrain a trigger-sized student:
   - 16 particles, phi ≤ 4-32-16, ReLU only (no DynamicTanh), 8-bit QAT with power-of-2 scales, quantized biases.
   - Then run `convert.py --io-type io_parallel --strategy Latency --pf 8`.
2. Try moving multiplies into DSPs, and dropping the dataflow FIFOs, before shrinking the model further.
3. Rerun `python mini_parallel.py --n 16 --phi 32,16 --rho 16 --pf 16 --synth` alone (nothing else running on
   rdsrv409). It is the most likely point to reach II ≤ 5 cycles at modest resources.
4. Get the real budget from the trigger group: per-algorithm latency, and the TMUX period. With TMUX 6 an
   II of 150 ns is acceptable, which the current mini points already meet.
5. Run Vivado synth/implementation on the best point for real timing and resources (multi-SLR).
6. For the NERSC full-quant graph: first check C-sim parity with io_stream. Then try io_parallel at a small PF and
   check the three items in section 3.

## Code changes this session

- `convert.py` (committed in 7f85b07):
  - New `--strategy {Resource,Latency}` and `--pf N` flags. PF is set on `MatMul_0..3`.
  - Each option set writes its own `hls_prj/deepsets_distillnet_8bit_<io>_<strategy>_rf<N>[_pf<N>]`.
    The first run stays in `hls_prj/deepsets_distillnet_8bit`.
- `mini_parallel.py` (new, uncommitted): builds, converts, runs C-sim on and optionally synthesizes the random-weight mini
  DeepSets. Independent of `convert.py`.
- `README.md`: Running section (new flags and the mini script), and the "io_parallel attempt" section.
- `logs/`: moved `convert_log.txt` and `synth_log.txt` here. New logs `convert_parallel_pf16.txt`,
  `synth_parallel_pf16.txt` and `mini_*.txt`.
- `hls_prj/` is not in git.
