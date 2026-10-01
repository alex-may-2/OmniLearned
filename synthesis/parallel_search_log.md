# parallel-search run log

One row per run on either machine. Plan: `synthesis/PARALLEL_SEARCH_PLAN.md`. II ns = II cycles x target clock.
Resources are % of one SLR (432k LUT, 864k FF, 3072 DSP, 1344 BRAM_18K) from csynth estimates.

| tag | machine | family | k | clock | config | status | II cyc / ns | LUT / FF / DSP % SLR | est. clock | AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0-chk | rdsrv | both | - | 5 ns | C-sim parity: default, full-quant (onnx identical to kept), r7-matched distillnet d16 n16 PF2 and io_stream | done | | | | | max dlogit 0 everywhere. Fix: an alpha-1 Mul made hls4ml drop the tanh-input Quant; mini now omits Mul by 1 |
| L0 | rdsrv | io_parallel | 1 | 5 ns | mini ds d8 n16 PF2 | done | 8 / 40 | 59 / 14 / 0 | 5.79 MISS | | 256k LUT, 2.5 min, 3.8 GB. Miss: pool-out Quant chained with rho alpha+Quant (fixed after L2) |
| L1 | rdsrv | io_parallel | 1 | 5 ns | mini ds d8 n16 PF4 | done | 8 / 40 | 64 / 15 / 0 | 5.79 MISS | | floor confirmed: 4 partitions x 2 cyc; 279k LUT |
| L2 | rdsrv | io_parallel | 1 | 5 ns | mini ds d8 n16 PF4 --mult-limit-fix | done | 4 / 20 | 68 / 15 / 0 | 5.79 MISS | | fix works: II = n/PF; 295k LUT, 4 GB. Miss = pool chain (fix landed after start) |
| T2 | rdsrv | io_stream | 2 | 2.78 ns | mini ds d32 n12 | refused | | | | | C-sim 0.006 off: mean over n=12 truncated; fixed with 6 more pool accum fraction bits (rerun T2b) |
| T1 | rdsrv | io_stream | 4 | 2.78 ns | mini ds d32 n32 | done | 38 / 105.6 | 83 / 27 / 81 | 3.04 MISS | | II = n + 6 here; misses k=4 budget (100 ns) and clock; 360k LUT, 2516 DSP |
| L3 | rdsrv | io_parallel | 1 | 5 ns | mini ds d8 n16 PF16 RF5 --mult-limit-fix | queued | | | | | RAM risk |
| S-ctl | nersc 59151837 | float KD | - | - | d32p2r1 n64, 20 ep, 1 node | queued | | | | | short-schedule control |
| S-d8n16 | nersc 59151838 | float KD | - | - | d8p2r1 n16, 20 ep | queued | | | | | io_parallel k=1 |
| S-d12n16 | nersc 59151839 | float KD | - | - | d12p2r1 n16, 20 ep | queued | | | | | io_parallel k=1/2 |
| S-d32n32 | nersc 59151840 | float KD | - | - | d32p2r1 n32, 20 ep | queued | | | | | io_stream k=4 |
| L2b | rdsrv | io_parallel | 1 | 5 ns | L2 + pool-result Quant type | queued | | | | | timing fix check |
| T3 | rdsrv | io_stream | 4 | 3.125 ns | mini ds d32 n24 | queued | | | | | k=4 at 320 MHz |
| L6a | rdsrv | io_parallel | 1 | 3.125 ns | d8 n16 PF2 mlf | queued | | | | | k=1 at 320 MHz |
| T2b | rdsrv | io_stream | 2 | 2.78 ns | mini ds d32 n12 | queued | | | | | k=2 at 360 MHz |
| L4 | rdsrv | io_parallel | 1 | 5 ns | L2b + --dsp-mult | queued | | | | | |
| L5 | rdsrv | io_parallel | 1 | 5 ns | L2b + --pipeline-style pipeline | queued | | | | | |
| L6b | rdsrv | io_parallel | 1 | 2.78 ns | d8 n16 PF2 mlf | queued | | | | | k=1 at 360 MHz |
