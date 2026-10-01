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
| L3 | rdsrv | io_parallel | 1 | 5 ns | mini ds d8 n16 PF16 RF5 --mult-limit-fix | done | 5 / 25 | 102 / 5 / 0 | 3.53 | | RF not applied (all reuse_factor 1); PF=n made hls4ml pick pipeline style: FF 47k, LUT 439k, 50 GB, 18 min. Loses to L2 |
| S-ctl | nersc 59152539 | float KD | - | - | d32p2r1 n64, 20 ep, 1 node | pending (interactive) | | | | | regular QOS had ~4800 GPU jobs pending: moved to salloc -q interactive (max 2 jobs/user) |
| S-d8n16 | nersc 59152495 | float KD | - | - | d8p2r1 n16, 20 ep | running (interactive) | | | | | io_parallel k=1 |
| S-d12n16 | nersc | float KD | - | - | d12p2r1 n16, 20 ep | not started | | | | | interactive QOS allows 2 jobs/user |
| S-d32n32 | nersc | float KD | - | - | d32p2r1 n32, 20 ep | not started | | | | | interactive QOS allows 2 jobs/user |
| L2b | rdsrv | io_parallel | 1 | 5 ns | L2 + pool-result Quant type | done | 4 / 20 | 68 / 15 / 0 | 5.84 MISS | | chain moved: pool (3.0) -> rho alpha+Quant (2.8). Next fix: register pool output (LATENCY min=1) |
| T3 | rdsrv | io_stream | 4 | 3.125 ns | mini ds d32 n24 | done | 32 / 99.8 | 83 / 26 / 86 | 3.29 MISS | | II meets k=4; clock misses in Resource dense of the stream conv. Try Latency strategy (T4), 240 MHz (T5) |
| L6a | rdsrv | io_parallel | 1 | 3.125 ns | d8 n16 PF2 mlf (+pool LATENCY min=1) | done | 8 / 25.0 | 57 / 16 / 0 | 4.55 MISS | | II meets k=1; pool register did not help. Cause: pool output and residual skip arrays have 2 readers in DATAFLOW -> clone fix (L6c) |
| T2b | rdsrv | io_stream | 2 | 2.78 ns | mini ds d32 n12 | done | 23 / 63.9 | 84 / 29 / 86 | 3.04 MISS | | II = n + 11: io_stream cannot meet k=2 (50 ns) |
| L4 | rdsrv | io_parallel | 1 | 5 ns | L2b + --dsp-mult | done | 4 / 20 | 60 / 15 / 26 | 5.84 MISS | | 804 mults to DSP, LUT 263k (-11% vs L2b). Keep (free). Miss = fan-out chain (pre-clone) |
| L5 | rdsrv | io_parallel | 1 | 5 ns | L2b + --pipeline-style pipeline | queued | | | | | |
| L6b | rdsrv | io_parallel | 1 | 2.78 ns | d8 n16 PF2 mlf + fan-out clones | done | 8 / 22.2 | 60 / 18 / 0 | 2.01 | | FIRST k=1 PASS (csynth only): 263k LUT, latency 98 cyc = 272 ns. Clone fix removes the chain |
| L2c | rdsrv | io_parallel | 1 | 5 ns | L2b + pool output register | queued | | | | | |
| M1-M4 | rdsrv | io_parallel | 1 | 2.78 ns | L6b + p1 / r0 / ratio 1 / no embed DyT | queued | | | | | model knobs (re-planned on L6b) |
| B1-B5 | rdsrv | io_parallel | 1-2 | 2.78 ns | d12n16 PF2, d8n32 PF4 (k=1); d16n8 PF1 (k=1); d16n16 PF1, d12n16 PF1 (k=2) | queued | | | | | Phase 1b at 360 MHz |
| T4 | rdsrv | io_stream | 4 | 3.125 ns | mini ds d32 n24 --strategy Latency | done | 32 / 99.8 | 115 / 20 / 5 | 2.28 | | meets II and clock but mults went to LUT (500k). T6 adds --dsp-mult |
| T5 | rdsrv | io_stream | 4 | 4.17 ns | mini ds d32 n16 | done | 20 / 83.4 | 82 / 21 / 81 | 3.04 | | k=4 PASS (csynth only); LUT 82% > 80% target; n 20 would give II 24 = 100 ns |
| L6c | rdsrv | io_parallel | 1 | 3.125 ns | L6a with fan-out clones | done | 8 / 25.0 | 58 / 17 / 0 | 2.28 | | k=1 PASS at 320 MHz too (254k LUT). ~25 min of rdsrv idle after it (stuck queue waiter, fixed) |
| T6 | rdsrv | io_stream | 4 | 3.125 ns | T4 + --dsp-mult | queued | | | | | |
