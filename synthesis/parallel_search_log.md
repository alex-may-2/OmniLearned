# parallel-search run log

One row per run on either machine. Plan: `synthesis/PARALLEL_SEARCH_PLAN.md`. II ns = II cycles x target clock.
Resources are % of one SLR (432k LUT, 864k FF, 3072 DSP, 1344 BRAM_18K) from csynth estimates.

| tag | machine | family | k | clock | config | status | II cyc / ns | LUT / FF / DSP % SLR | est. clock | AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0-chk | rdsrv | both | - | 5 ns | C-sim parity: default, full-quant (onnx identical to kept), r7-matched distillnet d16 n16 PF2 and io_stream | done | | | | | max dlogit 0 everywhere. Fix: an alpha-1 Mul made hls4ml drop the tanh-input Quant; mini now omits Mul by 1 |
