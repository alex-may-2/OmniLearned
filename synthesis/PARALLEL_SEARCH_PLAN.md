# Single-SLR search for the full-quant DeepSets at II ≈ 25 ns (rdsrv409 + NERSC)

## How to read paths in this plan

You (the agent) run on Alex's local machine. You reach two remote machines over ssh. Nothing in this plan is on
the local machine. Every path is prefixed with the machine it lives on:

- **`rdsrv:`** is rdsrv409 (ssh alias `rdsrv`, through the `slac3` proxy). It does hls4ml, Vitis HLS and C-sim.
  Repo: `rdsrv:/u1/alexmay/working/OmniLearned`. Paths such as `rdsrv:synthesis/...` are relative to that repo.
- **`nersc:`** is Perlmutter (ssh alias `nersc`). It does GPU training, QAT and export. Repo (your own clone):
  `nersc:/pscratch/sd/a/alexmay/OmniLearned`. Paths such as `nersc:src/...` are relative to that clone.
- Training code (`src/`, `tools/quantize/`, `scripts/`) exists in both repos, but runs only on nersc. Read and
  edit it on nersc.
- `synthesis/` is used only on rdsrv.
- Code moves between the two repos only through git (branch `parallel-search`).
- Data files (ONNX graphs, logs) move with `scp -3 nersc:... rdsrv:...`, or through `/tmp` on the local machine.
- Before you start, confirm that `ssh rdsrv true` and `ssh nersc true` both work.
- The NERSC sshproxy cert lasts 24 h. Check it with `ssh-keygen -L -f ~/.ssh/nersc-cert.pub`. If it has expired, ask
  Alex to renew it.

## Context

The r7 full-quant graph (`rdsrv:synthesis/onnx_graphs/qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant_r7_clean.onnx`)
is bit-exact in HLS. Details are in `rdsrv:synthesis/io_parallel_report.md`.

- io_stream (d32 n64) fits one SLR: 300k LUT (69% SLR), 1080 DSP, II 68 cycles = 340 ns at 5 ns.
- io_parallel at d32 n64 does not fit the device.

**Requirement (2026-09-30): the L1T data rate needs one jet every 25 ns.** II(ns) = II cycles × clock period.

- **Copies.** k copies of the tagger IP take turns on jets (round-robin), one copy per SLR. A jet never crosses an
  SLR. Each copy needs II ≤ 25·k ns. Scan k ∈ {1, 2, 4}.
- **Clock.** Free choice among 200, 240, 320 and 360 MHz.
- **Cycle budget per copy:** C = 25·k ns × f.

| f | k=1 | k=2 | k=4 |
|---|---|---|---|
| 200 MHz (5 ns) | 5 | 10 | 20 |
| 240 MHz (4.17 ns) | 6 | 12 | 24 |
| 320 MHz (3.125 ns) | 8 | 16 | 32 |
| 360 MHz (2.78 ns) | 9 | 18 | 36 |

**Goal:** for each k, find the design with the best AUC that meets all of these:
- **II (hard):** csynth top-level Interval ≤ C cycles at the target clock.
- **One SLR (hard):** ≤ 432k LUT (target < 80%), ≤ 864k FF, ≤ 3072 DSP, ≤ 1344 BRAM_18K. Read the
  `Utilization SLR (%)` row of `*_csynth.rpt`.
- **Timing (hard, csynth only):** the estimated clock is ≤ the target period. There is no Vivado P&R. Label every
  result at 320 or 360 MHz "csynth only, not P&R-confirmed".
- **AUC (soft):** reference 0.9644 (r7 − 0.01). Report each winner's AUC and its shortfall.

**Two design families:**

- **io_parallel** (k=1, k=2).
  - II cycles = n_partitions × RF, with n_partitions = n/PF. This assumes the multiplier-limit fix below holds.
  - LUT ≈ 1.2k × n × dim for elementwise + FIFO (fixed, whatever the II), plus ~60 × MACs/jet ÷ II cycles for the
    convs when they sit in LUT.
  - One SLR therefore allows roughly n × dim ≲ 250, unless the elementwise levers cut the 1.2k coefficient.
  - MACs/jet for `d<dim>p2r1`, ratio 2 ≈ n(6d² + 8d) + 4d² + 2d.
- **io_stream** (k=2, k=4).
  - II ≈ n + 4 cycles, since it reads one particle per clock.
  - LUT barely depends on n, so the r7 width d32 stays affordable.
  - Reachable n ≤ C − 4:
    - k=4: n 32 at 360 MHz, 28 at 320, 20 at 240, 16 at 200;
    - k=2: n ≤ 14 at 360 MHz, 12 at 320;
    - k=1: none.

**II floor (found 2026-09-30).**
- In `nnet_conv1d_latency.h`, the partition loop has `PIPELINE II=reuse_factor rewind` and
  `ALLOCATION mul limit=multiplier_limit`.
- `parameters.h` sets `multiplier_limit = n_chan × n_filt / RF`, which counts one particle. Each iteration, however,
  processes `n_pixels = PF` particles.
- Evidence: kept project `mini_fq_n16_phi32-16_rho16_pf8` has conv II 8 cycles with `n_partitions = 2`. At PF 2
  (`mini_ds_d16r2_phi1_rho1_n32_pf2`), II = n/PF.
- Hypothesis: scaling the limit by `n_pixels` gives II = n_partitions × RF at any PF.

## Facts

**rdsrv409**

- Env: `rdsrv:/u1/alexmay/conda/envs/omnilearned-hls`, with Vitis HLS 2024.1 and hls4ml `0.1.0.dev2733+g1d85133b3`.
- Target: part `xcvu13p-flga2577-2-e`. The clock is now a knob. The default is still 5 ns, because neither script has
  a `--clock` flag yet.
- Per SLR: 432k LUT, 864k FF, 3072 DSP, 1344 BRAM_18K. Current designs use 0 DSP.
- Machine: 61 GB RAM, 24 cores. csynth memory grows with the unrolled multiplies per partition: 30 GB at
  n × dim = 512 with PF 2, and an OOM at PF 16 with 10.5k MACs.
- Read first:
  - `rdsrv:synthesis/README.md`: env setup, hls4ml patches, full-quant section;
  - `rdsrv:synthesis/io_parallel_report.md`;
  - `rdsrv:synthesis/mini_parallel.py`: random-weight mini builder, with `--full-quant`, `--distillnet`,
    `--phi-blocks`, `--rho-blocks`, `--ratio` and `--pf`;
  - `rdsrv:synthesis/convert.py`: real-graph conversion, with `--io-type --strategy --pf --reuse-factor --synth`;
  - the kept r7 io_stream project: `rdsrv:synthesis/hls_prj/deepsets_distillnet_8bit_fullQuant_r7_io_stream_resource_rf1`.

**NERSC**

- Account `m3246`.
- Env: `nersc:/global/cfs/cdirs/m2616/alexm/conda/envs/omnilearned-fpga`. Call its python by absolute path; a
  bare `python` after `conda activate` has failed.
- Your clone does not exist yet. Clone `git@github.com:alex-may-2/OmniLearned.git` into
  `/pscratch/sd/a/alexmay/OmniLearned`.
- Your checkpoints: `nersc:/pscratch/sd/a/alexmay/omnilearned/checkpoints` (r1–r8 full-quant QAT are there).
- Your exported graphs: `nersc:/pscratch/sd/a/alexmay/omnilearned/qonnx/fpga/`.
- Data: `nersc:/global/cfs/cdirs/m4567/www/`.
- Teacher logits are under twamorka's pscratch (the `qat_deepsets.py` defaults). Read only.

**Training chain (nersc)**

1. Float KD: `nersc:scripts/configs/train/top_deepsets_distillnet_fpga.sh`, run through `nersc:scripts/distill_loop_top.sh`.
2. QAT: `nersc:tools/quantize/qat_deepsets.py`.
3. Eval: `nersc:tools/quantize/qat_deepsets_eval.py`.
4. Export: `nersc:tools/quantize/qat_deepsets_export_qonnx.py`. It writes `*_clean.onnx`.

`nersc:scripts/fpga_distillnet_pipeline.sh` chains all of them. QAT cannot train from scratch: every new shape
needs a float KD checkpoint first.

**Architecture knobs (nersc)**

- Size: `--size d<dim>p<phi>r<rho>` (`nersc:src/omnilearned/utils.py:get_deepsets_parameters`).
  - `p` = embed + (p−1) residual phi blocks. `r` = residual rho blocks; `r0` means pool → Linear.
  - `distillnet` = `d32p2r1`. **Fewer blocks (p1, r0) need no code.**
  - In `mini_parallel.py`, `--phi-blocks` counts residual blocks only, so p1 corresponds to `--phi-blocks 0`.
- `--deepsets-fixed-n` sets the particle count (64 now).
- mlp_ratio (2) is not on the CLI. `DeepSets(mlp_ratio=2)` is the default in `network.py`.
- DyT: `norm_layer=DynamicTanh` is used in the embed MLP, in each phi pre-norm and in each rho pre-norm.
- Add a training flag for mlp_ratio or DyT **only if Phase 1a shows that it pays** (see Phase 2).

**r7 QAT recipe**

- Flags: `--full-quant --tanh-in-max 4 --res-bits 10 --relu-uint`.
- Schedule: warm start, 15 epochs at lr 5e-5. r7 itself was r4 + 15 epochs at 2e-5.
- Export with the same flags.
- Uses one GPU node, ~215–250 s/epoch at dim 32, n 64.

**Reference metrics**

| model | acc | AUC | 1/eB |
|---|---|---|---|
| float distillnet | — | 0.9770 | — |
| r7 | 0.9173 | 0.9744 | 123.9 |

QAT costs about 0.003 AUC.

## Rules

**rdsrv**

- Run **one synth at a time.**
- Start each synth detached (`nohup` or `tmux`) and poll its log. Never hold an ssh session open for a run.
- Synth command: `timeout 2h /usr/bin/time -v python ... --synth 2>&1 | tee logs/<tag>.txt`, run in `rdsrv:synthesis/`.
  Record the peak RAM.
- Kill a stuck run by PID, never with `pkill -f`.
- Synth only after C-sim is bit-exact (max |Δ logit| = 0).
- Add lever flags to `mini_parallel.py` only when you test that lever. Default off, so existing modes do not change.
- Edit `rdsrv:synthesis/convert.py` only to add flags (`--clock`, plus each lever that Phase 1a keeps).

**nersc**

- Work only in the clone at `nersc:/pscratch/sd/a/alexmay/OmniLearned`.
- Code changes are **additive only**: new configs, new sbatch scripts, and new CLI flags whose default reproduces
  today's model.
- Do not change existing presets or defaults. Never touch twamorka's repo or checkpoints.
- Submit with `sbatch`. Run at most 4 jobs at once.
- Use new save tags only: `ps_d<dim>p<p>r<r>_n<n>[_m<ratio>][_nodyt<...>]`.

**git (both machines)**

- Branch `parallel-search`, created from `synthesis`.
- Commit and push only to sync code between rdsrv and nersc. Remote: `git@github.com:alex-may-2/OmniLearned.git`.
- Never merge into `synthesis` (or `parallelize`).

**Run log**

- Keep `rdsrv:synthesis/parallel_search_log.md`, one row per run on either machine: tag, machine, family, k, clock,
  config, status, II cycles/ns, %SLR LUT, estimated clock, AUC.
- Update it whenever a run is launched or finishes.

## Phases

### Phase 0: inspect and mirror r7 (rdsrv, ~1–2 h)

1. Dump `rdsrv:synthesis/onnx_graphs/..._fullQuant_r7_clean.onnx` node by node with onnx/qonnx: op, shapes,
   Quant scale/bits/signed/narrow, Mul alpha values, residual Adds, and the 10-bit residual/pool Quants.
2. Compare it with `rdsrv:synthesis/mini_parallel.py --distillnet`. Known gaps:
   - 10-bit Quants on the residual stream, the pool input and the pool output;
   - unsigned Quants on the `phi.fc2` and `rho.fc2` inputs.
3. Make `--distillnet` match r7 exactly. Add `--res-bits` and `--relu-uint` flags (defaults on).
4. Add `--clock <ns>` to `mini_parallel.py` and `convert.py`. Pass it to hls4ml as `clock_period`.
5. Add `--io-type {io_parallel,io_stream}` to `mini_parallel.py`. Copy the io_stream settings from `convert.py`.
6. Check that the other modes are unchanged.
7. C-sim must be bit-exact at d16 n16 in both io types before going further.

### Phase 1a: II and lever probe (rdsrv, sequential, d8 n16 `--distillnet` unless noted)

Run cheapest first. Keep a lever only if it lowers %SLR at a fixed II, or lowers II at a fixed %SLR.

| tag | family | change | clock | question |
|---|---|---|---|---|
| L0 | io_parallel PF 2 | baseline | 5 ns | II 8? Refit the 1.2k coefficient at d8 |
| L1 | io_parallel PF 4 | baseline | 5 ns | II 4, or the floor? |
| L2 | io_parallel PF 4 | `--mult-limit-fix` | 5 ns | II = n/PF? |
| L3 | io_parallel PF 16, RF 5 | `--mult-limit-fix --rf 5` | 5 ns | II 5? Peak RAM? |
| L4 | best of L1–L3 | `--dsp-mult` | 5 ns | conv LUT moves to DSP (≤ 3072)? |
| L5 | best so far | `--pipeline-style pipeline` | 5 ns | FIFO LUT gone? (hls4ml may refuse it with convs) |
| L6 | best so far | — | 3.125 and 2.78 ns | estimated clock, FF growth, latency cycles |
| T1 | io_stream d32 n32 | — | 2.78 ns | II ≤ 36? LUT ≈ r7? estimated clock? |
| T2 | io_stream d32 n12 | — | 2.78 ns | II ≤ 18 for k=2? |

How each lever works:
- **`--mult-limit-fix`:** scale the conv `multiplier_limit` by `n_pixels`. Patch it in the generated config, the same
  way `convert.py` patches the rest of hls4ml.
- **`--rf`:** set ReuseFactor on the per-particle MatMuls.
- **`--dsp-mult`:** add `config_op mul -impl dsp` to `build_prj.tcl` before csynth. Check the syntax in UG1399
  (2024.1). If the total multiplier count exceeds 3072, bind only the convs.
- **`--pipeline-style`:** set the hls4ml `Model: PipelineStyle` config.
- Fusing ReLU → alpha → Quant → tanh into one lookup is tried **only if** L5 fails. A pipeline-style design should
  let Vitis merge those stages itself.

Model-side hardware probes (random weights, cheap), run on the best L-config:

| tag | knob | mini flag |
|---|---|---|
| M1 | p1 (no residual phi block) | `--phi-blocks 0` |
| M2 | r0 (no rho block) | `--rho-blocks 0` |
| M3 | mlp_ratio 1 | `--ratio 1` |
| M4 | no embed DyT | new `--no-embed-dyt` |

Carry a model knob into training only if it cuts LUT per n × dim by about 15% or more.

### Phase 1b: shape grid per k (rdsrv, sequential, ≤ 15 synths)

Use the kept levers and the refit cost rule to predict, for each (family, k, clock), the largest shape that fits.
Synthesize the cheapest first.

- **k=1, io_parallel** (C 5–9): (d8,n16), (d8,n24), (d12,n16), (d16,n16), plus any kept model knob.
  - Choose n_partitions × RF ≤ C.
  - Prefer more partitions and RF 1, because a lower PF unrolls less and uses less csynth RAM.
- **k=2:**
  - io_parallel (C 10–18): (d12,n16), (d8,n32), (d16,n16);
  - io_stream: d32, n 12–14 at 320/360 MHz.
- **k=4:**
  - io_stream: d32 n32 at 360 MHz, d32 n28 at 320 MHz, d32 n16–20 at 200/240 MHz;
  - io_parallel only if it beats io_stream on AUC.

### Phase 2: NERSC accuracy screens (start once L0–L2 and T1 confirm the families)

1. Clone the repo on nersc and create branch `parallel-search` there (pull it from rdsrv's push).
2. Launch float KD screens:
   - 1 GPU node each, 15–20 epochs;
   - flags: `--size d<dim>p<p>r<r> --deepsets-fixed-n <n>`;
   - KD recipe copied from the distillnet config.
3. Screen list, about 11 runs, waves of 4:
   - io_parallel shapes: d8p2r1 n16, d8p2r1 n24, d12p2r1 n16, d16p2r1 n16, plus the variants that Phase 1a kept
     (for example d12p1r1 n16);
   - io_stream shapes: d32p2r1 at n 32, 28, 16 and 12;
   - control: d32p2r1 n64 on the same short schedule, to measure the undertraining.
4. AUC is soft, so there is no pass threshold.
   - Rank the shapes per k by the corrected float AUC.
   - Estimated QAT AUC = float AUC − 0.003.
5. Add the training flag for mlp_ratio or DyT only if Phase 1a kept that knob:
   - `--deepsets-mlp-ratio` (default 2);
   - a DyT-off flag (default off).

   Thread the flag through `train.py`, `qat_deepsets.py`, `qat_deepsets_eval.py` and `qat_deepsets_export_qonnx.py`.
   Additions only.
6. Phase 1b continues on rdsrv while the screens train.

### Phase 3: finalists (nersc training, then rdsrv HLS)

1. **Pick per k:** the shape with the best screen AUC among those that meet the k's II and one-SLR limits on rdsrv.
   Break ties by %SLR margin, then by II margin.
   - Take 1–2 per k.
   - One shape may serve several k.
2. **Train on nersc:** full float KD (50 epochs), QAT with the r7 recipe, `qat_deepsets_eval.py`, then export.
3. **Copy:** move each `nersc:/pscratch/sd/a/alexmay/omnilearned/qonnx/fpga/<tag>_clean.onnx` into
   `rdsrv:synthesis/onnx_graphs/` with `scp -3`.
4. **Synthesize on rdsrv:** `python convert.py --onnx onnx_graphs/<tag>_clean.onnx --io-type <family> --clock <ns>`,
   plus `--strategy`, `--pf`, `--reuse-factor` and the kept lever flags, then `--synth`.
   - Check it is bit-exact on 9984 jets.
   - Read the HLS acc, AUC and 1/eB from its output.
   - The real-weight resources replace the random-weight estimates.
5. **Pass:** II ≤ C at the target clock, fits one SLR, and the estimated clock is ≤ the target. Report the AUC and its
   shortfall from 0.9644.

### Phase 4: report (rdsrv)

Write `rdsrv:synthesis/io_parallel_slr_report.md` containing:
- the r7 topology summary and any deviations;
- the II-floor finding and the lever table (Phase 1a), with what each lever gained;
- **per-k winner table:** family, shape, clock, II cycles and ns, effective II with k copies, latency, LUT/FF/DSP/BRAM
  as % of one SLR, estimated clock ("csynth only" at ≥ 320 MHz), csynth time, peak RAM;
- an accuracy table: float screen, QAT, HLS, and the shortfall from 0.9644;
- the AUC vs %SLR front per k;
- what is not covered: the round-robin distributor and merger for k copies, and Vivado P&R;
- the winners' nersc training commands.

Add a 3-line pointer in `rdsrv:synthesis/README.md`. Commit to `parallel-search` and push. Stop, and ask Alex
before any merge.

## Verification

- **Every synth (rdsrv):**
  - C-sim max |Δ logit| = 0;
  - io_parallel: `firmware/parameters.h` shows the intended `n_partitions` and RF;
  - with `--mult-limit-fix`, `multiplier_limit` covers `n_pixels`;
  - tanh `table_size` 256;
  - numbers come from `*_csynth.rpt`: top Interval, the `Utilization SLR (%)` row, and the Timing "Estimated" value;
  - the II check is Interval × target period ≤ 25·k ns.
- **Every finalist:** the nersc QAT eval AUC equals the rdsrv HLS C-sim AUC.
- **Training code untouched:** `git diff synthesis -- src/ tools/` on `parallel-search` shows additions only. Any new
  flag defaults to today's behavior, and with it unset, an r7-config forward pass gives identical logits. No existing
  checkpoint, graph, preset or default was touched.

## Kickoff prompt for the local agent (~10 lines)

- `ssh rdsrv true` and `ssh nersc true`. Check the NERSC cert expiry, and ask Alex to renew it if it has expired.
- On rdsrv: `cd /u1/alexmay/working/OmniLearned && git pull`, then read `synthesis/PARALLEL_SEARCH_PLAN.md` fully.
- Create branch `parallel-search` from `synthesis` and push it.
- Clone the repo on nersc into `/pscratch/sd/a/alexmay/OmniLearned` and check out `parallel-search`.
- Create `synthesis/parallel_search_log.md`.
- Start Phase 0. Report after Phase 0 and after Phase 1a (L0–L2, T1), before launching nersc screens.
