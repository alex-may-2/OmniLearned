# scripts/

Job launchers, training/eval loops, queue managers, and one-shot helpers for
running OmniLearned on Perlmutter. Everything that was previously loose in the
repo root now lives here.

## Run them from the repo root

```
bash scripts/<name>.sh
salloc -C gpu -q interactive ... bash scripts/<name>.sh
sbatch scripts/<name>.sbatch
screen -dmS <sess> bash scripts/<name>.sh
```

The scripts assume the current working directory is the **repo root**, not
`scripts/`. In particular many training scripts do `source export_ddp.sh`
from within `srun bash -c "..."`, which resolves against the submission CWD.

## Config-driven training (`run_train.sh` + `configs/train/`)

The `top`-tagging distillation family used to be ~22 near-copy scripts
(`distill_train_top*.sh` + `distill_loop_top*.sh`). It is now:

- `run_train.sh <config> [--dry-run]` — assembles and runs one
  `omnilearned train` invocation. `--dry-run` prints the command and exits.
- `configs/train/_defaults.sh` — every value shared across the family, in one
  place. `configs/train/<config>.sh` — only what that run changes.
- `lib/common.sh` — module loads, NCCL env, the `srun … source export_ddp.sh`
  wrapper (`run_ddp`).
- `lib/resubmit_loop.sh` — the generic salloc/tee/resubmit loop
  (`LOOP_LOG_DIR`, `LOOP_TAG`, `NODES`, `WALLTIME`, `MAX_LOOPS` via env).
- `distill_loop_top.sh [<config>]` — one resubmit-loop driver for every
  top config: picks the per-config log dir + `MAX_LOOPS` from a table
  (override via env), then wraps `resubmit_loop.sh` + `run_train.sh`.

```
scripts/run_train.sh top_micro_a05 --dry-run          # inspect
salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
       --gpus-per-node 4 -A m3246 bash scripts/run_train.sh top_micro_a05
SAVE_TAG=..._r3 scripts/run_train.sh top_micro_a05    # replicate (env override)
screen -dmS distill_top bash scripts/distill_loop_top.sh top_medium_a00_b10
```

### Worked example: distil a PET2-small top-tagging student

Goal: train the `top_small_a05` student (large fine-tuned top teacher →
PET2-small, KD α=β=0.5, T=4). All commands run from the **repo root**.

1. **See what it will run** — no allocation needed:

   ```
   scripts/run_train.sh top_small_a05 --dry-run
   ```

   `configs/train/top_small_a05.sh` sets only `SAVE_TAG=distill_top_small_scratch_a05_T4`;
   everything else (`--size small`, `--interaction --local-interaction`,
   `--batch 128`, `--epoch 50`, `--lr 5e-4`, `--distill … --distill-t 4`,
   teacher `companion_fine_tune_top_l`, `--wandb`) comes from
   `configs/train/_defaults.sh`.

2. **Launch it the normal way** — resubmit loop inside `screen`, so it
   survives disconnects and re-`salloc`s on walltime/preemption
   (`--resuming` picks up the checkpoint):

   ```
   screen -dmS distill_top_small bash scripts/distill_loop_top.sh top_small_a05
   ```

   Logs: `/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top/`
   (`summary.log` = one line per session; `session_*.out` = full output).
   Checkpoint: `/pscratch/.../checkpoints/best_model_distill_top_small_scratch_a05_T4.pt`.
   Stops on its own when epoch 50/50 is reached; caps at `MAX_LOOPS=20`
   sessions (`MAX_LOOPS=40 bash scripts/distill_loop_top.sh top_small_a05`
   to raise it).

3. **Or run a single session by hand** — one 4-node × 4-GPU interactive
   allocation, no loop:

   ```
   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
          --gpus-per-node 4 -A m3246 bash scripts/run_train.sh top_small_a05
   ```

4. **A replicate** (fresh random init) — override `SAVE_TAG`; the loop keys
   its per-session logs off `LOOP_TAG` so replicates share one log dir:

   ```
   bash scripts/distill_loop_top_rep.sh distill_top_small_scratch_a05_T4_r1
   ```

5. **Evaluate the trained student** on the top test split:

   ```
   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
          --gpus-per-node 4 -A m3246 bash scripts/run_eval.sh top_distill
   python tools/metrics/compute_metrics_top.py \
       --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill/ \
       --tag distill_top_small_scratch_a05_T4
   ```

To train a *different* config, swap `top_small_a05` for any name under
`configs/train/` (`ls scripts/configs/train/`). They differ only in their
`_defaults.sh` overrides, and each file's header comment names the old
per-experiment script it replaced.

### Customizing a run

Every knob is a shell variable. `run_train.sh` sources `_defaults.sh`, then
your **one** named config (configs don't chain — a new config inherits from
`_defaults.sh` only, so copy whatever deltas you need), then assembles the
CLI. `--dry-run` to see the result.

**Change a value** (lr, wd, batch, epoch, size, …) — set it in a config.
Prefer a new config over editing an existing one:

```
# scripts/configs/train/top_small_a05_lr1e4.sh
LR=1e-4
EXTRA_FLAGS="--warmup-epoch 1"
SAVE_TAG=distill_top_small_scratch_a05_T4_lr1e4_warmup
```

Variables `run_train.sh` reads: `LR WD BATCH ITERATIONS EPOCH SIZE ARCH
NUM_WORKERS INTERACTION LOCAL_INTERACTION WANDB DISTILL ALPHA BETA DISTILL_T
TEACHER_TAG TEACHER_DIR OUTDIR DATASET MODE NUM_CLASSES`.

**Add a flag that has no variable** — `EXTRA_FLAGS` is appended verbatim:
`EXTRA_FLAGS="--feature-drop 0.1 --lr-factor 5 --optim lion"`.

**Remove a flag** — toggle its variable: `INTERACTION=0`,
`LOCAL_INTERACTION=0`, `WANDB=0`, `DISTILL=0` (drops `--distill` and every
`--teacher-*`/`--distill-*` arg), `ARCH=` (empty → PET2 default, no
`--arch`). The always-on flags (`--resuming`, `--path`, `--mode`,
`--num-classes`) need an edit to `run_train.sh` itself.

**Override at call time** — only `SAVE_TAG ALPHA BETA DISTILL_T` (the
replicate/sweep whitelist): `SAVE_TAG=..._r2 scripts/run_train.sh
top_small_a05`. Anything else must live in a config.

`configs/eval/` + `run_eval.sh` work the same way; its call-time whitelist
is `SAVE_TAG DATASET_TYPE OUTDIR`.

**Fine-tune runs** use the same `run_train.sh` — a config sets `FINETUNE=1
DISTILL=0 PRETRAIN_TAG=<tag>` (plus `LR_FACTOR`, `WARMUP_EPOCH`, `NUM_FEAT`
as needed). See `configs/train/ft_*.sh`, which replace the old per-run
`fine_tune_{top_distill_pretrain*,jetclass,dctr_*}.sh`. Their resubmit-loop
wrappers (`distill_loop_{top_pretrain,dctr_*}_finetune.sh`,
`fine_tune_loop_jetclass.sh`) now call `run_train.sh ft_<config>`.

The per-config `distill_loop_top_*.sh` shims are gone, folded into
`distill_loop_top.sh <config>`. Still their own scripts: the arg-taking
`distill_loop_top_{rep,micro_rep,micro_noint_rep,sweep}.sh`. `distill_queue_*`
entries now name `distill_loop_top.sh:<config>`. `configs/train/` each note
which old script they replace.

**DeepSets / MLP students** are just `ARCH=` configs of the same
`run_train.sh`: `configs/train/top_{deepsets,mlp}_*.sh` set `ARCH=deep-sets`
/ `ARCH=mlp` with the interaction flags off. The old
`distill_train_top_deepsets.sh` + `distill_loop_top_deepsets.sh` `CONFIG=`
table is gone — its `CONFIG=<name>` is now
`distill_loop_top.sh top_deepsets_<name>`. For a non-default DeepSets width
(`nano|distillnet|micro|tiny|small|medium|large`), copy a `top_deepsets_*`
config and set `SIZE` (replaces the old `SIZE_OVERRIDE` env knob).

## Config-driven eval (`run_eval.sh` + `configs/eval/`)

Same shape for `omnilearned evaluate`: `run_eval.sh <config> [--dry-run]`
sources `configs/eval/_defaults.sh` + `configs/eval/<config>.sh`, assembles
the CLI, and runs it via `lib/common.sh`'s `run_ddp`. Output lands in
`$EVAL_ROOT/<config>/outputs_<save-tag>_<dataset>_<split>_rank*.npz`.

```
scripts/run_eval.sh top_distill --dry-run
SAVE_TAG=..._r2 scripts/run_eval.sh top_micro_ce      # env override
DATASET_TYPE=val scripts/run_eval.sh top_distill      # pick a split
```

Replaces the old per-checkpoint `evaluate_{top_distill*,jetclass_*,
top_micro_ce}.sh` (13 scripts). Still their own scripts (structurally
different): `evaluate_chunk.sh` / `evaluate_top_large.sh` (sharded/chunked
teacher dumps), `evaluate_metrics_top_{,T_}sweep.sh` (multi-tag metrics
tables), `evaluate_top_distill_deepsets.sh` + `run_eval_deepsets.sh`,
`evaluate_top_for_distill.sh`, `evaluate_train.sh`, `evaluate.sh`.

Not yet migrated: `distill_loop_top_T_sweep*.sh`, the JetClass/pretrain
`distill_{train,loop}_{jetclass,pretrain}*.sh` families, and the cosmology
`fine_tune_{camels,quijote}.sh` outliers.

### quark/gluon (qg) configs

Ported from `distill_dev`:

- `configs/train/ft_qg_pretrain_l.sh` — CE-only fine-tune of `pretrain_l` on qg.
- `configs/train/qg_a05.sh` — distil that teacher into PET2-small, a=b=0.5/T=4.
  Needs `companion_fine_tune_qg_pretrain_l` logits built first.
- `configs/eval/qg_finetune_pretrain_l.sh`, `configs/eval/qg_distill.sh` — test-split evals.

The `distill_dev` `QUANTIZE=int8|bf16` eval knob needs the torchao path in
`evaluate.py`, not on this branch; omitted for now.

## `export_ddp.sh`

The real file is `../export_ddp.sh` (repo root); `scripts/export_ddp.sh` is a
symlink to it. This keeps both reference styles working:

- bare `source export_ddp.sh` (CWD = repo root) — used inside `srun bash -c`
- `source "$SCRIPT_DIR/export_ddp.sh"` — used by a few generator/watch scripts

## Cross-references

- Loop/queue scripts call their sibling launchers via
  `bash "$SCRIPT_DIR/<sibling>.sh"`, so they resolve regardless of CWD.
- Scripts that call a Python helper reference it under `tools/` (live:
  `tools/metrics/compute_metrics_top.py`, `tools/preprocess/build_teacher_h5.py`,
  …) or `analysis/` (frozen one-offs) — either by path after a `cd` to the
  root, or as `"$SCRIPT_DIR/../<dir>/<file>.py"`.

## Notable entry points

- `configs/train/top_deepsets_*.sh` + `distill_loop_top.sh` /
  `run_eval_deepsets.sh` — DeepSets-KD family
  (see `../docs/EXPERIMENTS_deepsets_kd.md`)
- `run_all_chunks.sh` / `submit_datasets.sh` — chunked teacher-logit generation
- `train_omnifold_pythia_herwig.sh` / `distill_loop_omnifold.sh` — OmniFold unfolding
- `qat_train_deepsets_distillnet_fpga_8bit.sh` / `fullquant_chain.sbatch` — full-quant QAT for the FPGA student
