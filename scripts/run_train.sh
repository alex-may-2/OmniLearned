#!/bin/bash
# Config-driven launcher for `omnilearned train`.
#
#   scripts/run_train.sh <config> [--dry-run]
#   SAVE_TAG=..._r2 scripts/run_train.sh top_micro_a05    # env override, replicates
#
# <config> names a file under scripts/configs/train/ (without .sh).
# run_train.sh sources configs/train/_defaults.sh, then that config, assembles
# the CLI, and (unless --dry-run) runs it under srun across the current SLURM
# allocation. For a walltime-surviving resubmit loop, wrap this in
# lib/resubmit_loop.sh (or one of the distill_loop_top_*.sh shims).
#
# Handles both from-scratch training and --fine-tune runs (set FINETUNE=1 +
# PRETRAIN_TAG in the config); see configs/train/ft_*.sh.
#
# Env vars a caller may override per run: SAVE_TAG ALPHA BETA DISTILL_T.
#
# (Not named train.sh: scripts/train.sh is an unrelated pre-existing
# example cheat-sheet.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${1:?usage: run_train.sh <config> [--dry-run]}"
DRY_RUN="${2:-}"

OVERRIDES="SAVE_TAG ALPHA BETA DISTILL_T"
for _v in $OVERRIDES; do eval "_ovr_${_v}=\${${_v}-}"; done

# shellcheck source=configs/train/_defaults.sh
source "$SCRIPT_DIR/configs/train/_defaults.sh"

CFG="$SCRIPT_DIR/configs/train/${CONFIG}.sh"
[ -f "$CFG" ] || { echo "run_train.sh: no such config: $CFG" >&2; exit 2; }
# shellcheck source=/dev/null
source "$CFG"

for _v in $OVERRIDES; do
    eval "_o=\${_ovr_${_v}}"
    [ -n "$_o" ] && eval "${_v}=\$_o"
done

: "${SAVE_TAG:?config must set SAVE_TAG}"
[ -n "$TEACHER_DIR" ] || TEACHER_DIR="$TEACHER_ROOT/companion_$TEACHER_TAG"

args=(
    -o "$OUTDIR"
    --save-tag "$SAVE_TAG"
)
[ "$FINETUNE" = 1 ]           && args+=(--fine-tune --pretrain-tag "$PRETRAIN_TAG")
args+=(
    --dataset "$DATASET" --mode "$MODE" --num-classes "$NUM_CLASSES"
    --path "$DATA_PATH"
)
[ -n "$ARCH" ]                && args+=(--arch "$ARCH")
args+=(--size "$SIZE")
[ -n "$NUM_FEAT" ]            && args+=(--num-feat "$NUM_FEAT")
[ "$INTERACTION" = 1 ]        && args+=(--interaction)
[ "$LOCAL_INTERACTION" = 1 ]  && args+=(--local-interaction)
args+=(--batch "$BATCH")
[ -n "$ITERATIONS" ]         && args+=(--iterations "$ITERATIONS")
args+=(--epoch "$EPOCH" --lr "$LR")
[ -n "$LR_FACTOR" ]          && args+=(--lr-factor "$LR_FACTOR")
args+=(--wd "$WD")
[ -n "$WARMUP_EPOCH" ]       && args+=(--warmup-epoch "$WARMUP_EPOCH")
args+=(--num-workers "$NUM_WORKERS")
if [ "$DISTILL" = 1 ]; then
    args+=(
        --distill
        --teacher-labels-dir "$TEACHER_DIR"
        --teacher-tag "$TEACHER_TAG"
        --distill-alpha "$ALPHA" --distill-beta "$BETA" --distill-t "$DISTILL_T"
    )
fi
[ "$WANDB" = 1 ] && args+=(--wandb)
args+=(--resuming)
# shellcheck disable=SC2206
[ -n "$EXTRA_FLAGS" ] && args+=($EXTRA_FLAGS)

echo "run_train.sh: config=$CONFIG  save_tag=$SAVE_TAG"
echo "+ omnilearned train ${args[*]}"

[ "$DRY_RUN" = "--dry-run" ] && exit 0

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
run_ddp omnilearned train "${args[@]}"
