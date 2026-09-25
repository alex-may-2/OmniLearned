#!/bin/bash

# needed to get the signal evluate outputs

# Evaluate an already-fine-tuned lhco_ad checkpoint on the leftover-signal
# file build_lhco_eval_signal.py builds (pure, never-injected signal) --
# same checkpoint as evaluate_lhco_ad.sh, different --path/-o so this run's
# outputs_*.npz can't collide with the real test-set evaluate output.
#
# EDIT THESE to match the fine_tune_lhco_ad.sh run being evaluated -- same
# SAVE_TAG_BASE/NSIG/DESCRIPT_TAG/SIZE as evaluate_lhco_ad.sh used.
# --interaction/--local-interaction below must ALSO match that run's flags
# exactly -- --local-interaction changes local_physics' MLP input width
# (4 -> 7, see layers.py's LocalEmbeddingBlock), so a mismatch here fails
# checkpoint loading with a shape-mismatch RuntimeError, not a silent bug.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246
# then (from scripts/lhco/) run bash evaluate_lhco_eval_signal.sh

module load conda
conda activate ol_distill

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

export MASTER_ADDR=$(hostname)

# ============================================================
#  EDIT THESE PER RUN -- must match the fine-tune script's values
# ============================================================
SAVE_TAG_BASE=fine_tune_pretrain_s
DATASET=lhco_ad
SIZE=small
NSIG=2000
DESCRIPT_TAG=r1
QUANTIZATION=none
# ============================================================

SAVE_TAG="${SAVE_TAG_BASE}_${DATASET}_nsig${NSIG}_${DESCRIPT_TAG}"
CHECKPOINT_DIR=/pscratch/sd/m/mbenyas/LHCO
LHCO_PATH="/global/cfs/cdirs/m3246/mbenyas/OmniLearned_distillation/LHCO/nsig_${NSIG}/eval_signal"
OUTPUT_DIR=/pscratch/sd/m/mbenyas/${SAVE_TAG}_eval_signal_${QUANTIZATION}

mkdir -p "$OUTPUT_DIR"
export QUANTIZE=${QUANTIZATION}

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset ${DATASET} \
    --path ${LHCO_PATH} \
    --size ${SIZE} \
    --use-add --num-add 2 \
    --conditional --num-cond 11 \
    --interaction --local-interaction \
    --num-classes 2 \
    --batch 16 \
    --num-workers 4 \
    --dataset-type test"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
