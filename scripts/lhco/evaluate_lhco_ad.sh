#!/bin/bash

# needed to get the held-out background test set evaluate output

# Evaluate the model fine-tuned on the idealized LHCO anomaly-detection
# dataset (lhco_ad) on a split -- scores go to outputs_<save-tag>_*.npz,
# with `prediction`/`logits` = the classifier's data-vs-background score
# and `cond` = global (mjj, then per-jet log pT/eta/phi/log mass/mult).
#
# EDIT THESE to match the fine_tune_lhco_ad.sh run being evaluated.
# --interaction/--local-interaction below must ALSO match that run's flags
# exactly -- --local-interaction changes local_physics' MLP input width
# (4 -> 7, see layers.py's LocalEmbeddingBlock), so a mismatch here fails
# checkpoint loading with a shape-mismatch RuntimeError, not a silent bug.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246
# then (from scripts/lhco/) run bash evaluate_lhco_ad.sh

module load conda
conda activate ol_distill
# module load pytorch

# See fine_tune_lhco_ad.sh: ol_distill's editable install points at the main
# checkout, so this worktree's code only wins if it is on PYTHONPATH.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

export MASTER_ADDR=$(hostname)

# ============================================================
#  EDIT THESE PER RUN -- must match the fine-tune script's values
#  NSIG must match a folder convert_lhco.py already produced
#  (LHCO/nsig_<NSIG>/lhco_ad/...) -- "all" means --nsig was left unset.
# ============================================================
SAVE_TAG_BASE=fine_tune_pretrain_s
DATASET=lhco_ad
SIZE=small
NSIG=2000
DESCRIPT_TAG=r1
DATASET_TYPE=${DATASET_TYPE:-test}
QUANTIZATION=none                          # "none", "int8", "int8dq", or "bf16"
# ============================================================

case "$QUANTIZATION" in
    none|int8|int8dq|bf16) ;;
    *)
        echo "ERROR: QUANTIZATION must be 'none', 'int8', 'int8dq', or 'bf16', got '$QUANTIZATION'" >&2
        exit 1
        ;;
esac

SAVE_TAG="${SAVE_TAG_BASE}_${DATASET}_nsig${NSIG}_${DESCRIPT_TAG}"
CHECKPOINT_DIR=/pscratch/sd/m/mbenyas/LHCO
LHCO_PATH="/global/cfs/cdirs/m3246/mbenyas/OmniLearned_distillation/LHCO/nsig_${NSIG}"
# LHCO_PATH="/global/cfs/cdirs/m3246/mbenyas/OmniLearned_distillation/LHCO/pure_test"
OUTPUT_DIR=/pscratch/sd/m/mbenyas/${SAVE_TAG}_${QUANTIZATION}

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
    --dataset-type $DATASET_TYPE"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    which omnilearned
    python -c 'import omnilearned; print(omnilearned.__file__)'
    $cmd
    "
