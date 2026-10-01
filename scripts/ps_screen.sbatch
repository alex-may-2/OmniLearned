#!/bin/bash
# parallel-search float KD screen (synthesis/PARALLEL_SEARCH_PLAN.md, Phase 2): the top_deepsets_distillnet_fpga
# KD recipe (teacher fine_tune_top_l, a0.5/b0.5/T4, lr 5e-4, wd 0.5) at another size / particle count, on one node,
# then a test-set evaluate and one AUC line in $BASE/logs/ps/screens.txt.
#
#   sbatch --export=ALL,SIZE=d8p2r1,N=16 scripts/ps_screen.sbatch            # EPOCH defaults to 20
#   sbatch --export=ALL,SIZE=d32p2r1,N=64 scripts/ps_screen.sbatch           # short-schedule control
#
# Runs this clone's code (PYTHONPATH), not the env's editable install. Writes only under
# /pscratch/sd/a/alexmay/omnilearned/{checkpoints,eval/ps,logs/ps}; the teacher logits are only read.
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH -J ps_screen
#SBATCH -o /pscratch/sd/a/alexmay/omnilearned/logs/ps/slurm_%j_%x.out

set -uo pipefail
: "${SIZE:?SIZE must be set via --export}"
: "${N:?N must be set via --export}"
EPOCH=${EPOCH:-20}
SAVE_TAG=${SAVE_TAG:-ps_${SIZE}_n${N}_e${EPOCH}}
REPO=/pscratch/sd/a/alexmay/OmniLearned
BASE=/pscratch/sd/a/alexmay/omnilearned
DATA=/global/cfs/cdirs/m4567/www/
TEACHER=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l
mkdir -p "$BASE/eval/ps" "$BASE/logs/ps"

export PATH=/global/cfs/cdirs/m2616/alexm/conda/envs/omnilearned-fpga/bin:$PATH
export PYTHONPATH=$REPO/src
export HDF5_USE_FILE_LOCKING=FALSE NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
MODEL="--arch deep-sets --size $SIZE --act-layer relu --deepsets-fixed-n $N --num-classes 2"

echo "[$(date '+%F %T')] job $SLURM_JOB_ID tag=$SAVE_TAG size=$SIZE n=$N epochs=$EPOCH"
srun -l -u bash -c "source $REPO/scripts/export_ddp.sh; omnilearned train -o $BASE/checkpoints/ --save-tag $SAVE_TAG \
    --dataset top --mode classifier --path $DATA $MODEL --batch 128 --iterations 1000 --epoch $EPOCH --lr 5e-4 \
    --wd 0.5 --num-workers 4 --distill --teacher-labels-dir $TEACHER --teacher-tag fine_tune_top_l \
    --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 --resuming" || exit 1

srun -l -u bash -c "source $REPO/scripts/export_ddp.sh; omnilearned evaluate -i $BASE/checkpoints/ -o $BASE/eval/ps/ \
    --save-tag $SAVE_TAG --dataset top --path $DATA $MODEL --batch 128 --num-workers 4 --dataset-type test" || exit 1

python - "$BASE/eval/ps" "$SAVE_TAG" <<'EOF' | tee -a "$BASE/logs/ps/screens.txt"
import glob, sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

paths = sorted(glob.glob(f"{sys.argv[1]}/outputs_{sys.argv[2]}_top_test_rank*.npz"))
pred = np.concatenate([np.load(p)["prediction"].astype(np.float64) for p in paths])
y = np.concatenate([np.load(p)["pid"] for p in paths])
fpr, tpr, _ = roc_curve(y, pred[:, 1])
print(f"{sys.argv[2]} n_jets={len(y)} acc={np.mean(pred.argmax(1) == y):.4f} AUC={roc_auc_score(y, pred[:, 1]):.4f} "
      f"1/eB@0.5={1 / np.interp(0.5, tpr, fpr):.1f}")
EOF
