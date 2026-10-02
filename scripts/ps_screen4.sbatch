#!/bin/bash
# parallel-search quick knob screen: up to 4 float KD runs side by side on one node, one GPU each (same KD recipe as
# ps_screen.sbatch), then a test-set AUC line each in $BASE/logs/ps/screens.txt. Fits the 30 min debug QOS at ~9
# epochs. One GPU per run sees a quarter of the jets per epoch of a 4-GPU run: compare only runs from this script.
#
#   sbatch --export=ALL,SCREENS="d8p2r1:16 d16p1r1:16 d12p2r1m1:16 d8p1r1:32",EPOCH=9 scripts/ps_screen4.sbatch
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J ps_screen4
#SBATCH -o /pscratch/sd/a/alexmay/omnilearned/logs/ps/slurm_%j_%x.out

set -uo pipefail
: "${SCREENS:?SCREENS must be set via --export}"
EPOCH=${EPOCH:-9}
REPO=/global/homes/a/alexmay/working/OmniLearned
BASE=/pscratch/sd/a/alexmay/omnilearned
DATA=/global/cfs/cdirs/m4567/www/
TEACHER=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l
mkdir -p "$BASE/eval/ps" "$BASE/logs/ps"
export PATH=/global/cfs/cdirs/m2616/alexm/conda/envs/omnilearned-fpga/bin:$PATH
export PYTHONPATH=$REPO/src
export HDF5_USE_FILE_LOCKING=FALSE
run1() {  # one single-GPU job step (the job exports SLURM_GPUS_PER_NODE=4, which a step would inherit); $PORT: own
    # rendezvous port per run, since export_ddp.sh derives one port from the job id
    env -u SLURM_GPUS_PER_NODE -u SLURM_TRES_PER_NODE srun -n 1 -c 32 --gpus=1 --mem=55G --exact \
        bash -c "source $REPO/scripts/export_ddp.sh; export MASTER_PORT=$PORT; $*"
}

i=0
for s in $SCREENS; do
    i=$((i + 1)) PORT=$((29500 + SLURM_JOB_ID % 10000 + 100 * i))
    SIZE=${s%:*} N=${s#*:} TAG=ps1g_${s%:*}_n${s#*:}_e$EPOCH
    MODEL="--arch deep-sets --size $SIZE --act-layer relu --deepsets-fixed-n $N --num-classes 2"
    (run1 omnilearned train -o $BASE/checkpoints/ --save-tag $TAG --dataset top --mode classifier --path $DATA $MODEL \
        --batch 128 --iterations 1000 --epoch $EPOCH --lr 5e-4 --wd 0.5 --num-workers 8 --distill \
        --teacher-labels-dir $TEACHER --teacher-tag fine_tune_top_l --distill-alpha 0.5 --distill-beta 0.5 \
        --distill-t 4 --resuming \
     && run1 omnilearned evaluate -i $BASE/checkpoints/ -o $BASE/eval/ps/ --save-tag $TAG --dataset top --path $DATA \
        $MODEL --batch 512 --num-workers 8 --dataset-type test \
     && python - "$BASE/eval/ps" "$TAG" <<'EOF' | tee -a "$BASE/logs/ps/screens.txt"
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
    ) > "$BASE/logs/ps/${TAG}.out" 2>&1 &
done
wait
