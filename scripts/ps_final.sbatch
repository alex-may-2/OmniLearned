#!/bin/bash
# Single-SLR finalist chain (see synthesis/io_parallel_report.md), one node:
#   1. float KD, top_deepsets_distillnet_fpga recipe, EPOCH (default 50) epochs   -> $FLOAT_TAG
#   2. 8-bit full-quant QAT (qat_deepsets.py defaults = r7 recipe), 15 ep at 5e-5
#   3. QONNX export -> $BASE/qonnx/fpga/$QAT_TAG_clean.onnx (scp -3 to rdsrv; convert.py gives qonnx + HLS AUC)
# Float KD resumes and finished stages are skipped on a resubmit (QAT restarts from the float checkpoint).
#
#   sbatch --export=ALL,SIZE=d8p2r1,N=16 scripts/ps_final.sbatch
#
# Runs this clone's code (PYTHONPATH). Writes only under /pscratch/sd/a/alexmay/omnilearned/.
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH -J ps_final
#SBATCH -o /pscratch/sd/a/alexmay/omnilearned/logs/ps/slurm_%j_%x.out

set -uo pipefail
: "${SIZE:?SIZE must be set via --export}"
: "${N:?N must be set via --export}"
EPOCH=${EPOCH:-50}
FLOAT_TAG=${FLOAT_TAG:-ps_${SIZE}_n${N}_e${EPOCH}}
QAT_TAG=${QAT_TAG:-qat_${FLOAT_TAG}_8bit_fullQuant}
REPO=/global/homes/a/alexmay/working/OmniLearned
BASE=/pscratch/sd/a/alexmay/omnilearned
CKPT=$BASE/checkpoints/
DATA=/global/cfs/cdirs/m4567/www/
TEACHER=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l
mkdir -p "$BASE/qonnx/fpga" "$BASE/logs/ps"

export PATH=/global/cfs/cdirs/m2616/alexm/conda/envs/omnilearned-fpga/bin:$PATH
export PYTHONPATH=$REPO/src
export HDF5_USE_FILE_LOCKING=FALSE NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$REPO"
SHAPE="--size $SIZE --act-layer relu --deepsets-fixed-n $N"
KD="--distill-alpha 0.5 --distill-beta 0.5 --distill-t 4"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$BASE/logs/ps/final_${FLOAT_TAG}.log"; }

log "job $SLURM_JOB_ID size=$SIZE n=$N float=$FLOAT_TAG qat=$QAT_TAG"
if [ ! -f "$CKPT/training_${FLOAT_TAG}.json" ]; then
    log "[1/3] float KD $EPOCH epochs"
    srun -l -u bash -c "source $REPO/scripts/export_ddp.sh; omnilearned train -o $CKPT --save-tag $FLOAT_TAG \
        --dataset top --mode classifier --num-classes 2 --path $DATA --arch deep-sets $SHAPE --batch 128 \
        --iterations 1000 --epoch $EPOCH --lr 5e-4 --wd 0.5 --num-workers 4 --distill --teacher-labels-dir $TEACHER \
        --teacher-tag fine_tune_top_l $KD --resuming" || { log "float KD failed"; exit 1; }
fi

if [ ! -f "$BASE/logs/ps/${QAT_TAG}.done" ]; then
    log "[2/3] full-quant QAT"
    srun -l -u bash -c "source $REPO/scripts/export_ddp.sh; python tools/quantize/qat_deepsets.py --tag $FLOAT_TAG \
        --save-tag $QAT_TAG --init-dir $CKPT --output-dir $CKPT --epochs 15 --warmup-epoch 1 \
        --lr 5e-5 --wd 0.5 --batch 128 --iterations 1000 --num-workers 4 --teacher-dir $TEACHER \
        --teacher-tag fine_tune_top_l $KD" || { log "QAT failed"; exit 1; }
    touch "$BASE/logs/ps/${QAT_TAG}.done"
fi

log "[3/3] QONNX export"
python tools/quantize/qat_deepsets_export_qonnx.py --tag "$QAT_TAG" --checkpoint-dir "$CKPT" \
    --out-dir "$BASE/qonnx/fpga/" || { log "export failed"; exit 1; }
log "done: $BASE/qonnx/fpga/${QAT_TAG}_clean.onnx"
