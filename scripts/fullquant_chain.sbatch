#!/bin/bash
# Full-quant DeepSets chain for one fixed-N student shape, on one 4-GPU node (~3 h for d12p2r1m1 N=16):
#   1. float KD training, EPOCH (default 50) epochs                          -> $CKPT/best_model_$FLOAT_TAG.pt
#   2. 8-bit full-quant QAT (qat_deepsets.py defaults), 15 epochs at 5e-5    -> $CKPT/best_model_$QAT_TAG.pt
#   3. QONNX export                                                          -> $BASE/qonnx/fpga/${QAT_TAG}_clean.onnx
# On a resubmit, float KD resumes and finished stages are skipped (QAT restarts from the float checkpoint).
#
# Run from the repo root. SIZE is a --size (e.g. d12p2r1m1 = dim 12, embed + 1 phi block, 1 rho layer, mlp_ratio 1)
# and N the fixed constituent count (--deepsets-fixed-n):
#   SIZE=d12p2r1m1 N=16 salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 \
#       -A m3246 bash scripts/fullquant_chain.sbatch
#   sbatch --export=ALL,SIZE=d12p2r1m1,N=16 scripts/fullquant_chain.sbatch
#
# Optional env vars:
#   BASE      output root: checkpoints/, qonnx/fpga/, logs/fullquant/ (default $PSCRATCH/omnilearned)
#   FPGA_ENV  conda env with Brevitas/QONNX (default the m2616 omnilearned-fpga env)
#   EPOCH, FLOAT_TAG, QAT_TAG, TEACHER, DATA
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH -J fullquant_chain
#SBATCH -o slurm_%j_%x.out

set -uo pipefail
: "${SIZE:?set SIZE, e.g. SIZE=d12p2r1m1}"
: "${N:?set N, the fixed constituent count, e.g. N=16}"
EPOCH=${EPOCH:-50}
FLOAT_TAG=${FLOAT_TAG:-deepsets_${SIZE}_n${N}_e${EPOCH}}
QAT_TAG=${QAT_TAG:-qat_${FLOAT_TAG}_8bit_fullQuant}
REPO=$PWD
BASE=${BASE:-$PSCRATCH/omnilearned}
CKPT=$BASE/checkpoints/
LOGS=$BASE/logs/fullquant
DATA=${DATA:-/global/cfs/cdirs/m4567/www/}
TEACHER=${TEACHER:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l}
FPGA_ENV=${FPGA_ENV:-/global/cfs/cdirs/m2616/alexm/conda/envs/omnilearned-fpga}
[ -f "$REPO/tools/quantize/qat_deepsets.py" ] || { echo "run from the repo root"; exit 1; }
mkdir -p "$CKPT" "$BASE/qonnx/fpga" "$LOGS"

export PATH=$FPGA_ENV/bin:$PATH
export PYTHONPATH=$REPO/src
export HDF5_USE_FILE_LOCKING=FALSE NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
KD="--distill-alpha 0.5 --distill-beta 0.5 --distill-t 4"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/${FLOAT_TAG}.log"; }

log "job $SLURM_JOB_ID size=$SIZE n=$N float=$FLOAT_TAG qat=$QAT_TAG repo=$REPO base=$BASE"
if [ ! -f "$CKPT/training_${FLOAT_TAG}.json" ]; then
    log "[1/3] float KD $EPOCH epochs"
    srun -l -u bash -c "source $REPO/scripts/export_ddp.sh; omnilearned train -o $CKPT --save-tag $FLOAT_TAG \
        --dataset top --mode classifier --num-classes 2 --path $DATA --arch deep-sets \
        --size $SIZE --act-layer relu --deepsets-fixed-n $N --batch 128 \
        --iterations 1000 --epoch $EPOCH --lr 5e-4 --wd 0.5 --num-workers 4 --distill --teacher-labels-dir $TEACHER \
        --teacher-tag fine_tune_top_l $KD --resuming" || { log "float KD failed"; exit 1; }
fi

if [ ! -f "$LOGS/${QAT_TAG}.done" ]; then
    log "[2/3] full-quant QAT"
    srun -l -u bash -c "source $REPO/scripts/export_ddp.sh; python tools/quantize/qat_deepsets.py --tag $FLOAT_TAG \
        --save-tag $QAT_TAG --init-dir $CKPT --output-dir $CKPT --epochs 15 --warmup-epoch 1 \
        --lr 5e-5 --wd 0.5 --batch 128 --iterations 1000 --num-workers 4 --teacher-dir $TEACHER \
        --teacher-tag fine_tune_top_l $KD" || { log "QAT failed"; exit 1; }
    touch "$LOGS/${QAT_TAG}.done"
fi

log "[3/3] QONNX export"
python tools/quantize/qat_deepsets_export_qonnx.py --tag "$QAT_TAG" --checkpoint-dir "$CKPT" \
    --out-dir "$BASE/qonnx/fpga/" || { log "export failed"; exit 1; }
log "done: $BASE/qonnx/fpga/${QAT_TAG}_clean.onnx"
