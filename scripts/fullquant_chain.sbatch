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
#   QAT_ARGS  extra qat_deepsets.py args, appended so they override the defaults (e.g. "--epochs 30")
#   SRUN_OPTS step placement, e.g. "-N1 -n4 -c 32 --exact" for one chain per node of a multi-node allocation
#             (scripts/fullquant_multi.sh); each step then rendezvouses on its own node
#   EVAL=1    stage 4: qat_deepsets_eval.py on the full test set -> $LOGS/${QAT_TAG}_eval.txt
#   NO_KD=1   float stage trains from scratch on truth labels (no --distill); QAT still distills
#   SEED=N    --seed N for the float stage (default unseeded)
#   SCORES=1  per-jet test scores for ROC plots in $BASE/eval/ps/: float eval after stage 1, and
#             qat_deepsets_eval.py --save-scores in stage 4 (needs EVAL=1)
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
SRUN_OPTS=${SRUN_OPTS:-}
DDP="source $REPO/scripts/export_ddp.sh"
# export_ddp.sh points every step at the allocation's first node; one chain per node needs its own rendezvous.
[ -n "$SRUN_OPTS" ] && DDP+='; export MASTER_ADDR=$(hostname) MASTER_PORT=$((29500 + SLURM_JOB_ID % 5000 + SLURM_STEP_ID % 1000))'
FLOAT_KD="--distill --teacher-labels-dir $TEACHER --teacher-tag fine_tune_top_l $KD"
[ -n "${NO_KD:-}" ] && FLOAT_KD=""
[ -n "${SEED:-}" ] && FLOAT_KD+=" --seed $SEED"
MODEL="--arch deep-sets --size $SIZE --act-layer relu --deepsets-fixed-n $N --num-classes 2"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/${FLOAT_TAG}.log"; }

log "job $SLURM_JOB_ID size=$SIZE n=$N float=$FLOAT_TAG qat=$QAT_TAG repo=$REPO base=$BASE"
if [ ! -f "$CKPT/training_${FLOAT_TAG}.json" ]; then
    log "[1/3] float ${NO_KD:+from-scratch }$EPOCH epochs: $FLOAT_KD"
    srun -l -u $SRUN_OPTS bash -c "$DDP; omnilearned train -o $CKPT --save-tag $FLOAT_TAG \
        --dataset top --mode classifier --path $DATA $MODEL --batch 128 \
        --iterations 1000 --epoch $EPOCH --lr 5e-4 --wd 0.5 --num-workers 4 $FLOAT_KD --resuming" \
        || { log "float KD failed"; exit 1; }
fi
if [ -n "${SCORES:-}" ] && ! ls "$BASE/eval/ps/outputs_${FLOAT_TAG}_top_test_rank"*.npz >/dev/null 2>&1; then
    log "float test scores -> $BASE/eval/ps/"
    srun -u $SRUN_OPTS -n 1 -c 32 bash -c "$DDP; omnilearned evaluate -i $CKPT -o $BASE/eval/ps/ --save-tag $FLOAT_TAG \
        --dataset top --path $DATA $MODEL --batch 512 --num-workers 8 --dataset-type test" || log "float eval failed"
fi

if [ ! -f "$LOGS/${QAT_TAG}.done" ]; then
    log "[2/3] full-quant QAT"
    srun -l -u $SRUN_OPTS bash -c "$DDP; python tools/quantize/qat_deepsets.py --tag $FLOAT_TAG \
        --save-tag $QAT_TAG --init-dir $CKPT --output-dir $CKPT --epochs 15 --warmup-epoch 1 \
        --lr 5e-5 --wd 0.5 --batch 128 --iterations 1000 --num-workers 4 --teacher-dir $TEACHER \
        --teacher-tag fine_tune_top_l $KD --data-path $DATA ${QAT_ARGS:-}" || { log "QAT failed"; exit 1; }
    touch "$LOGS/${QAT_TAG}.done"
fi

log "[3/3] QONNX export"
python tools/quantize/qat_deepsets_export_qonnx.py --tag "$QAT_TAG" --checkpoint-dir "$CKPT" \
    --out-dir "$BASE/qonnx/fpga/" || { log "export failed"; exit 1; }
log "done: $BASE/qonnx/fpga/${QAT_TAG}_clean.onnx"

if [ -n "${EVAL:-}" ]; then
    log "[4/4] test-set eval"
    srun -u $SRUN_OPTS -n 1 -c 32 bash -c "$DDP; python tools/quantize/qat_deepsets_eval.py --tag $QAT_TAG \
        --checkpoint-dir $CKPT --data-path $DATA ${SCORES:+--save-scores $BASE/eval/ps/}" > "$LOGS/${QAT_TAG}_eval.txt" 2>&1 || log "eval failed"
    log "eval $QAT_TAG: $(grep -E 'Accuracy|AUC|1/FPR' "$LOGS/${QAT_TAG}_eval.txt" | tr -s ' ' | paste -sd ';')"
fi
