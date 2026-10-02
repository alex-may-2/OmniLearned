#!/bin/bash
# 8-bit full-quant QAT fine-tune of the hls4ml-friendly plain distillnet student
# (fixed-N=64, ReLU, no mask). Warm-started from the float KD checkpoint
# distill_top_deepsets_distillnet_fpga_a05_T4, with the same distillation recipe
# (alpha=0.5/beta=0.5/T=4 vs fine_tune_top_l) the float run used. qat_deepsets.py
# is always full-quant (see its docstring); its defaults are the recipe of the
# r7 reference graph. That float checkpoint predates arch_config, so --size /
# --act-layer / --deepsets-fixed-n are passed to rebuild its shape.
#
# CRITICAL: uses omnilearned-fpga/env (has Brevitas), NOT omnilearned-clean/env.
#
# ~2 hrs for 1000 it x 15 ep; run as a single one-shot salloc (qat_deepsets.py
# always warm-starts from the float --tag, so a preempted rerun just restarts
# the 15-epoch fine-tune -- no corruption):
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/qat_train_deepsets_distillnet_fpga_8bit.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-fpga/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="/global/homes/t/twamorka/omnilearned-fpga/env/bin/python tools/quantize/qat_deepsets.py \
  --tag distill_top_deepsets_distillnet_fpga_a05_T4 \
  --size distillnet --bits 8 \
  --act-layer relu --deepsets-fixed-n 64 \
  --save-tag qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant \
  --epochs 15 --warmup-epoch 1 --lr 5e-5 --wd 0.5 \
  --batch 128 --iterations 1000 --num-workers 4 \
  --teacher-dir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4.0"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
