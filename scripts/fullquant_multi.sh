#!/bin/bash
# Several fullquant_chain.sbatch chains side by side in one multi-node allocation, one chain per node.
# SPEC has one chain per line as env assignments (blank lines and # comments are skipped), e.g.
#   SIZE=d14p2r1m1 N=16 FLOAT_TAG=ps_d14p2r1m1_n16_e50 EVAL=1
#   SIZE=d12p2r1m1 N=16 FLOAT_TAG=ps_d12p2r1m1_n16_e50 QAT_TAG=qat_ps_d12p2r1m1_n16_e50_q30 QAT_ARGS="--epochs 30" EVAL=1
# At most $SLURM_NNODES chains run at once; the next line starts when one finishes, so fewer granted nodes just
# means more waiting. From the repo root:
#   setsid nohup salloc -C gpu -q interactive -t 240 --nodes 1-4 --ntasks-per-node 4 --gpus-per-node 4 -A m2616 \
#       bash scripts/fullquant_multi.sh <spec> > <log> 2>&1 < /dev/null &
set -u
mapfile -t LINES < "${1:?usage: fullquant_multi.sh SPEC}"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID: $SLURM_NNODES node(s)"
running=0
for line in "${LINES[@]}"; do
    [[ -z "${line// /}" || "$line" == \#* ]] && continue
    if ((running >= SLURM_NNODES)); then wait -n; running=$((running - 1)); fi
    echo "[$(date '+%F %T')] start: $line"
    (eval "export $line"; SRUN_OPTS="-N1 -n4 -c 32 --exact" bash scripts/fullquant_chain.sbatch < /dev/null) &
    running=$((running + 1))
done
wait
echo "[$(date '+%F %T')] all chains done"
