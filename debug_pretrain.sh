#!/bin/bash
#SBATCH --job-name=r3m_debug
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=12
#SBATCH --tmp=200G
#SBATCH --constraint=localdisk
#SBATCH --error=./out_debug/r3m_%A/e.err
#SBATCH --output=./out_debug/r3m_%A/o.out

# Smoke test of run_pretrain.sh: the same steps, a few hundred training steps.
# Checks the shard copy to /tmp, R3M with ResNet-50 and the language loss
# (DistilBERT downloads on first use), validation, checkpointing, and resume
# (a second run continues the first). Needs the shards from run_prepare_ssv2.sh.
# In o.out look for: the copy time, "Encoder: resnet50", the steps/s and
# data_wait of the log lines, EVAL lines, "Resumed from ... at step 100".
# COPY_TO_LOCAL=0 sbatch debug_pretrain.sh reads the shards from $WRKDIR instead.

set -euo pipefail
module load mamba
source activate pvr_env/

export HF_HOME="$WRKDIR/hf"
DATA="$WRKDIR/data/ssv2"
RUN_DIR="$WRKDIR/pretrain_runs/debug_r3m_$SLURM_JOB_ID"     # delete afterwards
[ -f "$DATA/manifest_train.csv" ] || { echo "no shards at $DATA (run run_prepare_ssv2.sh first)"; exit 1; }

LOCAL="/tmp/$SLURM_JOB_ID"
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
trap 'exit 143' TERM
if [ "${COPY_TO_LOCAL:-1}" = 1 ]; then
    echo "copying $DATA -> $LOCAL"; time cp -r "$DATA" "$LOCAL/ssv2"
    DATA="$LOCAL/ssv2"
fi

COMMON=(+experiment=r3m embedding=resnet50 pretrain_data.root="$DATA"
        num_workers=$((SLURM_CPUS_PER_TASK - 1)) log_frequency=20 eval_frequency=50
        save.keep_every=null wandb.enabled=false hydra.run.dir="$RUN_DIR")

echo "=== run 1: 100 steps"
srun python3 pretrain.py "${COMMON[@]}" train_steps=100
echo "=== run 2: resume to 150 steps"
srun python3 pretrain.py "${COMMON[@]}" train_steps=150
echo "=== saved:"; ls -lh "$RUN_DIR"
