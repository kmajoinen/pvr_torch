#!/bin/bash
#SBATCH --job-name=r3m_vit
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=17       # 16 loader workers (JPEG decoding: 64 videos x 5 frames per step) + main
#SBATCH --tmp=200G               # local disk for the shard copy (~100-150 GB). Not reserved:
                                 #   the script checks free space first
#SBATCH --constraint=localdisk   # a node with a local disk (diskless nodes' /tmp is RAM)
#SBATCH --error=./outs/r3m_vit_%A/e.err
#SBATCH --output=./outs/r3m_vit_%A/o.out

# R3M's objective on MVP's ViT-S/16, from scratch, on SSv2
# (configs/experiment/r3m_vit.yaml). Needs the shards from run_prepare_ssv2.sh.
# Each job copies them to the node's /tmp/$SLURM_JOB_ID and trains from there;
# the copy is removed at the end. 1M steps won't fit one job: resubmit with the
# same RUN_DIR and it resumes from checkpoint.pt, or chain jobs:
#   jid=$(sbatch --parsable run_r3m_vit.sh); sbatch --dependency=afterany:$jid run_r3m_vit.sh
# Check steps/s and data_wait in the first log lines: hours for the full run
# = 1,000,000 / steps/s / 3600. Override the tuning knobs on the srun line
# (lr=..., weight_decay=..., objectives.0.l1_weight=..., lr_schedule.warmup=...).

set -euo pipefail
module load mamba
source activate pvr_env/

export HF_HOME="$WRKDIR/hf"         # DistilBERT (R3M's text encoder) downloads here on first use
RUN_DIR="$WRKDIR/pretrain_runs/r3m_vit_s16_ssv2_s1"
DATA="$WRKDIR/data/ssv2"              # the converted shards and manifests

LOCAL="/tmp/$SLURM_JOB_ID"            # /tmp is per user, not per job
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
trap 'exit 143' TERM                  # cancel/timeout: stop here (EXIT then cleans up)
need_kb=$(du -sk "$DATA" | cut -f1); free_kb=$(df -k --output=avail /tmp | tail -1)
if [ "$need_kb" -lt "$free_kb" ]; then
    echo "copying $DATA -> $LOCAL"; time cp -r "$DATA" "$LOCAL/ssv2"
    DATA="$LOCAL/ssv2"
else
    echo "only $((free_kb / 1048576)) GB free in /tmp: reading $DATA directly"
fi

srun python3 pretrain.py \
        +experiment=r3m_vit \
        pretrain_data.root="$DATA" \
        num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
        wandb.enabled=true \
        wandb.project=pvr-pretrain \
        hydra.run.dir="$RUN_DIR"
