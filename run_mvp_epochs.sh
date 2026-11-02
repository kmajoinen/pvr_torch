#!/bin/bash
#SBATCH --job-name=mvp_epochs
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=24       # loader workers: 256 JPEG decodes + crops per step
#SBATCH --tmp=200G               # local disk for the shard copy. Not reserved: the script checks free space first
#SBATCH --constraint=localdisk   # a node with a local disk (diskless nodes' /tmp is RAM)
#SBATCH --error=./outs/mvp_epochs_%A/e.err
#SBATCH --output=./outs/mvp_epochs_%A/o.out

# MVP pretraining (pretrain.py +experiment=mvp): MAE, ViT-S/16, on SSv2's
# frames at 0.3 fps (every 40th frame of each video), budget:
#   MVP's 1600 epochs over our frames (same repetition per frame as MVP; ~40% of its compute, ~1.75M steps)
# Reads the SSv2 shards run_prepare_ssv2.sh wrote (the same ones R3M/VIP use).
#
# The run takes longer than one job: chain jobs. Each resumes from
# checkpoint.pt in RUN_DIR (saved at every validation, every 10 epochs), and
# continues the data stream exactly as long as --cpus-per-task (hence
# num_workers) and batch_size stay the same:
#   jid=$(sbatch --parsable run_mvp_epochs.sh)
#   for i in 1 2 3 4; do jid=$(sbatch --parsable --dependency=afterany:$jid run_mvp_epochs.sh); done
# Check the first log lines for steps/s and data_wait (a high data_wait wants
# more CPUs), and the startup line for the frame count ("... images",
# "every=40: per video") and the number of epochs/steps.

module load mamba
source activate pvr_env/

RUN_DIR="$WRKDIR/pretrain_runs/mvp_ssv2_vits16_epochs_s1"
DATA="$WRKDIR/data/ssv2"              # the converted shards and manifests
[ -f "$DATA/train/index.npz" ] || { echo "no SSv2 shards at $DATA (run run_prepare_ssv2.sh first)"; exit 1; }

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
        +experiment=mvp \
        pretrain_data.root="$DATA" \
        num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
        wandb.enabled=true \
        wandb.project=pvr-pretrain \
        wandb.run_name=mvp_ssv2_vits16_epochs_s1 \
        hydra.run.dir="$RUN_DIR"
