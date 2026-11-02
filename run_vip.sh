#!/bin/bash
#SBATCH --job-name=vip_ssv2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=12       # loader workers (JPEG decoding) + main
#SBATCH --tmp=200G               # local disk for the shard copy. Not reserved: the script checks free space first
#SBATCH --constraint=localdisk   # a node with a local disk (diskless nodes' /tmp is RAM)
#SBATCH --error=./outs/vip_%A/e.err
#SBATCH --output=./outs/vip_%A/o.out

# VIP on Something-Something v2 (pretrain.py +experiment=vip,
# configs/experiment/vip.yaml): a ResNet-50 from scratch with the released
# VIP model's settings -- batch 32 videos x 4 frames, Adam 1e-4, 2M steps,
# gamma 0.98, L1/L2 weights 1e-3, 3 negatives, a 1024-wide fc head
# (embedding.head=fc). No text needed. Reads the same SSv2 shards as R3M
# (run_prepare_ssv2.sh).
#
# 2M steps may not fit in one job: chain jobs, each resuming from
# checkpoint.pt in RUN_DIR (saved at every validation, every 20k steps):
#   jid=$(sbatch --parsable run_vip.sh); sbatch --dependency=afterany:$jid run_vip.sh
# Read steps/s in the first log lines: 2M / steps/s / 3600 = hours.
# Smaller encoder: ARCH=resnet18 sbatch run_vip.sh (its own run directory).

module load mamba
source activate pvr_env/

ARCH="${ARCH:-resnet50}"
RUN_DIR="$WRKDIR/pretrain_runs/vip_ssv2_${ARCH}_s1"
DATA="$WRKDIR/data/ssv2"              # the converted shards and manifests
[ -f "$DATA/manifest_train.csv" ] || { echo "no SSv2 shards at $DATA (run run_prepare_ssv2.sh first)"; exit 1; }

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
        +experiment=vip \
        embedding=$ARCH \
        pretrain_data.root="$DATA" \
        num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
        wandb.enabled=true \
        wandb.project=pvr-pretrain \
        wandb.run_name=vip_ssv2_${ARCH}_s1 \
        hydra.run.dir="$RUN_DIR"
