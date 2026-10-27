#!/bin/bash
#SBATCH --job-name=r3m_ssv2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=12       # loader workers (JPEG decoding) + main
#SBATCH --tmp=200G               # local disk for the shard copy (~100-150 GB). Not reserved:
                                 #   the script checks free space first
#SBATCH --constraint=localdisk   # a node with a local disk (diskless nodes' /tmp is RAM)
#SBATCH --error=./outs/pretrain_%A/e.err
#SBATCH --output=./outs/pretrain_%A/o.out

# R3M on Something-Something v2 (pretrain.py, configs/experiment/r3m.yaml).
# Before the first run:
#   - convert Triton's SSv2 archive once into frame shards (a few hundred files
#     instead of millions, within the file quota): sbatch run_prepare_ssv2.sh
# Each job copies the shards to the node's /tmp/$SLURM_JOB_ID and trains from
# there (scicomp.aalto.fi/triton/usage/localstorage); the copy is removed at
# the end. A job that hits the time limit resumes from checkpoint.pt when
# resubmitted with the same RUN_DIR:
#   jid=$(sbatch --parsable run_pretrain.sh); sbatch --dependency=afterany:$jid run_pretrain.sh

module load mamba
source activate pvr_env/

export HF_HOME="$WRKDIR/hf"         # DistilBERT (R3M's text encoder) downloads here on first use
RUN_DIR="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1"
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
        +experiment=r3m \
        embedding=resnet50 \
        pretrain_data.root="$DATA" \
        num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
        wandb.enabled=true \
        wandb.project=pvr-pretrain \
        hydra.run.dir="$RUN_DIR"
