#!/bin/bash
#SBATCH --job-name=imagenet
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=17       # num_workers=16 JPEG-decoding loader processes + main
#SBATCH --tmp=200G               # local disk for the data (~150 GB extracted). Not reserved:
                                 #   the script checks free space first
#SBATCH --constraint=localdisk   # a node with a local disk (diskless nodes' /tmp is RAM)
#SBATCH --error=./outs/imagenet_%A/e.err
#SBATCH --output=./outs/imagenet_%A/o.out

# ImageNet pretraining (pretrain.py): supervised (torchvision recipe) or MoCo v2,
# from the node's local disk, as in scicomp.aalto.fi/triton/usage/localstorage:
# the data goes to /tmp/$SLURM_JOB_ID at job start, training reads only that,
# and the directory is removed when the job ends (also on cancel or timeout).
#
# Data, one of:
#   SOURCE_TAR  torchvision's ImageFolder layout in one tar (Triton's shared
#               copy): extracted straight from the shared drive to /tmp -- one
#               sequential read, no conversion, ~10-30 min per job
#   SHARDS      shards from src/data/prepare_imagenet.py (optionally
#               --short-side 256 for cheaper decoding): copied to /tmp, minutes
# Once, before the first run: check that every image decodes with the
# decoder training uses (the folder path reads images unchecked, so a bad one
# would fail mid-run). The converter decodes all of them; its output is
# thrown away, and its train/val counts (1,281,167 / 50,000) confirm the
# tar's layout. In a CPU job (--cpus-per-task=16 --mem=32G --tmp=200G
# --constraint=localdisk --time=4:00:00):
#   python -m src.data.prepare_imagenet --source "$SOURCE_TAR" \
#       --out /tmp/$SLURM_JOB_ID/check --workers 16; rm -rf /tmp/$SLURM_JOB_ID
# Resubmitting with the same RUN_DIR resumes where the job stopped. MoCo v2 (200
# epochs, two encoder passes per image) runs several days on one GPU: chain jobs,
#   jid=$(sbatch --parsable run_imagenet.sh); sbatch --dependency=afterany:$jid run_imagenet.sh

module load mamba
source activate pvr_env/

METHOD=imagenet_supervised            # or moco_v2
ARCH=resnet18
RUN_DIR="$WRKDIR/pretrain_runs/${METHOD}_${ARCH}_s1"
SOURCE_TAR=/scratch/shareddata/dldata/imagenet_imagefolder/ILSVRC2012_imagefolder.tar
SHARDS=""                             # e.g. "$WRKDIR/data/imagenet" (then leave SOURCE_TAR empty)

LOCAL="/tmp/$SLURM_JOB_ID"            # /tmp is per user, not per job
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
trap 'exit 143' TERM                  # cancel/timeout: stop here (EXIT then cleans up)
free_kb=$(df -k --output=avail /tmp | tail -1)
if [ -n "$SOURCE_TAR" ]; then
    need_kb=$(du -k --apparent-size "$SOURCE_TAR" | cut -f1)
    [ "$need_kb" -lt "$free_kb" ] || { echo "only $((free_kb / 1048576)) GB free in /tmp"; exit 1; }
    echo "extracting $SOURCE_TAR -> $LOCAL"; time tar xf "$SOURCE_TAR" -C "$LOCAL"
    # train/ and val/ may sit under a top folder inside the tar
    TRAIN=$(find "$LOCAL" -maxdepth 3 -type d -name train -print -quit)
    [ -n "$TRAIN" ] && [ -d "$(dirname "$TRAIN")/val" ] || { echo "no train/ and val/ found under $LOCAL"; exit 1; }
    DATA=$(dirname "$TRAIN")
else
    need_kb=$(du -sk "$SHARDS" | cut -f1)
    [ "$need_kb" -lt "$free_kb" ] || { echo "only $((free_kb / 1048576)) GB free in /tmp"; exit 1; }
    echo "copying $SHARDS -> $LOCAL"; time cp -r "$SHARDS" "$LOCAL/imagenet"
    DATA="$LOCAL/imagenet"
fi

srun python3 pretrain.py \
        +experiment=$METHOD \
        embedding=$ARCH \
        pretrain_data.root="$DATA" \
        num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
        perf.amp_bf16=true \
        wandb.enabled=true \
        wandb.project=pvr-pretrain \
        hydra.run.dir="$RUN_DIR"
