#!/bin/bash
#SBATCH --job-name=debug_ddp
#SBATCH --mem=128G
#SBATCH --gpus=2
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --tmp=200G
#SBATCH --constraint=localdisk
#SBATCH --error=./out_debug/ddp_%A/e.err
#SBATCH --output=./out_debug/ddp_%A/o.out

module load mamba
source activate pvr_env/

METHOD=imagenet_supervised
ARCH=resnet50
NGPU=2
SOURCE_TAR=/scratch/shareddata/dldata/imagenet_imagefolder/ILSVRC2012_imagefolder.tar
OUT=$WRKDIR/pretrain_runs/debug_ddp_${METHOD}_${ARCH}_$SLURM_JOB_ID

LOCAL="/tmp/$SLURM_JOB_ID"
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
trap 'exit 143' TERM
time tar xf "$SOURCE_TAR" -C "$LOCAL"
TRAIN=$(find "$LOCAL" -maxdepth 3 -type d -name train -print -quit)
DATA=$(dirname "$TRAIN")

COMMON=(+experiment=$METHOD
        embedding=$ARCH
        pretrain_data.root="$DATA"
        epochs=null
        train_steps=1000
        lr_schedule.name=constant
        eval_every_epochs=null
        eval_frequency=1000
        save.keep_every_epochs=null
        pretrain_data.eval_batches=20
        log_frequency=100
        num_workers=15
        perf.amp_bf16=true)

srun python3 pretrain.py "${COMMON[@]}" hydra.run.dir="$OUT/gpus1"

srun python3 -m torch.distributed.run --standalone --nproc_per_node=$NGPU pretrain.py \
        "${COMMON[@]}" \
        hydra.run.dir="$OUT/gpus$NGPU"
