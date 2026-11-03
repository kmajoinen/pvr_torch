#!/bin/bash
#SBATCH --job-name=buf_ssd
#SBATCH --array=0-1
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --tmp=40G
#SBATCH --constraint=localdisk
#SBATCH --error=./outs/buf_ssd_%A_%a/e.err
#SBATCH --output=./outs/buf_ssd_%A_%a/o.out

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl
TYPES=(none zstd)
TYPE=${TYPES[$SLURM_ARRAY_TASK_ID]}

BUF=/tmp/$SLURM_JOB_ID
mkdir -p $BUF
trap 'rm -rf $BUF' EXIT
findmnt -T $BUF -o TARGET,SOURCE,FSTYPE

srun python3 train_sac.py \
        env=adroit_hammer_pixels \
        env.image_size=224 \
        embedding=resnet18 \
        encoder.grads=critic \
        algo.buffer_size=100000 \
        algo.total_timesteps=150000 \
        algo.buffer_storage=disk \
        algo.buffer_dir=$BUF \
        algo.buffer_compression=$TYPE \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-buffer \
        wandb.run_name=buf-ssd-$TYPE
