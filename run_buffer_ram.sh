#!/bin/bash
#SBATCH --job-name=buf_ram
#SBATCH --array=0-2
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/buf_ram_%A_%a/e.err
#SBATCH --output=./outs/buf_ram_%A_%a/o.out

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl
TYPES=(none zstd lz4)
TYPE=${TYPES[$SLURM_ARRAY_TASK_ID]}

srun python3 train_sac.py \
        env=adroit_hammer_pixels \
        env.image_size=224 \
        embedding=resnet18 \
        encoder.grads=critic \
        algo.buffer_size=100000 \
        algo.total_timesteps=150000 \
        algo.buffer_storage=ram \
        algo.buffer_compression=$TYPE \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-buffer \
        wandb.run_name=buf-ram-$TYPE
