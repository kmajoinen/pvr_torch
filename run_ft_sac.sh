#!/bin/bash
#SBATCH --job-name=ham_rn18ft
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --error=./outs/rn18_%A/e.err
#SBATCH --output=./outs/rn18_%A/o.out


# module --ignore-cache load mamba
module load mamba
source activate pvr_env/

export MUJOCO_GL=egl
#MODEL_DIR="$WRKDIR/models/"

srun python3 train_sac.py \
        env=adroit_hammer_pixels \
        embedding=resnet18 \
        env.image_size=84 \
        save.enabled=false \
        algo.total_timesteps=2000000 \
        wandb.enabled=true \
        wandb.project=pvr-sac \
        encoder.grads=critic
