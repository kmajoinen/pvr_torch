#!/bin/bash
#SBATCH --job-name=ham_rn50
#SBATCH --mem=32G
#SBATCH --gpus=1
# #SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-grace-h200-141g,gpu-h100-80g
#SBATCH --partition=gpu-a100-80g
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/rn18_%A/e.err
#SBATCH --output=./outs/rn18_%A/o.out


module load mamba
source activate pvr_env/

export MUJOCO_GL=egl
MODEL_DIR="$WRKDIR/models/"

srun python3 train_sac.py \
        env=adroit_hammer_pixels \
        embedding=resnet18 \
        save.enabled=false \
        algo.total_timesteps=200000 \
        wandb.enabled=true \
        wandb.project=pvr-sac
