#!/bin/bash
#SBATCH --job-name=xqc_ft_cheetah
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/xqc_ft_cheetah_%j/e.err
#SBATCH --output=./outs/xqc_ft_cheetah_%j/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl

SEED=${SEED:-1}

srun python3 train_xqc.py \
        env=dm_control_pixels \
        env.image_size=84 \
        embedding=resnet18 \
        encoder.grads=critic \
        algo.total_timesteps=200000 \
        algo.buffer_size=200000 \
        save.enabled=false \
        seed=$SEED \
        wandb.enabled=true \
        wandb.project=pvr-sac \
        "$@"
