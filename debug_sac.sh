#!/bin/bash
#SBATCH --job-name=sac_debug
#SBATCH --mem=32G
#SBATCH --gpus=1
# #SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h100-80g,gpu-h100-80g
#SBATCH --partition=gpu-debug
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./out_debug/deb_%A/e.err
#SBATCH --output=./out_debug/deb_%A/o.out


module load mamba
source activate pvr_env_cu126/

export MUJOCO_GL=egl


srun python3 train_sac.py \
       	env=dm_control_pixels \
	embedding=resnet18 \
        finetune.enabled=true \
        save.enabled=false \
        algo.total_timesteps=5000 \
        algo.buffer_size=1000 \
        algo.learning_starts=200 \
        algo.eval_frequency=500 \
        algo.log_frequency=250 \
