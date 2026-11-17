#!/bin/bash
#SBATCH --job-name=xqc_ft_cheetah
#SBATCH --account=project_2019621
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=72
#SBATCH --error=./outs/xqc_ft_cheetah_%j/e.err
#SBATCH --output=./outs/xqc_ft_cheetah_%j/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load python-pytorch
source "${PROJAPPL_DIR:-/projappl/project_2019621/pvr/pvr_torch}/pvr_env/bin/activate"
export WRKDIR="${SCRATCH_DIR:-/scratch/project_2019621/pvr}"

export MUJOCO_GL=egl

SEED=${SEED:-1}

srun python3 train_xqc.py \
        env=dm_control_pixels \
        env.image_size=84 \
        embedding=resnet18 \
        encoder.grads=critic \
        algo.action_repeat=4 \
        algo.gamma=0.99 \
        algo.total_timesteps=200000 \
        algo.buffer_size=200000 \
        save.enabled=false \
        seed=$SEED \
        wandb.enabled=true \
        wandb.project=pvr-sac \
        "$@"
