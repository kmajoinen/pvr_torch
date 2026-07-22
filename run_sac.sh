#!/bin/bash
#SBATCH --job-name=ft_rn18
#SBATCH --account=project_2019621
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/scratch/project_2019621/pvr/xft_logs/outs/%j/o.out
#SBATCH --error=/scratch/project_2019621/pvr/xft_logs/outs/%j/e.err

module load python-pytorch
source pvr_env/bin/activate

export MUJOCO_GL=egl
export WANDB_DIR=/scratch/project_2019621/pvr/xft_logs/wandb_me
OUT_ROOT_DIR="/scratch/project_2019621/pvr/xft_logs"

srun python3 train_sac.py \
        env=dm_control_pixels \
        embedding=resnet18 \
        save.enabled=false \
        algo.total_timesteps=200000 \
        wandb.enabled=true \
        wandb.project=pvr-sac \
        finetune.enabled=true \
        hydra.run.dir="$OUT_ROOT_DIR/outputs_me/sac/\${env.id}/\${embedding.name}/\${now:%Y-%m-%d_%H-%M-%S}"
