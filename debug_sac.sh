#!/bin/bash
#SBATCH --job-name=sac_debug
#SBATCH --account=project_2019621
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:14:30
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=32G
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
        finetune.enabled=true \
        save.enabled=false \
        algo.total_timesteps=5000 \
        algo.buffer_size=1000 \
        algo.learning_starts=200 \
        algo.eval_frequency=500 \
        algo.log_frequency=250 \
        hydra.run.dir="$OUT_ROOT_DIR/outputs_me/sac/\${env.id}/\${embedding.name}/\${now:%Y-%m-%d_%H-%M-%S}"
