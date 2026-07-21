#!/bin/bash
#SBATCH --job-name=bc_test
#SBATCH --account=project_2019621
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:14:30
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=32G
#SBATCH --output=outs/%j/o.out
#SBATCH --error=outs/%j/e.err

module load python-pytorch
source pvr_env/bin/activate

export MUJOCO_GL=egl

srun python3 train_sac_sb3.py mode=pvr env=dmc_cheetah embedding=resnet18 no_save=true \
    algo.total_timesteps=5000 algo.buffer_size=1000 algo.eval_freq=500 algo.checkpoint_freq=500
