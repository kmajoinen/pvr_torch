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

srun python3 train_sac.py --mode pixels --env dmc_cheetah --debug --no-save --total-timesteps 5000
