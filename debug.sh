#!/bin/bash
#SBATCH --job-name=bc_test
#SBATCH --account=project_2019621
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:14:30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --output=outs/%j/o.out
#SBATCH --error=outs/%j/e.err

module load pytorch
source pvr_env/bin/activate

export MUJOCO_GL=egl

srun python3 train_bc.py \
	dataset=vd4rl_cheetah_expert \
	embedding=resnet50 \
    	wandb.enabled=true
