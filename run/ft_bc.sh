#!/bin/bash
#SBATCH --job-name=bc_rn18ft
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/bc_rn18_%A/e.err
#SBATCH --output=./outs/bc_rn18_%A/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }


module load mamba
source activate pvr_env/

export MUJOCO_GL=egl
export HF_HOME="$WRKDIR/hf"

srun python3 train_bc.py \
        dataset=vd4rl_cheetah_expert \
        embedding=resnet18 \
        algo.max_epochs=100 \
        algo.eval_frequency=10 \
        algo.n_episodes_test=10 \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-bc \
        wandb.run_name=bc_cheetah_rn18_ft \
        encoder.grads=policy
