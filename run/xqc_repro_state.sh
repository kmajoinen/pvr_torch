#!/bin/bash
#SBATCH --job-name=xqc_repro_state
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --error=./outs/xqc_repro_state_%A/%a/e.err
#SBATCH --output=./outs/xqc_repro_state_%A/%a/o.out
#SBATCH --array=0-2

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl

TASK=dog-trot
SEED=$((SLURM_ARRAY_TASK_ID + 1))

srun python3 train_xqc.py \
        env=dm_control_state \
        env.id=dm_control/${TASK}-v0 \
        env.name=dmc_${TASK}_state \
        embedding=none \
        algo.action_repeat=2 \
        algo.buffer_size=500000 \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-xqc-repro \
        wandb.run_name=${TASK}_state_xqc_s${SEED} \
        seed=$SEED \
        hydra.run.dir="outputs/xqc_repro/${TASK}_state/xqc_s${SEED}_${SLURM_ARRAY_JOB_ID}" \
        "$@"
