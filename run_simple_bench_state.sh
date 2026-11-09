#!/bin/bash
#SBATCH --job-name=simple_bench_state
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --error=./outs/simple_bench_state_%A/%a/e.err
#SBATCH --output=./outs/simple_bench_state_%A/%a/o.out
#SBATCH --array=0-2

module load mamba
source activate pvr_env/

ENVS=(pendulum bipedal_walker_state fourrooms_state)
SCRIPTS=(train_sac.py train_sac.py train_discrete.py)

E=$SLURM_ARRAY_TASK_ID

srun python3 ${SCRIPTS[$E]} \
        env=${ENVS[$E]} \
        embedding=none \
        algo.buffer_size=500000 \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-simple-bench \
        wandb.run_name=${ENVS[$E]}_state \
        hydra.run.dir="outputs/simple_bench/${ENVS[$E]}/state_${SLURM_ARRAY_JOB_ID}" \
        "$@"
