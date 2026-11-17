#!/bin/bash
#SBATCH --job-name=simple_bench_xqc_state
#SBATCH --account=project_2019621
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=72
#SBATCH --error=./outs/simple_bench_xqc_state_%A/%a/e.err
#SBATCH --output=./outs/simple_bench_xqc_state_%A/%a/o.out
#SBATCH --array=0-1

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load python-pytorch
source "${PROJAPPL_DIR:-/projappl/project_2019621/pvr/pvr_torch}/pvr_env/bin/activate"
export WRKDIR="${SCRATCH_DIR:-/scratch/project_2019621/pvr}"

ENVS=(pendulum bipedal_walker_state)
E=$SLURM_ARRAY_TASK_ID

srun python3 train_xqc.py \
        env=${ENVS[$E]} \
        embedding=none \
        algo.buffer_size=500000 \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-simple-bench \
        wandb.run_name=${ENVS[$E]}_state_xqc \
        hydra.run.dir="outputs/simple_bench/${ENVS[$E]}/state_xqc_${SLURM_ARRAY_JOB_ID}" \
        "$@"
