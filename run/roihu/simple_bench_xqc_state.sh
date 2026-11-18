#!/bin/bash
#SBATCH --job-name=simple_bench_xqc_state
#SBATCH --account=project_2019621
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00  # TODO: set from pack_scaling.sh
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

SEEDS=(${SEEDS:-${SEED:-1 2 3}})
CPUS=($(python3 -c 'import os; print(*sorted(os.sched_getaffinity(0)))'))
C=$(( ${#CPUS[@]} / ${#SEEDS[@]} ))
LOGS=outs/simple_bench_xqc_state_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p "$LOGS"

PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    OMP_NUM_THREADS=$C taskset -c "$(IFS=,; echo "${CPUS[*]:i*C:C}")" \
    python3 train_xqc.py \
        env=${ENVS[$E]} \
        embedding=none \
        algo.buffer_size=500000 \
        save.enabled=false \
        seed=$SEED \
        wandb.enabled=true \
        wandb.project=pvr-simple-bench \
        wandb.run_name=${ENVS[$E]}_state_xqc_s${SEED} \
        hydra.run.dir="outputs/simple_bench/${ENVS[$E]}/state_xqc_s${SEED}_${SLURM_ARRAY_JOB_ID}" \
        "$@" > "$LOGS/s$SEED.out" 2> "$LOGS/s$SEED.err" &
    PIDS+=($!)
done

RC=0
for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} || { echo "seed ${SEEDS[$i]} failed"; RC=1; }; done
exit $RC
