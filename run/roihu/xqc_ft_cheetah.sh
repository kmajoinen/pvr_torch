#!/bin/bash
#SBATCH --job-name=xqc_ft_cheetah
#SBATCH --account=project_2019621
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=36:00:00  # TODO: set from pack_scaling.sh; gpumedium's maximum
#SBATCH --cpus-per-task=72
#SBATCH --error=./outs/xqc_ft_cheetah_%j/e.err
#SBATCH --output=./outs/xqc_ft_cheetah_%j/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load python-pytorch
source "${PROJAPPL_DIR:-/projappl/project_2019621/pvr/pvr_torch}/pvr_env/bin/activate"
export WRKDIR="${SCRATCH_DIR:-/scratch/project_2019621/pvr}"

export MUJOCO_GL=egl

SEEDS=(${SEEDS:-${SEED:-1 2 3}})
CPUS=($(python3 -c 'import os; print(*sorted(os.sched_getaffinity(0)))'))
C=$(( ${#CPUS[@]} / ${#SEEDS[@]} ))
LOGS=outs/xqc_ft_cheetah_${SLURM_JOB_ID}
mkdir -p "$LOGS"

PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    OMP_NUM_THREADS=$C taskset -c "$(IFS=,; echo "${CPUS[*]:i*C:C}")" \
    python3 train_xqc.py \
        env=dm_control_pixels \
        env.image_size=84 \
        embedding=resnet18 \
        encoder.grads=critic \
        algo.action_repeat=4 \
        algo.gamma=0.99 \
        algo.total_timesteps=200000 \
        algo.buffer_size=200000 \
        save.enabled=false \
        seed=$SEED \
        wandb.enabled=true \
        wandb.project=pvr-sac \
        hydra.run.dir="outputs/xqc_ft_cheetah/s${SEED}_${SLURM_JOB_ID}" \
        "$@" > "$LOGS/s$SEED.out" 2> "$LOGS/s$SEED.err" &
    PIDS+=($!)
done

RC=0
for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} || { echo "seed ${SEEDS[$i]} failed"; RC=1; }; done
exit $RC
