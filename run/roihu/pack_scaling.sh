#!/bin/bash
#SBATCH --job-name=pack_scaling
#SBATCH --account=project_2019621
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=72
#SBATCH --error=./outs/pack_scaling_%j/e.err
#SBATCH --output=./outs/pack_scaling_%j/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load python-pytorch
source "${PROJAPPL_DIR:-/projappl/project_2019621/pvr/pvr_torch}/pvr_env/bin/activate"
export WRKDIR="${SCRATCH_DIR:-/scratch/project_2019621/pvr}"

export MUJOCO_GL=egl

R3M="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1/encoder_step1000000.pt"
RUN=("$@")
[ ${#RUN[@]} -gt 0 ] || RUN=(
    train_xqc.py
    env=dm_control_pixels
    embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$R3M embedding.name=r3m50_ours
    ++embedding.frame_stack=3
    algo.action_repeat=4
    algo.total_timesteps=500000
    algo.buffer_size=500000
)

PACK=${PACK:-1}
CPUS=($(python3 -c 'import os; print(*sorted(os.sched_getaffinity(0)))'))
C=$(( ${#CPUS[@]} / PACK ))
LOGS=outs/pack_scaling_${SLURM_JOB_ID}
mkdir -p "$LOGS"

if [ "${MPS:-0}" = 1 ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 5 > "$LOGS/gpu.csv" &
SMI=$!

PIDS=()
for i in $(seq 0 $((PACK - 1))); do
    OMP_NUM_THREADS=$C taskset -c "$(IFS=,; echo "${CPUS[*]:i*C:C}")" \
    python3 "${RUN[@]}" \
        seed=$((i + 1)) \
        timing.benchmark_steps=1500 \
        algo.learning_starts=1000 \
        algo.log_frequency=250 \
        save.enabled=false \
        wandb.enabled=false \
        hydra.run.dir="outputs/pack_scaling/${SLURM_JOB_ID}_run$i" \
        > "$LOGS/run$i.out" 2> "$LOGS/run$i.err" &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
kill $SMI
[ "${MPS:-0}" = 1 ] && echo quit | nvidia-cuda-mps-control

echo "PACK=$PACK MPS=${MPS:-0} ${RUN[*]:0:2}"
grep -h "^  training " "$LOGS"/run*.out | awk '{s += $2; n++} END {printf "aggregate %.2f steps/s over %d runs\n", s, n}'
grep -h "^Predicted time" "$LOGS"/run*.out
awk -F', ' '{u += $1; if ($2 > m) m = $2; n++} END {if (n) printf "GPU load %.0f%% mean, %.1f GiB peak\n", u / n, m / 1024}' "$LOGS/gpu.csv"
