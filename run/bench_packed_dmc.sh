#!/bin/bash
#SBATCH --job-name=bench_dmc
#SBATCH --mem=52G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --error=./outs/bench_dmc_%A/%a/e.err
#SBATCH --output=./outs/bench_dmc_%A/%a/o.out
#SBATCH --array=0-13

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl

VIP=$(ls -v "$WRKDIR"/pretrain_runs/vip_ssv2_resnet50_s1/encoder_step*.pt | tail -1)
R3M="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1/encoder_step1000000.pt"
MVP=$(ls -v "$WRKDIR"/pretrain_runs/mvp_ssv2_vits16_epochs_s1/encoder_step*.pt | tail -1)
echo "checkpoints: VIP=$VIP R3M=$R3M MVP=$MVP"

ENVS=(dm_control_pixels dmc_walker)
ENV_NAMES=(cheetah_run walker_walk)
ACTION_REPEATS=(4 2)

MODELS=(
  "embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$VIP embedding.head=fc embedding.head_dim=1024 embedding.name=vip50_ours"
  "embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$R3M embedding.name=r3m50_ours"
  "embedding=custom_checkpoint embedding.arch=vit_s16 embedding.checkpoint=$MVP embedding.name=mvp_vits16_ours"
  "embedding=vip_resnet50"
  "embedding=r3m_resnet50"
  "embedding=voltron_r_mvp"
  "embedding=voltron_r_r3m_rn50"
)
NAMES=(vip50_ours r3m50_ours mvp_ours vip50 r3m50 r_mvp r_r3m50)

E=$((SLURM_ARRAY_TASK_ID / 7))
M=$((SLURM_ARRAY_TASK_ID % 7))

SEEDS=(${SEEDS:-${SEED:-1 2 3}})
CPUS=($(python3 -c 'import os; print(*sorted(os.sched_getaffinity(0)))'))
C=$(( ${#CPUS[@]} / ${#SEEDS[@]} ))
LOGS=outs/bench_dmc_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p "$LOGS"

PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    OMP_NUM_THREADS=$C taskset -c "$(IFS=,; echo "${CPUS[*]:i*C:C}")" \
    python3 train_xqc.py \
        env=${ENVS[$E]} \
        ${MODELS[$M]} \
        ++embedding.frame_stack=3 \
        algo.action_repeat=${ACTION_REPEATS[$E]} \
        algo.total_timesteps=500000 \
        algo.buffer_size=500000 \
        save.enabled=false \
        seed=$SEED \
        wandb.enabled=true \
        wandb.project=pvr-benchmark \
        wandb.run_name=${ENV_NAMES[$E]}_${NAMES[$M]}_xqc_s${SEED} \
        hydra.run.dir="outputs/benchmark/${ENV_NAMES[$E]}/${NAMES[$M]}_xqc_s${SEED}_${SLURM_ARRAY_JOB_ID}" \
        "$@" > "$LOGS/s$SEED.out" 2> "$LOGS/s$SEED.err" &
    PIDS+=($!)
done

RC=0
for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} || { echo "seed ${SEEDS[$i]} failed"; RC=1; }; done
exit $RC
