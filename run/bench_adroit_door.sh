#!/bin/bash
#SBATCH --job-name=bench_door
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/bench_door_%A/%a/e.err
#SBATCH --output=./outs/bench_door_%A/%a/o.out
#SBATCH --array=0-6

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl

VIP=$(ls -v "$WRKDIR"/pretrain_runs/vip_ssv2_resnet50_s1/encoder_step*.pt | tail -1)
R3M="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1/encoder_step1000000.pt"
MVP=$(ls -v "$WRKDIR"/pretrain_runs/mvp_ssv2_vits16_epochs_s1/encoder_step*.pt | tail -1)
echo "checkpoints: VIP=$VIP R3M=$R3M MVP=$MVP"

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

M=$SLURM_ARRAY_TASK_ID

SEED=${SEED:-1}

srun python3 train_xqc.py \
        env=adroit_door_pixels \
        ${MODELS[$M]} \
        ++embedding.frame_stack=3 \
        algo.total_timesteps=5000000 \
        algo.buffer_size=1000000 \
        save.enabled=false \
        seed=$SEED \
        wandb.enabled=true \
        wandb.project=pvr-benchmark \
        wandb.run_name=adroit_door_${NAMES[$M]}_xqc_s${SEED} \
        hydra.run.dir="outputs/benchmark/adroit_door/${NAMES[$M]}_xqc_s${SEED}_${SLURM_ARRAY_JOB_ID}" \
        "$@"
