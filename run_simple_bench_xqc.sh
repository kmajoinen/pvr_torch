#!/bin/bash
#SBATCH --job-name=simple_bench_xqc
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --error=./outs/simple_bench_xqc_%A/%a/e.err
#SBATCH --output=./outs/simple_bench_xqc_%A/%a/o.out
#SBATCH --array=0-5

module load mamba
source activate pvr_env/

VIP=$(ls "$WRKDIR"/pretrain_runs/vip_ssv2_resnet50_s1/encoder_step*.pt | tail -1)
R3M="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1/encoder_step1000000.pt"
MVP=$(ls "$WRKDIR"/pretrain_runs/mvp_ssv2_vits16_epochs_s1/encoder_step*.pt | tail -1)

ENVS=(pendulum_pixels bipedal_walker_pixels)

MODELS=(
  "embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$VIP embedding.head=fc embedding.head_dim=1024 embedding.name=vip50_ours"
  "embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$R3M embedding.name=r3m50_ours"
  "embedding=custom_checkpoint embedding.arch=vit_s16 embedding.checkpoint=$MVP embedding.name=mvp_vits16_ours"
)
NAMES=(vip50 r3m50 mvp)

E=$((SLURM_ARRAY_TASK_ID / 3))
M=$((SLURM_ARRAY_TASK_ID % 3))

srun python3 train_xqc.py \
        env=${ENVS[$E]} \
        ${MODELS[$M]} \
        +embedding.frame_stack=3 \
        algo.buffer_size=500000 \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-simple-bench \
        wandb.run_name=${ENVS[$E]}_${NAMES[$M]}_xqc \
        hydra.run.dir="outputs/simple_bench/${ENVS[$E]}/${NAMES[$M]}_xqc_${SLURM_ARRAY_JOB_ID}" \
        "$@"
