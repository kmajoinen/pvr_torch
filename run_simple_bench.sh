#!/bin/bash
#SBATCH --job-name=simple_bench
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --error=./outs/simple_bench_%A/%a/e.err
#SBATCH --output=./outs/simple_bench_%A/%a/o.out
#SBATCH --array=0-8

module load mamba
source activate pvr_env/

VIP=$(ls "$WRKDIR"/pretrain_runs/vip_ssv2_resnet50_s1/encoder_step*.pt | tail -1)
R3M="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1/encoder_step1000000.pt"
MVP=$(ls "$WRKDIR"/pretrain_runs/mvp_ssv2_vits16_epochs_s1/encoder_step*.pt | tail -1)

ENVS=(pendulum_pixels bipedal_walker_pixels fourrooms_pixels)
SCRIPTS=(train_sac.py train_sac.py train_discrete.py)
ENV_ARGS=(
  "+embedding.frame_stack=3"
  "+embedding.frame_stack=3"
  ""
)

MODELS=(
  "embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$VIP embedding.head=fc embedding.head_dim=1024 embedding.name=vip50_ours"
  "embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$R3M embedding.name=r3m50_ours"
  "embedding=custom_checkpoint embedding.arch=vit_s16 embedding.checkpoint=$MVP embedding.name=mvp_vits16_ours"
)
NAMES=(vip50 r3m50 mvp)

E=$((SLURM_ARRAY_TASK_ID / 3))
M=$((SLURM_ARRAY_TASK_ID % 3))

srun python3 ${SCRIPTS[$E]} \
        env=${ENVS[$E]} \
        ${MODELS[$M]} \
        ${ENV_ARGS[$E]} \
        algo.buffer_size=500000 \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-simple-bench \
        wandb.run_name=${ENVS[$E]}_${NAMES[$M]} \
        hydra.run.dir="outputs/simple_bench/${ENVS[$E]}/${NAMES[$M]}_${SLURM_ARRAY_JOB_ID}" \
        "$@"
