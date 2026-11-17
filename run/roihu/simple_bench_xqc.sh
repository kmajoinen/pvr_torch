#!/bin/bash
#SBATCH --job-name=simple_bench_xqc
#SBATCH --account=project_2019621
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=72
#SBATCH --error=./outs/simple_bench_xqc_%A/%a/e.err
#SBATCH --output=./outs/simple_bench_xqc_%A/%a/o.out
#SBATCH --array=0-5

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load python-pytorch
source "${PROJAPPL_DIR:-/projappl/project_2019621/pvr/pvr_torch}/pvr_env/bin/activate"
export WRKDIR="${SCRATCH_DIR:-/scratch/project_2019621/pvr}"

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
