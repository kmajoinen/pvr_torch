#!/bin/bash
#SBATCH --job-name=pieg_cheetah
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
# #SBATCH --partition=gpu-a100-80g
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/pieg_%A/%a/e.err
#SBATCH --output=./outs/pieg_%A/%a/o.out
#SBATCH --array=0-4

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

# PIE-G on cheetah-run: DrQ-v2 + configs/experiment/pieg.yaml, 500k agent
# steps (1M frames). --time: set from run/debug_pieg.sh's SPS,
# hours = 500000 / SPS / 3600, plus a margin.

module load mamba
source activate pvr_env/

SEEDS=(
  1
  2
  3
  4
  5
)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}


export MUJOCO_GL=egl
DATA_DIR="$WRKDIR/data"     # Places365 images for the overlay: $DATA_DIR/val_256

# The preset sets every encoder option (the old finetune.* keys are gone):
# encoder.grads=critic, encoder.trainable=head, encoder.lr=1e-4,
# encoder.grad_clip=null, encoder.bn=adapt, no aux losses.
# hydra.run.dir: one directory per task (array tasks start in the same
# second, so the default timestamped directories would collide).
srun python3 train_drqv2.py \
        +experiment=pieg \
        env=dm_control_pixels \
        augmentation.strong.params.image_dir="$DATA_DIR/val_256" \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-drqv2 \
        seed=$SEED \
        hydra.run.dir="outputs/drqv2/pieg_cheetah_run/${SLURM_ARRAY_JOB_ID}_s${SEED}"
