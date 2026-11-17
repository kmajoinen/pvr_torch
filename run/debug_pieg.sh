#!/bin/bash
#SBATCH --job-name=pieg_debug
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./out_debug/pieg_%A/e.err
#SBATCH --output=./out_debug/pieg_%A/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

# PIE-G smoke test: DrQ-v2 + the pieg preset (ResNet18 layer2 at 84px, BatchNorm
# adapt, 3 frames fused by differences, linear head trained by the critic,
# random shift + random overlay). Needs Places365 images for the overlay:
#   cd $WRKDIR/data && wget http://data.csail.mit.edu/places/places365/val_256.tar && tar xf val_256.tar
# Check in o.out: the run reaches EVAL, "Encoder head: 63.4M parameters",
# "Replay buffer: ... (84, 84, 9) uint8", and the SPS value (sizes the full run).

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl

srun python3 train_drqv2.py \
        +experiment=pieg \
        env=dm_control_pixels \
        augmentation.strong.params.image_dir="$WRKDIR/data/val_256" \
        save.enabled=false \
        algo.total_timesteps=3000 \
        algo.buffer_size=5000 \
        algo.learning_starts=1000 \
        algo.log_frequency=250 \
        algo.eval_frequency=3000 \
        algo.n_episodes_test=1
