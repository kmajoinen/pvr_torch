#!/bin/bash
#SBATCH --job-name=run_ft_r3mours
#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=12:00:00          # ResNet-50 fine-tuning is slower than the RN18 run's 7h
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/r3m50_%A/e.err
#SBATCH --output=./outs/r3m50_%A/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

# SAC on our R3M encoder (pretrain.py +experiment=r3m, run/pretrain/r3m.sh).
# CKPT: the final snapshot; any encoder_step<N>.pt of the run works too
# (earlier ones: does downstream improve with more pretraining?).
# Compare against embedding=r3m_resnet50 (released R3M) and embedding=resnet50
# (ImageNet), same overrides. encoder.grads=none tests the representation itself.

# module --ignore-cache load mamba
module load mamba
source activate pvr_env/

export MUJOCO_GL=egl
RUN_DIR="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1"
CKPT="${CKPT:-$RUN_DIR/encoder_step1000000.pt}"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT (ls $RUN_DIR/encoder_step*.pt)"; exit 1; }

srun python3 train_sac.py \
        env=dm_control_pixels \
        env.image_size=84 \
        save.enabled=false \
        algo.total_timesteps=500000 \
        wandb.enabled=true \
        wandb.project=pvr-sac \
        encoder.grads=critic \
        embedding=custom_checkpoint \
        embedding.name=r3m50_ssv2_ours \
        embedding.arch=resnet50 \
        embedding.checkpoint="$CKPT"
