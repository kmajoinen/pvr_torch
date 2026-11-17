#!/bin/bash
#SBATCH --job-name=xqc_repro_px
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --error=./outs/xqc_repro_px_%A/%a/e.err
#SBATCH --output=./outs/xqc_repro_px_%A/%a/o.out
#SBATCH --array=0-5

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl

ALGOS=(drqv2_xqc drqv2)
A=${ALGOS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3 + 1))

srun python3 train_drqv2.py \
        algo=$A \
        env=dmc_walker \
        env.id=dm_control/walker-run-v0 \
        env.name=dmc_walker_run \
        env.image_size=84 \
        embedding=random \
        embedding.frame_stack=3 \
        '+embedding.preprocess={resize:null,crop:null,mean:[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],std:[1,1,1,1,1,1,1,1,1]}' \
        encoder.grads=critic \
        encoder.lr=1e-4 \
        encoder.grad_clip=null \
        augmentation=random_shift \
        algo.total_timesteps=1500000 \
        algo.stddev_schedule.duration=500000 \
        algo.buffer_compression=zstd \
        save.enabled=false \
        wandb.enabled=true \
        wandb.project=pvr-xqc-repro \
        wandb.run_name=walker_run_pixels_${A}_s${SEED} \
        seed=$SEED \
        hydra.run.dir="outputs/xqc_repro/walker_run_pixels/${A}_s${SEED}_${SLURM_ARRAY_JOB_ID}" \
        "$@"
