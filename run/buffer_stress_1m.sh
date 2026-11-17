#!/bin/bash
#SBATCH --job-name=buf_stress_1m
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --tmp=60G
#SBATCH --constraint=localdisk
#SBATCH --error=./outs/buf_stress_%A/e.err
#SBATCH --output=./outs/buf_stress_%A/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load mamba
source activate pvr_env/

export MUJOCO_GL=egl

BUF=/tmp/$SLURM_JOB_ID
mkdir -p $BUF
trap 'rm -rf $BUF' EXIT
findmnt -T $BUF -o TARGET,SOURCE,FSTYPE

srun python3 -m src.rl.buffer_stress \
        --env adroit_hammer_pixels \
        --image-size 224 \
        --capacity 1000000 \
        --frames 1000000 \
        --compression zstd \
        --max-gb 45 \
        --storage disk \
        --dir $BUF \
        --sample-every 4 \
        --report-every 100000
