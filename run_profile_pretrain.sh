#!/bin/bash
#SBATCH --job-name=r3m_profile
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=12
#SBATCH --error=./out_debug/profile_%A/e.err
#SBATCH --output=./out_debug/profile_%A/o.out

# Where an R3M step's time goes (profile_pretrain.py), for run_pretrain.sh's
# setup: ResNet-50, batch 16, fp32. Run 1 reads the real shards (straight from
# $WRKDIR: the loader isn't the bottleneck, data_wait ~0.1%); runs 2-5 use
# synthetic frames, so they compare the GPU side alone. Each run prints
# "[1] timing" (steps/s) ... "[4] synchronising calls". Compare [1] with the
# training run's ~12 steps/s, and note which GPU this job got (first line).

set -euo pipefail
module load mamba
source activate pvr_env/

export HF_HOME="$WRKDIR/hf"
DATA="$WRKDIR/data/ssv2"
OUT="$WRKDIR/profile_runs/r3m_$SLURM_JOB_ID"
COMMON=(+experiment=r3m embedding=resnet50 num_workers=$((SLURM_CPUS_PER_TASK - 1)))

run() {   # run <name> <overrides...>
    echo; echo "=================== $1"
    local name=$1; shift
    srun python3 profile_pretrain.py "${COMMON[@]}" hydra.run.dir="$OUT/$name" "$@"
}

run real_data         pretrain_data.root="$DATA"
run synthetic         profile.synthetic=true
run no_text_dropout   profile.synthetic=true +objectives.0.text_dropout=false
run no_language       profile.synthetic=true objectives.0.lang_weight=0
run bf16              profile.synthetic=true perf.amp_bf16=true
echo; echo "traces: $OUT/*/trace.json"
