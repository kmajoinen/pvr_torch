#!/bin/bash
#SBATCH --job-name=prep_oxe
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --error=./outs/prep_oxe_%A/e.err
#SBATCH --output=./outs/prep_oxe_%A/o.out

set -euo pipefail
module load mamba
source activate pvr_env/

OXE=/scratch/shareddata/dldata/open_x_embodiment/tensorflow_datasets
DATASET=${DATASET:-bridge_0.1.0}
CAMERA=${CAMERA:-}
EVERY=${EVERY:-}
LIMIT=${LIMIT:-}
OUT="$WRKDIR/data/oxe/${DATASET%_[0-9]*}${CAMERA:+_$CAMERA}${EVERY:+_every$EVERY}${LIMIT:+_limit$LIMIT}"

srun python -m src.data.prepare_rlds \
    --source "$OXE/$DATASET" \
    --out "$OUT" \
    --workers $SLURM_CPUS_PER_TASK \
    ${CAMERA:+--camera $CAMERA} \
    ${EVERY:+--every $EVERY} \
    ${LIMIT:+--limit $LIMIT}
python -m src.data.check "$OUT"
