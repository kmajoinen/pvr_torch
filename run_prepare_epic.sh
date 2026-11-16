#!/bin/bash
#SBATCH --job-name=prep_epic
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --error=./outs/prep_epic_%A/e.err
#SBATCH --output=./outs/prep_epic_%A/o.out

set -euo pipefail
module load mamba
source activate pvr_env/

VIDEOS=/scratch/shareddata/dldata/epic-kitchens/EPIC-KITCHENS
ANNOTATIONS="$WRKDIR/data/epic-kitchens-100-annotations"
LIMIT=${LIMIT:-}
OUT="$WRKDIR/data/epic${LIMIT:+_limit$LIMIT}"

for split in train validation; do
    srun python -m src.data.prepare_epic \
        --videos "$VIDEOS" \
        --annotations "$ANNOTATIONS" \
        --out "$OUT" \
        --split $split \
        --window 300 \
        --workers $SLURM_CPUS_PER_TASK \
        ${LIMIT:+--limit $LIMIT}
done
python -m src.data.check "$OUT"
