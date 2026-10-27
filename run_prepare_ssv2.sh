#!/bin/bash
#SBATCH --job-name=prep_ssv2
#SBATCH --cpus-per-task=16       # one conversion worker per CPU
#SBATCH --mem=48G                # the extracted archive (~25 GB, fixed: /tmp is a RAM disk on
                                 #   diskless CPU nodes) + workers (~0.5 GB each) + main
#SBATCH --time=24:00:00          # conversion can't resume: leave margin
# No --tmp/--constraint=localdisk: Triton's CPU partitions may not offer them;
# /tmp is then RAM, which --mem above covers.
#SBATCH --error=./outs/prep_ssv2_%A/e.err
#SBATCH --output=./outs/prep_ssv2_%A/o.out

# One-time conversion of Triton's Something-Something v2 archive into frame
# shards for pretrain.py (R3M). The archive (videos + labels) is extracted on
# the node's local disk -- its 220k small files never touch the shared
# filesystem -- and the shards (a few hundred ~1 GB files) are written to
# $OUT. Training jobs (run_pretrain.sh) then copy only the shards.

set -euo pipefail
module load mamba
source activate pvr_env/

ARCHIVE=/scratch/shareddata/dldata/something-something-v2.tar
OUT="$WRKDIR/data/ssv2"
LOCAL="/tmp/ssv2-$SLURM_JOB_ID"       # /tmp is per user, not per job
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
trap 'exit 143' TERM                  # cancel/timeout: stop here (EXIT then cleans up)

# Extracted straight from the shared drive (one sequential read): no copy of
# the archive needed next to its contents.
need_kb=$(du -k --apparent-size "$ARCHIVE" | cut -f1); free_kb=$(df -k --output=avail /tmp | tail -1)
[ "$need_kb" -lt "$free_kb" ] || { echo "/tmp has $((free_kb / 1048576)) GB free, the archive needs $((need_kb / 1048576)) GB"; exit 1; }
df -h /tmp
echo "extracting $ARCHIVE -> $LOCAL"; time tar -xf "$ARCHIVE" -C "$LOCAL"
VIDEOS="$LOCAL/20bn-something-something-v2"
LABELS="$LOCAL/something-something-v2-labels/labels"

for split in train validation; do
    srun python -m src.data.prepare_ssv2 --videos "$VIDEOS" --labels "$LABELS/$split.json" \
        --out "$OUT" --split $split --workers $SLURM_CPUS_PER_TASK
done
