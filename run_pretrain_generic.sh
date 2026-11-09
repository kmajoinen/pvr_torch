#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=17       # 16 loader workers (JPEG decoding) + main
#SBATCH --tmp=200G               # local disk for the data copy (ImageNet extracted ~150 GB,
                                 #   SSv2 shards ~100-150 GB). Not reserved: checked below
#SBATCH --constraint=localdisk   # a node with a local disk (diskless nodes' /tmp is RAM)
#SBATCH --error=./outs/pretrain_%A/e.err
#SBATCH --output=./outs/pretrain_%A/o.out

# "Network X, trained on data Y, with method Z and augmentation K" -- any
# combination pretrain.py supports, picked by the four variables below.
#
#   ARCH    the network: any embedding config pretrain.py can train from
#           scratch -- vit_s16 | vit_b16 | vit_l16 (MAE/MVP-layout ViTs),
#           resnet18 | resnet34 | resnet50, dinov2_vits14 ...
#   DATA    imagenet     ImageNet, extracted from Triton's shared tar to /tmp
#           ssv2         SSv2 as video clips (shards from run_prepare_ssv2.sh)
#           ssv2_frames  SSv2's frames as unlabelled images (same shards)
#           epic         EPIC-KITCHENS-100 narrated segments (run_prepare_epic.sh)
#           epic_frames  EPIC's frames as unlabelled images (same shards)
#           /a/path      any prepared dataset folder (pretrain_data=auto: pretrain.py
#                        picks clips or images to match METHOD); checked before the copy
#   METHOD  the objective and its recipe, a configs/experiment/ preset:
#             image data (imagenet, ssv2_frames):
#               imagenet_supervised (imagenet only: needs labels), moco_v2, mae, mvp
#             clip data (ssv2):  r3m, r3m_vit, vip
#   AUG     a configs/augmentation/ file (color_jitter, random_grayscale,
#           gaussian_blur, horizontal_flip, random_erasing, random_shift,
#           random_resized_crop, moco_v2, ...), or "" for the method's own.
#           Applied on the GPU after the loader's crop; for clip methods the
#           same draw goes on all of a video's frames (colour transforms then
#           switch to per-frame draws, see below).
#   EXTRA   any further Hydra overrides (e.g. "lr=1e-4 batch_size=128").
#
# Examples (set the variables, or pass them in: METHOD=moco_v2 AUG=color_jitter sbatch ...):
#   ARCH=vit_s16 DATA=imagenet    METHOD=imagenet_supervised AUG=color_jitter
#   ARCH=vit_s16 DATA=ssv2_frames METHOD=mae                 AUG=""
#   ARCH=vit_s16 DATA=ssv2        METHOD=r3m_vit             AUG=random_resized_crop
#   ARCH=resnet18 DATA=imagenet   METHOD=moco_v2             AUG=""
# DRY_RUN=1 bash run_pretrain_generic.sh prints the checks and the command only.
# Resubmitting with the same RUN_DIR resumes; chain long runs with
#   jid=$(sbatch --parsable run_pretrain_generic.sh); sbatch --dependency=afterany:$jid run_pretrain_generic.sh

set -euo pipefail

ARCH="${ARCH:-vit_s16}"
DATA="${DATA:-imagenet}"
METHOD="${METHOD:-imagenet_supervised}"
AUG="${AUG-color_jitter}"
SEED="${SEED:-1}"
EXTRA="${EXTRA:-}"

IMAGENET_TAR=/scratch/shareddata/dldata/imagenet_imagefolder/ILSVRC2012_imagefolder.tar
DATA_NAME=$(basename "$DATA")
RUN_DIR="${RUN_DIR:-$WRKDIR/pretrain_runs/${METHOD}_${DATA_NAME}_${ARCH}_aug-${AUG:-default}_s${SEED}}"

# ---- check the combination before using any resources -----------------------
case "$METHOD" in
    r3m|r3m_vit|vip)                       kind=clips ;;
    imagenet_supervised|moco_v2|mae|mvp)   kind=images ;;
    *) echo "unknown METHOD=$METHOD (see configs/experiment/)"; exit 1 ;;
esac
case "$DATA" in
    ssv2|epic)                         data_kind=clips;  SRC="$WRKDIR/data/${DATA}" ;;
    imagenet)                          data_kind=images; SRC="" ;;
    ssv2_frames|epic_frames)           data_kind=images; SRC="$WRKDIR/data/${DATA%_frames}" ;;
    */*) [ -d "$DATA" ] || { echo "DATA=$DATA: no such folder"; exit 1; }
                                       data_kind=auto;   SRC="$DATA" ;;
    *) echo "unknown DATA=$DATA (imagenet | ssv2 | ssv2_frames | epic | epic_frames | a dataset folder's path)"; exit 1 ;;
esac
[ "$data_kind" = auto ] || [ "$kind" = "$data_kind" ] || { echo "METHOD=$METHOD reads $kind, but DATA=$DATA is $data_kind"; exit 1; }
[ "$METHOD" != imagenet_supervised ] || [ "$DATA" = imagenet ] || [ "$data_kind" = auto ] || { echo "imagenet_supervised needs labels (ImageNet, or a labelled dataset folder)"; exit 1; }
[ -z "$AUG" ] || [ -f "configs/augmentation/$AUG.yaml" ] || { echo "no configs/augmentation/$AUG.yaml"; exit 1; }
[ -f "configs/embedding/$ARCH.yaml" ] || { echo "no configs/embedding/$ARCH.yaml"; exit 1; }

if [ "$data_kind" = auto ]; then DATA_CFG=auto; else DATA_CFG="$DATA"; fi
OVERRIDES=(+experiment="$METHOD" embedding="$ARCH" pretrain_data="$DATA_CFG" seed="$SEED")
[ -z "$AUG" ] || OVERRIDES+=(augmentation="$AUG")
if [ "$kind" = clips ]; then
    case "$AUG" in
        # Colour transforms need one RGB frame at a time: draw them per frame.
        color_jitter|random_grayscale|moco_v2) OVERRIDES+=(clip_augmentation=per_frame)
            echo "note: $AUG on clips is drawn per frame (clip_augmentation=per_frame)" ;;
    esac
fi
case "$METHOD" in r3m|r3m_vit) export HF_HOME="$WRKDIR/hf" ;; esac   # R3M's DistilBERT downloads here

echo "network=$ARCH  data=$DATA  method=$METHOD  augmentation=${AUG:-the method default}  seed=$SEED"
echo "run dir: $RUN_DIR"

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "python3 pretrain.py ${OVERRIDES[*]} pretrain_data.root=<local copy> $EXTRA hydra.run.dir=$RUN_DIR"
    exit 0
fi

module load mamba
source activate pvr_env/

# ---- data onto the node's local disk ------------------------------------------
LOCAL="/tmp/$SLURM_JOB_ID"                     # /tmp is per user, not per job
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
trap 'exit 143' TERM                           # cancel/timeout: stop here (EXIT then cleans up)
free_kb=$(df -k --output=avail /tmp | tail -1)
if [ "$DATA" = imagenet ]; then
    need_kb=$(du -k --apparent-size "$IMAGENET_TAR" | cut -f1)
    [ "$need_kb" -lt "$free_kb" ] || { echo "only $((free_kb / 1048576)) GB free in /tmp"; exit 1; }
    echo "extracting $IMAGENET_TAR -> $LOCAL"; time tar xf "$IMAGENET_TAR" -C "$LOCAL"
    TRAIN=$(find "$LOCAL" -maxdepth 3 -type d -name train -print -quit)
    [ -n "$TRAIN" ] && [ -d "$(dirname "$TRAIN")/val" ] || { echo "no train/ and val/ under $LOCAL"; exit 1; }
    ROOT=$(dirname "$TRAIN")
else
    python3 -m src.data.check "$SRC" --quick || { echo "$SRC isn't a dataset pretrain.py reads (above)"; exit 1; }
    need_kb=$(du -sk "$SRC" | cut -f1)
    [ "$need_kb" -lt "$free_kb" ] || { echo "only $((free_kb / 1048576)) GB free in /tmp"; exit 1; }
    ROOT="$LOCAL/$(basename "$SRC")"
    echo "copying $SRC -> $ROOT"; time cp -r "$SRC" "$ROOT"
fi

# shellcheck disable=SC2086  # EXTRA is a list of overrides
srun python3 pretrain.py \
        "${OVERRIDES[@]}" \
        pretrain_data.root="$ROOT" \
        num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
        wandb.enabled=true \
        wandb.project=pvr-pretrain \
        hydra.run.dir="$RUN_DIR" \
        $EXTRA
