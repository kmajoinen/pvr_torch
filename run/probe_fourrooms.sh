#!/bin/bash
#SBATCH --job-name=probe_4rooms
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h200-71g-ia-ellis,gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --error=./outs/probe_4rooms_%A/%a/e.err
#SBATCH --output=./outs/probe_4rooms_%A/%a/o.out
#SBATCH --array=0-7

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load mamba
source activate pvr_env/

R3M="$WRKDIR/pretrain_runs/r3m_ssv2_resnet50_s1/encoder_step1000000.pt"
IMAGENET="$WRKDIR/pretrain_runs/imagenet_supervised_resnet18_s1/encoder_step0450360.pt"
MVP=$(ls -v "$WRKDIR"/pretrain_runs/mvp_ssv2_vits16_epochs_s1/encoder_step*.pt | tail -1)
echo "checkpoints: R3M=$R3M IMAGENET=$IMAGENET MVP=$MVP"

MODELS=(
  "embedding=custom_checkpoint embedding.arch=resnet50 embedding.checkpoint=$R3M embedding.name=r3m50_ours"
  "embedding=custom_checkpoint embedding.arch=resnet18 embedding.checkpoint=$IMAGENET embedding.name=imagenet18_ours"
  "embedding=custom_checkpoint embedding.arch=vit_s16 embedding.checkpoint=$MVP embedding.name=mvp_vits16_ours"
  "embedding=resnet18 embedding.pretrained=false"
)
UNPOOLED=(
  "++embedding.layer=layer4 ++embedding.head=flatten"
  "++embedding.layer=layer4 ++embedding.head=flatten"
  "++embedding.layer=12 ++embedding.head=flatten"
  "++embedding.layer=layer4 ++embedding.head=flatten"
)
NAMES=(r3m50 imagenet18 mvp random18)

M=$((SLURM_ARRAY_TASK_ID / 2))
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 1 ]; then
  FEATS="${UNPOOLED[$M]}"; TAG=unpooled
else
  FEATS=""; TAG=pooled
fi

srun python3 probe_encoder.py \
        env=fourrooms_pixels \
        ${MODELS[$M]} \
        $FEATS \
        hydra.run.dir="outputs/probe/fourrooms/${NAMES[$M]}_${TAG}_${SLURM_ARRAY_JOB_ID}"
