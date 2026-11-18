#!/bin/bash
#SBATCH --job-name=roihu_debug
#SBATCH --account=project_2019621
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --error=./outs/roihu_debug_%j/e.err
#SBATCH --output=./outs/roihu_debug_%j/o.out

[ -f pretrain.py ] || { echo "submit from the repo root"; exit 1; }

module load python-pytorch
source "${PROJAPPL_DIR:-/projappl/project_2019621/pvr/pvr_torch}/pvr_env/bin/activate"
export WRKDIR="${SCRATCH_DIR:-/scratch/project_2019621/pvr}"
export MUJOCO_GL=egl

srun python3 -m pytest tests/test_env_seeding.py tests/test_metaworld.py tests/test_xqc.py tests/test_voltron.py -q

for ENV in dm_control_pixels adroit_door_pixels metaworld_pixels; do
    srun python3 train_xqc.py \
            env=$ENV \
            embedding=resnet18 \
            embedding.pretrained=false \
            ++embedding.frame_stack=3 \
            algo.total_timesteps=1500 \
            algo.learning_starts=1000 \
            algo.log_frequency=250 \
            algo.eval_frequency=1500 \
            algo.n_episodes_test=1 \
            algo.buffer_size=10000 \
            save.enabled=false \
            wandb.enabled=false \
            hydra.run.dir="outputs/roihu_debug/${ENV}_${SLURM_JOB_ID}"
done
