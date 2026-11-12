#!/bin/bash
# sbatch slurm/train.sbatch STAGE [--quick] [overrides...]       (STAGE: pretrain, finetune or grpo)
# Uses every GPU of one node. Run ./run.sh prepare and ./run.sh data first (prepare needs internet
# access): training reads only the local data/ folder, since compute nodes often have no network.
#SBATCH --job-name=abc-gpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load cuda   (if the cluster needs it)

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # read data and models from the cache only
# export HF_HOME=/path/to/shared/hf_cache       # if the cache is not in $HOME
export WANDB_MODE=${WANDB_MODE:-offline}           # no network: `uv run wandb sync wandb/offline-run-*` later
export UV_NO_SYNC=1                             # the environment was created with `uv sync` beforehand

gpus=${SLURM_GPUS_ON_NODE:-1}
launch=(--num_processes "$gpus")                # arguments that trl hands to `accelerate launch`
(( gpus > 1 )) && launch+=(--multi_gpu)
./run.sh "$@" "${launch[@]}"
