#!/usr/bin/env bash
# ./run.sh STAGE [--quick] [overrides...]
#   STAGE: prepare | data | pretrain | finetune | grpo | evaluate | export | demo
#   --quick appends a few overrides for a run of minutes; any other argument goes to the stage's
#   command and overrides the YAML file (./run.sh pretrain --learning_rate 3e-4 --num_train_epochs 1).
set -euo pipefail
cd "$(dirname "$0")"

usage="usage: ./run.sh {prepare|data|pretrain|finetune|grpo|evaluate|export|demo} [--quick] [overrides...]"
stage=${1:?$usage}
shift
quick=()
case $stage in
  prepare)  cmd=(python -m abcgpt.prepare --config configs/prepare.yaml) ;;
  data)     cmd=(python -m abcgpt.data --config configs/data.yaml) ;;
  pretrain) cmd=(trl sft --config configs/pretrain.yaml);  quick=(--max_steps 300 --eval_steps 150) ;;
  finetune) cmd=(trl sft --config configs/finetune.yaml);  quick=(--max_steps 200 --eval_steps 100) ;;
  grpo)     cmd=(trl grpo --config configs/grpo.yaml);     quick=(--max_steps 20) ;;
  evaluate) cmd=(python -m abcgpt.evaluate --config configs/evaluate.yaml); quick=(--num_samples 20) ;;
  export)   cmd=(python -m abcgpt.export_gguf --config configs/export.yaml) ;;
  demo)     cmd=(python -m abcgpt.app --config configs/app.yaml) ;;
  *) echo "$usage" >&2; exit 1 ;;
esac
if [[ ${1:-} == --quick ]]; then shift; else quick=(); fi

# Weights & Biases, used when the YAML file says report_to: wandb (run `uv run wandb login` once).
# Set these variables before ./run.sh to change them: WANDB_ENTITY (team or user), WANDB_MODE=offline
# (no network: sync later with `uv run wandb sync wandb/offline-run-*`), WANDB_MODE=disabled,
# WANDB_LOG_MODEL=end (upload the final model as an artifact).
export WANDB_PROJECT=${WANDB_PROJECT:-abc-gpt}

set -x
exec uv run "${cmd[@]}" "${quick[@]}" "$@"
