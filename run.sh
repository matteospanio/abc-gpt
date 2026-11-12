#!/usr/bin/env bash
# ./run.sh STAGE [--quick] [overrides...]
#   STAGE: prepare | pretrain | finetune | grpo | evaluate | export | demo
#   --quick appends a few overrides for a run of minutes; any other argument goes to the stage's
#   command and overrides the YAML file (./run.sh pretrain --learning_rate 3e-4 --num_train_epochs 1).
set -euo pipefail
cd "$(dirname "$0")"

usage="usage: ./run.sh {prepare|pretrain|finetune|grpo|evaluate|export|demo} [--quick] [overrides...]"
stage=${1:?$usage}
shift
quick=()
case $stage in
  prepare)  cmd=(python -m abcgpt.prepare --config configs/prepare.yaml) ;;
  pretrain) cmd=(trl sft --config configs/pretrain.yaml);  quick=(--max_steps 300 --eval_steps 150) ;;
  finetune) cmd=(trl sft --config configs/finetune.yaml);  quick=(--max_steps 200 --eval_steps 100) ;;
  grpo)     cmd=(trl grpo --config configs/grpo.yaml);     quick=(--max_steps 20) ;;
  evaluate) cmd=(python -m abcgpt.evaluate --config configs/evaluate.yaml); quick=(--num_samples 20) ;;
  export)   cmd=(python -m abcgpt.export_gguf --config configs/export.yaml) ;;
  demo)     cmd=(python -m abcgpt.app --config configs/app.yaml) ;;
  *) echo "$usage" >&2; exit 1 ;;
esac
if [[ ${1:-} == --quick ]]; then shift; else quick=(); fi

set -x
exec uv run "${cmd[@]}" "${quick[@]}" "$@"
