#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export JEV_TRAIN="${JEV_TRAIN:-$PWD/data/prepared/train.jsonl}"
export TOKENIZERS_PARALLELISM=false
swift sft \
  --model "${MODEL:-Qwen/Qwen3.5-0.8B-Base}" --use_hf true \
  --external_plugins jev_alpha/plugin.py --template jev_decision \
  --dataset jev_train --loss_type jev_options --callbacks jev_epoch \
  --lazy_tokenize true --remove_unused_columns false --strict true \
  --dataloader_num_workers 0 --use_logits_to_keep false \
  --packing false --padding_free false --split_dataset_ratio 0 \
  --tuner_type lora --target_modules all-linear --lora_rank 8 --lora_alpha 16 \
  --torch_dtype bfloat16 --attn_impl sdpa --max_length 4096 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 0.0001 --num_train_epochs 1 --gradient_checkpointing true \
  --eval_strategy no --save_strategy epoch --logging_steps 10 \
  --report_to none --seed 42 --output_dir runs/boolq --add_version true "$@"
