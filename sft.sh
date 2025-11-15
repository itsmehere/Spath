#!/bin/bash
# Script to train Qwen3-1.7B with Spath data generation dataset

export CUDA_VISIBLE_DEVICES=7

python unsloth-cli.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train.json" \
    --eval_dataset "spath_data_gen/data/val.json" \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.0 \
    --use_gradient_checkpointing "unsloth" \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 5 \
    --max_steps 100000 \
    --learning_rate 2e-4 \
    --optim "adamw_8bit" \
    --output_dir "Models" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --report_to "wandb" \
    --wandb_entity "spath" \
    --wandb_project "Spath" \
    --wandb_run_name "Spath15" \
    --save_model \
    --save_path "Qwen3-1.7b-Spath" \
    --save_method "merged_16bit" \
    --logging_steps 1

