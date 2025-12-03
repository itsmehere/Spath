#!/bin/bash
# Script to train Qwen3-4B with curriculum few-shot dataset
# Uses curriculum learning (3→7 nodes) with 1 example + 1 query format
# Includes Dijkstra's algorithm reasoning in outputs

# Activate conda environment - use absolute path for tmux compatibility
source /nas/ucb/aaryanchandna/anaconda3/etc/profile.d/conda.sh
conda activate spath

# Verify conda environment is activated
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python

# Set TMPDIR to use /nas/ucb filesystem (has space) instead of /tmp (root filesystem is full)
export TMPDIR=/nas/ucb/aaryanchandna/code/SPath/tmp
mkdir -p "$TMPDIR"
echo "Using TMPDIR: $TMPDIR (on /nas/ucb filesystem with available space)"

export CUDA_VISIBLE_DEVICES=7

# Use accelerate for distributed training across 4 GPUs
python unsloth-cli-curriculum-icl.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_icl_5_shuffled.json" \
    --eval_dataset "spath_data_gen/data/val_icl_5_shuffled.json" \
    --eval_strategy "steps" \
    --eval_steps 25 \
    --r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.0 \
    --use_gradient_checkpointing "unsloth" \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --max_steps 1000 \
    --learning_rate 4e-5 \
    --optim "adamw_8bit" \
    --output_dir "Models" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --wandb_entity "spath" \
    --wandb_project "Spath" \
    --wandb_run_name "Spath_icl_5_shuffled" \
    --save_model \
    --save_path "Qwen3-4b-icl5-shuffled" \
    --save_method "merged_16bit" \
    --logging_steps 1