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

export CUDA_VISIBLE_DEVICES=6

echo "========================================"
echo "CURRICULUM FEW-SHOT TRAINING"
echo "Format: 1 example + 1 query per batch"
echo "Training: 3→4 nodes (40K each = 80K total, 2x prompts)"
echo "Eval: 3,4,5,6 nodes (comprehensive evaluation)"
echo "Includes: Dijkstra's algorithm reasoning"
echo "Total: ~40,000 batches (80,000 graphs / 2 per batch)"
echo "Steps: ~1000 total (batch_size=10, grad_accum=4 → effective=40)"
echo "Eval: Every 25 steps (~40 evals total)"
echo "Save: Every 50 steps (~20 saves total)"
echo "========================================"
echo ""
echo "⚠️  CURRICULUM LEARNING ACTIVE"
echo "Training data is in SEQUENTIAL ORDER (easy→hard)"
echo "Data will NOT be shuffled to preserve curriculum!"
echo ""

# Use accelerate for distributed training across 4 GPUs
python unsloth-cli-curriculum-icl.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_curriculum_fewshot.json" \
    --eval_dataset "spath_data_gen/data/val_curriculum_fewshot.json" \
    --eval_strategy "steps" \
    --eval_steps 25 \
    --r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.0 \
    --use_gradient_checkpointing "unsloth" \
    --per_device_train_batch_size 10 \
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
    --wandb_run_name "Spath_curriculum_fewshot" \
    --save_model \
    --save_path "Qwen-3-4b-1202434" \
    --save_method "merged_16bit" \
    --logging_steps 1