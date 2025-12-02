#!/bin/bash
# Script to train Qwen3-4B with curriculum few-shot dataset
# Uses curriculum learning (3→7 nodes) with 4 examples + 1 query format
# Includes Dijkstra's algorithm reasoning in outputs

# Activate conda environment - use absolute path for tmux compatibility
source /nas/ucb/aaryanchandna/anaconda3/etc/profile.d/conda.sh
conda activate spath

# Verify conda environment is activated
echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python

export CUDA_VISIBLE_DEVICES=6

echo "========================================"
echo "CURRICULUM FEW-SHOT TRAINING"
echo "Format: 4 examples + 1 query per batch"
echo "Stages: 3→4→5→6→7 nodes (curriculum order)"
echo "Includes: Dijkstra's algorithm reasoning"
echo "Total: ~10,780 batches (53,900 graphs / 5 per batch)"
echo "Steps: 270 per epoch (batch_size=10, grad_accum=4 → effective=40)"
echo "Eval: Every 25 steps (~11 evals per epoch)"
echo "Save: Every 100 steps (~3 saves per epoch)"
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
    --warmup_steps 5 \
    --max_steps 270 \
    --learning_rate 2e-4 \
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
    --save_path "Qwen3-1.7b-Spath" \
    --save_method "merged_16bit" \
    --logging_steps 1

