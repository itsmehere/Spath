#!/bin/bash
# Curriculum learning training script for shortest path model
# Data is ordered from easy (3-4 nodes) to hard (6-8 nodes)

source /nas/ucb/aaryanchandna/anaconda3/etc/profile.d/conda.sh
conda activate spath

echo "Active conda environment: $CONDA_DEFAULT_ENV"
which python

export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "CURRICULUM LEARNING TRAINING"
echo "Stage 1: 3-4 nodes (15K samples)"
echo "Stage 2: 5-6 nodes (15K samples)"  
echo "Stage 3: 7-8 nodes (20K samples)"
echo "Stage 4: 9-10 nodes (20K samples)"
echo "Stage 5: 11-12 nodes (15K samples)"
echo "Stage 6: 13-15 nodes (15K samples)"
echo "Total: 100K samples, 3→15 nodes"
echo "Training: 98,000 samples | Validation: 2,000 samples"
echo "Max steps: 6,125 (1 epoch = 6,125 steps with batch size 16)"
echo "Eval: Every 100 steps (2 samples per stage = 12 total per eval)"
echo "========================================"
echo ""
echo "⚠️  CURRICULUM LEARNING ACTIVE"
echo "Training data is in SEQUENTIAL ORDER (easy→hard)"
echo "Data will NOT be shuffled to preserve curriculum!"
echo "Each curriculum stage will be seen EXACTLY ONCE"
echo ""

# Single GPU training with 4-bit quantization
python unsloth-cli.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_curriculum.json" \
    --eval_dataset "spath_data_gen/data/val_curriculum.json" \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --use_gradient_checkpointing "unsloth" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --max_steps 6125 \
    --learning_rate 2e-4 \
    --optim "adamw_8bit" \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --output_dir "Models_curriculum" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --report_to "wandb" \
    --wandb_entity "spath" \
    --wandb_project "Spath" \
    --wandb_run_name "Spath_curriculum_v1" \
    --save_model \
    --save_path "Qwen3-4B-Spath-Curriculum" \
    --save_method "merged_16bit" \
    --logging_steps 10


