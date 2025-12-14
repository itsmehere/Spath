#!/bin/bash
# Script to train gpt-2-medium with Spath data generation dataset

# Activate conda environment
# Try to find conda in common locations
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    # Conda is in PATH, try to initialize
    eval "$(conda shell.bash hook)"
else
    echo "Error: Could not find conda. Please ensure conda is installed and in your PATH."
    exit 1
fi

# Activate the spath environment
conda activate spath

# Verify conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "spath" ]; then
    echo "Warning: Conda environment 'spath' may not be activated correctly."
    echo "Current environment: $CONDA_DEFAULT_ENV"
    echo "Attempting to continue anyway..."
else
    echo "✓ Conda environment 'spath' activated successfully"
    echo "Python path: $(which python)"
fi

export CUDA_VISIBLE_DEVICES=7

python unsloth-cli-kevin.py \
    --model_name "gpt2-medium" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_len_baseline.json" \
    --eval_dataset "spath_data_gen/data/val_len_baseline.json" \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --r 128 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --use_gradient_checkpointing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 2 \
    --warmup_steps 5 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --optim "adamw_8bit" \
    --output_dir "Models" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --report_to "wandb" \
    --wandb_entity "spath" \
    --wandb_project "Spath" \
    --wandb_run_name "Spath_len_fast_zero_shot" \
    --save_model \
    --save_path "gpt2-medium-Spath" \
    --save_method "merged_16bit" \
    --logging_steps 1 \
    --max_eval_samples -1 \

