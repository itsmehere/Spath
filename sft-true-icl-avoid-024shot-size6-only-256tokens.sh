#!/bin/bash
# Script to train Qwen3-4B with TRUE ICL (In-Context Learning) setup with AVOID NODE evaluation (0, 2, 4-shot)
# NOTE: This version uses max_new_tokens=256 (standard setting)
# NOTE: Training and evaluation both on SIZE 6 ONLY
# Training: Single examples only (no few-shot format)
#   Format: "Find shortest path from node X to node Y"
#   Size: 6 ONLY
# Evaluation: Tests if model can learn NEW task format from examples
#   NEW Format: "Find shortest path from node X to node Y that avoids node Z"
#   Size: 6 ONLY
#   Tests 0-shot, 2-shot, and 4-shot prompts

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

export CUDA_VISIBLE_DEVICES=4

echo "========================================"
echo "TRUE ICL (IN-CONTEXT LEARNING) TRAINING"
echo "WITH AVOID NODE EVALUATION (0, 2, 4-shot)"
echo "SIZE 6 ONLY (256 TOKENS)"
echo "========================================"
echo ""
echo "📚 TRAINING:"
echo "   - Format: Single examples only (no few-shot)"
echo "   - Size: 6 ONLY"
echo "   - Samples: 10,000 total"
echo "   - Task: Find shortest path from node X to node Y"
echo "   - Purpose: Model learns to solve shortest path problems"
echo ""
echo "📊 EVALUATION (NEW TASK FORMAT):"
echo "   - Tests if model can learn NEW task format from examples"
echo "   - Size: 6 ONLY - 50 queries total (from 700 total samples)"
echo "   - 600 remaining samples used as example pool for ICL"
echo "   - Total completions: 50 queries × 3 modes = 150 completions"
echo ""
echo "🧪 NEW TASK FORMAT:"
echo "   - Training: 'Find shortest path from node X to node Y'"
echo "   - Evaluation: 'Find shortest path from node X to node Y that avoids node Z'"
echo "   - Avoid node: Must be different from start and end, path must exist"
echo "   - Model must learn this new format from examples at inference time"
echo ""
echo "🧪 EVALUATION MODES:"
echo "   1. ZERO-SHOT: Just the query, no examples"
echo "   2. 2-SHOT ICL: 2 examples + 1 query (constructed at inference)"
echo "   3. 4-SHOT ICL: 4 examples + 1 query (constructed at inference)"
echo ""
echo "📈 METRICS:"
echo "   - accuracy/val_0shot: Zero-shot accuracy on new task format"
echo "   - accuracy/val_2shot: 2-shot ICL accuracy on new task format"
echo "   - accuracy/val_4shot: 4-shot ICL accuracy on new task format"
echo "   - Per-size accuracy for all three modes"
echo "   - Tests true in-context learning: can model adapt to new format?"
echo "========================================"
echo ""

# Run training
python unsloth-cli-true-icl-avoid-024shot-size6-only-256tokens.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_icl_true_avoid_size6_only.json" \
    --eval_dataset "spath_data_gen/data/val_icl_true_avoid_024shot_size6_only.json" \
    --eval_strategy "steps" \
    --eval_steps 100 \
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
    --wandb_run_name "Spath_true_icl_avoid_024shot_size6_only_256tokens" \
    --save_model \
    --save_path "Qwen-3-4b-true-icl-avoid-024shot-size6-only-256tokens" \
    --save_method "merged_16bit" \
    --logging_steps 1

