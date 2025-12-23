#!/bin/bash
# Script to train Qwen3-4B with TRUE ICL (In-Context Learning) setup with SAME PARITY evaluation (0, 2, 4-shot)
# NOTE: This version uses max_new_tokens=256 (standard setting)
# NOTE: Training on SIZES 3-9, evaluation on SIZE 6 ONLY
# NOTE: Training: Mixed format with 0-shot, 2-shot, and 4-shot examples (equal proportion, 1/3 each)
# NOTE: Training format matches inference format (not alpaca)
# Training: Mixed format
#   Format: "Find shortest path from node X to node Y"
#   Sizes: 3, 4, 5, 6, 7, 8, 9
#   Distribution: 1/3 zero-shot, 1/3 2-shot, 1/3 4-shot (shuffled together)
# Evaluation: Tests if model can learn NEW task format from examples
#   NEW Format: "Find shortest path from node X to node Y only using nodes of the same parity"
#   Size: 6 ONLY
#   Tests 0-shot, 2-shot, and 4-shot prompts
#   IMPORTANT: Path must only use nodes with same parity as start node

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

export CUDA_VISIBLE_DEVICES=2

echo "========================================"
echo "TRUE ICL (IN-CONTEXT LEARNING) TRAINING"
echo "WITH SAME PARITY EVALUATION (0, 2, 4-shot)"
echo "TRAIN ON SIZES 3-9, TEST ON SIZE 6 (256 TOKENS)"
echo "MIXED TRAINING: 1/3 ZERO-SHOT, 1/3 2-SHOT, 1/3 4-SHOT"
echo "========================================"
echo ""
echo "📚 TRAINING:"
echo "   - Format: Mixed (1/3 zero-shot, 1/3 2-shot, 1/3 4-shot)"
echo "   - Sizes: 3, 4, 5, 6, 7, 8, 9"
echo "   - Samples: 10,000 total (distributed across sizes)"
echo "   - Task: Find shortest path from node X to node Y"
echo "   - Purpose: Model learns to solve shortest path problems across multiple sizes"
echo "   - Format: Uses inference format (not alpaca)"
echo "   - ⚠️  Training data is SHUFFLED (graph sizes and prompt types mixed together)"
echo ""
echo "📊 EVALUATION (NEW TASK FORMAT):"
echo "   - Tests if model can learn NEW task format from examples"
echo "   - Size: 6 ONLY - 50 queries total (from 700 total samples)"
echo "   - 650 remaining samples used as example pool for ICL"
echo "   - Total completions: 50 queries × 3 modes = 150 completions"
echo ""
echo "🧪 NEW TASK FORMAT:"
echo "   - Training: 'Find shortest path from node X to node Y'"
echo "   - Evaluation: 'Find shortest path from node X to node Y only using nodes of the same parity'"
echo "   - Constraint: Path must only use nodes with same parity (even/odd) as start node"
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
python unsloth-cli-true-icl-same-parity-024shot-train3to9-test6-mixed-training-256tokens.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_icl_true_same_parity_train3to9_test6_mixed_training.json" \
    --eval_dataset "spath_data_gen/data/val_icl_true_same_parity_024shot_train3to9_test6_mixed_training.json" \
    --eval_strategy "steps" \
    --eval_steps 200 \
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
    --wandb_run_name "Spath_true_icl_same_parity_024shot_train3to9_test6_mixed_training_256tokens" \
    --save_model \
    --save_path "Qwen-3-4b-true-icl-same-parity-024shot-train3to9-test6-mixed-training-256tokens" \
    --save_method "merged_16bit" \
    --logging_steps 1 \
    --dataloader_shuffle

