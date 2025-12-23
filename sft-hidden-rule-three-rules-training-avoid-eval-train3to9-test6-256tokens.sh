#!/bin/bash
# Script to train Qwen3-4B with HIDDEN RULE learning pipeline
# NOTE: Training uses 4-shot prompts with 3 hidden rules (33.33% each): max edge weight, no adjacent index, monotonic path
# NOTE: Evaluation uses 0-shot/4-shot with hidden "avoid node" rule
# NOTE: Training on sizes 3-9, testing on size 6 only
# NOTE: max_new_tokens=256 at evaluation

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
echo "HIDDEN RULE LEARNING PIPELINE"
echo "TRAINING: 4-shot with 3 hidden rules (33.33% each)"
echo "EVALUATION: 0-shot/4-shot with hidden 'avoid node' rule"
echo "TRAINING SIZES: 3-9, TEST SIZE: 6 ONLY, 256 TOKENS"
echo "========================================"
echo ""
echo "📚 TRAINING:"
echo "   - Format: Always 4-shot prompts"
echo "   - Hidden rules: 33.33% 'maximum edge weight', 33.33% 'no adjacent index edges', 33.33% 'monotonic path'"
echo "   - All 4 examples in a prompt share the same hidden rule"
echo "   - Model must infer rule from examples"
echo "   - true path != constrained path"
echo "   - Sizes: 3-9 (randomly selected per prompt)"
echo "   - Format: Uses inference format (not alpaca)"
echo ""
echo "📊 EVALUATION:"
echo "   - Hidden rule: 'avoid some node' (random node, not stated)"
echo "   - Modes: 0-shot or 4-shot at runtime"
echo "   - Same queries for 0-shot and 4-shot"
echo "   - true path != constrained path"
echo "   - Size: 6 ONLY"
echo "   - Format matches training format"
echo "   - max_new_tokens: 256"
echo ""
echo "🧪 HIDDEN RULES:"
echo "   - Training: Model learns to infer 3 different rules from 4 examples each"
echo "   - Rules: max edge weight, no adjacent index edges (i to i+1 or i+1 to i), monotonic path (indices only increase or only decrease)"
echo "   - Evaluation: Model must infer 'avoid node' rule from examples (or zero-shot)"
echo "   - Model never explicitly told the rule - must learn from patterns"
echo ""
echo "📈 METRICS:"
echo "   - accuracy/val_0shot: Zero-shot accuracy on avoid node rule"
echo "   - accuracy/val_4shot: 4-shot accuracy on avoid node rule"
echo "   - Per-avoid-node accuracy for both modes"
echo "========================================"
echo ""

# Run training
python unsloth-cli-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-256tokens.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_hidden_rule_three_rules_4shot_train3to9_test6_256tokens.json" \
    --eval_dataset "spath_data_gen/data/val_hidden_rule_avoid_0shot_4shot_test6_256tokens.json" \
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
    --wandb_run_name "Spath_hidden_rule_three_rules_training_avoid_eval_train3to9_test6_256tokens" \
    --save_model \
    --save_path "Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-256tokens" \
    --save_method "merged_16bit" \
    --logging_steps 1 \
    --dataloader_shuffle
