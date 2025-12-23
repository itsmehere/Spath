#!/bin/bash
# Script to train Qwen3-4B with HIDDEN RULE learning pipeline
# NOTE: Training uses 4-shot prompts with hidden "max edge weight" rule
# NOTE: Evaluation uses 0-shot/4-shot with hidden "avoid node" rule
# NOTE: All graphs are size 6
# NOTE: max_new_tokens=256

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

export CUDA_VISIBLE_DEVICES=5

echo "========================================"
echo "HIDDEN RULE LEARNING PIPELINE"
echo "TRAINING: 4-shot with hidden 'max edge weight' rule"
echo "EVALUATION: 0-shot/4-shot with hidden 'avoid node' rule"
echo "SIZE 6 GRAPHS, 256 TOKENS"
echo "========================================"
echo ""
echo "📚 TRAINING:"
echo "   - Format: Always 4-shot prompts"
echo "   - Hidden rule: 'maximum edge weight' (random max weight per prompt, not stated)"
echo "   - All 4 examples in a prompt share the same max edge weight"
echo "   - Model must infer rule from examples"
echo "   - true path != constrained path"
echo "   - Size: 6 ONLY"
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
echo "   - Training: Model learns to infer 'max edge weight' rule from 4 examples"
echo "   - Evaluation: Model must infer 'avoid node' rule from examples (or zero-shot)"
echo "   - Model never explicitly told the rule - must learn from patterns"
echo ""
echo "📈 METRICS:"
echo "   - accuracy/val_0shot: Zero-shot accuracy on odd/even target rule"
echo "   - accuracy/val_4shot: 4-shot accuracy on odd/even target rule"
echo "   - Per-parity accuracy for both modes"
echo "========================================"
echo ""

# Run training
python unsloth-cli-hidden-rule-max-edge-weight-training-avoid-eval-256tokens.py \
    --model_name "unsloth/Qwen3-4B" \
    --max_seq_length 2048 \
    --load_in_4bit \
    --dataset "spath_data_gen/data/train_hidden_rule_max_edge_weight_4shot_size6_256tokens.json" \
    --eval_dataset "spath_data_gen/data/val_hidden_rule_avoid_0shot_4shot_size6_256tokens.json" \
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
    --wandb_run_name "Spath_hidden_rule_max_edge_weight_training_avoid_eval_256tokens" \
    --save_model \
    --save_path "Qwen-3-4b-hidden-rule-max-edge-weight-training-avoid-eval-256tokens" \
    --save_method "merged_16bit" \
    --logging_steps 1 \
    --dataloader_shuffle

