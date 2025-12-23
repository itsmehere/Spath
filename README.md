# Constrained Shortest Path Learning with In-Context Learning

This repository contains code for training and evaluating language models on constrained shortest path problems using in-context learning. We explore two experimental paradigms: **Hidden Rule Transfer** and **Dijkstra Finetuning**.

## Table of Contents

- [Setup](#setup)
- [Model Checkpoints](#model-checkpoints)
- [Replicating Experiments](#replicating-experiments)
  - [Hidden Rule Transfer Experiments](#hidden-rule-transfer-experiments)
  - [Dijkstra Finetuning Experiments](#dijkstra-finetuning-experiments)
- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)

## Setup

1. **Create conda environment:**
   ```bash
   conda env create -f spath-env.yaml
   conda activate spath
   ```

2. **Verify installation:**
   ```bash
   python -c "import torch; import unsloth; print('Setup complete!')"
   ```

## Model Checkpoints

All trained model checkpoints are available on Hugging Face under the `basebala` organization:

### Hidden Rule Transfer Models

- **Three Rules Training (Variable Sizes 3-9, Test Size 6):**
  - [Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-256tokens](https://huggingface.co/basebala/Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-256tokens)
  - [Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-0shot-4shot-6shot-256tokens](https://huggingface.co/basebala/Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-0shot-4shot-6shot-256tokens)

- **Three Rules Training (Fixed Size 6):**
  - [Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-256tokens](https://huggingface.co/basebala/Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-256tokens)

- **Single Rule Training (Max Edge Weight, Size 6):**
  - [Qwen-3-4b-hidden-rule-max-edge-weight-training-avoid-eval-256tokens](https://huggingface.co/basebala/Qwen-3-4b-hidden-rule-max-edge-weight-training-avoid-eval-256tokens)

- **Variable Shot Training:**
  - [Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-variable-shot-256tokens](https://huggingface.co/basebala/Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-variable-shot-256tokens)

### Dijkstra Finetuning Models

- **Same Parity Constraint (Variable Sizes):**
  - [Qwen-3-4b-true-icl-same-parity-024shot-train3to9-test6-mixed-training-256tokens](https://huggingface.co/basebala/Qwen-3-4b-true-icl-same-parity-024shot-train3to9-test6-mixed-training-256tokens)

- **Same Parity Constraint (Size 6 Only):**
  - [Qwen-3-4b-true-icl-same-parity-024shot-size6-only-256tokens](https://huggingface.co/basebala/Qwen-3-4b-true-icl-same-parity-024shot-size6-only-256tokens)

- **Avoid Node Constraint (Size 6):**
  - [Qwen-3-4b-true-icl-avoid-024shot-size6-only-256tokens](https://huggingface.co/basebala/Qwen-3-4b-true-icl-avoid-024shot-size6-only-256tokens)

- **Set Target Constraint (Size 6):**
  - [Qwen-3-4b-true-icl-set-target-024shot-size6-only-256tokens](https://huggingface.co/basebala/Qwen-3-4b-true-icl-set-target-024shot-size6-only-256tokens)

- **Avoid Node Constraint (Size 10, trained on Size 6):**
  - [Qwen-3-4b-true-icl-avoid-024shot-size10-only-256tokens](https://huggingface.co/basebala/Qwen-3-4b-true-icl-avoid-024shot-size10-only-256tokens)

## Replicating Experiments

### Hidden Rule Transfer Experiments

In hidden rule transfer experiments, models are trained on constrained shortest path problems where the constraint (hidden rule) is never explicitly stated. During training, models see 4-shot prompts where all examples share the same hidden rule (maximum edge weight, no adjacent index edges, or monotonic path), but the rule is never mentioned. At evaluation time, we test whether models can transfer their rule-inference capability to a novel constraint (avoid node) that was never seen during training.

#### Experiment 1: Variable Sizes Training (3-9 nodes, test on size 6)

**Results:** 4-shot: 22%, 0-shot: 6%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_hidden_rule_three_rules_training_avoid_eval_train3to9_test6_256tokens.py
   ```

2. **Train the model:**
   ```bash
   bash sft-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-256tokens.sh
   ```

   This will:
   - Train on 10,000 prompts with 4-shot format
   - Use 3 hidden rules (33.33% each): max edge weight, no adjacent index edges, monotonic path
   - Train on graph sizes 3-9 (randomly selected per prompt)
   - Evaluate on size 6 graphs only with "avoid node" constraint
   - Save model to `Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-256tokens/`

#### Experiment 2: Fixed Size Training (size 6 only)

**Results:** 4-shot: 38%, 0-shot: 10%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_hidden_rule_three_rules_training_avoid_eval_256tokens.py
   ```

2. **Train the model:**
   ```bash
   bash sft-hidden-rule-three-rules-training-avoid-eval-256tokens.sh
   ```

   This trains exclusively on size 6 graphs, matching the evaluation size.

#### Experiment 3: Single Rule Training (Max Edge Weight only)

**Results:** 4-shot: 26%, 0-shot: 10%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_hidden_rule_max_edge_weight_training_avoid_eval_256tokens.py
   ```

2. **Train the model:**
   ```bash
   bash sft-hidden-rule-max-edge-weight-training-avoid-eval-256tokens.sh
   ```

   This trains on only one hidden rule (maximum edge weight) instead of three.

#### Experiment 4: Variable Shot Training

**Results:** 4-shot: 34%, 0-shot: 34%, 6-shot: 2%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_hidden_rule_three_rules_training_avoid_eval_train3to9_test6_variable_shot_256tokens.py
   ```

2. **Train the model:**
   ```bash
   bash sft-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-variable-shot-256tokens.sh
   ```

   This uses variable-shot training where prompts contain 0, 2, 4, or 6 examples (distributed evenly).

### Dijkstra Finetuning Experiments

In dijkstra finetuning experiments, models are trained on standard shortest path problems (no constraints) using single examples. At evaluation time, we test whether models can learn new constraint-based task formats from examples. We construct 0-shot, 2-shot, and 4-shot prompts where examples demonstrate a constraint (same parity, avoid node, or set target). Unlike hidden rule transfer, the constraint is explicitly stated in the problem description.

#### Experiment 1: Same Parity Constraint (Variable Sizes)

**Results:** 4-shot: 62%, 2-shot: 54%, 0-shot: 29%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_icl_true_same_parity_024shot_train3to9_test6_mixed_training.py
   ```

2. **Train the model:**
   ```bash
   bash sft-true-icl-same-parity-024shot-train3to9-test6-mixed-training-256tokens.sh
   ```

   This trains on sizes {3,4,6,8,9,10} and evaluates on unseen sizes (5 and 7).

#### Experiment 2: Same Parity Constraint (Size 6 Only)

**Results:** 4-shot: 74%, 2-shot: 46%, 0-shot: 24%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_icl_true_same_parity_024shot_size6_only.py
   ```

2. **Train the model:**
   ```bash
   bash sft-true-icl-same-parity-024shot-size6-only-256tokens.sh
   ```

   This trains and evaluates exclusively on size 6 graphs.

#### Experiment 3: Avoid Node Constraint (Size 6)

**Results:** 4-shot: 46%, 2-shot: 46%, 0-shot: 28%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_icl_true_avoid_024shot_size6_only.py
   ```

2. **Train the model:**
   ```bash
   bash sft-true-icl-avoid-024shot-size6-only-256tokens.sh
   ```

#### Experiment 4: Set Target Constraint (Size 6)

**Results:** 4-shot: 38%, 2-shot: 44%, 0-shot: 28%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_icl_true_set_target_024shot_size6_only.py
   ```

2. **Train the model:**
   ```bash
   bash sft-true-icl-set-target-024shot-size6-only-256tokens.sh
   ```

#### Experiment 5: Avoid Node Constraint (Size 10, trained on Size 6)

**Results:** 4-shot: 0%, 2-shot: 2%, 0-shot: 16%

1. **Generate training and evaluation data:**
   ```bash
   cd spath_data_gen
   python data_gen_icl_true_avoid_024shot_size10_only.py
   ```

2. **Train the model:**
   ```bash
   bash sft-true-icl-avoid-024shot-size10-only-256tokens.sh
   ```

   This tests size generalization by training on size 6 and evaluating on size 10.

## Data Generation

All data generation scripts are in the `spath_data_gen/` directory. Each script generates:
- **Training data:** JSON files with training prompts/examples
- **Evaluation data:** JSON files with evaluation queries

Data generation scripts follow a naming convention:
- `data_gen_hidden_rule_*` - For hidden rule transfer experiments
- `data_gen_icl_true_*` - For dijkstra finetuning experiments

Key parameters in data generation:
- `TRAIN_NODE_SIZES`: Graph sizes used for training
- `EVAL_NODE_SIZE`: Graph size used for evaluation
- `TRAIN_PROMPTS`: Number of training prompts/examples
- `VAL_QUERIES`: Number of evaluation queries
- `SEED`: Random seed for reproducibility (default: 182)

## Training

Training scripts are shell scripts (`.sh` files) that:
1. Activate the conda environment
2. Set CUDA device
3. Call the corresponding Python training script

All training scripts use:
- **Model:** `unsloth/Qwen3-4B`
- **LoRA:** rank=64, alpha=64, dropout=0.0
- **Quantization:** 4-bit
- **Batch size:** 10 per device with gradient accumulation of 4 (effective batch size 40)
- **Learning rate:** 4e-5
- **Optimizer:** AdamW 8-bit
- **Max steps:** 1000
- **Max sequence length:** 2048 tokens
- **Evaluation:** Every 200 steps, max_new_tokens=256

Training logs are sent to Weights & Biases (wandb). Make sure to set your wandb credentials:
```bash
export WANDB_API_KEY="your_wandb_key"
```

Models are saved to directories matching the experiment name (e.g., `Qwen-3-4b-hidden-rule-three-rules-training-avoid-eval-train3to9-test6-256tokens/`).

## Evaluation

Evaluation is performed automatically during training at specified intervals. The evaluation computes:
- **Exact match accuracy:** Predicted path must exactly match ground truth path
- **Per-shot accuracy:** Separate metrics for 0-shot, 2-shot, 4-shot, and 6-shot (when applicable)
- **Per-constraint accuracy:** Accuracy broken down by constraint type (for hidden rule transfer)

To evaluate a saved model separately, you can load it and run inference:

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Run inference
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your-paper,
  title={Constrained Shortest Path Learning with In-Context Learning},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

[Specify your license here]
