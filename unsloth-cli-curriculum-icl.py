#!/usr/bin/env python3

"""
🦥 Starter Script for Fine-Tuning FastLanguageModel with Unsloth

This script is designed as a starting point for fine-tuning your models using unsloth.
It includes configurable options for model loading, PEFT parameters, training arguments, 
and model saving/pushing functionalities.

You will likely want to customize this script to suit your specific use case 
and requirements.

Here are a few suggestions for customization:
    - Modify the dataset loading and preprocessing steps to match your data.
    - Customize the model saving and pushing configurations.

Usage: (most of the options have valid default values this is an extended example for demonstration purposes)
    python unsloth-cli.py --model_name "unsloth/llama-3-8b" --max_seq_length 8192 --dtype None --load_in_4bit \
    --r 64 --lora_alpha 32 --lora_dropout 0.1 --bias "none" --use_gradient_checkpointing "unsloth" \
    --random_state 3407 --use_rslora --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --warmup_steps 5 --max_steps 400 --learning_rate 2e-6 --logging_steps 1 --optim "adamw_8bit" \
    --weight_decay 0.005 --lr_scheduler_type "linear" --seed 3407 --output_dir "outputs" \
    --report_to "tensorboard" --save_model --save_path "model" --quantization_method "f16" \
    --push_model --hub_path "hf/model" --hub_token "your_hf_token"

To see a full list of configurable options, use:
    python unsloth-cli.py --help

Happy fine-tuning!
"""

import argparse
import os
import random
import json
import re


def run(args):
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers.utils import strtobool
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    import logging

    logging.getLogger("hf-to-gguf").setLevel(logging.WARNING)

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = args.dtype,
        load_in_4bit = args.load_in_4bit,
    )

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.r,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = args.bias,
        use_gradient_checkpointing = args.use_gradient_checkpointing,
        random_state = args.random_state,
        use_rslora = args.use_rslora,
        loftq_config = args.loftq_config,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    use_modelscope = strtobool(os.environ.get("UNSLOTH_USE_MODELSCOPE", "False"))
    if use_modelscope:
        from modelscope import MsDataset

        dataset = MsDataset.load(args.dataset, split = "train")
    else:
        # Load and format dataset
        # Support local JSON files
        if args.dataset.endswith('.json') or os.path.isfile(args.dataset):
            dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
        else:
            dataset = load_dataset(args.dataset, split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
    print("Training data is formatted and ready!")

    # Load evaluation dataset if provided
    eval_dataset = None
    if args.eval_dataset:
        if args.eval_dataset.endswith('.json') or os.path.isfile(args.eval_dataset):
            eval_dataset = load_dataset("json", data_files={"train": args.eval_dataset}, split="train")
        else:
            eval_dataset = load_dataset(args.eval_dataset, split = "train")
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)
        print("Evaluation data is formatted and ready!")

    # Configure wandb if using it
    wandb_initialized = False
    if args.report_to == "wandb" or (isinstance(args.report_to, list) and "wandb" in args.report_to):
        if args.wandb_entity or args.wandb_project or args.wandb_run_name:
            import wandb
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.wandb_run_name,
                resume="allow",
            )
            wandb_initialized = True

    # Configure training arguments
    training_args = SFTConfig(
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = args.logging_steps,
        optim = args.optim,
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        seed = args.seed,
        output_dir = args.output_dir,
        report_to = args.report_to,
        max_length = args.max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        # Evaluation settings
        # Disable standard Trainer evaluation - we use custom callback for evaluation
        # This avoids processing the entire eval dataset (4320 samples) unnecessarily
        eval_strategy = "no",  # Disable Trainer's standard eval, use custom callback instead
        eval_steps = None,
        per_device_eval_batch_size = None,
        # Checkpoint saving settings
        save_strategy = args.save_strategy,
        save_steps = args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit = args.save_total_limit,
    )

    # Create custom callback for logging completions to wandb
    if wandb_initialized:
        from transformers import TrainerCallback
        import wandb
        
        class CompletionsCallback(TrainerCallback):
            def __init__(self, model, tokenizer, train_dataset, eval_dataset):
                self.model = model
                self.tokenizer = tokenizer
                self.train_dataset = train_dataset
                self.eval_dataset = eval_dataset
                self.logged_steps = set()  # Track which steps we've already logged to avoid duplicates
                self.trainer = None  # Will be set in on_train_begin
                
                # Check if dataset has curriculum_stage field
                has_curriculum = False
                if len(train_dataset) > 0:
                    sample = train_dataset[0]
                    if isinstance(sample, dict) and "curriculum_stage" in sample:
                        has_curriculum = True
                
                if has_curriculum:
                    # Don't sample from train dataset - only eval dataset
                    print("\n📊 Sampling completions for curriculum stages:")
                    self.train_samples = []  # No train dataset sampling
                    if eval_dataset and len(eval_dataset) > 0:
                        print("   Validation set:")
                        # Sample 50 per stage from eval dataset
                        self.eval_all_indices = self._sample_from_curriculum_stages(eval_dataset, samples_per_stage=50)
                        print(f"   Will log {len(self.eval_all_indices)} eval completions during evaluation (50 per stage)")
                        # No separate preview samples needed
                        self.eval_samples = []
                    else:
                        self.eval_samples = []
                        self.eval_all_indices = []
                    print(f"   Total: {len(self.train_samples)} train samples, {len(self.eval_all_indices) if hasattr(self, 'eval_all_indices') else 0} eval samples\n")
                else:
                    # Fallback: Sample 5 datapoints from train set
                    self.train_samples = random.sample(range(len(train_dataset)), min(5, len(train_dataset))) if len(train_dataset) > 0 else []
                    # Sample datapoints from eval set
                    self.eval_samples = list(range(min(5, len(eval_dataset)))) if eval_dataset and len(eval_dataset) > 0 else []
                    # Store only first 50 eval indices for evaluation (instead of all)
                    if eval_dataset and len(eval_dataset) > 0:
                        eval_limit = min(50, len(eval_dataset))
                        self.eval_all_indices = list(range(eval_limit))
                        print(f"   Will log {len(self.eval_all_indices)} eval completions during evaluation (limited to 50)")
                    else:
                        self.eval_all_indices = []
            
            def _sample_from_curriculum_stages(self, dataset, total_samples=25, samples_per_stage=None):
                """Sample examples from each curriculum stage.
                
                Args:
                    dataset: The dataset to sample from
                    total_samples: Total number of samples to return (distributed across stages)
                    samples_per_stage: If provided, samples this many per stage (ignores total_samples)
                """
                if len(dataset) == 0:
                    return []
                
                # Group indices by curriculum stage
                stage_indices = {}
                for idx in range(len(dataset)):
                    sample = dataset[idx]
                    if isinstance(sample, dict) and "curriculum_stage" in sample:
                        stage = sample["curriculum_stage"]
                        if stage not in stage_indices:
                            stage_indices[stage] = []
                        stage_indices[stage].append(idx)
                
                # If samples_per_stage is provided, use old behavior
                if samples_per_stage is not None:
                    samples = []
                    for stage, indices in sorted(stage_indices.items()):
                        if len(indices) >= samples_per_stage:
                            sampled = random.sample(indices, samples_per_stage)
                            samples.extend(sampled)
                            print(f"   {stage}: sampled {samples_per_stage} indices ({len(indices)} total)")
                        elif len(indices) > 0:
                            samples.extend(indices)
                            print(f"   {stage}: only {len(indices)} samples available, sampled all")
                    return samples
                
                # Otherwise, distribute total_samples across stages evenly
                num_stages = len(stage_indices)
                if num_stages == 0:
                    return []
                
                samples_per_stage_rounded = max(1, total_samples // num_stages)
                remainder = total_samples % num_stages
                
                samples = []
                for i, (stage, indices) in enumerate(sorted(stage_indices.items())):
                    # Distribute remainder across first few stages
                    num_to_sample = samples_per_stage_rounded + (1 if i < remainder else 0)
                    num_to_sample = min(num_to_sample, len(indices))
                    
                    if num_to_sample > 0:
                        sampled = random.sample(indices, num_to_sample)
                        samples.extend(sampled)
                        print(f"   {stage}: sampled {num_to_sample} indices ({len(indices)} total)")
                
                return samples
            
            def _extract_instruction_and_output(self, text):
                """Extract instruction/input and output from formatted text"""
                # The text is formatted as: instruction + input + response
                # We need to extract the instruction part and the output part
                if "### Instruction:" in text and "### Response:" in text:
                    parts = text.split("### Response:")
                    instruction = parts[0].replace("### Instruction:", "").replace("### Input:", "").strip()
                    output = parts[1].strip() if len(parts) > 1 else ""
                    return instruction, output
                return text
            
            def _parse_path_from_output(self, text):
                """Parse shortest path from model output - uses LAST set of brackets found"""
                # Find ALL matches of bracket patterns [0, 1, 2]
                all_matches = re.findall(r'\[([-\d,\s]+)\]', text)
                
                if all_matches:
                    # Use the LAST match (most likely the final answer)
                    last_match = all_matches[-1]
                    try:
                        numbers = [int(item.strip()) for item in last_match.split(",") if item.strip()]
                        # Only return if it looks like a valid path (at least 2 nodes)
                        if len(numbers) >= 2:
                            return numbers
                    except ValueError:
                        pass
                
                return None
            
            def _extract_ground_truth_path(self, output_text):
                """Extract ground truth shortest path from output text"""
                # Look for "Final Answer: The shortest path ... is [0, 1, 2]"
                match = re.search(r'is\s+(\[[\d,\s]+\])', output_text)
                if match:
                    try:
                        path = json.loads(match.group(1))
                        if isinstance(path, list):
                            return path
                    except (json.JSONDecodeError, ValueError):
                        pass
                # Try to find any JSON array
                match = re.search(r'\[([\d,\s]+)\]', output_text)
                if match:
                    try:
                        numbers = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
                        return numbers
                    except ValueError:
                        pass
                return None
            
            def _generate_prediction_batch(self, instruction_texts):
                """Generate predictions for multiple prompts in batch (much faster)"""
                try:
                    # Extract just the instruction part (before Response) for each prompt
                    prompts = []
                    for instruction_text in instruction_texts:
                        if "### Response:" in instruction_text:
                            prompt = instruction_text.split("### Response:")[0] + "### Response:"
                        else:
                            prompt = instruction_text
                        prompts.append(prompt)
                    
                    # Tokenize all prompts at once (batch processing)
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_seq_length - 50
                    ).to(self.model.device)
                    
                    # Set model to eval mode for generation
                    was_training = self.model.training
                    self.model.eval()
                    
                    pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    eos_token_id = self.tokenizer.eos_token_id
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,  # Reduced from 2048 - much faster, sufficient for path + reasoning
                            do_sample=True,  # Sampling with temperature for more diverse outputs
                            temperature=0.7,  # Moderate temperature for balanced randomness
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                        )
                    
                    # Restore training mode
                    if was_training:
                        self.model.train()
                    
                    # Decode all outputs
                    generated_texts = []
                    input_lengths = inputs['input_ids'].shape[1]
                    for i, output in enumerate(outputs):
                        # Decode only the new tokens (skip the prompt)
                        generated_text = self.tokenizer.decode(
                            output[input_lengths:], 
                            skip_special_tokens=True
                        )
                        generated_texts.append(generated_text.strip())
                    
                    return generated_texts
                except Exception as e:
                    # Return error for all if batch fails
                    return [f"Error: {str(e)}"] * len(instruction_texts)
            
            def _generate_prediction(self, instruction_text):
                """Generate prediction for single prompt (backward compatibility)"""
                results = self._generate_prediction_batch([instruction_text])
                return results[0] if results else "Error: Generation failed"
            
            def _log_completions_table(self, dataset, sample_indices, table_name, step):
                """Log completions table to wandb with accuracy calculation"""
                if not sample_indices:
                    return
                
                import wandb
                
                # Prepare all samples for batch generation
                all_texts = []
                all_samples = []
                for idx in sample_indices:
                    sample = dataset[idx]
                    text = sample.get("text", "")
                    all_texts.append(text)
                    all_samples.append((idx, sample))
                
                # Generate predictions in smaller batches to avoid OOM
                # Process in chunks of 10 to stay within memory bounds
                eval_batch_size = 10
                print(f"   Generating {len(all_texts)} predictions in batches of {eval_batch_size}...")
                prediction_texts = []
                for i in range(0, len(all_texts), eval_batch_size):
                    batch_texts = all_texts[i:i+eval_batch_size]
                    batch_predictions = self._generate_prediction_batch(batch_texts)
                    prediction_texts.extend(batch_predictions)
                    print(f"      Completed {min(i+eval_batch_size, len(all_texts))}/{len(all_texts)} predictions...")
                
                # Process results
                table_data = []
                correct_count = 0
                total_count = len(sample_indices)
                
                # Track accuracy per stage
                stage_stats = {}  # {stage_name: {"correct": count, "total": count}}
                
                for (idx, sample), prediction_text in zip(all_samples, prediction_texts):
                    text = sample.get("text", "")
                    instruction, ground_truth_text = self._extract_instruction_and_output(text)
                    
                    # Extract ground truth path from output text
                    ground_truth_path = self._extract_ground_truth_path(ground_truth_text)
                    
                    # Parse predicted path from model output
                    predicted_path = self._parse_path_from_output(prediction_text)
                    
                    # Check if prediction matches ground truth
                    is_correct = (predicted_path is not None and 
                                ground_truth_path is not None and 
                                predicted_path == ground_truth_path)
                    
                    if is_correct:
                        correct_count += 1
                    
                    # Get curriculum stage if available
                    stage = None
                    if isinstance(sample, dict) and "curriculum_stage" in sample:
                        stage = sample["curriculum_stage"]
                    # Use the raw stage name (e.g., "stage1_baby") for consistency
                    stage_name = stage if stage else "unknown"
                    
                    # Track per-stage statistics
                    if stage_name not in stage_stats:
                        stage_stats[stage_name] = {"correct": 0, "total": 0}
                    stage_stats[stage_name]["total"] += 1
                    if is_correct:
                        stage_stats[stage_name]["correct"] += 1
                    
                    table_data.append({
                        "stage": stage_name,
                        "sample_idx": idx,
                        "instruction": instruction,  # No truncation - show full text
                        "ground_truth_path": str(ground_truth_path) if ground_truth_path else "N/A",
                        "predicted_path": str(predicted_path) if predicted_path else "N/A",
                        "ground_truth_text": ground_truth_text,  # No truncation - show full text
                        "prediction_text": prediction_text,  # No truncation - show full text
                        "correct": is_correct,
                    })
                
                # Calculate overall accuracy
                accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
                
                # Calculate accuracy per stage
                stage_accuracies = {}
                
                # Build metrics dictionary - start with overall accuracy
                wandb_log_dict = {
                    f"accuracy/{table_name}": accuracy,
                    f"accuracy/{table_name}_count": f"{correct_count}/{total_count}",
                }
                
                # Add per-stage accuracy metrics
                for stage_name, stats in sorted(stage_stats.items()):
                    stage_correct = stats["correct"]
                    stage_total = stats["total"]
                    stage_accuracy = (stage_correct / stage_total * 100) if stage_total > 0 else 0.0
                    stage_accuracies[stage_name] = stage_accuracy
                    # Log per-stage accuracy to wandb - use sanitized stage name
                    # Stage names are already in format like "stage1_baby", just ensure lowercase
                    sanitized_stage = stage_name.lower().replace(' ', '_')
                    wandb_log_dict[f"accuracy/{table_name}_{sanitized_stage}"] = stage_accuracy
                    wandb_log_dict[f"accuracy/{table_name}_{sanitized_stage}_count"] = f"{stage_correct}/{stage_total}"
                
                # Create wandb table with accuracy information
                table = wandb.Table(
                    columns=["stage", "sample_idx", "ground_truth_path", "predicted_path", "correct", "instruction", "ground_truth_text", "prediction_text"], 
                    data=[
                        [row["stage"], row["sample_idx"], row["ground_truth_path"], row["predicted_path"], 
                         row["correct"], row["instruction"], row["ground_truth_text"], row["prediction_text"]] 
                        for row in table_data
                    ]
                )
                # Make table name unique per step to avoid overwriting
                # Tables in Wandb overwrite if same name is used, so include step number
                log_step = int(step) if step is not None else 0
                table_key = f"completions/{table_name}_step_{log_step}"
                wandb_log_dict[table_key] = table
                
                # Also keep the main table name for latest view (will overwrite, but that's okay for "latest")
                wandb_log_dict[f"completions/{table_name}"] = table
                
                # Log metrics using trainer.log() for proper step tracking (like PPO example)
                # This ensures metrics are properly tracked in Trainer's log_history
                metrics_only = {k: v for k, v in wandb_log_dict.items() if k.startswith('accuracy/')}
                
                # Log tables separately via wandb.log (trainer.log doesn't handle tables well)
                tables_only = {k: v for k, v in wandb_log_dict.items() if k.startswith('completions/')}
                
                print(f"   📊 Logging to wandb at step {log_step}")
                print(f"   📊 Metrics being logged: {list(metrics_only.keys())}")
                print(f"   📊 Metrics values: {[(k, v) for k, v in metrics_only.items()]}")
                print(f"   📊 Tables being logged: {list(tables_only.keys())}")
                print(f"   📊 Trainer available: {self.trainer is not None}")
                
                # Log metrics - let Wandb handle step tracking automatically
                # Don't specify step or commit - Wandb will use current step and handle commits
                import wandb
                
                if metrics_only:
                    print(f"   📊 Logging {len(metrics_only)} metrics to wandb...")
                    wandb.log(metrics_only)
                    print(f"   ✅ Metrics logged successfully")
                
                # For tables, let Wandb handle step tracking automatically
                if tables_only:
                    print(f"   📊 Logging {len(tables_only)} tables to wandb...")
                    wandb.log(tables_only)
                    print(f"   ✅ Tables logged successfully")
                
                # Print accuracy to console
                print(f"\n📊 {table_name.upper()} Overall Accuracy: {correct_count}/{total_count} = {accuracy:.2f}%")
                print(f"\n📊 {table_name.upper()} Accuracy by Stage:")
                for stage_name in sorted(stage_accuracies.keys()):
                    stats = stage_stats[stage_name]
                    stage_acc = stage_accuracies[stage_name]
                    print(f"   {stage_name}: {stats['correct']}/{stats['total']} = {stage_acc:.2f}%")
            
            def on_train_begin(self, args, state, control, **kwargs):
                """Store trainer reference at training start"""
                # Store trainer reference for logging
                self.trainer = kwargs.get("trainer")
                # Note: We don't log here - we'll log at step 0 in on_evaluate instead
                # This avoids timing issues where Wandb has already moved past step 0
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Store trainer reference when available (not used for evaluation logging)"""
                # Store trainer reference if available (for potential future use)
                if self.trainer is None:
                    self.trainer = kwargs.get("trainer")
                # Note: We don't log evaluation metrics here - only in on_evaluate
                # This avoids timing issues where state.global_step has advanced past the eval step
            
            def on_step_end(self, args, state, control, **kwargs):
                """Log ground truth accuracy metrics and completions at evaluation steps"""
                # Store trainer reference if available
                if self.trainer is None:
                    self.trainer = kwargs.get("trainer")
                
                # Check if this is an evaluation step (every eval_steps, or step 0)
                # Use the eval_steps from training args (default 25)
                eval_steps = 25  # Default, matches --eval_steps 25 in script
                if hasattr(args, 'eval_steps') and args.eval_steps:
                    eval_steps = args.eval_steps
                
                # Only run evaluation at step 0 and every eval_steps
                if state.global_step % eval_steps != 0 and state.global_step != 0:
                    return
                
                # This is an evaluation step
                eval_step = state.global_step
                
                # Skip if we've already logged at this step (shouldn't happen, but safety check)
                if eval_step in self.logged_steps:
                    print(f"⏭️  Skipping duplicate log at step {eval_step} (already logged)")
                    return
                
                print(f"\n🔍 Running custom evaluation at step {eval_step}")
                print(f"   hasattr(self, 'eval_all_indices'): {hasattr(self, 'eval_all_indices')}")
                if hasattr(self, 'eval_all_indices'):
                    print(f"   eval_all_indices: {self.eval_all_indices[:5] if self.eval_all_indices else None}... (length: {len(self.eval_all_indices) if self.eval_all_indices else 0})")
                print(f"   eval_dataset exists: {self.eval_dataset is not None}")
                
                # Log eval completions and accuracy metrics (50 per stage)
                # This logs both the completions table AND the accuracy metrics
                if hasattr(self, 'eval_all_indices') and self.eval_all_indices and self.eval_dataset:
                    print(f"\n📊 Logging {len(self.eval_all_indices)} eval completions and accuracy metrics to wandb (50 per stage)...")
                    try:
                        self._log_completions_table(
                            self.eval_dataset, 
                            self.eval_all_indices, 
                            "val", 
                            eval_step
                        )
                        self.logged_steps.add(eval_step)
                        print(f"✅ Successfully logged completions and accuracy metrics at step {eval_step}")
                    except Exception as e:
                        print(f"❌ Error logging completions at step {eval_step}: {e}")
                        import traceback
                        traceback.print_exc()
                elif self.eval_samples and self.eval_dataset:
                    # Fallback if eval_all_indices not available
                    print(f"📊 Using fallback eval_samples (length: {len(self.eval_samples)})")
                    self._log_completions_table(
                        self.eval_dataset, 
                        self.eval_samples, 
                        "val", 
                        eval_step
                    )
                    self.logged_steps.add(eval_step)
                else:
                    print(f"⚠️  No eval samples available to log! eval_all_indices={hasattr(self, 'eval_all_indices')}, eval_samples={len(self.eval_samples) if hasattr(self, 'eval_samples') else 'N/A'}")
            
            def on_evaluate(self, args, state, control, logs=None, **kwargs):
                """This won't be called since eval_strategy='no', but kept for compatibility"""
                pass
        
        completions_callback = CompletionsCallback(model, tokenizer, dataset, eval_dataset)
        callbacks = [completions_callback]
    else:
        callbacks = None
    
    # Initialize trainer with shuffle control for curriculum learning
    class CurriculumSFTTrainer(SFTTrainer):
        def get_train_dataloader(self):
            """Override to control shuffling based on args.dataloader_shuffle"""
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            
            train_dataset = self.train_dataset
            if hasattr(train_dataset, "__len__") and len(train_dataset) == 0:
                raise ValueError("Trainer: train_dataset must have a length > 0.")
            
            # Import here to avoid circular imports
            from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
            
            # Use SequentialSampler if shuffle is disabled, RandomSampler if enabled
            if args.dataloader_shuffle:
                train_sampler = RandomSampler(train_dataset)
                print("🔄 Using RandomSampler - data will be shuffled each epoch")
            else:
                train_sampler = SequentialSampler(train_dataset)
                print("🔄 Using SequentialSampler - data order preserved (curriculum)")
                
                # Verify order by checking first few samples
                if len(train_dataset) >= 3:
                    try:
                        sample_0 = train_dataset[0]
                        sample_1 = train_dataset[1]
                        sample_2 = train_dataset[2]
                        if isinstance(sample_0, dict) and "instruction" in sample_0:
                            print(f"   ✓ Verified: Sample 0 curriculum_stage = {sample_0.get('curriculum_stage', 'N/A')}")
                            print(f"   ✓ Verified: Sample 1 curriculum_stage = {sample_1.get('curriculum_stage', 'N/A')}")
                            print(f"   ✓ Verified: Sample 2 curriculum_stage = {sample_2.get('curriculum_stage', 'N/A')}")
                    except Exception as e:
                        print(f"   ⚠️  Could not verify sample order: {e}")
            
            # CRITICAL: When using a sampler, shuffle parameter is ignored
            # SequentialSampler = no shuffling, RandomSampler = shuffling
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=train_sampler,  # This controls order - SequentialSampler preserves order
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
    
    trainer = CurriculumSFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        eval_dataset = eval_dataset,
        args = training_args,
        callbacks = callbacks,
    )

    # Train model
    trainer_stats = trainer.train()

    # Save model
    if args.save_model:
        # if args.quantization_method is a list, we will save the model for each quantization method
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(
                        f"Saving model with quantization method: {quantization_method}"
                    )
                    model.save_pretrained_gguf(
                        args.save_path,
                        tokenizer,
                        quantization_method = quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path = args.hub_path,
                            hub_token = args.hub_token,
                            quantization_method = quantization_method,
                        )
            else:
                print(f"Saving model with quantization method: {args.quantization}")
                model.save_pretrained_gguf(
                    args.save_path, tokenizer, quantization_method = args.quantization
                )
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path = args.hub_path,
                        hub_token = args.hub_token,
                        quantization_method = quantization_method,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        print("Warning: The model is not saved!")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(
        description = "🦥 Fine-tune your llm faster using unsloth!"
    )

    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument(
        "--model_name",
        type = str,
        default = "unsloth/llama-3-8b",
        help = "Model name to load",
    )
    model_group.add_argument(
        "--max_seq_length",
        type = int,
        default = 2048,
        help = "Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!",
    )
    model_group.add_argument(
        "--dtype",
        type = str,
        default = None,
        help = "Data type for model (None for auto detection)",
    )
    model_group.add_argument(
        "--load_in_4bit",
        action = "store_true",
        help = "Use 4bit quantization to reduce memory usage",
    )
    model_group.add_argument(
        "--dataset",
        type = str,
        default = "yahma/alpaca-cleaned",
        help = "Huggingface dataset or local JSON file to use for training",
    )
    model_group.add_argument(
        "--eval_dataset",
        type = str,
        default = None,
        help = "Huggingface dataset or local JSON file to use for evaluation (optional)",
    )

    lora_group = parser.add_argument_group(
        "🧠 LoRA Options", "These options are used to configure the LoRA model."
    )
    lora_group.add_argument(
        "--r",
        type = int,
        default = 16,
        help = "Rank for Lora model, default is 16.  (common values: 8, 16, 32, 64, 128)",
    )
    lora_group.add_argument(
        "--lora_alpha",
        type = int,
        default = 16,
        help = "LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)",
    )
    lora_group.add_argument(
        "--lora_dropout",
        type = float,
        default = 0.0,
        help = "LoRA dropout rate, default is 0.0 which is optimized.",
    )
    lora_group.add_argument(
        "--bias", type = str, default = "none", help = "Bias setting for LoRA"
    )
    lora_group.add_argument(
        "--use_gradient_checkpointing",
        type = str,
        default = "unsloth",
        help = "Use gradient checkpointing",
    )
    lora_group.add_argument(
        "--random_state",
        type = int,
        default = 3407,
        help = "Random state for reproducibility, default is 3407.",
    )
    lora_group.add_argument(
        "--use_rslora", action = "store_true", help = "Use rank stabilized LoRA"
    )
    lora_group.add_argument(
        "--loftq_config", type = str, default = None, help = "Configuration for LoftQ"
    )

    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument(
        "--per_device_train_batch_size",
        type = int,
        default = 2,
        help = "Batch size per device during training, default is 2.",
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type = int,
        default = 4,
        help = "Number of gradient accumulation steps, default is 4.",
    )
    training_group.add_argument(
        "--warmup_steps",
        type = int,
        default = 5,
        help = "Number of warmup steps, default is 5.",
    )
    training_group.add_argument(
        "--max_steps", type = int, default = 400, help = "Maximum number of training steps."
    )
    training_group.add_argument(
        "--learning_rate",
        type = float,
        default = 2e-4,
        help = "Learning rate, default is 2e-4.",
    )
    training_group.add_argument(
        "--optim", type = str, default = "adamw_8bit", help = "Optimizer type."
    )
    training_group.add_argument(
        "--weight_decay",
        type = float,
        default = 0.01,
        help = "Weight decay, default is 0.01.",
    )
    training_group.add_argument(
        "--lr_scheduler_type",
        type = str,
        default = "linear",
        help = "Learning rate scheduler type, default is 'linear'.",
    )
    training_group.add_argument(
        "--seed",
        type = int,
        default = 3407,
        help = "Seed for reproducibility, default is 3407.",
    )
    training_group.add_argument(
        "--eval_strategy",
        type = str,
        default = "steps",
        choices = ["no", "steps", "epoch"],
        help = "Evaluation strategy: 'no', 'steps', or 'epoch'. Default is 'steps'.",
    )
    training_group.add_argument(
        "--eval_steps",
        type = int,
        default = 100,
        help = "Run evaluation every N steps. Only used if eval_strategy='steps'. Default is 100.",
    )
    training_group.add_argument(
        "--per_device_eval_batch_size",
        type = int,
        default = 2,
        help = "Batch size per device for evaluation. Default is 2.",
    )
    training_group.add_argument(
        "--save_strategy",
        type = str,
        default = "steps",
        choices = ["no", "steps", "epoch"],
        help = "Checkpoint saving strategy: 'no', 'steps', or 'epoch'. Default is 'steps'.",
    )
    training_group.add_argument(
        "--save_steps",
        type = int,
        default = 500,
        help = "Save checkpoint every N steps. Only used if save_strategy='steps'. Default is 500.",
    )
    training_group.add_argument(
        "--save_total_limit",
        type = int,
        default = 3,
        help = "Limit the total number of checkpoints. Older checkpoints are deleted. Default is 3.",
    )
    training_group.add_argument(
        "--dataloader_shuffle",
        action = "store_true",
        help = "Shuffle training data. Default is False to preserve curriculum order.",
    )

    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument(
        "--report_to",
        type = str,
        default = "tensorboard",
        choices = [
            "azure_ml",
            "clearml",
            "codecarbon",
            "comet_ml",
            "dagshub",
            "dvclive",
            "flyte",
            "mlflow",
            "neptune",
            "tensorboard",
            "wandb",
            "all",
            "none",
        ],
        help = "The list of integrations to report the results and logs to. Supported platforms are: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations installed, 'none' for no integrations.",
    )
    report_group.add_argument(
        "--logging_steps", type = int, default = 1, help = "Logging steps, default is 1"
    )
    report_group.add_argument(
        "--wandb_entity",
        type = str,
        default = None,
        help = "Wandb entity (username or team name). If not set, uses default from wandb config.",
    )
    report_group.add_argument(
        "--wandb_project",
        type = str,
        default = None,
        help = "Wandb project name. If not set, uses default from wandb config.",
    )
    report_group.add_argument(
        "--wandb_run_name",
        type = str,
        default = None,
        help = "Wandb run name. If not set, wandb auto-generates a name.",
    )

    # Saving and pushing arguments
    save_group = parser.add_argument_group("💾 Save Model Options")
    save_group.add_argument(
        "--output_dir", type = str, default = "outputs", help = "Output directory"
    )
    save_group.add_argument(
        "--save_model", action = "store_true", help = "Save the model after training"
    )
    save_group.add_argument(
        "--save_method",
        type = str,
        default = "merged_16bit",
        choices = ["merged_16bit", "merged_4bit", "lora"],
        help = "Save method for the model, default is 'merged_16bit'",
    )
    save_group.add_argument(
        "--save_gguf",
        action = "store_true",
        help = "Convert the model to GGUF after training",
    )
    save_group.add_argument(
        "--save_path", type = str, default = "model", help = "Path to save the model"
    )
    save_group.add_argument(
        "--quantization",
        type = str,
        default = "q8_0",
        nargs = "+",
        help = "Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf ",
    )

    push_group = parser.add_argument_group("🚀 Push Model Options")
    push_group.add_argument(
        "--push_model",
        action = "store_true",
        help = "Push the model to Hugging Face hub after training",
    )
    push_group.add_argument(
        "--push_gguf",
        action = "store_true",
        help = "Push the model as GGUF to Hugging Face hub after training",
    )
    push_group.add_argument(
        "--hub_path",
        type = str,
        default = "hf/model",
        help = "Path on Hugging Face hub to push the model",
    )
    push_group.add_argument(
        "--hub_token", type = str, help = "Token for pushing the model to Hugging Face hub"
    )

    args = parser.parse_args()
    run(args)
