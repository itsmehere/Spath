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
            # Check if file exists
            if os.path.isfile(args.dataset):
                file_size = os.path.getsize(args.dataset) / (1024 * 1024)  # Size in MB
                print(f"📁 Loading dataset from: {args.dataset} ({file_size:.1f} MB)")
            dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
        else:
            dataset = load_dataset(args.dataset, split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
    # IMPORTANT: Ensure dataset is NOT shuffled - preserve order from JSON file
    # The dataset from load_dataset should already be in order, but let's verify
    print("Training data is formatted and ready!")
    print(f"Training dataset size: {len(dataset)} samples")
    print("⚠️  Note: Data will be loaded in SEQUENTIAL order (no shuffling)")
    
    # Calculate steps per epoch and verify max_steps
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = len(dataset) // effective_batch_size
    epochs = args.max_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    print(f"📊 Training configuration:")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Epochs: {epochs:.2f}")
    if epochs > 1.0:
        print(f"   ⚠️  WARNING: Training will go through data {epochs:.2f} times!")
        print(f"   ⚠️  For curriculum learning, set max_steps <= {steps_per_epoch} to see each stage once")
    else:
        print(f"   ✅ Each curriculum stage will be seen exactly once")
    
    # Verify first few samples to check order (for curriculum learning)
    if len(dataset) >= 3:
        try:
            sample_0 = dataset[0]
            sample_1 = dataset[1]
            sample_2 = dataset[2]
            if "instruction" in sample_0:
                print(f"📋 Dataset order verification:")
                print(f"   Sample 0: {sample_0['instruction']}")
                print(f"   Sample 1: {sample_1['instruction']}")
                print(f"   Sample 2: {sample_2['instruction']}")
        except Exception as e:
            print(f"⚠️  Could not verify dataset order: {e}")

    # Load evaluation dataset if provided
    eval_dataset = None
    if args.eval_dataset:
        if args.eval_dataset.endswith('.json') or os.path.isfile(args.eval_dataset):
            eval_dataset = load_dataset("json", data_files={"train": args.eval_dataset}, split="train")
        else:
            eval_dataset = load_dataset(args.eval_dataset, split = "train")
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)
        print("Evaluation data is formatted and ready!")
        print(f"Evaluation dataset size: {len(eval_dataset)} samples")
        print("✅ Eval data is SEPARATE - never used for training, only for validation!")

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
        # Shuffling control - default False to preserve curriculum order
        dataloader_drop_last = False,
        dataloader_num_workers = 0,  # Sequential loading
        # Evaluation settings
        eval_strategy = args.eval_strategy if eval_dataset else "no",
        eval_steps = args.eval_steps if eval_dataset and args.eval_strategy == "steps" else None,
        per_device_eval_batch_size = args.per_device_eval_batch_size if eval_dataset else None,
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
                
                # Calculate stage boundaries dynamically based on actual dataset size
                # Stage distribution: 15% / 15% / 20% / 20% / 15% / 15%
                train_len = len(train_dataset) if train_dataset else 0
                eval_len = len(eval_dataset) if eval_dataset else 0
                
                print(f"\n📊 Calculating curriculum stage boundaries:")
                print(f"   Train dataset size: {train_len}")
                print(f"   Eval dataset size: {eval_len}")
                
                # Calculate boundaries based on actual dataset sizes
                self.train_stage_boundaries = self._calculate_stage_boundaries(train_len, [0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
                self.eval_stage_boundaries = self._calculate_stage_boundaries(eval_len, [0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
                
                print(f"   Train stage boundaries: {self.train_stage_boundaries}")
                print(f"   Eval stage boundaries: {self.eval_stage_boundaries}")
                
                # Sample 2 examples from each curriculum stage
                print("\n📊 Sampling completions for curriculum stages:")
                print("   Training set:")
                self.train_samples = self._sample_from_stages(train_dataset, self.train_stage_boundaries, samples_per_stage=2)
                if eval_dataset:
                    print("   Validation set:")
                    self.eval_samples = self._sample_from_stages(eval_dataset, self.eval_stage_boundaries, samples_per_stage=2)
                else:
                    self.eval_samples = []
                print(f"   Total: {len(self.train_samples)} train samples, {len(self.eval_samples)} eval samples\n")
            
            def _calculate_stage_boundaries(self, dataset_len, stage_proportions):
                """Calculate stage boundaries based on dataset size and proportions"""
                if dataset_len == 0:
                    return []
                
                boundaries = []
                start = 0
                for prop in stage_proportions:
                    end = start + int(dataset_len * prop) - 1
                    boundaries.append((start, min(end, dataset_len - 1)))
                    start = end + 1
                return boundaries
            
            def _sample_from_stages(self, dataset, stage_boundaries, samples_per_stage=2):
                """Sample specified number of examples from each curriculum stage"""
                samples = []
                dataset_len = len(dataset) if dataset else 0
                
                if dataset_len == 0:
                    print(f"   ⚠️  Dataset is empty, cannot sample")
                    return samples
                
                for stage_idx, (start, end) in enumerate(stage_boundaries):
                    # Ensure boundaries are within dataset
                    stage_start = max(0, min(start, dataset_len - 1))
                    stage_end = max(stage_start, min(end, dataset_len - 1))
                    
                    if stage_start <= stage_end:
                        stage_indices = list(range(stage_start, stage_end + 1))
                        stage_size = len(stage_indices)
                        
                        if stage_size >= samples_per_stage:
                            sampled = random.sample(stage_indices, samples_per_stage)
                            samples.extend(sampled)
                            print(f"   Stage {stage_idx + 1}: sampled {samples_per_stage} indices {sampled} (range {stage_start}-{stage_end}, {stage_size} total)")
                        elif stage_size > 0:
                            # If stage has fewer samples than requested, take all and warn
                            samples.extend(stage_indices)
                            print(f"   ⚠️  Stage {stage_idx + 1}: only {stage_size} samples available, sampled all (range {stage_start}-{stage_end})")
                        else:
                            print(f"   ⚠️  Stage {stage_idx + 1}: no samples in range {stage_start}-{stage_end}")
                    else:
                        print(f"   ⚠️  Stage {stage_idx + 1}: invalid range {stage_start}-{stage_end}")
                
                expected_samples = len(stage_boundaries) * samples_per_stage
                if len(samples) < expected_samples:
                    print(f"   ⚠️  WARNING: Only got {len(samples)} samples, expected {expected_samples}")
                
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
                return text[:200] + "..." if len(text) > 200 else text, ""
            
            def _generate_prediction(self, instruction_text):
                """Generate prediction from model"""
                try:
                    # Extract just the instruction part (before Response)
                    if "### Response:" in instruction_text:
                        prompt = instruction_text.split("### Response:")[0] + "### Response:"
                    else:
                        prompt = instruction_text
                    
                    # Tokenize and generate
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_length - 50).to(self.model.device)
                    
                    # Set model to eval mode for generation
                    was_training = self.model.training
                    self.model.eval()
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    # Restore training mode
                    if was_training:
                        self.model.train()
                    
                    # Decode only the new tokens
                    generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    return generated_text.strip()
                except Exception as e:
                    return f"Error: {str(e)}"
            
            def _get_stage_for_index(self, idx, is_train=True):
                """Determine which curriculum stage an index belongs to"""
                boundaries = self.train_stage_boundaries if is_train else self.eval_stage_boundaries
                for stage_idx, (start, end) in enumerate(boundaries):
                    if start <= idx <= end:
                        return stage_idx + 1
                return None
            
            def _log_completions_table(self, dataset, sample_indices, table_name, step):
                """Log completions table to wandb"""
                if not sample_indices:
                    return
                
                import wandb
                
                is_train = (table_name == "train")
                table_data = []
                for idx in sample_indices:
                    sample = dataset[idx]
                    text = sample.get("text", "")
                    instruction, ground_truth = self._extract_instruction_and_output(text)
                    
                    # Generate prediction
                    prediction = self._generate_prediction(text)
                    
                    # Determine curriculum stage
                    stage = self._get_stage_for_index(idx, is_train=is_train)
                    stage_name = f"Stage {stage}" if stage else "Unknown"
                    
                    table_data.append({
                        "stage": stage_name,
                        "sample_idx": idx,
                        "instruction": instruction,  # No truncation - display full text
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                    })
                
                # Create wandb table with stage information
                table = wandb.Table(columns=["stage", "sample_idx", "instruction", "ground_truth", "prediction"], data=[
                    [row["stage"], row["sample_idx"], row["instruction"], row["ground_truth"], row["prediction"]] 
                    for row in table_data
                ])
                
                # Log to wandb
                wandb.log({f"completions/{table_name}": table}, step=step)
            
            def on_train_begin(self, args, state, control, **kwargs):
                """Log initial completions at training start"""
                # Log initial predictions at step 0
                if self.train_samples:
                    self._log_completions_table(
                        self.train_dataset, 
                        self.train_samples, 
                        "train", 
                        0
                    )
                
                if self.eval_samples and self.eval_dataset:
                    self._log_completions_table(
                        self.eval_dataset, 
                        self.eval_samples, 
                        "val", 
                        0
                    )
            
            def on_evaluate(self, args, state, control, logs=None, **kwargs):
                """Log completions when evaluation actually runs"""
                # This callback is only called during actual evaluation
                # Log train completions
                if self.train_samples:
                    self._log_completions_table(
                        self.train_dataset, 
                        self.train_samples, 
                        "train", 
                        state.global_step
                    )
                
                # Log eval completions
                if self.eval_samples and self.eval_dataset:
                    self._log_completions_table(
                        self.eval_dataset, 
                        self.eval_samples, 
                        "val", 
                        state.global_step
                    )
        
        completions_callback = CompletionsCallback(model, tokenizer, dataset, eval_dataset)
        callbacks = [completions_callback]
    else:
        callbacks = None
    
    # Initialize trainer
    # Override get_train_dataloader to control shuffling
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
                        if "instruction" in sample_0:
                            print(f"   ✓ Verified: Sample 0 = {sample_0['instruction']}")
                            print(f"   ✓ Verified: Sample 1 = {sample_1['instruction']}")
                            print(f"   ✓ Verified: Sample 2 = {sample_2['instruction']}")
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
