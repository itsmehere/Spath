#!/usr/bin/env python3

"""
Starter Script for Fine-Tuning GPT-2 with Transformers and PEFT

This script is designed for fine-tuning GPT-2 models using transformers and peft.
It includes configurable options for model loading, PEFT parameters, training arguments, 
and model saving/pushing functionalities.

You will likely want to customize this script to suit your specific use case 
and requirements.

Here are a few suggestions for customization:
    - Modify the dataset loading and preprocessing steps to match your data.
    - Customize the model saving and pushing configurations.

Usage: (most of the options have valid default values this is an extended example for demonstration purposes)
    python unsloth-cli-kevin.py --model_name "gpt2-medium" --max_seq_length 2048 --load_in_4bit \
    --r 64 --lora_alpha 32 --lora_dropout 0.1 --bias "none" --use_gradient_checkpointing \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --warmup_steps 5 --max_steps 400 --learning_rate 2e-4 --logging_steps 1 --optim "adamw_8bit" \
    --weight_decay 0.005 --lr_scheduler_type "linear" --seed 3407 --output_dir "outputs" \
    --report_to "tensorboard" --save_model --save_path "model" \
    --push_model --hub_path "hf/model" --hub_token "your_hf_token"

To see a full list of configurable options, use:
    python unsloth-cli-kevin.py --help

Happy fine-tuning!
"""

import argparse
import os
import random


def run(args):
    import torch
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
    from transformers import DataCollatorForLanguageModeling
    from datasets import load_dataset
    from transformers.utils import strtobool
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from peft import PeftModel
    import logging

    logging.basicConfig(level=logging.INFO)

    # Determine device and dtype
    device_map = "auto"
    torch_dtype = None
    if args.dtype:
        if args.dtype == "float16" or args.dtype == "fp16":
            torch_dtype = torch.float16
        elif args.dtype == "bfloat16" or args.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif args.dtype == "float32" or args.dtype == "fp32":
            torch_dtype = torch.float32
    
    # Check if bfloat16 is supported
    is_bfloat16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if is_bfloat16_supported else torch.float16

    # Configure quantization if needed
    quantization_config = None
    if args.load_in_4bit:
        try:
            # BitsAndBytesConfig is imported from transformers, but we need to check if bitsandbytes is installed
            import bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization. "
                "Install it with: pip install bitsandbytes\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error configuring bitsandbytes quantization: {e}\n"
                "Please ensure bitsandbytes is properly installed and compatible with your system."
            )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # GPT-2 specific tokenizer fixes
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    
    # Ensure model config matches tokenizer settings
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, 'bos_token_id'):
        model.config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model.config, 'eos_token_id'):
        model.config.eos_token_id = tokenizer.eos_token_id
    
    # Update generation config if it exists
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.generation_config, 'bos_token_id'):
            model.generation_config.bos_token_id = tokenizer.bos_token_id
        if hasattr(model.generation_config, 'eos_token_id'):
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    
    # Verify vocabulary sizes match
    vocab_size_tokenizer = len(tokenizer)
    vocab_size_model = model.config.vocab_size
    if vocab_size_tokenizer != vocab_size_model:
        print(f"Warning: Tokenizer vocab size ({vocab_size_tokenizer}) != Model vocab size ({vocab_size_model})")
        print(f"Using model vocab size: {vocab_size_model}")
    
    # Ensure pad_token_id is valid
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= vocab_size_model:
        raise ValueError(
            f"pad_token_id ({tokenizer.pad_token_id}) is >= model vocab_size ({vocab_size_model}). "
            "This will cause CUDA errors."
        )
    
    # Disable SDPA for GPT-2 if it's causing issues (can be re-enabled if needed)
    # SDPA can sometimes cause issues with certain configurations
    if hasattr(model.config, '_attn_implementation'):
        # Keep the default, but ensure it's compatible
        pass
    # For GPT-2, we might need to use eager attention instead of SDPA
    try:
        # Try to set attention implementation to eager if SDPA is causing issues
        if hasattr(model.config, 'attn_implementation'):
            # Only change if there are known issues
            pass
    except:
        pass

    # Prepare model for k-bit training if using quantization
    # This is required for 4-bit/8-bit models to enable training
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
        )
    
    # Configure PEFT model
    # GPT-2 target modules: c_attn (query, key, value), c_proj (attention output), c_fc (MLP input)
    target_modules = ["c_attn", "c_proj", "c_fc"]
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing if requested (if not already enabled by prepare_model_for_kbit_training)
    if args.use_gradient_checkpointing and not args.load_in_4bit:
        model.gradient_checkpointing_enable()
    
    # Ensure model is in training mode
    model.train()

    # alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    #
    # ### Instruction:
    # {}
    #
    # ### Input:
    # {}
    #
    # ### Response:
    # {}"""

    modified_prompt = "{input} {output}"
    RESPONSE_TEMPLATE = "Output: "

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        #instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        #for instruction, input, output in zip(instructions, inputs, outputs):
        for input, output in zip(inputs, outputs):
            text = modified_prompt.format(input=input, output=output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # Load and format dataset
    # Support local JSON files
    if args.dataset.endswith('.json') or os.path.isfile(args.dataset):
        dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
    else:
        dataset = load_dataset(args.dataset, split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Tokenize without padding - data collator will handle it
        # Important: Don't skip special tokens - we need EOS tokens for the model to learn when to stop
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=False,  # We already added EOS in formatting, don't add again
        )
        return tokenized
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print("Training data is formatted and ready!")

    # Load evaluation dataset if provided
    eval_dataset = None
    if args.eval_dataset:
        if args.eval_dataset.endswith('.json') or os.path.isfile(args.eval_dataset):
            eval_dataset = load_dataset("json", data_files={"train": args.eval_dataset}, split="train")
        else:
            eval_dataset = load_dataset(args.eval_dataset, split = "train")
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
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
    training_args = TrainingArguments(
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported,
        bf16 = is_bfloat16_supported,
        logging_steps = args.logging_steps,
        optim = args.optim,
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        seed = args.seed,
        output_dir = args.output_dir,
        report_to = args.report_to,
        # Evaluation settings
        eval_strategy = args.eval_strategy if eval_dataset else "no",
        eval_steps = args.eval_steps if eval_dataset and args.eval_strategy == "steps" else None,
        per_device_eval_batch_size = args.per_device_eval_batch_size if eval_dataset else None,
        prediction_loss_only = True,  # Only compute loss, don't accumulate predictions (saves memory)
        # Checkpoint saving settings
        save_strategy = args.save_strategy,
        save_steps = args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit = args.save_total_limit,
        remove_unused_columns = False,
        # Memory-efficient data loader settings
        dataloader_num_workers = 2,  # Reduced to save memory
        dataloader_pin_memory = False,  # Disable to save VRAM
    )

    # Create custom callback for logging completions to wandb
    if wandb_initialized:
        from transformers import TrainerCallback
        import wandb
        
        class CompletionsCallback(TrainerCallback):
            def __init__(self, model, tokenizer, train_dataset, eval_dataset, max_eval_samples=100):
                self.model = model
                self.tokenizer = tokenizer
                self.train_dataset = train_dataset
                self.eval_dataset = eval_dataset
                self.max_eval_samples = max_eval_samples
                # Sample 5 datapoints from train set
                self.train_samples = random.sample(range(len(train_dataset)), min(20, len(train_dataset))) if len(train_dataset) > 0 else []
                # Sample datapoints from eval set
                self.eval_samples = random.sample(range(len(eval_dataset)), min(20, len(eval_dataset))) if eval_dataset and len(eval_dataset) > 0 else []
                self.should_log_completions = False  # Flag to trigger completion logging
                self.last_logged_step = -1  # Track last step we logged to avoid duplicates
            
            def _decode_sample(self, sample):
                """Decode a tokenized sample back to text"""
                if "text" in sample:
                    return sample["text"]
                elif "input_ids" in sample:
                    # Decode the input_ids back to text
                    input_ids = sample["input_ids"]
                    # Convert to list if it's a tensor or numpy array
                    if hasattr(input_ids, 'cpu'):
                        input_ids = input_ids.cpu().tolist()
                    elif hasattr(input_ids, 'tolist'):
                        input_ids = input_ids.tolist()
                    elif not isinstance(input_ids, list):
                        input_ids = list(input_ids)
                    
                    # Remove padding tokens (pad_token_id) from the end
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is not None and len(input_ids) > 0:
                        # Remove trailing padding tokens
                        while len(input_ids) > 0 and input_ids[-1] == pad_token_id:
                            input_ids.pop()
                    
                    if len(input_ids) == 0:
                        return ""
                    
                    text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                    return text
                else:
                    return ""
            
            def _extract_instruction_and_output(self, text):
                """Extract input and output from formatted text"""
                # The text is formatted as: "{input (ends with 'Output: ')} {output}<eos>"
                eos_token = self.tokenizer.eos_token
                # Remove EOS token if present
                text_without_eos = text.rstrip(eos_token) if eos_token else text
                # Split on "Output: " marker (the input ends with this)
                if "Output: " in text_without_eos:
                    parts = text_without_eos.rsplit("Output: ", 1)
                    input_text = parts[0] + "Output: "  # Keep "Output: " as part of the prompt
                    output = parts[1] if len(parts) > 1 else ""
                else:
                    # No marker found, treat entire text as input
                    input_text = text_without_eos
                    output = ""
                return input_text, output
            
            def _extract_number_from_text(self, text):
                """Extract the first number from text, handling various formats"""
                import re
                # Try to find the first integer in the text
                # This handles cases like "160", "The answer is 160", "160.", etc.
                match = re.search(r'\d+', text)
                if match:
                    return int(match.group())
                return None
            
            def _generate_prediction(self, input_text):
                """Generate prediction from model given input text"""
                try:
                    # input_text already ends with "Output: " so use it directly
                    # Tokenize the input
                    inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=args.max_seq_length - 50).to(self.model.device)
                    
                    # Check if input is empty
                    if inputs['input_ids'].shape[1] == 0:
                        return "Error: Empty input"
                    
                    # Set model to eval mode for generation
                    was_training = self.model.training
                    self.model.eval()
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=10,  # Allow more tokens but stop at EOS
                            temperature=0.2,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.1,  # Slight penalty to avoid repetition
                        )
                    
                    # Restore training mode
                    if was_training:
                        self.model.train()
                    
                    # Decode only the new tokens (generated part)
                    # Stop at EOS token if present
                    input_length = inputs['input_ids'].shape[1]
                    if outputs[0].shape[0] > input_length:
                        # Find EOS token in generated sequence
                        generated_ids = outputs[0][input_length:]
                        eos_positions = (generated_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_positions) > 0:
                            # Stop at first EOS token
                            generated_ids = generated_ids[:eos_positions[0]]
                        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                        # Remove EOS token from decoded text if it's still there
                        if self.tokenizer.eos_token and generated_text.endswith(self.tokenizer.eos_token):
                            generated_text = generated_text[:-len(self.tokenizer.eos_token)]
                    else:
                        generated_text = ""
                    return generated_text.strip()
                except Exception as e:
                    import traceback
                    return f"Error: {str(e)}\n{traceback.format_exc()}"
            
            def _compute_accuracy(self, dataset, max_samples=None):
                """Compute accuracy on dataset by comparing predictions to ground truth"""
                if not dataset or len(dataset) == 0:
                    return 0.0, 0, 0
                
                # Limit number of samples for efficiency (evaluate on subset)
                num_samples = min(max_samples or len(dataset), len(dataset))
                sample_indices = list(range(num_samples))
                
                correct = 0
                total = 0
                
                for idx in sample_indices:
                    try:
                        sample = dataset[idx]
                        # Decode the tokenized sample back to text
                        text = self._decode_sample(sample)
                        if not text:
                            continue
                        
                        input_text, ground_truth = self._extract_instruction_and_output(text)
                        
                        # Only proceed if we have both input and ground truth
                        if not input_text or not ground_truth:
                            continue
                        
                        # Generate prediction using just the input
                        prediction_text = self._generate_prediction(input_text)
                        
                        # Extract numbers from both ground truth and prediction
                        gt_number = self._extract_number_from_text(ground_truth)
                        pred_number = self._extract_number_from_text(prediction_text)
                        
                        if gt_number is not None:
                            total += 1
                            if pred_number is not None and gt_number == pred_number:
                                correct += 1
                    except Exception as e:
                        # Skip samples that cause errors
                        continue
                
                accuracy = correct / total if total > 0 else 0.0
                return accuracy, correct, total
            
            def _compute_mae(self, dataset, max_samples=None):
                """Compute mean absolute error on dataset by comparing predictions to ground truth"""
                if not dataset or len(dataset) == 0:
                    return 0.0, 0
                
                # Limit number of samples for efficiency (evaluate on subset)
                num_samples = min(max_samples or len(dataset), len(dataset))
                sample_indices = list(range(num_samples))
                
                absolute_errors = []
                total = 0
                
                for idx in sample_indices:
                    try:
                        sample = dataset[idx]
                        # Decode the tokenized sample back to text
                        text = self._decode_sample(sample)
                        if not text:
                            continue
                        
                        input_text, ground_truth = self._extract_instruction_and_output(text)
                        
                        # Only proceed if we have both input and ground truth
                        if not input_text or not ground_truth:
                            continue
                        
                        # Generate prediction using just the input
                        prediction_text = self._generate_prediction(input_text)
                        
                        # Extract numbers from both ground truth and prediction
                        gt_number = self._extract_number_from_text(ground_truth)
                        pred_number = self._extract_number_from_text(prediction_text)
                        
                        if gt_number is not None and pred_number is not None:
                            absolute_error = abs(gt_number - pred_number)
                            absolute_errors.append(absolute_error)
                            total += 1
                    except Exception as e:
                        # Skip samples that cause errors
                        continue
                
                mae = sum(absolute_errors) / total if total > 0 else 0.0
                return mae, total
            
            def _log_completions_table(self, dataset, sample_indices, table_name, state=None):
                """Log completions table to wandb following best practices"""
                if not sample_indices:
                    return
                
                import wandb
                
                table_data = []
                for idx in sample_indices:
                    try:
                        sample = dataset[idx]
                        # Decode the tokenized sample back to text
                        text = self._decode_sample(sample)
                        if not text:
                            continue
                        
                        input_text, ground_truth = self._extract_instruction_and_output(text)
                        
                        # Only proceed if we have input
                        if not input_text:
                            continue
                        
                        # Generate prediction using just the input
                        prediction = self._generate_prediction(input_text)
                        
                        table_data.append({
                            "sample_idx": idx,
                            "input": input_text if input_text else "",  # Truncate for display
                            "ground_truth": ground_truth if ground_truth else "",
                            "prediction": prediction if prediction else "",
                        })
                    except Exception as e:
                        # Skip samples that cause errors, but log a row with error info
                        table_data.append({
                            "sample_idx": idx,
                            "input": f"Error decoding sample {idx}",
                            "ground_truth": "",
                            "prediction": str(e)[:200],
                        })
                
                if table_data:
                    # Create wandb table
                    table = wandb.Table(columns=["input", "ground_truth", "prediction"], data=[
                        [row["input"], row["ground_truth"], row["prediction"]] 
                        for row in table_data
                    ])
                    
                    # Following wandb best practices: don't specify step parameter
                    # Let wandb use its internal step tracking to avoid conflicts
                    # Transformers' WandbCallback handles step synchronization automatically
                    # When called from on_log, this will be batched with metrics logged by Transformers
                    wandb.log({f"completions/{table_name}": table})
            
            def on_train_begin(self, args, state, control, **kwargs):
                """Log initial completions at training start"""
                # Log initial predictions at step 0
                if self.train_samples:
                    self._log_completions_table(
                        self.train_dataset, 
                        self.train_samples, 
                        "train", 
                        state
                    )
                
                if self.eval_samples and self.eval_dataset:
                    self._log_completions_table(
                        self.eval_dataset, 
                        self.eval_samples, 
                        "val", 
                        state
                    )
            
            def on_evaluate(self, args, state, control, logs=None, **kwargs):
                """Compute accuracy and MAE during evaluation and set flag to log completions"""
                # Compute accuracy and MAE during evaluation on a subset of eval dataset for speed
                # This happens during the eval pass, so it's concurrent with eval_loss computation
                if self.eval_dataset and len(self.eval_dataset) > 0:
                    # Use a limited number of samples for accuracy calculation to speed up evaluation
                    # If max_eval_samples is -1, use all samples
                    max_samples = None if self.max_eval_samples == -1 else self.max_eval_samples
                    eval_accuracy, correct, total = self._compute_accuracy(
                        self.eval_dataset, 
                        max_samples=max_samples
                    )
                    # Compute mean absolute error
                    eval_mae, mae_total = self._compute_mae(
                        self.eval_dataset,
                        max_samples=max_samples
                    )
                    
                    # Store metrics to add to logs
                    self.eval_accuracy = eval_accuracy
                    self.eval_correct = correct
                    self.eval_total = total
                    self.eval_mae = eval_mae
                    self.eval_mae_total = mae_total
                    
                    # Add metrics to logs directly so they appear with eval_loss
                    # When report_to="wandb" is set, transformers automatically logs
                    # all metrics in the logs dict to wandb with proper step tracking
                    # Following wandb best practices: metrics added to logs dict are automatically
                    # logged with the correct step by Transformers' WandbCallback
                    if logs is not None:
                        # Use consistent metric naming convention (eval_ prefix for evaluation metrics)
                        logs["eval_accuracy"] = eval_accuracy
                        logs["eval_correct"] = correct
                        logs["eval_total"] = total
                        
                        # Also log as percentage for better visualization in wandb
                        logs["eval_accuracy_pct"] = eval_accuracy * 100.0
                        
                        # Log mean absolute error
                        logs["eval_mae"] = eval_mae
                        logs["eval_mae_total"] = mae_total
                    
                    # Print one example prompt and completion
                    try:
                        if len(self.eval_dataset) > 0:
                            # Use a rotating example index based on step to show different examples
                            example_idx = state.global_step % len(self.eval_dataset)
                            sample = self.eval_dataset[example_idx]
                            text = self._decode_sample(sample)
                            if text:
                                input_text, ground_truth = self._extract_instruction_and_output(text)
                                if input_text:
                                    # Generate prediction
                                    prediction = self._generate_prediction(input_text)
                                    
                                    # Print example
                                    print("\n" + "="*80)
                                    print(f"Example at step {state.global_step} (sample {example_idx}):")
                                    print("-"*80)
                                    print(f"Prompt: {input_text[:200]}{'...' if len(input_text) > 200 else ''}")
                                    print(f"Ground Truth: {ground_truth}")
                                    print(f"Prediction: {prediction}")
                                    print("="*80 + "\n")
                    except Exception as e:
                        print(f"Error printing example: {e}")
                else:
                    self.eval_accuracy = None
                
                # Set flag to log completions in on_log callback
                self.should_log_completions = True
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Log completions, accuracy, and MAE when metrics are logged, ensuring correct step"""
                import wandb
                
                # Only log if flag is set and we haven't logged at this step yet
                if self.should_log_completions and state.global_step != self.last_logged_step:
                    # Following wandb best practices: explicitly log metrics to ensure they're recorded
                    # Even though Transformers' WandbCallback should log metrics from logs dict,
                    # explicitly logging ensures they're definitely captured
                    
                    metrics_to_log = {}
                    
                    # Explicitly log accuracy and MAE metrics
                    if hasattr(self, 'eval_accuracy') and self.eval_accuracy is not None:
                        metrics_to_log["eval_accuracy"] = self.eval_accuracy
                        metrics_to_log["eval_accuracy_pct"] = self.eval_accuracy * 100.0
                        metrics_to_log["eval_correct"] = self.eval_correct
                        metrics_to_log["eval_total"] = self.eval_total
                        print(f"Accuracy: {self.eval_accuracy:.4f} ({self.eval_correct}/{self.eval_total})")
                    
                    if hasattr(self, 'eval_mae') and self.eval_mae is not None:
                        metrics_to_log["eval_mae"] = self.eval_mae
                        metrics_to_log["eval_mae_total"] = self.eval_mae_total
                        print(f"MAE: {self.eval_mae:.4f} (computed on {self.eval_mae_total} samples)")
                    
                    # Log metrics explicitly to wandb (wandb handles step tracking automatically)
                    if metrics_to_log:
                        wandb.log(metrics_to_log)
                    
                    # Log train completions
                    if self.train_samples:
                        self._log_completions_table(
                            self.train_dataset, 
                            self.train_samples, 
                            "train", 
                            state
                        )
                    
                    # Log eval completions
                    if self.eval_samples and self.eval_dataset:
                        self._log_completions_table(
                            self.eval_dataset, 
                            self.eval_samples, 
                            "val", 
                            state
                        )
                    
                    # Reset flag and track step
                    self.should_log_completions = False
                    self.last_logged_step = state.global_step
        
        completions_callback = CompletionsCallback(model, tokenizer, dataset, eval_dataset, max_eval_samples=args.max_eval_samples)
        callbacks = [completions_callback]
    else:
        callbacks = None
    
    # Define compute_metrics function for accuracy calculation during evaluation
    def compute_metrics(eval_pred):
        """Compute accuracy metric during evaluation"""
        # This function is called during evaluation, but for text generation tasks,
        # we need to generate text which is expensive. So we'll compute accuracy
        # in the callback instead. This function is here for compatibility.
        return {}
    
    # Data collator - masks prompt tokens so loss is only computed on the response
    # Custom collator that finds the LAST occurrence of "Output: " since there may be multiple
    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    
    class DataCollatorForCompletionOnlyLMLastOccurrence(DataCollatorForLanguageModeling):
        """Custom data collator that finds the LAST occurrence of the response template and masks everything before it."""
        
        def __init__(self, response_template, tokenizer, **kwargs):
            # Initialize base class with mlm=False for causal LM
            super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
            # Store the response template token IDs
            if isinstance(response_template, list):
                self._response_template_ids = response_template
            else:
                self._response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        
        def torch_call(self, examples):
            import torch
            # First, get the batch from the parent class (handles padding, etc.)
            batch = super().torch_call(examples)
            
            # Now mask everything before (and including) the LAST occurrence of the response template
            for i, input_ids in enumerate(batch["input_ids"]):
                labels = batch["labels"][i].clone()
                input_ids_list = input_ids.tolist()
                
                # Find ALL occurrences of response_template_ids in input_ids
                response_token_ids = self._response_template_ids
                template_len = len(response_token_ids)
                occurrences = []
                
                # Find all occurrences of the template
                for j in range(len(input_ids_list) - template_len + 1):
                    if input_ids_list[j:j + template_len] == response_token_ids:
                        occurrences.append(j)
                
                if occurrences:
                    # Use the LAST occurrence
                    last_idx = occurrences[-1]
                    # Mask everything before (and including) the response template
                    # Response starts AFTER the template
                    response_start = last_idx + template_len
                    labels[:response_start] = -100
                    batch["labels"][i] = labels
                else:
                    # If no template found, mask everything (shouldn't happen in normal training)
                    labels[:] = -100
                    batch["labels"][i] = labels
            
            return batch
    
    data_collator = DataCollatorForCompletionOnlyLMLastOccurrence(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )
    print(f"✅ Using DataCollatorForCompletionOnlyLM (LAST occurrence) - prompt tokens are MASKED from loss")
    print(f"   Response template: '{RESPONSE_TEMPLATE}' -> token ids: {response_template_ids}")
    
    # Custom Trainer class to control data shuffling for curriculum learning
    class CurriculumTrainer(Trainer):
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
                print("   ⚠️  IMPORTANT: Training data is in CURRICULUM ORDER!")
                print("   ⚠️  Easy examples (smaller graphs) come first, harder examples (larger graphs) come later.")
            
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
    
    # Initialize trainer
    trainer = CurriculumTrainer(
        model = model,
        processing_class = tokenizer,  # Use processing_class instead of tokenizer (new API)
        train_dataset = dataset,
        eval_dataset = eval_dataset,
        args = training_args,
        data_collator = data_collator,
        callbacks = callbacks,
        compute_metrics = compute_metrics if eval_dataset else None,
    )

    # Train model
    trainer_stats = trainer.train()

    # Save model
    if args.save_model:
        if args.save_method == "lora":
            # Save only LoRA weights
            model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
            print(f"LoRA model saved to {args.save_path}")
        else:
            # Merge and save full model
            # First, merge LoRA weights
            if hasattr(model, 'merge_and_unload'):
                model = model.merge_and_unload()
            model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
            print(f"Full model saved to {args.save_path}")
        
        if args.push_model:
            model.push_to_hub(args.hub_path, token=args.hub_token)
            tokenizer.push_to_hub(args.hub_path, token=args.hub_token)
            print(f"Model pushed to hub: {args.hub_path}")
    else:
        print("Warning: The model is not saved!")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(
        description = "Fine-tune GPT-2 using transformers and peft!"
    )

    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument(
        "--model_name",
        type = str,
        default = "gpt2-medium",
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
        action = "store_true",
        help = "Use gradient checkpointing",
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
        "--max_eval_samples",
        type = int,
        default = 100,
        help = "Maximum number of samples to use for accuracy/MAE computation during evaluation. Default is 100. Set to -1 to use all samples (slower).",
    )
    training_group.add_argument(
        "--dataloader_shuffle",
        action = "store_true",
        default = False,
        help = "Shuffle training data each epoch. Default is False to preserve curriculum order. Set this flag to enable shuffling.",
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
        "--save_path", type = str, default = "model", help = "Path to save the model"
    )

    push_group = parser.add_argument_group("🚀 Push Model Options")
    push_group.add_argument(
        "--push_model",
        action = "store_true",
        help = "Push the model to Hugging Face hub after training",
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
