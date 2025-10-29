import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate
from datetime import timedelta
from tqdm import tqdm

# ------------------------
# Argument parser
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="deepseek-ai/deepseek-coder-1b-base")
    p.add_argument("--output_dir", type=str, default="deepseek_coder1B_lora_out")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    # Accept either a single dataset name or multiple comma-separated names
    p.add_argument("--dataset", type=str, default="codeparrot",
                   help="Single dataset name or comma-separated list. Supported values: synthetic, codeparrot, iamtarun/python_code_instructions_18k_alpaca")

    # Backwards-compatible new flag for mixture
    p.add_argument("--dataset_mix", type=str, default=None,
                   help="Comma-separated dataset names. If provided, overrides --dataset")

    # lr scheduler choice
    p.add_argument("--lr_scheduler", type=str, default="linear",
                   choices=["linear", "cosine"],
                   help="Learning rate scheduler to use: linear or cosine (with warmup)")

    p.add_argument("--save_every", type=int, default=0,
                   help="Save checkpoint every N steps (0=disabled)")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume training from checkpoint directory")
    p.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                   help="Keep only the last N checkpoints to save disk space")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()

# ------------------------
# Checkpoint management
# ------------------------
def save_checkpoint(accelerator, model, tokenizer, optimizer, scheduler, epoch, step, output_dir, 
                   metrics, keep_last_n=3):
    """Save training checkpoint with state"""
    if not accelerator.is_main_process:
        return
    
    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch}-step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save model
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    
    # Save training state
    state = {
        "epoch": epoch,
        "step": step,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }
    torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
    
    print("\n", f"💾 Checkpoint saved: {ckpt_dir}")
    
    # Clean up old checkpoints
    cleanup_old_checkpoints(output_dir, keep_last_n)

def cleanup_old_checkpoints(output_dir, keep_last_n):
    """Remove old checkpoints to save disk space"""
    if keep_last_n <= 0:
        return
    
    if not os.path.isdir(output_dir):
        return

    checkpoints = [d for d in os.listdir(output_dir) 
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by modification time (oldest first)
    checkpoints = sorted(checkpoints, 
                        key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    
    # Remove oldest checkpoints
    to_remove = checkpoints[:-keep_last_n]
    for ckpt in to_remove:
        ckpt_path = os.path.join(output_dir, ckpt)
        import shutil
        shutil.rmtree(ckpt_path)
        print(f"🗑️  Removed old checkpoint: {ckpt}")

def load_checkpoint(checkpoint_dir):
    """Load training checkpoint"""
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path):
        return None
    
    state = torch.load(state_path, map_location="cpu")
    print(f"📂 Loaded checkpoint from epoch {state['epoch']}, step {state['step']}")
    return state

# ------------------------
# Main training function
# ------------------------
def main():
    args = parse_args()
    start_time = time.time()
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"🔹 Using model: {args.model_id}")
        print(f"🔹 Dataset (requested): {args.dataset if args.dataset_mix is None else args.dataset_mix}")
        print(f"🔹 Output directory: {args.output_dir}")
        print(f"🔹 Learning rate: {args.learning_rate}")
        print(f"🔹 Max grad norm: {args.max_grad_norm}")
        print(f"🔹 LR scheduler: {args.lr_scheduler}")
        print(f"🔹 Dry run: {args.dry_run}")
        if args.resume_from:
            print(f"🔹 Resuming from: {args.resume_from}")

    # ------------------------
    # Load tokenizer + model
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Load model in FP32, let Accelerate handle mixed precision conversion
    if is_main:
        print("🔧 Loading model in FP32 for stability...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_safetensors=True,
    )

    # CRITICAL: Disable cache before enabling gradient checkpointing
    model.config.use_cache = False
    # Use the simple call; model implementations vary
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        # Some models may not implement this method or accept kwargs
        pass

    model.resize_token_embeddings(len(tokenizer))

    # ------------------------
    # LoRA configuration
    # ------------------------
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["W_pack", "o_proj", "down_proj", "up_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()
    
    # ------------------------
    # Dataset loading & filtering
    # ------------------------
    dataset_arg = args.dataset_mix if args.dataset_mix else args.dataset
    dataset_names = [d.strip() for d in dataset_arg.split(",") if d.strip()]
    processed_datasets = []

    for name in dataset_names:

        if name == "codeparrot" or name.startswith("codeparrot"):
            if is_main:
                print("📘 Streaming CodeParrot dataset...")
            ds_stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

            if args.dry_run:
                samples_to_take = 500
            else:
                samples_to_take = 2500

            buffer = []

            def filter_and_format(example):
                text = example.get("content", "") or ""
                if len(text.split("\n")) > 50:
                    return None
                lines = [l for l in text.split("\n")
                        if not l.strip().startswith("Copyright")
                        and not l.strip().startswith("Author")
                        and not l.strip().startswith("# This program is free")]
                if not lines:
                    return None
                text_clean = "\n".join(lines)
                formatted_text = f"# Task:\n{text_clean}\n# Solution:\n"
                tokenized = tokenizer(formatted_text, truncation=True, padding="max_length", max_length=args.max_length)
                return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

            count = 0
            for example in ds_stream:
                if count >= samples_to_take:
                    break
                out = filter_and_format(example)
                if out is None:
                    continue
                buffer.append(out)
                count += 1

            if len(buffer) == 0:
                raise RuntimeError("No CodeParrot examples collected (check filters).")

            codeparrot_ds = HFDataset.from_list(buffer)
            processed_datasets.append(codeparrot_ds)

        elif name == "iamtarun/python_code_instructions_18k_alpaca":
            if is_main:
                print("📘 Loading iamtarun/python_code_instructions_18k_alpaca dataset...")
            ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

            if args.dry_run:
                ds = ds.select(range(500))

            def preprocess(batch):
                inputs = [
                    f"### Instruction:\n{inst}\n### Response:\n{out}"
                    for inst, out in zip(batch["instruction"], batch["output"])
                ]
                model_inputs = tokenizer(
                    inputs,
                    truncation=True,
                    padding="max_length",
                    max_length=args.max_length,
                )

                # Create labels (copy of input_ids)
                model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
                return model_inputs

            ds = ds.map(preprocess, batched=True, num_proc=4, remove_columns=ds.column_names)
            alpaca_ds = ds
            processed_datasets.append(alpaca_ds)

        elif name == "synthetic":
            samples = [
                {"text": "# Task: Reverse a string\n# Solution:\ndef reverse_string(s): return s[::-1]"},
                {"text": "# Task: Check palindrome\n# Solution:\ndef is_palindrome(s): return s==s[::-1]"},
            ]
            ds = HFDataset.from_list(samples)

            def preprocess_synth(batch):
                tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)
                tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
                return tokenized

            ds = ds.map(preprocess_synth, batched=True)
            synthetic_ds = ds
            processed_datasets.append(synthetic_ds)

        else:
            raise ValueError(f"Unknown dataset name: {name}")

    if len(processed_datasets) == 0:
        raise ValueError("No datasets were processed. Check --dataset or --dataset_mix values.")

    # concatenate all processed datasets
    merged = concatenate_datasets(processed_datasets, axis=0)

    # Ensure labels exist
    if "labels" not in merged.column_names:
        def add_labels(ex):
            ex["labels"] = ex["input_ids"]
            return ex
        merged = merged.map(add_labels)

    merged.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_loader = DataLoader(
        merged,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------
    # Optimizer + scheduler
    # ------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # compute total steps properly (ceil division if needed)
    steps_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation - 1) // args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if args.lr_scheduler == "linear":
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    # ------------------------
    # Resume from checkpoint if specified
    # ------------------------
    start_epoch = 0
    start_step = 0
    resumed_metrics = None
    
    if args.resume_from:
        checkpoint_state = load_checkpoint(args.resume_from)
        if checkpoint_state:
            opt_state = checkpoint_state.get("optimizer_state")
            sched_state = checkpoint_state.get("scheduler_state")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
            if sched_state is not None:
                try:
                    scheduler.load_state_dict(sched_state)
                except Exception:
                    if is_main:
                        print("⚠️ Could not load scheduler state (incompatible shapes). Continuing without loading scheduler state.")
            start_epoch = checkpoint_state["epoch"]
            start_step = checkpoint_state["step"]
            resumed_metrics = checkpoint_state.get("metrics", {})
            if is_main:
                print(f"✅ Resumed from epoch {start_epoch}, step {start_step}")

    # Prepare for distributed training: prepare model, optimizer, dataloader
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # ------------------------
    # Training loop
    # ------------------------
    model.train()
    global_step = start_step
    total_tokens = resumed_metrics.get("total_tokens", 0) if resumed_metrics else 0
    epoch_losses = resumed_metrics.get("epoch_losses", []) if resumed_metrics else []

    # ------------------------
    # Helper: get progress bar
    # ------------------------
    def get_progress(loader, desc, is_iterable):
        """Returns a tqdm progress bar."""
        if is_main:
            if is_iterable:
                return tqdm(loader, desc=desc, leave=True)
            else:
                return tqdm(loader, desc=desc, total=len(loader), leave=True)
        else:
            return loader

    if is_main:
        print("🚀 Starting LoRA fine-tuning with gradient clipping...")

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        smoothed_loss = None
        epoch_start_time = time.time()

        is_iterable = isinstance(train_loader.dataset, IterableDataset)
        progress_bar = get_progress(train_loader, f"Epoch {epoch+1}/{args.epochs}", is_iterable)

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            # Ensure use_cache is False during forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels", batch["input_ids"]),
                use_cache=False
            )
            loss = outputs.loss / args.gradient_accumulation
            accelerator.backward(loss)
            running_loss += loss.item() * args.gradient_accumulation
            total_tokens += batch["input_ids"].numel()

            # EMA smoothing
            smoothed_loss = loss.item() if smoothed_loss is None else 0.9*smoothed_loss + 0.1*loss.item()

            # --- Dynamic ETA calculation ---
            elapsed = time.time() - epoch_start_time
            steps_done = step + 1
            avg_step_time = elapsed / steps_done
            if not is_iterable:
                total_steps_local = len(train_loader)
                steps_left = total_steps_local - (step + 1)
                eta_sec = avg_step_time * steps_left
            else:
                eta_sec = avg_step_time * (step + 1)

            eta_str = str(timedelta(seconds=int(eta_sec)))

            if is_main:
                # scheduler might be on CPU; get_last_lr may not be available until warmup; handle safely
                try:
                    lr_val = scheduler.get_last_lr()[0]
                except Exception:
                    lr_val = args.learning_rate
                progress_bar.set_postfix({
                    "loss": f"{smoothed_loss:.4f}", 
                    "lr": f"{lr_val:.2e}",
                    "step": global_step,
                    "ETA": eta_str
                })

            # Gradient accumulation step with clipping
            if (step + 1) % args.gradient_accumulation == 0:
                # CRITICAL: Clip gradients before optimizer step
                if args.max_grad_norm > 0:
                    try:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Save periodic checkpoints
                if args.save_every > 0 and global_step % args.save_every == 0:
                    metrics = {
                        "epoch_losses": epoch_losses,
                        "total_tokens": total_tokens,
                        "current_loss": running_loss / max(steps_done, 1)
                    }
                    save_checkpoint(accelerator, model, tokenizer, optimizer, 
                                  scheduler, epoch, global_step, args.output_dir, metrics,
                                  keep_last_n=args.keep_last_n_checkpoints)

            if args.dry_run and step > 200:
                if is_main:
                    print("🧩 Dry-run stopping early")
                break

        # Calculate epoch loss
        avg_loss = running_loss / max(step + 1, 1)
        epoch_losses.append(avg_loss)
        
        if is_main:
            print(f"✅ Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f}")

        # Save end-of-epoch checkpoint
        accelerator.wait_for_everyone()
        if is_main:
            metrics = {
                "epoch_losses": epoch_losses,
                "total_tokens": total_tokens,
                "current_loss": avg_loss
            }
            save_checkpoint(accelerator, model, tokenizer, optimizer,
                          scheduler, epoch, global_step, args.output_dir, metrics,
                          keep_last_n=args.keep_last_n_checkpoints)

    # ------------------------
    # Metrics and final save
    # ------------------------
    total_time = time.time() - start_time
    metrics = {
        "model": args.model_id,
        "dataset": dataset_arg,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "total_steps": global_step,
        "total_tokens": total_tokens,
        "avg_loss_per_epoch": epoch_losses,
        "train_time_sec": total_time,
    }

    timestamp = int(time.time())
    
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, f"metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to: {metrics_path}")

        # Plot loss curve
        if epoch_losses:
            plt.figure(figsize=(8,5))
            plt.plot(range(len(epoch_losses)), epoch_losses, marker="o", color="blue", linewidth=2)
            plt.title("Training Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.grid(True)
            plot_path = os.path.join(args.output_dir, f"loss_plot_{timestamp}.png")
            plt.savefig(plot_path)
            print(f"📈 Loss plot saved to: {plot_path}")
            plt.close()

        # Save final model
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n✅ Training complete! LoRA adapters saved to: {args.output_dir}")
        print(f"⏱️  Total training time: {timedelta(seconds=int(total_time))}")

if __name__ == "__main__":
    main()
