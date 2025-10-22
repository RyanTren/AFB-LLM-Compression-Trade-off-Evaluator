import os
import time
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
import argparse
from datetime import timedelta

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

# ------------------------------
# Parse arguments
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--dataset", type=str, default="codeparrot")
parser.add_argument("--save_every", type=int, default=10000)
parser.add_argument("--max_runtime_hours", type=float, default=3.75, help="Stop after this many hours")
args = parser.parse_args()

# ------------------------------
# Accelerator setup
# ------------------------------
accelerator = Accelerator()
device = accelerator.device
is_main = accelerator.is_main_process
if is_main:
    os.makedirs(args.output_dir, exist_ok=True)

# ------------------------------
# Load model and tokenizer
# ------------------------------
if is_main:
    print(f"🔍 Loading model {args.model_id} ...")

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Try resuming from latest checkpoint
latest_ckpt = None
if os.path.exists(args.output_dir):
    checkpoints = sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")])
    if checkpoints:
        latest_ckpt = os.path.join(args.output_dir, checkpoints[-1])

if latest_ckpt:
    if is_main:
        print(f"🔁 Resuming from checkpoint: {latest_ckpt}")
    model = AutoModelForCausalLM.from_pretrained(latest_ckpt)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_id)

# ------------------------------
# LoRA setup
# ------------------------------
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_proj", "c_attn", "q_attn", "v_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model = model.to(device)

# ------------------------------
# Dataset setup
# ------------------------------
if args.dataset == "codeparrot":
    raw_datasets = load_dataset("codeparrot/codeparrot-clean-valid")
    dataset = raw_datasets["train"].shuffle(seed=42).select(range(5000))  # small subset
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

def tokenize_function(examples):
    outputs = tokenizer(examples["content"], truncation=True, max_length=args.max_length, padding="max_length")
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["content"])
train_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

# ------------------------------
# Optimizer setup
# ------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# ------------------------------
# Training loop
# ------------------------------
start_time = time.time()
global_step = 0
loss_history = []
max_runtime_sec = args.max_runtime_hours * 3600
total_steps = len(train_loader) * args.epochs

if is_main:
    print("🚀 Starting training...")

for epoch in range(args.epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        if (global_step + 1) % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        loss_value = loss.item()
        loss_history.append(loss_value)

        # --- Progress tracking ---
        if is_main and global_step % 100 == 0:
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0
            remaining_steps = total_steps - global_step
            eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_sec)))
            progress_pct = (global_step / total_steps) * 100

            print(
                f"[Epoch {epoch+1}/{args.epochs}] Step {global_step}/{total_steps} "
                f"({progress_pct:.1f}%) | Loss: {loss_value:.4f} | "
                f"⏱ Elapsed: {elapsed_str} | ETA: {eta_str}"
            )

        # --- Periodic checkpoint saving ---
        if is_main and args.save_every > 0 and global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_step{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"💾 Saved checkpoint: {ckpt_dir}")

        # --- Hard stop after max runtime ---
        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime_sec:
            if is_main:
                print(f"⏰ Max runtime of {args.max_runtime_hours} hours reached. Saving checkpoint and exiting...")
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint_runtime_stop_step{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"💾 Saved runtime checkpoint: {ckpt_dir}")
            accelerator.wait_for_everyone()
            exit(0)

# ------------------------------
# Save final model
# ------------------------------
if is_main:
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Training complete. Model saved to {args.output_dir}")

# ------------------------------
# Plot loss
# ------------------------------
if is_main:
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    print(f"📉 Saved loss plot at {args.output_dir}/loss_curve.png")
