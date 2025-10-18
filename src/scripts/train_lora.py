import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# ------------------------
# Argument parser
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, default="lora_out_clean")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--dataset", type=str, default="codeparrot",
                   choices=["synthetic", "codeparrot"])
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()

# ------------------------
# Main training function
# ------------------------
def main():
    args = parse_args()
    start_time = time.time()
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    print(f"🔹 Using model: {args.model_id}")
    print(f"🔹 Dataset: {args.dataset}")

    # ------------------------
    # Load tokenizer + model
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    # ------------------------
    # LoRA configuration
    # ------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(accelerator.device)
    model = accelerator.prepare(model)

    # ------------------------
    # Dataset loading & filtering
    # ------------------------
    if args.dataset == "codeparrot":
        print("📘 Streaming CodeParrot dataset...")
        ds_stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

        # Optional dry-run for debugging
        if args.dry_run:
            ds_stream = ds_stream.take(500)

        def filter_and_format(batch):
            text = batch.get("content", "")
            # skip long files
            if len(text.split("\n")) > 50:
                text = ""
            # remove copyright/license lines
            lines = [l for l in text.split("\n")
                    if not l.strip().startswith("Copyright")
                    and not l.strip().startswith("Author")
                    and not l.strip().startswith("# This program is free")]
            if not lines:
                text = ""
            else:
                text = "\n".join(lines)
            # wrap as Task + Solution
            formatted_text = f"# Task:\n{text}\n# Solution:\n"
            return tokenizer(formatted_text, truncation=True, padding="max_length", max_length=args.max_length)

        # Map and then filter out empty sequences safely
        ds_stream = ds_stream.map(filter_and_format)
        ds_stream = ds_stream.filter(lambda x: len(x["input_ids"]) > 0)


        # Wrap streaming dataset in iterable DataLoader
        class StreamWrapper(IterableDataset):
            def __iter__(self):
                for sample in ds_stream:
                    yield {k: torch.tensor(v) for k, v in sample.items() if k in ["input_ids", "attention_mask"]}

        train_loader = DataLoader(StreamWrapper(), batch_size=args.batch_size)

    elif args.dataset == "synthetic":
        from datasets import Dataset
        ds = Dataset.from_list([
            {"text": "# Task: Reverse a string\n# Solution:\ndef reverse_string(s): return s[::-1]"},
            {"text": "# Task: Check palindrome\n# Solution:\ndef is_palindrome(s): return s==s[::-1]"},
        ])
        def preprocess(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)
        ds = ds.map(preprocess)
        class SyntheticWrapper(IterableDataset):
            def __iter__(self):
                for sample in ds:
                    yield {k: torch.tensor(v) for k, v in sample.items() if k in ["input_ids", "attention_mask"]}
        train_loader = DataLoader(SyntheticWrapper(), batch_size=args.batch_size)

    # ------------------------
    # Optimizer
    # ------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ------------------------
    # Training loop
    # ------------------------
    model.train()
    global_step = 0
    total_tokens = 0
    epoch_losses = []

    def get_progress(loader, desc):
        return tqdm(loader, desc=desc, leave=True) if is_main else loader

    if is_main: print("🚀 Starting LoRA fine-tuning...")

    for epoch in range(args.epochs):
        running_loss = 0.0
        smoothed_loss = None
        progress_bar = get_progress(train_loader, f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["input_ids"])
            loss = outputs.loss / args.gradient_accumulation
            accelerator.backward(loss)
            running_loss += loss.item() * args.gradient_accumulation
            total_tokens += batch["input_ids"].numel()

            # EMA smoothing
            smoothed_loss = loss.item() if smoothed_loss is None else 0.9*smoothed_loss + 0.1*loss.item()
            if is_main:
                progress_bar.set_postfix({"smoothed_loss": f"{smoothed_loss:.4f}"})

            if (global_step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            if args.save_every and global_step > 0 and global_step % args.save_every == 0 and is_main:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint_step{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"💾 Saved intermediate checkpoint: {ckpt_dir}")

            global_step += 1
            if args.dry_run and step > 200:
                if is_main: print("🧩 Dry-run stopping early")
                break

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        if is_main: print(f"✅ Epoch {epoch} complete | Avg loss: {avg_loss:.4f}")

        # Epoch checkpoint
        accelerator.wait_for_everyone()
        if is_main:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"💾 Epoch checkpoint saved to {ckpt_dir}")

    # ------------------------
    # Metrics and final save
    # ------------------------
    total_time = time.time() - start_time
    metrics = {
        "model": args.model_id,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "total_steps": global_step,
        "total_tokens": total_tokens,
        "avg_loss_per_epoch": epoch_losses,
        "train_time_sec": total_time,
    }

    timestamp = int(time.time())
    metrics_path = os.path.join(args.output_dir, f"metrics_{timestamp}.json")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to: {metrics_path}")

        # Plot loss curve
        plt.figure(figsize=(8,5))
        plt.plot(range(len(epoch_losses)), epoch_losses, marker="o", color="blue", linewidth=2)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True)
        plot_path = os.path.join(args.output_dir, f"loss_plot_{timestamp}.png")
        plt.savefig(plot_path)
        print(f"📈 Loss plot saved to: {plot_path}")

        # Final save
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n✅ Training complete! LoRA adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
