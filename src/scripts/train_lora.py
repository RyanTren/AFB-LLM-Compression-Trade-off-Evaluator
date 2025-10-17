import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")  # avoid matplotlib permission issues

import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, default="lora_out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--dataset", type=str, default="synthetic",
                   choices=["synthetic", "codeparrot", "dolly", "openwebtext"])
    p.add_argument("--save_every", type=int, default=0,
                   help="Optional step interval to save intermediate checkpoints (0 = disable)")
    p.add_argument("--dry_run", action="store_true",
                   help="Run only a few batches for debugging")
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    print(f"🔹 Using model: {args.model_id}")
    print(f"🔹 Dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    # === LoRA setup ===
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

    # === Dataset loading ===
    if args.dataset == "codeparrot":
        print("📘 Streaming CodeParrot dataset (progressive load)...")
        ds_stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    # Use --dry_run in command to convert stream into limited iterable for test/debug mode
    if args.dry_run:
        ds_stream = ds_stream.take(500)  # only 500 samples for testing

    # Simple text extraction and tokenization pipeline
    def preprocess(batch):
        text = batch.get("content", "")
        return tokenizer(text, truncation=True, padding="max_length", max_length=args.max_length)

    # Apply lazy map (on the fly tokenization)
    ds_stream = ds_stream.map(preprocess)

    # Wrap in an iterable-style DataLoader
    from torch.utils.data import IterableDataset

    class StreamWrapper(IterableDataset):
        def __iter__(self):
            for sample in ds_stream:
                yield {k: torch.tensor(v) for k, v in sample.items() if k in ["input_ids", "attention_mask"]}

    train_loader = DataLoader(StreamWrapper(), batch_size=args.batch_size)


    # elif args.dataset == "dolly":
    #     if is_main: print("💬 Loading Dolly 15k...")
    #     ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    #     ds = ds.map(lambda ex: {"text": f"### Instruction:\n{ex['instruction']}\n### Response:\n{ex['response']}"})

    # elif args.dataset == "openwebtext":
    #     if is_main: print("🌐 Loading OpenWebText (subset 2%)...")
    #     ds = load_dataset("yhavinga/openwebtext", split="train[:2%]")

    # else:
    #     if is_main: print("🧪 Using synthetic dataset.")
    #     from datasets import Dataset
    #     ds = Dataset.from_list([
    #         {"text": "# Write a Python function that reverses a string.\n def rev(s): return s[::-1]"},
    #         {"text": "# Write a Python function that returns Fibonacci sequence.\n def fib(n): a,b=0,1; arr=[]; [arr.append(a) or (a,b:=b,a+b) for _ in range(n)]; return arr"},
    #     ])

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    global_step = 0
    total_tokens = 0
    epoch_losses = []
    metrics_log = []

    # Helper: use tqdm only on main process
    def get_progress_iterable(loader, desc):
        return tqdm(loader, desc=desc, leave=True) if is_main else loader

    if is_main: print("🚀 Starting LoRA fine-tuning...")

    for epoch in range(args.epochs):
        running_loss = 0.0
        smoothed_loss = None
        progress_bar = get_progress_iterable(train_loader, f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )

            loss = outputs.loss / args.gradient_accumulation
            accelerator.backward(loss)

            running_loss += loss.item() * args.gradient_accumulation
            total_tokens += batch["input_ids"].numel()

            # EMA smoothing for live loss
            smoothed_loss = loss.item() if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * loss.item()
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
                if is_main: print("🧩 Dry-run mode active — stopping early.")
                break

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        metrics_log.append({"epoch": epoch, "avg_loss": avg_loss})
        if is_main:
            print(f"✅ Epoch {epoch} complete | Avg loss: {avg_loss:.4f}")

        # Epoch checkpoint (main process only)
        accelerator.wait_for_everyone()
        if is_main:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"💾 Epoch {epoch} checkpoint saved to {ckpt_dir}")

    total_time = time.time() - start_time

    # === Metrics export ===
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

    accelerator.wait_for_everyone()
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to: {metrics_path}")

        # === Plot loss curve ===
        plt.figure(figsize=(8, 5))
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
        print(f"\n✅ Training complete! LoRA adapters + tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
