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

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, default="lora_out_fast")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--dataset", type=str, default="codeparrot",
                   choices=["synthetic", "codeparrot"])
    p.add_argument("--save_every", type=int, default=500,  # checkpoint every N steps
                   help="Save checkpoint every N steps (0 disables)")
    p.add_argument("--dry_run", action="store_true",
                   help="Limit data for quick tests (~10 min)")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    start_time = time.time()
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        print(f"🔹 Model: {args.model_id}")
        print(f"🔹 Dataset: {args.dataset}")

    # -----------------------------
    # Tokenizer + Model
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    # LoRA setup
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

    # -----------------------------
    # Dataset
    # -----------------------------
    if args.dataset == "codeparrot":
        if is_main:
            print("📘 Loading CodeParrot-clean stream...")
        ds_stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

        if args.dry_run:
            ds_stream = ds_stream.take(2000)  # ~10–15 min run

        def filter_and_format(batch):
            text = batch.get("content", "")
            if not text or len(text) < 100 or len(text) > 2000:
                return None
            if "Copyright" in text or "license" in text.lower():
                return None
            formatted = f"# Task:\n{text[:200]}\n# Solution:\n"
            return tokenizer(formatted, truncation=True, padding="max_length", max_length=args.max_length)

        def clean_stream(ds_stream):
            for ex in ds_stream:
                out = filter_and_format(ex)
                if out is not None:
                    yield out

        ds_stream = clean_stream(ds_stream)

        class StreamWrapper(IterableDataset):
            def __iter__(self):
                for sample in ds_stream:
                    yield {k: torch.tensor(v) for k, v in sample.items() if k in ["input_ids", "attention_mask"]}

        train_loader = DataLoader(StreamWrapper(), batch_size=args.batch_size)

    else:
        from datasets import Dataset
        ds = Dataset.from_list([
            {"text": "# Task: Reverse string\n# Solution:\ndef rev(s): return s[::-1]"},
            {"text": "# Task: Fibonacci\n# Solution:\ndef fib(n): a,b=0,1; arr=[]; [arr.append(a) or (a,b:=(b,a+b)) for _ in range(n)]; return arr"},
        ])
        def preprocess(ex): return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=args.max_length)
        ds = ds.map(preprocess)
        class SyntheticWrapper(IterableDataset):
            def __iter__(self):
                for sample in ds:
                    yield {k: torch.tensor(v) for k, v in sample.items() if k in ["input_ids", "attention_mask"]}
        train_loader = DataLoader(SyntheticWrapper(), batch_size=args.batch_size)

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # -----------------------------
    # Training
    # -----------------------------
    model.train()
    global_step, total_tokens = 0, 0
    epoch_losses, smoothed_loss = [], None

    def get_progress(loader, desc):
        return tqdm(loader, desc=desc, leave=True) if is_main else loader

    if is_main:
        print("🚀 Starting fine-tuning...")

    for epoch in range(args.epochs):
        running_loss, step = 0.0, 0
        progress_bar = get_progress(train_loader, f"Epoch {epoch}")

        for batch in progress_bar:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss / args.gradient_accumulation
            accelerator.backward(loss)

            running_loss += loss.item()
            total_tokens += batch["input_ids"].numel()
            smoothed_loss = loss.item() if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * loss.item()

            if (global_step + 1) % args.gradient_accumulation == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if is_main:
                progress_bar.set_postfix({"smoothed_loss": f"{smoothed_loss:.4f}"})

            # Save every N steps
            if args.save_every and global_step > 0 and global_step % args.save_every == 0 and is_main:
                ckpt = os.path.join(args.output_dir, f"checkpoint_step{global_step}")
                os.makedirs(ckpt, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"💾 Saved checkpoint at step {global_step}")

            step += 1
            global_step += 1

            # Limit runtime in dry-run or fast mode (~1h cap)
            if args.dry_run and global_step >= 1000:
                if is_main:
                    print("🧩 Dry-run cap reached (1000 steps).")
                break
            if global_step >= 5000:  # safety cap (~1 hour on GPU)
                if is_main:
                    print("⏹ Early stop for 1-hour cap (5000 steps).")
                break

        avg_loss = running_loss / (step + 1)
        epoch_losses.append(avg_loss)
        if is_main:
            print(f"✅ Epoch {epoch} done | Avg loss: {avg_loss:.4f}")

        accelerator.wait_for_everyone()
        if is_main:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"💾 Saved epoch checkpoint: {ckpt_dir}")

        if global_step >= 5000:  # stop after 5k steps max
            break

    # -----------------------------
    # Metrics & final save
    # -----------------------------
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

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = int(time.time())
        metrics_path = os.path.join(args.output_dir, f"metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to {metrics_path}")

        # Loss plot
        plt.figure(figsize=(8,5))
        plt.plot(range(len(epoch_losses)), epoch_losses, marker="o")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, f"loss_plot_{timestamp}.png"))

        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n✅ Training complete! Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
