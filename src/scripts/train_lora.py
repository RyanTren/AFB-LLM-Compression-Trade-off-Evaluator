import os
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, default="lora_out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "codeparrot", "dolly", "openwebtext"],
        help="Which dataset to use for LoRA fine-tuning",
    )
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print(f"🔹 Using model: {args.model_id}")
    print(f"🔹 Dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    # LoRA config
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

    accelerator = Accelerator()
    model.to(accelerator.device)
    model = accelerator.prepare(model)

    # === Dataset handling ===
    if args.dataset == "codeparrot":
        print("📘 Loading CodeParrot (subset 1%)...")
        ds = load_dataset("codeparrot/codeparrot-clean", split="train[:1%]")
        ds = ds.rename_column("content", "text")

    elif args.dataset == "dolly":
        print("💬 Loading Dolly 15k...")
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        ds = ds.map(lambda ex: {"text": f"### Instruction:\n{ex['instruction']}\n### Response:\n{ex['response']}"})

    elif args.dataset == "openwebtext":
        print("🌐 Loading OpenWebText (subset 2%)...")
        ds = load_dataset("yhavinga/openwebtext", split="train[:2%]")

    else:
        print("🧪 No dataset found; creating tiny synthetic dataset for demo.")
        from datasets import Dataset
        ds = Dataset.from_list([
            {"text": "# Write a Python function that reverses a string.\n def rev(s): return s[::-1]"},
            {"text": "# Write a Python function that returns Fibonacci sequence.\n def fib(n): a,b=0,1; arr=[]; [arr.append(a) or (a,b:=b,a+b) for _ in range(n)]; return arr"},
        ])

    # === Tokenization ===
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=[col for col in ds.column_names if col != "text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    global_step = 0
    total_tokens = 0
    epoch_losses = []
    metrics_log = []

    print("🚀 Starting LoRA fine-tuning...")

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in train_loader:
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

            if (global_step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % 10 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item() * args.gradient_accumulation:.4f}")

            global_step += 1

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        metrics_log.append({"epoch": epoch, "avg_loss": avg_loss})
        print(f"✅ Epoch {epoch} complete | Avg loss: {avg_loss:.4f}")

    total_time = time.time() - start_time

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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

    print(f"\n✅ Training complete! LoRA adapters + tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
