import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import Accelerator
import json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="facebook/opt-1.3b")
    p.add_argument("--output_dir", type=str, default="lora_out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    return p.parse_args()

def main():
    args = parse_args()

    # Basic logging
    print("Using model:", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<|pad|>"})

    # Load model with no quantization (we're doing LoRA only)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,  # ✅ change from "dropout" → "lora_dropout"
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Use Accelerate with DeepSpeed offload via config (accelerate launch will pick accelerate_config.yaml)
    accelerator = Accelerator()
    model, = accelerator.prepare(model)

    # Simple dataset: replace with your code-generation dataset
    ds = load_dataset("codeparrot/github-code", split="train[:1%]") if False else load_dataset("json", data_files={"train":"data/code_train.json"}) if os.path.exists("data/code_train.json") else None

    # fallback tiny synthetic dataset if none present
    if ds is None:
        print("No dataset found; creating tiny synthetic dataset for demonstration.")
        texts = [
            {"text":"# Write a python function that reverses a string\n def rev(s):\n     return s[::-1]"},
            {"text":"# Write a python function that returns fibonacci\n def fib(n):\n     a,b=0,1\n     arr=[]\n     for _ in range(n): arr.append(a); a,b=b,a+b\n     return arr"}
        ]
        from datasets import Dataset
        ds = Dataset.from_list(texts)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask"])

    # DataLoader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            # Move to device handled by Accelerator
            outputs = model(**{k: batch[k] for k in ["input_ids","attention_mask"]})
            loss = outputs.loss if hasattr(outputs,'loss') else outputs[0].mean()
            loss = loss / args.gradient_accumulation
            accelerator.backward(loss)
            if (global_step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            if global_step % 10 == 0:
                print(f"Epoch {epoch} step {global_step} loss {loss.item() * args.gradient_accumulation:.4f}")
            global_step += 1

    # Save adapters only (PEFT supports saving just adapter weights)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapters to", args.output_dir)

if __name__ == "__main__":
    main()
