import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sacrebleu
import argparse
import statistics

def load_model(model_dir_or_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
    model = AutoModelForCausalLM.from_pretrained(model_dir_or_name, torch_dtype=torch.float32)
    model.to(device)
    return tokenizer, model, device

def generate_one(prompt, tokenizer, model, device, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.time() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text, latency

def memory_mb():
    p = psutil.Process()
    mem = p.memory_info().rss / 1024.0**2
    return mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--lora_model", type=str, default="lora_out")
    parser.add_argument("--prompts_file", type=str, default="data/code_eval_prompts.json")
    args = parser.parse_args()

    # Dummy prompts if none
    prompts = [
        {"prompt":"Write a Python function that checks if a string is palindrome." , "reference":"def is_pal(s):\n    return s==s[::-1]"},
        {"prompt":"Write a function that sorts a list using bubble sort.", "reference":"def bubble_sort(a):\n    n=len(a)\n    for i in range(n):\n        for j in range(0,n-i-1):\n            if a[j] > a[j+1]: a[j], a[j+1]=a[j+1],a[j]\n    return a"}
    ]

    # Load both models (base and LoRA adapter model). We'll compare both.
    print("Loading base model...")
    tok_base, model_base, dev = load_model(args.base_model)
    print("Loading LoRA model (adapters)...")
    tok_lora, model_lora, _ = load_model(args.lora_model)

    results = []
    for p in prompts:
        print("Prompt:", p["prompt"])
        mem_before = memory_mb()
        out_base, lat_base = generate_one(p["prompt"], tok_base, model_base, dev)
        mem_after = memory_mb()
        print("Base latency:", lat_base, "mem (MB):", mem_after - mem_before)
        out_lora, lat_lora = generate_one(p["prompt"], tok_lora, model_lora, dev)
        print("LoRA latency:", lat_lora)
        # compute BLEU
        ref = [p["reference"]]
        cand_base = out_base
        cand_lora = out_lora
        bleu_base = sacrebleu.corpus_bleu([cand_base], [ref]).score
        bleu_lora = sacrebleu.corpus_bleu([cand_lora], [ref]).score
        results.append({
            "prompt": p["prompt"],
            "bleu_base": bleu_base,
            "bleu_lora": bleu_lora,
            "latency_base": lat_base,
            "latency_lora": lat_lora
        })
        print("BLEU base:", bleu_base, "lora:", bleu_lora)

    # Aggregate
    print("Summary:")
    for k in ["bleu_base","bleu_lora","latency_base","latency_lora"]:
        vals = [r[k] for r in results]
        print(k, statistics.mean(vals), statistics.pstdev(vals))

if __name__ == "__main__":
    main()
