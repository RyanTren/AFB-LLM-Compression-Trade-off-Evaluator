#!/usr/bin/env python3
"""
Quantize a LLaMA-family model to 4-bit GPTQ and save it for later inference.

Examples:

  # From HF Hub (pass HUGGINGFACE_HUB_TOKEN if gated):
  python quantize_llama_gptq.py \
      --model meta-llama/Meta-Llama-3-8B \
      --output /out/llama3-8b-gptq \
      --nsamples 128 --seqlen 512

  # From a local model directory on the VM:
  python quantize_llama_gptq.py \
      --model /models/llama3.3 \
      --output /out/llama3.3-gptq
"""

import argparse, os, json, time, random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def build_calib_examples(tokenizer, nsamples=128, seqlen=512,
                         dataset="wikitext", subset="wikitext-2-raw-v1", split="train"):
    """
    Create a list[dict] of {"input_ids","attention_mask"} for AutoGPTQ.quantize.
    Keeps RAM small and avoids long tokenization times.
    """
    ds = load_dataset(dataset, subset, split=split)
    texts = [t for t in ds["text"] if t and len(t.strip()) > 0]
    random.seed(42)
    random.shuffle(texts)

    examples = []
    acc = ""
    for t in texts:
        acc += ("\n\n" + t)
        if len(examples) >= nsamples:
            break
        toks = tokenizer(acc, return_tensors="pt", add_special_tokens=False)
        ids = toks["input_ids"][0]
        if ids.numel() >= seqlen:
            start = random.randint(0, ids.numel() - seqlen)
            chunk = ids[start:start + seqlen].unsqueeze(0)  # [1, seqlen]
            examples.append({
                "input_ids": chunk,
                "attention_mask": torch.ones_like(chunk)
            })
            acc = ""  # reset accumulator to diversify samples
    if len(examples) < nsamples:
        print(f"[WARN] Only built {len(examples)} samples (target {nsamples}).")
    return examples


def main():
    ap = argparse.ArgumentParser("LLaMA GPTQ 4-bit quantizer (Maxwell-safe)")
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--output", required=True, help="Where to save the GPTQ model")
    ap.add_argument("--bits", type=int, default=4, choices=[2,3,4,8],
                    help="Quantization bits (4 recommended)")
    ap.add_argument("--group-size", type=int, default=128,
                    help="Grouping for GPTQ (128 is a common sweet spot)")
    ap.add_argument("--desc-act", action="store_true",
                    help="Quantize with desc_act (slower but can improve quality)")
    ap.add_argument("--nsamples", type=int, default=128,
                    help="Calibration sample count")
    ap.add_argument("--seqlen", type=int, default=512,
                    help="Sequence length for calibration")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"[INFO] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    quant_cfg = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
    )

    print(f"[INFO] Loading base model into CPU memory (AutoGPTQ recommended default)")
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quantize_config=quant_cfg,
        use_safetensors=True,
        trust_remote_code=True
    )

    print(f"[INFO] Building calibration set nsamples={args.nsamples} seqlen={args.seqlen}")
    examples = build_calib_examples(tokenizer, args.nsamples, args.seqlen)

    t0 = time.time()
    print("[INFO] Starting GPTQ quantization … this can take a while.")
    # Quantization primarily runs on CPU for safety/compat; it works on Maxwell.
    model.quantize(examples)
    mins = (time.time() - t0) / 60.0
    print(f"[INFO] Quantization done in {mins:.1f} minutes. Saving to {args.output}")

    model.save_quantized(args.output, use_safetensors=True)
    tokenizer.save_pretrained(args.output)

    meta = {
        "source_model": args.model,
        "quantize": {"bits": args.bits, "group_size": args.group_size, "desc_act": args.desc_act},
        "calibration": {"nsamples": args.nsamples, "seqlen": args.seqlen, "dataset": "wikitext-2"},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "gpu_cc52_note": "Built to avoid kernels requiring cc>=7.5; safe for Tesla M40 (Maxwell)."
    }
    with open(os.path.join(args.output, "gptq_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[TEST LOAD] After this completes you can load with:")
    print(f"  from auto_gptq import AutoGPTQForCausalLM")
    print(f"  from transformers import AutoTokenizer")
    print(f"  model = AutoGPTQForCausalLM.from_quantized('{args.output}', device='cuda:0',"
          f" use_safetensors=True, use_triton=False)")
    print(f"  tok = AutoTokenizer.from_pretrained('{args.output}', use_fast=True)")
    print("")


if __name__ == "__main__":
    main()
