#!/usr/bin/env python3
"""
Quantize a LLaMA-family model to 4-bit GPTQ with code-aware calibration.

Examples:

  # Code-centric calibration (streaming from The Stack dedup)
  python quantize_llama_gptq.py \
      --model meta-llama/Meta-Llama-3-8B \
      --output /out/llama3-8b-gptq-code \
      --calib-source code \
      --code-dataset bigcode/the-stack-dedup \
      --code-langs python,javascript,cpp \
      --nsamples 384 --seqlen 1024 --desc-act --true-sequential --damp-percent 0.01

  # Natural-language calibration (as in your original, WikiText-2)
  python quantize_llama_gptq.py \
      --model meta-llama/Meta-Llama-3-8B \
      --output /out/llama3-8b-gptq-nl \
      --calib-source nl \
      --nl-dataset wikitext --nl-subset wikitext-2-raw-v1 --nl-split train \
      --nsamples 128 --seqlen 512

  # Mixed calibration (e.g., 85% code + 15% NL for chat-style prompting)
  python quantize_llama_gptq.py \
      --model meta-llama/Meta-Llama-3-8B \
      --output /out/llama3-8b-gptq-mix \
      --calib-source mix --calib-mix code=0.85,nl=0.15 \
      --code-dataset bigcode/the-stack-dedup --code-langs python,java \
      --nl-dataset wikitext --nl-subset wikitext-2-raw-v1
"""

import argparse, os, json, time, random
from typing import Dict, Iterable, Optional, Tuple
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# ----------------------------
# Helpers: data iteration
# ----------------------------
def _iter_nl_texts(dataset: str, subset: Optional[str], split: str, text_field: str) -> Iterable[str]:
    """Yield natural-language texts from a small HF dataset (non-streaming)."""
    if subset and subset.lower() not in ("none", ""):
        ds = load_dataset(dataset, subset, split=split)
    else:
        ds = load_dataset(dataset, split=split)
    cols = ds.column_names
    col = text_field if text_field in cols else cols[0]
    for t in ds[col]:
        if t and str(t).strip():
            yield str(t)


def _iter_code_texts(dataset: str, split: str, langs_csv: str, code_text_field: Optional[str]) -> Iterable[str]:
    """
    Yield code snippets from a large HF dataset (streaming). Falls back to any of
    ['content', 'code', 'text'] unless --code-text-field is provided.
    """
    ds = load_dataset(dataset, split=split, streaming=True)
    want_langs = {l.strip().lower() for l in langs_csv.split(",")} if langs_csv else set()
    for row in ds:
        lang = str(row.get("language", "")).lower()
        if want_langs and lang not in want_langs:
            continue
        text = None
        if code_text_field and code_text_field in row and row[code_text_field]:
            text = row[code_text_field]
        else:
            for f in ("content", "code", "text"):
                if f in row and row[f]:
                    text = row[f]
                    break
        if text:
            s = str(text)
            if s.strip():
                yield s


def _build_examples_from_iter(
    tokenizer,
    text_iter: Iterable[str],
    nsamples: int,
    seqlen: int,
    prefer_code_boundaries: bool = False,
    seed: int = 42,
) -> Tuple[list, int]:
    """
    Assemble GPTQ calibration examples from an iterator of raw strings.
    Returns (examples, used_count).
    """
    random.seed(seed)
    examples = []
    acc = ""

    def _massage_code_boundaries(txt: str) -> str:
        if not prefer_code_boundaries:
            return txt
        # Cheap heuristics to reduce mid-symbol splits
        txt = txt.replace("\r\n", "\n")
        txt = txt.replace("\nclass ", "\n\nclass ")
        txt = txt.replace("\ndef ", "\n\ndef ")
        txt = txt.replace("```", "\n```\n")
        return txt

    used = 0
    for t in text_iter:
        used += 1
        acc += ("\n\n" + _massage_code_boundaries(str(t)))
        if len(examples) >= nsamples:
            break
        toks = tokenizer(acc, return_tensors="pt", add_special_tokens=False)
        ids = toks["input_ids"][0]
        if ids.numel() >= seqlen:
            start = random.randint(0, ids.numel() - seqlen)
            chunk = ids[start:start + seqlen].unsqueeze(0)  # [1, seqlen]
            examples.append({"input_ids": chunk, "attention_mask": torch.ones_like(chunk)})
            acc = ""  # reset to diversify samples
    return examples, used


def _parse_mix(s: str) -> Dict[str, float]:
    """
    Parse mix like 'code=0.85,nl=0.15'.
    """
    out: Dict[str, float] = {"code": 0.85, "nl": 0.15}  # sensible default
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip().lower()
            try:
                out[k] = float(v)
            except ValueError:
                pass
    total = sum(out.values())
    if total <= 0:
        return {"code": 0.85, "nl": 0.15}
    # normalize
    for k in list(out.keys()):
        out[k] = out[k] / total
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("LLaMA GPTQ 4-bit quantizer (code-aware)")
    # Core
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--output", required=True, help="Where to save the GPTQ model")

    # GPTQ knobs
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], help="Quantization bits")
    ap.add_argument("--group-size", type=int, default=128, help="GPTQ group size (64/128 common)")
    ap.add_argument("--desc-act", action="store_true", help="Use desc_act (often helps at 4-bit)")
    ap.add_argument("--true-sequential", action="store_true", help="Enable true_sequential pass")
    ap.add_argument("--damp-percent", type=float, default=0.01, help="Hessian dampening percent")
    ap.add_argument("--static-groups", action="store_true", help="Use static_groups during GPTQ")

    # Calibration shape
    ap.add_argument("--nsamples", type=int, default=256, help="Calibration sample count")
    ap.add_argument("--seqlen", type=int, default=1024, help="Tokens per calibration example")

    # Calibration source & datasets
    ap.add_argument("--calib-source", default="code", choices=["code", "nl", "mix"],
                    help="Domain for calibration examples")
    # NL dataset args (small, non-streaming)
    ap.add_argument("--nl-dataset", default="wikitext", help="HF dataset id for NL")
    ap.add_argument("--nl-subset", default="wikitext-2-raw-v1", help="HF subset/config for NL")
    ap.add_argument("--nl-split", default="train", help="Split for NL dataset")
    ap.add_argument("--nl-text-field", default="text", help="Text field name for NL dataset")

    # Code dataset args (large, streaming)
    ap.add_argument("--code-dataset", default="codeparrot/codeparrot-clean",
                    help="HF dataset id for code (streaming strongly suggested for very large sets)")
    ap.add_argument("--code-split", default="train", help="Split for code dataset (often 'train')")
    ap.add_argument("--code-text-field", default="", help="Override text field for code dataset if needed")
    ap.add_argument("--code-langs", default="python,javascript,cpp",
                    help="Comma-separated list (filters by 'language' column if present)")
    ap.add_argument("--prefer-code-boundaries", action="store_true",
                    help="Heuristics to avoid mid-identifier/mid-fence splits")

    # Mixed calibration
    ap.add_argument("--calib-mix", default="code=0.85,nl=0.15",
                    help="For --calib-source mix: e.g., 'code=0.85,nl=0.15'")

    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Tokenizer
    print(f"[INFO] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Avoid padding warnings if you later batch/pad
    if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantize config
    quant_cfg = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        true_sequential=args.true_sequential,
        damp_percent=args.damp_percent,
        static_groups=args.static_groups,
    )

    print(f"[INFO] Loading base model (AutoGPTQ) …")
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quantize_config=quant_cfg,
        use_safetensors=True,
        trust_remote_code=True
    )

    # Build calibration
    print(f"[INFO] Building calibration set: nsamples={args.nsamples} seqlen={args.seqlen} source={args.calib_source}")
    examples = []
    calib_meta: Dict[str, object] = {
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "source": args.calib_source,
    }

    if args.calib_source == "nl":
        nl_iter = _iter_nl_texts(args.nl_dataset, args.nl_subset, args.nl_split, args.nl_text_field)
        examples, _ = _build_examples_from_iter(
            tokenizer, nl_iter, args.nsamples, args.seqlen, prefer_code_boundaries=False
        )
        calib_meta.update({
            "nl": {"dataset": args.nl_dataset, "subset": args.nl_subset, "split": args.nl_split,
                   "text_field": args.nl_text_field}
        })

    elif args.calib_source == "code":
        code_iter = _iter_code_texts(args.code_dataset, args.code_split, args.code_langs, args.code_text_field or None)
        examples, _ = _build_examples_from_iter(
            tokenizer, code_iter, args.nsamples, args.seqlen,
            prefer_code_boundaries=args.prefer_code_boundaries
        )
        calib_meta.update({
            "code": {"dataset": args.code_dataset, "split": args.code_split,
                     "langs": args.code_langs, "text_field": args.code_text_field or "(auto)"},
        })

    else:  # mix
        mix = _parse_mix(args.calib_mix)
        n_code = int(round(args.nsamples * mix.get("code", 0.85)))
        n_nl = max(0, args.nsamples - n_code)

        code_iter = _iter_code_texts(args.code_dataset, args.code_split, args.code_langs, args.code_text_field or None)
        code_examples, _ = _build_examples_from_iter(
            tokenizer, code_iter, n_code, args.seqlen, prefer_code_boundaries=args.prefer_code_boundaries
        )

        nl_iter = _iter_nl_texts(args.nl_dataset, args.nl_subset, args.nl_split, args.nl_text_field)
        nl_examples, _ = _build_examples_from_iter(
            tokenizer, nl_iter, n_nl, args.seqlen, prefer_code_boundaries=False
        )

        examples = code_examples + nl_examples
        random.shuffle(examples)
        calib_meta.update({
            "mix": {"weights": mix,
                    "code": {"dataset": args.code_dataset, "split": args.code_split,
                             "langs": args.code_langs, "text_field": args.code_text_field or "(auto)"},
                    "nl": {"dataset": args.nl_dataset, "subset": args.nl_subset, "split": args.nl_split,
                           "text_field": args.nl_text_field}
                    }
        })

    if len(examples) < args.nsamples:
        print(f"[WARN] Built {len(examples)} calibration examples (target {args.nsamples}). "
              f"Consider increasing dataset size or switching sources.")

    # Quantize
    t0 = time.time()
    print("[INFO] Starting GPTQ quantization … this can take a while.")
    # Your original quantization path (CPU-safe / Maxwell-friendly). :contentReference[oaicite:1]{index=1}
    model.quantize(examples)
    mins = (time.time() - t0) / 60.0
    print(f"[INFO] Quantization done in {mins:.1f} minutes. Saving to {args.output}")

    # Save
    model.save_quantized(args.output, use_safetensors=True)
    tokenizer.save_pretrained(args.output)

    meta = {
        "source_model": args.model,
        "quantize": {
            "bits": args.bits, "group_size": args.group_size, "desc_act": args.desc_act,
            "true_sequential": args.true_sequential, "damp_percent": args.damp_percent,
            "static_groups": args.static_groups
        },
        "calibration": calib_meta,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "note": "Quantized with AutoGPTQ 4-bit; keep use_triton=False for older GPUs (e.g., Maxwell)."
    }
    with open(os.path.join(args.output, "gptq_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Test-load hint (matches your previous print, triton off)
    print("\n[TEST LOAD] Example:")
    print(f"  from auto_gptq import AutoGPTQForCausalLM")
    print(f"  from transformers import AutoTokenizer")
    print(f"  model = AutoGPTQForCausalLM.from_quantized('{args.output}', device='cuda:0',"
          f" use_safetensors=True, use_triton=False)")
    print(f"  tok = AutoTokenizer.from_pretrained('{args.output}', use_fast=True)")
    print("")


if __name__ == "__main__":
    main()
