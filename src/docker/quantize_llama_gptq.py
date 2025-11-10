#!/usr/bin/env python3
"""
Quantize a LLaMA-family model to 4-bit GPTQ with code-aware calibration.

Highlights:
- Calibrate on code, natural language, or a mix.
- Robust dataset loading: normal HF datasets, plus fallbacks for files-only repos
  (e.g., JSON shard repos like codeparrot/*-clean-*, or Parquet repos).
- Escape hatch: --code-data-files to point directly at shard globs on the Hub.

Examples

  # Code-centric calibration (JSON shards from a files-only repo)
  python quantize_llama_gptq.py \
      --model meta-llama/Meta-Llama-3-8B \
      --output /out/llama3-8b-gptq \
      --calib-source code \
      --code-dataset codeparrot/codeparrot-clean-train \
      --code-data-files 'hf://datasets/codeparrot/codeparrot-clean-train/*.json.gz' \
      --code-split train \
      --code-langs '' \
      --nsamples 256 --seqlen 1024 \
      --bits 4 --group-size 64 \
      --desc-act --true-sequential --damp-percent 0.01

  # Using a scripted dataset (easier resolution)
  python quantize_llama_gptq.py \
      --model meta-llama/Meta-Llama-3-8B \
      --output /out/llama3-8b-gptq \
      --calib-source code \
      --code-dataset bigcode/the-stack-smol \
      --code-split train \
      --code-langs python,javascript,cpp \
      --nsamples 256 --seqlen 1024 \
      --bits 4 --group-size 64 \
      --desc-act --true-sequential --damp-percent 0.01

  # Natural-language calibration (WikiText-2)
  python quantize_llama_gptq.py \
      --model meta-llama/Meta-Llama-3-8B \
      --output /out/llama3-8b-gptq-nl \
      --calib-source nl \
      --nl-dataset wikitext --nl-subset wikitext-2-raw-v1 --nl-split train \
      --nsamples 128 --seqlen 512
"""

import argparse
import json
import os
import random
import time
from typing import Dict, Iterable, Optional, Tuple

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer


# ----------------------------
# Data iteration helpers
# ----------------------------
def _iter_nl_texts(dataset: str, subset: Optional[str], split: str, text_field: str) -> Iterable[str]:
    """Yield NL texts from a (non-streaming) HF dataset."""
    ds = load_dataset(dataset, subset, split=split) if subset else load_dataset(dataset, split=split)
    col = text_field if text_field in ds.column_names else ds.column_names[0]
    for v in ds[col]:
        s = str(v)
        if s.strip():
            yield s


def _iter_code_texts(
    dataset: str,
    split: str,
    langs_csv: str,
    code_text_field: Optional[str],
    data_files: Optional[str] = None,
    data_format: str = "auto",
) -> Iterable[str]:
    """
    Yield code snippets from a HF dataset, robust to files-only repos.

    Resolution order:
      1) If data_files provided:
           - 'parquet' -> load parquet builder
           - 'json' or anything else -> load json builder
      2) Else try the standard dataset builder (load_dataset(dataset, ...))
      3) Fallbacks:
           - parquet: hf://datasets/{dataset}/data/train-*.parquet
           - json:    hf://datasets/{dataset}/*.json.gz
    """
    def _from_files(files: str, fmt: str):
        if fmt.lower() == "parquet" or (isinstance(files, str) and files.endswith(".parquet")):
            return load_dataset("parquet", data_files=files, split=split, streaming=True)
        return load_dataset("json", data_files=files, split=split, streaming=True)

    # Try to build the streaming dataset
    try:
        if data_files:
            ds = _from_files(data_files, data_format)
        else:
            ds = load_dataset(dataset, split=split, streaming=True)
    except Exception:
        # Heuristics for common layouts
        try:
            ds = load_dataset(
                "parquet",
                data_files=f"hf://datasets/{dataset}/data/train-*.parquet",
                split=split,
                streaming=True,
            )
        except Exception:
            ds = load_dataset(
                "json",
                data_files=f"hf://datasets/{dataset}/*.json.gz",
                split=split,
                streaming=True,
            )

    want_langs = {l.strip().lower() for l in langs_csv.split(",")} if (langs_csv is not None and len(langs_csv.strip()) > 0) else set()

    for row in ds:
        # Optional language filter (only if a 'language' column is present)
        if want_langs:
            lang = str(row.get("language", "")).lower()
            if lang and lang not in want_langs:
                continue

        # Choose a text field
        txt = None
        if code_text_field and code_text_field in row and row[code_text_field]:
            txt = row[code_text_field]
        else:
            for cand in ("content", "code", "text", "content_text", "raw_content"):
                if cand in row and row[cand]:
                    txt = row[cand]
                    break

        if txt:
            s = str(txt)
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
    """Assemble GPTQ calibration examples from an iterator of raw strings."""
    random.seed(seed)
    examples = []
    acc = ""

    def massage(s: str) -> str:
        if not prefer_code_boundaries:
            return s
        s = s.replace("\r\n", "\n")
        s = s.replace("\nclass ", "\n\nclass ").replace("\ndef ", "\n\ndef ")
        s = s.replace("```", "\n```\n")
        return s

    used = 0
    for t in text_iter:
        used += 1
        acc += ("\n\n" + massage(str(t)))
        if len(examples) >= nsamples:
            break
        toks = tokenizer(acc, return_tensors="pt", add_special_tokens=False)
        ids = toks["input_ids"][0]
        if ids.numel() >= seqlen:
            start = random.randint(0, ids.numel() - seqlen)
            chunk = ids[start:start + seqlen].unsqueeze(0)  # [1, seqlen]
            examples.append({"input_ids": chunk, "attention_mask": torch.ones_like(chunk)})
            acc = ""  # reset to diversify samples
    if len(examples) < nsamples:
        print(f"[WARN] Built {len(examples)} samples (target {nsamples}).")
    return examples, used


def _parse_mix(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {"code": 0.85, "nl": 0.15}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                out[k.strip().lower()] = float(v)
            except ValueError:
                pass
    total = sum(out.values())
    if total <= 0:
        return {"code": 0.85, "nl": 0.15}
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

    # GPTQ options
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], help="Quantization bits")
    ap.add_argument("--group-size", type=int, default=128, help="GPTQ group size (64/128 typical)")
    ap.add_argument("--desc-act", action="store_true", help="Use desc_act (helps at 4-bit)")
    ap.add_argument("--true-sequential", action="store_true", help="Enable true_sequential pass")
    ap.add_argument("--damp-percent", type=float, default=0.01, help="Hessian dampening percent")
    ap.add_argument("--static-groups", action="store_true", help="Use static_groups during GPTQ")

    # Calibration shape
    ap.add_argument("--nsamples", type=int, default=256, help="Calibration sample count")
    ap.add_argument("--seqlen", type=int, default=1024, help="Tokens per calibration example")

    # Calibration source
    ap.add_argument("--calib-source", default="code", choices=["code", "nl", "mix"],
                    help="Domain for calibration examples")

    # NL dataset args
    ap.add_argument("--nl-dataset", default="wikitext", help="HF dataset id for NL")
    ap.add_argument("--nl-subset", default="wikitext-2-raw-v1", help="HF subset/config for NL")
    ap.add_argument("--nl-split", default="train", help="Split for NL dataset")
    ap.add_argument("--nl-text-field", default="text", help="Text field for NL dataset")

    # Code dataset args
    ap.add_argument("--code-dataset", default="codeparrot/codeparrot-clean-train",
                    help="HF dataset id for code; works with scripts or files-only repos")
    ap.add_argument("--code-split", default="train", help="Split for code dataset")
    ap.add_argument("--code-text-field", default="", help="Override text field if needed (e.g., 'content')")
    ap.add_argument("--code-langs", default="python,javascript,cpp",
                    help="Comma-separated languages to keep (empty string disables filtering)")
    ap.add_argument("--prefer-code-boundaries", action="store_true",
                    help="Heuristics to avoid mid-identifier/mid-fence splits")

    # Files-only escape hatch / explicit format selection
    ap.add_argument("--code-data-files", default="",
                    help="Optional glob/list for shard files, e.g. "
                         "'hf://datasets/codeparrot/codeparrot-clean-train/*.json.gz' "
                         "or 'hf://datasets/codeparrot/github-code-clean/data/train-*.parquet'")
    ap.add_argument("--code-data-format", default="auto", choices=["auto", "json", "parquet"],
                    help="Force data format for --code-data-files if needed")

    # Mixed calibration
    ap.add_argument("--calib-mix", default="code=0.85,nl=0.15",
                    help="For --calib-source mix: e.g., 'code=0.85,nl=0.15'")

    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Tokenizer
    print(f"[INFO] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    quant_cfg = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        true_sequential=args.true_sequential,
        damp_percent=args.damp_percent,
        static_groups=args.static_groups,
    )

    # Load model
    print(f"[INFO] Loading base model (AutoGPTQ) …")
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quantize_config=quant_cfg,
        use_safetensors=True,
        trust_remote_code=True
    )

    # Build calibration examples
    print(f"[INFO] Building calibration set: nsamples={args.nsamples} seqlen={args.seqlen} source={args.calib_source}")
    examples = []
    calib_meta: Dict[str, object] = {
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "source": args.calib_source,
    }

    if args.calib_source == "nl":
        nl_iter = _iter_nl_texts(args.nl_dataset, args.nl_subset, args.nl_split, args.nl_text_field)
        examples, _ = _build_examples_from_iter(tokenizer, nl_iter, args.nsamples, args.seqlen)
        calib_meta.update({
            "nl": {"dataset": args.nl_dataset, "subset": args.nl_subset, "split": args.nl_split,
                   "text_field": args.nl_text_field}
        })

    elif args.calib_source == "code":
        code_iter = _iter_code_texts(
            dataset=args.code_dataset,
            split=args.code_split,
            langs_csv=args.code_langs,
            code_text_field=(args.code_text_field or None),
            data_files=(args.code_data_files or None),
            data_format=args.code_data_format,
        )
        examples, _ = _build_examples_from_iter(
            tokenizer, code_iter, args.nsamples, args.seqlen,
            prefer_code_boundaries=args.prefer_code_boundaries
        )
        calib_meta.update({
            "code": {
                "dataset": args.code_dataset,
                "split": args.code_split,
                "langs": args.code_langs,
                "text_field": args.code_text_field or "(auto)",
                "data_files": args.code_data_files or "(none)",
                "data_format": args.code_data_format,
            }
        })

    else:  # mix
        mix = _parse_mix(args.calib_mix)
        n_code = int(round(args.nsamples * mix.get("code", 0.85)))
        n_nl = max(0, args.nsamples - n_code)

        code_iter = _iter_code_texts(
            dataset=args.code_dataset,
            split=args.code_split,
            langs_csv=args.code_langs,
            code_text_field=(args.code_text_field or None),
            data_files=(args.code_data_files or None),
            data_format=args.code_data_format,
        )
        code_examples, _ = _build_examples_from_iter(
            tokenizer, code_iter, n_code, args.seqlen, prefer_code_boundaries=args.prefer_code_boundaries
        )

        nl_iter = _iter_nl_texts(args.nl_dataset, args.nl_subset, args.nl_split, args.nl_text_field)
        nl_examples, _ = _build_examples_from_iter(tokenizer, nl_iter, n_nl, args.seqlen)

        examples = code_examples + nl_examples
        random.shuffle(examples)
        calib_meta.update({
            "mix": {
                "weights": mix,
                "code": {
                    "dataset": args.code_dataset, "split": args.code_split,
                    "langs": args.code_langs, "text_field": args.code_text_field or "(auto)",
                    "data_files": args.code_data_files or "(none)",
                    "data_format": args.code_data_format
                },
                "nl": {"dataset": args.nl_dataset, "subset": args.nl_subset, "split": args.nl_split,
                       "text_field": args.nl_text_field}
            }
        })

    if len(examples) < args.nsamples:
        print(f"[WARN] Built {len(examples)} calibration examples (target {args.nsamples}).")

    # Quantize
    t0 = time.time()
    print("[INFO] Starting GPTQ quantization …")
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
        "note": "Use use_triton=False on older GPUs; this script is robust to files-only HF datasets."
    }
    with open(os.path.join(args.output, "gptq_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Test-load hint
    print("\n[TEST LOAD] Example:")
    print(f"  from auto_gptq import AutoGPTQForCausalLM")
    print(f"  from transformers import AutoTokenizer")
    print(f"  model = AutoGPTQForCausalLM.from_quantized('{args.output}', device='cuda:0',"
          f" use_safetensors=True, use_triton=False)")
    print(f"  tok = AutoTokenizer.from_pretrained('{args.output}', use_fast=True)")
    print("")


if __name__ == "__main__":
    main()
