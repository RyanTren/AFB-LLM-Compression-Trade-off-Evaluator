#!/usr/bin/env python3
import os, json, time, torch
from contextlib import contextmanager
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# ---------------- env helpers ----------------
def env_str(name, default): return os.getenv(name, default)
def env_int(name, default): return int(os.getenv(name, str(default)))
def env_float(name, default): return float(os.getenv(name, str(default)))
def env_bool(name, default=False):
    v = os.getenv(name)
    return (str(v).lower() in {"1","true","yes","y","on"}) if v is not None else default

MODEL       = env_str("MODEL", "meta-llama/Meta-Llama-3-8B")
OUTDIR      = env_str("OUTDIR", "/out/llama3-8b-gptq")
DATA_FILES  = env_str("DATA_FILES", "hf://datasets/codeparrot/codeparrot-clean-train/*.json.gz")
DATA_FORMAT = env_str("DATA_FORMAT", "json").lower()   # 'json' or 'parquet'
TEXT_FIELD  = env_str("TEXT_FIELD", "content")

NSAMPLES    = env_int("NSAMPLES", 256)
SEQLEN      = env_int("SEQLEN", 1024)

BITS        = env_int("BITS", 4)
GROUP_SIZE  = env_int("GROUP_SIZE", 64)
DESC_ACT    = env_bool("DESC_ACT", True)
TRUE_SEQ    = env_bool("TRUE_SEQUENTIAL", True)
DAMP_PCT    = env_float("DAMP_PERCENT", 0.01)

# ------------- avoid dataset 401s -------------
@contextmanager
def without_hf_token():
    saved = {k: os.environ.pop(k, None) for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

# ---------------- tokenizer ----------------
print(f"[INFO] tokenizer for {MODEL} …")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

# ----------- dataset → token stream ----------
def token_stream():
    """Stream token ids from HF dataset shards. Token is **suppressed** here."""
    with without_hf_token():
        if DATA_FORMAT == "parquet":
            ds = load_dataset("parquet", data_files=DATA_FILES, split="train", streaming=True)
        else:
            ds = load_dataset("json",    data_files=DATA_FILES, split="train", streaming=True)

        for r in ds:
            c = r.get(TEXT_FIELD)
            if not c:
                continue
            ids = tok(c, add_special_tokens=False).input_ids
            if ids:
                yield ids

# --------- build calibration examples --------
def build_examples(n=NSAMPLES, seqlen=SEQLEN):
    buf, ex, seen = [], [], 0
    for ids in token_stream():
        seen += 1
        buf.extend(ids)
        while len(buf) >= seqlen and len(ex) < n:
            chunk = torch.tensor(buf[:seqlen]).unsqueeze(0)
            buf = buf[seqlen:]
            ex.append({"input_ids": chunk, "attention_mask": torch.ones_like(chunk)})
        if len(ex) >= n:
            break
    print(f"[INFO] rows_seen={seen}, tokens_buffered~={len(buf)}, examples={len(ex)}")
    return ex

print(f"[INFO] building packed examples (n={NSAMPLES}, seqlen={SEQLEN}) …")
examples = build_examples()
if len(examples) == 0:
    raise SystemExit(
        "No examples built. Check DATA_FILES/TEXT_FIELD and that the dataset is reachable."
    )

# ---------------- quantize ----------------
qcfg = BaseQuantizeConfig(
    bits=BITS, group_size=GROUP_SIZE, desc_act=DESC_ACT,
    true_sequential=TRUE_SEQ, damp_percent=DAMP_PCT
)
print("[INFO] loading base model …")
model = AutoGPTQForCausalLM.from_pretrained(
    MODEL, quantize_config=qcfg, use_safetensors=True, trust_remote_code=True
)

print("[INFO] quantizing …")
t0 = time.time()
model.quantize(examples)
print(f"[INFO] done in {(time.time()-t0)/60:.1f} min; saving to {OUTDIR}")

os.makedirs(OUTDIR, exist_ok=True)
model.save_quantized(OUTDIR, use_safetensors=True)
tok.save_pretrained(OUTDIR)

meta = {
    "source_model": MODEL,
    "nsamples": NSAMPLES, "seqlen": SEQLEN, "source": "code",
    "dataset": {"files": DATA_FILES, "format": DATA_FORMAT, "text_field": TEXT_FIELD},
    "quantize": {"bits": BITS, "group_size": GROUP_SIZE, "desc_act": DESC_ACT,
                 "true_sequential": TRUE_SEQ, "damp_percent": DAMP_PCT},
    "packing": "token-pack"
}
with open(os.path.join(OUTDIR, "gptq_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("[INFO] saved.")
