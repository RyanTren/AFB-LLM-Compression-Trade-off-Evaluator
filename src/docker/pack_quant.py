import os, json, time, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

MODEL = "meta-llama/Meta-Llama-3-8B"
OUT   = "/out/llama3-8b-gptq"
NS, SEQ = 256, 1024  # try 64/512 first for a smoke test

print("[INFO] tokenizer …")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

def token_stream():
    ds = load_dataset("json",
        data_files="hf://datasets/codeparrot/codeparrot-clean-train/*.json.gz",
        split="train", streaming=True)
    for r in ds:
        c = r.get("content")
        if c:
            ids = tok(c, add_special_tokens=False).input_ids
            if ids: 
                yield ids

def build_examples(n=NS, seqlen=SEQ):
    buf, ex, seen = [], [], 0
    for ids in token_stream():
        seen += 1
        buf.extend(ids)
        while len(buf) >= seqlen and len(ex) < n:
            chunk = torch.tensor(buf[:seqlen]).unsqueeze(0)
            buf = buf[seqlen:]
            ex.append({"input_ids": chunk, "attention_mask": torch.ones_like(chunk)})
        if len(ex) >= n: break
    print(f"[INFO] rows_seen={seen}, tokens_buffered~={len(buf)}, examples={len(ex)}")
    return ex

print("[INFO] building packed examples …")
examples = build_examples()
if len(examples) == 0:
    raise SystemExit("No examples built – check dataset access/text field")

qcfg = BaseQuantizeConfig(bits=4, group_size=64, desc_act=True, true_sequential=True, damp_percent=0.01)
print("[INFO] loading model …")
m = AutoGPTQForCausalLM.from_pretrained(MODEL, quantize_config=qcfg, use_safetensors=True, trust_remote_code=True)

print("[INFO] quantizing …"); t0=time.time()
m.quantize(examples)
print(f"[INFO] done in {(time.time()-t0)/60:.1f} min; saving to {OUT}")
os.makedirs(OUT, exist_ok=True)
m.save_quantized(OUT, use_safetensors=True)
tok.save_pretrained(OUT)
with open(os.path.join(OUT,"gptq_metadata.json"),"w") as f:
    json.dump({"source_model":MODEL,"nsamples":NS,"seqlen":SEQ,"source":"code",
               "dataset":"codeparrot/codeparrot-clean-train",
               "text_field":"content","packing":"token-pack"}, f, indent=2)
print("[INFO] saved.")
