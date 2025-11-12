# serve_llama3_base.py — CPU-safe, dashboard-compatible

import os

# --- Force CPU (must run BEFORE importing torch) ---
# Usage: start the server with:  CUDA_VISIBLE_DEVICES="" USE_CPU=1  uvicorn ...
if os.getenv("USE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide GPUs from PyTorch

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose the HF repo you have access to (base or instruct)
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B")

# ---- Robust device/dtype selection (safe on old GPUs and CPU) ----
use_gpu = torch.cuda.is_available() and os.getenv("USE_CPU", "0") != "1"
bf16_ok = False
if use_gpu:
    major, minor = torch.cuda.get_device_capability(0)
    bf16_ok = major >= 8  # Ampere+ supports bf16 well

# On M40 (sm_52) or CPU: use float32; on newer GPUs optionally bf16
dtype = (
    torch.bfloat16 if (use_gpu and bf16_ok) else
    (torch.float32 if not use_gpu else torch.float32)
)
device_map = "auto" if use_gpu else None

# ---- Load model/tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map=device_map,
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # safest across architectures (no flash/xformers)
)

# Avoid pad-token warnings
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

app = FastAPI(title="Llama 3 Base Server", version="1.0")

# ---- Schemas ----
class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None

class GenResponse(BaseModel):
    response: str

# ---- Health ----
@app.get("/health")
def health():
    dmap = getattr(model, "hf_device_map", "cpu")
    return {
        "cuda": torch.cuda.is_available(),
        "use_gpu": use_gpu,
        "dtype": str(dtype),
        "device_map": str(dmap),
        "model_id": MODEL_ID,
    }

# ---- Generate (matches your dashboard's contract) ----
@app.post("/generate", response_model=GenResponse)
def generate(req: GenerateReq):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    if use_gpu:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = (req.temperature or 0.0) > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(req.max_new_tokens),
            do_sample=do_sample,
            temperature=max(req.temperature, 1e-5) if do_sample else None,
            top_p=req.top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)

    if req.stop:
        for s in req.stop:
            if s and s in text:
                text = text.split(s)[0]

    return GenResponse(response=text)
