import os
from fastapi import FastAPI
from pydantic import BaseModel
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
import torch

MODEL_PATH = os.getenv("MODEL_PATH", "/models")  # mount your quantized folder here
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="LLM Serving (GPTQ 4-bit)", version="1.0")

# ---- Load once at startup ----
print(f"[BOOT] Loading model from {MODEL_PATH} on {DEVICE} …")
model = AutoGPTQForCausalLM.from_quantized(
    MODEL_PATH,
    device=DEVICE,
    use_safetensors=True,
    use_triton=False,        # important for Tesla M40 (Maxwell)
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
print("[BOOT] Model loaded.")

# ---- Schemas ----
class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

class GenResponse(BaseModel):
    response: str

# ---- Routes ----
@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_tokens = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
    # return only the completion after the prompt
    completion = text[len(req.prompt):].lstrip()
    return GenResponse(response=completion)
