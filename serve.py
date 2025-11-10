import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

MODEL_PATH = os.getenv("MODEL_PATH", "/models")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="LLM Serving (GPTQ 4-bit)", version="1.0")

# ---- Load once at startup ----
print(f"[BOOT] Loading model from {MODEL_PATH} on {DEVICE} …")

load_kwargs = dict(
    device=DEVICE,
    use_triton=False,     # correct path for older GPUs (no exllama/triton)
    use_safetensors=True,
)

try:
    # Preferred flags on newer AutoGPTQ (0.5.x/0.6.x)
    model = AutoGPTQForCausalLM.from_quantized(
        MODEL_PATH,
        inject_fused_attention=False,
        inject_fused_mlp=False,
        **load_kwargs
    )
except TypeError:
    # Back-compat for older API name
    model = AutoGPTQForCausalLM.from_quantized(
        MODEL_PATH,
        no_inject_fused_attention=True,
        **load_kwargs
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
print("[BOOT] Model loaded.")

# ---- Schemas ----
class GenRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

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
    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out_tokens = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
    # Return only the completion after the prompt
    completion = text[len(req.prompt):].lstrip()
    return GenResponse(response=completion)
