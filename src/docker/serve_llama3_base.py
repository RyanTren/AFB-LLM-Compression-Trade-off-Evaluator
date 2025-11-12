from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose one you have access to:
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B")  # or "...-Instruct"

# If GPU is available, bf16 is ideal. Otherwise float32 on CPU (slow).
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Avoid pad token warnings
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

app = FastAPI()

class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None

@app.post("/generate")
def generate(req: GenerateReq):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    if torch.cuda.is_available():
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

    return {"response": text}
