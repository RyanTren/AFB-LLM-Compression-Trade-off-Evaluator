import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Force Hugging Face to use /tmp cache
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

BASE_MODEL = "gpt2"
ADAPTER_PATH = "lora_out_codegen_final"

print("🔹 Loading GPT-2 + LoRA adapter...")

# Load tokenizer and ensure same pad token
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Load base model and resize embedding layer
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.resize_token_embeddings(len(tokenizer))  # ✅ critical fix

# Now safely load LoRA weights
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Send model to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompts = [
    "Write a Python function that reverses a string.",
    "Explain how a neural network learns.",
    "Generate a short poem about AI and the Air Force."
]
for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(f"\n🔹 Prompt: {p}")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=80)

print("\n🔹 Result:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
