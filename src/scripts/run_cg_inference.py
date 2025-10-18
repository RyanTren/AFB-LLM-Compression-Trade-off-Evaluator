import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIG ---
BASE_MODEL = "gpt2"
LORA_DIR = "lora_out_codegen_final"
PROMPTS_PATH = "data/code_prompts.json"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SETUP ---
print(f"🔹 Using device: {DEVICE}")
print(f"🔹 Loading base model: {BASE_MODEL}")
print(f"🔹 Loading LoRA adapters from: {LORA_DIR}")

# Load tokenizer (do NOT add pad tokens to avoid embedding mismatch)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

# --- Load LoRA adapter ---
model = PeftModel.from_pretrained(model, LORA_DIR)
model.to(DEVICE)
model.eval()

# --- Load prompts ---
if os.path.exists(PROMPTS_PATH):
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)
else:
    print(f"⚠️ Prompt file not found at {PROMPTS_PATH}, using default prompts.")
    prompts = [
        "Write a Python function that reverses a string.",
        "Implement a Fibonacci sequence generator in Python.",
    ]

print(f"🧠 Loaded {len(prompts)} code prompts.")

# --- Generate ---
for i, prompt in enumerate(prompts):
    print("\n" + "=" * 80)
    print(f"📝 Prompt {i+1}: {prompt}")
    print("-" * 80)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,  # use eos instead of pad
        )

    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(gen_text)

print("\n✅ Inference complete!")
