import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from torch import nn

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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

# --- Handle tokenizer & embedding size mismatch ---
old_emb = model.get_input_embeddings()
old_vocab_size, hidden_size = old_emb.weight.shape
new_vocab_size = len(tokenizer)

if new_vocab_size != old_vocab_size:
    print(f"⚠️ Adjusting embeddings: {old_vocab_size} → {new_vocab_size}")
    new_emb = nn.Embedding(new_vocab_size, hidden_size)
    # Copy old embeddings
    new_emb.weight.data[:old_vocab_size, :] = old_emb.weight.data
    # Initialize new embeddings if vocab expanded
    if new_vocab_size > old_vocab_size:
        new_emb.weight.data[old_vocab_size:, :].normal_(mean=0.0, std=0.02)
    model.set_input_embeddings(new_emb)

# --- Load LoRA adapters ---
model = PeftModel.from_pretrained(model, LORA_DIR)
model.to(DEVICE)
model.eval()

# --- Load prompts ---
if os.path.exists(PROMPTS_PATH):
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)
else:
    print(f"⚠️ Prompt file not found at {PROMPTS_PATH}, using default.")
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
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(gen_text)

print("\n✅ Inference complete!")
