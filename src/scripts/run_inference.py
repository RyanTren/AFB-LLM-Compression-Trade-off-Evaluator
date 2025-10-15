from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

BASE_MODEL = "gpt2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "../lora_out")
ADAPTER_PATH = os.path.abspath(ADAPTER_PATH)

print("🔹 Loading GPT-2 + LoRA adapter...")

# Load tokenizer from base model (not the adapter folder)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Bug Fix: this ensures same vocab size
model.resize_token_embeddings(len(tokenizer))

# Load LoRA weights
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

# 🔹 Run a quick test
prompt = "Write a simple Python function that calculates factorial."
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
