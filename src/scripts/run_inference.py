from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "gpt2"
ADAPTER_PATH = "../lora_out"   # ✅ relative to scripts/ folder

print("🔹 Loading GPT-2 + LoRA adapter...")

# Load tokenizer from base model (not the adapter folder)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Load LoRA weights
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

# 🔹 Run a quick test
prompt = "Write a simple Python function that calculates factorial."
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
