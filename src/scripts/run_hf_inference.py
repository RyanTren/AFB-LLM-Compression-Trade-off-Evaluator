import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

print("🔹 Loading tokenizer and model (CPU mode)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Force CPU, disable quantization
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map=None
).to(device)

print("✅ Model loaded.")

prompt = "Write a Python function that reverses a string."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("🔹 Generating...")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=80)

print("🔹 Result:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
