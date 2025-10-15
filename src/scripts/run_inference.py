import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Use your GPT-2 + LoRA adapter
BASE_MODEL = "gpt2"
ADAPTER_PATH = "lora_out"

print("🔹 Loading GPT-2 + LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

prompt = "Write a Python function to check if a number is prime:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print("\n🧠 Output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
