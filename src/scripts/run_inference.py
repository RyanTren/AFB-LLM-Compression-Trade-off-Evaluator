import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3.1-8B"

print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("🔹 Setting quantization config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("🔹 Loading model (this may take a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16
)

prompt = "Write a Python function that reverses a string."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("🔹 Generating output...")
with torch.inference_mode():
    output_tokens = model.generate(**inputs, max_new_tokens=100)
print("🔹 Result:")
print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
