import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Cache setup ---
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# --- Configuration ---
BASE_MODEL = "gpt2"
ADAPTER_PATH = "lora_out_codegen_final"
PROMPTS_PATH = "src/data/code_prompts.json"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50

# Specify category to generate (e.g., "string_algorithms", "math_algorithms", "all")
CATEGORY = "string_algorithms"

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# --- Load base model and LoRA adapter ---
print(f"🔹 Loading GPT-2 + LoRA adapter from: {ADAPTER_PATH}")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# --- Load prompts from JSON ---
if os.path.exists(PROMPTS_PATH):
    with open(PROMPTS_PATH, "r") as f:
        prompt_data = json.load(f)
    if CATEGORY == "all":
        prompts = [p for category_prompts in prompt_data.values() for p in category_prompts]
    else:
        if CATEGORY in prompt_data:
            prompts = prompt_data[CATEGORY]
        else:
            raise ValueError(f"Category '{CATEGORY}' not found in prompts JSON")
else:
    print(f"⚠️ Prompt file not found at {PROMPTS_PATH}, using default prompts.")
    prompts = [
        "Write a Python function that reverses a string.",
        "Explain how a neural network learns.",
        "Generate a short poem about AI and the Air Force."
    ]

print(f"🧠 Loaded {len(prompts)} prompts for category '{CATEGORY}'.")

# --- Generation settings ---
gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

# --- Run inference ---
for i, prompt in enumerate(prompts):
    formatted_prompt = f"# Task: {prompt}\n# Solution:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print(f"📝 Prompt {i+1}: {prompt}")
    print("-"*80)
    print(result)

print("\n✅ Inference complete!")
