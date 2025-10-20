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
ADAPTER_PATH = "./lora_out_clean"
PROMPTS_PATH = "../src/data/code_prompts.json"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.2

CATEGORY = "string_algorithms"
NUM_FEWSHOT = 2  # number of examples to prepend

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# --- Load base model + LoRA adapter ---
print(f"🔹 Loading GPT-2 + LoRA adapter from: {ADAPTER_PATH}")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# --- Load prompts ---
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
        "# Implement a stack using a Python list.",
        "# Implement a queue using collections.deque.",
        "# Implement a linked list with insert and delete methods.",
        "# Implement a binary search tree with insert and search.",
        "# Implement a simple hash table class."
    ]

print(f"🧠 Loaded {len(prompts)} prompts for category '{CATEGORY}'.")

# --- Few-shot examples for prepending ---
few_shot_examples = [
    "# Task: Reverse a string\n# Solution:\ndef reverse_string(s):\n    return s[::-1]\n",
    "# Task: Check if a string is palindrome\n# Solution:\ndef is_palindrome(s):\n    return s == s[::-1]\n"
]

# --- Generation settings ---
gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

# --- Run inference ---
for i, prompt in enumerate(prompts):
    # Construct few-shot prompt
    few_shot_prompt = "".join(few_shot_examples[:NUM_FEWSHOT])
    formatted_prompt = f"{few_shot_prompt}# Task: {prompt}\n# Solution:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt + few-shot examples from the output for clarity
    if result.startswith(formatted_prompt):
        result = result[len(formatted_prompt):]

    print("\n" + "="*80)
    print(f"📝 Prompt {i+1}: {prompt}")
    print("-"*80)
    print(result.strip())

print("\n✅ Inference complete!")
