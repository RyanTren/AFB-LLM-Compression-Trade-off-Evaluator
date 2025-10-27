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
BASE_MODEL = "deepseek-ai/deepseek-coder-1b-base"  # Must match training model!
ADAPTER_PATH = os.path.abspath("./lora_out_deepseek_1b_v2/checkpoint-epoch4-step2905")  # Point to specific checkpoint
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
# First try loading from checkpoint, then fallback to base model
tokenizer = None
checkpoint_tokenizer_path = os.path.join(ADAPTER_PATH, "tokenizer_config.json")

if os.path.exists(checkpoint_tokenizer_path):
    print(f"🔹 Loading tokenizer from checkpoint: {ADAPTER_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_PATH, 
            use_fast=True,
            local_files_only=True  # CRITICAL: Load from local directory
        )
    except Exception as e:
        print(f"⚠️  Failed to load tokenizer from checkpoint: {e}")

if tokenizer is None:
    print(f"🔹 Loading tokenizer from base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# --- Load base model + LoRA adapter ---
print(f"🔹 Loading {BASE_MODEL} + LoRA adapter from: {ADAPTER_PATH}")

# Check if the checkpoint has the required LoRA files
adapter_config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    print(f"\n❌ ERROR: Missing 'adapter_config.json' in {ADAPTER_PATH}")
    print("\nFiles in checkpoint directory:")
    for f in sorted(os.listdir(ADAPTER_PATH)):
        print(f"  - {f}")
    print("\n💡 This checkpoint may not have been saved correctly during training.")
    print("   Please re-run training with the fixed script to generate proper LoRA checkpoints.")
    exit(1)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.float16,  # Use FP16 for faster inference
    device_map="auto"  # Automatically place on available device
)
model.resize_token_embeddings(len(tokenizer))

# Load LoRA weights from checkpoint
print(f"🔹 Loading LoRA adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(
    model, 
    ADAPTER_PATH, 
    local_files_only=True,
    is_trainable=False
)
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
        "Implement a function to reverse a string",
        "Write a function to check if a string is a palindrome",
        "Create a function to find the longest substring without repeating characters",
        "Implement a function to count character frequency in a string",
        "Write a function to check if two strings are anagrams"
    ]

print(f"🧠 Loaded {len(prompts)} prompts for category '{CATEGORY}'.")

# --- Few-shot examples for prepending ---
few_shot_examples = [
    "# Task: Reverse a string\n# Solution:\ndef reverse_string(s):\n    return s[::-1]\n\n",
    "# Task: Check if a string is palindrome\n# Solution:\ndef is_palindrome(s):\n    return s == s[::-1]\n\n"
]

# --- Generation settings ---
# Using greedy decoding to avoid NaN issues
gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,  # Greedy decoding (deterministic)
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# --- Run inference ---
print("\n" + "="*80)
print(f"🚀 Running inference with checkpoint: {ADAPTER_PATH}")
print("="*80)

# --- Prepare results folder ---
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


for i, prompt in enumerate(prompts):
    # Construct few-shot prompt
    few_shot_prompt = "".join(few_shot_examples[:NUM_FEWSHOT])
    formatted_prompt = f"{few_shot_prompt}# Task: {prompt}\n# Solution:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **gen_kwargs
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt + few-shot examples from the output for clarity
    if result.startswith(formatted_prompt):
        result = result[len(formatted_prompt):]

    print("\n" + "="*80)
    print(f"📝 Prompt {i+1}: {prompt}")
    print("-"*80)
    print(result.strip())

    result_file_path = os.path.join(RESULTS_DIR, f"result_{i+1}.json")
    with open(result_file_path, "w") as f:
        f.write(f"Prompt:\n{prompt}\n\n")
        f.write(f"Generated Solution:\n{result.strip()}\n")

print("\n✅ Inference complete!")