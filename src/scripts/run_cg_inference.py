import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==== CONFIGURATION ====
BASE_MODEL = "gpt2"  # or "codeparrot/codeparrot-small"
LORA_DIR = "lora_out_codegen_final"  # folder from your LoRA fine-tuning
PROMPTS_FILE = "data/code_prompts.json"  # <-- updated path
OUTPUT_FILE = "results/inference_outputs.json"
MAX_NEW_TOKENS = 150

# ==== SETUP ====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

print(f"🔹 Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

print(f"🔹 Loading LoRA adapters from: {LORA_DIR}")
model = PeftModel.from_pretrained(model, LORA_DIR)
model.to(device)
model.eval()

os.makedirs("results", exist_ok=True)

# ==== LOAD PROMPTS ====
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", PROMPTS_FILE)
PROMPTS_PATH = os.path.abspath(PROMPTS_PATH)

print(f"🔹 Loading prompts from: {PROMPTS_PATH}")
with open(PROMPTS_PATH, "r") as f:
    prompt_groups = json.load(f)

outputs = {}

# ==== GENERATION LOOP ====
for group, prompts in prompt_groups.items():
    print(f"\n🧩 Generating group: {group} ({len(prompts)} prompts)")
    outputs[group] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print(f"\n--- PROMPT ---\n{prompt}\n\n--- OUTPUT ---\n{gen_text}\n{'-'*60}")
        outputs[group].append({"prompt": prompt, "output": gen_text})

# ==== SAVE RESULTS ====
with open(OUTPUT_FILE, "w") as f:
    json.dump(outputs, f, indent=2)

print(f"\n✅ All generations saved to: {OUTPUT_FILE}")
