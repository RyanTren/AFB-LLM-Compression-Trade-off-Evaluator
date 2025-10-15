import time
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------
# CONFIG
# ---------------------------
BASE_MODEL_ID = "gpt2"
ADAPTER_PATH = "lora_out"
OUTPUT_CSV = "benchmark_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "Write a Python function to check if a number is prime.",
    "Explain what LoRA fine-tuning is in 2 sentences.",
    "Generate a short poem about artificial intelligence.",
    "What is the difference between GPT-2 and LoRA fine-tuning?",
]
MAX_NEW_TOKENS = 60

# ---------------------------
# LOAD MODELS
# ---------------------------
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("🔹 Loading base GPT-2 model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32).to(DEVICE)
base_model.eval()

print("🔹 Loading LoRA fine-tuned GPT-2...")
lora_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32)
lora_model = PeftModel.from_pretrained(lora_model, ADAPTER_PATH).to(DEVICE)
lora_model.eval()

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    end = time.time()
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, end - start

def compute_loss(model, prompt):
    """Compute average cross-entropy loss for comparison."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item() if outputs.loss is not None else float("nan")

# ---------------------------
# RUN COMPARISON
# ---------------------------
results = []
print("\n🚀 Running benchmark...\n")

for prompt in PROMPTS:
    base_text, base_time = generate(base_model, prompt)
    lora_text, lora_time = generate(lora_model, prompt)

    base_loss = compute_loss(base_model, prompt)
    lora_loss = compute_loss(lora_model, prompt)

    results.append({
        "prompt": prompt,
        "base_time": base_time,
        "lora_time": lora_time,
        "base_loss": base_loss,
        "lora_loss": lora_loss,
        "base_text": base_text,
        "lora_text": lora_text,
    })

# ---------------------------
# DISPLAY RESULTS
# ---------------------------
print("📊 Benchmark Results\n" + "-" * 80)
for r in results:
    print(f"\n🧩 Prompt: {r['prompt']}")
    print(f"⏱️  Base Time: {r['base_time']:.2f}s | LoRA Time: {r['lora_time']:.2f}s")
    print(f"📉  Base Loss: {r['base_loss']:.4f} | LoRA Loss: {r['lora_loss']:.4f}\n")
    print(f"🤖 Base GPT-2:\n{r['base_text']}\n")
    print(f"🧠 LoRA GPT-2:\n{r['lora_text']}\n")
    print("-" * 80)

# ---------------------------
# SAVE RESULTS TO CSV
# ---------------------------
fieldnames = [
    "prompt", "base_time", "lora_time", "base_loss", "lora_loss", "base_text", "lora_text"
]

os.makedirs(os.path.dirname(OUTPUT_CCSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Results saved to {OUTPUT_CSV}")
