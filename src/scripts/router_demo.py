from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

BASE_MODEL = "facebook/opt-1.3b"
LORA_MODEL = "lora_out"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
model_lora = AutoModelForCausalLM.from_pretrained(LORA_MODEL).to(device)

def route_model(prompt: str):
    # Simple heuristic: short prompts => use lora; long prompts => base
    tokens = tokenizer(prompt, return_tensors="pt")
    length = tokens["input_ids"].shape[1]
    if length <= 40:
        return model_lora, "lora"
    else:
        return model_base, "base"

@app.route("/generate", methods=["POST"])
def generate():
    payload = request.json
    prompt = payload.get("prompt","")
    model, name = route_model(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=128)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return jsonify({"model": name, "output": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
