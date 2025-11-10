import os
import subprocess
import json
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch

st.set_page_config(page_title="AFB Robins | LLM-LoRA Compression Dashboard", layout="wide")

st.title("AFB Robins | LLM Compression Dashboard (LoRA vs Full Model)")

# --------------------------
# Sidebar configuration
# --------------------------
st.sidebar.header("⚙️ Training Configuration")
model_id = st.sidebar.text_input("Base Model", "deepseek-ai/deepseek-coder-1b-base")
epochs = st.sidebar.number_input("Epochs", 1, 10, 3)
batch_size = st.sidebar.number_input("Batch Size", 1, 8, 2)
learning_rate = st.sidebar.number_input("Learning Rate", 1e-6, 1e-3, 1e-5, format="%.1e")
dataset = st.sidebar.selectbox("Dataset", ["synthetic", "iamtarun/python_code_instructions_18k_alpaca"])
output_dir = st.sidebar.text_input("Output Directory", "./lora_output")
dry_run = st.sidebar.checkbox("Dry Run", False)

# --------------------------
# TRAINING
# --------------------------
st.subheader("🚀 Train LoRA Model")

train_btn = st.button("Start Training")

if train_btn:
    st.write("🔄 Starting training... please wait.")
    cmd = [
        "python", "deepseek_coder1B_train_lora.py",
        "--model_id", model_id,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--dataset", dataset,
        "--output_dir", output_dir
    ]
    if dry_run:
        cmd.append("--dry_run")

    start_time = time.time()
    with st.spinner("Training in progress... this may take a while"):
        process = subprocess.run(cmd, capture_output=True, text=True)
    st.write("✅ Training complete.")
    st.text_area("Training Output", process.stdout, height=250)
    
    # Find latest metrics file
    metrics_files = [f for f in os.listdir(output_dir) if f.startswith("metrics_")]
    if metrics_files:
        latest = max(metrics_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
        with open(os.path.join(output_dir, latest)) as f:
            metrics = json.load(f)
        st.json(metrics)

        # Plot loss curve if available
        if "avg_loss_per_epoch" in metrics:
            plt.plot(metrics["avg_loss_per_epoch"], marker="o")
            plt.title("Training Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            st.pyplot(plt)

# --------------------------
# INFERENCE
# --------------------------
st.subheader("🧠 Inference")

checkpoint_path = st.text_input("Checkpoint Path", f"{output_dir}")
run_infer = st.button("Run Inference")

if run_infer:
    st.write("⚙️ Running inference...")
    cmd = ["python", "run_inference_deepseek.py"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    st.text_area("Inference Output", process.stdout, height=250)
    st.success("Inference complete!")

# --------------------------
# EVALUATION
# --------------------------
st.subheader("📊 Evaluation Metrics")

if st.button("Evaluate BLEU / CodeBLEU"):
    # You can use CodeBLEU library: pip install codebleu
    from codebleu import calc_code_bleu

    refs = []
    hyps = []

    for file in os.listdir("results"):
        if file.endswith(".json"):
            with open(os.path.join("results", file)) as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    refs.append(lines[1])  # expected answer (optional)
                    hyps.append(lines[3])  # generated output

    if refs and hyps:
        codebleu_score = calc_code_bleu(refs, hyps, lang="python")
        st.write("✅ CodeBLEU Score:")
        st.json(codebleu_score)
    else:
        st.warning("No inference results found to evaluate.")

# --------------------------
# SYSTEM METRICS
# --------------------------
st.subheader("🧮 System Metrics")

if torch.cuda.is_available():
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    st.write(f"💾 GPU Memory Used: {gpu_mem:.2f} GB")
else:
    st.write("⚠️ Running on CPU (GPU not available).")

model_size_mb = sum(p.numel() for p in torch.load(output_dir + "/pytorch_model.bin", map_location="cpu").values()) * 4 / (1024 ** 2)
st.write(f"📦 Model Size: {model_size_mb:.2f} MB")

