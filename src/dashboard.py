import os
import subprocess
import json
import time
import streamlit as st
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator


os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = ""
os.environ["NCCL_P2P_LEVEL"] = "PXB"

st.set_page_config(page_title="AFB Robins | LLM-LoRA Compression Dashboard", layout="wide")

st.title("AFB Robins | LLM Compression Dashboard (LoRA vs Full Model)")

# --------------------------
# Sidebar configuration
# --------------------------
st.sidebar.header("⚙️ Training Configuration")
model_id = st.sidebar.text_input("Base Model", "deepseek-ai/deepseek-coder-1b-base")
epochs = st.sidebar.number_input("Epochs", 1, 10, 3)
batch_size = st.sidebar.number_input("Batch Size", 1, 8, 2)
gradient_accumulation = st.sidebar.number_input("Gradient Accumulation", 1, 32, 8)
learning_rate = st.sidebar.number_input("Learning Rate", 1e-6, 1e-3, 2e-5, format="%.1e")
max_length = st.sidebar.number_input("Max Sequence Length", 64, 2048, 128)
dataset = st.sidebar.text_input("Dataset", "iamtarun/python_code_instructions_18k_alpaca")
output_dir = st.sidebar.text_input("Output Directory", "lora_out_deepseek_1b_v4")
num_processes = st.sidebar.number_input("Num Processes (GPUs)", 1, 8, 4)
save_every = st.sidebar.number_input("Save Every (steps)", 100, 2000, 500)
keep_last_n_checkpoints = st.sidebar.number_input("Keep Last N Checkpoints", 1, 10, 3)
mixed_precision = st.sidebar.selectbox("Mixed Precision", ["fp16", "bf16", "no"], index=0)
dry_run = st.sidebar.checkbox("Dry Run", False)

# --------------------------
# TRAINING
# --------------------------
st.subheader("AFB Robins 🚀 Train/Inference/CodeBLEU LoRA Model")

train_btn = st.button("Start Training")

if train_btn:
    st.write("🔄 Starting Accelerate training...")

    # --------------------------
    # Multi-GPU environment setup
    # --------------------------
    num_gpus = torch.cuda.device_count()
    if num_processes > num_gpus:
        st.warning(f"Requested {num_processes} GPUs, but only {num_gpus} available. Using {num_gpus}.")
        num_processes = num_gpus

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_processes)))
 
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_processes),
        "--mixed_precision", mixed_precision,
        "scripts/deepseek_coder1B_train_lora.py",
        "--model_id", model_id,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--gradient_accumulation", str(gradient_accumulation),
        "--learning_rate", str(learning_rate),
        "--max_length", str(max_length),
        "--dataset", dataset,
        "--save_every", str(save_every),
        "--keep_last_n_checkpoints", str(keep_last_n_checkpoints),
    ]
    if dry_run:
        cmd.append("--dry_run")

    st.info("Launching training with command:")
    st.code(" ".join(cmd), language="bash")

    # --------------------------
    # Live log streaming
    # --------------------------
    log_placeholder = st.empty()
    metric_placeholder = st.empty()
    plot_placeholder = st.empty()
    os.makedirs(output_dir, exist_ok=True)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        logs = ""
        last_metrics_update = 0
        start_time = time.time()

        st.info("🚀 Training started — live logs and metrics below:")

        for line in iter(process.stdout.readline, ''):
            if line:
                logs += line
                log_placeholder.markdown(f"<pre>{logs}</pre>", unsafe_allow_html=True)

            
            # Refresh metrics every 10 seconds
            if time.time() - last_metrics_update > 10:
                last_metrics_update = time.time()
                if os.path.exists(output_dir):
                    metrics_files = [f for f in os.listdir(output_dir) if f.startswith("metrics_")]
                    if metrics_files:
                        latest = max(metrics_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
                        try:
                            with open(os.path.join(output_dir, latest)) as f:
                                metrics = json.load(f)
                                metric_placeholder.json(metrics)

                                if "avg_loss_per_epoch" in metrics:
                                    plt.clf()
                                    plt.plot(metrics["avg_loss_per_epoch"], marker="o", color="royalblue")
                                    plt.title("Training Loss per Epoch")
                                    plt.xlabel("Epoch")
                                    plt.ylabel("Loss")
                                    plot_placeholder.pyplot(plt)
                        except Exception as e:
                            st.warning(f"Could not read metrics: {e}")

        process.wait()
        duration = time.time() - start_time
        st.success(f"✅ Training completed in {duration/60:.1f} minutes")

    except Exception as e:
        st.error(f"Training failed: {e}")
        if process:
            process.terminate()


# --------------------------
# INFERENCE
# --------------------------
st.subheader("🧠 Inference")

checkpoint_path = st.text_input("Checkpoint Path", f"{output_dir}")
run_infer = st.button("Run Inference")

if run_infer:
    st.write("⚙️ Running inference...")
    cmd = ["python3", "run_inference_deepseek.py", "--checkpoint", checkpoint_path]
    process = subprocess.run(cmd, capture_output=True, text=True)
    st.text_area("Inference Output", process.stdout, height=250)
    st.success("Inference complete!")

# --------------------------
# EVALUATION
# --------------------------
st.subheader("📊 Evaluation Metrics")

if st.button("Evaluate BLEU / CodeBLEU"):
    try:
        from codebleu import calc_code_bleu
    except ImportError:
        st.error("CodeBLEU not installed. Run: pip install codebleu")
        st.stop()

    refs, hyps = [], []
    if os.path.exists("results"):
        for file in os.listdir("results"):
            if file.endswith(".json"):
                with open(os.path.join("results", file)) as f:
                    lines = f.readlines()
                    if len(lines) >= 4:
                        refs.append(lines[1])
                        hyps.append(lines[3])

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
    gpu_name = torch.cuda.get_device_name(0)
    st.write(f"💾 GPU: {gpu_name}")
    st.write(f"💽 GPU Memory Used: {gpu_mem:.2f} GB")
else:
    st.write("⚠️ Running on CPU (GPU not available).")

if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
    model_size_mb = (
        sum(p.numel() for p in torch.load(os.path.join(output_dir, "pytorch_model.bin"), map_location="cpu").values())
        * 4 / (1024 ** 2)
    )
    st.write(f"📦 Model Size: {model_size_mb:.2f} MB")
