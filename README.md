# 🧠 LLM Compression Trade-off Evaluator — LoRA Technique Branch

This branch implements **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** as part of the _LLM Compression Trade-off Evaluator_ project sponsored by **Robins AFB (AFSC/EN)**.  
It focuses on evaluating **compression–accuracy trade-offs** in large language models, especially for **code-generation** tasks.

---

## 📘 Overview

This implementation demonstrates **parameter-efficient fine-tuning** (LoRA/PEFT) as a valid model compression strategy, enabling smaller and faster fine-tuning while preserving model performance.  

Although the school VM GPUs (Tesla M40, compute capability 5.2) do **not support QLoRA**, LoRA fine-tuning still fulfills project objectives by providing measurable trade-offs between:
- **Model size**
- **Inference latency**
- **Accuracy (BLEU / Code-BLEU)**
- **Memory and resource usage**

This branch includes:
- LoRA fine-tuning pipeline using **PEFT + Accelerate + DeepSpeed**
- Inference benchmarking and BLEU evaluation
- Dynamic routing prototype for model selection
- Dockerized setup for reproducible experiments

---

## 🧩 Project Structure
```text
P10-T1 LLM Compression Trade-Off Accelerator
├── src/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── accelerate_config.yaml
│   ├── deepspeed_config.json
│   ├── scripts/
│   │   ├── train_lora.py           # LoRA fine-tuning pipeline (DeepSpeed + PEFT)
│   │   ├── eval_and_profile.py     # Evaluates BLEU, latency, memory, and performance
│   │   ├── router_demo.py          # Dynamic query router (LoRA vs base model)
│   │   ├── check_gpu.py            # Detects CUDA capability for compatibility checks
│   │   └── __init__.py
│
├── data/
│   ├── code_train.json             # Optional fine-tuning dataset (code examples)
│   └── code_eval_prompts.json      # Evaluation prompts for Code-BLEU testing
│
├── venv/                           # Local virtual environment (optional)
└── README.md
```



---

## ⚙️ Environment Setup

### Option 1: Local (Recommended for KSU VM)
Create a virtual environment to isolate dependencies.

```bash
cd src
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# this creates Local temporary env vars since we don't have write access
mkdir -p /tmp/hf_cache
export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
```

#### Dry-Run to Test
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch ../src/scripts/train_lora.py \
  --model_id codeparrot/codeparrot-small \
  --output_dir lora_dryrun_test \
  --epochs 1 \
  --batch_size 2 \
  --gradient_accumulation 4 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --dataset codeparrot \
  --dry_run
```

#### Full training with checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch ../src/scripts/train_lora.py \
  --model_id codeparrot/codeparrot-small \
  --output_dir lora_out_codeparrot_small \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --dataset codeparrot \
  --save_every 500 \
  --keep_last_n_checkpoints 3
```

#### Resume Training off last saved checkpoint
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch ../src/scripts/train_lora.py \
  --model_id codeparrot/codeparrot-small \
  --output_dir lora_out_codeparrot_small \
  --resume_from lora_out_codeparrot_small/checkpoint-epoch0-step500 \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --dataset codeparrot \
  --save_every 500
```
**Note** ``CUDA_VISIBLE_DEVICES=0`` is set to 0 for single-GPU training, our VM has 4 M40 GPUS.

**Once training is done create an Accelerate config for easier training in the future:**
Answer the prompts:
  - Compute environment: LOCAL_MACHINE
  - Distributed type: NO (for single GPU)
  - Mixed precision: fp16
  - Number of processes: 1

Now you can run:
```bash
accelerate launch ../src/scripts/train_lora.py ...

# or pass in flags directly

accelerate launch --num_processes=1 --mixed_precision=fp16 ../src/scripts/train_lora.py ...
```



- This is what the expected output is after running the test command (Dryrun):
```bash
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ CUDA_VISIBLE_DEVICES=0 accelerate launch ../src/scripts/train_lora.py \
  --model_id codeparrot/codeparrot-small \
  --output_dir lora_dryrun_test \
  --epochs 1 \
  --batch_size 2 \
  --gradient_accumulation 4 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --dataset codeparrot \
  --dry_run
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.11/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
The following values were not passed to `accelerate launch` and had defaults used instead:
        `--num_processes` was set to a value of `1`
        `--num_machines` was set to a value of `1`
        `--mixed_precision` was set to a value of `'no'`
        `--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.11/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🔹 Using model: codeparrot/codeparrot-small
🔹 Dataset: codeparrot
🔹 Output directory: lora_dryrun_test
🔹 Dry run: True
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.11/site-packages/huggingface_hub/file_download.py:945: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.11/site-packages/peft/tuners/lora/layer.py:1059: UserWarning: fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.
  warnings.warn(
trainable params: 811,008 || all params: 111,820,032 || trainable%: 0.7252797065913914
📘 Streaming CodeParrot dataset...
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:00<00:00, 71.18it/s]
🚀 Starting LoRA fine-tuning...
Epoch 1/1: 0it [00:00, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
Epoch 1/1: 201it [00:24,  7.66it/s, loss=nan, step=50, ETA=0:00:24]🧩 Dry-run st                                                                                                                                                             opping early
Epoch 1/1: 201it [00:24,  8.25it/s, loss=nan, step=50, ETA=0:00:24]
✅ Epoch 1 complete | Avg loss: nan
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.                                                                                                                                                             11/site-packages/huggingface_hub/file_download.py:945: FutureWarning: `resume_do                                                                                                                                                             wnload` is deprecated and will be removed in version 1.0.0. Downloads always res                                                                                                                                                             ume when possible. If you want to force a new download, use `force_download=True                                                                                                                                                             `.
  warnings.warn(
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.                                                                                                                                                             11/site-packages/peft/utils/save_and_load.py:168: UserWarning: Setting `save_emb                                                                                                                                                             edding_layers` to `True` as the embedding layer has been resized during finetuni                                                                                                                                                             ng.
  warnings.warn(
💾 Checkpoint saved: lora_dryrun_test/checkpoint-epoch0-step50
📊 Metrics saved to: lora_dryrun_test/metrics_1761154245.json
📈 Loss plot saved to: lora_dryrun_test/loss_plot_1761154245.png

✅ Training complete! LoRA adapters saved to: lora_dryrun_test
⏱️  Total training time: 0:00:34

```


You will only see this in the ssh terminal on the vm it will show up in the ``/src/lora_out`` path. It will contain the following files:

 ``
 adapter_model.safetensors  
 merges.txt               
 tokenizer.json         
 vocab.json
  adapter_config.json  
  added_tokens.json          special_tokens_map.json  tokenizer_config.json
``
This means our LoRA fine-tuning is done and our LoRA adapter is done and we can now run our inference script to load our Fine-tuned GPT-2 LoRA Model, this will generate a code snippet based on our fine-tuned LoRA adapter

- [brief read on why I didn't use QLoRA](src/why_QLoRA_won't_work.md)

### Progress as of 10/15/25

* ✅ LoRA training working with PEFT
* ✅ Model + adapter saved locally
* ✅ Proper vocab alignment (embedding fixed)
* ✅ Inference pipeline confirmed

**Screenshot from VM Terminal:**
![running inference script for LoRA/PEFT Model (codeparrot dataset not working in this test)](image.png)
![ss of codeparrot training ](image-1.png)


### Option 2: Docker (Reproducible)

Build and run the containerized environment:
```bash
docker build -t lora-m40 .
docker run --rm -it -v $(pwd):/app lora-m40

```

## 🧪 Running Experiments
### 1️⃣ Fine-tune with LoRA (Parameter-Efficient)

Trains LoRA adapters on a small code dataset using DeepSpeed offload (CPU-backed for M40s).
```bash
accelerate launch --config_file accelerate_config.yaml scripts/train_lora.py \
  --model_id facebook/opt-1.3b \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --output_dir lora_out
```

This produces LoRA adapter weights in lora_out/.

### 2️⃣ Evaluate & Profile

Compares the base model and LoRA-compressed model on BLEU, latency, and memory metrics.
```bash
python scripts/eval_and_profile.py \
  --base_model facebook/opt-1.3b \
  --lora_model lora_out
```

Outputs include:

* BLEU score comparison (base vs LoRA)

* Average latency per prompt

* Memory consumption (MB)

* Summary statistics for reporting

### 3️⃣ Dynamic Routing Demo

Routes incoming code-generation requests to either the full or LoRA model based on query complexity.

```bash
python scripts/router_demo.py
```


Send a test query:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"Write a Python function to reverse a string."}' \
  http://localhost:8080/generate
```

This proof-of-concept implements the “dynamic query allocation” requirement from the project brief.

### 📊 Metrics & Evaluation
Metric	Description	Tool
BLEU / Code-BLEU	Code generation accuracy	sacrebleu / custom Code-BLEU
Latency (s)	Inference time per query	Python timer
Memory (MB)	CPU memory footprint	psutil
Model Size (MB)	Serialized adapter size	Disk size comparison

- These metrics feed into the final dashboard and trade-off visualization stage (Phase 3).

### 🧱 Configuration Files
File	Purpose
accelerate_config.yaml	Configures Accelerate + DeepSpeed offload for low-memory training.
deepspeed_config.json	Defines ZeRO Stage 3 with CPU optimizer & param offloading.
requirements.txt	Project dependencies for fine-tuning, evaluation, and dashboard.
Dockerfile	Reproducible container image (CPU-friendly, switchable to GPU).

### 🧮 Technology Stack

Language: Python (PyTorch, Transformers)

Fine-tuning: PEFT (LoRA)

Optimization: DeepSpeed ZeRO-Offload + Accelerate

Evaluation: SacreBLEU / Code-BLEU, latency profiling

Deployment: Flask + Docker

Visualization: Streamlit (planned)

Hardware tested: Tesla M40 GPUs (11GB, compute 5.2), 500GB RAM

### 🧭 Deliverables Alignment
**Project Deliverable:	Implemented Component**
- LLM compression benchmarking scripts: ``train_lora.py, eval_and_profile.py``
- Dynamic query routing proof-of-concept:	``router_demo.py``
- Code-BLEU metric evaluation:	``eval_and_profile.py``
- Reporting dashboard:	(In progress via Streamlit)
- Final documentation & architecture:	This README and Docker environment

### 🧩 Next Steps

1. Integrate Code-BLEU metric implementation (syntax-aware scoring).

2. Build a Streamlit dashboard visualizing BLEU, latency, and memory trade-offs.

3. (Optional) Extend to QLoRA when modern GPUs are available (Ampere or newer).

### 🤝 Contributing

Contributions are welcome!
Please open a Pull Request or Issue to propose feature additions, experiment configurations, or dashboard improvements.

### ⚖️ License

This project is licensed under the MIT License.
See the LICENSE
 file for details.

### 🧠 References

- Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2022

- Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, 2023

- [Hugging Face PEFT library](https://github.com/huggingface/peft)

- [DeepSpeed ZeRO documentation](https://www.deepspeed.ai/tutorials/zero/)
