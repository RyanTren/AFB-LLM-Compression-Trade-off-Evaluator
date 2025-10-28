
# 📘 Overview of ``/src``

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
