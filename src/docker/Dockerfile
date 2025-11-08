# CUDA 11.8 + Torch 2.1.2 (Python 3.10) – safe for your GPU stack
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ensure kernels don’t target newer SMs
    TORCH_CUDA_ARCH_LIST=5.2

# Minimal OS deps (no uvicorn/tini here)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Install ONLY what quantization needs
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Copy just the quantizer entrypoint (and anything it imports)
COPY quantize_llama_gptq.py /app/quantize_llama_gptq.py

# We’ll mount /out and /models at runtime
ENV HF_HOME=/out/.hf_cache \
    TRANSFORMERS_CACHE=/out/.hf_cache \
    HF_DATASETS_CACHE=/out/.hf_cache

# No ENTRYPOINT that forces a server; default to bash so you can pass the CLI
CMD ["/bin/bash"]
