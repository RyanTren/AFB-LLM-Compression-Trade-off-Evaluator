# GPU runtime (CUDA 11.8 + PyTorch 2.1.x) – compatible with Tesla M40
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BUILD_CUDA_EXT=0 TORCH_CUDA_ARCH_LIST=5.2

# minimal system deps + uvicorn runner
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY ServingContainerRequirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r ServingContainerRequirements.txt

COPY serve.py /app/main.py

# Model gets mounted at /models (see docker run); do not bake weights into the image
ENV MODEL_PATH=/models
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
