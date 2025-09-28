ARG AUTOGPTQ_EXTRA_INDEX=""
ARG UID=10001
ARG CUDA_BASE_IMAGE=pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

###############################################################################
# Builder: CPU
FROM python:3.13-slim-bookworm AS builder-cpu
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build deps and rustup
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential gcc libssl-dev pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
ARG UID
RUN useradd --uid ${UID} --create-home --shell /bin/false appuser

COPY requirements.txt ./

# Install rustup for tiktoken and similar
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y --no-modify-path && rm /tmp/rustup.sh && \
    printf '%s\n' "export CARGO_HOME=/root/.cargo" "export RUSTUP_HOME=/root/.rustup" "export PATH=/root/.cargo/bin:$PATH" > /etc/profile.d/cargo.sh

ENV CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup PATH=/root/.cargo/bin:$PATH

# Create venv and install dependencies
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements.txt

COPY . .
RUN chown -R appuser:appuser /app /opt/venv

###############################################################################
# Builder: GPU (uses CUDA-enabled PyTorch base so CUDA wheels match)
FROM ${CUDA_BASE_IMAGE} AS builder-gpu
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential gcc libssl-dev pkg-config python3-dev tini \
    && rm -rf /var/lib/apt/lists/*

ARG UID
RUN useradd --uid ${UID} --create-home --shell /bin/false appuser

COPY requirements.txt ./

# If an extra index for AutoGPTQ cu118 is provided, use it
ARG AUTOGPTQ_EXTRA_INDEX
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y --no-modify-path && rm /tmp/rustup.sh && \
    printf '%s\n' "export CARGO_HOME=/root/.cargo" "export RUSTUP_HOME=/root/.rustup" "export PATH=/root/.cargo/bin:$PATH" > /etc/profile.d/cargo.sh

ENV CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup PATH=/root/.cargo/bin:$PATH

RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip && \
    if [ -n "${AUTOGPTQ_EXTRA_INDEX}" ]; then \
        /opt/venv/bin/pip install --extra-index-url "${AUTOGPTQ_EXTRA_INDEX}" -r requirements.txt; \
    else \
        /opt/venv/bin/pip install -r requirements.txt; \
    fi

COPY . .
RUN chown -R appuser:appuser /app /opt/venv

###############################################################################
# Final: CPU runtime (smaller)
FROM python:3.13-slim-bookworm AS final-cpu
WORKDIR /app

ARG UID

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy venv and app from cpu builder
COPY --from=builder-cpu /opt/venv /opt/venv
COPY --from=builder-cpu /app /app

ENV PATH=/opt/venv/bin:$PATH

RUN useradd --uid ${UID} --create-home --shell /bin/false appuser || true && chown -R appuser:appuser /app /opt/venv
USER appuser

EXPOSE 8000
CMD ["python", "/app/main.py"]

###############################################################################
# Final: GPU runtime
FROM ${CUDA_BASE_IMAGE} AS final-gpu
WORKDIR /app

ARG UID

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy venv and app from gpu builder
COPY --from=builder-gpu /opt/venv /opt/venv
COPY --from=builder-gpu /app /app

ENV PATH=/opt/venv/bin:$PATH

RUN useradd --uid ${UID} --create-home --shell /bin/false appuser || true && chown -R appuser:appuser /app /opt/venv
USER appuser

# Keep tini as entrypoint for clean signal handling and run quantize script by default
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/app/quantize_llama_gptq.py"]
CMD ["-h"]

### Usage
# Build CPU image (final-cpu):
# docker build --target final-cpu -t llama3:cpu .
# Build GPU image (final-gpu) with AutoGPTQ cu118 wheel index:
# docker build --build-arg AUTOGPTQ_EXTRA_INDEX=https://huggingface.github.io/autogptq-index/whl/cu118/ --target final-gpu -t llama3:gpu .
