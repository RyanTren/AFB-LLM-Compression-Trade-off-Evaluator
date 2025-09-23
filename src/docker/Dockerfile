FROM python:3.13.5-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required to build some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    libssl-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under
ARG UID=10001
RUN useradd --uid ${UID} --create-home --shell /bin/false appuser

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Install rustup and make sure cargo/rustc are available in PATH for subsequent pip builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    gcc \
    git \
    libssl-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    chmod +x /tmp/rustup.sh && \
    /tmp/rustup.sh -y --no-modify-path && \
    rm /tmp/rustup.sh && \
    # create a profile snippet that adds cargo to PATH for future shells and ensure variables are available
    printf '%s\n' "export CARGO_HOME=/root/.cargo" "export RUSTUP_HOME=/root/.rustup" "export PATH=/root/.cargo/bin:$PATH" > /etc/profile.d/cargo.sh && \
    chmod +x /etc/profile.d/cargo.sh

# Ensure the Rust toolchain is on PATH for this RUN by explicitly exporting it.
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies with pip; use cache directory
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy application source
COPY . .

# Ensure non-root user owns the app files
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default command: run the project's main Python module. Adjust if different.
CMD ["python", "/app/main.py"]
