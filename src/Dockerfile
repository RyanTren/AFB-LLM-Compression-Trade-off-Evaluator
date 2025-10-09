# CPU-only PyTorch base
FROM pytorch/pytorch:2.3.0-cpu

# Basic utilities
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App setup
WORKDIR /app
COPY . .

CMD ["python", "scripts/run_inference.py"]
