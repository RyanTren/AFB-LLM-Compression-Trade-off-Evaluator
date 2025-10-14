# Base (CPU). If you plan to run with GPUs in container later, change to nvidia/cuda runtime base.
FROM python:3.12-slim

# basics
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git build-essential wget libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy project
COPY . /app

# default command (you can override with docker run)
CMD ["bash"]


# command to run local (build and run interative)
# docker build -t lora-m40 .
# docker run --rm -it -v $(pwd):/app lora-m40

# If you later need CUDA in container, use an nvidia/cuda base image with a matching CUDA version and --gpus all at runtime.
