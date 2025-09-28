# Docker: CPU & GPU build (multi-stage)

This folder contains the multi-stage `Dockerfile` used to build two runtime images:

- `final-cpu` — a CPU-only image built from `python:3.13.5-slim` (smaller, default entrypoint runs `/app/main.py`).
- `final-gpu` — a CUDA-enabled image built from `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` (default entrypoint runs `/app/quantize_llama_gptq.py`).

Key points
- The Dockerfile uses builder stages to install build-time tooling (including `rustup`) so Rust-backed Python packages (for example, `tiktoken`) can be compiled.
- Each final image copies a virtualenv from its respective builder stage. This keeps the runtime images consistent with what was built but avoids keeping all build-layer artifacts in the final image layer history.

Quick build examples (run from `src/docker`)

Build the CPU runtime image (fastest):
```powershell
docker build --target final-cpu -t afb-llm:cpu .
```

Build the GPU runtime image (large; ensure your host has GPU support):
```powershell
docker build --build-arg AUTOGPTQ_EXTRA_INDEX=https://huggingface.github.io/autogptq-index/whl/cu118/ --target final-gpu -t afb-llm:gpu .
```

Notes on `AUTOGPTQ_EXTRA_INDEX`
- If you need prebuilt AutoGPTQ wheels for a specific CUDA version (for example `cu118`), pass the wheel index URL via `--build-arg AUTOGPTQ_EXTRA_INDEX=...` so pip can pull compatible binary wheels during the build.
- If you don't supply the extra index, the build uses the normal PyPI index and may attempt to compile GPU-specific packages from source (much slower) or skip them if not available.

Run examples
- Run the CPU image and expose the health server (port 8000):
```powershell
docker run --rm -p 8000:8000 afb-llm:cpu
```

- Run the GPU image (example, requires Docker host GPU support):
```powershell
docker run --rm --gpus all -p 8000:8000 afb-llm:gpu
```

Next steps & optional improvements
- Slim the final images further by removing rust/build-tooling in a later stage or by creating a smaller runtime-only image and copying installed site-packages into it.
- Change the GPU final image default entrypoint if you prefer to run a different script on container start (for example `quantize_llama_gptq.py`).
- Add a `docker-compose.yml` with named services for CPU/GPU and a healthcheck.

If you'd like, I can run a quick CPU build and a smoke run to verify the image build and startup on your machine (note: the GPU build is large and will take longer).

