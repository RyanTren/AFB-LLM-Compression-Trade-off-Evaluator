# Llama3.3 8B LLM Docker Setup

This repository contains instructions for building, running, and publishing a Docker image for the **Llama3.3** project. It includes a `Dockerfile`, `compose.yaml`, and an updated `requirements.txt` for Python dependencies.

---

## Prerequisites

- Docker installed on your machine ([Download Docker](https://www.docker.com/get-started))
- Docker Hub account for publishing images
- Git installed
- (Optional) Windows users may need to adjust the Dockerfile as noted below.

---

## Files in This Project

1. **Dockerfile** – Instructions to build the Docker image for your application.
2. **compose.yaml** – Defines a Docker Compose service called `server` and maps port 8000.
3. **requirements.txt** – Python dependencies for the Llama3.3 project.

---

## Docker Compose Service

The `compose.yaml` defines:

```yaml
services:
  server:
    build:
      context: .
    ports:
      - 8000:8000
```
This will build the Docker image from the ``Dockerfile`` and expose the application on port 8000.

## If you're on Windows...
You need to replace lines 33-46 in the ``Dockerfile`` with this:

```dockerfile
# Use Windows Server Core as the base image
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set environment variables to avoid prompts
ENV RUSTUP_HOME="C:\rustup" \
    CARGO_HOME="C:\cargo" \
    PATH="C:\cargo\bin;%PATH%" \
    RUST_VERSION=stable

# Download and install Rust
RUN powershell -Command \
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "rustup-init.exe" ; \
    Start-Process -FilePath "rustup-init.exe" -ArgumentList "-y --default-toolchain %RUST_VERSION%" -Wait ; \
    Remove-Item "rustup-init.exe"

# Set the working directory
WORKDIR /app

# Default command
CMD ["cmd.exe"]
```

## Build & Run Docker Image
Go to your project directory and run this command to build the Docker Image:
```bash
docker build -t llama3 .
```
- "llama3" is your project name so just replace this with whatever you named it with earlier...

Now run this command to run the application itself:
```bash
docker compose up
```
This will start the ``server`` service and you can view it on http://localhost:8000

## Publishing Docker Image to Docker Hub
```bash
docker login

docker tag llama3 <your-dockerhub-username>/llama3:latest

docker push <your-dockerhub-username>/llama3:latest
```
Here's a tutorial if this doesn't work: [Tutorial Video](https://www.youtube.com/watch?v=zO0O-EzXYhk)

## Python Dependencies
Use the following ``requirements.txt`` for your Llama3.3 project:

* ``src/docker/requirements.txt``

## References
* Docker Compose Reference: https://docs.docker.com/go/compose-spec-reference/

* Awesome Compose Examples: https://github.com/docker/awesome-compose

