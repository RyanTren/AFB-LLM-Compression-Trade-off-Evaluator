FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system updates and dependencies
RUN apt-get update && apt-get upgrade -y && apt-get clean && \
	pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY scripts/ ./scripts/

# Specify the command to run the application
CMD ["python", "./scripts/main.py"]