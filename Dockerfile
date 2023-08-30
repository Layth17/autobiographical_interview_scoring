# Use Debian Bullseye as the base image
FROM debian:bullseye

# Update package list and install necessary system packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required Python packages
RUN pip3 install --no-cache-dir pandas numpy tensorflow transformers pysbd

# Set environment variable for transformer models
ENV TRANSFORMERS_CACHE=/app/.transformers_cache

# The command to run your Python program
CMD ["python3", "aais.py"]



