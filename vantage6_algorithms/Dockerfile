# Unified Dockerfile for all vantage6 algorithms
# Usage: docker build --build-arg ALGORITHM=SIMI -t simi-algorithm:latest -f Dockerfile .
#        docker build --build-arg ALGORITHM=SIMICE -t simice-algorithm:latest -f Dockerfile .

FROM python:3.11-slim

# Define build argument (must be after FROM for multi-stage, but works here too)
ARG ALGORITHM=SIMI

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
# Both algorithms use the same requirements
COPY ${ALGORITHM}/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install vantage6 algorithm tools
RUN pip install --no-cache-dir vantage6>=4.0.0

# Copy Core utilities (shared across all algorithms)
COPY Core/ ./Core/

# Copy algorithm-specific code
COPY ${ALGORITHM}/algorithm.py ./algorithm.py
COPY ${ALGORITHM}/wrapper.py ./wrapper.py

# Set Python path
ENV PYTHONPATH=/app

# Vantage6 entry point - use wrapper
CMD ["python", "wrapper.py"]

