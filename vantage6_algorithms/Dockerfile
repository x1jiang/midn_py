# Unified Dockerfile for all vantage6 algorithms
# Usage: docker build --build-arg ALGORITHM=SIMI -t simi-algorithm:latest -f Dockerfile .
#        docker build --build-arg ALGORITHM=SIMICE -t simice-algorithm:latest -f Dockerfile .

# Use vantage6 algorithm base so the runtime tools are present
FROM harbor2.vantage6.ai/infrastructure/algorithm-base:4.13

# Define build argument (must be after FROM for multi-stage, but works here too)
ARG ALGORITHM=SIMI

WORKDIR /app

# Copy requirements first (for better caching)
# Both algorithms use the same requirements
COPY ${ALGORITHM}/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Core utilities (shared across all algorithms)
COPY Core/ ./Core/

# Copy algorithm-specific code
COPY ${ALGORITHM}/algorithm.py ./algorithm.py
COPY ${ALGORITHM}/wrapper.py ./wrapper.py
COPY ${ALGORITHM}/local_inference.py ./local_inference.py

# Set Python path
ENV PYTHONPATH=/app

# Tell wrap_algorithm where to look for decorated functions
ENV PKG_NAME=wrapper

# Vantage6 entry point - delegate to wrap_algorithm (handles tokens/env)
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
