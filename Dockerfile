# Dockerfile for PYMIDN (uses fuller Debian base with common utilities)
FROM python:3.11-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=none

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  libpq-dev \
  nginx \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code (everything except what .dockerignore excludes)
COPY . .

# Configure Nginx: place our site config and remove default
RUN mkdir -p /etc/nginx/conf.d && \
  rm -f /etc/nginx/conf.d/default.conf /etc/nginx/sites-enabled/default || true
COPY nginx/pymidn.conf /etc/nginx/nginx.conf

# Ensure entrypoint present & executable (explicit copy for clarity)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose internal service ports plus 8080 for Nginx
EXPOSE 8000 8001 8002 8080

# Healthcheck via Nginx proxy port (8080 or $PORT if platform sets it)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,socket; p=int(os.environ.get('PORT', '8080')); s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',p)); s.close()" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
