# Deployment

This guide explains how to deploy the project for a self-contained demo or local development.

## Prerequisites

- Docker/Docker Desktop
- macOS/Linux/Windows

Optional for local dev without Docker:
- Python 3.11

## One-container demo (recommended)

This repository ships a Dockerfile and an entrypoint that runs:
- central FastAPI at :8000
- two remote FastAPI instances at :8001 and :8002 (demo-only mapping)
- Nginx reverse proxy on :8080 (or `$PORT` if your platform injects it)

Build and run:

```sh
# Build image
docker build -t midn-py .

# Run container mapping Nginx port 8080 to host 8080
# You can override ports with env variables if needed.
docker run --rm -p 8080:8080 \
  -e CENTRAL_HOST=0.0.0.0 -e CENTRAL_PORT=8000 \
  -e REMOTE_HOST=0.0.0.0 -e REMOTE1_PORT=8001 -e REMOTE2_PORT=8002 \
  -e PORT=8080 \
  -v "$PWD/data":/app/data \
  --name midn midn-py
```

Then open:
- Central: http://localhost:8080/
- Remote 1: http://localhost:8080/remote1/
- Remote 2: http://localhost:8080/remote2/

Note: `/remote1` and `/remote2` are convenience routes for the bundled demo only. In production or non-demo setups, remotes can run on any host/port and need not be co-hosted behind the same Nginx.

Persistence: The SQLite DB (`data/central.db`) and the remote/central site configs live under `./data`. We mount it into the container so state survives restarts.

## Environment variables

- CENTRAL_HOST (default 0.0.0.0)
- REMOTE_HOST (default 0.0.0.0)
- CENTRAL_PORT (default 8000)
- REMOTE1_PORT (default 8001)
- REMOTE2_PORT (default 8002)
- PORT (public Nginx port, default 8080)
- ADMIN_PASSWORD (central GUI admin password, default admin123)
- SECRET_KEY (JWT signing, default a_very_secret_key)
- CENTRAL_URL (used in approval emails; defaults to http://localhost:8000)
- GMAIL_USER / GMAIL_APP_PASSWORD (optional; enable Gmail sending on approval)

All are consumed by `entrypoint.sh` and/or `central/app/core/config.py`.

## Local development (without Docker)

Launch services manually in separate terminals:

```sh
# Install deps (create venv recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip -r requirements.txt

# Terminal 1: central
uvicorn central.app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: remote 1 (example port; choose any free local port)
uvicorn remote.app.main:app --host 0.0.0.0 --port 8001

# Terminal 3: remote 2 (example port; choose any free local port)
uvicorn remote.app.main:app --host 0.0.0.0 --port 8002
```

Open the central at http://localhost:8000 and remotes at their chosen ports (e.g., http://localhost:8001 and http://localhost:8002).

In this mode you won't have the unified Nginx paths (/remote1, /remote2). Adjust URLs accordingly in the remote Settings to point HTTP URL to central `http://localhost:8000` and WebSocket URL to `ws://localhost:8000`. The remotes themselves can listen on any local port you choose.

## Cloud notes

- The container is suitable for platforms that inject `$PORT` (Cloud Run, ACI, etc.). `entrypoint.sh` patches the Nginx listen port to match `$PORT` and keeps internal services private.
- Terminate signals (SIGTERM) are propagated to uvicorn and Nginx for graceful shutdown.
- To expose HTTPS, place the container behind your cloud load balancer / reverse proxy that provides TLS, or extend the Nginx config to terminate TLS.

## Health checks

- Container-level: Dockerfile defines a HEALTHCHECK probing 127.0.0.1:$PORT (8080 by default).
- App-level: central exposes `/health` and Nginx provides `/nginx_healthz`.
