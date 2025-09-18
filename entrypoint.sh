#!/usr/bin/env bash
set -euo pipefail

# Allow optional environment overrides for host/ports
CENTRAL_HOST="${CENTRAL_HOST:-0.0.0.0}"
REMOTE_HOST="${REMOTE_HOST:-0.0.0.0}"
CENTRAL_PORT="${CENTRAL_PORT:-8000}"
REMOTE1_PORT="${REMOTE1_PORT:-8001}"
REMOTE2_PORT="${REMOTE2_PORT:-8002}"

# Function to terminate all background jobs gracefully
terminate() {
  echo "Received signal, terminating services..."
  jobs -p | xargs -r kill -TERM
  wait || true
}
trap terminate SIGINT SIGTERM

echo "Starting central service on ${CENTRAL_HOST}:${CENTRAL_PORT}" >&2
python -m uvicorn central.app.main:app --host "$CENTRAL_HOST" --port "$CENTRAL_PORT" &
CENTRAL_PID=$!

# For remote services, we assume separate instances pointing to same module.
echo "Starting remote service instance 1 on ${REMOTE_HOST}:${REMOTE1_PORT}" >&2
python -m uvicorn remote.app.main:app --host "$REMOTE_HOST" --port "$REMOTE1_PORT" &
REMOTE1_PID=$!

echo "Starting remote service instance 2 on ${REMOTE_HOST}:${REMOTE2_PORT}" >&2
python -m uvicorn remote.app.main:app --host "$REMOTE_HOST" --port "$REMOTE2_PORT" &
REMOTE2_PID=$!

# Wait for central to be ready before exposing Nginx on $PORT
echo "Waiting for central service to become ready on 127.0.0.1:${CENTRAL_PORT}..." >&2
CENTRAL_READY=0
for i in $(seq 1 60); do
  if python - <<'PY'
import os, socket, sys
host = '127.0.0.1'
port = int(os.environ.get('CENTRAL_PORT','8000'))
s = socket.socket()
s.settimeout(1)
try:
    s.connect((host, port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    CENTRAL_READY=1
    break
  fi
  sleep 1
done
if [ "$CENTRAL_READY" -ne 1 ]; then
  echo "Central service not ready after waiting; exiting to allow restart..." >&2
  kill -TERM $CENTRAL_PID $REMOTE1_PID $REMOTE2_PID 2>/dev/null || true
  wait || true
  exit 1
fi
echo "Central service is ready." >&2

# Best-effort wait for remote1 and remote2 (do not fail if they are slow)
for pair in "remote1:${REMOTE1_PORT}" "remote2:${REMOTE2_PORT}"; do
  REMOTE_NAME="${pair%%:*}"
  REMOTE_PORT="${pair#*:}"
  echo "Waiting briefly for $REMOTE_NAME on 127.0.0.1:${REMOTE_PORT}..." >&2
  for i in $(seq 1 15); do
    if CHECK_PORT="$REMOTE_PORT" python - <<'PY'
import os, socket, sys
host = '127.0.0.1'
port = int(os.environ.get('CHECK_PORT'))
s = socket.socket()
s.settimeout(1)
try:
    s.connect((host, port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
      echo "$REMOTE_NAME ready." >&2
      break
    fi
    sleep 1
  done
done

# Start Nginx proxy on 8080 (or $PORT if platform injects it)
NGINX_PORT="${PORT:-8080}"
echo "Starting nginx reverse proxy on :${NGINX_PORT}" >&2
# Ensure nginx can write to runtime dirs in some minimal containers
mkdir -p /run/nginx || true
# Replace the 'listen 8080' if platform requires a different PORT
if [ "$NGINX_PORT" != "8080" ]; then
  sed -i "s/listen 8080 default_server;/listen ${NGINX_PORT} default_server;/" /etc/nginx/nginx.conf || true
  sed -i "s/listen \[::\]:8080 default_server;/listen \[::\]:${NGINX_PORT} default_server;/" /etc/nginx/nginx.conf || true
fi
nginx -g 'daemon off;' &
NGINX_PID=$!

# Wait for any to exit
wait -n $CENTRAL_PID $REMOTE1_PID $REMOTE2_PID $NGINX_PID || true

echo "One service exited, shutting down others..." >&2
kill -TERM $CENTRAL_PID $REMOTE1_PID $REMOTE2_PID $NGINX_PID 2>/dev/null || true
wait || true

echo "All services stopped." >&2
