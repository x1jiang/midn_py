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

# Wait for any to exit
wait -n $CENTRAL_PID $REMOTE1_PID $REMOTE2_PID || true

echo "One service exited, shutting down others..." >&2
kill -TERM $CENTRAL_PID $REMOTE1_PID $REMOTE2_PID 2>/dev/null || true
wait || true

echo "All services stopped." >&2
