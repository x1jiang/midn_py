# PYMIDN - Federated Imputation (SIMI)

This repo provides a Python-only FastAPI implementation of a federated imputation system. The central server orchestrates imputation jobs; remote sites contribute only aggregates via WebSockets.

Only one algorithm is implemented: SIMI (Gaussian and Logistic paths).

## Prerequisites
- Python 3.10+
- macOS or Linux

## Setup
```bash
# From repo root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start services
- Central API (HTTP + WebSockets on port 8000):
```bash
uvicorn central.app.main:app --reload --host 0.0.0.0 --port 8000
```
- Remote UI (HTTP on port 8001):
```bash
uvicorn remote.app.main:app --reload --host 0.0.0.0 --port 8001
```

## Create a JWT for remote WebSocket auth
WebSockets require a JWT query param `token` that central verifies using `central/app/core/config.py -> SECRET_KEY` (default: `a_very_secret_key`).

From a Python REPL:
```python
from jose import jwt
from datetime import datetime, timedelta
SECRET_KEY = "a_very_secret_key"  # must match central
payload = {"sub": "remote-user", "exp": datetime.utcnow() + timedelta(hours=4)}
print(jwt.encode(payload, SECRET_KEY, algorithm="HS256"))
```
Copy the printed token.

## Create a user and approve (admin flow)
- Create user
```bash
curl -X POST http://127.0.0.1:8000/api/users/ \
  -H 'Content-Type: application/json' \
  -d '{
    "username": "siteA",
    "email": "siteA@example.com",
    "institution": "Test Hospital",
    "password": "pass123"
  }'
```
- Approve user (triggers email with site URL and site_id)
```bash
curl -X POST http://127.0.0.1:8000/api/users/1/approve
```
- Retrieve user (to confirm `site_id`)
```bash
curl http://127.0.0.1:8000/api/users/1
```
Note: Email sending uses test SMTP `129.106.31.45:7725`. Adjust in `central/app/services/user_service.py` if needed.

## Configure remote
Set environment variables or edit `remote/app/config.py`:
- CENTRAL_URL: `ws://127.0.0.1:8000`
- SITE_ID: the user’s `site_id` from central
- TOKEN: the JWT generated above

Example (shell):
```bash
export CENTRAL_URL=ws://127.0.0.1:8000
export SITE_ID=<SITE_ID_FROM_CENTRAL>
export TOKEN=<JWT_FROM_ABOVE>
```
Restart the remote app if it’s already running.

## Create a job (SIMI)
Jobs are validated against `algorithms/SIMI/params.json`, and SIMI execution uses `missing_spec` to decide the target column and method.

- Create a job
```bash
curl -X POST http://127.0.0.1:8000/api/jobs/ \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "SIMI demo",
    "description": "Gaussian SIMI",
    "algorithm": "SIMI",
    "parameters": {"target_column_index": 2, "is_binary": false},
    "participants": ["<SITE_ID_FROM_CENTRAL>"] ,
    "missing_spec": {"target_column_index": 2, "is_binary": false},
    "iteration_before_first_imputation": 0,
    "iteration_between_imputations": 0
  }'
```
- Start the job (upload central CSV)
```bash
curl -X POST "http://127.0.0.1:8000/api/jobs/1/start" \
  -F central_data_file=@/full/path/to/central.csv
```

## Ready the remote and upload local CSV
Open the remote UI in a browser:
- http://127.0.0.1:8001/
- Enter Job ID (e.g., 1)
- Enter Missing Variable Index (1-based) matching the job (e.g., 2)
- Choose the local site’s CSV
- Click Start

The remote connects to central via WebSocket and sends aggregates.

## Result
When all participants are ready and data have been aggregated:
- Central writes `imputed_data_<job_id>.csv` in the repo root.
- Job status changes to `completed`.

## Logistic SIMI (binary)
Use `is_binary: true` and binary target data (0/1):
- In job creation, set both `parameters.is_binary` and `missing_spec.is_binary` to `true`.
- Remote UI’s Missing Variable Index must still match the same column.

## Troubleshooting
- 401 on WebSocket: ensure TOKEN is a valid JWT with the same SECRET_KEY as central.
- Quorum wait: job starts only when all `participants` are connected and have sent data.
- Email failures: check SMTP host/port in `user_service.send_email_alert`.

## Project layout
- central/app/main.py: Central FastAPI app and WebSocket endpoint
- central/app/services/simi_service.py: SIMI orchestration (Gaussian + Logistic)
- remote/app/main.py: Remote FastAPI app + simple UI
- remote/app/websockets.py: Remote WebSocket client using SIMI remote module
- algorithms/SIMI: Uploaded SIMI modules and params schema
