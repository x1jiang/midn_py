# PYMIDN - Federated Imputation (SIMI & SIMICE)

This repo provides a Python-only FastAPI implementation of a federated imputation sy## Ready the remote and upload local CSV
Open the remote UI in a browser:
- http://127.0.0.1:8001/
- Enter Job ID (e.g., 1)
- **For SIMI**: Enter Missing Variable Index (1-based) matching the job (e.g., 2)
- **For SIMICE**: The UI will automatically handle multiple target variables based on job parameters
- Choose the local site's CSV
- Click Start

The remote connects to central via WebSocket and sends aggregates.

## Result
When all participants are ready and data have been aggregated:
- Central writes `imputed_data_<job_id>.csv` in the repo root.
- **SIMI**: Single imputed dataset
- **SIMICE**: Multiple complete datasets (currently saves the first one)
- Job status changes to `completed`.

## Algorithm Comparison

| Feature | SIMI | SIMICE |
|---------|------|--------|
| Target Variables | Single column | Multiple columns |
| Variable Types | One type per job | Mixed types (continuous + binary) |
| Approach | Direct imputation | Chained equations |
| Iterations | Fixed | Configurable (before first vs between) |
| Use Case | Simple missing data | Complex missing patterns |

## Logistic SIMI (binary)
Use `is_binary: true` and binary target data (0/1):
- In job creation, set both `parameters.is_binary` and `missing_spec.is_binary` to `true`.
- Remote UI's Missing Variable Index must still match the same column.

## SIMICE with Mixed Types
SIMICE can handle both continuous and binary variables in the same job:
- Set `target_column_indexes` to the list of 1-based column indices
- Set `is_binary_list` array where `true` indicates binary variables, `false` for continuous
- The algorithm will automatically apply Gaussian (continuous) or Logistic (binary) per column.

Two algorithms are implemented: 
- **SIMI** (Sequential Imputation using Model Imputation) - Gaussian and Logistic paths for single variables
- **SIMICE** (Sequential Imputation using Multiple Imputation with Chained Equations) - Multiple variables with mixed types

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

## Create a job (SIMI or SIMICE)
Jobs are validated against the respective algorithm's params.json schema.

### SIMI Job (Single Variable)
SIMI execution uses `missing_spec` to decide the target column and method.

- Create a SIMI job
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

### SIMICE Job (Multiple Variables)
SIMICE handles multiple target columns with mixed types using chained equations.

- Create a SIMICE job
```bash
curl -X POST http://127.0.0.1:8000/api/jobs/ \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "SIMICE demo",
    "description": "Multi-variable SIMICE with mixed types",
    "algorithm": "SIMICE",
    "parameters": {
      "target_column_indexes": [2, 4], 
  "is_binary_list": [false, true],
      "iteration_before_first_imputation": 5,
      "iteration_between_imputations": 3
    },
    "participants": ["<SITE_ID_FROM_CENTRAL>"],
    "missing_spec": {
      "target_column_indexes": [2, 4], 
  "is_binary_list": [false, true]
    },
    "iteration_before_first_imputation": 5,
    "iteration_between_imputations": 3
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

## Email Configuration
The system supports two email sending modes (see `central/app/services/user_service.py`):

1. Test SMTP Mode (send_email_alert_ut_smtp)
   - Uses an unencrypted test server at `129.106.31.45:7725`.
   - Sends approval details as an attachment `approve.txt`.

2. Gmail Mode (send_email_alert)
   - Uses Gmail with an App Password (recommended for production trials).
   - Provides identical attachment (`approve.txt`) and subject line for parity.
   - Configure via environment variables before starting the central server:
     ```bash
     export GMAIL_USER="your_account@gmail.com"
     export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"  # 16-character app password
     ```
   - (Optional) You can define `GMAIL_USER` / `GMAIL_APP_PASSWORD` on the `settings` object if you extend `central/app/core/config.py`.
   - Common issues:
     - App password required (not your normal Gmail password).
     - Ensure 2FA is enabled on the account before generating an app password.
     - Some corporate networks block outbound SMTP (ports 465/587).
   - The function will raise a clear error if credentials are missing.

Fallback: If Gmail credentials aren’t set, user approval still succeeds but email sending will log an error.

## Project layout
- central/app/main.py: Central FastAPI app and WebSocket endpoint
- central/app/services/simi_service.py: SIMI orchestration (Gaussian + Logistic)
- central/app/services/simice_service.py: SIMICE orchestration (Multiple Imputation with Chained Equations)
- remote/app/main.py: Remote FastAPI app + simple UI
- remote/app/websockets.py: Remote WebSocket client using SIMI remote module
- algorithms/SIMI: SIMI algorithm modules and params schema
- algorithms/SIMICE: SIMICE algorithm modules and params schema
- algorithms/R_Reference: Reference R implementations for validation
- common/core: Core statistical functions (Python equivalents of R Core/)
  - least_squares.py: LS(), SILSNet(), ImputeLS() - federated least squares
  - logistic.py: Logit(), SILogitNet(), ImputeLogit() - federated logistic regression
  - transfer.py: Serialization utilities for matrix/vector transmission

## Core Functions
The system now includes centralized statistical functions matching the R reference implementation:

### Least Squares (common/core/least_squares.py)
- `LS(X, y, offset, lam)`: Basic regularized least squares
- `SILSNet(D, idx, yidx, lam, remote_stats)`: Federated least squares aggregation
- `ImputeLS(yidx, beta, sig, manager, participants)`: Broadcast Gaussian imputation parameters

### Logistic Regression (common/core/logistic.py)  
- `Logit(X, y, offset, beta0, lam, maxiter)`: Newton-Raphson logistic with line search
- `SILogitNet(D, idx, yidx, remote_stats)`: Federated logistic regression aggregation
- `ImputeLogit(yidx, alpha, manager, participants)`: Broadcast logistic imputation parameters

### Data Transfer (common/core/transfer.py)
- `serialize_matrix/deserialize_matrix`: Network-safe matrix transmission
- `serialize_vector/deserialize_vector`: Network-safe vector transmission  
- `package_gaussian_stats/package_logistic_stats`: Ready-to-send statistics packages

These core functions ensure mathematical consistency with the R reference implementations.
