# PYMIDN Architecture

This document describes the system architecture for the Python-based MIDN project (ignore `archive/`).

- Central service: FastAPI app in `central/app/main.py`.
- Remote services: FastAPI apps in `remote/app/main.py`. In the demo container they're exposed twice under `/remote1` and `/remote2`; in real deployments, remotes can run on any host/port.
- Reverse proxy: Nginx fronts the services and exposes a single HTTP port.
- Runtime entrypoint: `entrypoint.sh` launches 3 uvicorn processes (central:8000, remote1:8001, remote2:8002) and Nginx on 8080 (or `$PORT`).
- Persistence: SQLite DB at `data/central.db` managed by SQLAlchemy models in `central/app/models/`.
- Configuration: Algorithm parameter schemas in `config/*.json`; remote site configuration in `data/site_config.json`.
- Algorithms: `MIDN_R_PY` holds core algorithms; SIMI and SIMICE are implemented in Python wrappers and invoked dynamically.

## Components

- Central (FastAPI)
  - GUI routes render templates in `central/app/templates/` and serve assets from `central/app/static/`.
  - Admin auth uses a simple cookie + CSRF token (`central/app/core/csrf.py`).
  - REST APIs grouped under `/api/*` (users, jobs, remote info).
  - WebSocket endpoint at `/ws/{site_id}` for algorithm coordination.
  - Job runtime state kept in-memory (`jobs_runtime`, `remote_websockets`).
  - Algorithm runner loads modules from `MIDN_R_PY` via `sys.path` and streams algorithm stdout/err into job logs.

- Remote (FastAPI)
  - UI in `remote/app/static/` (templated HTML) to configure central URL, site ID, JWT, and to start/monitor local work.
  - Talks to central over HTTP for job listing (`/api/remote/jobs`) and WS for algorithm messages.
  - Maintains per-site in-memory job state on `app.state`.

- Nginx
  - Listens on 8080 (or `$PORT`), proxies `/` to central (8000), `/remote1/*` to 8001, `/remote2/*` to 8002.
  - Note: `/remote1` and `/remote2` mappings are for the bundled demo only; production setups may point to different remote hosts/ports or separate services.
  - Upgrades WebSocket connections and forwards X-Forwarded-* headers.

## Data flow

1. Admin logs into central and creates a job using schemas in `config/` (SIMI or SIMICE). Jobs stored in SQLite.
2. Admin approves remote sites; central emails credentials including CENTRAL_URL, SITE_ID, JWT.
3. Each remote configures its `data/site_config.json` (or via UI) with the HTTP and WS URLs, site ID, token.
4. When a job is started (central GUI), central waits for all expected remotes to connect to `/ws/{site_id}`.
5. Central imports the algorithm entrypoint:
   - SIMI: `MIDN_R_PY/SIMI/SIMICentral.py::simi_central`
   - SIMICE: `MIDN_R_PY/SIMICE/SIMICECentral.py::simice_central`
6. Central streams messages to/from remotes via the maintained WebSocket connections and produces outputs under `central/app/static/results/`.

## Ports and paths

- Central uvicorn: 8000
- Remote1 uvicorn (demo): 8001 (mounted under `/remote1/` by Nginx)
- Remote2 uvicorn (demo): 8002 (mounted under `/remote2/` by Nginx)
- Nginx public: 8080 (or `$PORT` in managed platforms)
- SQLite file: `data/central.db`
- Remote site config: `data/site_config.json`
- Demo datasets: `central/app/static/demo_data/*.csv`
- Results: `central/app/static/results/job_<id>_<timestamp>/*.csv` and a zip per job

## Algorithms coverage

- Implemented in Python: SIMI, SIMICE (wrappers around MIDN_R_PY core).
- Other algorithms are kept in `MIDN_R_PY` (R/Python originals) but not wired via the Python central yet.
- Protocol notes: the runtime communication protocol follows the original R implementation's message structure.

## Security

- Admin GUI/API protected by cookie + CSRF; remote APIs use JWT tokens issued by central on approval.
- Nginx forwards X-Forwarded-Proto to ensure correct base URL handling behind HTTPS proxies.
