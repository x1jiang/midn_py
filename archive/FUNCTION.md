# Federated Imputation Requirements (FastAPI, Python-only)

## Scope & Goal
- Central site orchestrates federated imputation; **central has its own dataset** with missing values.
- **Remote sites help the central** impute; raw data never leaves remotes.
- **Final imputed dataset is produced only at the central** (no remote copies).

## Topology
- **Central FastAPI**: single **TLS port** for both HTTP management and **WebSocket** communication.
- **Remote FastAPI** (local-only UI): users upload local CSVs, view jobs, and manually start participation.
- **All central↔remote orchestration uses WebSockets** (persistent connections initiated by remotes).

## Registration & Access
- Remote users register at central with **username, password, email, institution**.
- Central admin approves; system issues a **site_id** (short hash). (will send to email)
- Remote has a config windows for them to stores **central URL/port** and **site_id**; remotes authenticate via JWT over WebSocket.

## Algorithms
- Each algorithm is provided as two Python modules:
  - **`xxxCentral.py`** (central-side logic)
  - **`xxxRemote.py`** (remote-side logic)
- Central **algorithm registration** uploads both files and a parameter schema; central keeps versions and distributes `xxxRemote.py` to sites as needed.
- Algorithms may support **multi-feature missingness** and must declare their parameter schema and capabilities.

## Jobs
- Central **registers a job** with:
  - **Participants**: list of `site_id`s.
  - **Algorithm** and **validated parameters**.
  - **Imputed output**: **central-only** (required).
  - ** Number of Imputing running**:.
  - **Missing specification**:
    - upload the central data to help to identify the **column names** ( by name in selection), but call the Algorithm with  **1-based column indices** 
    - **Missing type is per-feature** (each target can be binary or continuous).
  - **Iterations (for multi-feature)**:
    - `iteration_before_first_imputation` (int)
    - `iteration_between_imputations` (int)
- **Start semantics**:
  - Each remote **manually readies** by open a job and uploading CSV in its local UI and clicking **Start**.
  - **Central waits for all required sites to be online and ready**; only then begins orchestration.
  - Central also acts as a **participant** with its own dataset (uploaded at central).

## Data & Formats
- **CSV only** for datasets.
- the columns are the same across sites.
- Rows are dynamic; inputiaon targets are specified by **columns** (1-based indices).
- Each site must upload CSV to participate.

## Execution & Communication
- Central drives via `xxxCentral`:
- Remotes run `xxxRemote.py` steps on local data and return **only aggregates/partials** and status (no raw data).
- Asynchronous execution with progress reporting; remote or central can cancel the job.

## Privacy & Retention
- **Do not retain datasets or derived artifacts** at central or remotes—**keep logs only**.
- Central’s final imputed dataset is generated for **immediate download** and **not stored** afterward.
- No intermediate artifacts persist beyond runtime, except logs.

## Quorum & Availability
- **Job does not start** until **all** required sites are connected and have manually readied.
- If a site is offline before start, central continues waiting. (Default policy: wait for all.)

## Remote UI Expectations
- View assigned jobs and algorithm info.
- Upload CSV, see target feature specs, **manually start**, monitor progress, and cancel participation if needed.

## Security
- JWT-based authentication for users and sites; `site_id` identifies the remote.
- All comms over the single central TLS port (HTTP + WebSocket).

## Out of Scope / Assumptions
- Python-only stack; no non-Python services required.
- Datasets are **never** shared between sites; only algorithm-defined aggregates/partials move over WebSockets.
