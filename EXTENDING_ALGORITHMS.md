# Extending: Add a new algorithm

This guide shows how to add a new federated algorithm to the Python orchestration.

Scope: current Python orchestration implements SIMI and SIMICE via wrappers around `MIDN_R_PY`. Other algorithms remain in `MIDN_R_PY` and can be brought in similarly.

## 1) Define parameters (JSON schema)

Add a schema file in `config/YourAlgo.json` similar to `config/SIMI.json` or `config/SIMICE.json`:
- `title`: upper-case algorithm key (e.g., "MYALGO")
- `properties`: input fields with types, constraints, and optional UI hints (`ui:label`, `ui:order`, `ui:placeholder`, `ui:arrayInput`)
- `required`: list of required keys

The central GUI uses these to render forms and validate inputs. See `central/app/core/alg_config.py`.

## 2) Register algorithm name

- Add your upper-case code to `central/app/core/config.py` list `settings._ALG` so it appears in the Create Job page.

Example: `_ALG = ["SIMI","SIMICE","MYALGO"]`

## 3) Provide central entrypoint

Create a module under `MIDN_R_PY/<YourAlgo>/` that exports an async function with the contract:
- Inputs:
  - `D: np.ndarray` central dataset (float64)
  - `config: dict` parameters as submitted (already coerced to int/float/bool/list)
  - `site_ids: list[str]` ordered participant site IDs
  - `websockets: dict[str, WebSocket]` map of connected site_id -> FastAPI WebSocket
- Behavior: orchestrate computation, exchange messages with remotes per your protocol, and return either a single dataset or a list of datasets to persist.

Reference implementations:
- SIMI central entry: `MIDN_R_PY/SIMI/SIMICentral.py::simi_central`
- SIMICE central entry: `MIDN_R_PY/SIMICE/SIMICECentral.py::simice_central`

You can stream progress by printing; central captures stdout/stderr and shows it in the job log.

## 4) Provide remote-side logic (if needed)

If your algorithm needs remote computation:
- Add code under `MIDN_R_PY/<YourAlgo>/` for the remote logic.
- Reuse the existing WebSocket client in `remote/app/communication/connection_client.py` or send/recv using the same wire format as SIMI/SIMICE.
- Protocol: follow the original R message structure for your algorithm's interactions.

Remote UI can stay unchanged if central triggers jobs and the remote just reacts to WS messages and local dataset upload; otherwise, add any new form fields in `remote/app/static/index.html` and pass them in the `start_job` form.

## 5) Wire central to call your entrypoint

In `central/app/main.py` where the algorithm is selected, add a branch to import and call your module:

```python
if algorithm_name == "myalgo":
    algorithm_central = importlib.import_module("MYALGO.MYALGOCentral").myalgo_central
```

Ensure `sys.path` includes `MIDN_R_PY` (already handled at top of `central/app/main.py`).

## 6) Testing locally

- Use the DEMO datasets as scaffolding or create your own CSVs.
- Run the container and start a job for your algo.
- Validate that:
  - All remotes connect to `/ws/{site_id}`
  - Your algorithm log lines appear live on the Start page
  - Results are written under `central/app/static/results` and download works

## 7) Tips and conventions

- Indices: Prefer 1-based indices in user-facing forms when aligning with CSV columns, but normalize to 0-based internally where needed.
- Data types: Keep data numeric; convert to `float64` early. Catch conversion errors and report clearly.
- Wire protocol: Prefer lowercase method names ("gaussian"/"logistic") and a single `iterate` with a `mode` flag if that matches the original R behavior for your algorithm. The system currently adheres to the R-side message structure.
- Security: Keep JWT-protected APIs for remote listing and avoid exposing internals.

## 8) Optional: remote APIs

If your algorithm needs extra remote-side endpoints, add them under `remote/app/routes/` and include the router in `remote/app/main.py`.

## 9) Documentation

- Update `DEMO.md` with any algorithm-specific steps.
- Add `config/<YourAlgo>.json` to capture parameters and UI order.
