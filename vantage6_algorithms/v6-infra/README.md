# Vantage6 Demo Network Infrastructure

This repository provides a demo setup for running a [vantage6](https://vantage6.ai/) server, nodes, and UI locally for testing and development.

## Getting Started

1. **Clone** this repository.
2. **Check** or **edit** `config.env` to set any desired defaults (e.g., vantage6 version, Docker registry, UI port, etc.).
3. **Run** the setup script:
    ```bash
    # Development usage (won't stop on errors, opens browser)
    ENVIRONMENT=DEV ./setup.sh
    ```
   or
    ```bash
    # CI usage (stops on errors, no browser launch)
    ENVIRONMENT=CI ./setup.sh
    ```
    If you omit `ENVIRONMENT=...`, it falls back to whatever is in `config.env`.

4. **Verify** containers are running:
    ```bash
    docker ps
    ```
    You should see the vantage6 server, nodes, and UI container.

5. **Interact** with vantage6 (e.g., run an algorithm). The vantage6 UI can be accessed at http://localhost (configurable in `config.env`).

6. **Stop** and **remove** all containers:
    ```bash
    # Use the same ENVIRONMENT mode you started with, if desired
    ENVIRONMENT=DEV ./shutdown.sh
    ```
    This tears down the vantage6 environment and cleans up leftover files.

## Imputation Demo Setup
- Data for the imputation algorithms lives in `infrastructure/data/imputation` (gamma has missing values in `impute_continuous` [col 3] and `impute_binary` [col 5]; alpha/beta are complete). `config.env` now points `ALGO_DATA_DIRECTORY` to this folder and the node YAML files use the same paths.
- Build algorithm images from the repo root:
  - `docker build --build-arg ALGORITHM=SIMI -t simi-algorithm:latest -f Dockerfile .`
  - `docker build --build-arg ALGORITHM=SIMICE -t simice-algorithm:latest -f Dockerfile .`
- Algorithm presets are defined in `algorithms/settings/algorithms.toml`: `simi_gaussian` (single imputation, target col 3) and `simice_multi` (multi-column imputation, target cols 3 & 5).
- After the infrastructure is up and you have vantage6 client installed locally, run a task from the repo root:
  - `ALGORITHM=simi_gaussian python v6-infra/algorithms/run.py`
  - `ALGORITHM=simice_multi python v6-infra/algorithms/run.py`


