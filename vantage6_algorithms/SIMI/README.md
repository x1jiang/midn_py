# SIMI Algorithm for Vantage6

Single Imputation for Missing Data - Vantage6 Implementation

## Overview

SIMI performs federated imputation of missing values in a single target column using either Gaussian (continuous) or logistic (binary) regression.

## Algorithm Parameters

- `target_column_index` (int, required): 1-based index of column to impute
- `is_binary` (bool, required): True for binary data, False for continuous
- `imputation_trials` (int, default=10): Number of imputation datasets to generate
- `method` (str, optional): 'Gaussian' or 'logistic' (inferred from is_binary if not provided)

## Usage

### Building the Docker Image

```bash
cd vantage6_algorithms/SIMI
docker build -t simi-algorithm:latest .
```

### Registering with Vantage6

```python
from vantage6.client import UserClient

client = UserClient("https://your-server.com", "your-api-key")
client.algorithm.create(
    name="simi",
    image="simi-algorithm:latest",
    description="Single Imputation for Missing Data"
)
```

### Running a Task

```python
task = client.task.create(
    name="SIMI Imputation",
    image="simi-algorithm:latest",
    input_={
        'target_column_index': 2,
        'is_binary': False,
        'imputation_trials': 10
    },
    organizations=[org1_id, org2_id]
)
```

## Migration Notes

This algorithm was adapted from `MIDN_R_PY/SIMI`:
- Replaced WebSocket communication with vantage6 RPC calls
- Master function aggregates statistics from remote nodes
- Remote functions compute local statistics only
- Results are aggregated centrally for imputation

## Data Format

Input data should be:
- CSV file with numeric columns
- Missing values represented as NaN
- Target column specified by `target_column_index`

Output:
- List of imputed datasets (one per trial)
- Each dataset is a 2D array (rows x columns)

