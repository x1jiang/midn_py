# SIMICE Algorithm for Vantage6

Single Imputation for Multiple Columns - Vantage6 Implementation

## Overview

SIMICE performs federated imputation of missing values in multiple target columns using iterative chained equations with either Gaussian (continuous) or logistic (binary) regression.

## Algorithm Parameters

- `target_column_indexes` (list[int], required): 1-based indices of columns to impute
- `is_binary_list` (list[bool], required): Boolean list indicating binary (True) or continuous (False) for each target column
- `imputation_trials` (int, default=10): Number of imputation datasets to generate
- `iteration_before_first_imputation` (int, default=0): Iterations to run before first imputation
- `iteration_between_imputations` (int, default=0): Iterations to run between subsequent imputations

## Usage

### Building the Docker Image

```bash
cd vantage6_algorithms/SIMICE
docker build -t simice-algorithm:latest .
```

### Registering with Vantage6

```python
from vantage6.client import UserClient

client = UserClient("https://your-server.com", "your-api-key")
client.algorithm.create(
    name="simice",
    image="simice-algorithm:latest",
    description="Single Imputation for Multiple Columns"
)
```

### Running a Task

```python
task = client.task.create(
    name="SIMICE Imputation",
    image="simice-algorithm:latest",
    input_={
        'target_column_indexes': [2, 5, 7],
        'is_binary_list': [False, True, False],
        'imputation_trials': 10,
        'iteration_before_first_imputation': 5,
        'iteration_between_imputations': 3
    },
    organizations=[org1_id, org2_id]
)
```

## Migration Notes

This algorithm was adapted from `MIDN_R_PY/SIMICE`:
- Replaced WebSocket communication with vantage6 RPC calls
- Master function orchestrates iterative imputation
- Remote functions compute statistics for each iteration
- Handles multiple target columns with different methods

## Data Format

Input data should be:
- CSV file with numeric columns
- Missing values represented as NaN
- Multiple target columns specified by `target_column_indexes`

Output:
- List of imputed datasets (one per trial)
- Each dataset is a 2D array (rows x columns)


