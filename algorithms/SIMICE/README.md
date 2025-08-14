# SIMICE Algorithm Implementation

This directory contains the Python implementation of the SIMICE (Sequential Imputation using Multiple Imputation with Chained Equations) algorithm for the PYMIDN federated learning system.

## Overview

SIMICE extends the SIMI algorithm to handle multiple target columns with missing values simultaneously using a chained equations approach. Unlike SIMI which imputes one variable at a time, SIMICE:

1. **Handles multiple target variables**: Can impute several columns in the same run
2. **Uses chained equations**: Cycles through all target variables in each iteration
3. **Supports mixed variable types**: Handles both continuous (Gaussian) and binary (logistic) variables
4. **Iterative refinement**: Allows different iteration counts before first imputation and between subsequent imputations

## Algorithm Parameters

The algorithm accepts the following parameters (as defined in `params.json`):

- **`target_column_indexes`**: List of 1-based indices of columns to be imputed
- **`is_binary`**: List of booleans indicating if each target column is binary (true) or continuous (false)  
- **`iteration_before_first_imputation`**: Number of iterations to run before extracting the first complete dataset
- **`iteration_between_imputations`**: Number of iterations to run between subsequent complete datasets

## Files Structure

```
SIMICE/
├── __init__.py                 # Package initialization
├── params.json                 # Parameter schema definition
├── simice_central.py          # Central algorithm implementation (CentralAlgorithm interface)
├── simice_remote.py           # Remote algorithm implementation (RemoteAlgorithm interface)  
├── simice_central_base.py     # Base functionality for central site
├── simice_remote_base.py      # Base functionality for remote sites
├── SIMICECentral.py          # R-compatible central implementation
├── SIMICERemote.py           # R-compatible remote implementation
├── test_simice.py            # Test suite
└── README.md                 # This file
```

## Key Implementation Classes

### SIMICECentralAlgorithm
- Implements the `CentralAlgorithm` interface
- Coordinates imputation across multiple target variables
- Handles aggregation of statistics from remote sites
- Generates multiple complete datasets

### SIMICERemoteAlgorithm  
- Implements the `RemoteAlgorithm` interface
- Computes local statistics for each target variable
- Communicates with central site through message passing
- Updates local imputations based on central site instructions

### SIMICECentral & SIMICERemote
- R-compatible implementations matching the reference R code behavior
- Use WebSocket communication for distributed computation
- Handle the full MICE algorithm workflow

## Algorithm Workflow

### Initialization Phase
1. **Data Preparation**: Each site prepares its local data
2. **Missing Value Initialization**: Initialize missing values with simple imputation (mean for continuous, mode for binary)
3. **Site Registration**: Remote sites register with the central site

### Imputation Phase  
For each complete dataset (M total):
1. **Iteration Loop**: Run specified number of iterations
   - First dataset: `iteration_before_first_imputation` iterations
   - Subsequent datasets: `iteration_between_imputations` iterations
2. **Variable Cycling**: Within each iteration, cycle through all target variables
3. **Local Statistics**: Each site computes statistics for the current target variable
4. **Aggregation**: Central site aggregates statistics from all sites
5. **Imputation**: Generate new imputed values and update local datasets
6. **Dataset Storage**: After all iterations, store the complete dataset

## Usage Example

```python
import numpy as np
import pandas as pd
from algorithms.SIMICE.simice_central import SIMICECentralAlgorithm

# Create data with missing values
data = pd.DataFrame({
    'var1': [1, 2, np.nan, 4, 5],
    'var2': [1.1, np.nan, 3.1, 4.1, 5.1], 
    'var3': [1, 0, 1, np.nan, 0],
    'var4': [2.5, 3.5, 4.5, 5.5, 6.5]
})

# Setup algorithm
central = SIMICECentralAlgorithm()

# Run imputation
imputed_datasets = await central.impute(
    data=data,
    target_column_indexes=[2, 3],  # 1-based: var2 and var3
    is_binary=[False, True],       # var2 is continuous, var3 is binary
    iteration_before_first_imputation=5,
    iteration_between_imputations=3,
    imputation_count=10
)

print(f"Generated {len(imputed_datasets)} complete datasets")
```

## Differences from SIMI

| Aspect | SIMI | SIMICE |
|--------|------|--------|
| Target Variables | Single variable | Multiple variables |
| Approach | Direct imputation | Chained equations |
| Iterations | Fixed per imputation | Configurable (before first vs. between) |
| Variable Types | One type per run | Mixed types in same run |
| Convergence | Single model | Iterative refinement across variables |

## Integration with PYMIDN

The SIMICE algorithm is registered in the common algorithm registry and can be used through the standard PYMIDN interfaces:

```python
from common.algorithm.registry import AlgorithmRegistry

# Get SIMICE implementations
central_class = AlgorithmRegistry.get_central_algorithm("SIMICE")
remote_class = AlgorithmRegistry.get_remote_algorithm("SIMICE")

# Create instances
central = central_class()
remote = remote_class()
```

## Testing

Run the test suite to verify the implementation:

```bash
cd algorithms/SIMICE
python test_simice.py
```

The test suite validates:
- Basic algorithm functionality
- Parameter validation  
- Message processing
- Imputation quality (no missing values remain, binary variables stay binary)

## Dependencies

- numpy
- pandas  
- scipy
- scikit-learn
- websockets (for R-compatible implementations)
- asyncio (for asynchronous processing)

## Reference

Based on the R reference implementation in `algorithms/R_Reference/SIMICE/` and adapted for the PYMIDN federated learning framework.
