# Vantage6 Local Simulator Testing

This directory contains scripts for testing MIDN algorithms using a local simulator that mimics the vantage6 framework with multiple nodes using local CSV files.

## Quick Start

### 1. Setup Test Data

```bash
python3 simulator_setup.py
```

This creates:
- `test_data/` directory with CSV files for central and remote nodes
- `simulator_config.json` with configuration

### 2. Run Simulator Tests

```bash
python3 simulator_test.py
```

## What It Does

The simulator:
1. **Creates test data**: Generates sample CSV files for central node (with missing values) and 2 remote nodes
2. **Uses mock client**: Simulates vantage6's RPC pattern without requiring full vantage6 installation
3. **Tests algorithms**: Runs SIMI and SIMICE algorithms with the test data
4. **Validates results**: Checks that imputation works correctly

## Test Results

### ✅ SIMI Algorithm
- **Status**: Working correctly
- **Test**: Gaussian imputation with multiple datasets
- **Result**: Successfully generates 5 imputed datasets

### ⚠️ SIMICE Algorithm
- **Status**: Has minor issue with variable initialization
- **Note**: Works in main test suite, needs investigation for simulator

## Files

- `simulator_setup.py` - Creates test data and configuration
- `simulator_test.py` - Runs algorithm tests with mock client
- `test_data/` - Generated CSV files for testing
- `simulator_config.json` - Configuration file

## Using with Real Vantage6

For full vantage6 local simulator:

1. Install vantage6:
   ```bash
   pip install vantage6
   ```

2. Start vantage6 server (see vantage6 documentation)

3. Register algorithms:
   ```bash
   # Use the Docker images built in ../vantage6_algorithms
   docker images | grep -E "(simi|simice)-algorithm"
   ```

4. Create tasks with local data files

See `../vantage6_algorithms/INTEGRATION_GUIDE.md` for complete instructions.

## Notes

- The mock simulator uses the same RPC pattern as vantage6
- Test data is generated with known missing patterns for validation
- Results match expected behavior from main test suite

