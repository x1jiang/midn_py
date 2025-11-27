"""
Vantage6 Local Simulator Setup for MIDN Algorithms

This script sets up a local vantage6 simulator to test SIMI and SIMICE algorithms
with multiple nodes using local CSV files.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("VANTAGE6 LOCAL SIMULATOR SETUP")
print("="*80)

# Create test data directory
test_data_dir = Path(__file__).parent / "test_data"
test_data_dir.mkdir(exist_ok=True)

print(f"\n1. Creating test data in: {test_data_dir}")

# Generate sample data for 3 nodes
np.random.seed(42)

# Central node data (with missing values)
n_central = 50
central_data = np.random.randn(n_central, 5)
central_data[10:15, 1] = np.nan  # Missing in column 2 (1-based index)
central_df = pd.DataFrame(central_data, columns=[f'col_{i+1}' for i in range(5)])
central_df.to_csv(test_data_dir / "central_data.csv", index=False)
print(f"   ✓ Central data: {central_data.shape} ({np.isnan(central_data[:, 1]).sum()} missing in col 2)")

# Remote node 1 data
n_remote1 = 60
remote1_data = np.random.randn(n_remote1, 5)
remote1_df = pd.DataFrame(remote1_data, columns=[f'col_{i+1}' for i in range(5)])
remote1_df.to_csv(test_data_dir / "remote1_data.csv", index=False)
print(f"   ✓ Remote 1 data: {remote1_data.shape}")

# Remote node 2 data
n_remote2 = 70
remote2_data = np.random.randn(n_remote2, 5)
remote2_df = pd.DataFrame(remote2_data, columns=[f'col_{i+1}' for i in range(5)])
remote2_df.to_csv(test_data_dir / "remote2_data.csv", index=False)
print(f"   ✓ Remote 2 data: {remote2_data.shape}")

# Create simulator configuration
config = {
    "description": "MIDN Algorithm Testing Simulator",
    "version": "1.0.0",
    "nodes": [
        {
            "name": "central",
            "data": str(test_data_dir / "central_data.csv"),
            "port": 5001
        },
        {
            "name": "remote1",
            "data": str(test_data_dir / "remote1_data.csv"),
            "port": 5002
        },
        {
            "name": "remote2",
            "data": str(test_data_dir / "remote2_data.csv"),
            "port": 5003
        }
    ],
    "algorithms": {
        "simi": {
            "image": "simi-algorithm:latest",
            "description": "Single Imputation for Missing Data"
        },
        "simice": {
            "image": "simice-algorithm:latest",
            "description": "Single Imputation for Multiple Columns"
        }
    }
}

config_file = Path(__file__).parent / "simulator_config.json"
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n2. Configuration saved to: {config_file}")

print("\n" + "="*80)
print("SETUP COMPLETE")
print("="*80)
print(f"\nTest data directory: {test_data_dir}")
print(f"Configuration file: {config_file}")
print("\nNext steps:")
print("  1. Ensure Docker containers are built: cd ../vantage6_algorithms && ./build.sh")
print("  2. Run simulator test: python simulator_test.py")

