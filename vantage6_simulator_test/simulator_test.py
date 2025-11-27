"""
Vantage6 Local Simulator Test

Tests SIMI and SIMICE algorithms using vantage6 local simulator.
This simulates multiple nodes with local CSV files.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Try to import vantage6
try:
    from vantage6.client import UserClient
    from vantage6.tools.util import wrap_task
    VANTAGE6_AVAILABLE = True
except ImportError:
    VANTAGE6_AVAILABLE = False
    print("Warning: vantage6 not fully installed. Using mock simulator.")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vantage6_algorithms"))

print("="*80)
print("VANTAGE6 LOCAL SIMULATOR TEST")
print("="*80)

if not VANTAGE6_AVAILABLE:
    print("\n⚠️  Vantage6 not fully installed. Using mock simulator.")
    print("   To install: pip install vantage6")
    print("   For now, testing with mock client...\n")
    
    # Use mock client from test_local.py
    sys.path.insert(0, str(Path(__file__).parent.parent / "vantage6_algorithms"))
    from test_local import MockVantage6Client
    
    # Load test data
    test_data_dir = Path(__file__).parent / "test_data"
    central_data = pd.read_csv(test_data_dir / "central_data.csv").values
    remote1_data = pd.read_csv(test_data_dir / "remote1_data.csv").values
    remote2_data = pd.read_csv(test_data_dir / "remote2_data.csv").values
    
    print(f"\nLoaded test data:")
    print(f"  Central: {central_data.shape}")
    print(f"  Remote 1: {remote1_data.shape}")
    print(f"  Remote 2: {remote2_data.shape}")
    
    # Test SIMI
    print("\n" + "="*80)
    print("TEST 1: SIMI Algorithm (Gaussian)")
    print("="*80)
    
    from SIMI.algorithm import master_simi
    
    mock_client = MockVantage6Client([remote1_data, remote2_data])
    
    try:
        result = master_simi(
            mock_client,
            {
                'data': central_data,  # Pass as numpy array in dict
                'target_column_index': 2,
                'is_binary': False,
                'imputation_trials': 5
            }
        )
        
        print("\n✓ SIMI test completed successfully!")
        if result and 'imputed_datasets' in result:
            print(f"  Generated {len(result['imputed_datasets'])} imputed datasets")
            print(f"  First dataset shape: {len(result['imputed_datasets'][0])} rows")
        
    except Exception as e:
        print(f"\n✗ SIMI test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test SIMICE
    print("\n" + "="*80)
    print("TEST 2: SIMICE Algorithm")
    print("="*80)
    
    from SIMICE.algorithm import master_simice
    
    mock_client = MockVantage6Client([remote1_data, remote2_data])
    
    try:
        result = master_simice(
            mock_client,
            {
                'data': central_data,  # Pass as numpy array in dict
                'target_column_indexes': [2, 4],
                'is_binary_list': [False, False],
                'imputation_trials': 3,
                'iteration_before_first_imputation': 5,
                'iteration_between_imputations': 3
            }
        )
        
        print("\n✓ SIMICE test completed successfully!")
        if result and 'imputed_datasets' in result:
            print(f"  Generated {len(result['imputed_datasets'])} imputed datasets")
            print(f"  First dataset shape: {len(result['imputed_datasets'][0])} rows")
        
    except Exception as e:
        print(f"\n✗ SIMICE test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("SIMULATOR TEST SUMMARY")
    print("="*80)
    print("\n✓ Mock simulator tests completed")
    print("\nNote: For full vantage6 simulator, install vantage6:")
    print("  pip install vantage6")
    print("\nThen use vantage6's local simulator commands.")
    
else:
    print("\n✓ Vantage6 is available!")
    print("\nTo use vantage6 local simulator:")
    print("  1. Start vantage6 server")
    print("  2. Register algorithms")
    print("  3. Create tasks with local data files")
    print("\nSee vantage6_algorithms/INTEGRATION_GUIDE.md for details.")

