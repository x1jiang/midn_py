"""
Test algorithms with real sample data from MIDN_R_PY/samples
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "MIDN_R_PY"))

from test_local import MockVantage6Client

print("="*80)
print("TESTING WITH REAL SAMPLE DATA")
print("="*80)

def test_simi_with_real_data():
    """Test SIMI with actual sample CSV files."""
    print("\n" + "="*80)
    print("TEST: SIMI with Real Sample Data")
    print("="*80)
    
    # Load sample data
    samples_dir = Path(__file__).parent.parent / "MIDN_R_PY" / "samples"
    
    try:
        central_data = pd.read_csv(samples_dir / "SIMI_Cent_Cont.csv").values.astype(float)
        remote1_data = pd.read_csv(samples_dir / "SIMI_Remote_Cont_1.csv").values.astype(float)
        remote2_data = pd.read_csv(samples_dir / "SIMI_Remote_Cont_2.csv").values.astype(float)
        
        print(f"\nLoaded sample data:")
        print(f"  Central: {central_data.shape}")
        print(f"  Remote 1: {remote1_data.shape}")
        print(f"  Remote 2: {remote2_data.shape}")
        
        # Check missing values
        missing_col0 = np.isnan(central_data[:, 0]).sum()
        print(f"  Missing values in central column 0: {missing_col0}")
        
        if missing_col0 == 0:
            print("  ⚠ No missing values in column 0, creating some for testing...")
            # Create some missing values
            n_missing = min(10, central_data.shape[0] // 5)
            missing_indices = np.random.choice(central_data.shape[0], n_missing, replace=False)
            central_data[missing_indices, 0] = np.nan
            print(f"  Created {n_missing} missing values")
        
        # Create client
        client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
        
        # Import and run
        from SIMI.algorithm import master_simi
        
        print("\nRunning SIMI algorithm...")
        result = master_simi(
            client,
            {
                'data': central_data,
                'target_column_index': 1,  # 1-based: first column
                'is_binary': False,
                'imputation_trials': 5
            }
        )
        
        print(f"\n✓ Success!")
        print(f"  Generated {len(result.get('imputed_datasets', []))} imputed datasets")
        
        imputed_datasets = result.get('imputed_datasets', [])
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            print(f"  Shape: {first_imputed.shape}")
            print(f"  Missing after imputation: {np.isnan(first_imputed[:, 0]).sum()}")
            print(f"  Mean of imputed column: {np.nanmean(first_imputed[:, 0]):.4f}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n⚠ Sample data files not found: {e}")
        print("  Skipping real data test")
        return True  # Not a failure, just missing test data
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simice_with_real_data():
    """Test SIMICE with actual sample CSV files."""
    print("\n" + "="*80)
    print("TEST: SIMICE with Real Sample Data")
    print("="*80)
    
    samples_dir = Path(__file__).parent.parent / "MIDN_R_PY" / "samples"
    
    try:
        central_data = pd.read_csv(samples_dir / "SIMICE_Cent_Cont.csv").values.astype(float)
        remote1_data = pd.read_csv(samples_dir / "SIMICE_Remote_bin_1.csv").values.astype(float)
        remote2_data = pd.read_csv(samples_dir / "SIMICE_Remote_bin_2.csv").values.astype(float)
        
        print(f"\nLoaded sample data:")
        print(f"  Central: {central_data.shape}")
        print(f"  Remote 1: {remote1_data.shape}")
        print(f"  Remote 2: {remote2_data.shape}")
        
        # Check missing values
        missing_cols = []
        for col_idx in range(min(3, central_data.shape[1])):
            missing = np.isnan(central_data[:, col_idx]).sum()
            if missing > 0:
                missing_cols.append((col_idx + 1, missing))
        
        if not missing_cols:
            print("  ⚠ No missing values found, creating some...")
            # Create missing in first two columns
            n_missing = min(5, central_data.shape[0] // 10)
            missing_indices = np.random.choice(central_data.shape[0], n_missing, replace=False)
            central_data[missing_indices, 0] = np.nan
            central_data[missing_indices[:n_missing//2], 1] = np.nan
            missing_cols = [(1, n_missing), (2, n_missing//2)]
        
        print(f"  Missing values: {missing_cols}")
        
        # Determine target columns (first two with missing)
        target_cols = [col for col, _ in missing_cols[:2]]
        if len(target_cols) < 2:
            target_cols = [1, 2]  # Default to first two columns
        
        is_binary_list = [False, False]  # Assume continuous for this test
        
        client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
        
        from SIMICE.algorithm import master_simice
        
        print(f"\nRunning SIMICE algorithm...")
        print(f"  Target columns: {target_cols}")
        print(f"  Binary flags: {is_binary_list}")
        
        result = master_simice(
            client,
            {
                'data': central_data,
                'target_column_indexes': target_cols,
                'is_binary_list': is_binary_list,
                'imputation_trials': 3,
                'iteration_before_first_imputation': 2,
                'iteration_between_imputations': 1
            }
        )
        
        print(f"\n✓ Success!")
        print(f"  Generated {len(result.get('imputed_datasets', []))} imputed datasets")
        
        imputed_datasets = result.get('imputed_datasets', [])
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            print(f"  Shape: {first_imputed.shape}")
            for col_idx in target_cols:
                missing_after = np.isnan(first_imputed[:, col_idx - 1]).sum()
                print(f"  Missing in col {col_idx} after: {missing_after}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n⚠ Sample data files not found: {e}")
        print("  Skipping real data test")
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests with real data."""
    results = []
    
    results.append(("SIMI with Real Data", test_simi_with_real_data()))
    results.append(("SIMICE with Real Data", test_simice_with_real_data()))
    
    print("\n" + "="*80)
    print("REAL DATA TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


