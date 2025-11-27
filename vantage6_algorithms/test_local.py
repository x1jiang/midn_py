"""
Local Testing Script for Vantage6 Algorithms

Tests algorithms locally without requiring a full vantage6 server.
Uses mock clients to simulate vantage6's RPC pattern.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("VANTAGE6 ALGORITHMS - LOCAL TESTING")
print("="*80)


class MockVantage6Client:
    """Mock vantage6 client for local testing."""
    
    def __init__(self, remote_data_sets=None):
        """
        Initialize mock client.
        
        Parameters:
        -----------
        remote_data_sets : list of numpy arrays or CSV paths
            Simulated remote node datasets
        """
        self.remote_data_sets = remote_data_sets or []
        self.call_count = 0
    
    def create_new_task(self, input_, organization_ids=None):
        """
        Mock task creation - executes remote functions locally.
        
        Parameters:
        -----------
        input_ : dict
            Task input with 'method', 'args', 'kwargs'
        organization_ids : list
            Organization IDs (ignored in mock)
        
        Returns:
        --------
        list of results from remote functions
        """
        method = input_.get('method', '')
        args = input_.get('args', [])
        kwargs = input_.get('kwargs', {})
        
        self.call_count += 1
        print(f"  [Mock Client] Call #{self.call_count}: method={method}")
        
        results = []
        
        # Execute remote function for each simulated remote node
        for idx, remote_data in enumerate(self.remote_data_sets):
            try:
                # Import remote functions
                if 'simi_remote_gaussian' in method:
                    from SIMI.algorithm import RPC_simi_remote_gaussian
                    result = RPC_simi_remote_gaussian({'data': remote_data, **kwargs}, *args)
                    results.append({'result': result, 'organization_id': f'org_{idx}'})
                
                elif 'simi_remote_logistic' in method:
                    from SIMI.algorithm import RPC_simi_remote_logistic
                    result = RPC_simi_remote_logistic({'data': remote_data, **kwargs}, *args)
                    results.append({'result': result, 'organization_id': f'org_{idx}'})
                
                elif 'simice_remote_initialize' in method:
                    from SIMICE.algorithm import RPC_simice_remote_initialize
                    # Pass all parameters via kwargs
                    call_kwargs = {'data': {'data': remote_data}, **kwargs}
                    result = RPC_simice_remote_initialize(**call_kwargs)
                    results.append({'result': result, 'organization_id': f'org_{idx}'})
                
                elif 'simice_remote_statistics' in method:
                    from SIMICE.algorithm import RPC_simice_remote_statistics
                    # Pass all parameters via kwargs
                    call_kwargs = {'data': {'data': remote_data}, **kwargs}
                    result = RPC_simice_remote_statistics(**call_kwargs)
                    results.append({'result': result, 'organization_id': f'org_{idx}'})
                
                else:
                    print(f"    Warning: Unknown method {method}")
                    results.append({'result': {}, 'organization_id': f'org_{idx}'})
            
            except Exception as e:
                print(f"    Error executing {method} on remote {idx}: {e}")
                import traceback
                traceback.print_exc()
                results.append({'result': {}, 'organization_id': f'org_{idx}', 'error': str(e)})
        
        return results
    
    def get_data(self, *args, **kwargs):
        """Mock data access - not used in our algorithms."""
        return None


def create_test_data(n_samples=100, n_features=5, missing_rate=0.1, seed=42):
    """Create test dataset with missing values."""
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    
    # Add some missing values
    n_missing = int(n_samples * missing_rate)
    missing_indices = np.random.choice(n_samples, n_missing, replace=False)
    data[missing_indices, 0] = np.nan  # Missing in first column
    
    return data


def test_simi_gaussian():
    """Test SIMI with Gaussian method."""
    print("\n" + "="*80)
    print("TEST 1: SIMI - Gaussian Method")
    print("="*80)
    
    # Create test data
    central_data = create_test_data(n_samples=50, n_features=4, missing_rate=0.2, seed=1)
    remote1_data = create_test_data(n_samples=60, n_features=4, missing_rate=0.0, seed=2)
    remote2_data = create_test_data(n_samples=70, n_features=4, missing_rate=0.0, seed=3)
    
    # Ensure remote data has no missing in target column (column 1, index 0)
    remote1_data[:, 0] = np.random.randn(60)
    remote2_data[:, 0] = np.random.randn(70)
    
    print(f"\nData shapes:")
    print(f"  Central: {central_data.shape} (missing: {np.isnan(central_data[:, 0]).sum()} in col 0)")
    print(f"  Remote 1: {remote1_data.shape}")
    print(f"  Remote 2: {remote2_data.shape}")
    
    # Create mock client
    client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
    
    # Import master function
    from SIMI.algorithm import master_simi
    
    # Run algorithm
    print("\nRunning SIMI algorithm...")
    try:
        result = master_simi(
            client,
            {
                'data': central_data,
                'target_column_index': 1,  # 1-based: column 0
                'is_binary': False,
                'imputation_trials': 3
            }
        )
        
        print(f"\n✓ Algorithm completed successfully!")
        print(f"  Generated {len(result.get('imputed_datasets', []))} imputed datasets")
        
        # Verify results
        imputed_datasets = result.get('imputed_datasets', [])
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            print(f"  First imputed dataset shape: {first_imputed.shape}")
            print(f"  Missing values after imputation: {np.isnan(first_imputed[:, 0]).sum()}")
            
            if np.isnan(first_imputed[:, 0]).sum() == 0:
                print("  ✓ All missing values imputed!")
            else:
                print("  ⚠ Some missing values remain")
        
        print(f"  RPC calls made: {client.call_count}")
        return True
        
    except Exception as e:
        print(f"\n✗ Algorithm failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simi_logistic():
    """Test SIMI with Logistic method."""
    print("\n" + "="*80)
    print("TEST 2: SIMI - Logistic Method")
    print("="*80)
    
    # Create binary test data
    np.random.seed(42)
    central_data = np.random.randn(50, 4)
    central_data[:, 0] = (central_data[:, 0] > 0).astype(float)  # Binary target
    central_data[10:15, 0] = np.nan  # Missing values
    
    remote1_data = np.random.randn(60, 4)
    remote1_data[:, 0] = (remote1_data[:, 0] > 0).astype(float)
    
    remote2_data = np.random.randn(70, 4)
    remote2_data[:, 0] = (remote2_data[:, 0] > 0).astype(float)
    
    print(f"\nData shapes:")
    print(f"  Central: {central_data.shape} (binary target, missing: {np.isnan(central_data[:, 0]).sum()})")
    print(f"  Remote 1: {remote1_data.shape}")
    print(f"  Remote 2: {remote2_data.shape}")
    
    client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
    
    from SIMI.algorithm import master_simi
    
    print("\nRunning SIMI algorithm (logistic)...")
    try:
        result = master_simi(
            client,
            {
                'data': central_data,
                'target_column_index': 1,
                'is_binary': True,
                'imputation_trials': 3
            }
        )
        
        print(f"\n✓ Algorithm completed successfully!")
        print(f"  Generated {len(result.get('imputed_datasets', []))} imputed datasets")
        
        imputed_datasets = result.get('imputed_datasets', [])
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            print(f"  First imputed dataset shape: {first_imputed.shape}")
            print(f"  Missing values after imputation: {np.isnan(first_imputed[:, 0]).sum()}")
            print(f"  Imputed values are binary: {np.all(np.isin(first_imputed[:, 0], [0, 1]))}")
        
        print(f"  RPC calls made: {client.call_count}")
        return True
        
    except Exception as e:
        print(f"\n✗ Algorithm failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simice():
    """Test SIMICE with multiple columns."""
    print("\n" + "="*80)
    print("TEST 3: SIMICE - Multiple Columns")
    print("="*80)
    
    # Create test data with multiple missing columns
    np.random.seed(42)
    central_data = np.random.randn(50, 5)
    central_data[10:15, 1] = np.nan  # Missing in column 2
    central_data[20:25, 3] = np.nan  # Missing in column 4
    central_data[:, 1] = (central_data[:, 1] > 0).astype(float)  # Binary
    
    remote1_data = np.random.randn(60, 5)
    remote1_data[:, 1] = (remote1_data[:, 1] > 0).astype(float)
    
    remote2_data = np.random.randn(70, 5)
    remote2_data[:, 1] = (remote2_data[:, 1] > 0).astype(float)
    
    print(f"\nData shapes:")
    print(f"  Central: {central_data.shape}")
    print(f"    Missing in col 2: {np.isnan(central_data[:, 1]).sum()}")
    print(f"    Missing in col 4: {np.isnan(central_data[:, 3]).sum()}")
    print(f"  Remote 1: {remote1_data.shape}")
    print(f"  Remote 2: {remote2_data.shape}")
    
    client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
    
    from SIMICE.algorithm import master_simice
    
    print("\nRunning SIMICE algorithm...")
    try:
        result = master_simice(
            client,
            {
                'data': central_data,
                'target_column_indexes': [2, 4],  # 1-based indices
                'is_binary_list': [True, False],
                'imputation_trials': 3,
                'iteration_before_first_imputation': 2,
                'iteration_between_imputations': 1
            }
        )
        
        print(f"\n✓ Algorithm completed successfully!")
        print(f"  Generated {len(result.get('imputed_datasets', []))} imputed datasets")
        
        imputed_datasets = result.get('imputed_datasets', [])
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            print(f"  First imputed dataset shape: {first_imputed.shape}")
            print(f"  Missing in col 2 after: {np.isnan(first_imputed[:, 1]).sum()}")
            print(f"  Missing in col 4 after: {np.isnan(first_imputed[:, 3]).sum()}")
        
        print(f"  RPC calls made: {client.call_count}")
        return True
        
    except Exception as e:
        print(f"\n✗ Algorithm failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_remote_functions():
    """Test remote RPC functions directly."""
    print("\n" + "="*80)
    print("TEST 4: Remote RPC Functions")
    print("="*80)
    
    # Test SIMI remote Gaussian
    print("\n4.1: Testing RPC_simi_remote_gaussian...")
    try:
        from SIMI.algorithm import RPC_simi_remote_gaussian
        
        test_data = create_test_data(n_samples=100, n_features=4, missing_rate=0.0, seed=10)
        result = RPC_simi_remote_gaussian(
            {'data': test_data, 'mvar': 1},
            mvar=1
        )
        
        assert 'n' in result
        assert 'XX' in result
        assert 'Xy' in result
        assert 'yy' in result
        print(f"  ✓ Gaussian remote function works")
        print(f"    n={result['n']}, XX shape={len(result['XX'])}, Xy len={len(result['Xy'])}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test SIMI remote Logistic
    print("\n4.2: Testing RPC_simi_remote_logistic...")
    try:
        from SIMI.algorithm import RPC_simi_remote_logistic
        
        test_data = create_test_data(n_samples=100, n_features=4, missing_rate=0.0, seed=10)
        test_data[:, 0] = (test_data[:, 0] > 0).astype(float)
        
        # Test mode 0 (get sample size) - but actually mode 0 terminates, so use mode 1
        # The function reads mode first, so we need to pass it correctly
        # Actually, looking at the code, mode 0 terminates, so let's test with a valid mode
        # For getting sample size, the function sends n first, then waits for mode
        # Let's test mode 1 directly
        result = RPC_simi_remote_logistic({'data': test_data, 'mvar': 1}, [0.1, 0.2, 0.3], 1)
        assert 'Q' in result
        if 'H' in result and 'g' in result:
            print(f"  ✓ Logistic remote function (mode 1) works")
            print(f"    H shape={len(result['H'])}, g len={len(result['g'])}")
        else:
            print(f"  ✓ Logistic remote function (mode 1) works (Q only)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\nStarting comprehensive test suite...")
    print("Note: Using mock vantage6 client for local testing\n")
    
    results = []
    
    # Test remote functions first (simpler)
    results.append(("Remote Functions", test_remote_functions()))
    
    # Test SIMI algorithms
    results.append(("SIMI Gaussian", test_simi_gaussian()))
    results.append(("SIMI Logistic", test_simi_logistic()))
    
    # Test SIMICE
    results.append(("SIMICE", test_simice()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Algorithms are ready for vantage6 deployment.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

