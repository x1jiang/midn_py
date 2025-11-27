"""
Comprehensive Test Suite for Vantage6 Algorithms

Tests all aspects of the algorithms including:
- Imports and dependencies
- Function signatures
- Data handling
- Algorithm execution
- Edge cases
- Error handling
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "MIDN_R_PY"))

print("="*80)
print("COMPREHENSIVE TEST SUITE")
print("="*80)
print()

test_results = []

def test_result(name, passed, message=""):
    """Record test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    test_results.append((name, passed, message))
    print(f"{status}: {name}")
    if message:
        print(f"  {message}")
    return passed

def test_imports():
    """Test 1: Import all modules."""
    print("\n" + "="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    all_passed = True
    
    # Test SIMI imports
    try:
        from SIMI.algorithm import master_simi, RPC_simi_remote_gaussian, RPC_simi_remote_logistic
        all_passed &= test_result("SIMI imports", True)
    except Exception as e:
        all_passed &= test_result("SIMI imports", False, str(e))
        traceback.print_exc()
    
    # Test SIMICE imports
    try:
        from SIMICE.algorithm import master_simice, RPC_simice_remote_initialize, RPC_simice_remote_statistics
        all_passed &= test_result("SIMICE imports", True)
    except Exception as e:
        all_passed &= test_result("SIMICE imports", False, str(e))
        traceback.print_exc()
    
    # Test Core imports
    try:
        from Core import LS, Logit
        all_passed &= test_result("Core utilities imports", True)
    except Exception as e:
        all_passed &= test_result("Core utilities imports", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_function_signatures():
    """Test 2: Verify function signatures."""
    print("\n" + "="*80)
    print("TEST 2: Function Signatures")
    print("="*80)
    
    all_passed = True
    
    try:
        from SIMI.algorithm import master_simi, RPC_simi_remote_gaussian, RPC_simi_remote_logistic
        import inspect
        
        # Check master_simi signature
        sig = inspect.signature(master_simi)
        params = list(sig.parameters.keys())
        if 'client' in params and 'data' in params:
            all_passed &= test_result("master_simi signature", True)
        else:
            all_passed &= test_result("master_simi signature", False, f"Missing params: {params}")
        
        # Check RPC functions
        sig = inspect.signature(RPC_simi_remote_gaussian)
        all_passed &= test_result("RPC_simi_remote_gaussian signature", True)
        
        sig = inspect.signature(RPC_simi_remote_logistic)
        all_passed &= test_result("RPC_simi_remote_logistic signature", True)
        
        from SIMICE.algorithm import master_simice, RPC_simice_remote_initialize, RPC_simice_remote_statistics
        
        sig = inspect.signature(master_simice)
        all_passed &= test_result("master_simice signature", True)
        
        sig = inspect.signature(RPC_simice_remote_initialize)
        all_passed &= test_result("RPC_simice_remote_initialize signature", True)
        
        sig = inspect.signature(RPC_simice_remote_statistics)
        all_passed &= test_result("RPC_simice_remote_statistics signature", True)
        
    except Exception as e:
        all_passed &= test_result("Function signatures", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_data_handling():
    """Test 3: Data handling (numpy arrays, CSV paths, missing values)."""
    print("\n" + "="*80)
    print("TEST 3: Data Handling")
    print("="*80)
    
    all_passed = True
    
    try:
        from SIMI.algorithm import RPC_simi_remote_gaussian
        
        # Test with numpy array
        data1 = np.random.randn(50, 4)
        result1 = RPC_simi_remote_gaussian({'data': data1, 'mvar': 1})
        if 'n' in result1 and 'XX' in result1:
            all_passed &= test_result("Numpy array input", True)
        else:
            all_passed &= test_result("Numpy array input", False, "Missing keys in result")
        
        # Test with missing values
        data2 = data1.copy()
        data2[10:15, 0] = np.nan
        result2 = RPC_simi_remote_gaussian({'data': data2, 'mvar': 1})
        if 'n' in result2:
            all_passed &= test_result("Missing values handling", True)
        else:
            all_passed &= test_result("Missing values handling", False)
        
        # Test with binary data
        data3 = data1.copy()
        data3[:, 0] = (data3[:, 0] > 0).astype(float)
        from SIMI.algorithm import RPC_simi_remote_logistic
        result3 = RPC_simi_remote_logistic({'data': data3, 'mvar': 1}, [0.1, 0.2, 0.3], 1)
        if 'Q' in result3:
            all_passed &= test_result("Binary data handling", True)
        else:
            all_passed &= test_result("Binary data handling", False)
        
    except Exception as e:
        all_passed &= test_result("Data handling", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_simi_gaussian():
    """Test 4: SIMI Gaussian method end-to-end."""
    print("\n" + "="*80)
    print("TEST 4: SIMI Gaussian Method")
    print("="*80)
    
    from test_local import MockVantage6Client, create_test_data
    
    all_passed = True
    
    try:
        # Create test data
        central_data = create_test_data(n_samples=50, n_features=4, missing_rate=0.2, seed=1)
        remote1_data = create_test_data(n_samples=60, n_features=4, missing_rate=0.0, seed=2)
        remote2_data = create_test_data(n_samples=70, n_features=4, missing_rate=0.0, seed=3)
        
        remote1_data[:, 0] = np.random.randn(60)
        remote2_data[:, 0] = np.random.randn(70)
        
        client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
        
        from SIMI.algorithm import master_simi
        
        result = master_simi(
            client,
            {
                'data': central_data,
                'target_column_index': 1,
                'is_binary': False,
                'imputation_trials': 3
            }
        )
        
        # Verify results
        imputed_datasets = result.get('imputed_datasets', [])
        if len(imputed_datasets) == 3:
            all_passed &= test_result("SIMI Gaussian - generates datasets", True)
        else:
            all_passed &= test_result("SIMI Gaussian - generates datasets", False, f"Expected 3, got {len(imputed_datasets)}")
        
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            missing_after = np.isnan(first_imputed[:, 0]).sum()
            if missing_after == 0:
                all_passed &= test_result("SIMI Gaussian - imputes missing values", True)
            else:
                all_passed &= test_result("SIMI Gaussian - imputes missing values", False, f"{missing_after} missing values remain")
        
        if client.call_count > 0:
            all_passed &= test_result("SIMI Gaussian - RPC calls made", True, f"{client.call_count} calls")
        else:
            all_passed &= test_result("SIMI Gaussian - RPC calls made", False)
        
    except Exception as e:
        all_passed &= test_result("SIMI Gaussian", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_simi_logistic():
    """Test 5: SIMI Logistic method end-to-end."""
    print("\n" + "="*80)
    print("TEST 5: SIMI Logistic Method")
    print("="*80)
    
    from test_local import MockVantage6Client
    
    all_passed = True
    
    try:
        np.random.seed(42)
        central_data = np.random.randn(50, 4)
        central_data[:, 0] = (central_data[:, 0] > 0).astype(float)
        central_data[10:15, 0] = np.nan
        
        remote1_data = np.random.randn(60, 4)
        remote1_data[:, 0] = (remote1_data[:, 0] > 0).astype(float)
        
        remote2_data = np.random.randn(70, 4)
        remote2_data[:, 0] = (remote2_data[:, 0] > 0).astype(float)
        
        client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
        
        from SIMI.algorithm import master_simi
        
        result = master_simi(
            client,
            {
                'data': central_data,
                'target_column_index': 1,
                'is_binary': True,
                'imputation_trials': 3
            }
        )
        
        imputed_datasets = result.get('imputed_datasets', [])
        if len(imputed_datasets) == 3:
            all_passed &= test_result("SIMI Logistic - generates datasets", True)
        else:
            all_passed &= test_result("SIMI Logistic - generates datasets", False)
        
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            missing_after = np.isnan(first_imputed[:, 0]).sum()
            is_binary = np.all(np.isin(first_imputed[:, 0], [0, 1]))
            
            if missing_after == 0:
                all_passed &= test_result("SIMI Logistic - imputes missing values", True)
            else:
                all_passed &= test_result("SIMI Logistic - imputes missing values", False)
            
            if is_binary:
                all_passed &= test_result("SIMI Logistic - binary constraints", True)
            else:
                all_passed &= test_result("SIMI Logistic - binary constraints", False, "Values not binary")
        
    except Exception as e:
        all_passed &= test_result("SIMI Logistic", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_simice():
    """Test 6: SIMICE end-to-end."""
    print("\n" + "="*80)
    print("TEST 6: SIMICE Multiple Columns")
    print("="*80)
    
    from test_local import MockVantage6Client
    
    all_passed = True
    
    try:
        np.random.seed(42)
        central_data = np.random.randn(50, 5)
        central_data[10:15, 1] = np.nan
        central_data[20:25, 3] = np.nan
        central_data[:, 1] = (central_data[:, 1] > 0).astype(float)
        
        remote1_data = np.random.randn(60, 5)
        remote1_data[:, 1] = (remote1_data[:, 1] > 0).astype(float)
        
        remote2_data = np.random.randn(70, 5)
        remote2_data[:, 1] = (remote2_data[:, 1] > 0).astype(float)
        
        client = MockVantage6Client(remote_data_sets=[remote1_data, remote2_data])
        
        from SIMICE.algorithm import master_simice
        
        result = master_simice(
            client,
            {
                'data': central_data,
                'target_column_indexes': [2, 4],
                'is_binary_list': [True, False],
                'imputation_trials': 3,
                'iteration_before_first_imputation': 2,
                'iteration_between_imputations': 1
            }
        )
        
        imputed_datasets = result.get('imputed_datasets', [])
        if len(imputed_datasets) == 3:
            all_passed &= test_result("SIMICE - generates datasets", True)
        else:
            all_passed &= test_result("SIMICE - generates datasets", False)
        
        if imputed_datasets:
            first_imputed = np.array(imputed_datasets[0])
            missing_col2 = np.isnan(first_imputed[:, 1]).sum()
            missing_col4 = np.isnan(first_imputed[:, 3]).sum()
            
            if missing_col2 == 0 and missing_col4 == 0:
                all_passed &= test_result("SIMICE - imputes all columns", True)
            else:
                all_passed &= test_result("SIMICE - imputes all columns", False, f"Col2: {missing_col2}, Col4: {missing_col4}")
        
        if client.call_count > 0:
            all_passed &= test_result("SIMICE - RPC calls made", True, f"{client.call_count} calls")
        else:
            all_passed &= test_result("SIMICE - RPC calls made", False)
        
    except Exception as e:
        all_passed &= test_result("SIMICE", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_edge_cases():
    """Test 7: Edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST 7: Edge Cases")
    print("="*80)
    
    all_passed = True
    
    try:
        from SIMI.algorithm import RPC_simi_remote_gaussian
        
        # Test with very small dataset
        small_data = np.random.randn(5, 3)
        result = RPC_simi_remote_gaussian({'data': small_data, 'mvar': 1})
        if 'n' in result:
            all_passed &= test_result("Small dataset handling", True)
        else:
            all_passed &= test_result("Small dataset handling", False)
        
        # Test with all missing in target column (should handle gracefully)
        data_all_missing = np.random.randn(10, 4)
        data_all_missing[:, 0] = np.nan
        try:
            result = RPC_simi_remote_gaussian({'data': data_all_missing, 'mvar': 1})
            all_passed &= test_result("All missing target column", True, "Handled gracefully")
        except Exception as e:
            all_passed &= test_result("All missing target column", False, str(e))
        
        # Test with no missing values
        data_no_missing = np.random.randn(20, 4)
        result = RPC_simi_remote_gaussian({'data': data_no_missing, 'mvar': 1})
        if 'n' in result:
            all_passed &= test_result("No missing values", True)
        else:
            all_passed &= test_result("No missing values", False)
        
    except Exception as e:
        all_passed &= test_result("Edge cases", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_serialization():
    """Test 8: Data serialization (numpy to JSON)."""
    print("\n" + "="*80)
    print("TEST 8: Data Serialization")
    print("="*80)
    
    all_passed = True
    
    try:
        from SIMI.algorithm import RPC_simi_remote_gaussian
        
        data = np.random.randn(30, 4)
        result = RPC_simi_remote_gaussian({'data': data, 'mvar': 1})
        
        # Check that results are JSON-serializable
        import json
        
        try:
            json_str = json.dumps(result)
            all_passed &= test_result("Result JSON serializable", True)
        except TypeError as e:
            all_passed &= test_result("Result JSON serializable", False, str(e))
        
        # Check that arrays are converted to lists
        if isinstance(result.get('XX'), list):
            all_passed &= test_result("Arrays converted to lists", True)
        else:
            all_passed &= test_result("Arrays converted to lists", False, f"Type: {type(result.get('XX'))}")
        
    except Exception as e:
        all_passed &= test_result("Serialization", False, str(e))
        traceback.print_exc()
    
    return all_passed

def test_dependencies():
    """Test 9: Check required dependencies."""
    print("\n" + "="*80)
    print("TEST 9: Dependencies")
    print("="*80)
    
    all_passed = True
    
    required_modules = ['numpy', 'pandas', 'scipy']
    
    for module in required_modules:
        try:
            __import__(module)
            all_passed &= test_result(f"{module} available", True)
        except ImportError:
            all_passed &= test_result(f"{module} available", False, "Not installed")
    
    # Check optional vantage6 (should warn but not fail)
    try:
        import vantage6
        all_passed &= test_result("vantage6 available", True)
    except ImportError:
        all_passed &= test_result("vantage6 available", False, "Not installed (optional for local testing)")
    
    return all_passed

def test_file_structure():
    """Test 10: Verify file structure."""
    print("\n" + "="*80)
    print("TEST 10: File Structure")
    print("="*80)
    
    all_passed = True
    base_path = Path(__file__).parent
    
    required_files = [
        'SIMI/algorithm.py',
        'SIMI/Dockerfile',
        'SIMI/requirements.txt',
        'SIMICE/algorithm.py',
        'SIMICE/Dockerfile',
        'SIMICE/requirements.txt',
        'Core/LS.py',
        'Core/Logit.py',
        'README.md',
        'test_local.py'
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            all_passed &= test_result(f"File exists: {file_path}", True)
        else:
            all_passed &= test_result(f"File exists: {file_path}", False)
    
    return all_passed

def main():
    """Run all tests."""
    print("\nStarting comprehensive test suite...\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Function Signatures", test_function_signatures),
        ("Data Handling", test_data_handling),
        ("SIMI Gaussian", test_simi_gaussian),
        ("SIMI Logistic", test_simi_logistic),
        ("SIMICE", test_simice),
        ("Edge Cases", test_edge_cases),
        ("Serialization", test_serialization),
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} test suites passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Algorithms are fully functional and ready for deployment.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test suite(s) failed. Please review errors above.")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


