#!/usr/bin/env python
"""
Unified script to run SIMI or SIMICE imputation algorithms.

This script can start either SIMI or SIMICE with central server and remote clients
based on a specified algorithm parameter. Algorithm-specific parameters are provided
through a JSON configuration file.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
import multiprocessing
from typing import List, Dict, Any, Union, Optional

# These functions have been combined into start_central_site

def start_remote_site(data_file, algorithm, config, central_host, central_port, site_id, remote_port):
    """Start a remote site process (SIMI or SIMICE)
    
    Parameters:
    -----------
    data_file : str
        Path to the data file
    algorithm : str
        Algorithm to use ('SIMI' or 'SIMICE')
    config : dict
        Configuration parameters for the algorithm
    central_host : str
        Hostname of the central server
    central_port : int
        Port of the central server
    site_id : str
        ID of the remote site
    remote_port : int
        Port for the remote site
    """
    # Load data - same for both algorithms
    D = pd.read_csv(data_file).values
    
    # Debug the data
    print(f"Remote {site_id} data shape: {D.shape}", flush=True)
    print(f"Remote {site_id} data contains NaN: {np.isnan(D).any()}", flush=True)
    
    # Import the appropriate module based on algorithm
    if algorithm.lower() == "simi":
        from SIMI.SIMIRemote import run_remote_client
    elif algorithm.lower() == "simice":
        from SIMICE.SIMICERemote import run_remote_client
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Run the remote client with unified interface
    # Both implementations now have identical signatures and handle their own conversions
    run_remote_client(D, central_host, central_port, site_id, remote_port, config)

# Import necessary modules for run_imputation
import os
import numpy as np
import pandas as pd
import asyncio

# Define the run_imputation function at module level so it can be accessed by both algorithms
async def run_imputation(D, algorithm, config, site_ids, output_path=None):
    """Unified imputation function for both SIMI and SIMICE
    
    Parameters:
    -----------
    D : numpy.ndarray
        Data matrix
    algorithm : str
        Algorithm to use ('simi' or 'simice')
    config : dict
        Configuration parameters for the imputation algorithm
        Required keys depend on algorithm:
        - "M": Number of imputations (required)
        - "mvar": Index or list of indices of missing variables (required)
        - For SIMI: "method" ("Gaussian" or "logistic")
        - For SIMICE: "type_list", "iter_val", "iter0_val"
        - "output_path": Output file or directory path (optional)
    site_ids : list
        List of remote site IDs
    output_path : str, optional
        Output file or directory path, overrides config["output_path"] if provided
    """
    
    algorithm = algorithm.lower()
    if algorithm == "simi":
        from SIMI.SIMICentral import all_sites_connected
        from SIMI.SIMICentral import simi_central as algorithm_central
    else:  # algorithm == "simice"
        from SIMICE.SIMICECentral import all_sites_connected
        from SIMICE.SIMICECentral import simice_central as algorithm_central
    
    # Clear the event first to ensure we're waiting for fresh connections
    all_sites_connected.clear()
    
    # Use output_path from parameter or config
    output_path = output_path or config.get("output_path")
    if not output_path:
        raise ValueError("Output path not provided")
    
    print("Waiting for all sites to establish connection...", flush=True)
    await all_sites_connected.wait()
    print("All sites connected. Starting imputation...", flush=True)
    print(f"Using site IDs: {site_ids}", flush=True)
    
    try:
        # Use the algorithm_central function which is assigned based on the algorithm type
        imputed_data = await algorithm_central(D=D, config=config, site_ids=site_ids)
        # Process and save imputed data - unified approach
        # Prepare output directory and file naming pattern based on algorithm
        if algorithm == "simi":
            # For SIMI, output might be a file path
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # Define output file naming pattern
            file_pattern = lambda i: f"{os.path.splitext(output_path)[0]}_{i+1:02d}.csv"
        else:  # algorithm == "simice"
            # For SIMICE, output is a directory
            os.makedirs(output_path, exist_ok=True)
            # Define output file naming pattern
            file_pattern = lambda i: f"{output_path}/central_imp_{i+1:02d}.csv"
        # Identify binary columns based only on algorithm configuration
        binary_cols = []
        
        if algorithm == "simi":
            mvar = config["mvar"]
            if config.get("method", "").lower() == "logistic":
                binary_cols.append(mvar)
        else:  # algorithm == "simice"
            mvar_list = config["mvar"]
            type_list = config.get("type_list", [])
            for j, j_type in zip(mvar_list, type_list):
                if j_type.lower() == "logistic":
                    binary_cols.append(j)
        
        print(f"Binary columns identified from parameters: {binary_cols}", flush=True)
        
        # Save imputed datasets
        for i, D_imputed in enumerate(imputed_data):
            # Create a DataFrame from the imputed data
            df_imputed = pd.DataFrame(D_imputed)
            
            # Process binary columns (ensure 0/1 values)
            for col in binary_cols:
                print(f"Enforcing binary values for column index {col}", flush=True)
                if isinstance(col, int):  # Direct column index
                    df_imputed.iloc[:, col] = np.round(df_imputed.iloc[:, col]).clip(0, 1).astype(int)
                else:  # Column name
                    df_imputed[col] = np.round(df_imputed[col]).clip(0, 1).astype(int)
            
            # Create numbered output CSV file
            csv_filename = file_pattern(i)
            
            # Save as CSV 
            df_imputed.to_csv(csv_filename, index=False)
            print(f"Saved imputed dataset {i+1}/{len(imputed_data)} to {csv_filename}", flush=True)
        
        print(f"Saving results to {output_path}", flush=True)
        
        # Signal completion
        print("Imputation complete. Press Ctrl+C to exit.", flush=True)
        
        return imputed_data
        
    except Exception as e:
        print(f"Error in imputation: {str(e)}", flush=True)
        raise

def start_central_site(algorithm, port, expected_sites, data_file, config, output):
    """Start a central server process (SIMI or SIMICE)
    
    Parameters:
    -----------
    algorithm : str
        Algorithm to use ('SIMI' or 'SIMICE')
    port : int
        Port for the central server
    expected_sites : list
        List of expected remote site IDs
    data_file : str
        Path to the central data file
    config : dict
        Configuration parameters for the algorithm
    output : str
        Output file or directory for results
    """
    algorithm = algorithm.lower()
    
    if algorithm == "simi":
        # Start SIMI central process
        from SIMI.SIMICentral import app, set_expected_sites, simi_central
        import uvicorn
        import asyncio
        import pandas as pd
        
        # Extract config parameters
        mvar = config["mvar"]
        method = config["method"]
        M = config["m"]
        
        # Load data
        D = pd.read_csv(data_file).values
        
        # Set expected sites
        set_expected_sites(expected_sites)
        
        # Convert mvar from 1-based (R style) to 0-based (Python style)
        mvar_py = mvar - 1
        
        # Define startup event
        @app.on_event("startup")
        async def startup_event():
            # Start a background task for the imputation
            imputation_config = {
                "M": M,
                "mvar": mvar_py,
                "method": method,
                "output_path": output
            }
            # Pass algorithm as a separate parameter
            asyncio.create_task(run_imputation(D, algorithm, imputation_config, expected_sites))
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    elif algorithm == "simice":
        # Start SIMICE central process
        from SIMICE.SIMICECentral import app, set_expected_sites, simice_central
        import uvicorn
        import asyncio
        import pandas as pd
        import numpy as np
        
        # Extract config parameters
        mvar_list = config["mvar"]
        type_list = config["type_list"]
        M = config["m"]
        iter_val = config["iter"]
        iter0_val = config["iter0"]
        
        # Load data
        D = pd.read_csv(data_file).values
        
        # Debug the data to check for issues
        print(f"Data shape: {D.shape}", flush=True)
        print(f"Data contains NaN: {np.isnan(D).any()}", flush=True)
        
        # Set expected sites
        set_expected_sites(expected_sites)
        
        # Convert mvar from 1-based (R style) to 0-based (Python style)
        mvar_py = [m - 1 for m in mvar_list]
        
        # Define startup event
        @app.on_event("startup")
        async def startup_event():
            # Start a background task for the imputation
            imputation_config = {
                "M": M,
                "mvar": mvar_py,
                "type_list": type_list,
                "iter_val": iter_val,
                "iter0_val": iter0_val,
                "output_path": output
            }
            # Pass algorithm as a separate parameter
            asyncio.create_task(run_imputation(D, algorithm, imputation_config, expected_sites))
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=port)
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def run_imputation_processes(args):
    """Run the central and remote processes based on algorithm selection"""
    # Load algorithm-specific config
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Setup site IDs
    site_ids = ["remote1", "remote2"]
    remote_processes = []
    
    # Create remote processes with explicit site IDs using the unified start_remote_site function
    for i, remote_data in enumerate(args.remote_data[:2]):  # Limit to first two remote data files
        site_id = f"remote{i+1}"
        port = args.remote_ports[i] if i < len(args.remote_ports) else 8001 + i
        print(f"Creating {args.algorithm} remote site {site_id} on port {port}", flush=True)
        
        # All remote processes use the same unified interface regardless of algorithm
        remote_proc = multiprocessing.Process(
            target=start_remote_site,
            args=(remote_data, args.algorithm, config, args.central_host, args.central_port, site_id, port)
        )
        remote_processes.append((site_id, remote_proc))
    
    # Prepare output path/directory based on algorithm
    output_path = args.output_dir if args.algorithm.lower() == "simice" and args.output_dir else args.output
    
    # Create central process using the unified interface
    central_process = multiprocessing.Process(
        target=start_central_site,
        args=(args.algorithm, args.central_port, site_ids, args.central_data, config, output_path)
    )
    
    # Start processes
    print("Starting remote processes...", flush=True)
    for site_id, proc in remote_processes:
        print(f"Starting {site_id} process...", flush=True)
        proc.start()
    # Give remote sites time to start up
    time.sleep(2)
    print(f"Starting {args.algorithm} central process...", flush=True)
    central_process.start()
    
    # Wait for processes to finish
    try:
        central_process.join()
        for _, proc in remote_processes:
            proc.join()
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
        central_process.terminate()
        for _, proc in remote_processes:
            proc.terminate()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run imputation algorithms (SIMI or SIMICE)")
    parser.add_argument("--algorithm", required=True, choices=["SIMI", "SIMICE"], 
                        help="Imputation algorithm to use")
    parser.add_argument("--config_file", required=True, 
                        help="JSON configuration file with algorithm-specific parameters")
    parser.add_argument("--central_data", required=True, 
                        help="Path to central data file")
    parser.add_argument("--remote_data", required=True, nargs="+", 
                        help="Paths to remote site data files")
    parser.add_argument("--output", required=True, 
                        help="Output file or directory for results")
    parser.add_argument("--output_dir", 
                        help="Output directory for SIMICE results (for backward compatibility)")
    parser.add_argument("--central_host", default="localhost", 
                        help="Central server hostname")
    parser.add_argument("--central_port", type=int, default=8000, 
                        help="Central server port")
    parser.add_argument("--remote_ports", type=int, nargs="+", default=[8001, 8002], 
                        help="Remote site ports")
    
    args = parser.parse_args()
    
    # Run the imputation processes
    print(f"Running {args.algorithm} imputation...", flush=True)
    print(f"Using config file: {args.config_file}", flush=True)
    print(f"runtime args: {args}", flush=True)
    run_imputation_processes(args)
    
    print("Imputation complete. Results saved.", flush=True)

if __name__ == "__main__":
    main()
