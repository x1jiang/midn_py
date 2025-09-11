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
import asyncio
from fastapi import FastAPI, WebSocket
from typing import List, Dict, Optional, Set

# Global variables for WebSocket connections
app = FastAPI()
remote_websockets: Dict[str, WebSocket] = {}  # Keys are site_id, values are WebSocket connections - presence in this dict indicates connection
site_locks: Dict[str, asyncio.Lock] = {}
expected_sites: Optional[Set[str]] = None
all_sites_connected = asyncio.Event()
imputation_running = asyncio.Event()

# Helper functions for WebSocket handling
def set_expected_sites(sites: list):
    """Set the expected remote sites"""
    global expected_sites
    expected_sites = set(sites)
    print(f"Set expected sites: {expected_sites}", flush=True)

async def maintain_heartbeat(websockets, check_interval=30, ping_message="ping", 
                           expected_response="pong", timeout=10.0, 
                           locks=None, active_flag=None):
    """Maintain heartbeat with connected clients"""
    while True:
        await asyncio.sleep(check_interval)
        # Do not run heartbeat while a job is running to avoid recv/send conflicts
        if active_flag and active_flag.is_set():
            # Keep connections idle during active imputation
            continue

        print(f"Checking heartbeat for {len(websockets)} sites", flush=True)
        
        for site_id, websocket in list(websockets.items()):
            if locks and site_id in locks:
                # Skip heartbeat if a site is currently processing
                if locks[site_id].locked():
                    print(f"Site {site_id} is busy, skipping heartbeat", flush=True)
                    continue
                
                # Acquire lock for heartbeat
                async with locks[site_id]:
                    try:
                        print(f"Sending heartbeat to {site_id}", flush=True)
                        await websocket.send_text(ping_message)
                        
                        # Wait for response with timeout
                        try:
                            response = await asyncio.wait_for(websocket.receive_text(), timeout)
                            if response != expected_response:
                                print(f"Unexpected heartbeat response from {site_id}: {response}", flush=True)
                        except asyncio.TimeoutError:
                            print(f"Heartbeat timeout for site {site_id}", flush=True)
                            # Remove site from connected sites
                            if site_id in remote_websockets:
                                del remote_websockets[site_id]
                            
                    except Exception as e:
                        print(f"Error during heartbeat with {site_id}: {str(e)}", flush=True)
                        # Remove site from connected sites
                        if site_id in remote_websockets:
                            del remote_websockets[site_id]

@app.websocket("/ws/{site_id}")
async def websocket_endpoint(websocket: WebSocket, site_id: str):
    # Create a lock for this site if it doesn't exist
    if site_id not in site_locks:
        site_locks[site_id] = asyncio.Lock()
    
    await websocket.accept()
    
    # Store the WebSocket connection (this also indicates the site is connected)
    remote_websockets[site_id] = websocket
    
    print(f"Remote site {site_id} connected", flush=True)
    
    # Check if all expected sites are connected
    if expected_sites and expected_sites.issubset(set(remote_websockets.keys())):
        all_sites_connected.set()
        # Start the heartbeat task when all sites are connected
        heartbeat_task = asyncio.create_task(
            maintain_heartbeat(
                remote_websockets,
                check_interval=30,  # Check every 30 seconds
                ping_message="ping",
                expected_response="pong",
                timeout=10.0,
                locks=site_locks,  # Pass the locks to prevent conflicts
                active_flag=imputation_running  # Pass the active operations flag
            )
        )
        print(f"Started heartbeat monitoring for all connected sites", flush=True)
        print(f"All expected sites are now connected: {', '.join(expected_sites)}", flush=True)
    
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        print(f"WebSocket {site_id} cancelled during shutdown", flush=True)
    except Exception as e:
        print(f"WebSocket connection with {site_id} closed: {str(e)}", flush=True)
    finally:
        if site_id in remote_websockets:
            del remote_websockets[site_id]

# These functions have been combined into start_central_site

def start_remote_site(data_file, algorithm, config, central_host, central_port, site_id):
    """Start a remote site as a pure outbound WebSocket client (no local port)."""
    D = pd.read_csv(data_file).values
    print(f"Remote {site_id} data shape: {D.shape}", flush=True)
    print(f"Remote {site_id} data contains NaN: {np.isnan(D).any()}", flush=True)
    if algorithm.lower() == "simi":
        from SIMI.SIMIRemote import run_remote_client
    elif algorithm.lower() == "simice":
        from SIMICE.SIMICERemote import run_remote_client
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    run_remote_client(D, central_host, central_port, site_id, None, config)

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
        
    Note:
    -----
    This function passes the remote_websockets dictionary to the algorithm_central function,
    so that the algorithm can use the existing WebSocket connections instead of creating new ones.
    """
    
    algorithm = algorithm.lower()
    if algorithm == "simi":
        from SIMI.SIMICentral import simi_central as algorithm_central
    else:  # algorithm == "simice"
        from SIMICE.SIMICECentral import simice_central as algorithm_central
    
    # Clear the event first to ensure we're waiting for fresh connections
    all_sites_connected.clear()
    
    # Set the imputation running flag
    imputation_running.set()
    
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
        # Pass remote_websockets to the algorithm so it can use the existing WebSocket connections
        imputed_data = await algorithm_central(D=D, config=config, site_ids=site_ids, websockets=remote_websockets)
        # Process and save imputed data - unified approach
        # Prepare output directory and file naming pattern based on algorithm
        if algorithm == "simi":
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            file_pattern = lambda i: f"{os.path.splitext(output_path)[0]}_{i+1:02d}.csv"
        else:
            os.makedirs(output_path, exist_ok=True)
            file_pattern = lambda i: f"{output_path}/central_imp_{i+1:02d}.csv"
        binary_cols = []
        if algorithm == "simi" and config.get("method", "").lower() == "logistic":
            mvar = config["mvar"]
            binary_cols.append(mvar)
        elif algorithm == "simice":
            for var, t in zip(config.get("mvar", []), config.get("type_list", [])):
                if t.lower() == "logistic":
                    binary_cols.append(var)
        print(f"Binary columns identified from parameters: {binary_cols}", flush=True)
        for i, D_imputed in enumerate(imputed_data):
            df_imputed = pd.DataFrame(D_imputed)
            for col in binary_cols:
                if isinstance(col, int) and 0 <= col < df_imputed.shape[1]:
                    df_imputed.iloc[:, col] = np.round(df_imputed.iloc[:, col]).clip(0, 1).astype(int)
            csv_filename = file_pattern(i)
            df_imputed.to_csv(csv_filename, index=False)
            print(f"Saved imputed dataset {i+1}/{len(imputed_data)} to {csv_filename}", flush=True)
        print(f"Saving results to {output_path}", flush=True)
        print("Imputation complete.", flush=True)
        return imputed_data
        
    except Exception as e:
        print(f"Error in imputation: {str(e)}", flush=True)
        raise
    finally:
        # Clear the imputation running flag when done
        imputation_running.clear()

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
        from SIMI.SIMICentral import simi_central
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
            # Pass algorithm as a separate parameter and output path
            # Also pass remote_websockets to avoid creating new connections
            asyncio.create_task(run_imputation(D, algorithm, imputation_config, expected_sites, output))
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    elif algorithm == "simice":
        # Start SIMICE central process
        from SIMICE.SIMICECentral import simice_central
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
            # Pass algorithm as a separate parameter and output path
            asyncio.create_task(run_imputation(D, algorithm, imputation_config, expected_sites, output))
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=port)
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def run_imputation_processes(args):
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    site_ids = ["remote1", "remote2"]
    remote_processes = []
    for i, remote_data in enumerate(args.remote_data[:2]):
        site_id = site_ids[i]
        print(f"Creating {args.algorithm} remote site {site_id}", flush=True)
        proc = multiprocessing.Process(target=start_remote_site, args=(remote_data, args.algorithm, config, args.central_host, args.central_port, site_id))
        remote_processes.append((site_id, proc))
    output_path = args.output_dir if args.algorithm.lower() == "simice" and args.output_dir else args.output
    print("Starting remote processes...", flush=True)
    for sid, proc in remote_processes:
        print(f"Starting {sid} process...", flush=True)
        proc.start()
    time.sleep(1)
    print(f"Starting {args.algorithm} central process...", flush=True)
    try:
        start_central_site(args.algorithm, args.central_port, site_ids, args.central_data, config, output_path)
    finally:
        print("Terminating remote processes...", flush=True)
        for _, proc in remote_processes:
            if proc.is_alive():
                proc.terminate()
        for _, proc in remote_processes:
            proc.join()

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
    # remote ports removed (remote remotes use outbound connections only)
    
    args = parser.parse_args()
    
    # Run the imputation processes
    print(f"Running {args.algorithm} imputation...", flush=True)
    print(f"Using config file: {args.config_file}", flush=True)
    print(f"runtime args: {args}", flush=True)
    run_imputation_processes(args)
    
    print("Imputation complete. Results saved.", flush=True)

if __name__ == "__main__":
    main()
