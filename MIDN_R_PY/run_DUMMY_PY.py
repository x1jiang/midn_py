#!/usr/bin/env python
"""
Script to run DUMMY Federated Learning Central and Remote sites.

This script starts a central server and two remote clients for a simple
federated learning demonstration where random data is exchanged.
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd
import numpy as np
import multiprocessing
from typing import List, Dict

def start_central_server(port, expected_sites, num_iterations=1000):
    """Start the central server process"""
    from DUMMY.DUMMYCentral import app, set_expected_sites, dummy_federated_learning
    import uvicorn
    import asyncio
    
    # Set expected sites
    set_expected_sites(expected_sites)
    
    # Define startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        # Start a background task for the federated learning
        asyncio.create_task(run_federated_learning(expected_sites, num_iterations))
    
    async def run_federated_learning(site_ids, num_iterations):
        print(f"Waiting for all sites to connect: {site_ids}", flush=True)
        # Wait for all expected remote sites to connect using event from DUMMYCentral
        from DUMMY.DUMMYCentral import all_sites_connected
        # Clear the event first to ensure we're waiting for fresh connections
        all_sites_connected.clear()
        print("Waiting for all sites to establish connection...", flush=True)
        await all_sites_connected.wait()
        print("All sites connected. Starting federated learning...", flush=True)
        
        # Run federated learning
        total_sum = await dummy_federated_learning(site_ids, num_iterations)
        
        # Signal completion
        print(f"Federated learning complete. Final sum: {total_sum:.4f}", flush=True)
        print("Press Ctrl+C to exit.", flush=True)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)

def start_remote_site(port, central_host, central_port, site_id, remote_port=None):
    """Start a remote site process"""
    from DUMMY.DUMMYRemote import dummy_remote
    import numpy as np
    
    # Use provided remote_port if specified, otherwise use port
    if remote_port is None:
        remote_port = port
    
    # Create dummy data - not actually used in this implementation
    D = np.zeros((10, 10))
    
    # Run the remote site with site ID
    dummy_remote(D, remote_port, central_host, central_port, site_id)

def run_python_processes(args):
    """Run the Python processes for dummy federated learning"""
    # Setup fixed site IDs
    site_ids = ["remote1", "remote2"]
    remote_processes = []
    
    # Create remote processes with explicit site IDs
    print(f"Creating remote site remote1 on port {args.remote_ports[0]}", flush=True)
    remote1_proc = multiprocessing.Process(
        target=start_remote_site,
        args=(args.remote_ports[0], args.central_host, args.central_port, "remote1")
    )
    remote_processes.append(("remote1", remote1_proc))
    
    print(f"Creating remote site remote2 on port {args.remote_ports[1]}", flush=True)
    remote2_proc = multiprocessing.Process(
        target=start_remote_site,
        args=(args.remote_ports[1], args.central_host, args.central_port, "remote2")
    )
    remote_processes.append(("remote2", remote2_proc))
    
    # Start central process with the site IDs
    central_process = multiprocessing.Process(
        target=start_central_server,
        args=(args.central_port, site_ids, args.num_iterations)
    )
    # Start processes
    print("Starting remote processes...", flush=True)
    for site_id, proc in remote_processes:
        print(f"Starting {site_id} process...", flush=True)
        proc.start()
    # Give remote sites time to start up
    time.sleep(2)
    print("Starting central process...", flush=True)
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
    parser = argparse.ArgumentParser(description="Run DUMMY Federated Learning in Python")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of federated learning iterations")
    parser.add_argument("--central_host", default="localhost", help="Central server hostname")
    parser.add_argument("--central_port", type=int, default=8000, help="Central server port")
    parser.add_argument("--remote_ports", type=int, nargs="+", default=[8001, 8002], 
                        help="Remote site ports (default: [8001, 8002])")
    
    args = parser.parse_args()
    
    # Run Python implementation
    print("Running Dummy Federated Learning...", flush=True)
    run_python_processes(args)
    
    print("Federated Learning complete.", flush=True)

if __name__ == "__main__":
    main()
