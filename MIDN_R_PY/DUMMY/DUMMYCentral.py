"""
Python implementation of Dummy Federated Learning Central site

A simple implementation of a federated learning system where the central server 
requests random data from remote sites and aggregates it.
"""

import numpy as np
import asyncio
import time
import os
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict, Tuple, Optional, Any
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector, 
    write_string, read_string, write_integer, read_integer,
    WebSocketWrapper, get_wrapped_websocket
)
from Core.heartbeat import maintain_heartbeat

# Set to True to enable detailed debug information
print_debug_info = True

# Deterministic RNG: global seed from env or random
# If GLOBAL_SEED is not provided, use a random seed
seed_env = os.getenv("GLOBAL_SEED")
if seed_env is not None:
    GLOBAL_SEED = int(seed_env)
else:
    # Use current time for random seed
    GLOBAL_SEED = random.randint(0, 2**32-1)
    print(f"No GLOBAL_SEED provided, using random seed: {GLOBAL_SEED}", flush=True)

def _seed_combine(*xs: int) -> int:
    """Combine multiple seeds into one"""
    s = 0
    for x in xs:
        s = (s * 1315423911 + int(x)) & 0xFFFFFFFF
    return s

# Track iteration counts for debugging
debug_counters = {
    "iterations_completed": 0,
    "random_data_received": 0,
    "total_sum": 0.0
}

app = FastAPI()

# Keep track of connected clients
connected_remote_sites = {}
# Dict to map remote site ID to websocket

# Dictionary to store WebSocket connections for remote sites
remote_websockets: Dict[str, WebSocket] = {}

def set_expected_sites(site_ids, debug=None):
    """Set the expected remote sites and optionally enable debug mode
    
    Args:
        site_ids: List of site IDs to expect
        debug: Enable debug mode (default: None, uses global print_debug_info)
    """
    global expected_sites, print_debug_info
    if debug is not None:
        print_debug_info = debug
        
    expected_sites = set(site_ids)
    print(f"Set expected sites: {expected_sites}", flush=True)
    
    if print_debug_info:
        print(f"[CENTRAL] Debug mode ENABLED - detailed logging will be shown", flush=True)

# Event to signal all expected remote sites have connected
all_sites_connected = asyncio.Event()
expected_sites = set()

# Flag to indicate when federated learning is running (to avoid concurrent WebSocket access)
imputation_running = asyncio.Event()

# Dictionary to store locks for each site to ensure sequential communication
site_locks = {}

@app.websocket("/ws/{site_id}")
async def websocket_endpoint(websocket: WebSocket, site_id: str):
    # Create a lock for this site if it doesn't exist
    if site_id not in site_locks:
        site_locks[site_id] = asyncio.Lock()
    
    await websocket.accept()
    
    # Update connection status safely
    connected_remote_sites[site_id] = True
    
    # Store the WebSocket and mark it as pre-accepted
    remote_websockets[site_id] = websocket
    
    print(f"Remote site {site_id} connected", flush=True)
    
    # Check if all expected sites are connected
    if expected_sites and expected_sites.issubset(set(connected_remote_sites.keys())):
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
    
    # Keep this endpoint alive for the entire session
    try:
        while True:
            # If imputation is running, just sleep
            while imputation_running.is_set():
                await asyncio.sleep(1)
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Error in WebSocket connection with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
    finally:
        # Clean up on actual endpoint exit (real disconnect)
        if site_id in connected_remote_sites:
            del connected_remote_sites[site_id]
        if site_id in remote_websockets:
            del remote_websockets[site_id]
        if (site_id in expected_sites) and (not imputation_running.is_set()):
            all_sites_connected.clear()
            print(f"Cleared all_sites_connected flag since {site_id} was expected and disconnected", flush=True)
        print(f"Remote site {site_id} disconnected", flush=True)

# Helper functions to facilitate network communication
async def initialize_remote_sites(mvar_list):
    """Send initialization message to all remote sites with missing variables"""
    # Check if remote sites are still connected
    connected_sites = list(remote_websockets.keys())
    if not connected_sites:
        raise RuntimeError("No remote sites connected. Cannot initialize.")
    
    print(f"Initializing remote sites: {connected_sites}", flush=True)
    
    # Make a copy of the site list to avoid modification during iteration
    site_items = list(remote_websockets.items())
    
    for site_id, websocket in site_items:
        try:
            async with site_locks[site_id]:
                # Mark WebSocket as pre-accepted since we accepted it in the websocket_endpoint
                wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                try:
                    await write_string("Initialize", wrapped_ws)
                    await write_vector(np.array(mvar_list, dtype=np.float64), wrapped_ws)
                    print(f"Initialized remote site {site_id}", flush=True)
                except Exception as e:
                    print(f"Error initializing remote site {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                    if site_id in remote_websockets:
                        del remote_websockets[site_id]
                    if site_id in connected_remote_sites:
                        del connected_remote_sites[site_id]
        except Exception as e:
            print(f"Error acquiring lock for {site_id}: {type(e).__name__}: {str(e)}", flush=True)
            
    # Verify we still have required sites after initialization
    if not remote_websockets:
        raise RuntimeError("All remote sites disconnected during initialization.")

async def finalize_remote_sites(site_ids):
    """Send end message to all remote sites"""
    for site_id in site_ids:
        if site_id in remote_websockets:
            websocket = remote_websockets[site_id]
            try:
                async with site_locks[site_id]:
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                    await write_string("End", wrapped_ws)
                    print(f"Sent End message to {site_id}", flush=True)
            except Exception as e:
                print(f"[CENTRAL] Error sending End message to {site_id}: {type(e).__name__}: {str(e)}", flush=True)

async def dummy_federated_learning(site_ids, num_iterations=1000, debug=True):
    """
    Implement a dummy federated learning algorithm that exchanges random data
    
    Args:
        site_ids: List of remote site IDs
        num_iterations: Number of iterations for random data exchange
        debug: Enable debug mode (default: True)
    
    Returns:
        Total sum of random values collected
    """
    global print_debug_info
    if debug is not None:
        print_debug_info = debug
    # Declare who we expect, and wait until they're all connected
    set_expected_sites(site_ids, debug=debug)
    try:
        await asyncio.wait_for(all_sites_connected.wait(), timeout=60)
    except asyncio.TimeoutError:
        missing = set(site_ids) - set(remote_websockets.keys())
        raise RuntimeError(f"Not all remote sites connected within timeout. Missing: {sorted(missing)}")
    
    # Reset debug counters
    global debug_counters
    debug_counters = {
        "iterations_completed": 0,
        "random_data_received": 0,
        "total_sum": 0.0
    }
    
    # Start timing
    start_time = time.time()
    
    # Ensure we have locks for all sites
    for site_id in site_ids:
        if site_id not in site_locks:
            site_locks[site_id] = asyncio.Lock()
            
    # Set the flag to indicate federated learning is running
    imputation_running.set()
    print("Federated learning started, flag set", flush=True)
    
    if print_debug_info:
        print(f"[CENTRAL] Debug mode ENABLED - detailed logging will be shown", flush=True)
        print(f"[CENTRAL] Starting Federated Learning with {num_iterations} iterations", flush=True)
        print(f"[CENTRAL] Expected remote sites: {site_ids}", flush=True)
    
    try:
        # Initialize remote sites - no specific data needed for dummy federated learning
        await initialize_remote_sites([])
        
        # Variable to store total sum of random values
        total_sum = 0.0
        debug_counters["total_sum"] = 0.0
        
        # Main federated learning loop
        for iteration in range(num_iterations):
            debug_counters["iterations_completed"] += 1
            iter_start_time = time.time()
            
            if print_debug_info and iteration % 100 == 0:
                print(f"[CENTRAL][ITER {iteration+1}/{num_iterations}] Starting new iteration", flush=True)
            
            # Request random data from each site
            for site_id in site_ids:
                if site_id in remote_websockets:
                    websocket = remote_websockets[site_id]
                    try:
                        async with site_locks[site_id]:
                            wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                            
                            # 1. Send request for random data
                            await write_string("request_random", wrapped_ws)
                            
                            # 2. Wait for acknowledgment
                            ack = await read_string(wrapped_ws)
                            if ack != "request_random_ack":
                                print(f"[CENTRAL] Unexpected response from {site_id}: {ack}", flush=True)
                                continue
                            
                            # 3. Receive random data
                            random_value = await read_vector(wrapped_ws)
                            random_value = float(random_value[0])
                            
                            # 4. Add to total
                            total_sum += random_value
                            debug_counters["random_data_received"] += 1
                            debug_counters["total_sum"] = total_sum
                            
                            if print_debug_info and iteration % 100 == 0:
                                print(f"[CENTRAL][ITER {iteration+1}][SITE {site_id}] Received {random_value:.4f}, current total: {total_sum:.4f}", flush=True)
                    
                    except Exception as e:
                        print(f"[CENTRAL][ITER {iteration+1}][SITE {site_id}] Error communicating: {type(e).__name__}: {str(e)}", flush=True)
            
            # Optional delay to avoid overwhelming the network
            if iteration % 50 == 0:
                await asyncio.sleep(0.01)
                
            if print_debug_info and iteration % 100 == 0:
                iter_time = time.time() - iter_start_time
                print(f"[CENTRAL][ITER {iteration+1}] Iteration completed in {iter_time:.3f}s, total sum: {total_sum:.4f}", flush=True)
        
        # Finalize remote sites
        await finalize_remote_sites(site_ids)
        
        total_time = time.time() - start_time
        print(f"[CENTRAL][SUMMARY] Federated Learning completed in {total_time:.3f}s", flush=True)
        print(f"[CENTRAL][SUMMARY] Total iterations: {num_iterations}", flush=True)
        print(f"[CENTRAL][SUMMARY] Total sites: {len(site_ids)}", flush=True)
        print(f"[CENTRAL][SUMMARY] Total sum of random values: {total_sum:.4f}", flush=True)
        print(f"[CENTRAL][SUMMARY] Debug counters: {debug_counters}", flush=True)
        
        return total_sum
    
    finally:
        imputation_running.clear()
        print("Federated learning finished, flag cleared", flush=True)
