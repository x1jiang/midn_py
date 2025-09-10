"""
Python implementation of DUMMYRemote.py for Federated Learning

A simple implementation for a remote site in a federated learning system
that generates random data when requested by the central server.
"""

import numpy as np
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
import time
from datetime import datetime
import os
import random
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector, 
    read_string, write_string, read_integer, write_integer,
    read_number,
    WebSocketWrapper, get_wrapped_websocket
)

def create_remote_app(D, port, central_host, central_port, site_id=None):
    """Create a FastAPI app for the remote site
    
    Args:
        D: Data matrix (not used in dummy implementation)
        port: Port to listen on
        central_host: Host of central server
        central_port: Port of central server
        site_id: Site ID to use (default: None, will use "remote{port}")
    """
    # Create FastAPI app
    app = FastAPI()
    
    # Store connection status
    connection_status = {"connected": False, "site_id": site_id or f"remote{port}"}

    @app.on_event("startup")
    async def startup_event():
        # Optional one-time seeding
        seed_env = os.getenv("GLOBAL_SEED")
        if seed_env:
            try:
                np.random.seed(int(seed_env))
                random.seed(int(seed_env))
                print(f"[{connection_status['site_id']}] GLOBAL_SEED set; using seed {int(seed_env)}", flush=True)
            except Exception:
                pass
        await connect_to_central(central_host, central_port, port, connection_status, site_id)
        
    @app.get("/status")
    async def get_status():
        return connection_status
    
    return app

async def connect_to_central(central_host, central_port, local_port, connection_status, site_id=None):
    """Connect to the central server and handle communication"""
    actual_site_id = site_id if site_id else f"remote{local_port}"
    connection_status["site_id"] = actual_site_id

    central_url = f"ws://{central_host}:{central_port}/ws/{actual_site_id}"
    print(f"Remote site {actual_site_id} will connect to central at: {central_url}", flush=True)

    asyncio.create_task(maintain_connection(central_url, connection_status))

async def maintain_connection(central_url, connection_status):
    """Maintain connection to central server with automatic reconnection"""
    max_reconnect_delay = 30
    reconnect_delay = 1
    
    while True:
        try:
            async with websockets.connect(central_url, ping_interval=None) as websocket:
                connection_status["connected"] = True
                print(f"Connected to central server at {central_url}")
                wrapped_ws = get_wrapped_websocket(websocket)
                consecutive_errors = 0
                max_consecutive_errors = 3
                
                while consecutive_errors < max_consecutive_errors:
                    try:
                        inst = await asyncio.wait_for(read_string(wrapped_ws), timeout=30.0)
                        consecutive_errors = 0
                        
                        if inst == "Initialize":
                            try:
                                # We receive mvar list but ignore it in this dummy implementation
                                mvar = await read_vector(wrapped_ws)
                                site_id = connection_status['site_id']
                                print(f"[{site_id}] Initialized - ready for federated learning")
                            except Exception as e:
                                site_id = connection_status.get('site_id', 'unknown')
                                print(f"[{site_id}] Error during Initialize: {type(e).__name__}: {e}")
                                raise
                        
                        elif inst == "request_random":
                            try:
                                start_time = time.time()
                                site_id = connection_status['site_id']
                                
                                # Generate random value
                                random_value = random.random() * 10  # Random float between 0 and 10
                                
                                # Send acknowledgment
                                await write_string("request_random_ack", wrapped_ws)
                                
                                # Send random value
                                await write_vector(np.array([random_value]), wrapped_ws)
                                
                                # Track iterations with a counter
                                if 'iteration_counter' not in connection_status:
                                    connection_status['iteration_counter'] = 0
                                connection_status['iteration_counter'] += 1
                                
                                # Always show timing for iterations divisible by 100
                                if connection_status['iteration_counter'] % 100 == 0:
                                    end_time = time.time() - start_time
                                    print(f"[{site_id}][ITER {connection_status['iteration_counter']}] Generated random value: {random_value:.4f} in {end_time:.3f}s", flush=True)
                                # Sometimes show other values too (5% chance)
                                elif (random.random() < 0.05):
                                    print(f"[{site_id}][ITER {connection_status['iteration_counter']}] Generated random value: {random_value:.4f}", flush=True)
                                
                            except Exception as e:
                                site_id = connection_status.get('site_id', 'unknown')
                                print(f"[{site_id}] Error during request_random: {type(e).__name__}: {e}")
                                raise
                        
                        elif inst == "End":
                            print(f"[{connection_status['site_id']}] Received End command from central server")
                            return
                        
                        elif inst == "ping":
                            await write_string("pong", wrapped_ws)
                        else:
                            print(f"[{connection_status['site_id']}] Unknown instruction: {inst}")
                    
                    except asyncio.TimeoutError:
                        try:
                            await write_string("ping", wrapped_ws)
                        except Exception as e:
                            print(f"[{connection_status['site_id']}] Connection seems dead: {type(e).__name__}: {str(e)}", flush=True)
                            break
                    except ValueError as e:
                        consecutive_errors += 1
                        print(f"[{connection_status['site_id']}] Protocol error: {e}, error count: {consecutive_errors}/{max_consecutive_errors}")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"[{connection_status['site_id']}] Error processing message: {type(e).__name__}: {e}")
                        consecutive_errors += 1
                        if isinstance(e, (websockets.exceptions.ConnectionClosedError, 
                                          ConnectionResetError, 
                                          websockets.exceptions.ConnectionClosedOK)):
                            print(f"[{connection_status['site_id']}] Connection error, will reconnect")
                            break
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[{connection_status['site_id']}] Too many consecutive errors, reconnecting")
        except Exception as e:
            connection_status["connected"] = False
            print(f"Error connecting to {central_url}: {type(e).__name__}: {e}")
        
        print(f"[{connection_status['site_id']}] Waiting to reconnect...")
        await asyncio.sleep(5)

def dummy_remote(D, port, central_host, central_port, site_id=None):
    """Start a dummy federated learning remote client"""
    actual_site_id = site_id if site_id else f"remote{port}"
    app = create_remote_app(D, port, central_host, central_port, actual_site_id)
    uvicorn.run(app, host="0.0.0.0", port=port)
