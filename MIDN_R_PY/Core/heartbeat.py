"""
Heartbeat utilities for maintaining WebSocket connections.
"""

import asyncio
from typing import Dict, Any, Callable
from Core.transfer import (
    read_string, write_string, get_wrapped_websocket
)

async def maintain_heartbeat(
    websockets: Dict[str, Any],
    check_interval: int = 5,  # Reduced from 15 to 5 seconds
    ping_message: str = "ping",
    expected_response: str = "pong",
    send_func: Callable = None,
    receive_func: Callable = None,
    timeout: float = 3.0,  # Reduced from 5.0 to 3.0 seconds
    locks: Dict[str, asyncio.Lock] = None,
    active_flag: asyncio.Event = None
):
    """
    Maintain heartbeat with connected websockets.
    
    Args:
        websockets: Dictionary mapping site IDs to websocket objects
        check_interval: Time between heartbeats in seconds
        ping_message: Message to send for heartbeat
        expected_response: Expected response from clients
        send_func: Function to use for sending messages (if None, will use websocket.send_text)
        receive_func: Function to use for receiving messages (if None, will use websocket.receive_text)
        timeout: Timeout for waiting for responses in seconds
        locks: Dictionary of locks for each site ID to prevent concurrent access
        active_flag: Event that indicates whether active operations are in progress
    """
    while True:
        # Wait for the check interval
        await asyncio.sleep(check_interval)
        
        # Skip heartbeats if active operations are in progress
        if active_flag and active_flag.is_set():
            print(f"Skipping heartbeat check - active operations in progress")
            continue
            
        # Copy the keys to avoid modification during iteration
        site_ids = list(websockets.keys())
        
        for site_id in site_ids:
            websocket = websockets.get(site_id)
            if not websocket:
                continue
                
            try:
                # Acquire lock if provided
                if locks and site_id in locks:
                    # Try to acquire the lock but don't block if it's being used
                    if not locks[site_id].locked():
                        async with locks[site_id]:
                            # Wrap the websocket to use JSON-based protocol
                            wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                            
                            # Send heartbeat using write_string
                            if send_func:
                                await send_func(wrapped_ws, ping_message)
                            else:
                                await write_string(ping_message, wrapped_ws)
                            
                            # Wait for response with timeout using read_string
                            if receive_func:
                                response = await asyncio.wait_for(receive_func(wrapped_ws), timeout=timeout)
                            else:
                                response = await asyncio.wait_for(read_string(wrapped_ws), timeout=timeout)
                            
                            # Verify response
                            if response != expected_response:
                                print(f"Unexpected heartbeat response from {site_id}: {response}")
                    else:
                        # Skip if lock is in use
                        print(f"Skipping heartbeat for {site_id} - lock is in use")
                else:
                    # No locks provided, proceed directly
                    # Wrap the websocket to use JSON-based protocol
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                    
                    # Send heartbeat using write_string
                    if send_func:
                        await send_func(wrapped_ws, ping_message)
                    else:
                        await write_string(ping_message, wrapped_ws)
                    
                    # Wait for response with timeout using read_string
                    if receive_func:
                        response = await asyncio.wait_for(receive_func(wrapped_ws), timeout=timeout)
                    else:
                        response = await asyncio.wait_for(read_string(wrapped_ws), timeout=timeout)
                    
                    # Verify response
                    if response != expected_response:
                        print(f"Unexpected heartbeat response from {site_id}: {response}")
                
            except asyncio.TimeoutError:
                print(f"Heartbeat timeout for {site_id}")
                # Consider the connection dead
                if site_id in websockets:
                    del websockets[site_id]
                    print(f"Removed {site_id} due to heartbeat timeout")
                    
            except Exception as e:
                print(f"Heartbeat error for {site_id}: {type(e).__name__}: {str(e)}")
                # Remove the connection
                if site_id in websockets:
                    del websockets[site_id]
                    print(f"Removed {site_id} due to heartbeat error")
