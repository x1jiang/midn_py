"""
Connection manager for remote site.
Handles WebSocket connections to the central server.
"""

import websockets
import asyncio
import json
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple

from common.algorithm.job_protocol import create_message, parse_message, ProtocolMessageType


class ConnectionClient:
    """
    Client for managing connection to the central server.
    Handles connection establishment, retries, and message exchange.
    """
    
    def __init__(self, central_url: str, site_id: str, token: str, 
                 status_callback: Optional[Any] = None,
                 retry_delay: int = 15,  # Changed from 5 to 15 seconds
                 connect_timeout: int = 5,  # Quick connection (5 seconds)
                 message_timeout: int = 15,
                 completion_wait_time: int = 120):  # 2 minutes wait after completion
        """
        Initialize connection client.
        
        Args:
            central_url: URL of the central server
            site_id: ID of this site  
            token: Authentication token
            status_callback: Callback for status updates
            retry_delay: Delay between connection attempts (seconds) - 15s when no job running
            connect_timeout: Timeout for connection attempts (seconds)
            message_timeout: Timeout for sending/receiving messages (seconds)
            completion_wait_time: Time to wait after job completion before reconnecting (seconds)
        """
        self.central_url = central_url
        self.site_id = site_id
        self.token = token
        self.status_callback = status_callback
        self.retry_delay = retry_delay
        self.connect_timeout = connect_timeout
        self.message_timeout = message_timeout
        self.completion_wait_time = completion_wait_time
        self.attempt_count = 0
        self.is_connection_established = False
        self.job_completed = False
        
        # Debug connection information
        print(f"📡 ConnectionClient initialized for site: {site_id}")
        print(f"🔗 Central URL: {central_url}")
        print(f"📊 Has status callback: {status_callback is not None}")
        # Force logging of connection URI
        uri = self.get_uri()
        print(f"🌐 Will connect to: {uri}")
    
    def get_uri(self) -> str:
        """
        Get the WebSocket URI for connection.
        
        Returns:
            WebSocket URI
        """
        return f"{self.central_url}/ws/{self.site_id}?token={self.token}"
    
    def is_job_stopped(self) -> bool:
        """
        Check if the job has been stopped by the user or manually marked for stopping.
        This method now only stops on explicit completion, not on temporary disconnection.
        
        Returns:
            True if the job is permanently stopped, False otherwise
        """
        # Only check internal job_completed flag for current connection attempt
        # but don't consider this for the overall job state
        if self.job_completed:
            print("ℹ️ Current connection attempt marked as completed, but job continues")
            # Return False to keep reconnection loop going
            return False
            
        # Then check status callback (application state)
        if self.status_callback:
            # Get site ID and job ID for better debugging
            job_id = getattr(self.status_callback, "job_id", "unknown")
            site_id = getattr(self.status_callback, "site_id", "unknown")
            
            # Get job state
            job_state = self.status_callback.get_job_state()
            
            # If job state is not found, we can't continue
            if not job_state:
                print(f"⚠️ Job state for job {job_id} on site {site_id} not found in application state!")
                # Return True to stop reconnection attempts since we have no job
                return True
            
            # Check if explicitly marked as completed by user action
            # We now only respect explicit user-requested stops
            if job_state.get('status', '').startswith("Job stopped by user"):
                print(f"✓ Job {job_id} explicitly stopped by user")
                return True
                
            # For any other state, keep the job running
            print(f"ℹ️ Job {job_id} still active. Status: {job_state.get('status', 'unknown')}")
            return False
            
        # If no status callback, default to allowing reconnection
        return False
        
    async def request_job_stop(self) -> bool:
        """
        Request to update job status to prepare for reconnection.
        
        Returns:
            True if status update was successful, False otherwise
        """
        if self.status_callback:
            try:
                # Update job status instead of marking as completed
                await self.status_callback.on_complete()
                print("✅ Job status updated for reconnection")
                
                # Set internal flag to allow current connection attempt to finish
                # but don't remove the job
                self.job_completed = True
                
                # Verify the job is still in app state
                job_state = self.status_callback.get_job_state()
                if job_state:
                    print(f"✅ Job still present in app state with status: {job_state.get('status')}")
                    return True
                else:
                    print("⚠️ Warning: Job not found in app state")
                    return False
            except Exception as e:
                print(f"❌ Failed to update job status: {str(e)}")
        return False
    
    async def send_status(self, message: str) -> None:
        """
        Send a status message to the callback.
        
        Args:
            message: Status message
        """
        # Print the message to console for immediate feedback
        print(f"Status: {message}")
        
        # Also send to callback if available
        if self.status_callback:
            await self.status_callback.on_message(message)
    
    async def send_error(self, message: str) -> None:
        """
        Send an error message to the callback.
        
        Args:
            message: Error message
        """
        # Print error to console for immediate feedback
        print(f"Error: {message}")
        
        # Also send to callback if available
        if self.status_callback:
            await self.status_callback.on_error(message)
    
    async def connect(self) -> Tuple[bool, Optional[websockets.WebSocketClientProtocol]]:
        """
        Establish connection to the central server.
        
        Returns:
            (success, websocket) tuple
        """
        self.attempt_count += 1
        uri = self.get_uri()
        
        try:
            # Update status before connecting
            if self.attempt_count == 1:
                await self.send_status(f"Connecting to central server: {uri}")
            else:
                await self.send_status(f"Connection attempt #{self.attempt_count}: Reconnecting to central server...")
            
            # Add debug info for reconnection attempts
            print(f"📡 Connection attempt #{self.attempt_count}: Connecting to {uri}")
            
            # Try to establish connection with timeout
            websocket = await asyncio.wait_for(
                websockets.connect(
                    uri,
                    ping_interval=None,  # Disable automatic ping
                    close_timeout=5,     # Timeout for closing the connection
                    open_timeout=self.connect_timeout  # Timeout for opening the connection
                ),
                timeout=self.connect_timeout
            )
            
            # Success!
            print(f"✅ Connection established on attempt #{self.attempt_count}")
            await self.send_status("Connection established")
            self.is_connection_established = True
            return True, websocket
            
        except Exception as e:
            # Get more specific error message based on exception type
            error_message = str(e)
            if isinstance(e, websockets.exceptions.InvalidStatusCode):
                error_message = f"Server returned invalid status code: {str(e)}"
                # Check if this indicates a job already running
                if "403" in str(e) or "409" in str(e):
                    await self.send_status("Central server indicates a job is already running")
                    return False, None
            elif isinstance(e, websockets.exceptions.ConnectionClosed):
                error_message = f"Connection closed unexpectedly: {str(e)}"
            elif isinstance(e, asyncio.TimeoutError):
                error_message = "Connection timed out - central server may not be ready"
            elif "Timeout" in str(e):
                error_message = f"Operation timed out: {str(e)}"
            
            # Log the error
            print(f"Connection error (attempt #{self.attempt_count}): {error_message}")
            await self.send_status(f"Connection error: {error_message}")
            
            self.is_connection_established = False
            return False, None
    
    async def send_message(self, websocket: websockets.WebSocketClientProtocol, 
                          message_type: ProtocolMessageType, **payload) -> bool:
        """
        Send a message to the central server.
        
        Args:
            websocket: WebSocket connection
            message_type: Type of message to send
            **payload: Message payload
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        message = create_message(message_type, **payload)
        
        try:
            # Send message with timeout
            await asyncio.wait_for(websocket.send(message), timeout=self.message_timeout)
            
            msg_type = message_type.value if hasattr(message_type, 'value') else message_type
            await self.send_status(f"Sent {msg_type} message to central server")
            return True
            
        except Exception as e:
            await self.send_status(f"Error sending message: {str(e)}")
            return False
    
    async def send_message_dict(self, websocket: websockets.WebSocketClientProtocol, 
                               message_dict: Dict[str, Any]) -> bool:
        """
        Send a complete message dictionary to the central server.
        
        Args:
            websocket: WebSocket connection
            message_dict: Complete message dictionary to send
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        try:
            # Import NumpyJSONEncoder from job_protocol if not already imported
            from common.algorithm.job_protocol import NumpyJSONEncoder
            
            # Convert message dict to JSON string using NumpyJSONEncoder for proper NumPy serialization
            message_str = json.dumps(message_dict, cls=NumpyJSONEncoder)
            
            # Send message with timeout
            await asyncio.wait_for(websocket.send(message_str), timeout=self.message_timeout)
            
            msg_type = message_dict.get('type', 'unknown')
            await self.send_status(f"Sent {msg_type} message to central server")
            return True
            
        except Exception as e:
            await self.send_status(f"Error sending message: {str(e)}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            return False
    
    async def receive_message(self, websocket: websockets.WebSocketClientProtocol, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receive a message from the central server.
        
        Args:
            websocket: WebSocket connection
            timeout: Timeout in seconds, None for indefinite wait (like R implementation)
            
        Returns:
            Received message as dictionary, or None if an error occurred
        """
        try:
            # Use provided timeout or default to self.message_timeout
            actual_timeout = timeout if timeout is not None else self.message_timeout
            
            if timeout is None:
                await self.send_status("Waiting for message from central server (indefinite)...")
                message_str = await websocket.recv()  # Wait indefinitely like R
            else:
                await self.send_status("Waiting for message from central server...")
                message_str = await asyncio.wait_for(websocket.recv(), timeout=actual_timeout)
            
            # Parse message
            message = parse_message(message_str)
            
            message_type = message.get('type', 'unknown')
            await self.send_status(f"Received {message_type} message from central server")
            
            return message
            
        except asyncio.TimeoutError:
            if timeout is not None:
                await self.send_status(f"Timeout waiting for message from central server after {actual_timeout}s")
                # Don't mark connection as failed for explicit timeouts - let caller handle
                return None
            else:
                await self.send_status(f"Timeout waiting for message from central server after {actual_timeout}s")
                self.is_connection_established = False
                return None
        except websockets.exceptions.ConnectionClosed as e:
            await self.send_status(f"Connection closed while waiting for message: {str(e)}")
            self.is_connection_established = False
            return None
        except json.JSONDecodeError as e:
            await self.send_status(f"Error decoding message: {str(e)}")
            return None
        except Exception as e:
            await self.send_status(f"Unexpected error receiving message: {str(e)}")
            self.is_connection_established = False
            return None
    
    def mark_job_completed(self) -> None:
        """Mark that the current job has completed."""
        self.job_completed = True
        self.is_connection_established = False
    
    def reset_for_new_job(self) -> None:
        """Reset state for a new job attempt while keeping job in app state."""
        print("🔄 ConnectionClient: Resetting connection state for reconnection attempt")
        
        # Reset internal state variables
        self.job_completed = False
        self.attempt_count = 0
        self.is_connection_established = False
        
        # Update job status in application state but keep the job
        if self.status_callback:
            # Get job ID and site ID
            job_id = getattr(self.status_callback, "job_id", None)
            site_id = getattr(self, "site_id", None)
            
            if job_id and site_id:
                try:
                    # Update the job status without removing it
                    job_state = self.status_callback.get_job_state()
                    if job_state:
                        job_state['status'] = "Reconnecting to central server..."
                        job_state['messages'].append("Reconnecting to central server...")
                        print(f"✅ Updated job {job_id} status for reconnection attempt")
                except Exception as e:
                    print(f"❌ Error updating job status: {str(e)}")
        
        print(f"✅ Reset complete - job_completed={self.job_completed}, attempt_count={self.attempt_count}")
        
        # Force an immediate status update to show in logs
        asyncio.create_task(self.send_status("Connection state reset, reconnecting to central server..."))
        
        # Print URI to confirm connection details are preserved
        uri = self.get_uri()
        print(f"📡 Will reconnect to: {uri}")
