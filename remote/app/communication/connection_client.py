"""
Connection client for remote site.
Handles communication with the central server.
"""

import asyncio
import websockets
import json
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple

from common.algorithm.protocol import create_message, parse_message, MessageType


class ConnectionClient:
    """
    Client for managing connection to the central server.
    Handles connection establishment, retries, and message exchange.
    """
    
    def __init__(self, central_url: str, site_id: str, token: str, 
                 status_callback: Optional[Any] = None,
                 retry_delay: int = 5,
                 connect_timeout: int = 10,
                 message_timeout: int = 5):
        """
        Initialize connection client.
        
        Args:
            central_url: URL of the central server
            site_id: ID of this site
            token: Authentication token
            status_callback: Callback for status updates
            retry_delay: Delay between retry attempts (seconds)
            connect_timeout: Timeout for connection attempts (seconds)
            message_timeout: Timeout for message sending/receiving (seconds)
        """
        self.central_url = central_url
        self.site_id = site_id
        self.token = token
        self.status_callback = status_callback
        self.retry_delay = retry_delay
        self.connect_timeout = connect_timeout
        self.message_timeout = message_timeout
        self.websocket = None
        self.attempt_count = 0
    
    def get_uri(self) -> str:
        """
        Get the WebSocket URI for connection.
        
        Returns:
            WebSocket URI
        """
        return f"{self.central_url}/ws/{self.site_id}?token={self.token}"
    
    def is_job_stopped(self) -> bool:
        """
        Check if the job has been stopped by the user.
        
        Returns:
            True if the job is stopped, False otherwise
        """
        if self.status_callback:
            job_state = self.status_callback.get_job_state()
            return job_state and job_state.get('completed', False)
        return False
    
    async def send_status(self, message: str) -> None:
        """
        Send a status message to the callback.
        
        Args:
            message: Status message
        """
        if self.status_callback:
            await self.status_callback.on_message(message)
    
    async def send_error(self, message: str) -> None:
        """
        Send an error message to the callback.
        
        Args:
            message: Error message
        """
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
            
            await self.send_status("Connection established")
            return True, websocket
            
        except Exception as e:
            # Get more specific error message based on exception type
            error_message = str(e)
            if isinstance(e, websockets.exceptions.InvalidStatusCode):
                error_message = f"Server returned invalid status code: {str(e)}"
            elif isinstance(e, websockets.exceptions.ConnectionClosed):
                error_message = f"Connection closed unexpectedly: {str(e)}"
            elif isinstance(e, asyncio.TimeoutError):
                error_message = "Connection timed out"
            elif "Timeout" in str(e):
                error_message = f"Operation timed out: {str(e)}"
            
            # Log the error
            print(f"Connection error (attempt #{self.attempt_count}): {error_message}")
            await self.send_status(f"Connection error: {error_message}")
            
            return False, None
    
    async def send_message(self, websocket: websockets.WebSocketClientProtocol, 
                          message_type: MessageType, **payload) -> bool:
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
    
    async def receive_message(self, websocket: websockets.WebSocketClientProtocol) -> Optional[Dict[str, Any]]:
        """
        Receive a message from the central server.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            Received message as dictionary, or None if an error occurred
        """
        try:
            # Receive message with timeout
            message_str = await asyncio.wait_for(websocket.recv(), timeout=self.message_timeout)
            
            # Parse message
            message = parse_message(message_str)
            
            message_type = message.get('type', 'unknown')
            await self.send_status(f"Received {message_type} message from central server")
            
            return message
            
        except Exception as e:
            await self.send_status(f"Error receiving message: {str(e)}")
            return None
