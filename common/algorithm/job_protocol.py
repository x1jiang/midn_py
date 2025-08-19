"""
Common job protocol utilities for federated learning.

This module contains shared protocol definitions, constants, and utilities 
for implementing the standardized job management protocol across both
central services and remote clients.
"""

import json
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Union


class JobStatus(Enum):
    """Standard job status values"""
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class RemoteStatus(Enum):
    """Standard remote site status values"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    READY = "ready"
    COMPUTING = "computing"
    COMPLETED = "completed"


class ProtocolMessageType(Enum):
    """Standard protocol message types"""
    CONNECT = "connect"
    CONNECTION_CONFIRMED = "connection_confirmed"
    SITE_READY = "site_ready"
    START_COMPUTATION = "start_computation"
    JOB_COMPLETED = "job_completed"
    COMPLETION_ACK = "completion_ack"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    # Include message types from protocol.py
    METHOD = "method"
    DATA = "data"
    MODE = "mode"
    STATUS = "status"
    COMPLETE = "complete"


class ErrorCode(Enum):
    """Standard error codes for protocol errors"""
    NO_JOBS_AVAILABLE = "NO_JOBS_AVAILABLE"
    MISSING_JOB_ID = "MISSING_JOB_ID"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    UNAUTHORIZED_SITE = "UNAUTHORIZED_SITE"
    UNKNOWN_CONNECTION = "UNKNOWN_CONNECTION"
    COMPUTATION_ERROR = "COMPUTATION_ERROR"
    TIMEOUT = "TIMEOUT"


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle NumPy arrays and types"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Enum):  # Handle all Enum types including ProtocolMessageType
            return obj.value
        return super(NumpyJSONEncoder, self).default(obj)


def create_message(message_type: Union[ProtocolMessageType, str], **payload) -> str:
    """
    Create a standardized message for transmission.
    
    Args:
        message_type: Type of message to create
        **payload: Message payload key-value pairs
        
    Returns:
        JSON-encoded message string
    """
    if isinstance(message_type, Enum):
        msg_type = message_type.value
    else:
        msg_type = message_type
        
    message = {"type": msg_type}
    message.update(payload)
    
    return json.dumps(message, cls=NumpyJSONEncoder)


def parse_message(message_str: str) -> Dict[str, Any]:
    """
    Parse a message received over the network.
    
    Args:
        message_str: JSON-encoded message string
        
    Returns:
        Decoded message as dictionary
    """
    return json.loads(message_str)


class Protocol:
    """Common protocol utilities for both central and remote implementations"""
    
    @staticmethod
    def create_connect_message(job_id: int, site_id: str) -> Dict[str, Any]:
        """
        Create a standard connection request message.
        
        Args:
            job_id: ID of the job
            site_id: ID of the site
            
        Returns:
            Message dictionary
        """
        return {
            "type": ProtocolMessageType.CONNECT.value,
            "job_id": job_id,
            "site_id": site_id
        }
    
    @staticmethod
    def create_connection_confirmed_message(job_id: int, algorithm: str, status: str) -> Dict[str, Any]:
        """
        Create a standard connection confirmation message.
        
        Args:
            job_id: ID of the job
            algorithm: Algorithm name
            status: Current job status
            
        Returns:
            Message dictionary
        """
        return {
            "type": ProtocolMessageType.CONNECTION_CONFIRMED.value,
            "job_id": job_id,
            "algorithm": algorithm,
            "status": status
        }
    
    @staticmethod
    def create_site_ready_message(job_id: int, site_id: str) -> Dict[str, Any]:
        """
        Create a standard site ready message.
        
        Args:
            job_id: ID of the job
            site_id: ID of the site
            
        Returns:
            Message dictionary
        """
        return {
            "type": ProtocolMessageType.SITE_READY.value,
            "job_id": job_id,
            "site_id": site_id,
            "status": RemoteStatus.READY.value
        }
    
    @staticmethod
    def create_start_computation_message(job_id: int) -> Dict[str, Any]:
        """
        Create a standard start computation message.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Message dictionary
        """
        return {
            "type": ProtocolMessageType.START_COMPUTATION.value,
            "job_id": job_id
        }
    
    @staticmethod
    def create_job_completed_message(job_id: int, result_path: Optional[str] = None, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a standard job completion message.
        
        Args:
            job_id: ID of the job
            result_path: Optional path to results file
            message: Optional completion message
            
        Returns:
            Message dictionary
        """
        msg = {
            "type": ProtocolMessageType.JOB_COMPLETED.value,
            "job_id": job_id,
            "status": JobStatus.COMPLETED.value
        }
        
        if message:
            msg["message"] = message
            
        if result_path:
            msg["result_path"] = result_path
            
        return msg
    
    @staticmethod
    def create_completion_ack_message(job_id: int, site_id: str) -> Dict[str, Any]:
        """
        Create a standard completion acknowledgment message.
        
        Args:
            job_id: ID of the job
            site_id: ID of the site
            
        Returns:
            Message dictionary
        """
        return {
            "type": ProtocolMessageType.COMPLETION_ACK.value,
            "job_id": job_id,
            "site_id": site_id
        }
    
    @staticmethod
    def create_error_message(code: ErrorCode, message: str, **extra_data) -> Dict[str, Any]:
        """
        Create a standard error message.
        
        Args:
            code: Error code
            message: Error message
            **extra_data: Additional error data
            
        Returns:
            Message dictionary
        """
        msg = {
            "type": ProtocolMessageType.ERROR.value,
            "code": code.value if isinstance(code, ErrorCode) else code,
            "message": message
        }
        
        # Add any additional data
        msg.update(extra_data)
        
        return msg
    
    @staticmethod
    def create_heartbeat_message(job_id: int, site_id: str, require_ack: bool = False) -> Dict[str, Any]:
        """
        Create a standard heartbeat message.
        
        Args:
            job_id: ID of the job
            site_id: ID of the site
            require_ack: Whether acknowledgment is required
            
        Returns:
            Message dictionary
        """
        return {
            "type": ProtocolMessageType.HEARTBEAT.value,
            "job_id": job_id,
            "site_id": site_id,
            "require_ack": require_ack
        }
    
    @staticmethod
    def create_heartbeat_ack_message(job_id: int) -> Dict[str, Any]:
        """
        Create a standard heartbeat acknowledgment message.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Message dictionary
        """
        return {
            "type": ProtocolMessageType.HEARTBEAT_ACK.value,
            "job_id": job_id
        }
