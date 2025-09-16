"""
Unified protocol schema for federated learning communication.

This module contains all protocol definitions for message exchange and job management
in federated learning algorithms, consolidating both wire schema and job management
protocols into a single, consistent interface.
"""

import json
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, List, Set, Optional, Union

# ---- Core protocol types as Enums ----
class Method(str, Enum):
    """Method types for algorithm execution"""
    GAUSSIAN = "gaussian"
    LOGISTIC = "logistic"

class Channel(str, Enum):
    """Channel types for message routing"""
    CONTROL = "control"
    DATA = "data"
    UNKNOWN = "unknown"

class MessageType(str, Enum):
    """Message types for protocol communication"""
    # Wire schema message types
    INITIALIZE = "initialize"
    FINALIZE = "finalize"
    TERMINATE = "terminate"
    REQUEST_INFO = "request_info"
    INFO = "info"
    ITERATE = "iterate"
    ITER_RESPONSE = "iter_response"
    UPDATE_PARAMS = "update_params"
    ACK = "ack"
    IMPUTE = "impute"
    GET_FINAL_DATA = "get_final_data"
    FINAL_DATA = "final_data"
    
    # Job management message types
    CONNECT = "connect"
    CONNECTION_CONFIRMED = "connection_confirmed"
    SITE_READY = "site_ready"
    START_COMPUTATION = "start_computation"
    JOB_COMPLETED = "job_completed"
    COMPLETION_ACK = "completion_ack"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    
    # Legacy types (for backward compatibility)
    METHOD = "method"
    DATA = "data"
    
    # Special types
    UNKNOWN = "unknown"
    MODE = "mode"
    STATUS = "status"
    COMPLETE = "complete"

class JobStatus(str, Enum):
    """Standard job status values"""
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

class RemoteStatus(str, Enum):
    """Standard remote site status values"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    READY = "ready"
    COMPUTING = "computing"
    COMPLETED = "completed"

class ErrorCode(str, Enum):
    """Standard error codes for protocol errors"""
    NO_JOBS_AVAILABLE = "NO_JOBS_AVAILABLE"
    MISSING_JOB_ID = "MISSING_JOB_ID"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    UNAUTHORIZED_SITE = "UNAUTHORIZED_SITE"
    UNKNOWN_CONNECTION = "UNKNOWN_CONNECTION"
    COMPUTATION_ERROR = "COMPUTATION_ERROR"
    TIMEOUT = "TIMEOUT"

# ProtocolMessageType is now merged into the unified MessageType enum above

# ---- Message envelope (used everywhere a wire exists) ----
@dataclass
class Envelope:
    channel: Channel
    message_type: MessageType
    method: Optional[Method] = None
    yidx: Optional[int] = None            # 0-based index
    meta: Optional[Dict[str, Union[str, int, float]]] = None
    payload: Optional[Dict[str, object]] = None

# ---- Info/statistics payloads ----
@dataclass
class GaussianInfo:
    # One-shot sufficient stats OR curvature/grad per-iteration
    XTX: Optional[np.ndarray] = None   # (p,p) F-order
    XTy: Optional[np.ndarray] = None   # (p,)
    yTy: Optional[float] = None        
    H:   Optional[np.ndarray] = None   # (p,p)
    g:   Optional[np.ndarray] = None   # (p,)
    Q:   Optional[float] = None        
    n:   Optional[int] = None          

@dataclass
class LogisticInfo:
    H:       Optional[np.ndarray] = None  # (p,p)
    g:       Optional[np.ndarray] = None  # (p,)
    n:       Optional[int] = None         
    log_lik: Optional[float] = None       
    Q:       Optional[float] = None       

# ---- Iteration ask/response (covers SIMI modes + CSL* grad rounds) ----
@dataclass
class IterateAskPayload:
    mode: int                               # 1=H,g,(Q) | 2=Q-only | 0=terminate
    beta: Optional[np.ndarray] = None       # (p,) Gaussian
    beta_candidate: Optional[np.ndarray] = None  # (p,) SIMI mode 2
    alpha: Optional[np.ndarray] = None      # (p,) Logistic
    betabar: Optional[np.ndarray] = None    # (p,) Used by: CSLMI/CSLMICE gradient rounds

@dataclass
class IterateResponsePayload:
    H:   Optional[np.ndarray] = None  # (p,p)
    g:   Optional[np.ndarray] = None  # (p,)
    Q:   Optional[float] = None       
    n:   Optional[int] = None         

# ---- Parameter update / imputation broadcast ----
@dataclass
class ParamUpdatePayload:
    beta:  Optional[np.ndarray] = None  # (p,) Gaussian
    alpha: Optional[np.ndarray] = None  # (p,) Logistic
    sigma: Optional[float] = None       # Gaussian scale

# ---- HDMI packed parameter block (normalized) ----
@dataclass
class HDMIParamsPayload:
    beta_sel: Optional[np.ndarray] = None  # (p,)     selection model
    beta_out: Optional[np.ndarray] = None  # (p-1,)   outcome model
    sigma:    Optional[float] = None       # Gaussian only
    rho:      Optional[float] = None
    vcov:     Optional[np.ndarray] = None  # full vcov (Fortran order)
    n:        Optional[int] = None

# ---- JSON Encoding/Decoding ----
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
        if isinstance(obj, Enum):
            return obj.value
        return super(NumpyJSONEncoder, self).default(obj)

def create_message(message_type: Union[MessageType, str], **payload) -> str:
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

# ---- Protocol Utilities ----
class Protocol:
    """Common protocol utilities for both central and remote implementations"""
    
    @staticmethod
    def create_data_message(job_id: int, instruction: str, **payload) -> Dict[str, Any]:
        """
        Create a standard DATA message with embedded instruction for computation.
        All algorithm-specific computation instructions should use this message type.
        """
        message = {
            "type": MessageType.DATA.value,
            "job_id": job_id,
            "instruction": instruction
        }
        message.update(payload)
        return message
    
    @staticmethod
    def create_connect_message(job_id: int, site_id: str) -> Dict[str, Any]:
        """Create a standard connection request message."""
        return {
            "type": MessageType.CONNECT.value,
            "job_id": job_id,
            "site_id": site_id
        }
    
    @staticmethod
    def create_connection_confirmed_message(job_id: int, algorithm: str, status: JobStatus) -> Dict[str, Any]:
        """Create a standard connection confirmation message."""
        return {
            "type": MessageType.CONNECTION_CONFIRMED.value,
            "job_id": job_id,
            "algorithm": algorithm,
            "status": status.value if isinstance(status, JobStatus) else status
        }
    
    @staticmethod
    def create_site_ready_message(job_id: int, site_id: str) -> Dict[str, Any]:
        """Create a standard site ready message."""
        return {
            "type": MessageType.SITE_READY.value,
            "job_id": job_id,
            "site_id": site_id,
            "status": RemoteStatus.READY.value
        }
    
    @staticmethod
    def create_start_computation_message(job_id: int) -> Dict[str, Any]:
        """Create a standard start computation message."""
        return {
            "type": MessageType.START_COMPUTATION.value,
            "job_id": job_id
        }
    
    @staticmethod
    def create_job_completed_message(job_id: int, result_path: Optional[str] = None, message: Optional[str] = None) -> Dict[str, Any]:
        """Create a standard job completion message."""
        msg = {
            "type": MessageType.JOB_COMPLETED.value,
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
        """Create a standard completion acknowledgment message."""
        return {
            "type": MessageType.COMPLETION_ACK.value,
            "job_id": job_id,
            "site_id": site_id
        }
    
    @staticmethod
    def create_error_message(code: Union[ErrorCode, str], message: str, **extra_data) -> Dict[str, Any]:
        """Create a standard error message."""
        msg = {
            "type": MessageType.ERROR.value,
            "code": code.value if isinstance(code, ErrorCode) else code,
            "message": message
        }
        
        # Add any additional data
        msg.update(extra_data)
        
        return msg
    
    @staticmethod
    def create_heartbeat_message(job_id: int, site_id: str, require_ack: bool = False) -> Dict[str, Any]:
        """Create a standard heartbeat message."""
        return {
            "type": MessageType.HEARTBEAT.value,
            "job_id": job_id,
            "site_id": site_id,
            "require_ack": require_ack
        }
    
    @staticmethod
    def create_heartbeat_ack_message(job_id: int) -> Dict[str, Any]:
        """Create a standard heartbeat acknowledgment message."""
        return {
            "type": MessageType.HEARTBEAT_ACK.value,
            "job_id": job_id
        }

    @staticmethod
    def create_envelope(channel: Channel, message_type: MessageType, 
                       method: Optional[Method] = None,
                       yidx: Optional[int] = None,
                       meta: Optional[Dict[str, Any]] = None, 
                       payload: Optional[Dict[str, Any]] = None) -> Envelope:
        """
        Create a standardized message envelope.
        
        Args:
            channel: Message channel (control or data)
            message_type: Type of message 
            method: Optional method (Method.GAUSSIAN or Method.LOGISTIC)
            yidx: Optional target column index (0-based)
            meta: Optional metadata
            payload: Optional payload data
            
        Returns:
            Envelope object ready for transmission
        """
        return Envelope(
            channel=channel,
            message_type=message_type,
            method=method,
            yidx=yidx,
            meta=meta,
            payload=payload
        )
