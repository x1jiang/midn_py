"""
Standard communication protocol for PYMIDN algorithms.
Defines message formats and utilities for serialization/deserialization.
"""

import json
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, Union, List


class MessageType(Enum):
    """Standard message types for algorithm communication"""
    CONNECT = "connect"
    METHOD = "method"
    DATA = "data"
    MODE = "mode"
    STATUS = "status"
    ERROR = "error"
    COMPLETE = "complete"
    

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
        if isinstance(obj, MessageType):
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
    if isinstance(message_type, MessageType):
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
