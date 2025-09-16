"""
Adapter functions to convert between unified wire schema and legacy job protocol formats.
"""

from typing import Dict, Any, Optional

from .protocol import (
    Envelope, MessageType, Channel, Method,
    create_message
)


def envelope_to_protocol_message(envelope: Envelope) -> Envelope:
    """Convert an Envelope message to a ProtocolMessage."""
    # Map envelope types to protocol message types
    type_map = {
        MessageType.INITIALIZE: MessageType.METHOD,
        MessageType.INFO: MessageType.DATA,
        MessageType.ITERATE: MessageType.MODE,
        MessageType.TERMINATE: MessageType.JOB_COMPLETED
    }

    # Map envelope channels to protocol channels
    channel_map = {
        Channel.CONTROL: Channel.CONTROL,
        Channel.DATA: Channel.DATA
    }

    message_type = type_map.get(envelope.message_type, MessageType.UNKNOWN)
    channel = channel_map.get(envelope.channel, Channel.UNKNOWN)

    return Envelope(
        message_type=message_type,
        channel=channel,
        payload=envelope.payload,
        meta=envelope.meta
    )


def protocol_message_to_envelope(message: Dict[str, Any]) -> Envelope:
    """
    Convert a job protocol message to an Envelope.
    
    Args:
        message: Job protocol message
        
    Returns:
        Data protocol envelope
    """
    # Extract message type
    protocol_type = message.get("message_type")
    job_id = message.get("job_id")
    
    # Map protocol message types to envelope types
    type_map = {
        MessageType.METHOD: MessageType.INITIALIZE,
        MessageType.DATA: MessageType.INFO,
        MessageType.MODE: MessageType.ITERATE,
        MessageType.JOB_COMPLETED: MessageType.TERMINATE
    }
    
    # Determine channel based on message context
    channel: Channel = Channel.CONTROL if protocol_type in [
        MessageType.JOB_COMPLETED,
        MessageType.METHOD
    ] else Channel.DATA
    
    # Extract payload and metadata
    payload = {}
    meta = {"job_id": job_id} if job_id else {}
    
    # Copy relevant data to payload or meta
    for key, value in message.items():
        if key in ["message_type", "job_id", "method", "yidx"]:
            continue
        elif key in ["mode", "beta", "alpha", "H", "g", "Q", "n"]:
            payload[key] = value
        else:
            meta[key] = value
    
    # Create the envelope
    return Envelope(
        channel=channel,
        message_type=type_map.get(protocol_type, MessageType.INFO),
        method=message.get("method"),
        yidx=message.get("yidx"),
        meta=meta,
        payload=payload if payload else None
    )
