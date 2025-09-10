import numpy as np
import struct
import enum
from dataclasses import dataclass
from typing import Union, Any, Tuple, Optional
from fastapi import WebSocket

# WebSocketWrapper class to handle different WebSocket implementations
class WebSocketWrapper:
    """
    A wrapper class that provides a consistent interface for different WebSocket implementations.
    This allows us to use the same code with FastAPI WebSocket objects and the websockets library.
    """
    def __init__(self, websocket):
        self.websocket = websocket
        # Check websocket type to determine which methods to use
        self.ws_type = self._determine_websocket_type(websocket)
        self.is_active = True  # Flag to track if connection is still active
    
    def _determine_websocket_type(self, ws):
        """Determine the type of websocket implementation"""
        # Check for FastAPI WebSocket
        if hasattr(ws, 'send_bytes') and callable(ws.send_bytes) and hasattr(ws, 'send_text'):
            return "fastapi"
        # Check for websockets library Connection
        elif hasattr(ws, 'send') and callable(ws.send) and hasattr(ws, 'recv'):
            return "websockets"
        # Check for ClientConnection from websockets.legacy
        elif hasattr(ws, 'send') and callable(ws.send):
            return "websockets_legacy"
        else:
            print(f"Unknown websocket type: {type(ws)}, will try basic methods")
            return "unknown"
    
    async def send_bytes(self, data: bytes) -> None:
        """Send binary data over the WebSocket"""
        if not self.is_active:
            print("Warning: Attempting to send on closed WebSocket")
            return
        
        try:
            if self.ws_type == "fastapi":
                await self.websocket.send_bytes(data)
            elif self.ws_type in ["websockets", "websockets_legacy", "unknown"]:
                await self.websocket.send(data)
            else:
                raise NotImplementedError(f"WebSocket type {self.ws_type} does not support sending bytes")
        except Exception as e:
            self.is_active = False
            print(f"Error sending bytes: {type(e).__name__}: {str(e)}")
            raise
    
    async def receive_bytes(self) -> bytes:
        """Receive binary data from the WebSocket"""
        if not self.is_active:
            print("Warning: Attempting to receive on closed WebSocket")
            return b''
            
        try:
            if self.ws_type == "fastapi":
                return await self.websocket.receive_bytes()
            elif self.ws_type in ["websockets", "websockets_legacy"]:
                message = await self.websocket.recv()
                # Ensure we got bytes
                if isinstance(message, str):
                    return message.encode('utf-8')
                return message
            else:
                raise NotImplementedError(f"WebSocket type {self.ws_type} does not support receiving bytes")
        except Exception as e:
            self.is_active = False
            print(f"Error receiving bytes: {type(e).__name__}: {str(e)}")
            raise
    
    async def send_text(self, data: str) -> None:
        """Send text data over the WebSocket"""
        if not self.is_active:
            print("Warning: Attempting to send on closed WebSocket")
            return
            
        try:
            if self.ws_type == "fastapi":
                await self.websocket.send_text(data)
            elif self.ws_type in ["websockets", "websockets_legacy", "unknown"]:
                await self.websocket.send(data)
            else:
                raise NotImplementedError(f"WebSocket type {self.ws_type} does not support sending text")
        except Exception as e:
            self.is_active = False
            print(f"Error sending text: {type(e).__name__}: {str(e)}")
            raise
    
    async def receive_text(self) -> str:
        """Receive text data from the WebSocket"""
        if not self.is_active:
            print("Warning: Attempting to receive on closed WebSocket")
            return ""
            
        try:
            if self.ws_type == "fastapi":
                return await self.websocket.receive_text()
            elif self.ws_type in ["websockets", "websockets_legacy"]:
                message = await self.websocket.recv()
                # Ensure we got text
                if isinstance(message, bytes):
                    return message.decode('utf-8')
                return message
            else:
                raise NotImplementedError(f"WebSocket type {self.ws_type} does not support receiving text")
        except Exception as e:
            self.is_active = False
            print(f"Error receiving text: {type(e).__name__}: {str(e)}")
            raise
    
    def __getattr__(self, name):
        """Forward any other attribute access to the underlying WebSocket object"""
        try:
            return getattr(self.websocket, name)
        except (AssertionError, AttributeError) as e:
            # Handle cases where attributes are not accessible
            print(f"WebSocket wrapper couldn't access '{name}': {str(e)}")
            raise AttributeError(f"WebSocket wrapper couldn't access '{name}': {str(e)}")

# Function to get a wrapped WebSocket
def get_wrapped_websocket(websocket):
    """Create a wrapped WebSocket object for consistent interface"""
    # If it's already a wrapper, return it
    if isinstance(websocket, WebSocketWrapper):
        return websocket
    # Otherwise, create a new wrapper
    try:
        return WebSocketWrapper(websocket)
    except Exception as e:
        # Handle any initialization errors
        print(f"Warning: Failed to create WebSocketWrapper: {str(e)}")
        # Return a minimal wrapper with basic functionality
        return MinimalWebSocketWrapper(websocket)

# Fallback wrapper that only implements the minimum needed functions
class MinimalWebSocketWrapper:
    """A minimal WebSocket wrapper for when the full wrapper can't be created"""
    def __init__(self, websocket):
        self.websocket = websocket
        print(f"Using minimal wrapper for WebSocket type: {type(websocket)}")

    async def send_bytes(self, data: bytes) -> None:
        if hasattr(self.websocket, 'send_bytes'):
            await self.websocket.send_bytes(data)
        elif hasattr(self.websocket, 'send'):
            await self.websocket.send(data)
        else:
            raise NotImplementedError("Cannot send bytes")

    async def receive_bytes(self) -> bytes:
        if hasattr(self.websocket, 'receive_bytes'):
            return await self.websocket.receive_bytes()
        elif hasattr(self.websocket, 'recv'):
            message = await self.websocket.recv()
            if isinstance(message, str):
                return message.encode('utf-8')
            return message
        else:
            raise NotImplementedError("Cannot receive bytes")
            
    async def send_text(self, data: str) -> None:
        if hasattr(self.websocket, 'send_text'):
            await self.websocket.send_text(data)
        elif hasattr(self.websocket, 'send'):
            await self.websocket.send(data)
        else:
            raise NotImplementedError("Cannot send text")

    async def receive_text(self) -> str:
        if hasattr(self.websocket, 'receive_text'):
            return await self.websocket.receive_text()
        elif hasattr(self.websocket, 'recv'):
            message = await self.websocket.recv()
            if isinstance(message, bytes):
                return message.decode('utf-8')
            return message
        else:
            raise NotImplementedError("Cannot receive text")

# R2P1 protocol constants
MAGIC = b"R2P1"
VERSION = 0x01

class Cmd(enum.IntEnum):
    PUT = 0x01   # carries payload to store/send
    GET = 0x02   # requests data (no payload)
    ACK = 0x03   # acknowledges receipt (no payload)
    ERR = 0x7F   # error with optional UTF-8 payload

class Obj(enum.IntEnum):
    VEC = 0x00
    MAT = 0x01

class DType(enum.IntEnum):
    INT32 = 0x00
    FLOAT64 = 0x01

HEADER_FMT = "!4sBBBBIII"  # MAGIC, VER, CMD, OBJ, DTYPE, CORR, ROWS, COLS
HEADER_SIZE = struct.calcsize(HEADER_FMT)

@dataclass
class Header:
    magic: bytes
    version: int
    cmd: Cmd
    obj: Obj
    dtype: DType
    corr_id: int
    rows: int
    cols: int

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_FMT,
            self.magic,
            self.version,
            int(self.cmd),
            int(self.obj),
            int(self.dtype),
            self.corr_id,
            self.rows,
            self.cols,
        )

    @staticmethod
    def unpack(buf: bytes) -> "Header":
        if len(buf) < HEADER_SIZE:
            raise ValueError("incomplete header")
        magic, ver, cmd, obj, dtype, corr, rows, cols = struct.unpack(HEADER_FMT, buf[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic} (expected {MAGIC})")
        if ver != VERSION:
            raise ValueError(f"unsupported version: {ver}")
        return Header(magic, ver, Cmd(cmd), Obj(obj), DType(dtype), corr, rows, cols)

def _np_dtype_for(dt: DType):
    """Get the appropriate numpy dtype for the protocol's data type."""
    if dt == DType.INT32:
        return np.dtype(">i4")  # big-endian int32
    elif dt == DType.FLOAT64:
        return np.dtype(">f8")  # big-endian float64
    else:
        raise ValueError(f"Unsupported dtype {dt}")

def encode_vector(vec, *, dtype: DType, corr_id: int) -> bytes:
    """Encode a 1-D vector into a single binary WebSocket frame."""
    arr = np.asarray(vec)
    if arr.ndim != 1:
        raise ValueError("Vector must be 1-dimensional")
    be = arr.astype(_np_dtype_for(dtype), copy=False)
    hdr = Header(MAGIC, VERSION, Cmd.PUT, Obj.VEC, dtype, corr_id, be.shape[0], 1)
    return hdr.pack() + be.tobytes(order="C")

def encode_matrix(mat, *, dtype: DType, corr_id: int) -> bytes:
    """Encode a 2-D matrix (row-major) into a single binary WebSocket frame."""
    arr = np.asarray(mat)
    if arr.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional")
    rows, cols = arr.shape
    be = arr.astype(_np_dtype_for(dtype), copy=False)
    hdr = Header(MAGIC, VERSION, Cmd.PUT, Obj.MAT, dtype, corr_id, rows, cols)
    return hdr.pack() + be.tobytes(order="C")

def encode_get(obj: Obj, *, dtype: DType, corr_id: int, rows: int = 0, cols: int = 0) -> bytes:
    """Encode a GET request (no payload). rows/cols may be hints; set to 0 if unknown."""
    hdr = Header(MAGIC, VERSION, Cmd.GET, obj, dtype, corr_id, rows, cols)
    return hdr.pack()

def encode_ack(corr_id: int) -> bytes:
    """Encode an ACK response (no payload)."""
    hdr = Header(MAGIC, VERSION, Cmd.ACK, Obj.VEC, DType.INT32, corr_id, 0, 0)
    return hdr.pack()

def encode_err(corr_id: int, message: str) -> bytes:
    """Encode an error message."""
    payload = message.encode("utf-8")
    # We use the rows field to specify the error message length
    hdr = Header(MAGIC, VERSION, Cmd.ERR, Obj.VEC, DType.INT32, corr_id, len(payload), 0)
    return hdr.pack() + payload

def decode_frame(frame: bytes) -> Tuple[Header, Optional[np.ndarray], Optional[str]]:
    """Decode a full WebSocket binary frame into (Header, ndarray|None, err|None)."""
    hdr = Header.unpack(frame)
    payload = frame[HEADER_SIZE:]

    if hdr.cmd == Cmd.ERR:
        return hdr, None, payload.decode("utf-8", errors="replace") if payload else "Unknown error"

    if hdr.cmd in (Cmd.ACK, Cmd.GET) and not payload:
        return hdr, None, None

    if hdr.obj == Obj.VEC:
        dtype = _np_dtype_for(hdr.dtype)
        expected = hdr.rows * 1 * dtype.itemsize
        if len(payload) != expected:
            raise ValueError(f"Payload size mismatch for vector: got {len(payload)}, expected {expected}")
        arr = np.frombuffer(payload, dtype=dtype, count=hdr.rows)
        return hdr, arr, None

    if hdr.obj == Obj.MAT:
        dtype = _np_dtype_for(hdr.dtype)
        expected = hdr.rows * hdr.cols * dtype.itemsize
        if len(payload) != expected:
            raise ValueError(f"Payload size mismatch for matrix: got {len(payload)}, expected {expected}")
        arr = np.frombuffer(payload, dtype=dtype, count=hdr.rows * hdr.cols)
        arr = arr.reshape((hdr.rows, hdr.cols))
        return hdr, arr, None

    raise ValueError(f"Unknown object type: {hdr.obj}")

# WebSocket helpers for the R2P1 protocol
async def ws_send(websocket, data: bytes) -> None:
    """Send binary data over WebSocket."""
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    await ws.send_bytes(data)

async def ws_recv(websocket) -> Tuple[Header, Optional[np.ndarray], Optional[str]]:
    """Receive and decode a binary frame from WebSocket."""
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    msg = await ws.receive_bytes()
    if isinstance(msg, str):
        msg = msg.encode('utf-8')  # Convert to bytes if we somehow got text
    return decode_frame(msg)

# R2P1 protocol vector and matrix functions
async def read_vector(websocket: Union[WebSocket, Any]) -> np.ndarray:
    """
    Read a vector from WebSocket using the R2P1 protocol
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    try:
        # Read binary message
        message = await ws.receive_bytes()
        
        # Parse using R2P1 protocol
        hdr, arr, err = decode_frame(message)
        
        if err:
            print(f"ERROR: Received error message: {err}")
            raise ValueError(f"Protocol error: {err}")
        
        if hdr.obj != Obj.VEC:
            print(f"ERROR: Expected vector, but received {hdr.obj.name}")
            raise ValueError(f"Expected vector, but received {hdr.obj.name}")
        
        if arr is None:
            print(f"ERROR: No data payload in vector message")
            raise ValueError("No data payload in vector message")
        
        print(f"Received vector of length {len(arr)}")
        return arr
        
    except Exception as e:
        print(f"Error reading vector: {type(e).__name__}: {str(e)}")
        raise

async def write_vector(v: np.ndarray, websocket: Union[WebSocket, Any]) -> None:
    """
    Write a vector to WebSocket using the R2P1 protocol
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    try:
        # Static correlation ID (can be improved with a counter)
        corr_id = 1
        
        # Sanitize vector before sending
        v = np.asarray(v)
        a = len(v)
        
        # Check for problematic values
        nan_mask = np.isnan(v)
        inf_mask = np.isinf(v)
        
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            print(f"WARNING: Found {nan_count} NaN values in vector, replacing with 0.0")
            v = np.copy(v)  # Make a copy to avoid modifying the original
            v[nan_mask] = 0.0
        
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            print(f"WARNING: Found {inf_count} infinite values in vector")
            v = np.copy(v)  # Make a copy to avoid modifying the original
            # Replace infinities with large finite values
            v[inf_mask & (v > 0)] = 1.0e+308
            v[inf_mask & (v < 0)] = -1.0e+308
        
        # Encode and send the vector
        binary_data = encode_vector(v, dtype=DType.FLOAT64, corr_id=corr_id)
        await ws.send_bytes(binary_data)
        print(f"Sent vector of length {a}")
    except Exception as e:
        print(f"Error writing vector: {type(e).__name__}: {str(e)}")
        raise

async def read_matrix(websocket: Union[WebSocket, Any]) -> np.ndarray:
    """
    Read a matrix from WebSocket using the R2P1 protocol
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    try:
        # Read binary message
        message = await ws.receive_bytes()
        
        # Parse using R2P1 protocol
        hdr, arr, err = decode_frame(message)
        
        if err:
            print(f"ERROR: Received error message: {err}")
            raise ValueError(f"Protocol error: {err}")
        
        if hdr.obj != Obj.MAT:
            print(f"ERROR: Expected matrix, but received {hdr.obj.name}")
            raise ValueError(f"Expected matrix, but received {hdr.obj.name}")
        
        if arr is None:
            print(f"ERROR: No data payload in matrix message")
            raise ValueError("No data payload in matrix message")
        
        print(f"Received matrix with shape {arr.shape}")
        return arr
        
    except Exception as e:
        print(f"Error reading matrix: {type(e).__name__}: {str(e)}")
        raise

async def write_matrix(m: np.ndarray, websocket: Union[WebSocket, Any]) -> None:
    """
    Write a matrix to WebSocket using the R2P1 protocol
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    try:
        # Static correlation ID (can be improved with a counter)
        corr_id = 2
        
        # Sanitize matrix before sending
        m = np.asarray(m)
        
        # Check that it's 2D
        if m.ndim != 2:
            print(f"ERROR: Input is not a 2D matrix, reshaping to 2D")
            if m.ndim == 1:
                m = m.reshape(-1, 1)  # Convert vector to column matrix
            else:
                # For higher dimensions, flatten to 2D
                m = m.reshape(m.shape[0], -1)
        
        a, b = m.shape
        
        # Check for problematic values
        nan_mask = np.isnan(m)
        inf_mask = np.isinf(m)
        
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            print(f"WARNING: Found {nan_count} NaN values in matrix, replacing with 0.0")
            m = np.copy(m)  # Make a copy to avoid modifying the original
            m[nan_mask] = 0.0
        
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            print(f"WARNING: Found {inf_count} infinite values in matrix")
            m = np.copy(m)  # Make a copy to avoid modifying the original
            # Replace infinities with large finite values
            m[inf_mask & (m > 0)] = 1.0e+308
            m[inf_mask & (m < 0)] = -1.0e+308
        
        # Encode and send the matrix
        binary_data = encode_matrix(m, dtype=DType.FLOAT64, corr_id=corr_id)
        await ws.send_bytes(binary_data)
        print(f"Sent matrix with shape {a}x{b}")
    except Exception as e:
        print(f"Error writing matrix: {type(e).__name__}: {str(e)}")
        raise

async def write_string(s: str, websocket: Union[WebSocket, Any]) -> None:
    """
    Write a string to WebSocket using the R2P1 protocol
    
    This function uses Cmd.PUT with a custom string payload.
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    # Handle empty string specially
    if not s:
        s = ""  # Ensure it's an empty string not None
    
    try:
        # Convert string to UTF-8 bytes
        s_bytes = s.encode('utf-8')
        s_len = len(s_bytes)
        
        # Static correlation ID for strings
        corr_id = 3
        
        # Create header with string length in rows field
        hdr = Header(MAGIC, VERSION, Cmd.PUT, Obj.VEC, DType.INT32, corr_id, s_len, 0)
        
        # Send header and data in one message
        combined = hdr.pack() + s_bytes
        
        # Debug info for commands
        if s in ["Information", "Initialize", "Impute", "End", "ping", "pong"]:
            print(f"Sending command: '{s}' ({s_len} bytes)")
        
        await ws.send_bytes(combined)
        
        return HEADER_SIZE + s_len
    except Exception as e:
        print(f"Error writing string '{s[:20]}...': {type(e).__name__}: {str(e)}")
        raise

async def read_string(websocket: Union[WebSocket, Any]) -> str:
    """
    Read a string from WebSocket using the R2P1 protocol
    
    This function expects a string encoded using Cmd.PUT with the string length in the rows field.
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    try:
        # Read the message
        message = await ws.receive_bytes()
        
        # Parse the header
        hdr = Header.unpack(message)
        payload = message[HEADER_SIZE:]
        
        # Check if it's an error message
        if hdr.cmd == Cmd.ERR:
            error_msg = payload.decode('utf-8', errors='replace') if payload else "Unknown error"
            print(f"ERROR: Received error message: {error_msg}")
            return error_msg
        
        # Verify we got the expected command type
        if hdr.cmd != Cmd.PUT:
            raise ValueError(f"Expected PUT command, but received {hdr.cmd}")
        
        # Verify string length
        s_len = hdr.rows
        actual_len = len(payload)
        
        if actual_len != s_len:
            print(f"WARNING: String length mismatch. Expected {s_len} bytes but got {actual_len} bytes")
        
        # Decode the string
        result = payload.decode('utf-8')
        
        # Log received command for debugging
        if result in ["Information", "Initialize", "Impute", "End", "ping", "pong"]:
            print(f"Received command: '{result}'")
        elif len(result) > 100:
            # Log truncated version for large strings
            print(f"Read string of length {len(result)} (first 20 chars: '{result[:20]}...')")
        
        return result
    except UnicodeDecodeError as e:
        print(f"Unicode decode error: {e}")
        raise
    except Exception as e:
        print(f"Error reading string: {type(e).__name__}: {str(e)}")
        raise

async def write_integer(i: int, websocket: Union[WebSocket, Any]) -> None:
    """
    Write an integer to WebSocket using R2P1 protocol
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    try:
        # Use INT32 data type with 1 element vector
        vec = np.array([i], dtype=np.int32)
        binary_data = encode_vector(vec, dtype=DType.INT32, corr_id=4)
        await ws.send_bytes(binary_data)
    except Exception as e:
        print(f"Error writing integer {i}: {type(e).__name__}: {str(e)}")
        raise

async def read_integer(websocket: Union[WebSocket, Any]) -> int:
    """
    Read an integer from WebSocket using R2P1 protocol
    """
    # Wrap the websocket if it's not already wrapped
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    
    try:
        # Read the message
        message = await ws.receive_bytes()
        
        # Parse using R2P1 protocol
        hdr, arr, err = decode_frame(message)
        
        if err:
            print(f"ERROR: Received error message: {err}")
            raise ValueError(f"Protocol error: {err}")
        
        if hdr.dtype != DType.INT32:
            print(f"ERROR: Expected INT32 data, but received {hdr.dtype}")
            raise ValueError(f"Expected INT32 data, but received {hdr.dtype}")
        
        if arr is None or len(arr) == 0:
            print(f"ERROR: No data payload in integer message")
            raise ValueError("No data payload in integer message")
        
        return int(arr[0])
    except Exception as e:
        print(f"Error reading integer: {type(e).__name__}: {str(e)}")
        raise

# Client-server helper functions for R2P1 protocol
async def client_put_vector(vec: np.ndarray, websocket: Union[WebSocket, Any]) -> None:
    """Client helper to send a vector"""
    await write_vector(vec, websocket)

async def client_put_matrix(mat: np.ndarray, websocket: Union[WebSocket, Any]) -> None:
    """Client helper to send a matrix"""
    await write_matrix(mat, websocket)

async def client_get_vector(websocket: Union[WebSocket, Any], hint_size: int = 0) -> np.ndarray:
    """Client helper to request and receive a vector"""
    # Send GET request for vector
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    req = encode_get(Obj.VEC, dtype=DType.FLOAT64, corr_id=5, rows=hint_size)
    await ws.send_bytes(req)
    # Wait for response
    return await read_vector(websocket)

async def client_get_matrix(websocket: Union[WebSocket, Any], hint_rows: int = 0, hint_cols: int = 0) -> np.ndarray:
    """Client helper to request and receive a matrix"""
    # Send GET request for matrix
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    req = encode_get(Obj.MAT, dtype=DType.FLOAT64, corr_id=6, rows=hint_rows, cols=hint_cols)
    await ws.send_bytes(req)
    # Wait for response
    return await read_matrix(websocket)

async def handle_r2p1_message(websocket: Union[WebSocket, Any], handler_map: dict) -> None:
    """
    Handle an incoming R2P1 message and dispatch to appropriate handler
    
    handler_map should be a dictionary mapping from (cmd, obj) tuples to handler functions
    Each handler function should accept (header, data, websocket) parameters
    """
    hdr, data, err = await ws_recv(websocket)
    
    # Check for error messages
    if err:
        print(f"R2P1 ERROR: {err}")
        return
    
    # Find appropriate handler
    handler_key = (hdr.cmd, hdr.obj)
    if handler_key in handler_map:
        await handler_map[handler_key](hdr, data, websocket)
    else:
        print(f"No handler for message type: {hdr.cmd.name}, {hdr.obj.name}")
        # Send error response
        err_msg = f"No handler for {hdr.cmd.name}/{hdr.obj.name}"
        await ws_send(websocket, encode_err(hdr.corr_id, err_msg))
