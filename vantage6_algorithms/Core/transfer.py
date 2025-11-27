# transfer.py
# HIGHLIGHT: JSON-only transport with base64 payloads. All WebSocket frames are TEXT (strings).

import json  # HIGHLIGHT: JSON envelopes for all messages
import base64  # HIGHLIGHT: base64 payloads for numeric/binary data
import numpy as np
import enum
from dataclasses import dataclass
from typing import Union, Any, Optional
from fastapi import WebSocket
from datetime import datetime

# ================================================================
# WebSocket Wrapper (TEXT-only I/O)
# ================================================================

class WebSocketWrapper:
    """
    Consistent interface for FastAPI WebSocket (server) and websockets (client).
    HIGHLIGHT: Only TEXT frames are used; all data is JSON-encoded strings.
    """
    # set pre_accepted=True if the FastAPI WebSocket has already been accepted externally
    def __init__(self, websocket, pre_accepted=True):
        self.websocket = websocket
        self.ws_type = self._determine_websocket_type(websocket)
        self.is_active = True
        self._accepted = pre_accepted  # HIGHLIGHT: track if already accepted externally

    def _determine_websocket_type(self, ws):
        if hasattr(ws, "send_text") and callable(ws.send_text):
            return "fastapi"
        if hasattr(ws, "send") and callable(ws.send) and hasattr(ws, "recv"):
            return "websockets"
        return "unknown"

    async def _ensure_accept(self):
        # HIGHLIGHT: server-side FastAPI WS must be accepted before I/O
        if self.ws_type == "fastapi" and not self._accepted:
            await self.websocket.accept()
            self._accepted = True

    async def send_text(self, data: str) -> None:
        if not self.is_active:
            return
        try:
            await self._ensure_accept()
            if self.ws_type == "fastapi":
                await self.websocket.send_text(data)
            else:
                await self.websocket.send(data)  # websockets: text is str
        except Exception:
            self.is_active = False
            raise

    async def receive_text(self) -> str:
        if not self.is_active:
            return ""
        try:
            await self._ensure_accept()
            if self.ws_type == "fastapi":
                return await self.websocket.receive_text()
            else:
                msg = await self.websocket.recv()
                return msg.decode("utf-8") if isinstance(msg, (bytes, bytearray)) else msg
        except Exception:
            self.is_active = False
            raise

    async def close(self, code: int = 1000) -> None:
        try:
            await self.websocket.close(code=code)
        finally:
            self.is_active = False


def get_wrapped_websocket(websocket, pre_accepted=True):
    """HIGHLIGHT: Factory; JSON-only wrapper.
    
    Args:
        websocket: The WebSocket object to wrap
        pre_accepted: Whether the FastAPI WebSocket has already been accepted
    """
    return websocket if isinstance(websocket, WebSocketWrapper) else WebSocketWrapper(websocket, pre_accepted=pre_accepted)


# ================================================================
# JSON Protocol (R2P1J)
# ================================================================

MAGIC = "R2P1J"  # HIGHLIGHT: JSON protocol magic
VERSION = 1

class Cmd(str, enum.Enum):
    PUT = "PUT"
    GET = "GET"
    ACK = "ACK"
    ERR = "ERR"

class Obj(str, enum.Enum):
    VEC = "VEC"
    MAT = "MAT"
    STR = "STR"
    BIN = "BIN"   # reserved: raw bytes as base64 if needed later
    HB  = "HB"    # reserved: heartbeat envelope
    NUM = "NUM"   # HIGHLIGHT: scalar number

class DType(str, enum.Enum):
    INT32   = "int32"
    FLOAT64 = "float64"
    UTF8    = "utf8"
    BYTES   = "bytes"

# Envelope schema (all messages are TEXT JSON):
# {
#   "magic": "R2P1J", "ver": 1,
#   "cmd": "PUT"|"GET"|"ACK"|"ERR",
#   "obj": "VEC"|"MAT"|"STR"|"BIN"|"HB"|"NUM",
#   "corr": int,
#   "dtype": "int32"|"float64"|"utf8"|"bytes",
#   "rows": int, "cols": int,
#   "payload": base64-string (for VEC/MAT/BIN/NUM),
#   "text": string (for STR),
#   "error": string (for ERR),
#   "ts": int (for HB)
# }

@dataclass
class Envelope:
    magic: str
    ver: int
    cmd: Cmd
    obj: Obj
    corr: int
    dtype: Optional[DType] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    payload: Optional[str] = None  # base64
    text: Optional[str] = None
    error: Optional[str] = None
    ts: Optional[int] = None

    def to_json(self) -> str:
        # HIGHLIGHT: compact JSON to minimize bandwidth
        d = {
            "magic": self.magic, "ver": self.ver,
            "cmd": self.cmd.value, "obj": self.obj.value, "corr": self.corr
        }
        if self.dtype is not None: d["dtype"] = self.dtype.value
        if self.rows is not None:  d["rows"] = self.rows
        if self.cols is not None:  d["cols"] = self.cols
        if self.payload is not None: d["payload"] = self.payload
        if self.text is not None:    d["text"] = self.text
        if self.error is not None:   d["error"] = self.error
        if self.ts is not None:      d["ts"] = self.ts
        return json.dumps(d, separators=(",", ":"))

    @staticmethod
    def from_json(s: str) -> "Envelope":
        o = json.loads(s)
        if o.get("magic") != MAGIC:
            raise ValueError("bad magic")
        if int(o.get("ver", -1)) != VERSION:
            raise ValueError("unsupported version")
        return Envelope(
            magic=o["magic"],
            ver=int(o["ver"]),
            cmd=Cmd(o["cmd"]),
            obj=Obj(o["obj"]),
            corr=int(o.get("corr", 0)),
            dtype=DType(o["dtype"]) if "dtype" in o else None,
            rows=int(o["rows"]) if "rows" in o else None,
            cols=int(o["cols"]) if "cols" in o else None,
            payload=o.get("payload"),
            text=o.get("text"),
            error=o.get("error"),
            ts=int(o["ts"]) if "ts" in o else None,
        )


# ================================================================
# Encoding helpers (numeric/binary -> base64)
# ================================================================

def _np_dtype_for(dt: DType):
    if dt == DType.INT32:
        return np.dtype(">i4")  # big-endian int32
    if dt == DType.FLOAT64:
        return np.dtype(">f8")  # big-endian float64
    raise ValueError(f"Unsupported numeric dtype {dt}")

def _b64_encode(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64_decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


# ================================================================
# Public API (JSON-only I/O)
# ================================================================

async def write_vector(v: np.ndarray, websocket: Union[WebSocket, Any]) -> None:
    # HIGHLIGHT: send JSON with base64 numeric payload (FLOAT64)
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    v = np.asarray(v)
    # sanitize NaN/Inf
    nan_mask, inf_mask = np.isnan(v), np.isinf(v)
    if np.any(nan_mask) or np.any(inf_mask):
        v = np.copy(v)
        if np.any(nan_mask):
            v[nan_mask] = 0.0
        if np.any(inf_mask):
            v[inf_mask & (v > 0)] = 1.0e+308
            v[inf_mask & (v < 0)] = -1.0e+308
    be = v.astype(_np_dtype_for(DType.FLOAT64), copy=False).tobytes(order="C")
    env = Envelope(
        magic=MAGIC, ver=VERSION, cmd=Cmd.PUT, obj=Obj.VEC, corr=1,
        dtype=DType.FLOAT64, rows=v.shape[0], cols=1, payload=_b64_encode(be)
    )
    await ws.send_text(env.to_json())

async def read_vector(websocket: Union[WebSocket, Any]) -> np.ndarray:
    """Receive a FLOAT64 vector envelope, tolerating limited stray frames.

    Race condition context: A late heartbeat / ping (STR/HB) frame or a
    reconnect-induced status string can arrive just before the numeric
    payload the remote expects (e.g., mvar list). Previously we only
    tolerated exactly one stray frame which was occasionally insufficient
    when two quick administrative frames appeared (e.g., 'ping', 'pong').

    Strategy: Allow up to MAX_STRAY non-error frames that are clearly not
    the expected VEC/FLOAT64 numeric payload. Skip silently unless they
    are an ERR frame (which is raised immediately). After exceeding the
    stray allowance we raise the original descriptive error.
    """
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    stray_count = 0
    MAX_STRAY = 3  # small safety margin; keeps tight detection while robust
    while True:
        txt = await ws.receive_text()
        env = Envelope.from_json(txt)
        if env.cmd == Cmd.ERR:
            raise ValueError(env.error or "Protocol error")
        if env.obj == Obj.VEC and env.dtype == DType.FLOAT64:
            if env.payload is None or env.rows is None:
                raise ValueError("Missing payload/rows for vector")
            raw = _b64_decode(env.payload)
            return np.frombuffer(raw, dtype=_np_dtype_for(DType.FLOAT64), count=env.rows)
        # Skip benign stray frames (e.g., STR/HB/NUM) up to MAX_STRAY
        if stray_count < MAX_STRAY and env.obj in {Obj.STR, Obj.HB, Obj.NUM}:
            stray_count += 1
            continue
        raise ValueError(f"Expected VEC/float64, got {env.obj}/{env.dtype}")

async def write_matrix(m: np.ndarray, websocket: Union[WebSocket, Any]) -> None:
    # HIGHLIGHT: send JSON with base64 numeric payload (FLOAT64, 2D)
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    m = np.asarray(m)
    if m.ndim != 2:
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        else:
            m = m.reshape(m.shape[0], -1)
    nan_mask, inf_mask = np.isnan(m), np.isinf(m)
    if np.any(nan_mask) or np.any(inf_mask):
        m = np.copy(m)
        if np.any(nan_mask):
            m[nan_mask] = 0.0
        if np.any(inf_mask):
            m[inf_mask & (m > 0)] = 1.0e+308
            m[inf_mask & (m < 0)] = -1.0e+308
    rows, cols = m.shape
    be = m.astype(_np_dtype_for(DType.FLOAT64), copy=False).tobytes(order="C")
    env = Envelope(
        magic=MAGIC, ver=VERSION, cmd=Cmd.PUT, obj=Obj.MAT, corr=2,
        dtype=DType.FLOAT64, rows=rows, cols=cols, payload=_b64_encode(be)
    )
    await ws.send_text(env.to_json())

async def read_matrix(websocket: Union[WebSocket, Any]) -> np.ndarray:
    # HIGHLIGHT: receive JSON and decode base64 payload into 2D np.ndarray
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    txt = await ws.receive_text()
    env = Envelope.from_json(txt)
    if env.cmd == Cmd.ERR:
        raise ValueError(env.error or "Protocol error")
    if env.obj != Obj.MAT or env.dtype != DType.FLOAT64:
        raise ValueError(f"Expected MAT/float64, got {env.obj}/{env.dtype}")
    if env.payload is None or env.rows is None or env.cols is None:
        raise ValueError("Missing payload/shape for matrix")
    raw = _b64_decode(env.payload)
    arr = np.frombuffer(raw, dtype=_np_dtype_for(DType.FLOAT64), count=env.rows * env.cols)
    return arr.reshape((env.rows, env.cols))

async def write_string(s: str, websocket: Union[WebSocket, Any]) -> None:
    # HIGHLIGHT: strings are plain JSON field ("text") in the envelope
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    env = Envelope(
        magic=MAGIC, ver=VERSION, cmd=Cmd.PUT, obj=Obj.STR, corr=3,
        dtype=DType.UTF8, text=(s or "")
    )
    # print(f"######[{datetime.now()}] write_string: {s}", flush=True)
    await ws.send_text(env.to_json())

async def read_string(websocket: Union[WebSocket, Any]) -> str:
    # HIGHLIGHT: read JSON "text"
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    txt = await ws.receive_text()
    env = Envelope.from_json(txt)
    if env.cmd == Cmd.ERR:
        raise ValueError(env.error or "Protocol error")
    if env.obj != Obj.STR or env.dtype != DType.UTF8:
        raise ValueError(f"Expected STR/utf8, got {env.obj}/{env.dtype}")
    # print(f"######[{datetime.now()}] read_string: {env.text}", flush=True)
    return env.text or ""

async def write_integer(i: int, websocket: Union[WebSocket, Any]) -> None:
    # HIGHLIGHT: integers sent as 1-element INT32 vector
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    vec = np.array([int(i)], dtype=_np_dtype_for(DType.INT32))
    payload_b64 = _b64_encode(vec.tobytes(order="C"))
    env = Envelope(
        magic=MAGIC, ver=VERSION, cmd=Cmd.PUT, obj=Obj.VEC, corr=4,
        dtype=DType.INT32, rows=1, cols=1, payload=payload_b64
    )
    await ws.send_text(env.to_json())

async def read_integer(websocket: Union[WebSocket, Any]) -> int:
    # HIGHLIGHT: receive JSON and decode base64 payload into integer
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    txt = await ws.receive_text()
    env = Envelope.from_json(txt)
    if env.cmd == Cmd.ERR:
        raise ValueError(env.error or "Protocol error")
    if env.obj != Obj.VEC or env.dtype != DType.INT32:
        raise ValueError(f"Expected VEC/int32, got {env.obj}/{env.dtype}")
    if env.payload is None or env.rows is None or env.rows != 1:
        raise ValueError("Missing payload or invalid rows for integer")
    raw = _b64_decode(env.payload)
    arr = np.frombuffer(raw, dtype=_np_dtype_for(DType.INT32), count=1)
    return int(arr[0])


# HIGHLIGHT: scalar number support (int32 if in range, else float64)
async def write_number(x: Union[int, float], websocket: Union[WebSocket, Any]) -> None:
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)

    use_int32 = isinstance(x, (int, np.integer)) and not isinstance(x, bool)
    if use_int32:
        if x < -2147483648 or x > 2147483647:
            use_int32 = False

    if use_int32:
        dt = DType.INT32  # HIGHLIGHT
        arr = np.array([int(x)], dtype=_np_dtype_for(dt))
    else:
        dt = DType.FLOAT64  # HIGHLIGHT
        arr = np.array([float(x)], dtype=_np_dtype_for(dt))

    payload_b64 = _b64_encode(arr.tobytes(order="C"))
    env = Envelope(
        magic=MAGIC, ver=VERSION, cmd=Cmd.PUT, obj=Obj.NUM, corr=7,  # HIGHLIGHT
        dtype=dt, rows=1, cols=1, payload=payload_b64
    )
    await ws.send_text(env.to_json())

# HIGHLIGHT: read scalar number (returns int for INT32, float for FLOAT64)
async def read_number(websocket: Union[WebSocket, Any]) -> Union[int, float]:
    ws = websocket if isinstance(websocket, WebSocketWrapper) else get_wrapped_websocket(websocket)
    txt = await ws.receive_text()
    env = Envelope.from_json(txt)

    if env.cmd == Cmd.ERR:
        raise ValueError(env.error or "Protocol error")

    if env.obj != Obj.NUM or env.dtype not in (DType.INT32, DType.FLOAT64):
        raise ValueError(f"Expected NUM/(int32|float64), got {env.obj}/{env.dtype}")

    if env.payload is None or env.rows != 1:
        raise ValueError("Missing or invalid payload for number")

    raw = _b64_decode(env.payload)
    if env.dtype == DType.INT32:  # HIGHLIGHT
        val = np.frombuffer(raw, dtype=_np_dtype_for(DType.INT32), count=1)[0]
        return int(val)
    else:
        val = np.frombuffer(raw, dtype=_np_dtype_for(DType.FLOAT64), count=1)[0]
        return float(val)


# ================================================================
# Explicit public surface
# ================================================================

__all__ = [
    "read_matrix", "write_matrix", "read_vector", "write_vector",
    "read_string", "write_string", "read_integer", "write_integer",
    "read_number", "write_number",  # HIGHLIGHT: new scalar number APIs
    "WebSocketWrapper", "get_wrapped_websocket",
]
