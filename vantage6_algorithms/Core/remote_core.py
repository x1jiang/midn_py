"""Core remote utilities for MIDN algorithms.

This module provides shared functionality for remote client implementations
across different algorithms like SIMI and SIMICE.

Key features:
- RemoteClient base class with persistent WebSocket connection loop
- Pluggable method handlers (e.g., 'gaussian', 'logistic', 'initialize', ...)
- Default 'ping' and 'end' handlers
- Common parameter validation for mvar and method
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Callable, Dict, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from .transfer import (
    read_string, write_string,
    WebSocketWrapper, get_wrapped_websocket,
)


class RemoteClient:
    """Base class for remote algorithm clients.

    Subclasses should:
    - implement prepare_data(self, mvar: int)
    - register algorithm-specific handlers using self.register_handler(name, coro)

    Handler signature: async def handler(websocket: WebSocketWrapper) -> bool
        Return True to stop the client, False to continue.
    """

    def __init__(
        self,
        data: Union[str, np.ndarray],
        central_host: str,
        central_port: int,
        central_proto: str,
        site_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.site_id = site_id
        self.central_host = central_host
        self.central_port = int(central_port)
        self.central_proto = central_proto or "ws"
        self.parameters = parameters or {}
        self.data = self._load_data(data)
        self.method_handlers: Dict[str, Callable[[WebSocketWrapper], asyncio.Future]] = {}
        self._register_default_handlers()

    # ------------------------------------------------------------
    # Setup and registration
    # ------------------------------------------------------------
    def _load_data(self, data: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(data, str):
            return pd.read_csv(data).values
        return np.asarray(data)

    def _register_default_handlers(self) -> None:
        self.register_handler("ping", self._handle_ping)
        self.register_handler("end", self._handle_end)

    def register_handler(self, method: str, handler: Callable[[WebSocketWrapper], Any]) -> None:
        self.method_handlers[method.lower()] = handler

    # ------------------------------------------------------------
    # Default handlers
    # ------------------------------------------------------------
    async def _handle_ping(self, websocket: WebSocketWrapper) -> bool:
        await write_string("pong", websocket)
        return False

    async def _handle_end(self, websocket: WebSocketWrapper) -> bool:
        # By default, persist after End to allow reuse (can be changed via env by subclass if desired)
        # Keeping behavior consistent with existing remotes.
        return False

    # ------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------
    async def prepare_data(self, mvar: int) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------
    # Core connection loop
    # ------------------------------------------------------------
    async def remote_kernel(self) -> None:
        url = f"{self.central_proto}://{self.central_host}:{self.central_port}/ws/{self.site_id}"
        backoff = 1
        max_backoff = 30

        while True:
            try:
                print(f"[{self.site_id}] Connecting to {url}", flush=True)
                async with websockets.connect(url, ping_interval=None) as ws:
                    wrapped = get_wrapped_websocket(ws)
                    print(f"[{self.site_id}] Connected", flush=True)
                    backoff = 1

                    while True:
                        try:
                            method = await read_string(wrapped)
                        except asyncio.TimeoutError:
                            try:
                                await write_string("ping", wrapped)
                            except Exception:
                                raise
                            continue

                        if method is None:
                            print(f"[{self.site_id}] Null method (disconnect?)", flush=True)
                            break

                        m = method.lower()
                        handler = self.method_handlers.get(m)
                        if handler:
                            try:
                                should_exit = await handler(wrapped)
                                if should_exit:
                                    return
                            except Exception as e:
                                print(f"[{self.site_id}] Error in handler for {m}: {e}", flush=True)
                                traceback.print_exc()
                        else:
                            print(f"[{self.site_id}] Unknown method '{method}'", flush=True)

            except asyncio.CancelledError:
                print(f"[{self.site_id}] Cancelled - shutting down remote kernel", flush=True)
                return
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                print(f"[{self.site_id}] Connection closed: {type(e).__name__} - retry in {backoff}s", flush=True)
            except (ConnectionRefusedError, OSError) as e:
                msg = str(e).splitlines()[0]
                print(f"[{self.site_id}] Central not ready (will retry): {msg} | next attempt in {backoff}s", flush=True)
            except Exception as e:
                print(f"[{self.site_id}] Kernel error: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                print(f"[{self.site_id}] Reconnecting in {backoff}s", flush=True)

            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)

    async def run(self) -> None:
        try:
            await self.remote_kernel()
        except asyncio.CancelledError:
            print(f"[{self.site_id}] Cancelled - task terminated", flush=True)
            raise


# ------------------------------------------------------------
# Common utilities
# ------------------------------------------------------------
def validate_parameters(parameters: Dict[str, Any], data_shape: Tuple[int, int]) -> Tuple[int, str]:
    """Validate and extract (mvar_0based, method) from parameters.

    parameters:
      - target_column_index (preferred) OR mvar (legacy), 1-based
      - method: optional ('gaussian'|'logistic'), inferred from is_binary if absent
      - is_binary: optional bool for inference
    """
    params = parameters or {}
    n_cols = int(data_shape[1])

    if "target_column_index" in params:
        try:
            mvar_1 = int(params["target_column_index"])  # 1-based
        except Exception as e:
            raise ValueError(f"Invalid target_column_index value: {params.get('target_column_index')} ({e})")
    elif "mvar" in params:
        try:
            mvar_1 = int(params["mvar"])  # 1-based
        except Exception as e:
            raise ValueError(f"Invalid mvar value: {params.get('mvar')} ({e})")
    else:
        raise ValueError("Parameters must include 'target_column_index' or 'mvar' (1-based)")

    if not (1 <= mvar_1 <= n_cols):
        raise ValueError(f"mvar/target_column_index {mvar_1} out of bounds for data with {n_cols} columns (1..{n_cols})")

    mvar_py = mvar_1 - 1

    method = params.get("method")
    if method is None:
        is_binary = bool(params.get("is_binary", False))
        method = "logistic" if is_binary else "gaussian"
    method = str(method).lower()
    if method not in ("gaussian", "logistic"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'gaussian' or 'logistic'.")

    return mvar_py, method


async def run_remote_client_async(
    algorithm_class,
    data: Union[str, np.ndarray],
    central_host: str,
    central_port: int,
    central_proto: str,
    site_id: str,
    parameters: Dict[str, Any],
) -> None:
    """Generic async runner that constructs and runs the algorithm client."""
    client = algorithm_class(
        data=data,
        central_host=central_host,
        central_port=central_port,
        central_proto=central_proto,
        site_id=site_id,
        parameters=parameters,
    )
    await client.run()


__all__ = [
    "RemoteClient",
    "validate_parameters",
    "run_remote_client_async",
]
