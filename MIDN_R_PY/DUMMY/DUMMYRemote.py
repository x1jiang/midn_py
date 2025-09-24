"""DUMMY remote algorithm template

This module demonstrates the minimal structure for a remote-side algorithm
client. It uses Core.remote_core.RemoteClient to register message handlers and
interact with the central server over JSON-only WebSockets via Core.transfer.

Protocol shown:
- initialize: central may send a vector of 1-based target column indices.
- information: central may ask for stats for a given target (method + j).
- impute: central may send parameters to update local missing entries.

This DUMMY version does no heavy computation. It logs messages and performs
simple mean imputation during the "impute" phase for demonstration.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from Core.transfer import (
	write_string, write_vector,
	read_string, read_integer, read_vector,
	WebSocketWrapper,
)
from Core.remote_core import RemoteClient, run_remote_client_async


class DUMMYRemoteClient(RemoteClient):
	"""Minimal RemoteClient for demonstration."""

	def __init__(self, data, central_host, central_port, central_proto, site_id, parameters=None):
		super().__init__(data, central_host, central_port, central_proto, site_id, parameters)

		# State
		self.D: Optional[np.ndarray] = None   # full data (n x p)
		self.mvar_list: List[int] = []        # 0-based target columns
		self.col_means: Optional[np.ndarray] = None

		# Handlers to illustrate minimal verbs
		self.register_handler("initialize", self.handle_initialize)
		self.register_handler("information", self.handle_information)
		self.register_handler("impute", self.handle_impute)

	async def prepare_data(self, _mvar: int) -> None:
		# Convert input to float numpy array; compute simple column means
		self.D = np.asarray(self.data, dtype=float)
		with np.errstate(all='ignore'):
			self.col_means = np.nanmean(self.D, axis=0)
		if self.col_means is not None:
			self.col_means = np.where(np.isnan(self.col_means), 0.0, self.col_means)
		print(f"[{self.site_id}] DUMMY prepared data shape={self.D.shape}", flush=True)

	async def handle_initialize(self, websocket: WebSocketWrapper) -> bool:
		# Receive vector of 1-based target indices (optional)
		vec = await read_vector(websocket)
		self.mvar_list = [int(x) - 1 for x in vec]
		print(f"[{self.site_id}] initialize targets(0-based)={self.mvar_list}", flush=True)
		return False

	async def handle_information(self, websocket: WebSocketWrapper) -> bool:
		# Read method and 1-based target index (compat with other algos)
		method = (await read_string(websocket)).lower()
		j_1b = await read_integer(websocket)
		j = int(j_1b) - 1
		# For DUMMY, just reply with placeholders: send back the mean as a 1-length vector
		mu = float(self.col_means[j]) if self.col_means is not None else 0.0
		await write_vector(np.array([mu], dtype=float), websocket)
		print(f"[{self.site_id}] information method={method} j={j} -> mean={mu}", flush=True)
		return False

	async def handle_impute(self, websocket: WebSocketWrapper) -> bool:
		# Read method and 1-based target index, then fill NaNs with column mean
		method = (await read_string(websocket)).lower()
		j_1b = await read_integer(websocket)
		j = int(j_1b) - 1
		D = self.D
		if D is None:
			return False
		if self.col_means is None:
			with np.errstate(all='ignore'):
				self.col_means = np.nanmean(D, axis=0)
			if self.col_means is not None:
				self.col_means = np.where(np.isnan(self.col_means), 0.0, self.col_means)
		mu = float(self.col_means[j]) if self.col_means is not None else 0.0
		midx = np.where(np.isnan(D[:, j]))[0]
		if midx.size:
			D[midx, j] = mu
		print(f"[{self.site_id}] impute method={method} j={j} n_filled={midx.size}", flush=True)
		return False

	async def run(self) -> None:
		await self.prepare_data(0)
		job_id = (self.parameters or {}).get("job_id", "unknown")
		print(f"[async:{self.site_id}] Starting DUMMY remote job_id={job_id}", flush=True)
		await super().run()


def run_remote_client(data, central_host, central_port, central_proto, site_id=None, remote_port=None, config=None):
	# Legacy-compatible entry point
	if isinstance(data, str):
		D = pd.read_csv(data).values
	else:
		D = data
	site_id = site_id or "remote1"
	if remote_port is not None:
		print(f"[{site_id}] Ignoring remote_port={remote_port} (no local server)", flush=True)
	print(f"[{site_id}] Starting DUMMY remote with data shape={np.asarray(D).shape}", flush=True)
	asyncio.run(async_run_remote_client(D, central_host, central_port, central_proto, site_id, config or {}))


async def async_run_remote_client(data, central_host, central_port, central_proto, site_id=None, config=None):
	if isinstance(data, str):
		D = pd.read_csv(data).values
	else:
		D = data
	site_id = site_id or "remote1"
	print(f"[async:{site_id}] Loaded data shape={np.asarray(D).shape}", flush=True)
	await run_remote_client_async(
		DUMMYRemoteClient,
		data=D,
		central_host=central_host,
		central_port=central_port,
		central_proto=central_proto,
		site_id=site_id,
		parameters=config or {},
	)

