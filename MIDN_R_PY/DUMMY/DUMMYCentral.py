"""DUMMY central algorithm template

This module demonstrates the minimal signature and lifecycle expected for a
central-side algorithm in this project. Use it as a starting point when adding
new algorithms.

Contract (central side):
- Expose an async entrypoint: dummy_central(D, config, site_ids, websockets, debug)
- Accepts an n x p numpy array D (with NaNs) and returns a list of M imputed
  datasets (np.ndarray). This DUMMY version simply mean-imputes locally and
  does not depend on remote sites, but shows how to integrate with the shared
  WebSocket registry when needed (Initialize/End messages).

Globals used by the central runtime:
- remote_websockets: Dict[str, WebSocket]
- site_locks: Dict[str, asyncio.Lock]
- imputation_running: asyncio.Event

Wire-format conventions (if interacting with remotes):
- Column indices sent to remotes are 1-based.
- Scalars may be sent as length-1 vectors for consistency.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket

# Optional transport helpers if you want to message remotes in this template
from Core.transfer import (
	write_string,
	write_vector,
	get_wrapped_websocket,
)

# ---- Shared central runtime state (provided by outer central server) ----
remote_websockets: Dict[str, WebSocket] = {}
site_locks: Dict[str, asyncio.Lock] = {}
imputation_running = asyncio.Event()


def _parse_index_list(val: Any) -> List[int]:
	"""Parse an index list that could be 0-based list, 1-based list, scalar, or CSV string."""
	if val is None:
		return []
	if isinstance(val, (list, tuple, np.ndarray)):
		arr = list(map(int, list(val)))
	elif isinstance(val, str):
		parts = [p.strip() for p in val.split(',') if p.strip()]
		arr = list(map(int, parts))
	else:
		arr = [int(val)]
	return arr


def _normalize_config(config: Optional[dict]) -> dict:
	"""Minimal config normalization for demonstration.

	Accepts either:
	  - mvar: int or list[int] (0-based by default; set one_based=True to treat as 1-based)
	  - target_column_index or target_column_indexes: 1-based values
	Other keys:
	  - M or imputation_trials: number of output datasets
	"""
	raw = dict(config or {})
	norm: Dict[str, Any] = {}

	# Targets
	if 'mvar' in raw:
		mvar = _parse_index_list(raw['mvar'])
		if raw.get('one_based'):
			mvar = [i - 1 for i in mvar]
		norm['mvar'] = mvar
	elif 'target_column_index' in raw:
		norm['mvar'] = [int(raw['target_column_index']) - 1]
	elif 'target_column_indexes' in raw:
		mvar = _parse_index_list(raw['target_column_indexes'])
		norm['mvar'] = [i - 1 for i in mvar]
	else:
		# For a dummy example, default to no specific targets
		norm['mvar'] = []

	# Number of imputations
	if 'M' in raw:
		norm['M'] = int(raw['M'])
	else:
		norm['M'] = int(raw.get('imputation_trials', 1))

	return norm


async def _initialize_remote_sites(mvar_1based: List[int], site_ids: List[str]) -> None:
	"""Send an optional Initialize + mvar vector to remotes (no-op if none)."""
	if not site_ids:
		return
	for sid in list(site_ids):
		if sid not in remote_websockets:
			continue
		async with site_locks[sid]:
			ws = get_wrapped_websocket(remote_websockets[sid], pre_accepted=True)
			await write_string("Initialize", ws)
			await write_vector(np.array(mvar_1based, dtype=float), ws)


async def _finalize_remote_sites(site_ids: List[str]) -> None:
	if not site_ids:
		return
	for sid in list(site_ids):
		if sid not in remote_websockets:
			continue
		try:
			async with site_locks[sid]:
				ws = get_wrapped_websocket(remote_websockets[sid], pre_accepted=True)
				await write_string("End", ws)
		except Exception:
			# Ignore teardown issues in a dummy template
			pass


async def dummy_central(
	D: np.ndarray,
	config: Optional[dict] = None,
	site_ids: Optional[List[str]] = None,
	websockets: Optional[Dict[str, WebSocket]] = None,
	debug: bool = False,
) -> List[np.ndarray]:
	"""Minimal central entrypoint for an algorithm.

	Args:
	  D: np.ndarray (n x p) data matrix with possible NaNs.
	  config: dict with optional keys { 'M', 'mvar'|'target_column_indexes', 'one_based' }.
	  site_ids: list of remote site IDs to include (optional).
	  websockets: mapping of site_id -> WebSocket (provided by the central server).
	  debug: enable verbose prints.

	Returns: list of imputed datasets (length M). This DUMMY version performs
			 simple mean imputation locally and demonstrates the expected
			 control flow, globals, and WS initialization/finalization.
	"""
	cfg = _normalize_config(config)
	M = int(cfg['M'])
	mvar = list(cfg.get('mvar', []))

	# Provide websockets/locks (managed by central runtime)
	global remote_websockets, site_locks
	if websockets is not None:
		remote_websockets = websockets
	site_ids = list(site_ids or [])
	for sid in site_ids:
		site_locks.setdefault(sid, asyncio.Lock())

	# Derive active sites (optional)
	active_sites = [sid for sid in site_ids if sid in remote_websockets]
	if not active_sites and remote_websockets and site_ids:
		# Requested sites not connected; continue locally in this dummy
		if debug:
			print(f"[DUMMY] Requested sites not connected; running locally.", flush=True)
	if debug:
		print(f"[DUMMY] Active sites: {active_sites}", flush=True)

	# Start run
	imputation_running.set()
	start = time.time()

	# Optional seed for determinism across runs
	seed_env = os.getenv("GLOBAL_SEED") or os.getenv("SIMICE_GLOBAL_SEED")
	if seed_env:
		try:
			np.random.seed(int(seed_env))
		except Exception:
			pass
	else:
		random.seed()

	# Inform remotes which columns are targets (if any connected and provided)
	if active_sites and mvar:
		if debug:
			print(f"[DUMMY] Initializing remotes with targets (1-based) {[j+1 for j in mvar]}", flush=True)
		await _initialize_remote_sites([j + 1 for j in mvar], active_sites)

	try:
		imputations: List[np.ndarray] = []
		# Minimal local mean-imputation per output copy
		col_means = np.nanmean(D, axis=0)
		col_means = np.where(np.isnan(col_means), 0.0, col_means)  # if an all-NaN column
		for m in range(1, M + 1):
			D_imp = np.array(D, dtype=float, copy=True)
			inds = np.where(np.isnan(D_imp))
			if inds[0].size:
				D_imp[inds] = np.take(col_means, inds[1])
			imputations.append(D_imp)
		return imputations
	finally:
		# Clean remote sessions if we sent Initialize
		await _finalize_remote_sites(active_sites)
		imputation_running.clear()
		if debug:
			dur = time.time() - start
			print(f"[DUMMY] Completed {M} imputations in {dur:.3f}s", flush=True)

