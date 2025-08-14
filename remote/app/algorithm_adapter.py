"""
ADAPTER LAYER - Legacy entry point for algorithm execution.

This module serves as an adapter between the old code structure and the new
service-oriented architecture. It provides backward compatibility for existing
code that calls these functions.

For new code, please use the algorithm services directly:
    from remote.app.services.algorithm_factory import AlgorithmClientFactory
    client = AlgorithmClientFactory.create_client("ALGORITHM_NAME")
    await client.run_algorithm(...)
"""

import websockets
import asyncio
import numpy as np
import json
import pandas as pd

from remote.app.services.algorithm_factory import AlgorithmClientFactory
from common.algorithm.registry import AlgorithmRegistry

# Import service initialization to ensure all algorithms and clients are registered
import remote.app.services.init_clients


async def run_simi_remote(data: np.ndarray, mvar: int, job_id: int, site_id: str,
                        central_url: str, token: str, is_binary: bool = False,
                        extra_params=None, status_callback=None):
    """
    LEGACY ADAPTER - Run the SIMI algorithm remotely.
    
    This function exists for backward compatibility. New code should use
    the SIMI client service directly.
    
    Args:
        data: Data array
        mvar: Index of the target column
        job_id: ID of the job
        site_id: ID of this site
        central_url: URL of the central server
        token: Authentication token
        is_binary: Whether to use logistic regression (binary outcome)
        extra_params: Additional parameters for the algorithm
        status_callback: Callback for status updates
    """
    try:
        # Prepare extra parameters
        params = extra_params or {}
        params["is_binary"] = is_binary
        
        # Create the SIMI client
        client = AlgorithmClientFactory.create_client("SIMI")
        
        # Run the algorithm
        await client.run_algorithm(
            data=data,
            target_column=mvar,
            job_id=job_id,
            site_id=site_id,
            central_url=central_url,
            token=token,
            extra_params=params,
            status_callback=status_callback
        )
        
    except Exception as e:
        if status_callback:
            await status_callback.on_error(f"Error running SIMI algorithm: {str(e)}")
        raise


async def run_simice_remote(data: np.ndarray, target_column_indexes: list, job_id: int, site_id: str,
                           central_url: str, token: str, is_binary: list = None,
                           extra_params=None, status_callback=None):
    """
    LEGACY ADAPTER - Run the SIMICE algorithm remotely.
    
    This function exists for backward compatibility. New code should use
    the SIMICE client service directly.
    
    Args:
        data: Data array
        target_column_indexes: List of target column indices (1-based)
        job_id: ID of the job
        site_id: ID of this site
        central_url: URL of the central server
        token: Authentication token
        is_binary: List of binary flags for each target column
        extra_params: Additional parameters for the algorithm
        status_callback: Callback for status updates
    """
    try:
        # Prepare extra parameters
        params = extra_params or {}
        
        # Create the SIMICE client
        client = AlgorithmClientFactory.create_client("SIMICE")
        
        # Run the algorithm
        await client.run_algorithm(
            data=data,
            target_column_indexes=target_column_indexes,
            job_id=job_id,
            site_id=site_id,
            central_url=central_url,
            token=token,
            is_binary=is_binary,
            extra_params=params,
            status_callback=status_callback
        )
        
    except Exception as e:
        if status_callback:
            await status_callback.on_error(f"Error running SIMICE algorithm: {str(e)}")
        raise
