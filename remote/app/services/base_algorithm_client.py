"""
Base algorithm client for remote site.
Provides core functionality for running algorithm jobs.
"""

from abc import ABC, abstractmethod
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Type

from common.algorithm.base import RemoteAlgorithm
from common.algorithm.protocol import MessageType
from ..websockets.connection_client import ConnectionClient


class BaseAlgorithmClient(ABC):
    """
    Base client for all algorithm implementations on the remote site.
    Handles common aspects of connection management and algorithm execution.
    """
    
    def __init__(self, algorithm_class: Type[RemoteAlgorithm]):
        """
        Initialize base algorithm client.
        
        Args:
            algorithm_class: Algorithm class to use
        """
        self.algorithm_class = algorithm_class
        self.algorithm_instance = None
    
    @abstractmethod
    async def run_algorithm(self, data: np.ndarray, target_column: int,
                           job_id: int, site_id: str, central_url: str, token: str,
                           extra_params: Optional[Dict[str, Any]] = None,
                           status_callback: Optional[Any] = None) -> None:
        """
        Run the algorithm.
        
        Args:
            data: Data array
            target_column: Index of the target column
            job_id: ID of the job
            site_id: ID of this site
            central_url: URL of the central server
            token: Authentication token
            extra_params: Additional parameters for the algorithm
            status_callback: Callback for status updates
        """
        pass
    
    @abstractmethod
    async def handle_message(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a message from the central site.
        
        Args:
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        pass
    
    @abstractmethod
    async def process_method_message(self, method: str) -> None:
        """
        Process a method message from the central site.
        
        Args:
            method: Method to use for the algorithm
        """
        pass
