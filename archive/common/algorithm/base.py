"""
Base classes for algorithm implementations in the PYMIDN system.
Provides interfaces for both central and remote algorithm components.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class BaseAlgorithm(ABC):
    """
    Base class for all algorithms.
    """
    
    @classmethod
    @abstractmethod
    def get_algorithm_name(cls) -> str:
        """Get the unique name identifier for this algorithm"""
        pass
    
    @classmethod
    @abstractmethod
    def get_supported_methods(cls) -> List[str]:
        """Get the list of methods supported by this algorithm (e.g. 'gaussian', 'logistic')"""
        pass


class CentralAlgorithm(BaseAlgorithm):
    """
    Base class for central algorithm implementations.
    """
    
    @abstractmethod
    async def aggregate_data(self, local_data: Any, remote_data_list: List[Any]) -> Any:
        """
        Aggregate local and remote data.
        
        Args:
            local_data: Data from the central site
            remote_data_list: List of data from remote sites
            
        Returns:
            Aggregated data
        """
        pass
    
    @abstractmethod
    async def impute(self, data: pd.DataFrame, target_column: int, aggregated_data: Any, 
                    method: str, imputation_count: int = 10) -> pd.DataFrame:
        """
        Perform imputation using the aggregated data.
        
        Args:
            data: DataFrame with missing values to impute
            target_column: Index of the column to impute
            aggregated_data: Data aggregated from all sites
            method: Imputation method to use (e.g. 'gaussian', 'logistic')
            imputation_count: Number of imputations to generate
            
        Returns:
            DataFrame with imputed values
        """
        pass
    
    @abstractmethod
    async def process_message(self, site_id: str, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from a remote site.
        
        Args:
            site_id: ID of the site that sent the message
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        pass


class RemoteAlgorithm(BaseAlgorithm):
    """
    Base class for remote algorithm implementations.
    """
    
    @abstractmethod
    async def prepare_data(self, data: np.ndarray, target_column: int) -> Dict[str, Any]:
        """
        Prepare data for initial transmission to central.
        
        Args:
            data: Data array
            target_column: Index of the target column
            
        Returns:
            Data to send to the central site
        """
        pass
    
    @abstractmethod
    async def process_message(self, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from the central site.
        
        Args:
            message_type: Type of the message
            payload: Message payload
            
        Returns:
            Response payload (if any)
        """
        pass
