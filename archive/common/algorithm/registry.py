"""
Registry for algorithm implementations in the PYMIDN system.
Handles discovery and registration of available algorithms.
"""

from typing import Dict, Type, Any, Optional
from .base import CentralAlgorithm, RemoteAlgorithm


class AlgorithmRegistry:
    """
    Registry for algorithm implementations.
    Maps algorithm names to their central and remote implementations.
    """
    
    _algorithms: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_algorithm(cls, name: str, central_class: Type[CentralAlgorithm], 
                           remote_class: Type[RemoteAlgorithm]) -> None:
        """
        Register an algorithm with the system.
        
        Args:
            name: Unique name identifier for the algorithm
            central_class: Class implementing the central component
            remote_class: Class implementing the remote component
        """
        cls._algorithms[name] = {
            'central': central_class,
            'remote': remote_class
        }
        
    @classmethod
    def register_central_algorithm(cls, central_class: Type[CentralAlgorithm]) -> None:
        """
        Register just the central component of an algorithm.
        
        Args:
            central_class: Class implementing the central component
        """
        name = central_class.get_algorithm_name()
        if name not in cls._algorithms:
            cls._algorithms[name] = {}
        cls._algorithms[name]['central'] = central_class
        
    @classmethod
    def register_remote_algorithm(cls, remote_class: Type[RemoteAlgorithm]) -> None:
        """
        Register just the remote component of an algorithm.
        
        Args:
            remote_class: Class implementing the remote component
        """
        name = remote_class.get_algorithm_name()
        if name not in cls._algorithms:
            cls._algorithms[name] = {}
        cls._algorithms[name]['remote'] = remote_class
    
    @classmethod
    def get_central_algorithm(cls, name: str) -> Type[CentralAlgorithm]:
        """
        Get the central implementation for an algorithm.
        
        Args:
            name: Name of the algorithm
            
        Returns:
            The central algorithm class
            
        Raises:
            ValueError: If the algorithm is not registered
        """
        if name not in cls._algorithms:
            raise ValueError(f"Algorithm '{name}' not registered")
        return cls._algorithms[name]['central']
    
    @classmethod
    def get_remote_algorithm(cls, name: str) -> Type[RemoteAlgorithm]:
        """
        Get the remote implementation for an algorithm.
        
        Args:
            name: Name of the algorithm
            
        Returns:
            The remote algorithm class
            
        Raises:
            ValueError: If the algorithm is not registered
        """
        if name not in cls._algorithms:
            raise ValueError(f"Algorithm '{name}' not registered")
        return cls._algorithms[name]['remote']
    
    @classmethod
    def list_algorithms(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered algorithms.
        
        Returns:
            Dictionary mapping algorithm names to their implementations
        """
        return cls._algorithms.copy()
