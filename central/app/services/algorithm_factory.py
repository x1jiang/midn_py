"""
Factory for creating algorithm services on the central site.
"""

from typing import Dict, Type, Any

from common.algorithm.registry import AlgorithmRegistry
from ..websockets.connection_manager import ConnectionManager


class AlgorithmServiceFactory:
    """
    Factory for creating algorithm services.
    Maps algorithm names to their service implementations.
    """
    
    _service_classes: Dict[str, Type[Any]] = {}
    
    @classmethod
    def register_service(cls, algorithm_name: str, service_class: Type[Any]) -> None:
        """
        Register a service class for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            service_class: Class implementing the service
        """
        cls._service_classes[algorithm_name] = service_class
    
    @classmethod
    def create_service(cls, algorithm_name: str, manager: ConnectionManager) -> Any:
        """
        Create a service instance for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            manager: WebSocket connection manager
            
        Returns:
            An instance of the algorithm service
            
        Raises:
            ValueError: If no service is registered for the algorithm
        """
        if algorithm_name not in cls._service_classes:
            raise ValueError(f"No service registered for algorithm '{algorithm_name}'")
        
        service_class = cls._service_classes[algorithm_name]
        return service_class(manager)
