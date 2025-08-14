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
    _service_instances: Dict[str, Any] = {}  # Cache service instances
    
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
        Create or get a service instance for an algorithm.
        
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
        
        # Return existing instance if available
        if algorithm_name in cls._service_instances:
            print(f"🔄 AlgorithmFactory: Returning existing {algorithm_name} service instance")
            return cls._service_instances[algorithm_name]
        
        # Create new instance and cache it
        service_class = cls._service_classes[algorithm_name]
        service_instance = service_class(manager)
        cls._service_instances[algorithm_name] = service_instance
        print(f"🏭 AlgorithmFactory: Created new {algorithm_name} service instance")
        return service_instance
