"""
Factory for creating algorithm clients on the remote site.
"""

from typing import Dict, Type, Any

from common.algorithm.registry import AlgorithmRegistry


class AlgorithmClientFactory:
    """
    Factory for creating algorithm clients.
    Maps algorithm names to their client implementations.
    """
    
    _client_classes: Dict[str, Type[Any]] = {}
    
    @classmethod
    def register_client(cls, algorithm_name: str, client_class: Type[Any]) -> None:
        """
        Register a client class for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            client_class: Class implementing the client
        """
        cls._client_classes[algorithm_name] = client_class
    
    @classmethod
    def create_client(cls, algorithm_name: str) -> Any:
        """
        Create a client instance for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            An instance of the algorithm client
            
        Raises:
            ValueError: If no client is registered for the algorithm
        """
        print(f"🏭 AlgorithmClientFactory: Request for '{algorithm_name}' client")
        print(f"🔍 AlgorithmClientFactory: Available clients: {list(cls._client_classes.keys())}")
        
        if algorithm_name not in cls._client_classes:
            raise ValueError(f"No client registered for algorithm '{algorithm_name}'. Available: {list(cls._client_classes.keys())}")
        
        client_class = cls._client_classes[algorithm_name]
        
        # Get the algorithm class from the registry
        remote_algorithm_class = AlgorithmRegistry.get_remote_algorithm(algorithm_name)
        
        # Create the client with the algorithm class
        return client_class(remote_algorithm_class)
