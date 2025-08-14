"""
Initialize algorithm services and register them with the factory.
"""

from .algorithm_factory import AlgorithmClientFactory
from .simi_client import SIMIClient

# Register algorithm clients
AlgorithmClientFactory.register_client("SIMI", SIMIClient)
