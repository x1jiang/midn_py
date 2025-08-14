"""
Initialize algorithm clients for the remote site.
"""

from remote.app.services.algorithm_factory import AlgorithmClientFactory

# Import client implementations
from remote.app.services.simi_client import SIMIClient

# Register clients
AlgorithmClientFactory.register_client("SIMI", SIMIClient)

# Import algorithm registry to ensure all algorithms are registered
import common.algorithm.register
