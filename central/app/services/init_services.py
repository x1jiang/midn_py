"""
Initialize algorithm services for the central site.
"""

from central.app.services.algorithm_factory import AlgorithmServiceFactory

# Import service implementations
from central.app.services.simi_service import SIMIService

# Register services
AlgorithmServiceFactory.register_service("SIMI", SIMIService)

# Import algorithm registry to ensure all algorithms are registered
import common.algorithm.register
