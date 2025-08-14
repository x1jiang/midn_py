"""
Initialize algorithm services for the central site.
"""

from central.app.services.algorithm_factory import AlgorithmServiceFactory

# Import service implementations
from central.app.services.simi_service import SIMIService
from central.app.services.simice_service import SIMICEService

# Register services
AlgorithmServiceFactory.register_service("SIMI", SIMIService)
AlgorithmServiceFactory.register_service("SIMICE", SIMICEService)

# Import algorithm registry to ensure all algorithms are registered
import common.algorithm.register
