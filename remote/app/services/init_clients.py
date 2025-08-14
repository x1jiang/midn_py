"""
Initialize algorithm clients for the remote site.
"""

from remote.app.services.algorithm_factory import AlgorithmClientFactory

# Import client implementations
from remote.app.services.simi_client import SIMIClient
from remote.app.services.simice_client import SIMICEClient

# Register clients
AlgorithmClientFactory.register_client("SIMI", SIMIClient)
AlgorithmClientFactory.register_client("SIMICE", SIMICEClient)

print("🏭 Remote: Registered algorithm clients:")
print(f"   - SIMI: {SIMIClient}")
print(f"   - SIMICE: {SIMICEClient}")

# Import algorithm registry to ensure all algorithms are registered
import common.algorithm.register
