"""
Register all algorithms with the system.
"""

from common.algorithm.registry import AlgorithmRegistry

# Import algorithm implementations
from algorithms.SIMI.simi_central import SIMICentralAlgorithm
from algorithms.SIMI.simi_remote import SIMIRemoteAlgorithm

# Register algorithms
AlgorithmRegistry.register_algorithm("SIMI", SIMICentralAlgorithm, SIMIRemoteAlgorithm)

# Add more algorithms here as they are implemented
# AlgorithmRegistry.register_algorithm("SIMICE", SIMICECentralAlgorithm, SIMICERemoteAlgorithm)
# When implementing SIMICE, use:
# from algorithms.SIMICE.simice_central import SIMICECentralAlgorithm
# from algorithms.SIMICE.simice_remote import SIMICERemoteAlgorithm
