"""
Initialize algorithm registration.
This file is imported at application startup to ensure all algorithms are registered.
"""

from common.algorithm.registry import AlgorithmRegistry
from algorithms.SIMI.simi_remote import SIMIRemoteAlgorithm
from algorithms.SIMI.simi_central import SIMICentralAlgorithm

# Register all algorithms
# Option 1: Register both parts together
AlgorithmRegistry.register_algorithm("SIMI", SIMICentralAlgorithm, SIMIRemoteAlgorithm)

# Option 2: Register parts separately
# AlgorithmRegistry.register_remote_algorithm(SIMIRemoteAlgorithm)
# AlgorithmRegistry.register_central_algorithm(SIMICentralAlgorithm)
