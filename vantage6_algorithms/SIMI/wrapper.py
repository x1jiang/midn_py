"""
Vantage6 Wrapper for SIMI Algorithm

This is the entry point that vantage6 uses to execute the algorithm.
It registers the master and RPC functions with vantage6's framework.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import algorithm functions (exposed so wrap_algorithm can find them)
from algorithm import master_simi, simi_remote_gaussian, simi_remote_logistic

# Import vantage6 tools
try:
    from vantage6.algorithm.client import AlgorithmClient
    from vantage6.algorithm.tools.decorators import algorithm_client, data
    from vantage6.algorithm.tools.util import info, warn, error
    
    @data(1)
    @algorithm_client
    def master(client: AlgorithmClient, data, *args, **kwargs):
        """Entry point for SIMI master."""
        return master_simi(client, data, *args, **kwargs)
    
    # No direct execution; vantage6's wrap_algorithm will invoke this entry point.
        
except ImportError:
    # Fallback for local testing
    print("Warning: vantage6 not installed. This wrapper requires vantage6.")
    print("For local testing, use algorithm.py directly.")
    sys.exit(1)
