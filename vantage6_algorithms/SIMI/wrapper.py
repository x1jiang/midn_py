"""
Vantage6 Wrapper for SIMI Algorithm

This is the entry point that vantage6 uses to execute the algorithm.
It registers the master and RPC functions with vantage6's framework.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import algorithm functions
from algorithm import master_simi, RPC_simi_remote_gaussian, RPC_simi_remote_logistic

# Import vantage6 tools
try:
    from vantage6.algorithm.tools import AlgorithmClient
    from vantage6.algorithm.tools.decorators import algorithm_client
    
    # Register the algorithm with vantage6
    @algorithm_client
    def simi_wrapper(client: AlgorithmClient, data, *args, **kwargs):
        """
        Vantage6 wrapper for SIMI algorithm.
        
        This function is called by vantage6 when executing the algorithm.
        It routes to the appropriate function based on the method name.
        """
        method = kwargs.get('method', 'master')
        
        if method == 'master':
            # Execute master function
            return master_simi(client, data, *args, **kwargs)
        elif method == 'simi_remote_gaussian':
            # Execute remote Gaussian function
            return RPC_simi_remote_gaussian(data, *args, **kwargs)
        elif method == 'simi_remote_logistic':
            # Execute remote Logistic function
            return RPC_simi_remote_logistic(data, *args, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Set as main entry point
    if __name__ == "__main__":
        simi_wrapper()
        
except ImportError:
    # Fallback for local testing
    print("Warning: vantage6 not installed. This wrapper requires vantage6.")
    print("For local testing, use algorithm.py directly.")
    sys.exit(1)


