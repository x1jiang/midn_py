"""
Vantage6 Wrapper for SIMICE Algorithm

This is the entry point that vantage6 uses to execute the algorithm.
It registers the master and RPC functions with vantage6's framework.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import algorithm functions
from algorithm import master_simice, RPC_simice_remote_initialize, RPC_simice_remote_statistics

# Import vantage6 tools
try:
    from vantage6.algorithm.tools import AlgorithmClient
    from vantage6.algorithm.tools.decorators import algorithm_client
    
    # Register the algorithm with vantage6
    @algorithm_client
    def simice_wrapper(client: AlgorithmClient, data, *args, **kwargs):
        """
        Vantage6 wrapper for SIMICE algorithm.
        
        This function is called by vantage6 when executing the algorithm.
        It routes to the appropriate function based on the method name.
        """
        method = kwargs.get('method', 'master')
        
        if method == 'master':
            # Execute master function
            return master_simice(client, data, *args, **kwargs)
        elif method == 'simice_remote_initialize':
            # Execute remote initialize function
            return RPC_simice_remote_initialize(*args, **kwargs)
        elif method == 'simice_remote_statistics':
            # Execute remote statistics function
            return RPC_simice_remote_statistics(*args, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Set as main entry point
    if __name__ == "__main__":
        simice_wrapper()
        
except ImportError:
    # Fallback for local testing
    print("Warning: vantage6 not installed. This wrapper requires vantage6.")
    print("For local testing, use algorithm.py directly.")
    sys.exit(1)


