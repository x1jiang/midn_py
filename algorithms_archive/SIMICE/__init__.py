"""
SIMICE Algorithm Package
Multiple Imputation using Chained Equations for federated learning.
"""

from .simice_central import SIMICECentralAlgorithm
from .simice_remote import SIMICERemoteAlgorithm

try:
    from .SIMICECentral import SIMICECentral
    from .SIMICERemote import SIMICERemote
except ImportError:
    # Optional imports for backward compatibility
    SIMICECentral = None
    SIMICERemote = None

__all__ = [
    'SIMICECentralAlgorithm',
    'SIMICERemoteAlgorithm'
]

# Add optional exports if available
if SIMICECentral is not None:
    __all__.append('SIMICECentral')
if SIMICERemote is not None:
    __all__.append('SIMICERemote')

__version__ = '1.0.0'
