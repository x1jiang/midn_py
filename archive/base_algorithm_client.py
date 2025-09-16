"""
Deprecated: BaseAlgorithmClient is no longer used in the new architecture.
"""

class BaseAlgorithmClient:  # pragma: no cover - deprecated stub
    def __init__(self, *args, **kwargs):
        raise RuntimeError("BaseAlgorithmClient is deprecated; use MIDN_R_PY clients via remote/app/main.py")
