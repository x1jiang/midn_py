"""
Deprecated: algorithm_adapter is no longer used.
Algorithms are launched directly from remote/app/main.py using MIDN_R_PY modules.
"""

def get_algorithm_client(*args, **kwargs):  # pragma: no cover - deprecated stub
    raise RuntimeError("algorithm_adapter is deprecated; use remote/app/main.py to run algorithms")
