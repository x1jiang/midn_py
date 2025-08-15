"""
Transfer utilities - Python equivalent of R Core/Transfer.R
Provides serialization/deserialization for matrices and vectors.
"""

import numpy as np
import json
from typing import Dict, Any, List, Union


def serialize_matrix(matrix: np.ndarray) -> Dict[str, Any]:
    """
    Serialize a matrix for network transmission - Python equivalent of R writeMat().
    
    Args:
        matrix: NumPy array to serialize
        
    Returns:
        Dictionary with matrix dimensions and data
    """
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")
    
    rows, cols = matrix.shape
    
    return {
        'type': 'matrix',
        'rows': int(rows),
        'cols': int(cols),
        'data': matrix.flatten().tolist()  # Row-major order
    }


def deserialize_matrix(data: Dict[str, Any]) -> np.ndarray:
    """
    Deserialize a matrix from network transmission - Python equivalent of R readMat().
    
    Args:
        data: Dictionary containing matrix data
        
    Returns:
        Reconstructed NumPy array
    """
    if data.get('type') != 'matrix':
        raise ValueError("Data is not a serialized matrix")
    
    rows = data['rows']
    cols = data['cols']
    flat_data = data['data']
    
    return np.array(flat_data).reshape(rows, cols)


def serialize_vector(vector: np.ndarray) -> Dict[str, Any]:
    """
    Serialize a vector for network transmission - Python equivalent of R writeVec().
    
    Args:
        vector: NumPy array to serialize
        
    Returns:
        Dictionary with vector length and data
    """
    if vector.ndim != 1:
        raise ValueError("Input must be a 1D vector")
    
    return {
        'type': 'vector',
        'length': int(len(vector)),
        'data': vector.tolist()
    }


def deserialize_vector(data: Dict[str, Any]) -> np.ndarray:
    """
    Deserialize a vector from network transmission - Python equivalent of R readVec().
    
    Args:
        data: Dictionary containing vector data
        
    Returns:
        Reconstructed NumPy array
    """
    if data.get('type') != 'vector':
        raise ValueError("Data is not a serialized vector")
    
    return np.array(data['data'])


def serialize_statistics(stats: Dict[str, Any]) -> str:
    """
    Serialize a complete statistics dictionary for network transmission.
    
    Args:
        stats: Dictionary containing statistical results
        
    Returns:
        JSON string representation
    """
    serialized = {}
    
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                serialized[key] = serialize_vector(value)
            elif value.ndim == 2:
                serialized[key] = serialize_matrix(value)
            else:
                serialized[key] = value.tolist()  # Fallback
        else:
            serialized[key] = value
    
    return json.dumps(serialized)


def deserialize_statistics(json_str: str) -> Dict[str, Any]:
    """
    Deserialize a statistics dictionary from network transmission.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        Dictionary with NumPy arrays reconstructed
    """
    data = json.loads(json_str)
    deserialized = {}
    
    for key, value in data.items():
        if isinstance(value, dict) and 'type' in value:
            if value['type'] == 'matrix':
                deserialized[key] = deserialize_matrix(value)
            elif value['type'] == 'vector':
                deserialized[key] = deserialize_vector(value)
            else:
                deserialized[key] = value
        else:
            deserialized[key] = value
    
    return deserialized


# Convenience functions for common use cases
def package_gaussian_stats(n: int, XTX: np.ndarray, XTy: np.ndarray, yTy: float) -> str:
    """Package Gaussian statistics for transmission."""
    stats = {
        'n': n,
        'XTX': XTX,
        'XTy': XTy,
        'yTy': yTy,
        'method': 'gaussian'
    }
    return serialize_statistics(stats)


def package_logistic_stats(n: int, H: np.ndarray, g: np.ndarray, log_lik: float) -> str:
    """Package logistic statistics for transmission."""
    stats = {
        'n': n,
        'H': H,
        'g': g,
        'log_lik': log_lik,
        'method': 'logistic'
    }
    return serialize_statistics(stats)
