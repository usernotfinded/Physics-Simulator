# MIT License (see LICENSE)
"""
Utility functions for vector math and numeric operations.

Provides low-level 2D vector operations optimized for the physics engine,
including normalization, cross products, and array conversion helpers.
All functions operate on 2D vectors represented as numpy arrays of shape (2,).
"""
from __future__ import annotations
import os

import numpy as np


def f64(x) -> np.ndarray:
    """
    Convert any array-like to a float64 numpy array.
    
    Used throughout the codebase to ensure consistent numeric precision
    and allow tuple/list inputs for positions and velocities.
    """
    return np.array(x, dtype=np.float64)


def norm2(v: np.ndarray) -> float:
    """Squared magnitude of a 2D vector. Avoids sqrt for performance."""
    return float(v[0] * v[0] + v[1] * v[1])


def norm(v: np.ndarray) -> float:
    """Magnitude (length) of a 2D vector."""
    return float(np.sqrt(norm2(v)))


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return a unit (normalized) vector in the same direction as v.
    
    Returns zero vector if |v| < eps to avoid division by zero.
    """
    n = norm(v)
    if n < eps:
        return np.zeros(2, dtype=np.float64)
    return v / n


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """
    2D cross product (scalar result): a × b = ax*by - ay*bx.
    
    In 2D, the cross product yields a scalar representing the
    z-component of the 3D cross product (a, 0) × (b, 0).
    Positive result means b is counterclockwise from a.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def cross_z_scalar_vec(z: float, v: np.ndarray) -> np.ndarray:
    """
    Cross product of z-axis scalar with 2D vector: (0, 0, z) × (vx, vy, 0).
    
    Result: (-z*vy, z*vx). Used for converting angular velocity to
    linear velocity at a point offset from the center of mass.
    """
    return np.array([-z * v[1], z * v[0]], dtype=np.float64)


def vec_cross_z(v: np.ndarray, z: float) -> np.ndarray:
    """
    Cross product of 2D vector with z-axis scalar: (vx, vy, 0) × (0, 0, z).
    
    Result: (vy*z, -vx*z). Used in Lorentz force calculation where
    F = q(v × B) and B points in z-direction. See maths.md Eq (7).
    """
    return np.array([v[1] * z, -v[0] * z], dtype=np.float64)


def use_numba() -> bool:
    """Check if Numba JIT compilation is enabled via environment variable."""
    return os.environ.get("PHYSICS_SIM_USE_NUMBA", "0") == "1"
