# MIT License (see LICENSE)
"""
Utilities for calculating physical invariants and conserved quantities.

Used for verifying simulation correctness and debugging stability issues.
In a closed system with no dissipation (drag/friction) or external forces,
total energy and momentum should remain constant (within integration error).
"""
from __future__ import annotations
import numpy as np

from ..types import RigidBody2D


def kinetic_energy(bodies: list[RigidBody2D]) -> float:
    """
    Calculate the total kinetic energy of a system of bodies.
    
    T = Σ (0.5 * m * v² + 0.5 * I * ω²)
    
    Args:
        bodies: List of rigid bodies.
    
    Returns:
        Total kinetic energy in Joules.
    """
    ke = 0.0
    for b in bodies:
        if b.mass <= 0:
            continue
        v_sq = float(np.dot(b.velocity, b.velocity))
        ke += 0.5 * b.mass * v_sq
        ke += 0.5 * b.inertia * (b.omega * b.omega)
    return ke


def linear_momentum(bodies: list[RigidBody2D]) -> np.ndarray:
    """
    Calculate the total linear momentum of a system.
    
    P = Σ (m * v)
    
    Args:
        bodies: List of rigid bodies.
        
    Returns:
        Total momentum vector [Px, Py] in kg·m/s.
    """
    p = np.zeros(2, dtype=np.float64)
    for b in bodies:
        if b.mass <= 0:
            continue
        p += b.mass * b.velocity
    return p
