# MIT License (see LICENSE)
"""
Force generators for physics simulation.

This module provides functions that apply various forces to rigid bodies,
including gravity, linear drag, electromagnetic forces (Lorentz force),
and electrostatic interactions (Coulomb force).

All force functions modify body.force in-place and are designed to be called
during the force accumulation phase of the simulation step. See maths.md for
the underlying equations.

Key concepts:
- Forces are accumulated in body.force before integration.
- Electromagnetic forces (Lorentz, Coulomb) require body.charge != 0.
- Coulomb pairwise is O(N²); for large N, consider Barnes-Hut approximation.
"""
from __future__ import annotations

import numpy as np

from ..constants import K_COULOMB, DEFAULT_EPS
from ..types import RigidBody2D
from ..util import vec_cross_z, norm2


def apply_gravity(body: RigidBody2D, g: np.ndarray) -> None:
    """
    Apply gravitational force to a body.
    
    Implements F = m * g (maths.md Eq 5).
    
    Args:
        body: The rigid body to apply gravity to. Must have mass > 0.
        g: Gravitational acceleration vector as [gx, gy] in m/s².
        
    Note:
        Modifies body.force in-place. Has no effect if body.mass <= 0.
    """
    if body.mass > 0:
        body.force += body.mass * g


def apply_linear_drag(body: RigidBody2D, c: float) -> None:
    """
    Apply linear drag force proportional to velocity.
    
    Implements F = -c * v (maths.md Eq 6).
    
    Args:
        body: The rigid body to apply drag to. Must have mass > 0.
        c: Drag coefficient. Higher values = more damping.
        
    Note:
        Modifies body.force in-place. Has no effect if c == 0 or body.mass <= 0.
    """
    if c != 0.0 and body.mass > 0:
        body.force += -c * body.velocity


def apply_lorentz(body: RigidBody2D, E: np.ndarray, Bz: float) -> None:
    """
    Apply the Lorentz force from electric and magnetic fields.
    
    Implements F = q * (E + v × B) for 2D (maths.md Eq 7).
    The magnetic field B is assumed to point in the z-direction (Bz).
    
    Args:
        body: The charged rigid body. Must have charge != 0 and mass > 0.
        E: Electric field vector as [Ex, Ey] in V/m (or N/C).
        Bz: Magnetic field z-component in Tesla.
        
    Note:
        Modifies body.force in-place. Has no effect if body.charge == 0 or mass <= 0.
    """
    if body.charge != 0.0 and body.mass > 0:
        body.force += body.charge * (E + vec_cross_z(body.velocity, Bz))


def apply_coulomb_pairwise(bodies: list[RigidBody2D], eps: float = DEFAULT_EPS) -> None:
    """
    Apply Coulomb electrostatic forces between all pairs of charged bodies.
    
    Implements F = k * q1 * q2 * r / |r|³ (maths.md Eq 8).
    Uses Newton's third law: applies equal and opposite forces to each pair.
    
    Complexity: O(N²) where N is the number of charged bodies.
    For large N, consider Barnes-Hut or FMM algorithms (not implemented).
    
    Args:
        bodies: List of rigid bodies, some of which may be charged.
        eps: Softening parameter to prevent singularity at r=0. The effective
             distance is sqrt(r² + eps²).
             
    Note:
        Modifies body.force in-place for all charged bodies. Bodies with
        charge == 0 are skipped for efficiency.
    """
    n = len(bodies)
    for i in range(n):
        bi = bodies[i]
        qi = bi.charge
        if qi == 0.0:
            continue
        for j in range(i + 1, n):
            bj = bodies[j]
            qj = bj.charge
            if qj == 0.0:
                continue
            
            r = bj.position - bi.position
            r2 = norm2(r) + eps * eps
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))
            f = (K_COULOMB * qi * qj) * r * inv_r3
            
            # Newton's third law
            bi.force += f
            bj.force -= f
