# MIT License (see LICENSE)
"""
Continuous Collision Detection (CCD) for physics simulation.

This module provides time-of-impact (TOI) calculations to detect and prevent
tunneling (fast-moving objects passing through each other between discrete
timesteps). Currently supports circle-circle CCD using analytic quadratic
solving.

Key concepts:
- TOI (Time of Impact): The exact time within a timestep when two objects
  first make contact.
- Tunneling prevention: By detecting TOI, we can substep the simulation
  to resolve collisions that would otherwise be missed.
"""
from __future__ import annotations

import numpy as np


def circle_toi(
    p0: np.ndarray,
    v0: np.ndarray,
    r0: float,
    p1: np.ndarray,
    v1: np.ndarray,
    r1: float,
    dt: float,
) -> float | None:
    """
    Compute the time of impact between two moving circles.
    
    Solves the quadratic equation for when the distance between circle centers
    equals the sum of their radii: ||(p0 - p1) + t*(v0 - v1)|| = r0 + r1.
    
    Args:
        p0: Position of circle 0 at t=0 as [x, y].
        v0: Velocity of circle 0 as [vx, vy].
        r0: Radius of circle 0.
        p1: Position of circle 1 at t=0 as [x, y].
        v1: Velocity of circle 1 as [vx, vy].
        r1: Radius of circle 1.
        dt: Maximum time to search for collision.
        
    Returns:
        Time of first contact in [0, dt] if collision occurs, None otherwise.
        Returns 0.0 if circles are already overlapping at t=0.
        
    Note:
        Uses the earliest positive root of the quadratic. If both roots are
        negative or beyond dt, returns None.
    """
    dp = p0 - p1
    dv = v0 - v1
    R = r0 + r1
    
    # Quadratic coefficients: a*t^2 + b*t + c = 0
    a = float(np.dot(dv, dv))
    b = 2.0 * float(np.dot(dp, dv))
    c = float(np.dot(dp, dp)) - R * R
    
    # Already overlapping
    if c <= 0:
        return 0.0
    
    # No relative motion
    if a < 1e-15:
        return None
    
    # Solve quadratic
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    
    s = float(np.sqrt(disc))
    t0 = (-b - s) / (2 * a)
    t1 = (-b + s) / (2 * a)
    
    # Return earliest valid root
    if 0.0 <= t0 <= dt:
        return t0
    elif 0.0 <= t1 <= dt:
        return t1
    
    return None
