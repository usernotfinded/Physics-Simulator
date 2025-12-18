# MIT License (see LICENSE)
"""
Expanding Polytope Algorithm (EPA) for penetration depth calculation.

This module implements EPA to find the penetration depth and separation
normal for two intersecting convex shapes. EPA expands the GJK simplex
into a polytope and iteratively refines it to find the closest point
on the Minkowski boundary to the origin.

EPA should only be called after GJK confirms intersection (hit=True).
The returned normal points from shape A toward shape B.

Usage:
    hit, simplex = gjk_intersect(xf_a, xf_b)
    if hit:
        normal, depth = epa_penetration(xf_a, xf_b, simplex)
"""
from __future__ import annotations
from typing import Callable

import numpy as np

from .gjk import support, Transform, Vec2, Simplex


def epa_penetration(
    xfA: Transform,
    xfB: Transform,
    simplex: Simplex,
    max_iters: int = 48,
    tol: float = 1e-6,
) -> tuple[Vec2, float]:
    """
    Compute penetration normal and depth using the Expanding Polytope Algorithm.
    
    Given an intersecting simplex from GJK (must be a triangle), expands it
    into a convex polytope and iteratively adds vertices to find the minimum
    penetration depth.
    
    Args:
        xfA: Transform dict with "support" callable for shape A.
        xfB: Transform dict with "support" callable for shape B.
        simplex: Initial simplex from GJK (should have 3 points for EPA).
        max_iters: Maximum expansion iterations. Default 48.
        tol: Convergence tolerance for depth refinement.
        
    Returns:
        Tuple (normal, depth) where:
        - normal: Unit separation vector from A toward B as [nx, ny].
        - depth: Penetration depth in the normal direction.
        
    Note:
        Returns default values (x-axis normal, 0 depth) if simplex has
        fewer than 3 points or algorithm fails to converge.
    """
    # Copy simplex to avoid modifying original
    poly: list[Vec2] = simplex[:]
    
    if len(poly) < 3:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0

    for _ in range(max_iters):
        # Find the edge closest to the origin
        min_dist = 1e30
        min_i = -1
        best_n: Vec2 | None = None
        
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            e = b - a
            
            # Outward-pointing normal (perpendicular to edge)
            n = np.array([e[1], -e[0]], dtype=np.float64)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-12:
                continue
            
            n /= n_norm
            dist = float(np.dot(n, a))
            
            if dist < min_dist:
                min_dist = dist
                min_i = i
                best_n = n

        if best_n is None:
            break

        # Get support point in direction of closest edge normal
        p = support(None, xfA, None, xfB, best_n)
        d = float(np.dot(best_n, p))
        
        # Check convergence
        if d - min_dist < tol:
            return best_n, d
        
        # Expand polytope by inserting new vertex
        poly.insert(min_i + 1, p)

    # Fallback: return default values
    return np.array([1.0, 0.0], dtype=np.float64), 0.0
