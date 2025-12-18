# MIT License (see LICENSE)
"""
Gilbert-Johnson-Keerthi (GJK) algorithm for convex collision detection.

This module implements the GJK algorithm to test intersection between two
convex shapes using their support functions. GJK works in Minkowski space
and iteratively builds a simplex to determine if the origin is contained.

Key concepts:
- Support function: Returns the farthest point on a shape in a given direction.
- Minkowski difference: The set A - B contains the origin iff A and B intersect.
- Simplex: 1-3 points that evolve toward containing the origin.

Usage:
    xf_a = {"support": lambda d: circle_support(pos_a, radius_a, d)}
    xf_b = {"support": lambda d: box_support(pos_b, angle_b, extents_b, d)}
    hit, simplex = gjk_intersect(xf_a, xf_b)
    if hit:
        # Use EPA for penetration depth
        normal, depth = epa_penetration(xf_a, xf_b, simplex)
"""
from __future__ import annotations
from typing import Callable, Any

import numpy as np

# Type aliases for clarity
Vec2 = np.ndarray  # Shape (2,), dtype float64
SupportFunc = Callable[[Vec2], Vec2]
Transform = dict[str, SupportFunc]  # {"support": SupportFunc}
Simplex = list[Vec2]


def support(
    shapeA: Any,
    xfA: Transform,
    shapeB: Any,
    xfB: Transform,
    d: Vec2,
) -> Vec2:
    """
    Compute a support point in Minkowski difference space.
    
    The Minkowski support is: supA(d) - supB(-d), which gives a point
    on the boundary of the Minkowski difference A - B.
    
    Args:
        shapeA: Shape A (unused, kept for interface compatibility).
        xfA: Transform dict with "support" callable for shape A.
        shapeB: Shape B (unused, kept for interface compatibility).
        xfB: Transform dict with "support" callable for shape B.
        d: Direction vector to search.
        
    Returns:
        Support point in Minkowski space as [x, y].
    """
    pA = xfA["support"](d)
    pB = xfB["support"](-d)
    return pA - pB


def gjk_intersect(
    xfA: Transform,
    xfB: Transform,
    max_iters: int = 32,
    eps: float = 1e-12,
) -> tuple[bool, Simplex]:
    """
    Test if two convex shapes intersect using the GJK algorithm.
    
    Iteratively builds a simplex in Minkowski space, trying to enclose
    the origin. If successful, the shapes intersect.
    
    Args:
        xfA: Transform dict with "support" callable for shape A.
        xfB: Transform dict with "support" callable for shape B.
        max_iters: Maximum iterations before giving up. Default 32.
        eps: Tolerance for degenerate cases. Default 1e-12.
        
    Returns:
        Tuple (hit, simplex) where:
        - hit: True if shapes intersect, False otherwise.
        - simplex: Final simplex points (2-3 points if hit, for EPA input).
    """
    d: Vec2 = np.array([1.0, 0.0], dtype=np.float64)
    simplex: Simplex = []
    
    a = support(None, xfA, None, xfB, d)
    simplex.append(a)
    d = -a
    
    for _ in range(max_iters):
        a = support(None, xfA, None, xfB, d)
        
        # If new point doesn't pass origin, no intersection
        if float(np.dot(a, d)) < 0:
            return False, simplex
        
        simplex.append(a)
        hit, d = _handle_simplex(simplex, d, eps)
        
        if hit:
            return True, simplex
    
    return False, simplex


def _triple_product(a: Vec2, b: Vec2, c: Vec2) -> Vec2:
    """
    Compute the 2D triple product: (A × B) × C = B(A·C) - A(B·C).
    
    This gives a vector perpendicular to C in the direction away from A.
    
    Args:
        a: First vector.
        b: Second vector.
        c: Third vector.
        
    Returns:
        Result vector perpendicular to c.
    """
    return b * float(np.dot(a, c)) - a * float(np.dot(b, c))


def _handle_simplex(simplex: Simplex, d: Vec2, eps: float) -> tuple[bool, Vec2]:
    """
    Update simplex and search direction based on current simplex state.
    
    For a 2-simplex (line), finds the region containing the origin and
    updates the direction. For a 3-simplex (triangle), checks if the
    origin is inside.
    
    Args:
        simplex: Current simplex points (modified in-place).
        d: Current search direction.
        eps: Tolerance for degenerate edges.
        
    Returns:
        Tuple (hit, new_direction) where hit is True if origin is enclosed.
    """
    if len(simplex) == 2:
        # Line segment case
        b = simplex[0]
        a = simplex[1]
        ab = b - a
        ao = -a
        
        if float(np.dot(ab, ao)) > 0:
            # Origin is in region between a and b
            d_new = _triple_product(ab, ao, ab)
            if np.linalg.norm(d_new) < eps:
                # Degenerate: use perpendicular
                d_new = np.array([ab[1], -ab[0]], dtype=np.float64)
            return False, d_new
        else:
            # Origin is past a, keep only a
            simplex[:] = [a]
            return False, ao
            
    elif len(simplex) == 3:
        # Triangle case
        c = simplex[0]
        b = simplex[1]
        a = simplex[2]
        ab = b - a
        ac = c - a
        ao = -a
        
        # Check if origin is outside edge AB
        ab_perp = _triple_product(ac, ab, ab)
        if float(np.dot(ab_perp, ao)) > 0:
            simplex[:] = [b, a]
            return False, ab_perp
        
        # Check if origin is outside edge AC
        ac_perp = _triple_product(ab, ac, ac)
        if float(np.dot(ac_perp, ao)) > 0:
            simplex[:] = [c, a]
            return False, ac_perp
        
        # Origin is inside triangle
        return True, d
    
    else:
        return False, d
