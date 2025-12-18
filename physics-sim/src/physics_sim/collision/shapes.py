# MIT License (see LICENSE)
"""
Geometric shape definitions for collision detection.

This module defines general convex polygons, providing support functions
needed for GJK intersection testing.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..util import f64


from ..types import ConvexPolygon


def support_polygon(poly: ConvexPolygon, direction: np.ndarray) -> np.ndarray:
    """
    Find the farthest vertex in a given direction (support point).
    
    Used by GJK algorithm. The support point s(d) maximizes dot(s, d).
    
    Args:
        poly: The convex polygon.
        direction: Search direction vector.
        
    Returns:
        The vertex that is farthest in the search direction.
    """
    # Optimized: Compute dot product with all vertices and take max.
    # For very large polygons, hill-climbing would be O(log N).
    dots = poly.vertices @ direction
    return poly.vertices[int(np.argmax(dots))]
