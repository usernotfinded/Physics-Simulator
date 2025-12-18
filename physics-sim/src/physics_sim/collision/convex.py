# MIT License (see LICENSE)
"""
Convex collision detection using GJK + EPA.

This module provides unified collision detection for convex shapes (circles, boxes)
using the GJK algorithm for intersection testing and EPA for penetration depth.
"""
from __future__ import annotations
import numpy as np
from typing import Callable

from ..types import RigidBody2D, Circle, Box, ConvexPolygon
from .contact import Contact
from .gjk import gjk_intersect
from .epa import epa_penetration
from .shapes import support_polygon


def circle_support(center: np.ndarray, radius: float, direction: np.ndarray) -> np.ndarray:
    """
    Support function for a circle.
    
    The support point is the point on the circle's boundary that is
    farthest in the given direction.
    
    Args:
        center: Circle center position [x, y].
        radius: Circle radius.
        direction: Direction vector to search.
        
    Returns:
        Support point on the circle boundary.
    """
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-12:
        return center.copy()
    d_unit = direction / d_norm
    return center + d_unit * radius


def box_support(center: np.ndarray, angle: float, half_extents: tuple[float, float], 
                direction: np.ndarray) -> np.ndarray:
    """
    Support function for an oriented box (OBB).
    
    The support point is the vertex of the box that is farthest
    in the given direction.
    
    Args:
        center: Box center position [x, y].
        angle: Box rotation angle in radians.
        half_extents: (hx, hy) half-widths of the box.
        direction: Direction vector to search.
        
    Returns:
        Support point (one of the box's vertices).
    """
    hx, hy = half_extents
    c, s = np.cos(angle), np.sin(angle)
    
    # Rotation matrix columns (local x and y axes in world space)
    ax = np.array([c, s], dtype=np.float64)
    ay = np.array([-s, c], dtype=np.float64)
    
    # Project direction onto local axes to find the farthest vertex
    dx = hx if np.dot(direction, ax) >= 0 else -hx
    dy = hy if np.dot(direction, ay) >= 0 else -hy
    
    return center + ax * dx + ay * dy


def make_transform(body: RigidBody2D) -> dict:
    """
    Create a GJK/EPA transform dictionary with a support function for a body.
    
    The transform dict contains a 'support' key with a callable that takes
    a direction and returns the support point for the body's shape.
    
    Args:
        body: The rigid body with position, angle, and shape.
        
    Returns:
        Dictionary with 'support' callable for GJK/EPA algorithms.
        
    Raises:
        TypeError: If the body has an unsupported shape type.
    """
    pos = body.position
    angle = body.angle
    shape = body.shape
    
    if isinstance(shape, Circle):
        def support_fn(d: np.ndarray) -> np.ndarray:
            return circle_support(pos, shape.radius, d)
    elif isinstance(shape, Box):
        def support_fn(d: np.ndarray) -> np.ndarray:
            return box_support(pos, angle, shape.half_extents, d)
    elif isinstance(shape, ConvexPolygon):
        # Support point for generic polygon
        # Note: poly vertices are local, so we need to rotate/translate support vector?
        # NO, usually support_polygon takes poly in local space?
        # Let's check shapes.py logic.
        # shapes.support_polygon takes poly.vertices.
        # But poly.vertices are in LOCAL space.
        # But `d` is in WORLD space?
        # Yes, GJK is in world space usually.
        # So we can either:
        # A) Transform poly vertices to world space (expensive to do every frame)
        # B) Transform direction `d` to local space, find local support, transform support point to world.
        #    Support(d) = R * LocalSupport(R^T * d) + p
        
        # Checking shapes.py:
        # def support_polygon(poly: ConvexPolygon, direction: np.ndarray) -> np.ndarray:
        #     # argmax dot(v, d)
        #     dots = poly.vertices @ direction
        #     return poly.vertices[int(np.argmax(dots))]
        # This assumes `direction` is in same space as vertices.
        
        c_cos, s_sin = np.cos(angle), np.sin(angle)
        # Rotation matrix R = [[c, -s], [s, c]]
        # Inverse R^T = [[c, s], [-s, c]]
        
        def support_fn(d: np.ndarray) -> np.ndarray:
            # Transform direction to local space: d_local = R^T * d
            dx, dy = d
            d_local = np.array([
                dx * c_cos + dy * s_sin,
                -dx * s_sin + dy * c_cos
            ], dtype=np.float64)
            
            # Find support in local space
            local_support = support_polygon(shape, d_local)
            
            # Transform support point to world space: p_world = R * p_local + pos
            lx, ly = local_support
            wx = lx * c_cos - ly * s_sin + pos[0]
            wy = lx * s_sin + ly * c_cos + pos[1]
            return np.array([wx, wy], dtype=np.float64)
            
    else:
        raise TypeError(f"Unsupported shape type for GJK: {type(shape)}")
    
    return {"support": support_fn}


def convex_contact(a: RigidBody2D, b: RigidBody2D) -> Contact | None:
    """
    Detect contact between two convex bodies using GJK + EPA.
    
    This function works for any combination of Circle and Box shapes.
    
    Args:
        a: First rigid body.
        b: Second rigid body.
        
    Returns:
        Contact object if bodies are intersecting, None otherwise.
    """
    xf_a = make_transform(a)
    xf_b = make_transform(b)
    
    # GJK intersection test
    hit, simplex = gjk_intersect(xf_a, xf_b)
    
    if not hit:
        return None
    
    # EPA for penetration depth and normal
    normal, depth = epa_penetration(xf_a, xf_b, simplex)
    
    if depth < 1e-9:
        return None
    
    # Compute contact point
    # Improvement: For Circle collisions, using the average of supports is unstable
    # because the Box support function returns a vertex (corner) even for face collisions.
    # We should bias the contact point towards the Circle's surface point.
    
    point_a = xf_a["support"](normal)
    point_b = xf_b["support"](-normal)
    
    contact_point = (point_a + point_b) * 0.5 # Default
    
    if isinstance(a.shape, Circle):
        # A is Circle, point_a is on A's surface. Use it (or centered).
        # Actually, point_a is the deep point on A.
        # Ideally contact is on the surface between them.
        # But simply using point_a is better than averaging with a far Box corner.
        # Adjusted: point_a + normal * (depth * 0.5)?
        # Let's just use point_a - normal * (depth * 0.5) to centre it?
        # A is pushed by normal. point_a is extreme in 'normal'.
        # So point_a is "front".
        # Let's use the Circle's surface point. 
        # point_a is Support(A, n).
        # If A is Circle, point_a = center + r * n.
        contact_point = point_a
        
    elif isinstance(b.shape, Circle):
        # B is Circle. point_b = Support(B, -n).
        # point_b = center + r * (-n).
        contact_point = point_b

    return Contact(
        a=a,
        b=b,
        point=contact_point, # Uses Circle's point if available
        normal=normal,
        penetration=depth,
    )


def circle_circle_contact_fast(a: RigidBody2D, b: RigidBody2D) -> Contact | None:
    """
    Fast circle-circle contact detection (direct formula, no GJK).
    
    This is a specialized fast path for circle-circle collisions that
    avoids the GJK/EPA overhead.
    
    Args:
        a: First body (must have Circle shape).
        b: Second body (must have Circle shape).
        
    Returns:
        Contact object if circles are overlapping, None otherwise.
    """
    pa, pb = a.position, b.position
    ra, rb = a.shape.radius, b.shape.radius
    
    d = pb - pa
    dist = float(np.linalg.norm(d))
    R = ra + rb
    
    if dist >= R:
        return None
    
    # Normal from a to b
    if dist > 1e-12:
        n = d / dist
    else:
        n = np.array([1.0, 0.0], dtype=np.float64)
    
    penetration = R - dist
    contact_point = pa + n * (ra - 0.5 * penetration)
    
    return Contact(
        a=a,
        b=b,
        point=contact_point,
        normal=n,
        penetration=penetration,
    )


def detect_contact(a: RigidBody2D, b: RigidBody2D) -> Contact | None:
    """
    Unified contact detection dispatcher.
    
    Uses specialized fast paths when available (circle-circle),
    falls back to GJK+EPA for general convex shapes.
    
    Args:
        a: First rigid body.
        b: Second rigid body.
        
    Returns:
        Contact object if bodies are intersecting, None otherwise.
    """
    # Fast path for circle-circle
    if isinstance(a.shape, Circle) and isinstance(b.shape, Circle):
        return circle_circle_contact_fast(a, b)
    
    # General convex path (Box-Box, Circle-Box, etc.)
    return convex_contact(a, b)
