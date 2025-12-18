# MIT License (see LICENSE)
"""
Broadphase collision detection using spatial hashing.

This module provides efficient collision pair culling by partitioning space into
a uniform grid. Bodies are hashed into grid cells based on their axis-aligned
bounding boxes (AABBs), and only bodies sharing a cell are considered as potential
collision pairs.

Key concepts:
- AABB (Axis-Aligned Bounding Box): Conservative bounding region for a shape.
- Spatial hashing: O(1) expected cell lookup for broad phase culling.
- The output is a list of (bodyA, bodyB) pairs that may be colliding.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Iterator

import numpy as np

from ..types import RigidBody2D, Circle, Box, ConvexPolygon


def aabb_for_body(body: RigidBody2D) -> tuple[float, float, float, float]:
    """
    Calculate Axis-Aligned Bounding Box (min_x, min_y, max_x, max_y).
    """
    # Box: use conservative circular bound (rotation-invariant)
    # This is cheaper than transforming 4 vertices but looser.
    # For AABB broadphase, loose is fine.
    
    if isinstance(body.shape, Circle):
        r = body.shape.radius
        p = body.position
        return (p[0] - r, p[1] - r, p[0] + r, p[1] + r)
    
    if isinstance(body.shape, Box):
        # Optimized: rotate box extents locally
        # World extent W = |h . R| ...
        # Standard OBB AABB:
        # ex = |c*hx| + |s*hy|
        # ey = |s*hx| + |c*hy|
        hx, hy = body.shape.half_extents
        angle = body.angle
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        
        ex = abs(c * hx) + abs(s * hy)
        ey = abs(s * hx) + abs(c * hy)
        
        px, py = body.position
        return (px - ex, py - ey, px + ex, py + ey)
        
    if isinstance(body.shape, ConvexPolygon):
        # Must transform all vertices
        angle = body.angle
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        pos = body.position
        
        # Rotation matrix R = [[c, -s], [s, c]]
        # v_world = R * v_local + pos
        
        verts = body.shape.vertices
        
        # Vectorized transform
        # x_w = v_x * c - v_y * s + px
        # y_w = v_x * s + v_y * c + py
        
        vx = verts[:, 0]
        vy = verts[:, 1]
        
        wx = vx * c - vy * s + pos[0]
        wy = vx * s + vy * c + pos[1]
        
        return (float(np.min(wx)), float(np.min(wy)), float(np.max(wx)), float(np.max(wy)))

    raise TypeError(f"Unknown shape type: {type(body.shape)}")


class SpatialHashBroadphase:
    """
    Spatial hash grid for broadphase collision detection.
    
    Partitions 2D space into cells of uniform size and uses hashing to
    efficiently find potential collision pairs. Bodies are inserted into
    all cells their AABB overlaps, and pairs are generated from bodies
    sharing at least one cell.
    
    Attributes:
        cell: The size of each grid cell in world units.
        
    Example:
        broadphase = SpatialHashBroadphase(cell_size=2.0)
        pairs = broadphase.pairs(scene.bodies)
        for a, b in pairs:
            # Narrow phase collision check
            ...
    """
    
    def __init__(self, cell_size: float = 1.0) -> None:
        """
        Initialize the spatial hash grid.
        
        Args:
            cell_size: Size of each grid cell in world units. Larger cells
                       reduce insertion cost but increase false positives.
        """
        self.cell = float(cell_size)

    def _cells_for_aabb(self, aabb: tuple[float, float, float, float]) -> Iterator[tuple[int, int]]:
        """
        Yield all grid cell coordinates that overlap with an AABB.
        
        Args:
            aabb: Bounding box as (x_min, y_min, x_max, y_max).
            
        Yields:
            (ix, iy) integer cell coordinates.
        """
        x0, y0, x1, y1 = aabb
        cs = self.cell
        ix0, iy0 = int(np.floor(x0 / cs)), int(np.floor(y0 / cs))
        ix1, iy1 = int(np.floor(x1 / cs)), int(np.floor(y1 / cs))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield (ix, iy)

    def pairs(self, bodies: list[RigidBody2D]) -> list[tuple[RigidBody2D, RigidBody2D]]:
        """
        Find all potential collision pairs among the given bodies.
        
        Each body is inserted into all cells its AABB overlaps. Pairs are
        generated from bodies that share at least one cell. Duplicate pairs
        are eliminated using a seen set.
        
        Args:
            bodies: List of rigid bodies to check for potential collisions.
            
        Returns:
            List of (bodyA, bodyB) tuples representing potential collision pairs,
            sorted by body IDs for deterministic ordering.
        """
        grid: dict[tuple[int, int], list[RigidBody2D]] = defaultdict(list)
        
        for b in bodies:
            aabb = aabb_for_body(b)
            for c in self._cells_for_aabb(aabb):
                grid[c].append(b)
        
        seen: set[tuple[int, int]] = set()
        out: list[tuple[RigidBody2D, RigidBody2D]] = []
        
        for cell, bs in grid.items():
            bs = sorted(bs, key=lambda x: x.id)
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    a, b = bs[i], bs[j]
                    key = (a.id, b.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((a, b))
        
        return out
