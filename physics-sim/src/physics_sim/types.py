# MIT License (see LICENSE)
"""
Core type definitions for the 2D physics simulation.

Defines the fundamental data structures:
- Shape primitives (Circle, Box)
- RigidBody2D: The main simulation entity with position, velocity, mass, etc.

The equations of motion are documented in maths.md and follow standard
Newtonian mechanics for rigid bodies in 2D:
  - Linear:  F = m·a  →  a = F/m
  - Angular: τ = I·α  →  α = τ/I
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .materials import Material
from .util import f64


# =============================================================================
# Shape Definitions
# =============================================================================

@dataclass(frozen=True)
class Circle:
    """
    Circular shape defined by radius.
    
    Attributes:
        radius: Distance from center to edge in meters.
    """
    radius: float


@dataclass(frozen=True)
class Box:
    """
    Axis-aligned box shape defined by half-extents.
    
    The full width is 2*hx and full height is 2*hy.
    The box center is at the body's position.
    
    Attributes:
        half_extents: Tuple (hx, hy) representing half-width and half-height.
    """
    half_extents: tuple[float, float]


# Union type for shape dispatch
@dataclass(frozen=True)
class ConvexPolygon:
    """
    General convex polygon defined by vertices.
    
    Attributes:
        vertices: Array of vertices [N, 2] in local space, ordered counter-clockwise.
                  Must be convex and winding must be CCW.
                  The center of mass is assumed to be at (0,0) for simple simulation,
                  or users must adjust position accordingly.
    """
    vertices: np.ndarray

    def __post_init__(self) -> None:
        """Ensure vertices are stored as float64."""
        object.__setattr__(self, "vertices", f64(self.vertices))


# Union type for shape dispatch
Shape2D = Circle | Box | ConvexPolygon


# =============================================================================
# Rigid Body
# =============================================================================

@dataclass
class RigidBody2D:
    """
    A 2D rigid body with full kinematic and dynamic state.
    
    Implements the equations of motion from maths.md Eq (1)-(4):
      dx/dt = v           (position rate of change)
      dv/dt = F/m         (velocity rate of change)
      dθ/dt = ω           (angle rate of change)
      dω/dt = τ/I         (angular velocity rate of change)
    
    Attributes:
        shape: Collision geometry (Circle, Box, or ConvexPolygon).
        mass: Mass in kg. Use mass ≤ 0 for static/immovable bodies.
        position: Center of mass position [x, y] in meters.
        angle: Rotation angle in radians (counterclockwise from +x axis).
        velocity: Linear velocity [vx, vy] in m/s.
        omega: Angular velocity in rad/s (counterclockwise positive).
        material: Surface properties (friction, restitution).
        charge: Electric charge in Coulombs for EM simulation.
        force: Accumulated force vector [Fx, Fy] (cleared each step).
        torque: Accumulated torque scalar (cleared each step).
        id: Unique identifier assigned by Scene.add_body().
    
    Note:
        Position and velocity are converted to float64 numpy arrays on init.
        Force accumulation happens during Scene._apply_forces(), then
        forces are integrated and cleared for the next step.
    """
    shape: Shape2D
    mass: float
    position: np.ndarray | tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    velocity: np.ndarray | tuple[float, float] = (0.0, 0.0)
    omega: float = 0.0
    material: Material = field(default_factory=Material)
    charge: float = 0.0

    # Sleep state
    sleeping: bool = False
    sleep_time: float = 0.0
    can_sleep: bool = True  # Dynamic bodies can sleep by default
    
    # Runtime state (not user-specified)
    force: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    torque: float = 0.0
    id: int = -1

    def wake(self) -> None:
        """Force the body to wake up."""
        if self.sleeping:
            self.sleeping = False
            self.sleep_time = 0.0

    def sleep(self) -> None:
        """Force the body to sleep (zeroes velocity)."""
        if self.can_sleep:
            self.sleeping = True
            self.velocity.fill(0.0)
            self.omega = 0.0
            self.force.fill(0.0)
            self.torque = 0.0

    def __post_init__(self) -> None:
        """Convert position/velocity to float64 arrays for consistent numerics."""
        self.position = f64(self.position)
        self.velocity = f64(self.velocity)
        self.force = f64(self.force)

    @property
    def inv_mass(self) -> float:
        """Inverse mass (1/m). Returns 0 for static bodies (mass ≤ 0)."""
        return 0.0 if self.mass <= 0 else 1.0 / self.mass

    @property
    def inertia(self) -> float:
        """
        Moment of inertia about center of mass.
        
        Formulas for solid 2D shapes:
          Circle: I = (1/2) m r²
          Box:    I = (1/12) m (w² + h²)  where w=2*hx, h=2*hy
          Polygon: Computed using Shoelace-like formula for second moment of area.
                  See: https://en.wikipedia.org/wiki/Second_moment_of_area#Polygon
        
        Reference: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        """
        if isinstance(self.shape, Circle):
            r = self.shape.radius
            return 0.5 * self.mass * r * r
        if isinstance(self.shape, Box):
            hx, hy = self.shape.half_extents
            w, h = 2 * hx, 2 * hy
            return (1 / 12) * self.mass * (w * w + h * h)
        if isinstance(self.shape, ConvexPolygon):
            # Calculate polar moment of inertia for a polygon about (0,0)
            # Assumption: (0,0) is the center of mass (or close enough for this approx)
            # Numerator = sum( ||v_i x v_{i+1}|| * (v_i^2 + v_i.v_{i+1} + v_{i+1}^2) )
            # Denominator = 6 * sum( ||v_i x v_{i+1}|| )
            # Mass scaling: I = (mass / area) * integral(r^2 dA)
            
            verts = self.shape.vertices
            numerator = 0.0
            denominator = 0.0
            
            # Iterate edges
            for i in range(len(verts)):
                p1 = verts[i]
                p2 = verts[(i + 1) % len(verts)]
                
                # Signed double-area of triangle (0, p1, p2)
                cross = p1[0]*p2[1] - p1[1]*p2[0]
                
                numerator += cross * (np.dot(p1, p1) + np.dot(p1, p2) + np.dot(p2, p2))
                denominator += cross
                
            if abs(denominator) < 1e-9:
                return 1.0 # Degenerate polygon
                
            # I_polar = numerator / 12 (geometric property)
            # But we need Mass moment of inertia.
            # I_mass = mass * (I_polar / Area)
            # Area = 0.5 * denominator
            
            area = 0.5 * denominator
            I_polar = numerator / 6.0  # Wait, formula is sum / 6? 
            # Correct formula for Inertia about origin:
            # I = (1/6) * sum( |cross| * (v1^2 + v1.v2 + v2^2) ) / sum(|cross|) * mass?
            # Standard: I = (mass / 6) * (sum(cross * (..)) / sum(cross))
            return (self.mass / 6.0) * (numerator / denominator)

        raise TypeError(f"Unknown shape type: {type(self.shape)}")

    @property
    def inv_inertia(self) -> float:
        """Inverse moment of inertia (1/I). Returns 0 for static bodies."""
        I = self.inertia
        return 0.0 if I <= 0 else 1.0 / I

    def clear_forces(self) -> None:
        """Reset accumulated force and torque to zero for next timestep."""
        self.force[:] = 0.0
        self.torque = 0.0

    def local_to_world(self, local_point: tuple[float, float] | np.ndarray) -> np.ndarray:
        """Transform a point from local body coordinates to world coordinates."""
        # p_world = pos + Rot * p_local
        c, s = np.cos(self.angle), np.sin(self.angle)
        lx, ly = local_point[0], local_point[1]
        
        wx = lx * c - ly * s + self.position[0]
        wy = lx * s + ly * c + self.position[1]
        return np.array([wx, wy], dtype=np.float64)

    def world_to_local(self, world_point: tuple[float, float] | np.ndarray) -> np.ndarray:
        """Transform a point from world coordinates to local body coordinates."""
        # p_local = Rot^T * (p_world - pos)
        dx = world_point[0] - self.position[0]
        dy = world_point[1] - self.position[1]
        c, s = np.cos(self.angle), np.sin(self.angle)
        
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        return np.array([lx, ly], dtype=np.float64)
