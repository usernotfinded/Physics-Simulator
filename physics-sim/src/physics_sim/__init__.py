# MIT License (see LICENSE)
"""
physics_sim - A 2D rigid body physics simulation engine.

This package provides core simulation capabilities including rigid body
dynamics, collision detection, constraint solving, and optional rendering.

Main entry points:
    - Scene: The simulation world containing bodies and constraints.
    - RigidBody2D: A rigid body with shape, mass, and kinematic state.
    - Circle, Box: Shape definitions.
    - Material: Physical properties (friction, restitution).

Submodules:
    - collision: Broadphase, narrowphase, and contact resolution.
    - constraints: Distance and other constraint solvers.
    - core: Force generators and integrators.
    - io: JSON serialization/deserialization.
    - renderer: Optional visualization adapters.

Example:
    from physics_sim import Scene, RigidBody2D, Circle
    
    scene = Scene(gravity=(0, -9.81))
    ball = RigidBody2D(shape=Circle(0.5), mass=1.0, position=(0, 10))
    scene.add_body(ball)
    scene.step()
"""
from .scene import Scene
from .types import RigidBody2D, Circle, Box
from .materials import Material

__all__ = [
    # Core simulation
    "Scene",
    "RigidBody2D",
    # Shapes
    "Circle",
    "Box",
    # Materials
    "Material",
]
