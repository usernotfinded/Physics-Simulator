import numpy as np
import pytest
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Box, Circle
from physics_sim.io.json_io import scene_to_json, load_scene, save_scene
import os

def test_revolute_joint():
    """Verify separate anchor points and constraint satisfaction."""
    scene = Scene(gravity=(0, 0), dt=0.01)
    
    # Body A: Static at 0,0
    a = RigidBody2D(Box((1,1)), mass=0, position=(0,0))
    # Body B: Dynamic, initially at (2,0)
    b = RigidBody2D(Box((1,1)), mass=1, position=(2,0))
    
    scene.add_body(a)
    scene.add_body(b)
    
    # Connect Anchor A (1,0) [right edge of A] to Anchor B (-1,0) [left edge of B]
    # In global space, A is at 0,0 -> anchor at 1,0. B is at 2,0 -> anchor at 1,0.
    # So they match perfectly.
    # Use higher beta for stiffer constraint in test
    scene.add_revolute_constraint(a, b, anchor_a=(1.0, 0.0), anchor_b=(-1.0, 0.0), beta=0.5)
    
    # Apply initial velocity to B to swing it
    # Pushing UP means it should circle around anchor A.
    b.velocity = np.array([0.0, 1.0], dtype=np.float64)
    
    scene.step()
    
    # Check constraint maintenance
    # World anchor A
    wa = a.position + np.array([1.0, 0.0]) # Rot is 0
    # World anchor B (Rot B might change)
    c, s = np.cos(b.angle), np.sin(b.angle)
    wb = b.position + np.array([-1.0*c - 0.0*s, -1.0*s + 0.0*c])
    
    dist = np.linalg.norm(wa - wb)
    # Drift is expected with Baumgarte (velocity correction for next frame)
    # Unconstrained drift would be 1.0 * 0.01 = 0.01
    assert dist < 0.002, f"Constraint violation: {dist}"
    # Verify it actually moved
    assert np.linalg.norm(b.position - np.array([2.0, 0.0])) > 1e-5

def test_prismatic_joint():
    """Verify sliding constraint (locked angle, locked perp pos)."""
    scene = Scene(gravity=(0, 0), dt=0.01)
    
    a = RigidBody2D(Box((1,1)), mass=0, position=(0,0))
    b = RigidBody2D(Box((1,1)), mass=1, position=(2,0))
    
    scene.add_body(a)
    scene.add_body(b)
    
    # Constrain B to slide along X axis of A
    # Use higher beta
    scene.add_prismatic_constraint(
        a, b, anchor_a=(0,0), axis_a=(1,0), beta=0.5
    )
    
    # 1. Apply velocity in Y -> should be corrected/stopped
    b.velocity = np.array([0.0, 1.0], dtype=np.float64)
    scene.step() # Solver should kill Y velocity
    
    # Velocity constraint is checked on next step implicitly/explicitly
    # Check position drift
    assert abs(b.position[1]) < 0.002, f"Prismatic joint failed to lock Y axis, pos={b.position[1]}"
    assert abs(b.velocity[1]) < 0.1, f"Prismatic joint failed to kill Y velocity, vel={b.velocity[1]}"
    
    # 2. Apply velocity in X -> should move in X
    b.velocity = np.array([1.0, 0.0], dtype=np.float64)
    expected_x = b.position[0] + 1.0 * 0.01 # approx
    scene.step()
    assert b.position[0] > 2.005, "Prismatic joint prevented valid X movement"
    
    # 3. Apply angular velocity -> should stop rotation
    b.omega = 10.0
    scene.step()
    assert abs(b.angle) < 0.1, f"Prismatic joint failed to lock rotation, angle={b.angle}"


def test_constraints_json_io(tmp_path):
    """Verify JSON round-trip for all constraints."""
    scene = Scene()
    b1 = RigidBody2D(Circle(1), mass=0, position=(0,0))
    b2 = RigidBody2D(Circle(1), mass=1, position=(2,0))
    scene.add_body(b1)
    scene.add_body(b2)
    
    scene.add_distance_constraint(b1, b2, length=2.0)
    scene.add_revolute_constraint(b1, b2, (1,0), (-1,0))
    scene.add_prismatic_constraint(b1, b2, (0,0), (0,1))
    
    # Serialize
    data = scene_to_json(scene)
    assert len(data["constraints"]) == 3
    
    # Save/Load
    p = tmp_path / "test_scene.json"
    save_scene(scene, str(p))
    
    scene2 = load_scene(str(p))
    
    assert len(scene2.constraints) == 1
    assert len(scene2.revolute_constraints) == 1
    assert len(scene2.prismatic_constraints) == 1
    
    # Verify properties
    rc = scene2.revolute_constraints[0]
    assert rc.anchor_a == (1.0, 0.0)
    
    pc = scene2.prismatic_constraints[0]
    assert pc.axis_a == (0.0, 1.0)
