import numpy as np
import pytest
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Box, Circle
from physics_sim.constraints.solver import MouseJoint

def test_query_point():
    """Verify scene.query_point works for Boxes and Circles."""
    scene = Scene()
    
    # Circle at (0, 0) r=1
    circle = RigidBody2D(Circle(1.0), mass=1.0, position=(0, 0))
    scene.add_body(circle)
    
    assert scene.query_point((0, 0)) == circle
    assert scene.query_point((0.9, 0)) == circle
    assert scene.query_point((1.1, 0)) is None
    
    # Box at (5, 0) w=2, h=2 (half=1)
    box = RigidBody2D(Box((1.0, 1.0)), mass=1.0, position=(5, 0))
    scene.add_body(box)
    
    assert scene.query_point((5, 0)) == box
    assert scene.query_point((5.9, 0.9)) == box
    assert scene.query_point((6.1, 0)) is None

def test_mouse_drag():
    """Verify MouseJoint moves a body."""
    scene = Scene(gravity=(0, 0), dt=1/60) # Zero gravity to isolate drag
    
    # Box at origin
    box = RigidBody2D(Box((1, 1)), mass=1.0, position=(0, 0))
    scene.add_body(box)
    
    # Create MouseJoint
    # Click at (0,0) (center)
    target = (1.0, 0.0)
    local_anchor = box.world_to_local((0, 0)) # Should be (0,0)
    
    joint = MouseJoint(
        body=box,
        target=target, # Dragging to right immediately
        local_anchor=local_anchor,
        max_force=1000.0,
        frequency_hz=5.0,
        damping_ratio=0.7
    )
    scene.add_mouse_joint(joint)
    
    # Step
    for _ in range(10):
        scene.step()
        
    # Body should have moved right
    print(f"Box Pos: {box.position}")
    assert box.position[0] > 0.05, "Box should move towards target"
    assert abs(box.position[1]) < 1e-3
    
    # Move target further
    joint.set_target((2.0, 0.0))
    for _ in range(10):
        scene.step()
        
    assert box.position[0] > 0.5, "Box should follow target"

def test_max_force():
    """Verify MouseJoint limits force."""
    scene = Scene(gravity=(0, 0), dt=1/60)
    
    # Heavy Box
    box = RigidBody2D(Box((1, 1)), mass=100.0, position=(0, 0))
    scene.add_body(box)
    
    # Weak Joint
    joint = MouseJoint(
        body=box,
        target=(100.0, 0.0), # Far away
        local_anchor=(0, 0),
        max_force=10.0, # Very weak limit
        frequency_hz=5.0
    )
    scene.add_mouse_joint(joint)
    
    scene.step()
    
    # Force = mass * accel
    # acc constraint? 
    # Max force is 10. Mass 100. Max accel = 0.1.
    # Vel after 1/60 should be approx 0.1 * dt = 0.1/60 = 0.0016
    v = np.linalg.norm(box.velocity)
    print(f"Vel: {v}")
    
    # Check that force didn't explode despite huge distance
    # Impulse = change_in_momentum = m * v
    # Force = Impulse / dt
    force = (box.mass * v) / scene.dt
    print(f"Applied Force: {force}")
    
    # It might slightly exceed due to Baumgarte or discrete nature, but should be close to 10.
    # Note: Soft constraints are weird with max_force.
    # The max_force clamps the accumulated impulse.
    # So max impulse per step = 10 * dt.
    assert force < 12.0, f"Force {force} exceeded max_force 10"

if __name__ == "__main__":
    test_query_point()
    test_mouse_drag()
    test_max_force()
