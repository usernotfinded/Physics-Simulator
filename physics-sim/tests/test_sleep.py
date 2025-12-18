# MIT License (see LICENSE)
import pytest
import numpy as np
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle, Box

def test_single_body_sleep():
    """
    Verify a single body on the floor settles and goes to sleep.
    """
    scene = Scene(dt=1/60, substeps=4, solver_iters=10)
    
    # Floor (static)
    floor = RigidBody2D(Box((50, 1)), mass=0, position=(0, -1.0))
    # floor = RigidBody2D(Circle(50.0), mass=0, position=(0, -50.5)) # Top at -0.5
    scene.add_body(floor)
    
    # Dynamic Circle
    circle = RigidBody2D(Circle(0.5), mass=1.0, position=(0, 2.0))
    scene.add_body(circle)
    
    # Run simulation
    # 1. Fall (awake)
    for _ in range(60): # 1 sec
        scene.step()
    
    assert not circle.sleeping, "Should be awake while falling/bouncing"
    
    # 2. Settle (eventually sleep)
    # Circle on box is stable, settles fast.
    # Sleep threshold 0.5s.
    # Wait 3 seconds
    for _ in range(180):
        scene.step()
        
    assert circle.sleeping, "Body should be sleeping after settling"
    assert np.allclose(circle.velocity, 0), "Velocity should be zeroed"
    
    # 3. Wake by Force
    circle.wake()
    scene._apply_forces() # Force wake doesn't add force, just enables integration
    assert not circle.sleeping
    
    # Re-sleep
    for _ in range(60): # 1s wait should re-sleep
        scene.step()
    assert circle.sleeping

def test_wake_on_impact():
    """
    Verify a sleeping body wakes up when hit.
    """
    scene = Scene(dt=1/60, substeps=4, solver_iters=10)
    
    # Floor - Use Circle to avoid Box-GJK issues
    # floor = RigidBody2D(Circle(50.0), mass=0, position=(0, -50.5))
    floor = RigidBody2D(Box((50, 1)), mass=0, position=(0, -1.0))
    scene.add_body(floor)
    
    # Target (sleeping candidate)
    # Target Radius 0.5. Floor Radius 50. Floor Y -50.5. Top -0.5.
    # Target Center Y 0.0. Bottom -0.5.
    # target = RigidBody2D(Circle(0.5), mass=1.0, position=(0, 0.0)) 
    # target.position = np.array([0.0, 0.001], dtype=np.float64)    # Target (sleeping candidate)
    # Floor Top 0.0 (Box height 2, Pos -1).
    # Target Radius 0.5. Center at 0.5+.
    target = RigidBody2D(Circle(0.5), mass=1.0, position=(0, 0.501))
    scene.add_body(target)
    
    # Let it sleep
    for _ in range(60):
        scene.step()
        
    assert target.sleeping, "Target should be asleep"
    
    # Bullet
    # Target Center 0.5. Bullet must hit it. 
    # Bullet Y 0.5. Impact Head on.
    bullet = RigidBody2D(Circle(0.2), mass=0.5, position=(-5.0, 0.5)) 
    bullet.velocity = np.array([10.0, 0.0], dtype=np.float64)
    scene.add_body(bullet)
    
    # Simulate impact
    hit = False
    for _ in range(60):
        scene.step()
        if not target.sleeping:
            hit = True
            break
            
    assert hit, "Target should wake up after collision"
    assert np.linalg.norm(target.velocity) > 0.1, "Target should move"

def test_island_sleep():
    """
    Verify two touching bodies sleep together and wake together.
    """
    scene = Scene(dt=1/60, substeps=4, solver_iters=10)
    # Floor
    # floor = RigidBody2D(Circle(50.0), mass=0, position=(0, -50.5))
    floor = RigidBody2D(Box((50, 1)), mass=0, position=(0, -1.0))
    scene.add_body(floor)
    
    # A and B touching vertically (Stack)
    # Floor top 0.0.
    # A Radius 0.5. Center 0.5+.
    # B Radius 0.5. Center 1.5+.
    a = RigidBody2D(Circle(0.5), mass=1.0, position=(0.0, 0.501))
    b = RigidBody2D(Circle(0.5), mass=1.0, position=(0.0, 1.502))
    scene.add_body(a)
    scene.add_body(b)
    
    # Let them settle/sleep
    for _ in range(120):
        scene.step()
    
    # Verify contacts exist (Floor-A, A-B)
    assert len(scene.contact_manager.manifolds) >= 2, f"Contacts lost! {len(scene.contact_manager.manifolds)}"
    print(f"Contacts: {len(scene.contact_manager.manifolds)}")

    assert a.sleeping and b.sleeping, "Both should sleep"
    
    # Wake A only (e.g. via force)
    # We simulate this by manually setting velocity and waking?
    # Or just calling wake() on A.
    # Island Manager sees A moving -> Wakes B?
    # Ideally: calling wake() on A makes A active.
    # Next step: Island logic sees A active -> Checks connectivity -> Wakes B.
    
    a.wake()
    assert not a.sleeping
    assert b.sleeping # B still sleeping technically until update
    
    # Run 1 step
    scene.step()
    
    # Both should be awake
    # Note: If A doesn't move, IslandManager might put them back to sleep?
    # A.velocity is 0.
    # A.sleep_time resets to 0.
    # Island Check:
    # A (v=0) -> Sleepy? Yes.
    # B (v=0) -> Sleepy? Yes.
    # But A.sleep_time is 0.
    # So Island NOT ready to sleep.
    # So Island Wakes.
    # B.wake() called.
    
    assert not a.sleeping
    assert not b.sleeping, "B should wake because A woke (island constraint)"
