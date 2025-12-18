import numpy as np
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Box, Circle

def test_box_circle_repro():
    """
    Reproduction script for Box-Circle collision instability.
    A Circle falls onto a Box floor. It should settle, not fall through.
    """
    # Standard simulation parameters
    scene = Scene(dt=1/60, substeps=4, solver_iters=10)
    
    # 1. Box Floor (Static)
    # 50 width, 1 height.
    # Position (0, -1.0). Top surface at y = -0.5.
    floor = RigidBody2D(Box((50, 1)), mass=0, position=(0, -1.0)) 
    scene.add_body(floor)
    
    # 2. Dynamic Circle
    # Radius 0.5. Start at y = 2.0.
    circle = RigidBody2D(Circle(0.5), mass=1.0, position=(0, 2.0))
    scene.add_body(circle)
    
    print("Starting integration...")
    passed = True
    
    # Run for 2 seconds (120 frames)
    for i in range(120):
        scene.step()
        y = circle.position[1]
        
        # Check if it fell through
        # Floor top is -0.5. Circle radius is 0.5.
        # So Circle center should be >= 0.0 ideally.
        # Allow small penetration, but if y < -1.0 (center inside floor box center), it's definitely broken.
        if y < -1.0:
            print(f"FAIL: Frame {i}, Y={y:.4f} (Fell through floor)")
            passed = False
            break
            
        if i % 20 == 0:
            print(f"Frame {i}: Y={y:.4f}")

    if passed:
        # Check final position
        final_y = circle.position[1]
        print(f"Final Y: {final_y:.4f}")
        # Expect resting at ~0.5 (Floor Top 0.0 + Radius 0.5)
        if final_y > 0.4 and final_y < 0.6:
            print("SUCCESS: Circle resting on Box.")
        else:
            print("FAIL: Circle didn't settle correctly.")

if __name__ == "__main__":
    test_box_circle_repro()
