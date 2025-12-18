from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
from physics_sim.materials import Material
import numpy as np

scene = Scene(dt=1/240, integrator="rk4_adaptive", substeps=4, solver_iters=40)

anchor = RigidBody2D(Circle(0.05), mass=0.0, position=(0.0, 0.0), material=Material(friction=0, restitution=0))
bob = RigidBody2D(Circle(0.05), mass=1.0, position=(0.2, -1.0), material=Material(friction=0, restitution=0))
scene.add_body(anchor); scene.add_body(bob)

L = 1.0
scene.add_distance_constraint(anchor, bob, length=L, beta=0.25)

for _ in range(240):
    scene.step()

print("bob position:", bob.position, "distance:", float(np.linalg.norm(bob.position - anchor.position)))
