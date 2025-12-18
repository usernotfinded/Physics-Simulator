from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
import numpy as np

scene = Scene(gravity=(0,0), dt=1/2000, integrator="rk4_adaptive", substeps=2, solver_iters=10,
              enable_em=True, coulomb_eps=1e-3)

# Two opposite charges; not a stable "orbit" in classical EM without radiation, but shows interaction
a = RigidBody2D(Circle(0.05), mass=1.0, charge=+2e-6, position=(-0.5, 0.0), velocity=(0.0, 0.6))
b = RigidBody2D(Circle(0.05), mass=1.0, charge=-2e-6, position=(+0.5, 0.0), velocity=(0.0, -0.6))
scene.add_body(a); scene.add_body(b)

for _ in range(4000):
    scene.step()

print("a pos", a.position, "v", a.velocity)
print("b pos", b.position, "v", b.velocity)
