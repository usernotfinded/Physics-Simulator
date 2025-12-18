from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
from physics_sim.materials import Material
import numpy as np

scene = Scene(gravity=(0,0), dt=1/600, integrator="rk4_adaptive", substeps=6, solver_iters=30)

m1, m2 = 1.0, 2.0
a = RigidBody2D(Circle(0.2), mass=m1, position=(-1.0, 0.0), velocity=(+3.0, 0.0),
                material=Material(friction=0.0, restitution=1.0))
b = RigidBody2D(Circle(0.2), mass=m2, position=(+1.0, 0.0), velocity=(-1.0, 0.0),
                material=Material(friction=0.0, restitution=1.0))
scene.add_body(a); scene.add_body(b)

p0 = m1*np.array(a.velocity) + m2*np.array(b.velocity)
ke0 = 0.5*m1*np.dot(a.velocity,a.velocity) + 0.5*m2*np.dot(b.velocity,b.velocity)

for _ in range(1500):
    scene.step()

p1 = m1*a.velocity + m2*b.velocity
ke1 = 0.5*m1*np.dot(a.velocity,a.velocity) + 0.5*m2*np.dot(b.velocity,b.velocity)

print("p0", p0, "p1", p1, "dp", p1-p0)
print("ke0", ke0, "ke1", ke1, "dke", ke1-ke0)
print("v_final a,b:", a.velocity, b.velocity)
