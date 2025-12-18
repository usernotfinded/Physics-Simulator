# examples/minimal_freefall.py
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle

scene = Scene(gravity=(0.0, -9.81), dt=1/240, integrator="rk4_adaptive")

ball = RigidBody2D(
    shape=Circle(radius=0.1),
    mass=1.0,
    position=(0.0, 10.0),
    velocity=(0.0, 0.0),
)
scene.add_body(ball)

t_end = 1.0
while scene.time < t_end:
    scene.step()

print("t:", scene.time)
print("pos:", scene.bodies[0].position)
print("vel:", scene.bodies[0].velocity)
