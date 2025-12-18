import numpy as np
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle

def test_freefall_accuracy():
    """
    Analytic (constant g):
      y(t) = y0 + v0 t + 1/2 g t^2
      v(t) = v0 + g t
    """
    g = -9.81
    y0 = 10.0
    v0 = 0.0
    T = 1.0

    scene = Scene(gravity=(0,g), dt=1/240, integrator="rk4_adaptive", substeps=2, solver_iters=10)
    b = RigidBody2D(Circle(0.1), mass=1.0, position=(0.0, y0), velocity=(0.0, v0))
    scene.add_body(b)

    while scene.time < T - 1e-12:
        scene.step(min(scene.dt, T - scene.time))

    y_exp = y0 + v0*T + 0.5*g*T*T
    v_exp = v0 + g*T

    y_err = abs(b.position[1] - y_exp) / max(1e-9, abs(y_exp))
    v_err = abs(b.velocity[1] - v_exp) / max(1e-9, abs(v_exp))
    print("freefall y", b.position[1], "exp", y_exp, "relerr", y_err)
    print("freefall v", b.velocity[1], "exp", v_exp, "relerr", v_err)

    assert y_err <= 0.01
    assert v_err <= 0.01
