import numpy as np
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
from physics_sim.materials import Material

def test_pendulum_small_angle():
    """
    Small-angle analytic pendulum:
      θ(t) = θ0 cos( sqrt(g/L) t )
    We simulate a bob constrained at distance L from anchor using DistanceConstraint.
    Compare θ at t=T.
    """
    g = 9.81
    L = 1.0
    theta0 = 0.15  # small angle [rad]
    T = 1.0

    scene = Scene(gravity=(0,-g), dt=1/240, integrator="rk4_adaptive", substeps=4, solver_iters=60)

    anchor = RigidBody2D(Circle(0.05), mass=0.0, position=(0.0, 0.0), material=Material(0,0))
    bob = RigidBody2D(Circle(0.05), mass=1.0,
                      position=(L*np.sin(theta0), -L*np.cos(theta0)),
                      velocity=(0.0, 0.0),
                      material=Material(0,0))
    scene.add_body(anchor); scene.add_body(bob)
    scene.add_distance_constraint(anchor, bob, length=L, beta=0.3)

    while scene.time < T - 1e-12:
        scene.step(min(scene.dt, T - scene.time))

    # compute simulated angle from position
    x, y = bob.position[0], bob.position[1]
    theta_sim = np.arctan2(x, -y)

    omega0 = np.sqrt(g/L)
    theta_exp = theta0 * np.cos(omega0 * T)

    err = abs(theta_sim - theta_exp) / max(1e-9, abs(theta_exp))
    print("theta", theta_sim, "exp", theta_exp, "relerr", err)
    assert err <= 0.01
