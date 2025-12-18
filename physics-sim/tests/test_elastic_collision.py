import numpy as np
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
from physics_sim.materials import Material

def test_elastic_headon_conservation():
    """
    1D elastic collision analytic:
      v1' = (m1-m2)/(m1+m2)*v1 + (2m2)/(m1+m2)*v2
      v2' = (2m1)/(m1+m2)*v1 + (m2-m1)/(m1+m2)*v2
    Also check momentum and kinetic energy conserved (within tolerance).
    """
    m1, m2 = 1.0, 2.0
    v1, v2 = 3.0, -1.0
    v1p = (m1-m2)/(m1+m2)*v1 + (2*m2)/(m1+m2)*v2
    v2p = (2*m1)/(m1+m2)*v1 + (m2-m1)/(m1+m2)*v2

    scene = Scene(gravity=(0,0), dt=1/600, integrator="rk4_adaptive", substeps=6, solver_iters=40)
    a = RigidBody2D(Circle(0.2), mass=m1, position=(-1.0, 0.0), velocity=(v1, 0.0),
                    material=Material(friction=0.0, restitution=1.0))
    b = RigidBody2D(Circle(0.2), mass=m2, position=(+1.0, 0.0), velocity=(v2, 0.0),
                    material=Material(friction=0.0, restitution=1.0))
    scene.add_body(a); scene.add_body(b)

    p0 = m1*a.velocity.copy() + m2*b.velocity.copy()
    ke0 = 0.5*m1*np.dot(a.velocity,a.velocity) + 0.5*m2*np.dot(b.velocity,b.velocity)

    for _ in range(1500):
        scene.step()

    p1 = m1*a.velocity + m2*b.velocity
    ke1 = 0.5*m1*np.dot(a.velocity,a.velocity) + 0.5*m2*np.dot(b.velocity,b.velocity)

    # Compare final velocities to analytic
    err_v1 = abs(a.velocity[0] - v1p) / max(1e-9, abs(v1p))
    err_v2 = abs(b.velocity[0] - v2p) / max(1e-9, abs(v2p))
    print("v1'", a.velocity[0], "exp", v1p, "relerr", err_v1)
    print("v2'", b.velocity[0], "exp", v2p, "relerr", err_v2)

    # Conservation checks
    dp = np.linalg.norm(p1 - p0) / max(1e-9, np.linalg.norm(p0))
    dke = abs(ke1 - ke0) / max(1e-9, abs(ke0))
    print("dp rel", dp, "dke rel", dke)

    assert err_v1 <= 0.01
    assert err_v2 <= 0.01
    assert dp <= 0.01
    assert dke <= 0.01
