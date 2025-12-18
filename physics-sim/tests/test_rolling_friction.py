import numpy as np
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
from physics_sim.materials import Material

def test_disk_slip_to_roll_analytic():
    """
    Rolling+friction model (kinetic friction while slipping):
    For a solid disk: I = 1/2 m R^2.
    Define slip velocity: v_rel = v - ωR.
    With kinetic friction magnitude f = μ m g opposing slip:
      a = dv/dt = -μ g sgn(v_rel)
      α = dω/dt = +(f R)/I = (μ m g R)/(1/2 m R^2) = 2 μ g / R   (same sign as -v_rel)
    For v_rel > 0:
      dv/dt = -μ g
      dω/dt = +2 μ g / R
      => d(v_rel)/dt = -3 μ g
      => t* = v_rel0 / (3 μ g)
    At t*:
      v* = v0 - μ g t*
      ω* = ω0 + 2 μ g t* / R
    Energy loss due to slip work:
      W = -f * ∫ v_rel dt = -μ m g * (v_rel0^2 / (6 μ g)) = - m v_rel0^2 / 6
    """
    g = 9.81
    mu = 0.3
    m = 2.0
    R = 0.5

    v0 = 4.0
    w0 = 0.0
    vrel0 = v0 - w0*R
    tstar = vrel0 / (3*mu*g)

    v_exp = v0 - mu*g*tstar
    w_exp = w0 + (2*mu*g/R)*tstar

    # We implement friction here as a contact-like force against a "ground" via a custom loop:
    # for this verification, we integrate with the engine but apply the analytic friction force/torque directly.
    scene = Scene(gravity=(0,-g), dt=1/2000, integrator="rk4_adaptive", substeps=1, solver_iters=1)
    disk = RigidBody2D(Circle(R), mass=m, position=(0.0, 0.0), velocity=(v0, 0.0), omega=w0,
                       material=Material(friction=mu, restitution=0.0))
    scene.add_body(disk)

    # simulate only until t* applying kinetic friction on horizontal motion (no vertical dynamics used here)
    t_end = tstar
    while scene.time < t_end - 1e-12:
        dt = min(scene.dt, t_end - scene.time)
        # forces: cancel gravity vertically (ground normal) and apply kinetic friction horizontally + torque.
        disk.clear_forces()
        # normal force N = m g
        # friction force = -μ N sgn(v_rel) along +x
        vrel = disk.velocity[0] - disk.omega*R
        sgn = 1.0 if vrel > 0 else (-1.0 if vrel < 0 else 0.0)
        f = -mu * m * g * sgn
        disk.force += np.array([f, 0.0], dtype=np.float64)
        # torque τ = -f * R (opposes relative motion at contact), sign consistent with ω increase when vrel>0
        disk.torque += (-f) * R  # if f negative, torque positive
        scene._integrate(dt)
        scene.time += dt

    err_v = abs(disk.velocity[0] - v_exp) / max(1e-9, abs(v_exp))
    err_w = abs(disk.omega - w_exp) / max(1e-9, abs(w_exp))
    print("v*", disk.velocity[0], "exp", v_exp, "relerr", err_v)
    print("w*", disk.omega, "exp", w_exp, "relerr", err_w)

    assert err_v <= 0.01
    assert err_w <= 0.01
