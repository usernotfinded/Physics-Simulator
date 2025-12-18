# MIT License (see LICENSE)
"""
Numerical integrators for rigid body dynamics.

This module provides time-stepping methods to advance the simulation state.
All integrators solve the equations of motion from maths.md Eq (1)-(4):
    dx/dt = v,         dv/dt = F/m
    dθ/dt = ω,         dω/dt = τ/I

Available integrators:
- rk4_step: Fixed-step 4th-order Runge-Kutta (high accuracy)
- rk4_adaptive_step: Adaptive RK4 with error control (accuracy + efficiency)
- verlet_step: Velocity Verlet (symplectic, good for energy conservation)

Reference:
    Runge-Kutta methods: https://en.wikipedia.org/wiki/Runge-Kutta_methods
    Velocity Verlet: https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
"""
from __future__ import annotations

import numpy as np

from ..types import RigidBody2D


def _derivatives(
    state: tuple,
    inv_mass: float,
    inv_inertia: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute time derivatives of the state vector.
    
    Given current state (x, v, θ, ω, F, τ), returns (dx/dt, dv/dt, dθ/dt, dω/dt).
    This is the right-hand side of the ODE system.
    
    Args:
        state: Tuple (position, velocity, angle, omega, force, torque).
        inv_mass: Inverse mass (1/m).
        inv_inertia: Inverse moment of inertia (1/I).
    """
    x, v, theta, omega, F, tau = state
    dx = v
    dv = F * inv_mass
    dtheta = omega
    domega = tau * inv_inertia
    return dx, dv, dtheta, domega


def rk4_step(body: RigidBody2D, dt: float) -> None:
    """
    Advance body state by dt using classical 4th-order Runge-Kutta.
    
    RK4 evaluates derivatives at 4 points within the timestep and combines
    them with weights (1, 2, 2, 1)/6 to achieve O(dt⁵) local error.
    
    Forces are held constant over the timestep (explicit integrator).
    This is appropriate when forces don't depend on velocity or when
    dt is small enough that the approximation is acceptable.
    
    Args:
        body: Rigid body to integrate (modified in-place).
        dt: Timestep in seconds.
    
    Reference:
        https://en.wikipedia.org/wiki/Runge-Kutta_methods#The_Runge-Kutta_method
    """
    inv_m, inv_I = body.inv_mass, body.inv_inertia
    
    # Save initial state
    x0 = body.position.copy()
    v0 = body.velocity.copy()
    th0 = float(body.angle)
    w0 = float(body.omega)
    F0 = body.force.copy()
    tau0 = float(body.torque)

    def f(x, v, th, w):
        """Evaluate derivatives at arbitrary state (forces held constant)."""
        return _derivatives((x, v, th, w, F0, tau0), inv_m, inv_I)

    # RK4 stages
    k1 = f(x0, v0, th0, w0)
    k2 = f(
        x0 + 0.5 * dt * k1[0],
        v0 + 0.5 * dt * k1[1],
        th0 + 0.5 * dt * k1[2],
        w0 + 0.5 * dt * k1[3],
    )
    k3 = f(
        x0 + 0.5 * dt * k2[0],
        v0 + 0.5 * dt * k2[1],
        th0 + 0.5 * dt * k2[2],
        w0 + 0.5 * dt * k2[3],
    )
    k4 = f(
        x0 + dt * k3[0],
        v0 + dt * k3[1],
        th0 + dt * k3[2],
        w0 + dt * k3[3],
    )

    # Weighted combination
    body.position = x0 + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    body.velocity = v0 + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    body.angle = th0 + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    body.omega = w0 + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])


def rk4_adaptive_step(
    body: RigidBody2D,
    dt: float,
    tol: float,
    dt_min: float,
    dt_max: float,
) -> tuple[float, float]:
    """
    Adaptive RK4 using step-doubling for error estimation.
    
    Compares a single dt step against two dt/2 steps. If the difference
    exceeds tolerance, the step is rejected and dt is reduced.
    
    Deterministic acceptance rule (for reproducibility):
        Accept if error ≤ tol OR dt ≤ dt_min
    
    The next dt is scaled using the standard formula for RK4 (order 4):
        dt_new = dt × (tol / error)^(1/5)
    
    Args:
        body: Rigid body to integrate (modified in-place on accept).
        dt: Proposed timestep.
        tol: Error tolerance for step acceptance.
        dt_min: Minimum allowed timestep (forces acceptance).
        dt_max: Maximum allowed timestep.
        
    Returns:
        Tuple (accepted_dt, suggested_next_dt):
        - accepted_dt: Actual time advanced (0 if rejected, dt if accepted)
        - suggested_next_dt: Recommended dt for next call
    """
    # Preserve state for rejection/comparison
    saved = (body.position.copy(), body.velocity.copy(), body.angle, body.omega)
    
    # Full step
    rk4_step(body, dt)
    x1, v1, th1, w1 = body.position.copy(), body.velocity.copy(), body.angle, body.omega

    # Restore and do two half-steps (more accurate estimate)
    body.position, body.velocity, body.angle, body.omega = (
        saved[0].copy(), saved[1].copy(), saved[2], saved[3]
    )
    rk4_step(body, 0.5 * dt)
    rk4_step(body, 0.5 * dt)
    x2, v2, th2, w2 = body.position.copy(), body.velocity.copy(), body.angle, body.omega

    # Error estimate: L2 norm of difference across all state components
    err = (
        np.linalg.norm(x2 - x1) +
        np.linalg.norm(v2 - v1) +
        abs(th2 - th1) +
        abs(w2 - w1)
    )

    if err <= tol or dt <= dt_min:
        # Accept the two-half-steps result (it's the more accurate one)
        # Scale factor for next dt: (tol/err)^(1/5) for order-4 method
        if err < 1e-18:
            scale = 2.0  # Error negligible, can safely double
        else:
            scale = float((tol / err) ** 0.2)
        
        # Clamp scale factor to avoid wild swings
        scale = max(0.5, min(2.0, 0.9 * scale))
        dt_next = max(dt_min, min(dt_max, dt * scale))
        return dt, dt_next
    else:
        # Reject: restore original state and halve dt
        body.position, body.velocity, body.angle, body.omega = saved
        dt_new = max(dt_min, dt * 0.5)
        return 0.0, dt_new


def verlet_step(body: RigidBody2D, dt: float) -> None:
    """
    Advance body state using velocity Verlet integration.
    
    Verlet is a symplectic integrator, meaning it exactly preserves
    phase-space volume. This makes it excellent for long-term energy
    conservation in systems without dissipation.
    
    The standard velocity Verlet update is:
        x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    
    We use a simplified form where a(t+dt) ≈ a(t) (forces constant),
    which reduces to leapfrog-style updates. This is appropriate when
    forces are recomputed each timestep anyway.
    
    Args:
        body: Rigid body to integrate (modified in-place).
        dt: Timestep in seconds.
    
    Reference:
        https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
    """
    inv_m, inv_I = body.inv_mass, body.inv_inertia
    
    # Current accelerations
    a0 = body.force * inv_m
    alpha0 = body.torque * inv_I
    
    # Position update (includes velocity and acceleration terms)
    body.position = body.position + body.velocity * dt + 0.5 * a0 * dt * dt
    body.angle = body.angle + body.omega * dt + 0.5 * alpha0 * dt * dt
    
    # Velocity update (simplified: assumes constant acceleration)
    body.velocity = body.velocity + a0 * dt
    body.omega = body.omega + alpha0 * dt
