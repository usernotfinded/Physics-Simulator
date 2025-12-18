# MIT License (see LICENSE)
"""
Core physics simulation components.

This subpackage provides:
    - Force generators: Gravity, drag, Lorentz, Coulomb forces.
    - Integrators: RK4, adaptive RK4, Verlet integration.

Typical usage:
    from physics_sim.core import apply_gravity, rk4_step
    
    apply_gravity(body, np.array([0, -9.81]))
    rk4_step(body, dt=1/240)
"""
from .forces import (
    apply_gravity,
    apply_linear_drag,
    apply_lorentz,
    apply_coulomb_pairwise,
)
from .integrators import rk4_step, rk4_adaptive_step, verlet_step

__all__ = [
    # Forces
    "apply_gravity",
    "apply_linear_drag",
    "apply_lorentz",
    "apply_coulomb_pairwise",
    # Integrators
    "rk4_step",
    "rk4_adaptive_step",
    "verlet_step",
]
