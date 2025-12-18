# MIT License (see LICENSE)
"""
Constraint solvers for physics simulation.

This subpackage provides constraint types and solvers:
    - DistanceConstraint: Maintains fixed distance between two bodies.
    - solve_distance_constraints_pgs: PGS solver with Baumgarte stabilization.

Typical usage:
    from physics_sim.constraints import DistanceConstraint, solve_distance_constraints_pgs
    
    constraint = DistanceConstraint(a=body1, b=body2, length=2.0)
    solve_distance_constraints_pgs([constraint], dt=1/240)
"""
from .solver import DistanceConstraint, solve_distance_constraints_pgs

__all__ = [
    "DistanceConstraint",
    "solve_distance_constraints_pgs",
]
