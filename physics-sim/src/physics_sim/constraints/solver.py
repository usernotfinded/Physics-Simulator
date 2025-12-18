# MIT License (see LICENSE)
"""
Constraint solvers for physics simulation.

This module provides iterative constraint solvers using the Projected
Gauss-Seidel (PGS) method with Baumgarte stabilization. Constraints
maintain relationships between bodies (e.g., fixed distance for pendulums).

Key concepts:
- PGS (Projected Gauss-Seidel): Iterative solver that converges to valid
  constraint satisfaction over multiple iterations.
- Baumgarte stabilization: Adds positional correction to prevent drift.
- Warm starting: Reuses accumulated impulses from previous frames for
  faster convergence.

See maths.md Eq (11)-(12) for the underlying mathematics.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..types import RigidBody2D
from ..util import unit


@dataclass
class DistanceConstraint:
    """
    Maintains a fixed distance between two bodies (e.g., pendulum rod).
    
    This constraint enforces |pos_b - pos_a| = length at all times.
    Uses Baumgarte stabilization to correct position drift and warm
    starting for faster solver convergence.
    
    Attributes:
        a: First rigid body.
        b: Second rigid body.
        length: Target distance between body centers in meters.
        beta: Baumgarte stabilization factor (0-1). Higher = stiffer but
              potentially unstable. Default 0.2 is a good balance.
        lambda_accum: Accumulated constraint impulse for warm starting.
    """
    a: RigidBody2D
    b: RigidBody2D
    length: float
    beta: float = 0.2
    lambda_accum: float = 0.0


def solve_distance_constraints_pgs(
    constraints: list[DistanceConstraint],
    dt: float,
    iters: int = 20,
) -> None:
    """
    Solve distance constraints using Projected Gauss-Seidel iteration.
    
    Iteratively applies impulses to satisfy all distance constraints.
    Uses Baumgarte stabilization to correct position drift over time
    (maths.md Eq 11-12).
    
    Args:
        constraints: List of distance constraints to solve.
        dt: Timestep in seconds. Used for Baumgarte bias calculation.
        iters: Number of solver iterations. More iterations = better
               accuracy but higher cost. Default 20 is usually sufficient.
               
    Note:
        Modifies body velocities in-place. The accumulated impulse
        (constraint.lambda_accum) is updated for warm starting in
        subsequent frames.
    """
    for _ in range(iters):
        for c in constraints:
            a, b = c.a, c.b
            d = b.position - a.position
            dist = float(np.linalg.norm(d))
            
            if dist < 1e-12:
                continue
            
            n = d / dist

            # Constraint error: C = current_distance - target_length
            C = dist - c.length
            bias = (c.beta / dt) * C

            # Relative velocity along constraint axis
            vn = float(np.dot((b.velocity - a.velocity), n))

            # Effective mass (inverse mass sum)
            k = a.inv_mass + b.inv_mass
            if k < 1e-15:
                continue
            m_eff = 1.0 / k

            # Solve for impulse magnitude
            dlambda = -(vn + bias) * m_eff

            # Accumulate (unbounded for distance constraints)
            c.lambda_accum += dlambda
            impulse = dlambda * n

            # Apply impulse to bodies
            a.velocity -= impulse * a.inv_mass
            b.velocity += impulse * b.inv_mass


@dataclass
class RevoluteConstraint:
    """
    Constraints two bodies to share a common point (hinge/pivot).
    
    Constrains 2 degrees of freedom (translation x, y). Rotation is free.
    
    Attributes:
        a: First rigid body.
        b: Second rigid body.
        anchor_a: Local anchor point on body A.
        anchor_b: Local anchor point on body B.
        beta: Baumgarte factor.
        lambda_accum: Accumulated impulse (vector 2D).
    """
    a: RigidBody2D
    b: RigidBody2D
    anchor_a: tuple[float, float]
    anchor_b: tuple[float, float]
    beta: float = 0.2
    
    # Internal state
    lambda_accum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    
    def __post_init__(self):
        # Allow tuple inputs but store as array
        # Note: field default factory handles initialization, but we need to ensure type
        if isinstance(self.lambda_accum, list):
            self.lambda_accum = np.array(self.lambda_accum, dtype=np.float64)


def solve_revolute_constraints_pgs(
    constraints: list[RevoluteConstraint],
    dt: float,
    iters: int = 20,
) -> None:
    """
    Solve revolute (hinge) constraints.
    
    Enforces that: pos_a + rot_a(anchor_a) == pos_b + rot_b(anchor_b).
    """
    from ..util import cross_z_scalar_vec, vec_cross_z, cross2

    for _ in range(iters):
        for c in constraints:
            a, b = c.a, c.b
            
            # Calculate global anchor positions
            # Rot A
            ca, sa = np.cos(a.angle), np.sin(a.angle)
            ra = np.array([
                c.anchor_a[0]*ca - c.anchor_a[1]*sa,
                c.anchor_a[0]*sa + c.anchor_a[1]*ca
            ], dtype=np.float64)
            
            # Rot B
            cb, sb = np.cos(b.angle), np.sin(b.angle)
            rb = np.array([
                c.anchor_b[0]*cb - c.anchor_b[1]*sb,
                c.anchor_b[0]*sb + c.anchor_b[1]*cb
            ], dtype=np.float64)
            
            # Current constraint error (position difference)
            # C = (pa + ra) - (pb + rb)
            diff = (a.position + ra) - (b.position + rb)
            
            # Bias (Baumgarte)
            bias = (c.beta / dt) * diff
            
            # Jacobian J = [ -I, -skew(ra), I, skew(rb) ]
            # Effective mass K = J M^-1 J^T
            # K = inv_ma*I + inv_Ia*skew(ra)*skew(ra)^T + inv_mb*I + inv_Ib*skew(rb)*skew(rb)^T
            # For 2D point constraint, K is a 2x2 matrix.
            
            # Calculate K matrix terms
            inv_m_sum = a.inv_mass + b.inv_mass
            
            # Rotational terms: skew(r) = [[0, -ry], [rx, 0]]
            # r_skew * r_skew^T = [[ry^2, -rx*ry], [-rx*ry, rx^2]]
            
            rax, ray = ra[0], ra[1]
            rbx, rby = rb[0], rb[1]
            
            K = np.eye(2) * inv_m_sum
            
            if a.inv_inertia > 0:
                K[0,0] += a.inv_inertia * ray * ray
                K[0,1] += a.inv_inertia * -rax * ray
                K[1,0] += a.inv_inertia * -rax * ray
                K[1,1] += a.inv_inertia * rax * rax

            if b.inv_inertia > 0:
                K[0,0] += b.inv_inertia * rby * rby
                K[0,1] += b.inv_inertia * -rbx * rby
                K[1,0] += b.inv_inertia * -rbx * rby
                K[1,1] += b.inv_inertia * rbx * rbx
                
            # Invert K
            try:
                # Manual 2x2 inverse for speed/safety
                det = K[0,0]*K[1,1] - K[0,1]*K[1,0]
                if det < 1e-15:
                    continue
                inv_det = 1.0 / det
                K_inv = np.array([
                    [K[1,1], -K[0,1]],
                    [-K[1,0], K[0,0]]
                ]) * inv_det
            except:
                continue
                
            # Relative velocity at anchors
            va = a.velocity + cross_z_scalar_vec(a.omega, ra)
            vb = b.velocity + cross_z_scalar_vec(b.omega, rb)
            dv = va - vb # J*v = va - vb (sign convention depends on C def)
            # C = pos_a - pos_b ... implies we want va - vb -> 0
            
            # Impulse lambda = K_inv * -(Jv + bias)
            rhs = -(dv + bias)
            dlambda = K_inv @ rhs
            
            c.lambda_accum += dlambda
            P = dlambda
            
            # Apply impulses
            # For body A: force +P, torque +ra x P
            # For body B: force -P, torque -rb x P
            # Wait, C = (pa + ra) - (pb + rb).
            # J w.r.t A matches signs of C w.r.t A ?? 
            # dC/dt = va - vb.
            # So velocity constraint is va - vb = 0.
            # if we apply +P to A, vel change is +P/m.
            # New rel vel (va-vb) changes by (+P/ma - -P/mb) = P(1/ma+1/mb).
            # So signs are correct: +P on A, -P on B.
            
            # Force A
            a.velocity += P * a.inv_mass
            a.omega += a.inv_inertia * cross2(ra, P)
            
            # Force B
            b.velocity -= P * b.inv_mass
            b.omega -= b.inv_inertia * cross2(rb, P)


@dataclass
class PrismaticConstraint:
    """
    Constrains body B to slide along an axis fixed in body A.
    
    Locks relative rotation and perpendicular translation.
    
    Attributes:
        a: First rigid body.
        b: Second rigid body.
        anchor_a: Local anchor on A.
        axis_a: Local axis unit vector on A [x, y].
        beta: Baumgarte factor.
    """
    a: RigidBody2D
    b: RigidBody2D
    anchor_a: tuple[float, float]
    axis_a: tuple[float, float]
    beta: float = 0.2
    
    # Internal state
    # 2 DOF constrained: 1 angle, 1 linear perp
    lambda_accum_ang: float = 0.0
    lambda_accum_lin: float = 0.0
    
    # Reference angle difference (locked at initialization usually)
    ref_angle: float = 0.0


def solve_prismatic_constraints_pgs(
    constraints: list[PrismaticConstraint],
    dt: float,
    iters: int = 20,
) -> None:
    """
    Solve prismatic (slider) constraints.
    """
    from ..util import cross_z_scalar_vec, vec_cross_z, cross2

    for _ in range(iters):
        for c in constraints:
            a, b = c.a, c.b
            
            # 1. Angular Constraint (Lock relative rotation)
            # C_ang = angle_b - angle_a - ref
            C_ang = b.angle - a.angle - c.ref_angle
            bias_ang = (c.beta / dt) * C_ang
            
            # Relative angular vel
            w_rel = b.omega - a.omega
            
            # Effective mass for rotation
            k_ang = a.inv_inertia + b.inv_inertia
            if k_ang > 1e-15:
                dlambda_ang = -(w_rel + bias_ang) / k_ang
                c.lambda_accum_ang += dlambda_ang
                
                # Apply angular impulse
                # Torque on B: +dlambda
                # Torque on A: -dlambda
                a.omega -= dlambda_ang * a.inv_inertia
                b.omega += dlambda_ang * b.inv_inertia

            # 2. Linear Constraint (Perpendicular to axis)
            # World axis
            ca, sa = np.cos(a.angle), np.sin(a.angle)
            axis_w = np.array([
                c.axis_a[0]*ca - c.axis_a[1]*sa,
                c.axis_a[0]*sa + c.axis_a[1]*ca
            ], dtype=np.float64)
            perp_w = np.array([-axis_w[1], axis_w[0]]) # Rotate 90 deg
            
            # World anchor A
            ra = np.array([
                c.anchor_a[0]*ca - c.anchor_a[1]*sa,
                c.anchor_a[0]*sa + c.anchor_a[1]*ca
            ], dtype=np.float64)
            
            # We constrain position of B's COM (or anchor B?) relative to A's anchor projected on perp axis?
            # Standard prismatic: Anchor B is implicitly B.position (or specific anchor on B).
            # Let's assume Anchor B is COM of B for simplicity, or we should add anchor_b to class.
            # Assuming B's COM is on the line defined by (A.pos + ra, axis_w).
            
            # Vector from A's anchor to B's COM
            d = b.position - (a.position + ra)
            
            # Constraint: dot(d, perp_w) = 0
            C_lin = float(np.dot(d, perp_w))
            bias_lin = (c.beta / dt) * C_lin
            
            # Jv term ... 
            # v_anchor_a = va + cross(wa, ra)
            # v_point_b = vb (since point is COM)
            # dC/dt = dot(vb - v_anchor_a, perp_w) + dot(d, d(perp_w)/dt)
            # Term 2 is usually small/ignored or complex. Let's start with simple velocity projection.
            
            va_anchor = a.velocity + cross_z_scalar_vec(a.omega, ra)
            v_rel = b.velocity - va_anchor
            jv = float(np.dot(v_rel, perp_w))
            
            # Add term for axis rotation? 
            # d(perp)/dt = wa x perp
            # dC/dt includes: dot(d, wa x perp)
            # wa * cross(perp, d) = wa * dot(perp, cross(d, z))?
            # It's wa * dot(d, perp_rotated_90) ? 
            # Let's approximate by just projecting relative velocity on current perp axis.
            # Usually sufficient for Baumgarte stabilization.
            
            # Effective mass
            # J_lin maps v -> scalar.
            # J_lin = [ -perp, ... ]?
            # Linear part: inv_ma + inv_mb (projected on perp)
            # Angular part: A rotates -> affects anchor A -> affects constraint
            #               A rotates -> affects perp axis -> affects constraint (ignored for now)
            #               B rotates -> does not affect COM pos constraint (if anchor B is COM)
            
            k_lin = a.inv_mass + b.inv_mass
            
            # Tangential term for A's rotation effect on anchor velocity
            # Torque = cross(ra, Impulse_direction) = cross(ra, perp)
            rt = cross2(ra, perp_w)
            k_lin += a.inv_inertia * rt * rt
            
            if k_lin > 1e-15:
                dlambda_lin = -(jv + bias_lin) / k_lin
                c.lambda_accum_lin += dlambda_lin
                
                P = dlambda_lin * perp_w
                # Apply P to B, -P to A
                # Force
                b.velocity += P * b.inv_mass
                a.velocity -= P * a.inv_mass
                
                # Torque on A (from force at anchor)
                # Torque = ra x (-P) = - cross(ra, P)
                a.omega -= a.inv_inertia * cross2(ra, P)


@dataclass
class MouseJoint:
    """
    Soft constraint for dragging bodies with a cursor/mouse.
    
    Acts as a spring-damper between a fixed target point (cursor) and 
    a local anchor point on the body.
    
    Attributes:
        body: The dynamic body being dragged.
        target: World space target point (cursor position).
        local_anchor: Attachment point in body's local coordinates.
        max_force: Maximum force magnitude (prevents instability/explosions).
        frequency_hz: Response speed (approximate spring frequency).
        damping_ratio: Damping factor (0.0 - 1.0). 1.0 = Critical Damping.
    """
    body: RigidBody2D
    target: np.ndarray # World space [x, y]
    local_anchor: tuple[float, float]
    max_force: float = 1000.0
    frequency_hz: float = 5.0
    damping_ratio: float = 0.7
    
    # Internal solver state
    gamma: float = 0.0
    beta: float = 0.0
    k_inv: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float64))
    lambda_accum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64)) # Impulse
    
    def set_target(self, pos: tuple[float, float] | np.ndarray) -> None:
        """Update the target world position (e.g. on mouse move)."""
        self.target = np.array(pos, dtype=np.float64)
        # Wake the body to ensure it reacts
        self.body.wake()

def solve_mouse_constraints_pgs(
    constraints: list[MouseJoint],
    dt: float,
    iters: int = 20,
) -> None:
    """
    Solve mouse joints using PGS with soft constraint formulation.
    """
    from ..util import cross_z_scalar_vec, cross2
    
    # 1. Initialization / Pre-step
    for c in constraints:
        b = c.body
        
        # Calculate soft constraint parameters
        omega = 2.0 * np.pi * c.frequency_hz
        d = 2.0 * b.mass * c.damping_ratio * omega
        k = b.mass * omega * omega
        
        c.gamma = 1.0 / (dt * (d + dt * k)) if (d + dt * k) > 1e-9 else 0.0
        c.beta = (dt * k) / (d + dt * k) if (d + dt * k) > 1e-9 else 0.0
        
        # Calculate world anchor and effective mass
        ca, sa = np.cos(b.angle), np.sin(b.angle)
        ra = np.array([
            c.local_anchor[0]*ca - c.local_anchor[1]*sa,
            c.local_anchor[0]*sa + c.local_anchor[1]*ca
        ], dtype=np.float64)
        
        rax, ray = ra[0], ra[1]
        
        K = np.eye(2) * b.inv_mass
        
        if b.inv_inertia > 0:
            K[0,0] += b.inv_inertia * ray * ray
            K[0,1] += b.inv_inertia * -rax * ray
            K[1,0] += b.inv_inertia * -rax * ray
            K[1,1] += b.inv_inertia * rax * rax
            
        # Add soft constraint gamma
        K[0,0] += c.gamma
        K[1,1] += c.gamma
        
        # Invert K
        det = K[0,0]*K[1,1] - K[0,1]*K[1,0]
        if det < 1e-15:
            # Degenerate? Should rarely happen unless mass is infinite
            c.k_inv = np.zeros((2, 2), dtype=np.float64)
            continue
            
        inv_det = 1.0 / det
        c.k_inv = np.array([
            [K[1,1], -K[0,1]],
            [-K[1,0], K[0,0]]
        ]) * inv_det
        
        # Apply warm starting
        P = c.lambda_accum
        b.velocity += P * b.inv_mass
        b.omega += b.inv_inertia * cross2(ra, P)

    # 2. Iterative Solver
    for _ in range(iters):
        for c in constraints:
            b = c.body
            
            # Recalc world anchor vector
            ca, sa = np.cos(b.angle), np.sin(b.angle)
            ra = np.array([
                c.local_anchor[0]*ca - c.local_anchor[1]*sa,
                c.local_anchor[0]*sa + c.local_anchor[1]*ca
            ], dtype=np.float64)
            
            # C = world_anchor - target
            C = (b.position + ra) - c.target
            
            # Position Bias (Baumgarte / Soft)
            bias = C * (c.beta / dt)
            
            # Relative velocity
            v_anchor = b.velocity + cross_z_scalar_vec(b.omega, ra)
            
            # Constraint equation: J*v + bias + gamma * impulse = 0
            # Impulse = -K_inv * (v_anchor + bias + gamma * accumulated_impulse)
            rhs = -(v_anchor + bias + c.gamma * c.lambda_accum)
            dlambda = c.k_inv @ rhs
            
            old_impulse = c.lambda_accum.copy()
            c.lambda_accum += dlambda
            
            # Max force clamping
            max_impulse = c.max_force * dt
            current_mag = np.linalg.norm(c.lambda_accum)
            if current_mag > max_impulse:
                c.lambda_accum *= (max_impulse / current_mag)
            
            # Delta for this iteration
            dlambda = c.lambda_accum - old_impulse
            
            # Apply to body
            # P = dlambda
            b.velocity += dlambda * b.inv_mass
            b.omega += b.inv_inertia * cross2(ra, dlambda)

