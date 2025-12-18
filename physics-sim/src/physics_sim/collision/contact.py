# MIT License (see LICENSE)
"""
Contact detection and resolution for collision handling.

This module provides contact data structures and solver functions for
resolving collisions between rigid bodies. Uses Sequential Impulse
(Projected Gauss-Seidel) with Baumgarte stabilization and Coulomb friction.

Key concepts:
- Contact: Represents a collision between two bodies with penetration info.
- Warm starting: Reuses accumulated impulses for faster solver convergence.
- Baumgarte stabilization: Corrects position drift over time.
- Coulomb friction: Tangent impulse clamped by mu * normal impulse.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..types import RigidBody2D
from ..util import unit, cross2


@dataclass
class Contact:
    """
    Represents a contact point between two colliding bodies.
    
    Contains geometric information (point, normal, penetration) and
    solver state (accumulated impulses, effective masses, bias).
    
    Attributes:
        a: First rigid body involved in the contact.
        b: Second rigid body involved in the contact.
        point: World-space contact point in meters.
        normal: Unit normal vector from body a toward body b.
        penetration: Overlap depth in meters (positive = overlapping).
        jn_accum: Accumulated normal impulse for warm starting.
        jt_accum: Accumulated tangent (friction) impulse for warm starting.
        mass_normal: Precomputed effective mass for normal impulse.
        mass_tangent: Precomputed effective mass for tangent impulse.
        bias: Velocity bias from Baumgarte stabilization + restitution.
    """
    a: RigidBody2D
    b: RigidBody2D
    point: np.ndarray
    normal: np.ndarray
    penetration: float
    jn_accum: float = 0.0
    jt_accum: float = 0.0
    
    # Solver cache (computed by prepare_contacts)
    mass_normal: float = 0.0
    mass_tangent: float = 0.0
    bias: float = 0.0
    
    # Feature ID for contact matching (default 0 for single-point collision)
    feature_id: int = 0


def warm_start_contact(contact: Contact) -> None:
    """
    Apply accumulated impulses for warm starting.
    
    Applies the stored normal and tangent impulses from the previous frame
    as an initial guess for the current frame's velocity. This greatly
    improves stability of stacking and resting contacts.
    
    Args:
        contact: The contact with accumulated impulses (jn_accum, jt_accum).
    """
    if contact.jn_accum == 0.0 and contact.jt_accum == 0.0:
        return

    a, b = contact.a, contact.b
    n = contact.normal
    
    # Tangent vector (recomputed from current state? OR stored?)
    # Tangent direction depends on relative velocity usually, but for
    # warm starting static friction, we use the direction of the stored impulse?
    # Or strict tangent: cross(n, z)? or implied?
    # In solve_contact_pgs, tangent `t` is computed from relative velocity.
    # If starting relative velocity is zero, tangent is arbitrary.
    # To properly warm start friction, we should store the tangent vector OR 
    # assume the tangent hasn't changed much (if n hasn't changed much).
    # But usually we just Apply Impulse Vector P.
    # If we only stored scalars `jn` and `jt`, we need `n` and `t`.
    # `n` is from the current contact geometry.
    # `t` needs to be derived.
    # Strategy: Recompute t perpendicular to n (rotated 90 deg).
    # NOTE: Friction warm starting is unstable if tangent vector definition is not consistent.
    # Without storing the tangent basis from the previous frame, applying jt_accum
    # with an arbitrary tangent can add energy.
    # For stability, we ONLY warm start the normal impulse (non-penetration).
    
    # t = np.array([-n[1], n[0]], dtype=np.float64)
    # P = contact.jn_accum * n + contact.jt_accum * t
    
    # Normal impulse only
    # Normal impulse only
    P = contact.jn_accum * n
    
    ra = contact.point - a.position
    rb = contact.point - b.position
    
    a.velocity -= P * a.inv_mass
    b.velocity += P * b.inv_mass
    # Stability: GJK/EPA provides a single, moving contact point.
    # Warm starting torque with a shifting contact point can introduce massive energy (torque spikes).
    # Disabling angular warm start improves stability for box stacks using single-point collision.
    # a.omega -= a.inv_inertia * cross2(ra, P)
    # b.omega += b.inv_inertia * cross2(rb, P)


def circle_circle_contact(a: RigidBody2D, b: RigidBody2D) -> Contact | None:
    """
    Detect contact between two circular bodies.
    
    Uses direct geometric computation (no GJK overhead for circles).
    
    Args:
        a: First body (must have Circle shape).
        b: Second body (must have Circle shape).
        
    Returns:
        Contact object if circles overlap, None otherwise.
    """
    pa, pb = a.position, b.position
    ra, rb = a.shape.radius, b.shape.radius
    d = pb - pa
    dist = float(np.linalg.norm(d))
    R = ra + rb
    
    if dist >= R:
        return None
    
    n = unit(d) if dist > 1e-12 else np.array([1.0, 0.0], dtype=np.float64)
    penetration = R - dist
    p = pa + n * (ra - 0.5 * penetration)
    
    return Contact(a=a, b=b, point=p, normal=n, penetration=penetration)


def prepare_contacts(
    contacts: list[Contact],
    dt: float,
    baumgarte_beta: float = 0.2,
) -> None:
    """
    Precompute solver data for contacts before iteration.
    
    Calculates effective masses and velocity bias (Baumgarte stabilization
    + restitution) for each contact. Must be called once per timestep
    before solve_contact_pgs iterations.
    
    Args:
        contacts: List of contacts to prepare.
        dt: Timestep in seconds.
        baumgarte_beta: Position correction factor (0-1). Higher = stiffer.
        
    Note:
        Modifies contact.mass_normal and contact.bias in-place.
    """
    for c in contacts:
        a, b = c.a, c.b
        n = c.normal
        ra = c.point - a.position
        rb = c.point - b.position
        
        # Effective mass for normal direction
        ra_x_n = ra[0] * n[1] - ra[1] * n[0]
        rb_x_n = rb[0] * n[1] - rb[1] * n[0]
        k_n = (
            a.inv_mass + b.inv_mass +
            (ra_x_n * ra_x_n) * a.inv_inertia +
            (rb_x_n * rb_x_n) * b.inv_inertia
        )
        c.mass_normal = 1.0 / k_n if k_n > 1e-15 else 0.0
        
        # Relative velocity at contact point (pre-solve)
        va = a.velocity + np.array([-a.omega * ra[1], a.omega * ra[0]], dtype=np.float64)
        vb = b.velocity + np.array([-b.omega * rb[1], b.omega * rb[0]], dtype=np.float64)
        rv = vb - va
        vn = float(np.dot(rv, n))
        
        # Restitution bias (only for separating velocity above threshold)
        e = min(a.material.restitution, b.material.restitution)
        restitution_v = -e * vn if vn < -1.0 else 0.0

        # Baumgarte position correction bias
        slop = 1e-4  # allowed penetration
        pos_bias = (baumgarte_beta / dt) * max(c.penetration - slop, 0.0)
        pos_bias = min(pos_bias, 20.0)  # stability clamp
        
        c.bias = pos_bias + restitution_v


def solve_contact_pgs(contact: Contact) -> None:
    """
    Apply impulses to resolve a single contact constraint.
    
    Performs one iteration of Sequential Impulse solving:
    1. Apply normal impulse to prevent penetration.
    2. Apply tangent impulse for Coulomb friction.
    
    Uses accumulated impulse clamping for stability.
    
    Args:
        contact: The contact to solve (must have been prepared first).
        
    Note:
        Modifies body velocities and angular velocities in-place.
        Updates contact.jn_accum and contact.jt_accum.
    """
    if contact.mass_normal == 0.0:
        return

    a, b = contact.a, contact.b
    n = contact.normal
    ra = contact.point - a.position
    rb = contact.point - b.position
    
    # --- Normal Impulse ---
    va = a.velocity + np.array([-a.omega * ra[1], a.omega * ra[0]], dtype=np.float64)
    vb = b.velocity + np.array([-b.omega * rb[1], b.omega * rb[0]], dtype=np.float64)
    rv = vb - va
    vn = float(np.dot(rv, n))
    
    lambda_n = contact.mass_normal * (contact.bias - vn)
    
    # Clamp accumulated impulse (normal impulse must be non-negative)
    jn_old = contact.jn_accum
    contact.jn_accum = max(jn_old + lambda_n, 0.0)
    jn = contact.jn_accum - jn_old
    
    # Apply normal impulse
    P = jn * n
    a.velocity -= P * a.inv_mass
    b.velocity += P * b.inv_mass
    a.omega -= a.inv_inertia * cross2(ra, P)
    b.omega += b.inv_inertia * cross2(rb, P)
    
    # --- Tangent Impulse (Friction) ---
    # Re-evaluate relative velocity after normal impulse
    va = a.velocity + np.array([-a.omega * ra[1], a.omega * ra[0]], dtype=np.float64)
    vb = b.velocity + np.array([-b.omega * rb[1], b.omega * rb[0]], dtype=np.float64)
    rv = vb - va
    
    # Compute tangent direction (perpendicular to normal, in sliding direction)
    vt_vec = rv - float(np.dot(rv, n)) * n
    vt_len = float(np.linalg.norm(vt_vec))
    if vt_len < 1e-12:
        return
    t = vt_vec / vt_len
    
    # Tangent effective mass
    ra_x_t = ra[0] * t[1] - ra[1] * t[0]
    rb_x_t = rb[0] * t[1] - rb[1] * t[0]
    k_t = (
        a.inv_mass + b.inv_mass +
        (ra_x_t * ra_x_t) * a.inv_inertia +
        (rb_x_t * rb_x_t) * b.inv_inertia
    )
    mass_tangent = 1.0 / k_t if k_t > 1e-15 else 0.0
    
    # Tangent impulse to oppose sliding
    lambda_t = mass_tangent * (-vt_len)
    
    # Coulomb friction clamp: |Jt| <= mu * Jn
    mu = 0.5 * (a.material.friction + b.material.friction)
    max_jt = mu * contact.jn_accum
    
    jt_old = contact.jt_accum
    contact.jt_accum = max(-max_jt, min(max_jt, jt_old + lambda_t))
    jt = contact.jt_accum - jt_old
    
    # Apply tangent impulse
    Pt = jt * t
    a.velocity -= Pt * a.inv_mass
    b.velocity += Pt * b.inv_mass
    a.omega -= a.inv_inertia * cross2(ra, Pt)
    b.omega += b.inv_inertia * cross2(rb, Pt)
