# MIT License (see LICENSE)
"""
Collision detection and resolution subsystem.

This subpackage provides:
    - Broadphase: Spatial hashing for efficient pair culling.
    - Narrowphase: GJK/EPA for convex intersection and penetration.
    - Contact: Contact data and Sequential Impulse solver.
    - CCD: Continuous collision detection for tunneling prevention.

Typical usage:
    from physics_sim.collision import detect_contact, SpatialHashBroadphase
    
    broadphase = SpatialHashBroadphase(cell_size=1.0)
    for a, b in broadphase.pairs(bodies):
        contact = detect_contact(a, b)
        if contact:
            # handle collision
"""
from .broadphase import SpatialHashBroadphase, aabb_for_body
from .contact import Contact, circle_circle_contact, prepare_contacts, solve_contact_pgs
from .convex import detect_contact, convex_contact
from .gjk import gjk_intersect, support
from .epa import epa_penetration
from .ccd import circle_toi

__all__ = [
    # Broadphase
    "SpatialHashBroadphase",
    "aabb_for_body",
    # Contact
    "Contact",
    "circle_circle_contact",
    "prepare_contacts",
    "solve_contact_pgs",
    # Convex (GJK/EPA)
    "detect_contact",
    "convex_contact",
    "gjk_intersect",
    "support",
    "epa_penetration",
    # CCD
    "circle_toi",
]
