# MIT License (see LICENSE)
"""
Material properties for physics simulation.

Materials define surface interaction properties used during contact
resolution, including friction and restitution (bounciness).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """
    Physical surface properties for collision response.
    
    Attributes:
        friction: Coefficient of friction μ (Coulomb friction model).
                  Range [0, 1+], where 0 = frictionless, 1 = high friction.
                  Values > 1 are physically unusual but allowed.
        restitution: Coefficient of restitution e (bounciness).
                     Range [0, 1], where 0 = perfectly inelastic (no bounce),
                     1 = perfectly elastic (full energy preserved).
    
    Note:
        When two bodies collide, effective friction is averaged and
        effective restitution uses the minimum of the two materials.
        See collision/contact.py for the combination rules.
    """
    friction: float = 0.5
    restitution: float = 0.2
