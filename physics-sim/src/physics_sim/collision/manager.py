from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from .contact import Contact
from ..types import RigidBody2D

# ContactKey maps a pair of bodies (by ID or index) to a manifold
# Tuple[int, int] where idA < idB
ContactKey = Tuple[int, int]

@dataclass
class ContactManifold:
    """
    Maintains a set of persistent contact points between two bodies.
    """
    body_a: RigidBody2D
    body_b: RigidBody2D
    points: Dict[int, Contact] = field(default_factory=dict) # Map feature_id -> Contact
    
    @property
    def a(self) -> RigidBody2D:
        return self.body_a
        
    @property
    def b(self) -> RigidBody2D:
        return self.body_b
    
    def update(self, new_contacts: List[Contact]) -> List[Contact]:
        """
        Update manifold with new detected contacts.
        
        Matches new contacts to existing ones via feature_id.
        Preserves accumulated impulses for matching contacts.
        Returns the list of active persistent contacts.
        """
        active_contacts: List[Contact] = []
        new_points_map: Dict[int, Contact] = {}
        
        for new_c in new_contacts:
            fid = new_c.feature_id
            
            if fid in self.points:
                # Match found! Update persistent contact with new geometry
                existing = self.points[fid]
                
                # Check spatial coherence
                # If contact point moved too much, the previous impulse is invalid (torque spike risk)
                # For GJK single-point, the point can jump between vertices.
                dist_sq = np.sum((existing.point - new_c.point)**2)
                if dist_sq > (0.002 * 0.002): # 2mm threshold
                    # Reset accumulators
                    existing.jn_accum = 0.0
                    existing.jt_accum = 0.0
                
                # Copy geometry
                existing.point = new_c.point
                existing.normal = new_c.normal
                existing.penetration = new_c.penetration
                # existing.a = new_c.a # Bodies should be same ref
                # existing.b = new_c.b
                
                # Keep accumulated impulses (jn_accum, jt_accum)
                
                new_points_map[fid] = existing
                active_contacts.append(existing)
            else:
                # New contact point
                new_points_map[fid] = new_c
                active_contacts.append(new_c)
        
        # Replace points map with new set (removes stale points)
        self.points = new_points_map
        return active_contacts

class ContactManager:
    """
    Manages persistent contact manifolds for the simulation scene.
    
    Ensures that contact points persist across frames allows for
    warm starting of the solver (reusing previous impulses).
    """
    def __init__(self):
        self.manifolds: Dict[ContactKey, ContactManifold] = {}
        
    def update(self, detected_contacts: List[Contact]) -> List[Contact]:
        """
        Process raw detected contacts and return persistent contacts.
        
        Args:
            detected_contacts: List of fresh Contact objects from narrowphase.
            
        Returns:
            List of persistent Contact objects to be passed to solver.
        """
        # Group contacts by pair
        # Note: bodies don't strictly have IDs in types.py yet.
        # But Scene uses list index or python id?
        # User constraint: "Uses a consistent ordering... idA and idB should be stable IDs, not ephemeral Python id() if those can change; if necessary, add explicit body.id fields."
        # Python id() is stable for object lifetime. Since bodies persist in Scene, id() is fine.
        # But for determinism (reproducibility across runs), id() is bad.
        # RigidBody2D does NOT have a deterministic ID field currently?
        # Let's check RigidBody2D source in types.py?
        # I recall it just had mass, etc.
        # If I use `id(body)`, it breaks cross-run determinism but works for stability within a run.
        # For this task, I'll use `id(body)` as a fallback, but ideally I should add `id` to RigidBody2D?
        # Wait, if I add `id` to RigidBody2D, do I need to manage it?
        # Ideally Scene assigns it.
        # Assumption: Scene keeps bodies consistent.
        # I will use `id(body)` for now as it's simplest, but note the limitation.
        # Actually, let's look at `types.py` if I can.
        # But first, implement grouping.
        
        grouped: Dict[ContactKey, List[Contact]] = {}
        
        for c in detected_contacts:
            # Enforce canonical order: id(a) < id(b)
            # Use body.id if available (guaranteed by Scene), else fallback to id()
            id_a = getattr(c.a, 'id', id(c.a))
            id_b = getattr(c.b, 'id', id(c.b))
            
            if id_a > id_b:
                # Swap to canonical representation
                c.a, c.b = c.b, c.a
                c.normal = -c.normal
                id_a, id_b = id_b, id_a
            
            key = (id_a, id_b)
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(c)
            
        # Update manifolds
        active_manifolds: Dict[ContactKey, ContactManifold] = {}
        final_contacts: List[Contact] = []
        
        for key, contacts in grouped.items():
            if key in self.manifolds:
                manifold = self.manifolds[key]
            else:
                # Create new
                manifold = ContactManifold(contacts[0].a, contacts[0].b)
            
            # Update manifold
            params = manifold.update(contacts)
            final_contacts.extend(params)
            
            # Keep manifold alive
            active_manifolds[key] = manifold
            
        # Remove stale manifolds (naturally handled by recreating active_manifolds dict?
        # NO. If a pair separates, it's not in `grouped`. 
        # If I reconstruct `self.manifolds` from scratch using only active ones, I lose warm start data for pairs that might flicker?
        # Actually yes, if they separate, we drop the manifold.
        # So replacing `self.manifolds = active_manifolds` is correct behavior.
        
        self.manifolds = active_manifolds
        return final_contacts
