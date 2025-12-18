# MIT License (see LICENSE)
"""
Island management and rigid body sleeping logic.

This module implements the "Island" concept for physics simulation.
An island is a connected component of bodies linked by constraints (joints)
or contacts. Sleeping is decided at the island level to ensure stability.

Logic:
1. Build a graph of bodies (nodes) and interactions (edges).
2. Decompose graph into disjoint islands.
3. For each island:
   - Check if all bodies are below motion thresholds.
   - If yes and stable for enough time, put entire island to sleep.
   - If any body is awake/moving, wake the entire island.
"""
from __future__ import annotations
from typing import List, Set, Any, Protocol
import numpy as np

from ..types import RigidBody2D
from ..collision.contact import Contact


# Sleep Thresholds
SLEEP_LINEAR_THRESHOLD = 0.05       # m/s
SLEEP_ANGULAR_THRESHOLD = 0.05      # rad/s (approx 3 deg/s)
SLEEP_TIME_THRESHOLD = 0.5          # seconds


class ConstraintProtocol(Protocol):
    """Protocol for any constraint linking two bodies."""
    a: RigidBody2D
    b: RigidBody2D


class IslandManager:
    """
    Manages islands of bodies to handle sleeping and wake-up propagation.
    """
    def __init__(self) -> None:
        pass

    def update(
        self,
        bodies: List[RigidBody2D],
        contacts: List[Contact],
        constraints: List[ConstraintProtocol],
        dt: float
    ) -> None:
        """
        Update sleep state for all bodies based on island analysis.
        
        Args:
            bodies: All bodies in the scene.
            contacts: Active contacts for this frame.
            constraints: Active constraints/joints.
            dt: Timestep in seconds.
        """
        # 0. Reset visited state effectively by tracking processed bodies
        # We will build adjacency list dynamically
        adj: dict[int, List[RigidBody2D]] = {b.id: [] for b in bodies}
        
        # 1. Build Interaction Graph
        # Edges from Contacts
        for c in contacts:
            # Only enabled contacts connect bodies (assumed all active for now)
            # Static bodies act as isolators usually? 
            # Standard Box2D: Static bodies are nodes but don't propagate "union" if they are the bridge?
            # Actually, if A rests on StaticFloor, A is an island.
            # If A rests on B, and B on StaticFloor, A-B is island.
            # Static bodies usually participate but don't merge islands across themselves?
            # For simplicity: Treat Static bodies as nodes. 
            # But sleeping only applies to Dynamic bodies.
            adj[c.a.id].append(c.b)
            adj[c.b.id].append(c.a)

        # Edges from Constraints (All types)
        # Constraints passed as a flat list or we need to aggregate them in Scene
        # We assume `constraints` contains all types or we accept list of lists?
        # Scene passes a single iterable of all constraints preferably.
        for j in constraints:
            adj[j.a.id].append(j.b)
            adj[j.b.id].append(j.a)

        visited: Set[int] = set()
        
        # 2. Traverse Graph to find Islands
        for body in bodies:
            if body.id in visited:
                continue
            
            # Static/Kinematic bodies don't initiate sleeping logic usually,
            # but they can put other bodies to sleep.
            # However, we only care about "Dynamic" islands.
            if body.mass <= 0:
                continue

            # Start BFS/DFS for a new island
            stack = [body]
            visited.add(body.id)
            island_bodies: List[RigidBody2D] = []
            
            while stack:
                curr = stack.pop()
                if curr.mass > 0: # Only collect dynamic bodies for sleep check
                    island_bodies.append(curr)
                
                # Neighbors
                for neighbor in adj.get(curr.id, []):
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        # If neighbor is static, we add it to visited (so we don't process it as seed),
                        # but we DO NOT traverse *through* it to other dynamic bodies?
                        # Box2D rule: Static bodies bind islands but don't merge them?
                        # Actually Box2D merges everything connected.
                        # If two piles on same static floor are far apart, they should sleep independently.
                        # If we traverse through StaticFloor, we merge EVERYTHING.
                        # Fix: Do not push neighbor to stack if it is Static.
                        # This makes Static bodies "leaves" or "roots" but not "bridges".
                        if neighbor.mass > 0:
                            stack.append(neighbor)
            
            # 3. Process Island Sleep State
            self._process_island(island_bodies, dt)

    def _process_island(self, island: List[RigidBody2D], dt: float) -> None:
        """
        Check sleep criteria for a group of dynamic bodies.
        """
        if not island:
            return

        # Mixed State Check:
        # If any body is awake (e.g. manually woken or moving), the entire island must match state.
        # This propagates 'Wake' signals immediately.
        has_awake = False
        has_sleeping = False
        for b in island:
            if b.sleeping:
                has_sleeping = True
            else:
                has_awake = True
        
        if has_awake and has_sleeping:
            # Mixed state -> Wake everyone to maintain consistency
            for b in island:
                b.wake()
            return

        min_sleep_time = float('inf')
        all_sleepy = True
        
        # Assess motion for all bodies
        for b in island:
            # Check if body is allowed to seek sleep
            if not b.can_sleep:
                all_sleepy = False
                break
                
            # Compute motion magnitude
            # Note: If body is alrady sleeping, v=0, so it remains sleepy.
            v_mag = np.linalg.norm(b.velocity)
            w_mag = abs(b.omega)
            
            if v_mag < SLEEP_LINEAR_THRESHOLD and w_mag < SLEEP_ANGULAR_THRESHOLD:
                pass 
            else:
                # Body is awake/moving
                all_sleepy = False
                # Reset this body's timer immediately
                b.sleep_time = 0.0
                break

        if all_sleepy:
            # Pass 1: Update individual timers
            island_ready = True
            for b in island:
                if not b.can_sleep:
                    island_ready = False
                    b.sleep_time = 0.0
                    continue

                v = np.linalg.norm(b.velocity)
                w = abs(b.omega)
                
                if v < SLEEP_LINEAR_THRESHOLD and w < SLEEP_ANGULAR_THRESHOLD:
                    b.sleep_time += dt
                else:
                    b.sleep_time = 0.0
                    island_ready = False
            
            # Pass 2: If island is ready, check if ALL timers exceed threshold
            if island_ready:
                can_doze = True
                for b in island:
                    if b.sleep_time < SLEEP_TIME_THRESHOLD:
                        can_doze = False
                        break
                
                if can_doze:
                    # PUT TO SLEEP
                    for b in island:
                        b.sleep()
                # Else: They are sleepy candidates but waiting for timer. Keep Awake.
            else:
                # Island active. Wake everyone.
                for b in island:
                    b.wake()
        else:
            # Not all sleepy. Wake everyone.
            for b in island:
                b.wake()
