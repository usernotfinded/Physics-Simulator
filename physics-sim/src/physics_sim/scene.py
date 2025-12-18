# MIT License (see LICENSE)
"""
The main simulation scene and loop.

The Scene class acts as the world container and simulation controller.
It manages:
- The list of rigid bodies and constraints.
- Global simulation parameters (gravity, timestep, integrator choice).
- The main simulation loop (step), including:
    1. Force generation (gravity, EM, drag).
    2. Integration (RK4, Verlet).
    3. Collision detection (Broadphase + Narrowphase).
    4. Constraint solving (Contacts + Joints).
    5. Continuous Collision Detection (CCD) for fast-moving circles.

Structure:
    - User creates a Scene.
    - User adds bodies via add_body().
    - User calls scene.step(dt) in a loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .types import RigidBody2D, Circle, Box
from .util import f64
from .profiler import Profiler
from .core.forces import (
    apply_gravity, 
    apply_linear_drag, 
    apply_lorentz, 
    apply_coulomb_pairwise
)
from .core.integrators import rk4_step, rk4_adaptive_step, verlet_step
from .collision.broadphase import SpatialHashBroadphase
from .collision.ccd import circle_toi
from .collision.contact import (
    Contact,
    circle_circle_contact, 
    prepare_contacts, 
    solve_contact_pgs,
    warm_start_contact
)
from .collision.convex import detect_contact
from .collision.manager import ContactManager
from .core.island import IslandManager
from .constraints.solver import (
    DistanceConstraint, solve_distance_constraints_pgs,
    RevoluteConstraint, solve_revolute_constraints_pgs,
    PrismaticConstraint, solve_prismatic_constraints_pgs,
    MouseJoint, solve_mouse_constraints_pgs
)


@dataclass
class Scene:
    """
    Physics simulation world.
    
    Attributes:
        gravity: Global gravity vector (default: Earth gravity [0, -9.81]).
        dt: Base simulation timestep in seconds (default: 1/240).
        integrator: Integration scheme ("rk4", "rk4_adaptive", "verlet").
        substeps: Number of physics substeps per 'step' call. Higher = more stable
                  collisions but slower. Default: 4.
        solver_iters: Constraint solver iterations per substep. Default: 20.
        enable_em: Enable electromagnetic forces (Lorentz + Coulomb).
        drag_c: Global linear drag coefficient (simulates air resistance).
        E: Global electric field vector [Ex, Ey].
        Bz: Global uniform magnetic field z-component.
        coulomb_eps: Softening parameter for Coulomb interactions.
        profiler: Optional Profiler instance for timing statistics.
    """
    gravity: tuple[float, float] = (0.0, -9.81)
    dt: float = 1/240
    integrator: str = "rk4_adaptive" 
    rk_tol: float = 1e-9
    rk_dt_min: float = 1e-5
    rk_dt_max: float = 1/60
    substeps: int = 4
    solver_iters: int = 20
    enable_em: bool = True
    drag_c: float = 0.0
    E: tuple[float, float] = (0.0, 0.0)
    Bz: float = 0.0
    coulomb_eps: float = 1e-3
    profiler: Profiler | None = None

    # Internal state
    bodies: list[RigidBody2D] = field(default_factory=list)
    constraints: list[DistanceConstraint] = field(default_factory=list)
    revolute_constraints: list[RevoluteConstraint] = field(default_factory=list)
    prismatic_constraints: list[PrismaticConstraint] = field(default_factory=list)
    mouse_joints: list[MouseJoint] = field(default_factory=list)
    contact_manager: ContactManager = field(default_factory=ContactManager)
    island_manager: IslandManager = field(default_factory=IslandManager)
    time: float = 0.0

    def __post_init__(self) -> None:
        """Initialize internal structures after dataclass creation."""
        # Convert tuple configs to numpy arrays for speed
        self._g = f64(self.gravity)
        self._E = f64(self.E)
        
        # Broadphase for efficient collision pair finding
        # Cell size 1.0 is a reasonable default for bodies ~0.1-2.0m size
        self._broadphase = SpatialHashBroadphase(cell_size=1.0)
        
        self._next_id = 1

    def add_body(self, body: RigidBody2D) -> int:
        """
        Add a rigid body to the simulation.
        
        Assigns a unique ID to the body.
        
        Args:
            body: The body instance to add.
            
        Returns:
            The assigned body ID.
        """
        body.id = self._next_id
        self._next_id += 1
        self.bodies.append(body)
        return body.id

    def add_distance_constraint(self, a: RigidBody2D, b: RigidBody2D, length: float, beta: float = 0.2) -> None:
        """
        Add a distance constraint (pendulum rod) between two bodies.
        
        Args:
            a: First body.
            b: Second body.
            length: Target distance.
            beta: Baumgarte stabilization factor (0.0 - 1.0).
        """
        self.constraints.append(DistanceConstraint(a=a, b=b, length=length, beta=beta))

    def add_revolute_constraint(
        self, 
        a: RigidBody2D, 
        b: RigidBody2D, 
        anchor_a: tuple[float, float], 
        anchor_b: tuple[float, float],
        beta: float = 0.2
    ) -> None:
        """
        Add a revolute joint (hinge) connecting two bodies at a point.
        """
        self.revolute_constraints.append(RevoluteConstraint(
            a=a, b=b, anchor_a=anchor_a, anchor_b=anchor_b, beta=beta
        ))

    def add_prismatic_constraint(
        self, 
        a: RigidBody2D, 
        b: RigidBody2D, 
        anchor_a: tuple[float, float], 
        axis_a: tuple[float, float],
        beta: float = 0.2
    ) -> None:
        """
        Add a prismatic joint (slider) constraining B to line on A.
        """
        self.prismatic_constraints.append(PrismaticConstraint(
            a=a, b=b, anchor_a=anchor_a, axis_a=axis_a, beta=beta
        ))

    def add_mouse_joint(self, joint: MouseJoint) -> None:
        """Add a mouse joint to the scene."""
        self.mouse_joints.append(joint)

    def remove_mouse_joint(self, joint: MouseJoint) -> None:
        """Remove a specific mouse joint (e.g. on mouse up)."""
        if joint in self.mouse_joints:
            self.mouse_joints.remove(joint)

    def query_point(self, point: tuple[float, float]) -> RigidBody2D | None:
        """
        Find a dynamic body containing the world point.
        
        Used for mouse picking.
        
        Args:
            point: World space point [x, y].
            
        Returns:
            The first body found containing the point, or None.
        """
        # Simple bruteforce for now, or use Broadphase if point search implemented
        # Broadphase point query: get cells for point, check bodies in cell.
        # But we don't persist broadphase state fully (it's rebuild-ish).
        # However, bodies are always there.
        # Let's iterate all bodies for robustness in small scenes.
        
        # Optimization: Check AABB first?
        px, py = point
        
        # Reverse iterate (check top bodies first roughly based on insertion)
        for b in reversed(self.bodies):
            # Skip static if you only want dynamic? 
            # User might want to drag static? But prompt said "on dynamic body".
            # Usually static bodies are massive or fixed.
            
            # 1. Broad check (Circle Radius or AABB)
            # Just do full shape check for now.
            
            # Point in Circle?
            if isinstance(b.shape, Circle):
                dx = px - b.position[0]
                dy = py - b.position[1]
                if dx*dx + dy*dy <= b.shape.radius * b.shape.radius:
                    return b
                    
            # Point in Box?
            elif isinstance(b.shape, Box):
                # Transform to local
                local = b.world_to_local(point)
                hx, hy = b.shape.half_extents
                if abs(local[0]) <= hx and abs(local[1]) <= hy:
                    return b
                    
            # Point in Polygon? (Not yet fully implemented/common but good to handle)
            # Omitted for simplicity unless needed.
            
        return None

    def _apply_forces(self) -> None:
        """Apply all active force generators to bodies."""
        # Accumulate forces (cleared at start of step implicitly by clear_forces logic usually, 
        # but here we follow a specific order: clear -> apply -> integrate).
        # Actually, clear_forces is often done after integration or before accumulation.
        # In this loop:
        # 1. We clear forces.
        # 2. We add gravity, drag, EM.
        for b in self.bodies:
            # Skip sleeping bodies (optimization & prevent drift)
            if b.sleeping:
                continue
                
            b.clear_forces()
            apply_gravity(b, self._g)
            apply_linear_drag(b, self.drag_c)
            if self.enable_em:
                apply_lorentz(b, self._E, self.Bz)

        if self.enable_em:
            apply_coulomb_pairwise(self.bodies, eps=self.coulomb_eps)

    def _integrate(self, dt: float) -> None:
        """
        Advance kinematic state of all bodies by dt.
        
        Dispatches to the configured integrator.
        """
        if self.integrator == "rk4":
            for b in self.bodies:
                if not b.sleeping:
                    rk4_step(b, dt)
        elif self.integrator == "verlet":
            for b in self.bodies:
                if not b.sleeping:
                    verlet_step(b, dt)
        elif self.integrator == "rk4_adaptive":
            # Adaptive RK4: each body takes sub-steps as needed to satisfy error tolerance.
            # To maintain determinism, we sort bodies by ID.
            for b in sorted(self.bodies, key=lambda x: x.id):
                if b.sleeping:
                    continue
                    
                remaining = dt
                local_dt = min(dt, self.rk_dt_max)
                attempts = 0
                while remaining > 1e-15 and attempts < 64:
                    h = min(remaining, local_dt)
                    accepted, next_dt = rk4_adaptive_step(b, h, self.rk_tol, self.rk_dt_min, self.rk_dt_max)
                    if accepted > 0:
                        remaining -= accepted
                        local_dt = next_dt
                    else:
                        local_dt = next_dt
                    attempts += 1
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")

    def _contacts(self) -> list[Contact]:
        """
        Run valid collision detection pipeline.
        
        1. Broadphase: Find candidate pairs (spatial hash).
        2. Narrowphase: Exact intersection test (Circle-Circle or GJK+EPA).
        
        Returns:
            List of active Contacts, sorted by ID for deterministic solver order.
        """
        # OPTIMIZATION: Could filter sleeping bodies from broadphase updates 
        # inside SpatialHashBroadphase logic, but for now we just detect normally.
        pairs = self._broadphase.pairs(self.bodies)
        contacts = []
        for a, b in pairs:
            # detect_contact handles multiple shape types (Box, Circle)
            c = detect_contact(a, b)
            if c is not None:
                contacts.append(c)
        
        # Sort contacts to ensure deterministic solver order (essential for stability)
        contacts.sort(key=lambda c: (c.a.id, c.b.id))
        return contacts

    def _solve(self, contacts: list[Contact], dt: float) -> None:
        """
        Resolve constraints (collisions + joints).
        
        Uses Sequential Impulse (Projected Gauss-Seidel) method.
        """
        # 1. Pre-calculate solver constants (effective mass, bias)
        prepare_contacts(contacts, dt)
        
        # Apply warm starting (impulses from previous frame)
        for c in contacts:
            # Skip if both bodies are stationary (sleeping or static)
            is_a_stationary = c.a.sleeping or c.a.mass <= 0
            is_b_stationary = c.b.sleeping or c.b.mass <= 0
            
            if not (is_a_stationary and is_b_stationary):
                warm_start_contact(c)
        
        # 2. Iteratively resolve velocities
        for _ in range(self.solver_iters):
            for c in contacts:
                # Optimization: Could skip contacts between two sleeping bodies
                # But IslandManager usually ensures entire islands are awake or sleep.
                # Treat static bodies (mass <= 0) as compatible with sleeping.
                is_a_stationary = c.a.sleeping or c.a.mass <= 0
                is_b_stationary = c.b.sleeping or c.b.mass <= 0
                
                if is_a_stationary and is_b_stationary:
                    continue
                solve_contact_pgs(c)
        
        if self.constraints:
            solve_distance_constraints_pgs(self.constraints, dt, iters=self.solver_iters)
        if self.revolute_constraints:
            solve_revolute_constraints_pgs(self.revolute_constraints, dt, iters=self.solver_iters)
        if self.prismatic_constraints:
            solve_prismatic_constraints_pgs(self.prismatic_constraints, dt, iters=self.solver_iters)
        if self.mouse_joints:
            solve_mouse_constraints_pgs(self.mouse_joints, dt, iters=self.solver_iters)

    def _circle_ccd_substep(self, dt: float) -> float:
        """
        Perform Continuous Collision Detection (CCD) for circles.
        
        Finds the earliest Time of Impact (TOI) for any circle pair.
        If a collision occurs within the timestep (before dt):
            1. Integrate everything to TOI.
            2. Resolve that specific collision immediately.
            3. Return the time advanced (TOI).
        
        Returns:
            Amount of time advanced (<= dt).
        """
        pairs = self._broadphase.pairs(self.bodies)
        toi = None
        hit_pair = None
        
        # Find earliest collision time
        for a, b in pairs:
            if not (isinstance(a.shape, Circle) and isinstance(b.shape, Circle)):
                continue
            
            # Optimization: Skip CCD for sleeping/static pairs
            is_a_stationary = a.sleeping or a.mass <= 0
            is_b_stationary = b.sleeping or b.mass <= 0
            if is_a_stationary and is_b_stationary:
                continue

            # Analytic quadratic solution for circle-circle impact time
            t = circle_toi(
                a.position, a.velocity, a.shape.radius,
                b.position, b.velocity, b.shape.radius, 
                dt
            )
            
            if t is not None and (toi is None or t < toi):
                toi = t
                hit_pair = (a, b)
        
        # Logic:
        # If no TOI found, or TOI is very close to start (already touching),
        # we treat it as no 'tunneling' event and just return 0.
        # The regular discrete solver will handle the contact.
        if toi is None or toi <= 1e-12:
            return 0.0
            
        # Move system to TOI
        self._integrate(toi)
        
        # Create and resolve the specific contact
        a, b = hit_pair
        c = circle_circle_contact(a, b)
        if c is not None:
            # If hit, wake them up!
            a.wake()
            b.wake()
            
            # Prepare and solve just this contact to prevent tunneling
            prepare_contacts([c], max(toi, 1e-9))
            solve_contact_pgs(c)
            
        return toi

    def step(self, dt: float | None = None) -> None:
        """
        Advance the simulation by one frame (dt).
        
        Architecture:
        1. Island Sleep Update (Wake propagation & Sleep decision).
        2. Force Application.
        3. Substepping loop.
        """
        dt = float(self.dt if dt is None else dt)
        prof = self.profiler

        # 0. Island Management
        # Must run BEFORE forces/integration to correctly propagate 'Wake' 
        # signals through the graph (using contacts/constraints from previous frame).
        if prof:
            with prof.section("islands"):
                 all_cons = (
                    self.constraints + 
                    self.revolute_constraints + 
                    self.prismatic_constraints
                )
                 # Use persisted contacts from Manager
                 active_contacts = list(self.contact_manager.manifolds.values())
                 self.island_manager.update(self.bodies, active_contacts, all_cons, dt)
        else:
            all_cons = (
                self.constraints + 
                self.revolute_constraints + 
                self.prismatic_constraints
            )
            active_contacts = list(self.contact_manager.manifolds.values())
            self.island_manager.update(self.bodies, active_contacts, all_cons, dt)

        # Measure forces section
        if prof: 
            with prof.section("forces"):
                self._apply_forces()
        else:
            self._apply_forces()

        # Substep loop for stability
        # Larger dt is split into smaller 'h' chunks.
        h = dt / self.substeps
        
        for _ in range(self.substeps):
            # 1. Unknown Time: Handle CCD
            #    We might advance 'advanced' amount of time if a TOI is found.
            if prof: 
                with prof.section("ccd"):
                    advanced = self._circle_ccd_substep(h)
            else:
                advanced = self._circle_ccd_substep(h)

            # 2. Remaining Time: Standard Integration
            rem = h - advanced
            if rem > 1e-15:
                if prof:
                    with prof.section("integrate"):
                        self._integrate(rem)
                else:
                    self._integrate(rem)

            # 3. Collision Detection & Resolution
            if prof:
                with prof.section("contacts"):
                    contacts = self._contacts()
                    contacts = self.contact_manager.update(contacts)
                with prof.section("solve"):
                    self._solve(contacts, h)
            else:
                contacts = self._contacts()
                contacts = self.contact_manager.update(contacts)
                self._solve(contacts, h)

        self.time += dt
