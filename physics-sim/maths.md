# Maths (SI units)

## State (per rigid body, 2D)
- Position x ∈ R^2 [m]
- Velocity v = dx/dt [m/s]
- Orientation θ [rad]
- Angular velocity ω = dθ/dt [rad/s]
- Mass m [kg], inverse mass m⁻¹ [1/kg]
- Moment of inertia I [kg·m²], inverse I⁻¹ [1/(kg·m²)]
- Net force F [N = kg·m/s²]
- Net torque τ [N·m]

## Equations of motion (Newton–Euler)
(1) dx/dt = v
(2) dv/dt = F / m
(3) dθ/dt = ω
(4) dω/dt = τ / I

## Gravity
(5) F_g = m g, where g = (0, -9.81) [m/s²]

## Linear drag (optional)
(6) F_d = -c v   [N], c [kg/s]

## Quasi-static electromagnetism
Charge q [C]
Electric field E [V/m = N/C]
Magnetic field B [T = N·s/(C·m)] out of plane (scalar in 2D)

Lorentz force:
(7) F_L = q (E + v × B)
In 2D, for B = (0,0,Bz): v×B = (vy*Bz, -vx*Bz)

Pairwise Coulomb (softened):
(8) F_ij = k q_i q_j * r_ij / (|r_ij|^2 + ε^2)^(3/2)
k = 8.9875517923e9 [N·m²/C²], ε [m]

## Contacts (impulses)
Normal impulse magnitude j_n [N·s]
Relative velocity at contact along normal: v_n [m/s]
Restitution e ∈ [0,1]

(9) j_n = -(1+e) v_n / (m_eff)
m_eff = (m_a⁻¹ + m_b⁻¹ + (r_a×n)^2 I_a⁻¹ + (r_b×n)^2 I_b⁻¹)

Friction (Coulomb):
(10) |j_t| ≤ μ j_n
tangent impulse attempts to cancel tangential relative velocity.

## Constraints (distance constraint)
C(x) = |x_b - x_a| - L = 0
Jacobian J and velocity constraint J v + b = 0
Baumgarte stabilization:
(11) b = β/h * C(x)
β ∈ [0,1], h timestep

Solve via PGS:
(12) λ ← clamp(λ - (J v + b)/ (J M⁻¹ Jᵀ), limits)

# Integrators (algorithms / pseudocode)

## Fixed-step RK4 (maths.md Eq (1)-(4))
Given state y = (x,v,θ,ω) and derivative f(y) from Eq (1)-(4):
k1 = f(y_n)
k2 = f(y_n + (h/2) k1)
k3 = f(y_n + (h/2) k2)
k4 = f(y_n + h k3)
y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)

## Adaptive RK4 via step-doubling (deterministic policy)
Compute:
y_big  = RK4(y_n, h)
y_half = RK4(RK4(y_n, h/2), h/2)
err = ||y_half - y_big|| (component-wise mixed norm)
Accept if err ≤ tol OR h ≤ h_min.
If accepted: take y_{n+1} = y_half and propose next step:
h_next = clamp(h * 0.9 * (tol/err)^(1/5), h_min, h_max)
If rejected: restore y_n and set h = max(h_min, h/2), retry.

## Symplectic velocity Verlet (fallback)
x_{n+1} = x_n + v_n h + 1/2 a_n h^2
v_{n+1} = v_n + a_n h
(and same form for θ, ω using α)

# Collision detection stack

## Broadphase: spatial hashing
- Compute conservative AABB per body.
- Insert body id into grid cells overlapped by AABB.
- Candidate pairs are all unique id pairs co-resident in any cell.

Expected behavior: O(n) insertion; pair count depends on density (worst-case O(n^2)).

## Narrowphase: GJK + EPA (convex)
GJK:
- Uses support mapping S(d)=supA(d)-supB(-d) in Minkowski difference.
- Iteratively builds simplex; if simplex encloses origin => intersection.

EPA (if intersecting):
- Expand polytope edges toward origin until improvement < tol.
- Result: penetration normal n and depth δ.

## Continuous collision detection (TOI)
For circles:
Solve |(p0-p1)+t(v0-v1)| = (r0+r1), t∈[0,h] analytically (quadratic).
Choose earliest non-negative root as TOI.

For general convex:
Hook: conservative advancement using GJK distance queries (not fully implemented here).

# Contact / constraint resolution

## PGS / sequential impulses (contacts)
- For each contact:
  - compute relative velocity at contact
  - compute effective mass m_eff
  - compute normal impulse j_n from Eq (9) + Baumgarte bias
  - clamp accumulated normal impulse ≥ 0
  - compute friction impulse j_t and clamp by Eq (10)

Warm starting: reuse accumulated impulses from previous step for faster convergence (in this reference, per-step accumulation is local; production would cache by persistent contact id).

## Constraint stabilization
Baumgarte adds bias term b = β/h * C(x) (Eq (11)) to velocity-level solve, reducing positional drift.

# Numerical stability notes (sources & mitigation)

Drift sources:
- Non-symplectic integrators (RK4) can drift energy in long runs.
- Constraint projection/impulses introduce non-physical energy changes if under-iterated.
- Discrete collision detection causes tunneling at high velocities.

Mitigation:
- Substepping + CCD (TOI) reduces tunneling.
- More solver iterations + warm starting improves constraint satisfaction.
- For long conservative systems: choose Verlet integrator.

# Error handling policy
- Adaptive RK4 rejects steps deterministically if error > tol.
- Safety clamps: h_min ≤ h ≤ h_max; bounded retries.
- Optional invariant checks (momentum/energy) can emit diagnostic logs if thresholds exceeded.
