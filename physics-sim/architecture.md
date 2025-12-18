# Architecture

## Component diagram (logical)
Scene
 ├─ Bodies (RigidBody2D)
 ├─ Forces (gravity, drag, EM: Lorentz + Coulomb)
 ├─ Integrator (adaptive RK4 / fixed RK4 / Verlet)
 ├─ Collision
 │   ├─ Broadphase (spatial hash)
 │   ├─ Narrowphase (GJK + EPA for convex; circle special-case)
 │   └─ CCD (circle TOI; conservative advancement hook)
 ├─ Contact solver (PGS sequential impulses, warm start)
 ├─ Constraint solver (PGS, Baumgarte)
 ├─ IO (JSON scene)
 ├─ Profiler + Benchmarks
 └─ Renderer adapter (interface only)

## Public API (core)
- Scene(gravity, dt, integrator, rk_tol, substeps, solver_iters, enable_em)
  - add_body(body) -> id
  - step(dt=None)
  - to_json() / from_json()

- RigidBody2D(shape, mass, position, angle, velocity, omega, material, charge)

## Data model
- Stable integer ids for determinism
- float64 state arrays per body (small N) for clarity; can be vectorized later

## File layout
See repo tree.
