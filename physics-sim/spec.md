# Scope & domain

## Primary
- Classical mechanics (2D): kinematics, rigid-body dynamics, gravity, external forces/torques
- Collisions: discrete + continuous (CCD for circles), friction + restitution
- Constraints: distance constraint (pendulum); extensible to joints

## Optional (requested & included)
- Quasi-static electromagnetism:
  - Lorentz force: F = q (E + v × B)
  - Pairwise Coulomb force for small N (O(N^2)) with softening epsilon

## Exclusions
- Quantum mechanics
- Full Navier–Stokes / turbulence / CFD beyond particle-SPH (not included)
- Relativistic EM / Maxwell wave propagation (not included)

## Configurable parameters
- Body: mass, inertia, pose, velocities, forces/torques, charge
- Material: friction μ, restitution e
- Engine: timestep dt, integrator (adaptive RK4 / fixed RK4 / Verlet), tolerance, solver iterations, precision float64
