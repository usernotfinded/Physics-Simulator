# physics-sim

[![CI](https://github.com/YOUR_USERNAME/physics-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/physics-sim/actions/workflows/ci.yml)

Deterministic 2D rigid-body mechanics + quasi-static electromagnetism simulator with adaptive RK4, collisions (GJK/EPA + circle CCD), constraints, and automated verification (≤1% error vs analytic).

## Features

- **Rigid Body Dynamics**: Newton-Euler equations with RK4/adaptive RK4/Verlet integrators
- **Collision Detection**: Spatial hash broadphase + GJK/EPA narrowphase (Circles, Boxes, Polygons) + circle CCD
- **Constraints**: Distance, Revolute (Hinge), and Prismatic (Slider) joints with Baumgarte stabilization & PGS solver
- **Electromagnetism**: Quasi-static Lorentz force + pairwise Coulomb (softened)
- **Determinism**: float64, stable ordering, no `fastmath`, reproducible across runs
- **Profiling**: Built-in profiler for performance analysis
- **Serialization**: Complete JSON schema for scene persistence (bodies, shapes, constraints)

## Installation

```bash
# Clone and install
git clone https://github.com/natanmucelli/physics-sim.git
cd physics-sim
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src pytest tests/ -v

# Run example
PYTHONPATH=src python examples/minimal_freefall.py
```

## Quick Start

```python
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
from physics_sim.materials import Material

scene = Scene(gravity=(0.0, -9.81), dt=1/240)
mat = Material(friction=0.3, restitution=0.5)
body = RigidBody2D(Circle(0.5), mass=1.0, position=(0, 10), material=mat)
scene.add_body(body)

for _ in range(240):
    scene.step()
print(f"Position after 1s: {body.position}")
```

## Examples

```bash
PYTHONPATH=src python examples/elastic_collision.py
PYTHONPATH=src python examples/pendulum.py
PYTHONPATH=src python examples/em_coulomb_orbitish.py
```

## Benchmarks

```bash
PYTHONPATH=src python benchmarks/bench_steps.py
```

Sample output:
```
N=  10  step=   0.94 ms  steps/s= 1062
N= 250  step=  27.33 ms  steps/s=   37
```

## Project Structure

```
physics-sim/
├── src/physics_sim/
│   ├── scene.py          # Main simulation loop
│   ├── types.py          # RigidBody2D, Circle, shapes
│   ├── materials.py      # Friction, restitution
│   ├── collision/        # Broadphase, narrowphase, CCD
│   ├── constraints/      # Distance constraint, PGS solver
│   ├── core/             # Integrators (RK4, Verlet)
│   ├── io/               # JSON scene import/export
│   └── renderer/         # Visualization adapter
├── tests/                # Automated verification tests
├── benchmarks/           # Performance benchmarks
├── examples/             # Usage examples
└── maths.md              # Numerical algorithms documentation
```

## Determinism Checklist

- ✅ `float64` everywhere (default)
- ✅ Stable ordering: bodies and contacts sorted by id
- ✅ Fixed random seed when used
- ✅ No `numba` fastmath; no non-deterministic parallel reductions
- ✅ Adaptive RK4 step acceptance is deterministic

## Documentation

- [maths.md](maths.md) – Equations, integrators, collision stack, numerical stability
- [architecture.md](architecture.md) – System design overview
- [spec.md](spec.md) – Specification and requirements

## License

MIT – see [LICENSE](LICENSE)
