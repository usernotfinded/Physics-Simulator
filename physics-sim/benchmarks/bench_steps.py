"""
Microbenchmark: time per step vs number of bodies.
Run:
  python benchmarks/bench_steps.py
"""
import time
import numpy as np
from physics_sim.scene import Scene
from physics_sim.types import RigidBody2D, Circle
from physics_sim.materials import Material
from physics_sim.profiler import Profiler

def run(n: int, steps: int = 300):
    prof = Profiler()
    scene = Scene(
        gravity=(0.0, -9.81),
        dt=1/240,
        integrator="rk4_adaptive",
        substeps=2,
        solver_iters=15,
        profiler=prof,
        enable_em=False,
    )

    rng = np.random.default_rng(12345)  # determinism (no randomness elsewhere)
    mat = Material(friction=0.2, restitution=0.2)

    # spawn circles in a grid with small random jitter
    side = int(np.ceil(np.sqrt(n)))
    k = 0
    for iy in range(side):
        for ix in range(side):
            if k >= n:
                break
            x = 0.5 * ix + 0.01 * float(rng.normal())
            y = 0.5 * iy + 0.01 * float(rng.normal()) + 2.0
            b = RigidBody2D(Circle(0.2), mass=1.0, position=(x, y), velocity=(0.0, 0.0), material=mat)
            scene.add_body(b)
            k += 1

    # warmup
    for _ in range(30):
        scene.step()

    t0 = time.perf_counter()
    for _ in range(steps):
        scene.step()
    t1 = time.perf_counter()

    total = t1 - t0
    per_step = total / steps
    return per_step, prof.stats.summary()

if __name__ == "__main__":
    for n in [10, 50, 100, 250, 500]:
        per_step, summary = run(n)
        print(f"N={n:4d}  step={1e3*per_step:8.3f} ms  steps/s={1/per_step:8.1f}")
        # print top sections
        for k in ["forces", "ccd", "integrate", "contacts", "solve"]:
            if k in summary:
                print(" ", k, summary[k])
        print()
