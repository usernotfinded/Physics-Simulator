# Simulator

A high-precision deterministic 2D physics engine and visualization system.

This monorepo contains two main components:

- **[physics-sim](physics-sim/)**: A deterministic 2D rigid-body mechanics simulator written in Python. Features adaptive RK4 integration, GJK/EPA collision detection, and constraint solvers.
- **[sim-viewer](sim-viewer/)**: A high-performance, Apple-like Canvas2D + React viewer for visualizing simulation recordings.

## Quick Start

### 1. Run a Simulation
Generate a recording using the physics engine:

```bash
cd physics-sim
pip install -e .
python examples/elastic_collision.py
# This generates 'elastic_collision.json'
```

### 2. Visualize
Load the recording in the web viewer:

```bash
cd sim-viewer
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) and drop the generated JSON file into the window.

## License

MIT - See [LICENSE](LICENSE)
