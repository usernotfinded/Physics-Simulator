# MIT License (see LICENSE)
"""
JSON serialization and deserialization for physics scenes.

This module provides functions to save and load simulation scenes. The JSON
format is designed to be human-readable and compatible with the web viewer.

JSON Schema Overview:
---------------------
{
  "gravity": [float, float],       # Default: [0.0, -9.81]
  "dt": float,                     # Timestep (sec), default: 1/240
  "integrator": string,            # "rk4", "rk4_adaptive", or "verlet"
  "substeps": int,                 # Physics substeps per frame
  "bodies": [
    {
      "shape": {                   # Required
        "type": "circle" | "box" | "polygon",
        "radius": float,           # If circle
        "hx": float, "hy": float,  # If box (half-extents)
        "vertices": [[x,y], ...]   # If polygon (CCW)
      },
      "mass": float,               # Required (>0 for dynamic, <=0 for static)
      "position": [x, y],          # Default: [0, 0]
      "velocity": [vx, vy],        # Default: [0, 0]
      "angle": float,              # Radians, default: 0
      "omega": float,              # Rad/s, default: 0
      "material": {                # Optional
        "friction": float,         # Default: 0.5
        "restitution": float       # Default: 0.2
      },
      "charge": float              # Coulombs, default: 0
    }
  ],
  # Optional simulation constraints/parameters
  "solver_iters": int,
  "enable_em": bool,
  "drag_c": float,
  "E": [Ex, Ey],
  "Bz": float,
  "constraints": [                 # Optional
    {
      "type": "distance",
      "bodyA": int, "bodyB": int,  # Indices in bodies list
      "length": float,
      "beta": float                # Optional
    },
    {
      "type": "revolute",
      "bodyA": int, "bodyB": int,
      "anchorA": [x, y],
      "anchorB": [x, y],
      "beta": float
    },
    {
      "type": "prismatic",
      "bodyA": int, "bodyB": int,
      "anchorA": [x, y],
      "axisA": [x, y],
      "beta": float,
      "refAngle": float
    }
  ]
}
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import RigidBody2D, Circle, Box, ConvexPolygon
from ..materials import Material
from ..constraints.solver import (
    DistanceConstraint, RevoluteConstraint, PrismaticConstraint
)

if TYPE_CHECKING:
    from ..scene import Scene


def load_scene_raw(path: str) -> dict[str, Any]:
    """
    Load raw JSON data from a scene file without object construction.
    
    Useful for inspecting scene parameters or using custom parsers.
    
    Args:
        path: Absolute or relative path to the JSON file.
        
    Returns:
        Dictionary containing the raw JSON data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene(path: str) -> "Scene":
    """
    Load and construct a fully initialized Scene from a JSON file.
    
    Parses the JSON file, creates a Scene with specified global parameters
    (gravity, integrators, etc.), and instantiates all bodies.
    
    Args:
        path: Path to the JSON scene file.
        
    Returns:
        A ready-to-run Scene object.
        
    Raises:
        FileNotFoundError: If the file cannot be found.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If required fields (like body mass or shape) are missing.
    """
    # Import locally to avoid circular import (Scene imports json_io)
    from ..scene import Scene
    
    data = load_scene_raw(path)
    
    # Extract scene parameters with safe defaults
    gravity = tuple(data.get("gravity", [0.0, -9.81]))
    dt = float(data.get("dt", 1 / 240.0))
    integrator = data.get("integrator", "rk4_adaptive")
    substeps = int(data.get("substeps", 4))
    solver_iters = int(data.get("solver_iters", 20))
    enable_em = bool(data.get("enable_em", True))
    drag_c = float(data.get("drag_c", 0.0))
    E = tuple(data.get("E", [0.0, 0.0]))
    Bz = float(data.get("Bz", 0.0))
    
    scene = Scene(
        gravity=gravity,
        dt=dt,
        integrator=integrator,
        substeps=substeps,
        solver_iters=solver_iters,
        enable_em=enable_em,
        drag_c=drag_c,
        E=E,
        Bz=Bz,
    )
    
    # Instantiate and add bodies
    loaded_bodies = []
    for body_data in data.get("bodies", []):
        body = body_from_json(body_data)
        scene.add_body(body)
        loaded_bodies.append(body)

    # Load constraints
    for c_data in data.get("constraints", []):
        c_type = c_data["type"]
        idx_a = c_data["bodyA"]
        idx_b = c_data["bodyB"]
        
        # Verify indices
        if not (0 <= idx_a < len(loaded_bodies) and 0 <= idx_b < len(loaded_bodies)):
            raise ValueError(f"Constraint references invalid body index: {idx_a} or {idx_b}")
            
        body_a = loaded_bodies[idx_a]
        body_b = loaded_bodies[idx_b]
        beta = float(c_data.get("beta", 0.2)) # Default default
        
        if c_type == "distance":
            length = float(c_data["length"])
            scene.add_distance_constraint(body_a, body_b, length, beta)
            
        elif c_type == "revolute":
            anchor_a = tuple(c_data.get("anchorA", [0.0, 0.0]))
            anchor_b = tuple(c_data.get("anchorB", [0.0, 0.0]))
            scene.add_revolute_constraint(body_a, body_b, anchor_a, anchor_b, beta)
            
        elif c_type == "prismatic":
            anchor_a = tuple(c_data.get("anchorA", [0.0, 0.0]))
            axis_a = tuple(c_data.get("axisA", [1.0, 0.0]))
            scene.add_prismatic_constraint(body_a, body_b, anchor_a, axis_a, beta)
            # Handle manual refAngle setting if needed? 
            # add_prismatic_constraint doesn't take refAngle currently (defaults to 0).
            # If we want to restore exact state, we might need to set it manually on the object.
            # But the scene helper doesn't return the constraint object.
            # For now, ignore advanced refAngle restoration or assume 0 (initialized state).
            
        else:
            # Ignore unknown types for forward compatibility
            pass
    
    return scene


def body_from_json(d: dict[str, Any]) -> RigidBody2D:
    """
    Parse a single rigid body definition from a dictionary.
    
    Args:
        d: Dictionary containing body properties (shape, mass, etc.).
        
    Returns:
        Initialized RigidBody2D instance.
    """
    # 1. Parse Shape
    # -------------------------------------------------------------------------
    if "shape" not in d:
        raise ValueError("Body definition missing required 'shape' field.")
        
    shape_data = d["shape"]
    shape_type = shape_data.get("type")
    
    if shape_type == "circle":
        radius = float(shape_data["radius"])
        if radius <= 0:
            raise ValueError(f"Circle radius must be positive, got {radius}")
        shape = Circle(radius=radius)
    elif shape_type == "box":
        hx = float(shape_data["hx"])
        hy = float(shape_data["hy"])
        if hx <= 0 or hy <= 0:
            raise ValueError(f"Box extents must be positive, got ({hx}, {hy})")
        shape = Box(half_extents=(hx, hy))
    elif shape_type == "polygon":
        verts = shape_data["vertices"]
        if len(verts) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        # Ensure numpy array
        shape = ConvexPolygon(vertices=np.array(verts, dtype=np.float64))
    else:
        raise ValueError(f"Unknown shape type: '{shape_type}'")

    # 2. Parse Material (optional)
    # -------------------------------------------------------------------------
    mat_data = d.get("material", {})
    mat = Material(
        friction=float(mat_data.get("friction", 0.5)),
        restitution=float(mat_data.get("restitution", 0.2)),
    )

    # 3. Construct Body
    # -------------------------------------------------------------------------
    return RigidBody2D(
        shape=shape,
        mass=float(d["mass"]),
        position=tuple(d.get("position", [0.0, 0.0])),
        angle=float(d.get("angle", 0.0)),
        velocity=tuple(d.get("velocity", [0.0, 0.0])),
        omega=float(d.get("omega", 0.0)),
        material=mat,
        charge=float(d.get("charge", 0.0)),
    )


def body_to_json(body: RigidBody2D) -> dict[str, Any]:
    """
    Serialize a RigidBody2D to a dictionary (round-trip compatible).
    
    Only minimal/non-default fields are included to keep the output concise.
    """
    # Serialize shape
    if isinstance(body.shape, Circle):
        shape_data = {"type": "circle", "radius": body.shape.radius}
    elif isinstance(body.shape, Box):
        hx, hy = body.shape.half_extents
        shape_data = {"type": "box", "hx": hx, "hy": hy}
    elif isinstance(body.shape, ConvexPolygon):
        verts = body.shape.vertices.tolist() if isinstance(body.shape.vertices, np.ndarray) else list(body.shape.vertices)
        shape_data = {"type": "polygon", "vertices": verts}
    else:
        raise TypeError(f"Cannot serialize unknown shape type: {type(body.shape)}")
    
    # Base required fields
    result = {
        "shape": shape_data,
        "mass": body.mass,
        "position": _to_list(body.position),
        "velocity": _to_list(body.velocity),
    }
    
    # Optional fields (skip if default)
    if body.angle != 0.0:
        result["angle"] = body.angle
    if body.omega != 0.0:
        result["omega"] = body.omega
    if body.charge != 0.0:
        result["charge"] = body.charge
    
    # Material (skip if default)
    # Note: Default material values encoded here must match Material defaults
    if body.material.friction != 0.5 or body.material.restitution != 0.2:
        result["material"] = {
            "friction": body.material.friction,
            "restitution": body.material.restitution,
        }
    
    return result


def bodies_to_json(bodies: list[RigidBody2D]) -> list[dict[str, Any]]:
    """Serialize a list of bodies to a JSON-compatible list."""
    return [body_to_json(b) for b in bodies]


def scene_to_json(scene: "Scene") -> dict[str, Any]:
    """
    Serialize a complete Scene to a dictionary.
    
    Captured state includes:
    - Global physics parameters (gravity, dt, etc.)
    - All bodies and their kinematic state
    - EM field parameters
    """
    result = {
        "gravity": list(scene.gravity),
        "dt": scene.dt,
        "bodies": bodies_to_json(scene.bodies),
    }
    
    # Optional parameters (skip if standard defaults)
    if scene.integrator != "rk4_adaptive":
        result["integrator"] = scene.integrator
    if scene.substeps != 4:
        result["substeps"] = scene.substeps
    if scene.solver_iters != 20:
        result["solver_iters"] = scene.solver_iters
    if not scene.enable_em:
        result["enable_em"] = scene.enable_em
    if scene.drag_c != 0.0:
        result["drag_c"] = scene.drag_c
    if scene.E != (0.0, 0.0):
        result["E"] = list(scene.E)
    if scene.Bz != 0.0:
        result["Bz"] = scene.Bz
        
    # Serialize constraints
    if scene.constraints or scene.revolute_constraints or scene.prismatic_constraints:
        c_list = []
        
        # Helper to find index
        def get_idx(b: RigidBody2D) -> int:
            try:
                return scene.bodies.index(b)
            except ValueError:
                return -1
        
        for c in scene.constraints:
            c_list.append({
                "type": "distance",
                "bodyA": get_idx(c.a),
                "bodyB": get_idx(c.b),
                "length": c.length,
                "beta": c.beta
            })
            
        for rc in scene.revolute_constraints:
            c_list.append({
                "type": "revolute",
                "bodyA": get_idx(rc.a),
                "bodyB": get_idx(rc.b),
                "anchorA": _to_list(rc.anchor_a),
                "anchorB": _to_list(rc.anchor_b),
                "beta": rc.beta
            })
            
        for pc in scene.prismatic_constraints:
            c_list.append({
                "type": "prismatic",
                "bodyA": get_idx(pc.a),
                "bodyB": get_idx(pc.b),
                "anchorA": _to_list(pc.anchor_a),
                "axisA": _to_list(pc.axis_a),
                "beta": pc.beta,
                "refAngle": pc.ref_angle
            })
            
        result["constraints"] = c_list
    
    return result


def save_scene(scene: "Scene", path: str, indent: int = 2) -> None:
    """Save a Scene instance to a JSON file on disk."""
    data = scene_to_json(scene)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def _to_list(arr: Any) -> list[float]:
    """Helper: Convert numpy array or tuple to a clean list of floats."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return list(arr)
