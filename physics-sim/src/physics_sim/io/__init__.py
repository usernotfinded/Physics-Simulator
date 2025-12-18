# MIT License (see LICENSE)
"""
Input/Output utilities for physics simulation.

This subpackage provides:
    - JSON serialization: Save and load scenes to/from JSON files.
    - Round-trip support: Serialized scenes can be loaded back identically.

Typical usage:
    from physics_sim.io import load_scene, save_scene, scene_to_json
    
    # Load a scene from JSON
    scene = load_scene("my_scene.json")
    
    # Save a scene to JSON
    save_scene(scene, "output.json")
    
    # Get scene as dict (for custom serialization)
    data = scene_to_json(scene)
"""
from .json_io import (
    load_scene,
    load_scene_raw,
    save_scene,
    scene_to_json,
    bodies_to_json,
    body_to_json,
    body_from_json,
)

__all__ = [
    # Loading
    "load_scene",
    "load_scene_raw",
    # Saving
    "save_scene",
    # Serialization
    "scene_to_json",
    "bodies_to_json",
    "body_to_json",
    "body_from_json",
]
