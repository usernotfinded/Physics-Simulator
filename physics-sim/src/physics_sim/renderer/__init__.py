# MIT License (see LICENSE)
"""
Rendering adapters for visualization.

This subpackage provides abstract and concrete renderer implementations:
    - RendererAdapter: Abstract base class defining the rendering interface.
    - DebugRenderer: Text/console output for debugging.
    - NullRenderer: No-op renderer for performance testing.
    - BufferedRenderer: Records frames for playback or export.

The physics engine has no rendering dependency; these adapters are optional.

Typical usage:
    from physics_sim.renderer import DebugRenderer
    
    renderer = DebugRenderer()
    renderer.render_scene(scene)
"""
from .adapter import (
    RendererAdapter,
    DebugRenderer,
    NullRenderer,
    BufferedRenderer,
)

__all__ = [
    "RendererAdapter",
    "DebugRenderer",
    "NullRenderer",
    "BufferedRenderer",
]
