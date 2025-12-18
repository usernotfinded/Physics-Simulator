# MIT License (see LICENSE)
"""
Renderer adapters for physics simulation visualization.

This module provides an abstract base class for rendering and a concrete
debug implementation. The core engine has no rendering dependency - 
these adapters are optional for visualization.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TextIO
import sys
import numpy as np

from ..types import RigidBody2D, Circle, Box

if TYPE_CHECKING:
    from ..scene import Scene


class RendererAdapter(ABC):
    """
    Abstract base class for renderer implementations.
    
    Subclasses should implement the drawing methods to integrate with
    various graphics backends (matplotlib, pyglet, web frontend, etc.).
    
    Usage:
        renderer = MyRenderer()
        renderer.begin_frame(scene.time)
        for body in scene.bodies:
            renderer.draw_body(body)
        renderer.end_frame()
        
    Or use the convenience method:
        renderer.render_scene(scene)
    """
    
    @abstractmethod
    def begin_frame(self, time: float) -> None:
        """
        Begin a new frame for rendering.
        
        Args:
            time: Current simulation time in seconds.
        """
        ...
    
    @abstractmethod
    def draw_body(self, body: RigidBody2D) -> None:
        """
        Draw a single rigid body.
        
        Args:
            body: The body to draw.
        """
        ...
    
    @abstractmethod
    def end_frame(self) -> None:
        """
        Finalize the current frame.
        
        Called after all bodies have been drawn for this frame.
        """
        ...
    
    def render_scene(self, scene: "Scene") -> None:
        """
        Convenience method to render all bodies in a scene.
        
        Args:
            scene: The scene to render.
        """
        self.begin_frame(scene.time)
        for body in scene.bodies:
            self.draw_body(body)
        self.end_frame()


class DebugRenderer(RendererAdapter):
    """
    Console/text debug renderer for development and testing.
    
    Outputs human-readable text representation of bodies to a stream
    (stdout by default). Useful for debugging without graphics dependencies.
    
    Example:
        renderer = DebugRenderer()
        renderer.render_scene(scene)
        
    Output:
        === Frame t=0.0042 ===
        [1] Circle r=0.10 @ (0.00, 9.99) v=(0.00, -0.04) θ=0.00
        [2] Box 1.00x0.50 @ (2.00, 5.00) v=(1.00, 0.00) θ=0.79
    """
    
    def __init__(self, output: TextIO | None = None, verbose: bool = True):
        """
        Initialize the debug renderer.
        
        Args:
            output: Output stream (defaults to sys.stdout).
            verbose: If True, include velocity and angle info.
        """
        self.output = output or sys.stdout
        self.verbose = verbose
        self._current_time = 0.0
    
    def begin_frame(self, time: float) -> None:
        """Begin a new debug frame."""
        self._current_time = time
        self.output.write(f"=== Frame t={time:.4f} ===\n")
    
    def draw_body(self, body: RigidBody2D) -> None:
        """Draw a body as text output."""
        pos = body.position
        shape = body.shape
        
        # Format shape info
        if isinstance(shape, Circle):
            shape_str = f"Circle r={shape.radius:.2f}"
        elif isinstance(shape, Box):
            hx, hy = shape.half_extents
            shape_str = f"Box {2*hx:.2f}x{2*hy:.2f}"
        else:
            shape_str = f"Shape({type(shape).__name__})"
        
        # Position
        line = f"[{body.id}] {shape_str} @ ({pos[0]:.2f}, {pos[1]:.2f})"
        
        # Velocity and angle (if verbose)
        if self.verbose:
            vel = body.velocity
            line += f" v=({vel[0]:.2f}, {vel[1]:.2f}) θ={body.angle:.2f}"
        
        self.output.write(line + "\n")
    
    def end_frame(self) -> None:
        """End the debug frame."""
        self.output.write("\n")
        self.output.flush()


class NullRenderer(RendererAdapter):
    """
    No-op renderer that does nothing.
    
    Useful as a placeholder or for performance testing without rendering overhead.
    """
    
    def begin_frame(self, time: float) -> None:
        pass
    
    def draw_body(self, body: RigidBody2D) -> None:
        pass
    
    def end_frame(self) -> None:
        pass


class BufferedRenderer(RendererAdapter):
    """
    Renderer that buffers frame data for later retrieval.
    
    Stores body states for each frame, useful for recording simulations
    or batch processing.
    
    Example:
        renderer = BufferedRenderer()
        for _ in range(100):
            scene.step()
            renderer.render_scene(scene)
        
        # Access recorded frames
        for frame in renderer.frames:
            print(f"t={frame['time']}, bodies={len(frame['bodies'])}")
    """
    
    def __init__(self):
        self.frames: list[dict] = []
        self._current_frame: dict | None = None
    
    def begin_frame(self, time: float) -> None:
        """Begin buffering a new frame."""
        self._current_frame = {
            "time": time,
            "bodies": [],
        }
    
    def draw_body(self, body: RigidBody2D) -> None:
        """Buffer body state."""
        if self._current_frame is None:
            return
        
        self._current_frame["bodies"].append({
            "id": body.id,
            "position": body.position.tolist() if isinstance(body.position, np.ndarray) else list(body.position),
            "velocity": body.velocity.tolist() if isinstance(body.velocity, np.ndarray) else list(body.velocity),
            "angle": body.angle,
            "omega": body.omega,
        })
    
    def end_frame(self) -> None:
        """Finalize and store the buffered frame."""
        if self._current_frame is not None:
            self.frames.append(self._current_frame)
            self._current_frame = None
    
    def clear(self) -> None:
        """Clear all buffered frames."""
        self.frames.clear()
