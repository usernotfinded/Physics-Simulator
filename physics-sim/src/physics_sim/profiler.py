# MIT License (see LICENSE)
"""
Simple profiling utilities for performance measurement.

Provides lightweight instrumentation to measure execution time of
simulation phases (forces, integration, collision, solving) without
external dependencies.

Example:
    profiler = Profiler()
    with profiler.section("physics_step"):
        scene.step()
    print(profiler.stats.summary())
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class ProfileStats:
    """
    Accumulates timing samples for named sections.
    
    Stores raw timing data and provides summary statistics (count, mean, max).
    """
    samples: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, dt: float) -> None:
        """Record a timing sample (in seconds) for a named section."""
        self.samples.setdefault(name, []).append(dt)

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Compute summary statistics for all recorded sections.
        
        Returns:
            Dict mapping section name to stats dict with keys:
            - 'n': sample count
            - 'mean_ms': average time in milliseconds
            - 'max_ms': maximum time in milliseconds
        """
        out = {}
        for name, times in self.samples.items():
            n = len(times)
            out[name] = {
                "n": n,
                "mean_ms": 1e3 * (sum(times) / n),
                "max_ms": 1e3 * max(times),
            }
        return out


class Profiler:
    """
    Context-manager based profiler for timing code sections.
    
    Usage:
        profiler = Profiler()
        with profiler.section("my_operation"):
            do_expensive_work()
        
        # Get timing results
        stats = profiler.stats.summary()
        print(f"my_operation avg: {stats['my_operation']['mean_ms']:.2f}ms")
    """
    
    def __init__(self) -> None:
        self.stats = ProfileStats()

    def section(self, name: str):
        """
        Return a context manager that times the enclosed code.
        
        Args:
            name: Identifier for this timed section.
        """
        profiler = self

        class _Section:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, exc_type, exc, tb):
                elapsed = time.perf_counter() - self.t0
                profiler.stats.add(name, elapsed)

        return _Section()
