"""Shared task pipeline boundary.

The current large runtime modules remain callable as compatibility backends;
new task implementations should enter through this package.
"""

from .compiler import CompiledTask, TaskPipeline

__all__ = ["CompiledTask", "TaskPipeline"]
