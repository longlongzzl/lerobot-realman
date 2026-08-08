"""Task policies for the shared RM75 perception/planning/execution stack."""

from .registry import TASK_REGISTRY, get_task_adapter, list_task_adapters

__all__ = ["TASK_REGISTRY", "get_task_adapter", "list_task_adapters"]
